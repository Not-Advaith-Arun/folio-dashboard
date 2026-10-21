"""Offline issuer-anchor/action diagnostics, never reconstructed execution bars.

python -m research.upstox_actions /private/path/PLAN.json
Monthly agreement cannot prove daily raw prices, volume units or bonus liquidity.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np

from research.architecture import sha
from research.engine import atomic_json
from research.prospective import job_lock
from research.upstox_audit import frame

VERSION = "upstox-action-audit-1"
GATES = {"rawPriceVerified": False, "volumeUnitsVerified": False,
         "productionCalendarVerified": False, "bonusTradabilityVerified": False,
         "trainingApproved": False, "livePromotionAllowed": False}


def audit(payload, rules):
    action, anchor = rules["action"], rules["anchor"]
    factor = action["totalShareFactor"]
    if (action["type"] != "bonus" or action["bonusNumerator"] != 1
            or action["bonusDenominator"] != 1 or factor != 2
            or rules["priceMultipliers"] != [1, 2]
            or rules["volumeMultipliers"] != [1, 0.5, 2]
            or rules["priceAbsoluteTolerance"] != 0.01
            or rules["volumeAbsoluteTolerance"] != 0 or anchor["exchange"] != "NSE"):
        raise ValueError("Unsupported frozen action/comparison contract")
    # Record date and an F&O adjustment do not establish cash bonus availability.
    if action.get("bonusCreditDate") is not None or action.get("bonusTradableDate") is not None:
        raise ValueError("Credit/listing evidence needs a separately reviewed protocol")
    boundary = date.fromisoformat(action["diagnosticBoundaryDate"])
    date.fromisoformat(action["recordDate"])
    date.fromisoformat(anchor["month"] + "-01")
    start, end = (date.fromisoformat(rules["window"][k]) for k in ["start", "end"])
    if not start < boundary <= end or anchor["month"] >= boundary.strftime("%Y-%m"):
        raise ValueError("Anchor/window must bracket the declared action")
    if (not np.isfinite([anchor[k] for k in ["high", "low", "volume"]]).all()
            or not 0 < anchor["low"] <= anchor["high"] or anchor["volume"] <= 0):
        raise ValueError("Invalid issuer anchor")
    f = frame(payload)
    f[["open", "high", "low", "close", "volume"]] = f[["open", "high", "low", "close", "volume"]].astype(float)
    values = f[["open", "high", "low", "close", "volume"]].to_numpy(dtype=float)
    if (not np.isfinite(values).all() or (values[:, :4] <= 0).any()
            or (values[:, 4] < 0).any() or (values[:, 4] % 1 != 0).any()
            or (f.high < f[["open", "close", "low"]].max(axis=1)).any()
            or (f.low > f[["open", "close", "high"]].min(axis=1)).any()
            or len(set(f.index.date)) != len(f)):
        raise ValueError("Invalid daily OHLCV or duplicate source dates")
    month = f[f.index.strftime("%Y-%m") == anchor["month"]]
    price_checks, volume_checks = [], []
    if len(month):
        for multiplier in rules["priceMultipliers"]:
            high, low = float(month.high.max() * multiplier), float(month.low.min() * multiplier)
            price_checks.append({"multiplier": multiplier, "high": high, "low": low,
                "highDifference": high - anchor["high"], "lowDifference": low - anchor["low"],
                "agreesWithinTolerance": abs(high - anchor["high"]) <= rules["priceAbsoluteTolerance"]
                    and abs(low - anchor["low"]) <= rules["priceAbsoluteTolerance"]})
        for multiplier in rules["volumeMultipliers"]:
            volume = float(month.volume.sum() * multiplier)
            volume_checks.append({"multiplier": multiplier, "volume": volume,
                "difference": volume - anchor["volume"],
                "agreesExactly": volume == anchor["volume"]})
    window = f[(f.index.date >= start) & (f.index.date <= end)]
    before, after = window[window.index.date < boundary], window[window.index.date >= boundary]
    adjacent = {"status": "missing_side", "netReturnEvaluated": False}
    if len(before) and len(after):
        pre, post = before.iloc[-1], after.iloc[0]
        adjacent.update(status="observed_adjusted_scale_diagnostic",
            preDate=str(before.index[-1].date()), postDate=str(after.index[0].date()),
            preProviderClose=float(pre.close), postProviderOpen=float(post.open),
            providerOpenToPriorClose=float(post.open / pre.close - 1),
            declaredFactorPreClose=float(pre.close * factor),
            declaredFactorPostToPrePriceRatio=float(post.open / (pre.close * factor)))
    return {"version": VERSION, "anchor": anchor, "action": action,
        "anchorStatus": "compared_aggregate_only" if len(month) else "missing_anchor_month",
        "observedMonthDates": month.index.strftime("%Y-%m-%d").tolist(),
        "priceComparisons": price_checks, "volumeComparisons": volume_checks,
        "eventWindowProviderBars": [{"day": str(i.date()), **{k: float(row[k]) for k in
            ["open", "high", "low", "close", "volume"]}} for i, row in window.iterrows()],
        "boundaryDiagnostic": adjacent,
        "unresolved": ["Full dated adjustment/factor history and original daily prices",
            "Daily versus minute volume definitions/coverage and units beyond the anchor",
            "Bonus credit and cash-share trading availability; no immediate liquidity assumed",
            "Complete independently reviewed session calendar and historical eligibility"], **GATES}


def run(plan_path):
    path = Path(plan_path).resolve()
    plan = json.loads(path.read_text())
    frozen = datetime.fromisoformat(plan["frozenAt"])
    if plan["version"] != VERSION or frozen.tzinfo is None or frozen > datetime.now(timezone.utc):
        raise ValueError("Invalid frozen action audit")
    entries = [plan["parentPlan"], plan["parentCompletion"], *plan["sources"]]
    def verify_inputs():
        for e in entries:
            if sha(e["path"]) != e["sha256"]:
                raise ValueError("Source hash mismatch")
        parent = Path(plan["parentCompletion"]["path"]).parent
        artifacts = json.loads(Path(plan["parentCompletion"]["path"]).read_text())["artifacts"]
        if not artifacts or any(Path(n).name != n for n in artifacts):
            raise ValueError("Invalid parent artifact names")
        for name, digest in artifacts.items():
            if sha(parent / name) != digest:
                raise ValueError("Parent artifact mismatch")
        if plan["candleFile"] not in artifacts or "results.json" not in artifacts:
            raise ValueError("Candle is not sealed by parent")
        return parent
    parent = verify_inputs()
    ids = [e["id"] for e in plan["sources"]]
    required = [plan["rules"]["anchor"]["sourceId"], *plan["rules"]["action"]["sourceIds"]]
    if len(set(ids)) != len(ids) or not required or not set(required) <= set(ids):
        raise ValueError("Missing/duplicate source lineage")
    observations = json.loads((parent / "results.json").read_text())["observations"]
    matches = [o for o in observations if o["file"] == plan["candleFile"]]
    if (len(matches) != 1 or matches[0]["instrumentKey"] != plan["instrumentKey"]
            or matches[0]["window"]["unit"] != "days"):
        raise ValueError("Instrument/daily-source mismatch")
    dependencies = [path, __file__, frame.__code__.co_filename, atomic_json.__code__.co_filename,
                    sha.__code__.co_filename, job_lock.__wrapped__.__code__.co_filename]
    fingerprint = hashlib.sha256("".join(sha(p) for p in dependencies).encode()).hexdigest()
    out = path.parent / "run"
    out.mkdir(mode=0o700, exist_ok=True)
    with job_lock(out):
        completed = out / "completed.json"
        if completed.exists():
            saved = json.loads(completed.read_text())
            if saved["fingerprint"] != fingerprint or sha(out / "audit.json") != saved["auditSha256"]:
                raise ValueError("Immutable action audit mismatch")
            return {**json.loads((out / "audit.json").read_text()), "cacheHit": True}
        result = audit(json.loads((parent / plan["candleFile"]).read_text()), plan["rules"])
        result["lineage"] = {"planSha256": sha(path), "codeFingerprint": fingerprint,
                             "instrumentKey": plan["instrumentKey"]}
        result["cacheHit"] = False
        verify_inputs()
        atomic_json(out / "audit.json", result)
        atomic_json(completed, {"fingerprint": fingerprint, "auditSha256": sha(out / "audit.json")})
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    r = run(parser.parse_args().plan)
    print(json.dumps({k: r[k] for k in ["anchorStatus", "priceComparisons", "volumeComparisons", "cacheHit", *GATES]}))
