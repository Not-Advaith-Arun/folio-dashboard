"""Offline retained-source session/volume audit; no credential or network reads.

python -m research.upstox_audit /private/path/PLAN.json
Structural agreement cannot establish raw historical execution prices or rights.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from research.architecture import sha
from research.engine import atomic_json
from research.prospective import job_lock

VERSION = "upstox-retained-audit-1"
GATES = {"rawPriceVerified": False, "volumeUnitsVerified": False,
         "productionCalendarVerified": False, "trainingApproved": False,
         "livePromotionAllowed": False}


def frame(payload):
    rows = payload["data"]["candles"]
    f = pd.DataFrame(rows, columns=["at", "open", "high", "low", "close", "volume", "oi"])
    f["at"] = pd.to_datetime(f["at"])
    idx = pd.DatetimeIndex(f["at"])
    if idx.tz is None or idx.hasnans or idx.has_duplicates:
        raise ValueError("Unique timezone-aware source timestamps required")
    f["at"] = idx.tz_convert("Asia/Kolkata")
    return f.sort_values("at").set_index("at")


def audit(observations, payloads, rules):
    if (rules["prefixStart"] != "09:15" or rules["prefixEndExclusive"] != "15:15"
            or rules["barMinutes"] != 5):
        raise ValueError("Prefix differs from frozen conservative architecture")
    daily = {o["symbol"]: frame(payloads[o["file"]]) for o in observations if o["window"]["unit"] == "days"}
    sessions, reconciliations, retained = [], [], []
    for observation in observations:
        if observation["window"]["unit"] != "minutes":
            continue
        f = frame(payloads[observation["file"]])
        index_only = observation["instrumentKey"].startswith("NSE_INDEX|")
        for day, part in f.groupby(f.index.date):
            clock = part.index.strftime("%H:%M")
            prefix = part[(clock >= "09:15") & (clock < "15:15")]
            # Grid only within each observed day; this never generates trading
            # days, fills missing bars or installs a production calendar.
            expected = pd.date_range(f"{day} 09:15", f"{day} 15:10", freq="5min", tz="Asia/Kolkata")
            zeros = part[part.volume.eq(0)]
            sessions.append({"symbol": observation["symbol"], "sourceFile": observation["file"],
                "day": str(day), "rows": len(part), "benchmarkOnly": index_only,
                "currentCatalogueCASFlag": observation.get("catalogueMatch", {}).get("cas_eligible"),
                "prefixRows": len(prefix), "missingPrefixSlots": expected.difference(prefix.index).astype(str).tolist(),
                "unexpectedPrefixSlots": prefix.index.difference(expected).astype(str).tolist(),
                "zeroVolumeTimes": zeros.index.strftime("%H:%M").tolist(),
                "prefixZeroVolumeRows": int(prefix.volume.eq(0).sum()),
                "allStockZerosAfterPrefix": bool(index_only or all(t >= "15:15" for t in zeros.index.strftime("%H:%M"))),
                **GATES})
            for timestamp, row in part.iterrows():
                if timestamp not in prefix.index or row.volume == 0:
                    retained.append({"sourceFile": observation["file"], "symbol": observation["symbol"],
                        "at": str(timestamp), "volume": float(row.volume), "outsideFixedPrefix": timestamp not in prefix.index,
                        "status": "benchmark_not_trade_candidate" if index_only else
                                  "outside_predeclared_continuous_prefix" if timestamp not in prefix.index else "unresolved_zero_volume"})
            if index_only:
                continue
            d = daily.get(observation["symbol"])
            candidates = d[d.index.date == day] if d is not None else pd.DataFrame()
            comparison = {"symbol": observation["symbol"], "day": str(day), "sourceFile": observation["file"],
                          "status": "missing_daily_comparator", **GATES}
            if len(candidates) == 1:
                row = candidates.iloc[0]
                volume = float(part.volume.sum())
                comparison.update(status="compared", dailyVolume=float(row.volume), minuteVolume=volume,
                    dailyToMinuteVolumeRatio=float(row.volume / volume) if volume > 0 else None,
                    openDifference=float(row.open - part.open.iloc[0]),
                    highDifference=float(row.high - part.high.max()), lowDifference=float(row.low - part.low.min()),
                    officialDailyCloseMinusLastBar=float(row.close - part.close.iloc[-1]))
            reconciliations.append(comparison)
    fact = rules["calendarFact"]
    calendar = {"reviewedFact": fact, "observations": [], "productionCalendarVerified": False}
    for symbol, f in daily.items():
        present = fact["date"] in set(f.index.strftime("%Y-%m-%d"))
        calendar["observations"].append({"symbol": symbol, "holidayCandlePresent": present,
            "lastDailyDate": str(f.index.max().date()),
            "endDateGapConsistentWithReviewedHoliday": not present and str(f.index.max().date()) == "2026-10-19"})
    return {"version": VERSION, "sessions": sessions, "reconciliations": reconciliations,
            "retainedExceptions": retained, "calendar": calendar, **GATES}


def run(plan_path):
    path = Path(plan_path).resolve()
    plan = json.loads(path.read_text())
    frozen = datetime.fromisoformat(plan["frozenAt"])
    if plan["version"] != VERSION or frozen.tzinfo is None or frozen > datetime.now(timezone.utc):
        raise ValueError("Invalid frozen audit plan")
    inputs = plan["inputs"] + [plan["parentPlan"]]
    for entry in inputs:
        if sha(entry["path"]) != entry["sha256"]:
            raise ValueError("Retained source hash mismatch")
    files = {Path(e["path"]).name: json.loads(Path(e["path"]).read_text()) for e in plan["inputs"]}
    if len(files) != len(plan["inputs"]):
        raise ValueError("Duplicate audit input names")
    completion = files["completed.json"]
    for name, digest in completion["artifacts"].items():
        entries = [e for e in plan["inputs"] if Path(e["path"]).name == name]
        # The gzipped catalogue remains in the parent cache and is verified from
        # its manifest, without rewriting or treating today's membership as past.
        source = Path(plan["parentPlan"]["path"]).parent / "run" / name
        if entries:
            source = Path(entries[0]["path"])
        if sha(source) != digest:
            raise ValueError("Parent completion manifest mismatch")
    fingerprint = hashlib.sha256((sha(path) + sha(__file__)).encode()).hexdigest()
    out = path.parent / "run"
    out.mkdir(mode=0o700, exist_ok=True)
    with job_lock(out):
        completed = out / "completed.json"
        if completed.exists():
            saved = json.loads(completed.read_text())
            if saved["fingerprint"] != fingerprint or sha(out / "audit.json") != saved["auditSha256"]:
                raise ValueError("Immutable audit mismatch")
            return {**json.loads((out / "audit.json").read_text()), "cacheHit": True}
        result = audit(files["results.json"]["observations"], files, plan["rules"])
        result["lineage"] = {"planSha256": sha(path), "codeSha256": sha(__file__)}
        result["cacheHit"] = False
        if any(sha(e["path"]) != e["sha256"] for e in inputs):
            raise ValueError("Audit inputs changed")
        atomic_json(out / "audit.json", result)
        atomic_json(completed, {"fingerprint": fingerprint, "auditSha256": sha(out / "audit.json")})
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    result = run(parser.parse_args().plan)
    print(json.dumps({"sessions": len(result["sessions"]), "stockComparisons": len(result["reconciliations"]),
                      "retainedExceptions": len(result["retainedExceptions"]), "cacheHit": result["cacheHit"], **GATES}))
