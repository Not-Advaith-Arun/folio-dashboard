"""One controlled volatility-scaled target challenger against a frozen run.

python -m research.normalized /private/path/protocol.json
The original architecture, gates and evidence stay unchanged. No acquisition or
broker/cloud calls; parent JSON is verified and model pickle files are not loaded.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import pickle
import hashlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

from research import architecture as base
from research.engine import atomic_json
from research.execution import validated_calendar
from research.rebalance import replay_rebalanced

TARGET = {"version": "vol-target-20-1", "horizon": 20, "minimumHorizonScale": .01,
          "formula": "actual / max(0.01, sqrt(20) * signal-date volatility20)",
          "inverse": "Multiply each calibrated mean/quantile forecast by that security signal-date scale before unchanged allocation"}
BOOKS = {"normalized_dual": "dual_head", "normalized_mean": "ensemble_mean_only", "normalized_ridge": "ridge_mean_only"}
PREDICTIONS = ["mean", "ridge", "tree", "q10", "pastMean", "pastQ10"]


def normalize_panel(panel):
    normalized = panel.copy()
    ready_volatility = normalized.loc[normalized.featureReady, "volatility20"]
    if not np.isfinite(ready_volatility).all() or (ready_volatility < 0).any():
        raise ValueError("Invalid signal-date volatility on a ready row")
    normalized["targetScale"] = (np.sqrt(TARGET["horizon"]) * normalized.volatility20).clip(
        lower=TARGET["minimumHorizonScale"])
    normalized["actual"] = normalized.actual / normalized.targetScale
    return normalized


def fit_scaled(panel, signal):
    # Scale every historical label by its own dated feature, never by the
    # prediction date's scale or volatility from the outcome window.
    snapshot, lineage, models = base.fit_at(normalize_panel(panel), signal)
    lineage.update(target=TARGET, calibrationUnits="signal-volatility units")
    if snapshot is not None:
        for key in PREDICTIONS:
            snapshot[key] *= snapshot.targetScale
        snapshot["actual"] = panel.loc[snapshot.index, "actual"]
        models["target"] = TARGET
        models["lineage"] = lineage
    return snapshot, lineage, models


def load_reference(protocol):
    for key in ["referenceProtocol", "referenceCompleted"]:
        entry = protocol[key]
        if base.sha(entry["path"]) != entry["sha256"]:
            raise ValueError("Reference manifest hash mismatch")
    protocol_path = Path(protocol["referenceProtocol"]["path"]).resolve()
    directory = protocol_path.parent / "run"
    if Path(protocol["referenceCompleted"]["path"]).resolve() != directory / "completed.json":
        raise ValueError("Reference completion must belong to the referenced protocol")
    parent = json.loads(protocol_path.read_text())
    if parent["architecture"] != base.SPEC or parent["books"] != base.BOOKS:
        raise ValueError("Reference architecture does not match")
    base.verify_inputs(parent)
    manifest = json.loads((directory / "completed.json").read_text())
    for name, digest in manifest["artifacts"].items():
        target = (directory / name).resolve()
        if not target.is_relative_to(directory) or base.sha(target) != digest:
            raise ValueError("Reference artifact hash mismatch")
    reference = json.loads((directory / "results.json").read_text())
    if reference["lineage"]["protocolSha256"] != base.sha(protocol_path):
        raise ValueError("Reference result protocol mismatch")
    for name, version in {"sklearn": sklearn.__version__, "numpy": np.__version__, "pandas": pd.__version__}.items():
        if reference["lineage"][name] != version:
            raise ValueError("Reference runtime changed; target-only comparison requires the same versions")
    for name, digest in reference["lineage"]["code"].items():
        if base.sha(Path(base.__file__).with_name(name)) != digest:
            raise ValueError("Reference implementation changed; declare a new compatible comparison")
    return parent, directory, reference


def paired_metrics(forecasts, reference):
    lookup = {(r["date"], r["symbol"]): r for r in reference}
    if set(lookup) != {(r["date"], r["symbol"]) for r in forecasts}:
        raise ValueError("Prediction candidate universes differ")
    scored, differences = [], []
    for row in forecasts:
        previous = lookup[row["date"], row["symbol"]]
        if (row["actual"] != previous["actual"] or row["featureStatus"] != previous["featureStatus"]
                or row["labelEnd"] != previous["labelEnd"] or row["fitStatus"] != previous["fitStatus"]):
            raise ValueError("Prediction outcomes or availability differ from reference")
        if row["actual"] is not None and row["mean"] is not None:
            if previous["mean"] is None:
                raise ValueError("Reference prediction missing on scored candidate")
            scored.append(row)
            differences.append({"date": row["date"], "improvement": abs(row["actual"] - previous["mean"])
                                - abs(row["actual"] - row["mean"])})
    result = base.prediction_metrics(forecasts)
    result["reference"] = base.prediction_metrics(reference)
    result["normalizedMae"] = {key: float(np.mean([abs(r["actual"] - (0 if key == "zero" else r[key])) /
                                                  r["targetScale"] for r in scored]))
                               for key in ["mean", "ridge", "tree", "zero"]} if scored else {}
    daily = pd.DataFrame(differences).groupby("date").improvement.mean().to_numpy() if differences else np.array([])
    interval = None
    if len(daily) >= 4:
        rng = np.random.default_rng(42)
        samples = []
        for _ in range(1000):
            starts = rng.integers(0, len(daily) - 1, int(np.ceil(len(daily) / 2)))
            samples.append(np.concatenate([daily[i:i + 2] for i in starts])[:len(daily)].mean())
        lower, upper = np.quantile(samples, [.025, .975])
        interval = {"meanImprovement": float(daily.mean()), "lower95": float(lower), "upper95": float(upper),
                    "blockSignalDates": 2, "signalDates": len(daily), "repetitions": 1000}
    result["pairedRawMaeImprovement"] = interval
    return result


def run_experiment(protocol_path):
    path = Path(protocol_path).resolve()
    protocol = json.loads(path.read_text())
    if protocol["target"] != TARGET or protocol["books"] != list(BOOKS):
        raise ValueError("Unknown target experiment")
    frozen = datetime.fromisoformat(protocol["frozenAt"])
    if frozen.tzinfo is None or frozen > datetime.now(timezone.utc):
        raise ValueError("Invalid protocol timestamp")
    parent, directory, reference = load_reference(protocol)
    lineage = {"protocolSha256": base.sha(path), "referenceCompletedSha256": base.sha(directory / "completed.json"),
               "code": {**reference["lineage"]["code"], "normalized.py": base.sha(__file__)},
               "sklearn": sklearn.__version__, "numpy": np.__version__, "pandas": pd.__version__}
    fingerprint = hashlib.sha256(json.dumps(lineage, sort_keys=True).encode()).hexdigest()
    out = path.parent / "run"
    out.mkdir(mode=0o700, exist_ok=True)
    with (out / ".lock").open("a") as lock:
        os.chmod(out / ".lock", 0o600)
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        completed = out / "completed.json"
        if completed.exists():
            saved = json.loads(completed.read_text())
            if saved["fingerprint"] != fingerprint:
                raise ValueError("Completed run changed; declare a new protocol")
            for name, digest in saved["artifacts"].items():
                target = (out / name).resolve()
                if not target.is_relative_to(out) or base.sha(target) != digest:
                    raise ValueError("Completed artifact hash mismatch")
            return {**json.loads((out / "results.json").read_text()), "cacheHit": True}
        if (out / "started.json").exists():
            raise ValueError("Incomplete attempt preserved; declare a continuation")
        atomic_json(out / "started.json", {**lineage, "fingerprint": fingerprint,
                                         "startedAt": datetime.now(timezone.utc).isoformat()})
        try:
            result = evaluate(parent, directory, reference, out, lineage)
            load_reference(protocol)
            atomic_json(out / "results.json", result)
            artifacts = {str(p.relative_to(out)): base.sha(p) for p in out.iterdir() if p.is_file() and p.name != ".lock"}
            atomic_json(completed, {"fingerprint": fingerprint, "artifacts": artifacts})
            return {**result, "cacheHit": False}
        except Exception as exc:
            atomic_json(out / "failure.json", {"type": type(exc).__name__, "error": str(exc),
                                              "at": datetime.now(timezone.utc).isoformat()})
            raise


def evaluate(parent, directory, reference, out, lineage):
    frames, raw = base.load_frames(parent["featureInputs"]), base.load_frames(parent["executionInputs"])
    sessions = pd.DatetimeIndex(sorted(set.union(*(set(f.index) for f in frames.values()))))
    excluded = pd.to_datetime(parent["calendarExcluded"])
    raw = {s: f.loc[~f.index.isin(excluded)].copy() for s, f in raw.items()}
    start, end = pd.Timestamp(parent["start"]), pd.Timestamp(parent["end"])
    if list(sessions[(sessions >= start) & (sessions <= end)]) != validated_calendar(raw, start, end):
        raise ValueError("Execution calendar differs from feature calendar")
    if sessions.isin(excluded).any() or any(f.loc[start:end, "splits"].ne(0).any() for f in raw.values()):
        raise ValueError("Reference exclusion/action assumptions no longer hold")
    panel, coverage = base.architecture_panel(frames, sessions, calendar_source=parent["calendarSource"],
                                            price_basis=parent["priceBasis"])
    atomic_json(out / "coverage.json", coverage)
    reference_decisions = json.loads((directory / "decisions.json").read_text())
    decisions = {book: [] for book in BOOKS}
    forecasts, fits = [], []
    for decision in reference_decisions["cash"]:
        signal = pd.Timestamp(decision["signal"])
        snapshot, fit, models = fit_scaled(panel, signal)
        fit.update(signal=decision["signal"], entry=decision["entry"])
        fits.append(fit)
        model_sha = None
        if models is not None:
            model_path = out / f"model-{signal.date()}.pickle"
            with model_path.open("xb") as file:
                os.chmod(model_path, 0o600)
                pickle.dump(models, file, protocol=pickle.HIGHEST_PROTOCOL)
            model_sha = base.sha(model_path)
        for book, allocator in BOOKS.items():
            decisions[book].append({"signal": decision["signal"], "entry": decision["entry"],
                                    "weights": base.allocate(snapshot, allocator)})
        prediction = {} if snapshot is None else snapshot.set_index("symbol").to_dict("index")
        for row in panel[panel.date == signal].itertuples():
            p = prediction.get(row.symbol, {})
            observed = pd.notna(row.labelEnd) and row.labelEnd <= end and np.isfinite(row.actual)
            forecasts.append({"date": str(signal.date()), "symbol": row.symbol,
                "featureStatus": row.featureStatus, "fitStatus": fit["status"],
                "labelEnd": None if pd.isna(row.labelEnd) else str(row.labelEnd.date()),
                "actual": float(row.actual) if observed else None, "modelSha256": model_sha,
                **{key: float(p[key]) if key in p else None for key in [*PREDICTIONS, "targetScale"]}})
    for name, value in [("forecasts", forecasts), ("fits", fits), ("decisions", decisions)]:
        atomic_json(out / f"{name}.json", value)
    metrics = paired_metrics(forecasts, json.loads((directory / "forecasts.json").read_text()))
    old_replays = json.loads((directory / "replays.json").read_text())
    scenarios = {}
    for slip in parent["slippageScenarios"]:
        scenarios[str(slip)] = {book: replay_rebalanced(raw, choices, end, parent["lots"],
                                 initial=parent["capital"], slippage_bps=slip) for book, choices in decisions.items()}
    atomic_json(out / "replays.json", scenarios)
    primary_key = str(parent["primarySlippage"])
    candidate = scenarios[primary_key]["normalized_dual"]
    comparisons = ["dual_head", "ensemble_mean_only", "ridge_mean_only", "trend", "equal_buy_hold"]
    intervals = {b: base.excess_interval(candidate, old_replays[primary_key][b]) for b in comparisons}
    six_start = reference["sixMonthRestart"]["start"]
    six = {}
    for book, choices in decisions.items():
        subset = [d for d in choices if d["entry"] >= six_start]
        r = replay_rebalanced(raw, subset, end, parent["lots"], initial=parent["capital"], slippage_bps=parent["primarySlippage"])
        six[book] = {k: r[k] for k in ["ending", "netReturn", "dailyCloseDrawdown", "fees"]}
    checks = {}
    for slip in ["5.0", "10.0"]:
        tested = scenarios[slip]["normalized_dual"]
        checks[f"positive_{slip}bps"] = tested["netReturn"] > 0
        checks[f"beats_original_books_{slip}bps"] = all(tested["ending"] > old_replays[slip][b]["ending"] for b in comparisons)
    checks["drawdown_no_worse_than_original_mean"] = candidate["dailyCloseDrawdown"] >= old_replays[primary_key]["ensemble_mean_only"]["dailyCloseDrawdown"]
    checks["all_excess_intervals_positive"] = all(v is not None and v["lower95"] > 0 for v in intervals.values())
    predictive = {"raw_mae_better": metrics.get("count", 0) > 0 and metrics["mae"]["mean"] <
                  min(metrics["reference"]["mae"]["mean"], metrics["mae"]["zero"]),
                  "downside_no_worse": metrics.get("count", 0) > 0 and metrics["pinball"]["q10"] <= metrics["reference"]["pinball"]["q10"],
                  "paired_mae_interval_positive": metrics["pairedRawMaeImprovement"] is not None and metrics["pairedRawMaeImprovement"]["lower95"] > 0}
    return {"version": TARGET["version"], "createdAt": datetime.now(timezone.utc).isoformat(), "lineage": lineage,
            "fitCount": sum(f["status"] == "fitted" for f in fits), "scheduledDecisions": len(fits),
            "start": parent["start"], "end": parent["end"], "capital": parent["capital"], "predictionMetrics": metrics,
            "predictiveAcceptanceChecks": predictive, "predictiveAcceptancePassed": all(predictive.values()),
            "economicAcceptanceChecks": checks, "economicAcceptancePassed": all(checks.values()), "livePromotionAllowed": False,
            "comparison": {slip: {**reference["comparison"][slip], **{b: {k: r[k] for k in
                            ["ending", "netReturn", "dailyCloseDrawdown", "fees"]} for b, r in books.items()}}
                           for slip, books in scenarios.items()}, "excessIntervals": intervals,
            "sixMonthRestart": {"start": six_start, "end": parent["end"], "books": {**reference["sixMonthRestart"]["books"], **six}},
            "limitations": reference["limitations"] + ["Target scaling is a new exploratory hypothesis on the same viewed period.",
                f"Only {metrics.get('signalDates', 0)} scored decision dates; bootstrap intervals are low-power diagnostics.",
                "Signal-date volatility scaling is not future volatility knowledge or a guaranteed risk bound."]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("protocol", type=Path)
    result = run_experiment(parser.parse_args().protocol)
    print(json.dumps({key: result[key] for key in ["version", "fitCount", "cacheHit", "predictionMetrics",
                      "predictiveAcceptancePassed", "economicAcceptancePassed", "comparison"]}, indent=2))
