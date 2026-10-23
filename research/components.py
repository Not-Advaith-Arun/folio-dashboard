"""Frozen overnight/day return decomposition challenger; local CPU research only.

python -m research.components /private/path/protocol.json
Read verified prior JSON and hashes; never refit or unpickle old models.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from research import architecture as base
from research.engine import atomic_json
from research.execution import validated_calendar
from research.payoff import load_reference, date_weights, BOUNDARIES
from research.prospective import job_lock
from research.rebalance import replay_rebalanced

SPEC = {"version": "component-heads-20-1", "horizon": 20, "ridgeAlpha": 100.,
        "lags": [1, 5, 20, 60], "identityTolerance": 1e-10}
NIGHT = base.FEATURES + [f"night{k}" for k in SPEC["lags"]] + ["day20"]
DAY = base.FEATURES + [f"day{k}" for k in SPEC["lags"]] + ["night20"]
FEATURES = list(dict.fromkeys(NIGHT + DAY))
BOOKS = {"component_dual": ("componentMean", "dual_head"),
         "component_mean": ("componentMean", "ensemble_mean_only"),
         "direct_log_dual": ("directMean", "dual_head"),
         "direct_log_mean": ("directMean", "ensemble_mean_only")}
CODE = ["components.py", "payoff.py", "architecture.py", "panel.py", "execution.py",
        "rebalance.py", "engine.py", "prospective.py"]


def code_hashes():
    return {name: base.sha(Path(__file__).with_name(name)) for name in CODE}


def component_panel(frames, sessions, **provenance):
    panel, coverage = base.architecture_panel(frames, sessions, **provenance)
    parts = []
    for symbol, part in panel.groupby("symbol", sort=True):
        f = part.set_index("date").reindex(sessions).copy()
        # Use the already validated/masked OHLC columns, never repair invalid rows.
        day = np.log(f.close / f.open)
        night = np.log(f.open / f.close.shift(1))
        for lag in SPEC["lags"]:
            f[f"day{lag}"] = day.rolling(lag, min_periods=lag).sum()
            f[f"night{lag}"] = night.rolling(lag, min_periods=lag).sum()
        h = SPEC["horizon"]
        f["targetDay"] = pd.concat([day.shift(-i) for i in range(1, h + 1)], axis=1).sum(axis=1, min_count=h)
        f["targetNight"] = pd.concat([night.shift(-i) for i in range(2, h + 2)], axis=1).sum(axis=1, min_count=h)
        for target in ["targetDay", "targetNight"]:
            f[target] = f[target].where(f.actual.notna())
        complete = f.actual.notna()
        if not np.allclose(np.expm1(f.loc[complete, "targetDay"] + f.loc[complete, "targetNight"]),
                           f.loc[complete, "actual"], atol=SPEC["identityTolerance"], rtol=0):
            raise ValueError("Component labels do not reconstruct original open-to-open return")
        f["componentReady"] = f.featureReady & np.isfinite(f[FEATURES]).all(axis=1)
        f.index.name = "date"
        parts.append(f.reset_index())
    combined = pd.concat(parts, ignore_index=True)
    coverage.update(componentSpec=SPEC, nightFeatures=NIGHT, dayFeatures=DAY,
                    componentUnavailable=int((combined.featureReady & ~combined.componentReady).sum()))
    return combined, coverage


def fit_components(panel, signal):
    signal = pd.Timestamp(signal)
    snapshot = panel[(panel.date == signal) & panel.componentReady].copy().sort_values("symbol")
    if len(snapshot) < base.SPEC["minimumCandidates"]:
        return None, {"status": "insufficient_current_candidates"}, None
    past = panel[(panel.date < signal) & (panel.labelEnd < signal) & panel.componentReady
                 & np.isfinite(panel.actual) & np.isfinite(panel[["targetDay", "targetNight"]]).all(axis=1)].copy()
    past = past[past.groupby("date").symbol.transform("size") >= base.SPEC["minimumCandidates"]]
    dates = sorted(past.date.unique())
    if len(dates) < base.SPEC["minimumTrainDates"] + base.SPEC["calibrationDates"]:
        return None, {"status": "insufficient_history"}, None
    cal_dates = dates[-base.SPEC["calibrationDates"]:]
    start = pd.Timestamp(cal_dates[0])
    train = past[(past.date < start) & (past.labelEnd < start)].copy()
    train_dates = sorted(train.date.unique())[-base.SPEC["maximumTrainDates"]:]
    train = train[train.date.isin(train_dates)]
    if len(train_dates) < base.SPEC["minimumTrainDates"]:
        return None, {"status": "insufficient_purged_history"}, None
    calibration = past[past.date.isin(cal_dates)]
    weights = date_weights(train)
    models = {key: make_pipeline(StandardScaler(), Ridge(alpha=SPEC["ridgeAlpha"]))
              for key in ["night", "day", "direct"]}
    with threadpool_limits(limits=2):
        for key, features, target in [("night", NIGHT, train.targetNight), ("day", DAY, train.targetDay),
                                      ("direct", FEATURES, np.log1p(train.actual))]:
            models[key].fit(train[features], target, standardscaler__sample_weight=weights, ridge__sample_weight=weights)
        def predict(frame):
            n, d = models["night"].predict(frame[NIGHT]), models["day"].predict(frame[DAY])
            direct = models["direct"].predict(frame[FEATURES])
            return n, d, direct, np.expm1(n + d), np.expm1(direct)
        _, _, _, cal_component, cal_direct = predict(calibration)
        biases = {"component": float((calibration.actual - cal_component).groupby(calibration.date).mean().mean()),
                  "direct": float((calibration.actual - cal_direct).groupby(calibration.date).mean().mean())}
        n, d, direct, total, control = predict(snapshot)
    snapshot["nightLog"], snapshot["dayLog"], snapshot["directLog"] = n, d, direct
    snapshot["componentMean"], snapshot["directMean"] = total + biases["component"], control + biases["direct"]
    if not np.isfinite(snapshot[["componentMean", "directMean"]]).all(axis=None):
        raise ValueError("Nonfinite unmodified component prediction")
    lineage = {"status": "fitted", "trainRows": len(train), "trainDates": len(train_dates),
        "calibrationRows": len(calibration), "calibrationDates": len(cal_dates),
        "trainStart": str(train.date.min().date()), "trainLastOutcome": str(train.labelEnd.max().date()),
        "calibrationStart": str(start.date()), "calibrationLastOutcome": str(calibration.labelEnd.max().date()),
        "biases": biases, "spec": SPEC}
    return snapshot, lineage, {**models, "lineage": lineage}


def error_interval(rows, control):
    scored = [r for r in rows if r["actual"] is not None and r["componentMean"] is not None]
    f = pd.DataFrame([{"date": r["date"], "delta": abs(r["actual"] - r[control])
                      - abs(r["actual"] - r["componentMean"])} for r in scored])
    if not len(f) or f.date.nunique() < 4:
        return None
    daily, rng = f.groupby("date").delta.mean().to_numpy(), np.random.default_rng(42)
    samples = []
    for _ in range(1000):
        starts = rng.integers(0, len(daily) - 1, int(np.ceil(len(daily) / 2)))
        samples.append(np.concatenate([daily[i:i + 2] for i in starts])[:len(daily)].mean())
    lower, upper = np.quantile(samples, [.025, .975])
    return {"meanImprovement": float(daily.mean()), "lower95": float(lower), "upper95": float(upper),
            "signalDates": len(daily), "blockSignalDates": 2, "repetitions": 1000}


def verify(protocol):
    if (protocol["spec"] != SPEC or protocol["books"] != list(BOOKS)
            or protocol["code"] != code_hashes() or base.sha(protocol["plan"]["path"]) != protocol["plan"]["sha256"]):
        raise ValueError("Frozen component protocol/code changed")
    frozen = datetime.fromisoformat(protocol["frozenAt"])
    if frozen.tzinfo is None or frozen > datetime.now(timezone.utc):
        raise ValueError("Invalid freeze time")
    if any(base.sha(e["path"]) != e["sha256"] for e in protocol.get("sources", [])):
        raise ValueError("Research source evidence changed")
    refs = {k: load_reference(v) for k, v in protocol["references"].items()}
    plan = json.loads(Path(protocol["plan"]["path"]).read_text())
    if (plan["references"] != protocol["references"] or plan["books"] != list(BOOKS)
            or protocol["engineCompatibility"] != refs["payoff"][0]["engineCompatibility"]):
        raise ValueError("Unreviewed plan ancestry or engine compatibility")
    original = refs["original"]
    if original[0]["architecture"] != base.SPEC or original[0]["books"] != base.BOOKS:
        raise ValueError("Original contract changed")
    base.verify_inputs(original[0])
    for name, digest in original[2]["lineage"]["code"].items():
        current = protocol["code"][name]
        if current != digest and (name != "engine.py" or protocol["engineCompatibility"] != {"previous": digest, "current": current}):
            raise ValueError("Unreviewed numerical reference change")
    for _, _, result in refs.values():
        for name, version in {"numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__}.items():
            if result["lineage"][name] != version:
                raise ValueError("Reference runtime changed")
    return refs


def evaluate(refs, out, lineage):
    parent, directory, previous = refs["original"]
    frames, raw = base.load_frames(parent["featureInputs"]), base.load_frames(parent["executionInputs"])
    sessions = pd.DatetimeIndex(sorted(set.union(*(set(f.index) for f in frames.values()))))
    excluded = pd.to_datetime(parent["calendarExcluded"])
    raw = {s: f.loc[~f.index.isin(excluded)].copy() for s, f in raw.items()}
    start, end = pd.Timestamp(parent["start"]), pd.Timestamp(parent["end"])
    if (list(sessions[(sessions >= start) & (sessions <= end)]) != validated_calendar(raw, start, end)
            or sessions.isin(excluded).any() or any(f.loc[start:end, "splits"].ne(0).any() for f in raw.values())):
        raise ValueError("Original calendar/action assumptions changed")
    panel, coverage = component_panel(frames, sessions, calendar_source=parent["calendarSource"], price_basis=parent["priceBasis"])
    atomic_json(out / "coverage.json", coverage)
    original_rows = json.loads((directory / "forecasts.json").read_text())
    lookup = {(r["date"], r["symbol"]): r for r in original_rows}
    old_fits = json.loads((directory / "fits.json").read_text())
    decisions = {b: [] for b in BOOKS}
    fits, forecasts = [], []
    for old_fit, decision in zip(old_fits, json.loads((directory / "decisions.json").read_text())["cash"], strict=True):
        signal = pd.Timestamp(decision["signal"])
        snapshot, fit, models = fit_components(panel, signal)
        fit.update(signal=decision["signal"], entry=decision["entry"])
        if fit["status"] != old_fit["status"] or fit["status"] == "fitted" and any(fit[k] != old_fit[k] for k in BOUNDARIES):
            raise ValueError("Training partition differs from original")
        fits.append(fit)
        model_sha = None
        if models is not None:
            name = out / f"model-{signal.date()}.pickle"
            with name.open("xb") as f:
                os.chmod(name, 0o600)
                pickle.dump(models, f, protocol=pickle.HIGHEST_PROTOCOL)
            model_sha = base.sha(name)
            for key in ["q10", "pastQ10", "pastMean"]:
                snapshot[key] = [lookup[(decision["signal"], s)][key] for s in snapshot.symbol]
        for book, (column, allocator) in BOOKS.items():
            selected = None if snapshot is None else snapshot.assign(mean=snapshot[column])
            decisions[book].append({**decision, "weights": base.allocate(selected, allocator)})
        predictions = {} if snapshot is None else snapshot.set_index("symbol").to_dict("index")
        for row in panel[panel.date == signal].itertuples():
            old = lookup[(decision["signal"], row.symbol)]
            observed = pd.notna(row.labelEnd) and row.labelEnd <= end and np.isfinite(row.actual)
            actual = float(row.actual) if observed else None
            if actual != old["actual"] or row.featureStatus != old["featureStatus"]:
                raise ValueError("Reference outcome/availability changed")
            p = predictions.get(row.symbol, {})
            forecasts.append({"date": decision["signal"], "symbol": row.symbol, "actual": actual,
                "fitStatus": fit["status"], "featureStatus": row.featureStatus,
                "componentReady": bool(row.componentReady), "modelSha256": model_sha,
                "originalMean": old["mean"], "pastMean": old["pastMean"], "q10": old["q10"],
                **{k: float(p[k]) if k in p else None for k in ["componentMean", "directMean", "nightLog", "dayLog", "directLog"]}})
    for name, data in [("forecasts", forecasts), ("fits", fits), ("decisions", decisions)]:
        atomic_json(out / (name + ".json"), data)
    scored = [r for r in forecasts if r["actual"] is not None and r["componentMean"] is not None]
    metrics = {"count": len(scored), "signalDates": len({r["date"] for r in scored}),
        "mae": {k: float(np.mean([abs(r["actual"] - (0 if k == "zero" else r[k])) for r in scored]))
                for k in ["componentMean", "directMean", "originalMean", "pastMean", "zero"]} if scored else {}}
    paired = {k: error_interval(forecasts, k) for k in ["directMean", "originalMean"]}
    replays = {str(slip): {b: replay_rebalanced(raw, d, end, parent["lots"], initial=parent["capital"], slippage_bps=slip)
                         for b, d in decisions.items()} for slip in parent["slippageScenarios"]}
    atomic_json(out / "replays.json", replays)
    comparisons = {slip: {**{b: summary for _, _, result in refs.values() for b, summary in result["comparison"][slip].items()},
                         **{b: {k: r[k] for k in ["ending", "netReturn", "dailyCloseDrawdown", "fees"]} for b, r in books.items()}}
                   for slip, books in replays.items()}
    primary = str(parent["primarySlippage"])
    old_replays = json.loads((directory / "replays.json").read_text())
    intervals = {b: base.excess_interval(replays[primary]["component_dual"],
        replays[primary][b] if b == "direct_log_dual" else old_replays[primary][b])
        for b in ["direct_log_dual", "ensemble_mean_only", "equal_buy_hold"]}
    economic = {}
    for slip in ["5.0", "10.0"]:
        tested = comparisons[slip]["component_dual"]
        economic["positive_" + slip] = tested["netReturn"] > 0
        economic["beats_all_controls_" + slip] = all(tested["ending"] > r["ending"] for b, r in comparisons[slip].items()
                                                    if b not in {"component_dual", "component_mean"})
    economic["drawdown_no_worse"] = comparisons[primary]["component_dual"]["dailyCloseDrawdown"] >= comparisons[primary]["ensemble_mean_only"]["dailyCloseDrawdown"]
    economic["excess_intervals_positive"] = all(v is not None and v["lower95"] > 0 for v in intervals.values())
    predictive = {"same_scored_coverage": len(scored) == previous["predictionMetrics"]["count"],
        "mae_below_controls": bool(scored) and metrics["mae"]["componentMean"] < min(metrics["mae"][k] for k in ["directMean", "originalMean", "zero"]),
        "paired_intervals_positive": all(v is not None and v["lower95"] > 0 for v in paired.values())}
    six_start = previous["sixMonthRestart"]["start"]
    six = {b: replay_rebalanced(raw, [d for d in ds if d["entry"] >= six_start], end, parent["lots"], initial=parent["capital"],
                               slippage_bps=parent["primarySlippage"]) for b, ds in decisions.items()}
    return {"version": SPEC["version"], "lineage": lineage, "fitCount": sum(f["status"] == "fitted" for f in fits),
        "estimatorsFitted": 3 * sum(f["status"] == "fitted" for f in fits), "predictionMetrics": metrics,
        "pairedMaeIntervals": paired, "predictiveAcceptanceChecks": predictive, "predictiveAcceptancePassed": all(predictive.values()),
        "economicAcceptanceChecks": economic, "economicAcceptancePassed": all(economic.values()), "comparison": comparisons,
        "excessIntervals": intervals, "sixMonthRestart": {"start": six_start, "books": {
            **{b: summary for _, _, result in refs.values() for b, summary in result["sixMonthRestart"]["books"].items()},
            **{b: {k: v[k] for k in ["ending", "netReturn", "fees", "dailyCloseDrawdown"]} for b, v in six.items()}}},
        "limitations": previous["limitations"] + ["Exploratory component point forecasts, not conditional expected arithmetic returns.",
            "New feature subsets and two component estimators differ in capacity from the single matched control.",
            "Daily adjusted OHLC components do not verify intraday fillability or solve corporate actions."], "livePromotionAllowed": False}


def run(protocol_path):
    path = Path(protocol_path).resolve()
    protocol = json.loads(path.read_text())
    refs = verify(protocol)
    lineage = {"protocolSha256": base.sha(path), "code": protocol["code"],
               "numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__}
    fingerprint = hashlib.sha256(json.dumps(lineage, sort_keys=True).encode()).hexdigest()
    out = path.parent / "run"
    out.mkdir(mode=0o700, exist_ok=True)
    with job_lock(out):
        if (out / "completed.json").exists():
            saved = json.loads((out / "completed.json").read_text())
            if saved["fingerprint"] != fingerprint or any(base.sha(out / n) != h for n, h in saved["artifacts"].items()):
                raise ValueError("Immutable component result mismatch")
            return {**json.loads((out / "results.json").read_text()), "cacheHit": True}
        if (out / "started.json").exists():
            raise ValueError("Prior incomplete attempt preserved")
        atomic_json(out / "started.json", {**lineage, "startedAt": datetime.now(timezone.utc).isoformat()})
        try:
            result = evaluate(refs, out, lineage)
            verify(protocol)
            atomic_json(out / "results.json", result)
            atomic_json(out / "completed.json", {"fingerprint": fingerprint, "artifacts": {
                p.name: base.sha(p) for p in out.iterdir() if p.is_file() and p.name != ".lock"}})
            return {**result, "cacheHit": False}
        except Exception as e:
            atomic_json(out / "failure.json", {"type": type(e).__name__, "error": str(e)})
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("protocol", type=Path)
    r = run(parser.parse_args().protocol)
    print(json.dumps({k: r[k] for k in ["fitCount", "predictionMetrics", "predictiveAcceptancePassed", "economicAcceptancePassed", "cacheHit"]}))
