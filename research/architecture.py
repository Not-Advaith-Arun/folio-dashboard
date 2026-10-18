"""Offline, protocol-bound pooled return/downside training and costed evaluation.

Run with python -m research.architecture /private/path/protocol.json. Inputs are
explicit hashed local files; there is no acquisition, cloud write or broker API.
Persist models for audit, never load externally supplied pickle artifacts.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from research.engine import atomic_json
from research.execution import charges, validated_calendar
from research.panel import session_panel, STOCK_FEATURES
from research.rebalance import replay_rebalanced

SPEC = {
    "version": "dual-head-20-1", "horizon": 20, "calibrationDates": 126,
    "minimumTrainDates": 252, "maximumTrainDates": 756, "minimumCandidates": 4,
    "ridgeAlpha": 100., "treeIterations": 100, "treeLeaves": 7,
    "treeMinLeaf": 50, "treeL2": 10., "learningRate": .05, "quantile": .1,
    "randomState": 42, "downsidePenalty": .25, "riskTargetAnnual": .1,
    "nameCap": .25, "maximumNames": 4, "minimumDownside": .005,
    "selectionSlippageBps": 10., "referenceOrderValue": 25000.,
}
FEATURES = STOCK_FEATURES + ["cohortReturn20", "cohortVolatility20", "breadth", "relative20"]
BOOKS = ["dual_head", "ensemble_mean_only", "ridge_mean_only", "trend",
         "equal_ready", "equal_buy_hold", "cash"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def architecture_panel(frames, sessions, *, calendar_source, price_basis):
    panel, manifest = session_panel(frames, sessions, calendar_source=calendar_source,
                                    price_basis=price_basis, horizon=SPEC["horizon"])
    ready = panel[panel.featureReady].copy()
    ready["positiveTrend"] = ready.trend200.gt(0).astype(float)
    cohort = ready.groupby("date").agg(cohortReturn20=("return20", "median"),
        cohortVolatility20=("volatility20", "median"), breadth=("positiveTrend", "mean"))
    panel = panel.join(cohort, on="date")
    panel["relative20"] = panel.return20 - panel.cohortReturn20
    manifest["architectureFeatures"] = FEATURES
    return panel, manifest


def fit_at(panel, signal):
    """Fit on mature past labels; current candidates never require an outcome."""
    signal = pd.Timestamp(signal)
    snapshot = panel[(panel.date == signal) & panel.featureReady].copy().sort_values("symbol")
    if len(snapshot) < SPEC["minimumCandidates"]:
        return None, {"status": "insufficient_current_candidates", "ready": len(snapshot)}, None
    # A missing future label is only a training exclusion after its horizon has
    # elapsed. It must never be a current-stock selection condition.
    past = panel[(panel.date < signal) & (panel.labelEnd < signal) & panel.featureReady
                 & np.isfinite(panel.actual) & np.isfinite(panel[FEATURES]).all(axis=1)].copy()
    past = past[past.groupby("date").symbol.transform("size") >= SPEC["minimumCandidates"]]
    dates = sorted(past.date.unique())
    if len(dates) < SPEC["calibrationDates"] + SPEC["minimumTrainDates"]:
        return None, {"status": "insufficient_history"}, None
    calibration_dates = dates[-SPEC["calibrationDates"]:]
    calibration_start = pd.Timestamp(calibration_dates[0])
    train = past[(past.date < calibration_start) & (past.labelEnd < calibration_start)].copy()
    train_dates = sorted(train.date.unique())[-SPEC["maximumTrainDates"]:]
    train = train[train.date.isin(train_dates)]
    if len(train_dates) < SPEC["minimumTrainDates"]:
        return None, {"status": "insufficient_purged_history"}, None
    calibration = past[past.date.isin(calibration_dates)]
    weights = 1. / train.groupby("date").symbol.transform("size").to_numpy()
    weights /= weights.mean()
    ridge = make_pipeline(StandardScaler(), Ridge(alpha=SPEC["ridgeAlpha"]))
    tree_args = dict(max_iter=SPEC["treeIterations"], max_leaf_nodes=SPEC["treeLeaves"],
                     min_samples_leaf=SPEC["treeMinLeaf"], l2_regularization=SPEC["treeL2"],
                     learning_rate=SPEC["learningRate"], early_stopping=False, random_state=SPEC["randomState"])
    mean = HistGradientBoostingRegressor(**tree_args)
    downside = HistGradientBoostingRegressor(loss="quantile", quantile=SPEC["quantile"], **tree_args)
    with threadpool_limits(limits=2):
        ridge.fit(train[FEATURES], train.actual, standardscaler__sample_weight=weights,
                  ridge__sample_weight=weights)
        mean.fit(train[FEATURES], train.actual, sample_weight=weights)
        downside.fit(train[FEATURES], train.actual, sample_weight=weights)
        bias = {}
        for name, model in [("ridge", ridge), ("tree", mean)]:
            residual = calibration.actual - model.predict(calibration[FEATURES])
            bias[name] = float(residual.groupby(calibration.date).mean().mean())
            snapshot[name] = model.predict(snapshot[FEATURES]) + bias[name]
        correction = float(np.quantile(calibration.actual - downside.predict(calibration[FEATURES]), SPEC["quantile"]))
        snapshot["q10"] = downside.predict(snapshot[FEATURES]) + correction
    snapshot["mean"] = (snapshot.ridge + snapshot.tree) / 2
    snapshot["pastMean"] = float(train.groupby("date").actual.mean().mean())
    snapshot["pastQ10"] = float(train.actual.quantile(SPEC["quantile"]))
    lineage = {"status": "fitted", "signal": str(signal.date()), "trainRows": len(train),
               "trainDates": len(train_dates), "calibrationRows": len(calibration),
               "calibrationDates": len(calibration_dates), "trainStart": str(train.date.min().date()),
               "trainLastOutcome": str(train.labelEnd.max().date()),
               "calibrationStart": str(calibration_start.date()),
               "calibrationLastOutcome": str(calibration.labelEnd.max().date()),
               "meanBias": bias, "q10ResidualCorrection": correction,
               "features": FEATURES, "spec": SPEC}
    models = {"ridge": ridge, "mean": mean, "downside": downside, "lineage": lineage}
    return snapshot, lineage, models


def allocate(snapshot, book):
    if snapshot is None or len(snapshot) < SPEC["minimumCandidates"]:
        return {}
    if book == "equal_ready":
        return dict.fromkeys(snapshot.symbol, 1. / len(snapshot))
    value = SPEC["referenceOrderValue"]
    cost = (charges(value, "buy") + charges(value, "sell")) / value + 2 * SPEC["selectionSlippageBps"] / 10000
    if book == "trend":
        candidates = snapshot[(snapshot.return60 > 0) & (snapshot.trend200 > 0)].copy()
        candidates["utility"] = candidates.return60
    else:
        candidates = snapshot.copy()
        if book not in {"dual_head", "ensemble_mean_only", "ridge_mean_only"}:
            raise ValueError("Unknown allocation book")
        candidates["utility"] = candidates["ridge" if book == "ridge_mean_only" else "mean"] - cost
        if book == "dual_head":
            candidates["utility"] -= SPEC["downsidePenalty"] * (-candidates.q10).clip(lower=SPEC["minimumDownside"])
        candidates = candidates[candidates.utility > 0]
    candidates = candidates.sort_values(["utility", "symbol"], ascending=[False, True]).head(SPEC["maximumNames"])
    weights = {}
    for row in candidates.itertuples():
        if not np.isfinite(row.volatility20) or row.volatility20 <= 0:
            continue
        weights[row.symbol] = min(SPEC["nameCap"], SPEC["riskTargetAnnual"] /
                                 (np.sqrt(252) * row.volatility20) / SPEC["maximumNames"])
    return weights


def daily_returns(result):
    equity = pd.Series([p["equity"] for p in result["curve"]])
    return equity.pct_change().dropna().to_numpy()


def excess_interval(primary, benchmark):
    delta = daily_returns(primary) - daily_returns(benchmark)
    if len(delta) < 40:
        return None
    rng = np.random.default_rng(42)
    count, block = len(delta), 20
    # Paired moving blocks preserve within-month dependence, not full-regime
    # uncertainty. With one viewed year this is diagnostic, not proof of alpha.
    draws = []
    for _ in range(1000):
        starts = rng.integers(0, count - block + 1, int(np.ceil(count / block)))
        sample = np.concatenate([delta[start:start + block] for start in starts])[:count]
        draws.append(float(sample.mean() * 252))
    low, high = np.quantile(draws, [.025, .975])
    return {"annualizedMeanDailyExcess": float(delta.mean() * 252), "lower95": float(low),
            "upper95": float(high), "blockSessions": block, "repetitions": 1000}


def prediction_metrics(rows):
    scored = [r for r in rows if r.get("actual") is not None and r.get("mean") is not None]
    if not scored:
        return {"count": 0}
    y = np.array([r["actual"] for r in scored])
    result = {"count": len(y), "signalDates": len({r["date"] for r in scored}), "mae": {"zero": float(abs(y).mean())}}
    for key in ["mean", "ridge", "tree", "pastMean"]:
        result["mae"][key] = float(np.abs(y - np.array([r[key] for r in scored])).mean())
    result["pinball"] = {}
    for key in ["q10", "pastQ10"]:
        error = y - np.array([r[key] for r in scored])
        result["pinball"][key] = float(np.maximum(.1 * error, -.9 * error).mean())
        if key == "q10":
            result["fractionBelowQ10"] = float((error < 0).mean())
    return result


def verify_inputs(protocol):
    for entry in protocol["featureInputs"] + protocol["executionInputs"] + protocol.get("parents", []):
        if sha(entry["path"]) != entry["sha256"]:
            raise ValueError(f"Source hash mismatch: {Path(entry['path']).name}")


def load_frames(entries):
    frames = {}
    for entry in entries:
        symbol = entry["symbol"]
        if symbol in frames:
            raise ValueError("Duplicate input security")
        frame = pd.DataFrame(json.loads(Path(entry["path"]).read_text()))
        frame["date"] = pd.to_datetime(frame.date)
        frames[symbol] = frame.set_index("date")
    return frames


def run_experiment(protocol_path):
    """Immutable run: same inputs/code return saved result; changes need new protocol."""
    path = Path(protocol_path).resolve()
    protocol = json.loads(path.read_text())
    if protocol["architecture"] != SPEC or protocol["books"] != BOOKS or protocol["signalStride"] != SPEC["horizon"]:
        raise ValueError("Protocol does not match the declared architecture")
    # Refuse after-the-fact protocol dates. Historical evaluation is separately
    # disclosed; frozenAt records declaration, not a historical prediction time.
    frozen = datetime.fromisoformat(protocol["frozenAt"])
    if frozen.tzinfo is None or frozen > datetime.now(timezone.utc):
        raise ValueError("Invalid protocol freeze timestamp")
    verify_inputs(protocol)
    code = {name: sha(Path(__file__).with_name(name)) for name in
            ["architecture.py", "panel.py", "execution.py", "rebalance.py", "engine.py"]}
    lineage = {"protocolSha256": sha(path), "code": code, "sklearn": sklearn.__version__,
               "numpy": np.__version__, "pandas": pd.__version__}
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
                raise ValueError("Completed run differs; declare a new protocol, never overwrite")
            for name, digest in saved["artifacts"].items():
                if sha(out / name) != digest:
                    raise ValueError("Completed artifact hash mismatch")
            return {**json.loads((out / "results.json").read_text()), "cacheHit": True}
        if (out / "started.json").exists():
            raise ValueError("Incomplete attempt preserved; inspect failure and declare a continuation")
        atomic_json(out / "started.json", {**lineage, "fingerprint": fingerprint,
                                         "startedAt": datetime.now(timezone.utc).isoformat()})
        try:
            result = _evaluate(protocol, out, lineage)
            verify_inputs(protocol)
            atomic_json(out / "results.json", result)
            artifacts = {str(p.relative_to(out)): sha(p) for p in out.rglob("*") if p.is_file() and p.name != ".lock"}
            atomic_json(completed, {"fingerprint": fingerprint, "artifacts": artifacts})
            return {**result, "cacheHit": False}
        except Exception as exc:
            atomic_json(out / "failure.json", {"error": str(exc), "type": type(exc).__name__,
                                              "at": datetime.now(timezone.utc).isoformat()})
            raise


def _evaluate(protocol, out, lineage):
    frames, raw = load_frames(protocol["featureInputs"]), load_frames(protocol["executionInputs"])
    if set(frames) != set(raw) or set(raw) != set(protocol["lots"]):
        raise ValueError("Feature/execution/lot universes must match")
    sessions = pd.DatetimeIndex(sorted(set.union(*(set(f.index) for f in frames.values()))))
    excluded = pd.to_datetime(protocol["calendarExcluded"])
    if sessions.isin(excluded).any():
        raise ValueError("Feature inputs still contain excluded sessions")
    start, end = pd.Timestamp(protocol["start"]), pd.Timestamp(protocol["end"])
    raw = {s: f.loc[~f.index.isin(excluded)].copy() for s, f in raw.items()}
    expected = sessions[(sessions >= start) & (sessions <= end)]
    if list(expected) != validated_calendar(raw, start, end):
        raise ValueError("Execution calendar differs from feature calendar")
    if any(f.loc[start:end, "splits"].ne(0).any() for f in raw.values()):
        raise ValueError("This protocol requires a zero-split execution interval")
    panel, coverage = architecture_panel(frames, sessions, calendar_source=protocol["calendarSource"],
                                        price_basis=protocol["priceBasis"])
    atomic_json(out / "coverage.json", coverage)
    entries = sessions[(sessions >= start) & (sessions < end)][::SPEC["horizon"]]
    decisions = {book: [] for book in BOOKS}
    forecasts, fits = [], []
    for entry in entries:
        position = sessions.get_loc(entry)
        if position == 0:
            raise ValueError("Entry has no preceding signal session")
        signal = sessions[position - 1]
        snapshot, fit, models = fit_at(panel, signal)
        fit.update(signal=str(signal.date()), entry=str(entry.date()))
        fits.append(fit)
        model_sha = None
        if models is not None:
            target = out / f"model-{signal.date()}.pickle"
            with target.open("xb") as file:
                os.chmod(target, 0o600)
                pickle.dump(models, file, protocol=pickle.HIGHEST_PROTOCOL)
            model_sha = sha(target)
        ready = panel[(panel.date == signal) & panel.featureReady]
        for book in BOOKS:
            weights = (dict.fromkeys(frames, 1. / len(frames)) if book == "equal_buy_hold" else
                       {} if book == "cash" else allocate(ready if book in {"trend", "equal_ready"} else snapshot, book))
            if book != "equal_buy_hold" or not decisions[book]:
                decisions[book].append({"signal": str(signal.date()), "entry": str(entry.date()), "weights": weights})
        # Write every stock and its unavailability; outcomes are joined only
        # after allocations, never fed to allocate or current selection.
        predictions = {} if snapshot is None else snapshot.set_index("symbol").to_dict("index")
        for row in panel[panel.date == signal].itertuples():
            p = predictions.get(row.symbol, {})
            observed = pd.notna(row.labelEnd) and row.labelEnd <= end and np.isfinite(row.actual)
            forecasts.append({"date": str(signal.date()), "symbol": row.symbol,
                "featureStatus": row.featureStatus, "fitStatus": fit["status"],
                "labelEnd": None if pd.isna(row.labelEnd) else str(row.labelEnd.date()),
                "actual": float(row.actual) if observed else None, "modelSha256": model_sha,
                **{key: float(p[key]) if key in p else None for key in ["mean", "ridge", "tree", "q10", "pastMean", "pastQ10"]}})
    atomic_json(out / "forecasts.json", forecasts)
    atomic_json(out / "decisions.json", decisions)
    atomic_json(out / "fits.json", fits)
    scenarios = {}
    for slip in protocol["slippageScenarios"]:
        books = {book: replay_rebalanced(raw, choices, end, protocol["lots"], initial=protocol["capital"],
                                        slippage_bps=slip) for book, choices in decisions.items()}
        scenarios[str(slip)] = books
    atomic_json(out / "replays.json", scenarios)
    primary = scenarios[str(protocol["primarySlippage"])]
    intervals = {book: excess_interval(primary["dual_head"], primary[book]) for book in
                 ["equal_buy_hold", "trend", "ridge_mean_only", "ensemble_mean_only"]}
    six_start = next(pd.Timestamp(d["entry"]) for d in decisions["cash"]
                     if pd.Timestamp(d["entry"]) >= end - pd.DateOffset(months=6))
    six_month = {}
    for book, choices in decisions.items():
        subset = [d for d in choices if pd.Timestamp(d["entry"]) >= six_start]
        if book == "equal_buy_hold":
            subset = [{**decisions["cash"][[pd.Timestamp(d["entry"]) for d in decisions["cash"]].index(six_start)],
                       "weights": choices[0]["weights"]}]
        replay = replay_rebalanced(raw, subset, end, protocol["lots"], initial=protocol["capital"],
                                   slippage_bps=protocol["primarySlippage"])
        six_month[book] = {k: replay[k] for k in ["ending", "netReturn", "dailyCloseDrawdown", "fees"]}
    checks = {}
    for slip in [5., 10.]:
        books = scenarios[str(slip)]
        checks[f"positive_{slip}bps"] = books["dual_head"]["netReturn"] > 0
        checks[f"beats_baselines_{slip}bps"] = all(books["dual_head"]["ending"] > books[b]["ending"] for b in intervals)
    checks["drawdown_no_worse_than_mean_only"] = primary["dual_head"]["dailyCloseDrawdown"] >= primary["ensemble_mean_only"]["dailyCloseDrawdown"]
    checks["excess_intervals_positive"] = all(v is not None and v["lower95"] > 0 for v in intervals.values())
    return {"version": SPEC["version"], "createdAt": datetime.now(timezone.utc).isoformat(), "lineage": lineage,
            "start": str(start.date()), "end": str(end.date()), "capital": protocol["capital"],
            "fitCount": sum(f["status"] == "fitted" for f in fits), "scheduledDecisions": len(entries),
            "predictionMetrics": prediction_metrics(forecasts), "acceptanceChecks": checks,
            "researchAcceptancePassed": all(checks.values()), "livePromotionAllowed": False,
            "comparison": {slip: {b: {k: r[k] for k in ["ending", "netReturn", "dailyCloseDrawdown", "fees"]}
                                  for b, r in books.items()} for slip, books in scenarios.items()},
            "excessIntervals": intervals, "sixMonthRestart": {"start": str(six_start.date()), "end": str(end.date()), "books": six_month},
            "limitations": protocol["evidenceLimits"] + primary["dual_head"]["limitations"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("protocol", type=Path)
    args = parser.parse_args()
    result = run_experiment(args.protocol)
    print(json.dumps({k: result[k] for k in ["version", "fitCount", "cacheHit", "researchAcceptancePassed",
                                           "predictionMetrics", "comparison"]}, indent=2))
