"""Offline probability / conditional-payoff challenger, never a live signal.

python -m research.payoff /private/path/protocol.json
The net label is adjusted return less a reference cost, not account profit.
Reuse verified baseline JSON, never deserialize old model pickle files.
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
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from research import architecture as base
from research.engine import atomic_json
from research.execution import charges, validated_calendar
from research.rebalance import replay_rebalanced

SPEC = {"version": "payoff-heads-20-1", "horizon": 20, "iterations": 100,
        "leaves": 7, "minLeaf": 50, "l2": 10., "learningRate": .05,
        "seed": 42, "calibratorC": 1., "minClassRows": 100, "minClassDates": 20}
BOOKS = {"payoff_dual": "dual_head", "payoff_mean": "ensemble_mean_only"}
OUTPUTS = ["mean", "netProxy", "uncalibratedNetProxy", "proxyProbability",
           "gain", "loss", "pastProbability", "q10"]
CODE_FILES = ["payoff.py", "architecture.py", "panel.py", "execution.py",
              "rebalance.py", "engine.py"]
BOUNDARIES = ["trainRows", "trainDates", "calibrationRows", "calibrationDates",
              "trainStart", "trainLastOutcome", "calibrationStart", "calibrationLastOutcome"]


def code_hashes():
    return {name: base.sha(Path(__file__).with_name(name)) for name in CODE_FILES}


def reference_cost():
    value = base.SPEC["referenceOrderValue"]
    return ((charges(value, "buy") + charges(value, "sell")) / value
            + 2 * base.SPEC["selectionSlippageBps"] / 10000)


def date_weights(frame):
    weights = 1. / frame.groupby("date").symbol.transform("size").to_numpy()
    return weights / weights.mean()


def fit_payoff(panel, signal):
    """Three heads on mature labels, then disjoint past-only calibration."""
    signal = pd.Timestamp(signal)
    snapshot = panel[(panel.date == signal) & panel.featureReady].copy().sort_values("symbol")
    if len(snapshot) < base.SPEC["minimumCandidates"]:
        return None, {"status": "insufficient_current_candidates", "ready": len(snapshot)}, None
    # Keep the original partition contract; equality is checked against its
    # saved fit records before any new forecasts can be replayed.
    past = panel[(panel.date < signal) & (panel.labelEnd < signal) & panel.featureReady
                 & np.isfinite(panel.actual) & np.isfinite(panel[base.FEATURES]).all(axis=1)].copy()
    past = past[past.groupby("date").symbol.transform("size") >= base.SPEC["minimumCandidates"]]
    dates = sorted(past.date.unique())
    if len(dates) < base.SPEC["calibrationDates"] + base.SPEC["minimumTrainDates"]:
        return None, {"status": "insufficient_history"}, None
    calibration_dates = dates[-base.SPEC["calibrationDates"]:]
    calibration_start = pd.Timestamp(calibration_dates[0])
    train = past[(past.date < calibration_start) & (past.labelEnd < calibration_start)].copy()
    train_dates = sorted(train.date.unique())[-base.SPEC["maximumTrainDates"]:]
    train = train[train.date.isin(train_dates)]
    if len(train_dates) < base.SPEC["minimumTrainDates"]:
        return None, {"status": "insufficient_purged_history"}, None
    calibration = past[past.date.isin(calibration_dates)]
    net, cal_net = train.actual - reference_cost(), calibration.actual - reference_cost()
    positive, cal_positive = net.gt(0), cal_net.gt(0)
    lineage = {"status": "fitted", "signal": str(signal.date()), "trainRows": len(train),
        "trainDates": len(train_dates), "calibrationRows": len(calibration),
        "calibrationDates": len(calibration_dates), "trainStart": str(train.date.min().date()),
        "trainLastOutcome": str(train.labelEnd.max().date()), "calibrationStart": str(calibration_start.date()),
        "calibrationLastOutcome": str(calibration.labelEnd.max().date()),
        "positiveRows": int(positive.sum()), "nonpositiveRows": int((~positive).sum()),
        "costProxy": reference_cost(), "features": base.FEATURES, "spec": SPEC}
    if (cal_positive.nunique() < 2 or any(
            mask.sum() < SPEC["minClassRows"] or train.loc[mask, "date"].nunique() < SPEC["minClassDates"]
            for mask in [positive, ~positive])):
        return None, {**lineage, "status": "insufficient_payoff_classes"}, None
    args = dict(max_iter=SPEC["iterations"], max_leaf_nodes=SPEC["leaves"],
                min_samples_leaf=SPEC["minLeaf"], l2_regularization=SPEC["l2"],
                learning_rate=SPEC["learningRate"], early_stopping=False, random_state=SPEC["seed"])
    probability = HistGradientBoostingClassifier(**args)
    gain, loss = HistGradientBoostingRegressor(**args), HistGradientBoostingRegressor(**args)
    calibrator = LogisticRegression(C=SPEC["calibratorC"], max_iter=1000, random_state=SPEC["seed"])
    weights, cal_weights = date_weights(train), date_weights(calibration)
    with threadpool_limits(limits=2):
        probability.fit(train[base.FEATURES], positive.astype(int), sample_weight=weights)
        # Conditional weighting retains the original full-cohort date weights;
        # each head's mean weight is normalized to keep regularization fixed.
        for model, mask, target in [(gain, positive, net), (loss, ~positive, -net)]:
            w = weights[mask]
            model.fit(train.loc[mask, base.FEATURES], target[mask], sample_weight=w / w.mean())
        calibrator.fit(probability.decision_function(calibration[base.FEATURES]).reshape(-1, 1),
                       cal_positive.astype(int), sample_weight=cal_weights)

        def predict(frame):
            x = frame[base.FEATURES]
            p = calibrator.predict_proba(probability.decision_function(x).reshape(-1, 1))[:, 1]
            g, l = np.maximum(0., gain.predict(x)), np.maximum(0., loss.predict(x))
            return p, g, l, p * g - (1 - p) * l

        _, _, _, cal_prediction = predict(calibration)
        bias = float((cal_net - cal_prediction).groupby(calibration.date).mean().mean())
        p, g, l, prediction = predict(snapshot)
    snapshot["proxyProbability"], snapshot["gain"], snapshot["loss"] = p, g, l
    snapshot["uncalibratedNetProxy"] = prediction
    snapshot["netProxy"] = prediction + bias
    # Restore gross units for the unchanged allocator, which subtracts this
    # exact hurdle once. Actual replay costs are accounted independently.
    snapshot["mean"] = snapshot.netProxy + reference_cost()
    snapshot["pastProbability"] = float(np.average(cal_positive, weights=cal_weights))
    lineage.update(netBias=bias, calibrationPrevalence=float(snapshot.pastProbability.iloc[0]))
    models = {"probability": probability, "gain": gain, "loss": loss,
              "calibrator": calibrator, "lineage": lineage}
    return snapshot, lineage, models


def load_reference(entry):
    path = Path(entry["protocol"]["path"]).resolve()
    completed = Path(entry["completed"]["path"]).resolve()
    directory = path.parent / "run"
    if completed != directory / "completed.json":
        raise ValueError("Reference completion belongs to a different protocol")
    for item in entry.values():
        if base.sha(item["path"]) != item["sha256"]:
            raise ValueError("Reference manifest hash mismatch")
    parent, manifest = json.loads(path.read_text()), json.loads(completed.read_text())
    for name, digest in manifest["artifacts"].items():
        artifact = (directory / name).resolve()
        if not artifact.is_relative_to(directory) or base.sha(artifact) != digest:
            raise ValueError("Reference artifact hash mismatch")
    result = json.loads((directory / "results.json").read_text())
    if result["lineage"]["protocolSha256"] != entry["protocol"]["sha256"]:
        raise ValueError("Reference result protocol mismatch")
    return parent, directory, result


def verify_protocol(protocol):
    if protocol["spec"] != SPEC or protocol["books"] != list(BOOKS) or protocol["code"] != code_hashes():
        raise ValueError("Frozen specification or implementation changed")
    frozen = datetime.fromisoformat(protocol["frozenAt"])
    if frozen.tzinfo is None or frozen > datetime.now(timezone.utc):
        raise ValueError("Invalid protocol timestamp")
    if base.sha(protocol["plan"]["path"]) != protocol["plan"]["sha256"]:
        raise ValueError("Frozen plan changed")
    original = load_reference(protocol["references"]["original"])
    normalized = load_reference(protocol["references"]["normalized"])
    parent, _, reference = original
    if parent["architecture"] != base.SPEC or parent["books"] != base.BOOKS:
        raise ValueError("Original architecture mismatch")
    base.verify_inputs(parent)
    if (normalized[0]["referenceProtocol"] != protocol["references"]["original"]["protocol"]
            or normalized[0]["referenceCompleted"] != protocol["references"]["original"]["completed"]):
        raise ValueError("Normalized reference has different ancestry")
    for name, version in {"sklearn": sklearn.__version__, "numpy": np.__version__, "pandas": pd.__version__}.items():
        if any(r[2]["lineage"][name] != version for r in [original, normalized]):
            raise ValueError("Reference runtime changed")
    # The ledger extraction changed engine.py after the original experiments.
    # Admit only the explicitly reviewed engine revision; all numerical panel,
    # allocator and execution modules must still equal the reference bytes.
    for name, digest in reference["lineage"]["code"].items():
        current = protocol["code"][name]
        if current != digest and (name != "engine.py" or
                protocol["engineCompatibility"] != {"previous": digest, "current": current}):
            raise ValueError("Unreviewed reference implementation change")
    return original, normalized


def metrics(rows, previous):
    lookup = {(r["date"], r["symbol"]): r for r in previous}
    if len(lookup) != len(previous) or set(lookup) != {(r["date"], r["symbol"]) for r in rows}:
        raise ValueError("Prediction universes differ")
    scored, delta = [], []
    for row in rows:
        old = lookup[row["date"], row["symbol"]]
        if any(row[k] != old[k] for k in ["actual", "labelEnd", "featureStatus"]):
            raise ValueError("Reference outcomes or availability changed")
        if row["mean"] is not None and row["actual"] is not None:
            if old["mean"] is None:
                raise ValueError("Missing original prediction")
            scored.append(row)
            delta.append({"date": row["date"], "improvement":
                          abs(row["actual"] - old["mean"]) - abs(row["actual"] - row["mean"])})
    result = {"count": len(scored), "signalDates": len({r["date"] for r in scored}),
              "reference": base.prediction_metrics(previous), "pairedMae": None}
    if not scored:
        return result
    frame = pd.DataFrame(scored)
    y, positive = frame.actual, frame.actual.gt(reference_cost()).astype(float)
    result["mae"] = {"mean": float(abs(y - frame["mean"]).mean()), "zero": float(abs(y).mean())}
    result["brier"] = {name: float(((positive - (0.5 if name == "half" else frame[name])) ** 2).mean())
                       for name in ["proxyProbability", "pastProbability", "half"]}
    result["q10Pinball"] = float(np.maximum(.1 * (y - frame.q10), -.9 * (y - frame.q10)).mean())
    result["decisionRegretProxy"] = float((abs(y - reference_cost()) *
        (frame.netProxy.gt(0) != positive.astype(bool))).mean())
    daily = pd.DataFrame(delta).groupby("date").improvement.mean().to_numpy()
    if len(daily) >= 4:
        rng = np.random.default_rng(42)
        samples = [np.concatenate([daily[i:i + 2] for i in
                   rng.integers(0, len(daily) - 1, int(np.ceil(len(daily) / 2)))])[:len(daily)].mean()
                   for _ in range(1000)]
        low, high = np.quantile(samples, [.025, .975])
        result["pairedMae"] = {"meanImprovement": float(daily.mean()), "lower95": float(low),
                               "upper95": float(high), "blockSignalDates": 2, "repetitions": 1000}
    return result


def evaluate(original, normalized, out, lineage):
    parent, directory, reference = original
    frames, raw = base.load_frames(parent["featureInputs"]), base.load_frames(parent["executionInputs"])
    if set(frames) != set(raw) or set(raw) != set(parent["lots"]):
        raise ValueError("Feature/execution/lot universes differ")
    sessions = pd.DatetimeIndex(sorted(set.union(*(set(f.index) for f in frames.values()))))
    excluded = pd.to_datetime(parent["calendarExcluded"])
    raw = {s: f.loc[~f.index.isin(excluded)].copy() for s, f in raw.items()}
    start, end = pd.Timestamp(parent["start"]), pd.Timestamp(parent["end"])
    if list(sessions[(sessions >= start) & (sessions <= end)]) != validated_calendar(raw, start, end):
        raise ValueError("Execution calendar differs from feature calendar")
    if sessions.isin(excluded).any() or any(f.loc[start:end, "splits"].ne(0).any() for f in raw.values()):
        raise ValueError("Reference exclusion/action assumptions changed")
    panel, coverage = base.architecture_panel(frames, sessions, calendar_source=parent["calendarSource"],
                                             price_basis=parent["priceBasis"])
    atomic_json(out / "coverage.json", coverage)
    old_forecasts = json.loads((directory / "forecasts.json").read_text())
    lookup = {(r["date"], r["symbol"]): r for r in old_forecasts}
    old_fits = {r["signal"]: r for r in json.loads((directory / "fits.json").read_text())}
    old_decisions = json.loads((directory / "decisions.json").read_text())
    decisions, forecasts, fits = {book: [] for book in BOOKS}, [], []
    for decision in old_decisions["cash"]:
        signal = pd.Timestamp(decision["signal"])
        snapshot, fit, models = fit_payoff(panel, signal)
        old_fit = old_fits[decision["signal"]]
        if fit["status"] == "fitted":
            if any(fit[k] != old_fit.get(k) for k in BOUNDARIES):
                raise ValueError("Purged training boundaries changed")
            snapshot["q10"] = [lookup[decision["signal"], s]["q10"] for s in snapshot.symbol]
            if not np.isfinite(snapshot.q10).all():
                raise ValueError("Missing frozen downside prediction")
        elif fit["status"] != "insufficient_payoff_classes" and fit["status"] != old_fit["status"]:
            raise ValueError("Unexpected reference fit availability change")
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
            decisions[book].append({**decision, "weights": base.allocate(snapshot, allocator)})
        predictions = {} if snapshot is None else snapshot.set_index("symbol").to_dict("index")
        for row in panel[panel.date == signal].itertuples():
            p = predictions.get(row.symbol, {})
            observed = pd.notna(row.labelEnd) and row.labelEnd <= end and np.isfinite(row.actual)
            forecasts.append({"date": str(signal.date()), "symbol": row.symbol, "featureStatus": row.featureStatus,
                "fitStatus": fit["status"], "labelEnd": None if pd.isna(row.labelEnd) else str(row.labelEnd.date()),
                "actual": float(row.actual) if observed else None, "modelSha256": model_sha,
                **{k: float(p[k]) if k in p else None for k in OUTPUTS}})
    for name, value in [("forecasts", forecasts), ("fits", fits), ("decisions", decisions)]:
        atomic_json(out / f"{name}.json", value)
    measures = metrics(forecasts, old_forecasts)
    old_replays = json.loads((directory / "replays.json").read_text())
    normalized_replays = json.loads((normalized[1] / "replays.json").read_text())
    baseline_replays = {s: {**old_replays[s], **normalized_replays[s]} for s in old_replays}
    scenarios = {str(slip): {b: replay_rebalanced(raw, choices, end, parent["lots"],
                        initial=parent["capital"], slippage_bps=slip) for b, choices in decisions.items()}
                 for slip in parent["slippageScenarios"]}
    atomic_json(out / "replays.json", scenarios)
    primary_key = str(parent["primarySlippage"])
    candidate = scenarios[primary_key]["payoff_dual"]
    intervals = {b: base.excess_interval(candidate, r) for b, r in baseline_replays[primary_key].items()}
    summary = lambda r: {k: r[k] for k in ["ending", "netReturn", "dailyCloseDrawdown", "fees"]}
    six_start = reference["sixMonthRestart"]["start"]
    six = {b: summary(replay_rebalanced(raw, [d for d in choices if d["entry"] >= six_start], end,
                            parent["lots"], initial=parent["capital"], slippage_bps=parent["primarySlippage"]))
           for b, choices in decisions.items()}
    checks = {}
    for slip in ["5.0", "10.0"]:
        tested = scenarios[slip]["payoff_dual"]
        checks[f"positive_{slip}bps"] = tested["netReturn"] > 0
        checks[f"beats_all_ten_baselines_{slip}bps"] = all(
            tested["ending"] > r["ending"] for r in baseline_replays[slip].values())
    checks["drawdown_no_worse_than_original_mean"] = (
        candidate["dailyCloseDrawdown"] >= old_replays[primary_key]["ensemble_mean_only"]["dailyCloseDrawdown"])
    checks["all_excess_intervals_positive"] = all(v is not None and v["lower95"] > 0 for v in intervals.values())
    enough = measures["count"] == measures["reference"]["count"] and measures["count"] > 0
    predictive = {
        "same_scored_coverage": enough,
        "raw_mae_better": enough and measures["mae"]["mean"] <
            min(measures["mae"]["zero"], measures["reference"]["mae"]["mean"]),
        "brier_better_than_baselines": enough and measures["brier"]["proxyProbability"] <
            min(measures["brier"]["pastProbability"], measures["brier"]["half"]),
        "paired_mae_interval_positive": measures["pairedMae"] is not None and measures["pairedMae"]["lower95"] > 0,
        "unchanged_downside": enough and abs(measures["q10Pinball"] - measures["reference"]["pinball"]["q10"]) < 1e-12}
    return {"version": SPEC["version"], "createdAt": datetime.now(timezone.utc).isoformat(), "lineage": lineage,
        "fitCount": sum(f["status"] == "fitted" for f in fits), "scheduledDecisions": len(fits),
        "start": parent["start"], "end": parent["end"], "capital": parent["capital"],
        "predictionMetrics": measures, "predictiveAcceptanceChecks": predictive,
        "predictiveAcceptancePassed": all(predictive.values()), "economicAcceptanceChecks": checks,
        "economicAcceptancePassed": all(checks.values()), "livePromotionAllowed": False,
        "comparison": {s: {**{b: summary(r) for b, r in baseline_replays[s].items()},
                           **{b: summary(r) for b, r in books.items()}} for s, books in scenarios.items()},
        "excessIntervals": intervals, "sixMonthRestart": {"start": six_start, "end": parent["end"],
            "books": {**normalized[2]["sixMonthRestart"]["books"], **six}},
        "limitations": reference["limitations"] + [
            "Cost proxy uses adjusted returns and a fixed reference ticket; probability is not verified account net profit.",
            "Conditional heads plus calibrated expectation are a new hypothesis on the same viewed twelve-stock cohort.",
            "Saved original downside forecasts are reused; no baseline models are refitted or unpickled.",
            "Only a few scored dates; paired bootstrap diagnostics do not correct repeated research selection.",
            "No current-data, broad-universe, prospective profitability or live-promotion claim."]}


def run_experiment(protocol_path):
    path = Path(protocol_path).resolve()
    protocol = json.loads(path.read_text())
    original, normalized = verify_protocol(protocol)
    lineage = {"protocolSha256": base.sha(path), "code": code_hashes(), "references": protocol["references"],
               "referenceCode": original[2]["lineage"]["code"], "sklearn": sklearn.__version__,
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
                raise ValueError("Completed run changed; declare a new protocol")
            for name, digest in saved["artifacts"].items():
                artifact = (out / name).resolve()
                if not artifact.is_relative_to(out) or base.sha(artifact) != digest:
                    raise ValueError("Completed artifact hash mismatch")
            return {**json.loads((out / "results.json").read_text()), "cacheHit": True}
        if (out / "started.json").exists():
            raise ValueError("Incomplete attempt preserved; declare a continuation")
        atomic_json(out / "started.json", {**lineage, "fingerprint": fingerprint,
                                         "startedAt": datetime.now(timezone.utc).isoformat()})
        try:
            result = evaluate(original, normalized, out, lineage)
            verify_protocol(protocol)
            atomic_json(out / "results.json", result)
            artifacts = {p.name: base.sha(p) for p in out.iterdir() if p.is_file() and p.name != ".lock"}
            atomic_json(completed, {"fingerprint": fingerprint, "artifacts": artifacts})
            return {**result, "cacheHit": False}
        except Exception as exc:
            atomic_json(out / "failure.json", {"type": type(exc).__name__, "error": str(exc),
                                              "at": datetime.now(timezone.utc).isoformat()})
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("protocol", type=Path)
    result = run_experiment(parser.parse_args().protocol)
    print(json.dumps({k: result[k] for k in ["version", "fitCount", "cacheHit", "predictionMetrics",
        "predictiveAcceptancePassed", "economicAcceptancePassed", "comparison"]}, indent=2))
