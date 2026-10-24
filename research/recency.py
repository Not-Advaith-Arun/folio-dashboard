"""One frozen recency-weighted Ridge; exploratory local CPU research only.

python -m research.recency /private/path/protocol.json
Reuse old verified JSON, never refit or deserialize a reference model.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from research import architecture as base, components
from research.calibration_evidence import write_evidence
from research.engine import atomic_json
from research.execution import validated_calendar
from research.payoff import load_reference, BOUNDARIES, reference_cost
from research.prospective import job_lock
from research.rebalance import replay_rebalanced

SPEC = {"version": "recency-ridge-20-1", "horizon": 20, "halfLifeSessions": 126, "ridgeAlpha": 100.}
BOOKS = {"recency_dual": "dual_head", "recency_mean": "ensemble_mean_only"}
CODE = list(dict.fromkeys(components.CODE + ["recency.py", "calibration_evidence.py"]))
CONTROLS = {"original": {"uniformRidge": "ridge", "originalMean": "mean", "originalTree": "tree", "pastMean": "pastMean"},
    "normalized": {"normalizedMean": "mean", "normalizedRidge": "ridge"},
    "payoff": {"payoffMean": "mean"}, "component": {"componentMean": "componentMean", "directMean": "directMean"}}


def code_hashes():
    return {n: base.sha(Path(__file__).with_name(n)) for n in CODE}


def recency_weights(frame, sessions):
    positions = sessions.get_indexer(pd.DatetimeIndex(frame.date))
    if (positions < 0).any():
        raise ValueError("Training date outside frozen sessions")
    ages = positions.max() - positions
    weights = np.exp2(-ages / SPEC["halfLifeSessions"]) / frame.groupby("date").symbol.transform("size").to_numpy()
    weights /= weights.mean()
    mass = pd.Series(weights, index=frame.index).groupby(frame.date).sum().to_numpy()
    return weights, float(mass.sum() ** 2 / np.square(mass).sum())


def fit_weighted(panel, signal, sessions):
    signal = pd.Timestamp(signal)
    snapshot = panel[(panel.date == signal) & panel.featureReady].copy().sort_values("symbol")
    if len(snapshot) < base.SPEC["minimumCandidates"]:
        return None, {"status": "insufficient_current_candidates"}, None, None
    past = panel[(panel.date < signal) & (panel.labelEnd < signal) & panel.featureReady
                 & np.isfinite(panel.actual) & np.isfinite(panel[base.FEATURES]).all(axis=1)].copy()
    past = past[past.groupby("date").symbol.transform("size") >= base.SPEC["minimumCandidates"]]
    dates = sorted(past.date.unique())
    if len(dates) < base.SPEC["minimumTrainDates"] + base.SPEC["calibrationDates"]:
        return None, {"status": "insufficient_history"}, None, None
    cal_dates = dates[-base.SPEC["calibrationDates"]:]
    start = pd.Timestamp(cal_dates[0])
    train = past[(past.date < start) & (past.labelEnd < start)].copy()
    train_dates = sorted(train.date.unique())[-base.SPEC["maximumTrainDates"]:]
    train = train[train.date.isin(train_dates)].copy()
    if len(train_dates) < base.SPEC["minimumTrainDates"]:
        return None, {"status": "insufficient_purged_history"}, None, None
    calibration = past[past.date.isin(cal_dates)].copy()
    weights, ess = recency_weights(train, sessions)
    model = make_pipeline(StandardScaler(), Ridge(alpha=SPEC["ridgeAlpha"]))
    with threadpool_limits(limits=2):
        model.fit(train[base.FEATURES], train.actual, standardscaler__sample_weight=weights, ridge__sample_weight=weights)
        for f in [train, calibration, snapshot]:
            f["rawWeighted"] = model.predict(f[base.FEATURES])
    bias = float((calibration.actual - calibration.rawWeighted).groupby(calibration.date).mean().mean())
    for f in [train, calibration, snapshot]:
        f["weightedMean"] = f.rawWeighted + bias
        if not np.isfinite(f[["weightedMean", "rawWeighted"]]).all(axis=None):
            raise ValueError("Nonfinite unmodified weighted forecast")
    lineage = {"status": "fitted", "trainRows": len(train), "trainDates": len(train_dates),
        "calibrationRows": len(calibration), "calibrationDates": len(cal_dates),
        "trainStart": str(train.date.min().date()), "trainLastOutcome": str(train.labelEnd.max().date()),
        "calibrationStart": str(start.date()), "calibrationLastOutcome": str(calibration.labelEnd.max().date()),
        "bias": bias, "dateWeightEffectiveSize": ess, "features": base.FEATURES, "spec": SPEC,
        "fittedAt": datetime.now(timezone.utc).isoformat()}
    return snapshot, lineage, {"ridge": model, "lineage": lineage}, {"train": train, "calibration": calibration}


def capture(out, protocol, original, fit, frames, model_sha):
    marker = lambda value: pd.Timestamp(value).tz_localize("UTC").isoformat()
    records = []
    for part, frame in frames.items():
        for row in frame.itertuples():
            records.append({"partition": part, "symbol": row.symbol, "session": str(row.date.date()),
                "featureAt": marker(row.date), "entryAt": marker(row.entryDate), "labelEndAt": marker(row.labelEnd),
                "featureObservedAt": None, "labelObservedAt": None,
                "raw": float(row.rawWeighted), "calibrated": float(row.weightedMean), "label": float(row.actual),
                "cost": reference_cost(), "risk": None, "sourceIds": [row.symbol],
                "missing": {"featureObservedAt": "not_recorded", "labelObservedAt": "not_recorded", "risk": "unavailable"}})
    contract = {"version": "calibration-evidence-1", "experimentId": SPEC["version"], "mode": "retrospective",
        "sessionTimezone": "UTC", "signalAt": marker(fit["signal"]), "issuedAt": datetime.now(timezone.utc).isoformat(),
        "protocol": {"sha256": base.sha(protocol), "frozenAt": json.loads(Path(protocol).read_text())["frozenAt"]},
        "model": {"sha256": model_sha, "fittedAt": fit["fittedAt"]},
        "target": {"name": "20-session-open-to-open", "functional": "mean", "score": "squared_error", "quantile": None, "unit": "return_fraction"},
        "sources": [{"id": e["symbol"], "sha256": e["sha256"], "receiptSha256": None, "firstSeenAt": None,
                     "availability": "unknown"} for e in original["featureInputs"]],
        "expectedRows": {"train": len(frames["train"]), "calibration": len(frames["calibration"]), "forecast": 0}}
    parent = out / "calibration"
    parent.mkdir(mode=0o700, exist_ok=True)
    return write_evidence(parent / fit["signal"], contract, records)


def verify(protocol):
    if protocol["spec"] != SPEC or protocol["books"] != list(BOOKS) or protocol["code"] != code_hashes():
        raise ValueError("Frozen recency protocol/code changed")
    frozen = datetime.fromisoformat(protocol["frozenAt"])
    if frozen.tzinfo is None or frozen > datetime.now(timezone.utc) or base.sha(protocol["plan"]["path"]) != protocol["plan"]["sha256"]:
        raise ValueError("Invalid recency declaration")
    plan = json.loads(Path(protocol["plan"]["path"]).read_text())
    if plan["references"] != protocol["references"] or plan["books"] != list(BOOKS):
        raise ValueError("Plan ancestry changed")
    for key in ["horizon", "halfLifeSessions", "ridgeAlpha"]:
        if plan["spec"][key] != SPEC[key]:
            raise ValueError("Unplanned recency parameter")
    if any(base.sha(e["path"]) != e["sha256"] for e in plan.get("sources", [])):
        raise ValueError("Research source changed")
    refs = {k: load_reference(v) for k, v in protocol["references"].items()}
    if refs["component"][0]["references"] != {k: protocol["references"][k] for k in ["original", "normalized", "payoff"]}:
        raise ValueError("Unmatched baseline ancestry")
    components.verify(refs["component"][0])
    base.verify_inputs(refs["original"][0])
    return refs


def error_interval(rows):
    f = pd.DataFrame(rows)
    daily = f.assign(delta=(f.actual - f.uniformRidge) ** 2 - (f.actual - f.weightedMean) ** 2).groupby("date").delta.mean().to_numpy()
    if len(daily) < 4:
        return None
    rng, draws = np.random.default_rng(42), []
    for _ in range(1000):
        starts = rng.integers(0, len(daily) - 1, int(np.ceil(len(daily) / 2)))
        draws.append(float(np.concatenate([daily[i:i + 2] for i in starts])[:len(daily)].mean()))
    low, high = np.quantile(draws, [.025, .975])
    return {"meanImprovement": float(daily.mean()), "lower95": float(low), "upper95": float(high),
        "signalDates": len(daily), "blockSignalDates": 2, "repetitions": 1000}


def evaluate(refs, out, lineage, protocol_path):
    original, old_dir, previous = refs["original"]
    frames, raw = base.load_frames(original["featureInputs"]), base.load_frames(original["executionInputs"])
    sessions = pd.DatetimeIndex(sorted(set.union(*(set(f.index) for f in frames.values()))))
    excluded = pd.to_datetime(original["calendarExcluded"])
    raw = {s: f.loc[~f.index.isin(excluded)].copy() for s, f in raw.items()}
    start, end = pd.Timestamp(original["start"]), pd.Timestamp(original["end"])
    if (list(sessions[(sessions >= start) & (sessions <= end)]) != validated_calendar(raw, start, end)
            or sessions.isin(excluded).any() or any(f.loc[start:end, "splits"].ne(0).any() for f in raw.values())):
        raise ValueError("Original calendar/action assumptions changed")
    panel, coverage = base.architecture_panel(frames, sessions, calendar_source=original["calendarSource"], price_basis=original["priceBasis"])
    atomic_json(out / "coverage.json", coverage)
    old = {name: {(r["date"], r["symbol"]): r for r in json.loads((d / "forecasts.json").read_text())}
           for name, (_, d, _) in refs.items()}
    old_fits = json.loads((old_dir / "fits.json").read_text())
    decisions, forecasts, fits, receipts = {b: [] for b in BOOKS}, [], [], []
    for old_fit, decision in zip(old_fits, json.loads((old_dir / "decisions.json").read_text())["cash"], strict=True):
        signal = pd.Timestamp(decision["signal"])
        snapshot, fit, model, paths = fit_weighted(panel, signal, sessions)
        fit.update(signal=decision["signal"], entry=decision["entry"])
        if fit["status"] != old_fit["status"] or fit["status"] == "fitted" and any(fit[k] != old_fit[k] for k in BOUNDARIES):
            raise ValueError("Original partition changed")
        fits.append(fit)
        model_sha = None
        if model is not None:
            name = out / f"model-{signal.date()}.pickle"
            with name.open("xb") as f:
                os.chmod(name, 0o600)
                pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
            model_sha = base.sha(name)
            snapshot["q10"] = [old["original"][(decision["signal"], s)]["q10"] for s in snapshot.symbol]
            receipts.append({"signal": decision["signal"], **capture(out, protocol_path, original, fit, paths, model_sha)})
        for book, allocator in BOOKS.items():
            selected = None if snapshot is None else snapshot.assign(mean=snapshot.weightedMean)
            decisions[book].append({**decision, "weights": base.allocate(selected, allocator)})
        predictions = {} if snapshot is None else snapshot.set_index("symbol").to_dict("index")
        for row in panel[panel.date == signal].itertuples():
            key = (decision["signal"], row.symbol)
            saved = old["original"][key]
            observed = pd.notna(row.labelEnd) and row.labelEnd <= end and np.isfinite(row.actual)
            actual = float(row.actual) if observed else None
            if actual != saved["actual"] or row.featureStatus != saved["featureStatus"]:
                raise ValueError("Reference outcome/availability changed")
            p = predictions.get(row.symbol, {})
            controls = {name: old[ref][key][column] for ref, mapping in CONTROLS.items() for name, column in mapping.items()}
            forecasts.append({"date": decision["signal"], "symbol": row.symbol, "actual": actual,
                "featureStatus": row.featureStatus, "fitStatus": fit["status"], "modelSha256": model_sha,
                "weightedMean": float(p["weightedMean"]) if p else None, "rawWeighted": float(p["rawWeighted"]) if p else None,
                "q10": saved["q10"], "zero": 0., **controls})
    for name, data in [("forecasts", forecasts), ("fits", fits), ("decisions", decisions), ("evidence-receipts", receipts)]:
        atomic_json(out / (name + ".json"), data)
    scored = [r for r in forecasts if r["actual"] is not None and r["weightedMean"] is not None]
    names = ["weightedMean", "zero"] + [n for mapping in CONTROLS.values() for n in mapping]
    if any(r[n] is None for r in scored for n in names):
        raise ValueError("Missing matched point comparator")
    metrics = {"count": len(scored), "signalDates": len({r["date"] for r in scored}),
        "mse": {n: float(np.mean([(r["actual"] - r[n]) ** 2 for r in scored])) for n in names},
        "mae": {n: float(np.mean([abs(r["actual"] - r[n]) for r in scored])) for n in names}}
    paired = error_interval(scored)
    predictive = {"same_scored_coverage": len(scored) == previous["predictionMetrics"]["count"],
        "mse_below_all_controls": metrics["mse"]["weightedMean"] < min(v for k, v in metrics["mse"].items() if k != "weightedMean"),
        "paired_mse_interval_positive": paired is not None and paired["lower95"] > 0}
    replays = {str(s): {b: replay_rebalanced(raw, d, end, original["lots"], initial=original["capital"], slippage_bps=s)
                       for b, d in decisions.items()} for s in original["slippageScenarios"]}
    atomic_json(out / "replays.json", replays)
    comparison = {s: {**{b: v for _, _, result in refs.values() for b, v in result["comparison"][s].items()},
        **{b: {k: v[k] for k in ["ending", "netReturn", "dailyCloseDrawdown", "fees"]} for b, v in books.items()}}
        for s, books in replays.items()}
    primary = str(original["primarySlippage"])
    old_replays = json.loads((old_dir / "replays.json").read_text())
    intervals = {b: base.excess_interval(replays[primary]["recency_dual"], old_replays[primary][b])
                 for b in ["cash", "ridge_mean_only", "equal_buy_hold"]}
    economic = {}
    for s in ["5.0", "10.0"]:
        candidate = comparison[s]["recency_dual"]
        economic["positive_" + s] = candidate["netReturn"] > 0
        economic["beats_all_old_books_" + s] = all(candidate["ending"] > v["ending"] for b, v in comparison[s].items() if b not in BOOKS)
    economic["drawdown_no_worse"] = comparison[primary]["recency_dual"]["dailyCloseDrawdown"] >= comparison[primary]["ensemble_mean_only"]["dailyCloseDrawdown"]
    economic["excess_intervals_positive"] = all(v is not None and v["lower95"] > 0 for v in intervals.values())
    six_start = previous["sixMonthRestart"]["start"]
    six = {b: replay_rebalanced(raw, [d for d in ds if d["entry"] >= six_start], end, original["lots"],
        initial=original["capital"], slippage_bps=original["primarySlippage"]) for b, ds in decisions.items()}
    return {"version": SPEC["version"], "lineage": lineage, "fitCount": sum(f["status"] == "fitted" for f in fits),
        "predictionMetrics": metrics, "pairedMseImprovement": paired, "predictiveAcceptanceChecks": predictive,
        "predictiveAcceptancePassed": all(predictive.values()), "economicAcceptanceChecks": economic,
        "economicAcceptancePassed": all(economic.values()), "comparison": comparison, "excessIntervals": intervals,
        "sixMonthRestart": {"start": six_start, "books": {**{b: v for _, _, result in refs.values() for b, v in result["sixMonthRestart"]["books"].items()},
            **{b: {k: v[k] for k in ["ending", "netReturn", "fees", "dailyCloseDrawdown"]} for b, v in six.items()}}},
        "limitations": previous["limitations"] + ["One fixed126-session recency hypothesis, not estimated optimal break weights.",
            "Date-weight effective size is not independent information with overlapping labels.",
            "Calibration seals use retrospective unknown availability and UTC date markers, never verified exchange event times.",
            "Old uniform-control calibration paths remain unavailable; no baseline was refitted to obtain them."], "livePromotionAllowed": False}


def run(path):
    path = Path(path).resolve()
    protocol = json.loads(path.read_text())
    refs = verify(protocol)
    lineage = {"protocolSha256": base.sha(path), "code": protocol["code"], **{k: refs["original"][2]["lineage"][k] for k in ["numpy", "pandas", "sklearn"]}}
    fingerprint = hashlib.sha256(json.dumps(lineage, sort_keys=True).encode()).hexdigest()
    out = path.parent / "run"
    out.mkdir(mode=0o700, exist_ok=True)
    with job_lock(out):
        if (out / "completed.json").exists():
            saved = json.loads((out / "completed.json").read_text())
            if saved["fingerprint"] != fingerprint or any(base.sha(out / n) != h for n, h in saved["artifacts"].items()):
                raise ValueError("Immutable recency result changed")
            return {**json.loads((out / "results.json").read_text()), "cacheHit": True}
        if (out / "started.json").exists():
            raise ValueError("Incomplete attempt preserved")
        atomic_json(out / "started.json", {**lineage, "startedAt": datetime.now(timezone.utc).isoformat()})
        try:
            result = evaluate(refs, out, lineage, path)
            verify(protocol)
            atomic_json(out / "results.json", result)
            atomic_json(out / "completed.json", {"fingerprint": fingerprint, "artifacts": {
                str(p.relative_to(out)): base.sha(p) for p in out.rglob("*") if p.is_file() and p.name != ".lock"}})
            return {**result, "cacheHit": False}
        except Exception as exc:
            atomic_json(out / "failure.json", {"type": type(exc).__name__, "error": str(exc)})
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("protocol", type=Path)
    result = run(parser.parse_args().protocol)
    print(json.dumps({k: result[k] for k in ["fitCount", "predictionMetrics", "predictiveAcceptancePassed", "economicAcceptancePassed", "cacheHit"]}))
