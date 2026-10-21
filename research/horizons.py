"""Frozen multi-horizon prediction diagnostics, never an account-profit replay.

python -m research.horizons /private/path/PLAN.json
All horizons and blocked/pending cases remain visible. No live acquisition.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from research.architecture import load_frames, sha
from research.engine import atomic_json
from research.panel import STOCK_FEATURES, session_panel
from research.prospective import job_lock

SPEC = {"horizons": [20, 63, 126, 252], "signalStride": 63,
        "minimumTrainDates": 252, "maximumTrainDates": 756,
        "ridgeAlpha": 100.0, "minimumCandidates": 4}


def fit_snapshot(panel, signal):
    signal = pd.Timestamp(signal)
    snapshot = panel[(panel.date == signal) & panel.featureReady].copy()
    past = panel[(panel.date < signal) & (panel.labelEnd < signal)
                 & panel.featureReady & np.isfinite(panel.actual)].copy()
    past = past[past.groupby("date").symbol.transform("size") >= SPEC["minimumCandidates"]]
    dates = sorted(past.date.unique())[-SPEC["maximumTrainDates"]:]
    train = past[past.date.isin(dates)]
    lineage = {"signal": str(signal.date()), "trainDates": len(dates),
               "trainRows": len(train), "candidates": len(snapshot)}
    if len(dates) < SPEC["minimumTrainDates"] or len(snapshot) < SPEC["minimumCandidates"]:
        return snapshot, {**lineage, "status": "insufficient_history_or_candidates"}, None
    weights = 1 / train.groupby("date").symbol.transform("size").to_numpy()
    weights /= weights.mean()
    model = make_pipeline(StandardScaler(), Ridge(alpha=SPEC["ridgeAlpha"]))
    with threadpool_limits(limits=2):
        model.fit(train[STOCK_FEATURES], train.actual,
                  standardscaler__sample_weight=weights, ridge__sample_weight=weights)
        snapshot["prediction"] = model.predict(snapshot[STOCK_FEATURES])
    snapshot["pastMean"] = float(train.groupby("date").actual.mean().mean())
    lineage.update(status="fitted", trainStart=str(train.date.min().date()),
                   lastTrainOutcome=str(train.labelEnd.max().date()))
    scaler, ridge = model.steps[0][1], model.steps[1][1]
    parameters = {"features": STOCK_FEATURES, "scaleMean": scaler.mean_.tolist(),
                  "scale": scaler.scale_.tolist(), "coefficient": ridge.coef_.tolist(),
                  "intercept": float(ridge.intercept_)}
    return snapshot, lineage, parameters


def metrics(rows):
    scored = rows.dropna(subset=["actual", "prediction", "pastMean"])
    if scored.empty:
        return {"rows": 0, "dates": 0}
    mae = {name: float((scored.actual - value).abs().mean()) for name, value in
           [("ridge", scored.prediction), ("zero", 0.), ("pastMean", scored.pastMean)]}
    return {"rows": len(scored), "dates": int(scored.date.nunique()), "mae": mae,
            "meanCohortAdjustedOutcome": float(scored.groupby("date").actual.mean().mean()),
            "belowBothBaselines": mae["ridge"] < min(mae["zero"], mae["pastMean"])}


def evaluate(frames, sessions, plan):
    signals = sessions[(sessions >= plan["start"]) & (sessions <= plan["end"])][::SPEC["signalStride"]]
    if len(signals) == 0:
        raise ValueError("No declared signal sessions")
    forecasts, fits, models, coverage = [], [], {}, {}
    for horizon in SPEC["horizons"]:
        panel, manifest = session_panel(frames, sessions, calendar_source=plan["calendarSource"],
                                        price_basis=plan["priceBasis"], horizon=horizon)
        coverage[str(horizon)] = manifest
        for signal in signals:
            snapshot, fit, parameters = fit_snapshot(panel, signal)
            fits.append({"horizon": horizon, **fit})
            if parameters is not None:
                models[f"{horizon}/{signal.date()}"] = parameters
            # Preserve non-ready instruments and absent predictions alongside
            # candidates; future outcome availability never determines selection.
            full = panel[panel.date == signal].copy()
            if parameters is not None:
                full = full.merge(snapshot[["symbol", "prediction", "pastMean"]], on="symbol", how="left")
            else:
                full["prediction"], full["pastMean"] = np.nan, np.nan
            full["horizon"] = horizon
            forecasts.append(full[["date", "symbol", "horizon", "featureStatus", "labelStatus",
                                   "labelEnd", "actual", "prediction", "pastMean"]])
    rows = pd.concat(forecasts, ignore_index=True)
    summary = {}
    for horizon, group in rows.groupby("horizon"):
        summary[str(horizon)] = {**metrics(group),
            "bySignalYear": {str(year): metrics(part) for year, part in group.groupby(group.date.dt.year)},
            "pendingOrMissingOutcomes": int(group.actual.isna().sum()),
            "unfittedOrUnreadyRows": int(group.prediction.isna().sum())}
    payload = rows.copy()
    for column in ["date", "labelEnd"]:
        payload[column] = payload[column].dt.strftime("%Y-%m-%d")
    return {"summary": summary, "fitCount": len(models), "signalDates": signals.strftime("%Y-%m-%d").tolist(),
            "livePromotionAllowed": False, "portfolioProfitEvaluated": False}, {
            "forecasts.json": json.loads(payload.to_json(orient="records", double_precision=15)),
            "fits.json": fits, "models.json": models, "coverage.json": coverage}


def run(plan_path):
    path = Path(plan_path).resolve()
    plan = json.loads(path.read_text())
    frozen = datetime.fromisoformat(plan["frozenAt"])
    if (plan["version"] != "long-horizon-study-1" or plan["spec"] != SPEC
            or frozen.tzinfo is None or frozen > datetime.now(timezone.utc)):
        raise ValueError("Invalid or mismatched frozen plan")
    inputs = plan["featureInputs"] + plan["parents"]
    for entry in inputs:
        if sha(entry["path"]) != entry["sha256"]:
            raise ValueError("Input hash mismatch")
    code = {name: sha(Path(__file__).with_name(name)) for name in
            ["horizons.py", "panel.py", "architecture.py", "engine.py", "prospective.py"]}
    lineage = {"planSha256": sha(path), "code": code, "sklearn": sklearn.__version__,
               "numpy": np.__version__, "pandas": pd.__version__}
    fingerprint = hashlib.sha256(json.dumps(lineage, sort_keys=True).encode()).hexdigest()
    out = path.parent / "run"
    out.mkdir(mode=0o700, exist_ok=True)
    with job_lock(out):
        completed = out / "completed.json"
        if completed.exists():
            record = json.loads(completed.read_text())
            if record["fingerprint"] != fingerprint:
                raise ValueError("Run lineage mismatch; preserve and declare a new experiment")
            for name, digest in record["artifacts"].items():
                if sha(out / name) != digest:
                    raise ValueError("Artifact hash mismatch")
            return {**json.loads((out / "results.json").read_text()), "cacheHit": True}
        if (out / "started.json").exists():
            raise ValueError("Incomplete run preserved; inspect before a declared continuation")
        frames = load_frames(plan["featureInputs"])
        sessions = pd.DatetimeIndex(sorted(set().union(*(set(f.index) for f in frames.values()))))
        excluded = pd.to_datetime(plan["calendarExcluded"])
        sessions = sessions[~sessions.isin(excluded)]
        frames = {s: f.loc[~f.index.isin(excluded)] for s, f in frames.items()}
        atomic_json(out / "started.json", {**lineage, "fingerprint": fingerprint,
                    "startedAt": datetime.now(timezone.utc).isoformat()})
        result, artifacts = evaluate(frames, sessions, plan)
        # Do not publish a completion if any input changed during the fit.
        if sha(path) != lineage["planSha256"] or any(sha(e["path"]) != e["sha256"] for e in inputs):
            raise ValueError("Inputs changed during evaluation")
        result.update(lineage=lineage, limitations=plan["limitations"], cacheHit=False)
        artifacts["results.json"] = result
        for name, value in artifacts.items():
            atomic_json(out / name, value)
        atomic_json(completed, {"fingerprint": fingerprint,
                    "artifacts": {name: sha(out / name) for name in artifacts}})
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    print(json.dumps(run(parser.parse_args().plan), indent=2))
