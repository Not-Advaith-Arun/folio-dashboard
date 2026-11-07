"""Fixed cohort-dispersion input with shared Ridge; offline research only.

python -m research.dispersion /private/path/protocol.json
Reuse old verified JSON, never refit or deserialize a reference model.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
import stat
import platform
import math
import fcntl
import sklearn
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from research import architecture as base, components, recency, intercepts, slopes, replay_evidence
from research.calibration_evidence import write_evidence
from research.engine import atomic_json
from research.execution import validated_calendar, FEE_SOURCE, FEE_CHECKED
from research.payoff import load_reference, BOUNDARIES, reference_cost
from research.prospective import job_lock
from research.rebalance import replay_rebalanced

SPEC = {"version": "cohort-dispersion-20-1", "horizon": 20, "ridgeAlpha": 100.}
BOOKS = {"cohort_dispersion_dual": "dual_head", "cohort_dispersion_mean": "ensemble_mean_only"}
CODE = list(dict.fromkeys(slopes.CODE + ["dispersion.py", "test_dispersion_workflow.py", "replay_evidence.py"]))
CONTROLS = {"original": {"uniformRidge": "ridge", "originalMean": "mean", "originalTree": "tree", "pastMean": "pastMean"},
    "normalized": {"normalizedMean": "mean", "normalizedRidge": "ridge"},
    "payoff": {"payoffMean": "mean"}, "component": {"componentMean": "componentMean", "directMean": "directMean"},
    "recency": {"recencyMean": "weightedMean"},
    "stockIntercept": {"stockInterceptMean": "interceptMean"},
    "stockSlope": {"stockSlopeMean": "slopeMean"}}


FEATURES = base.FEATURES + ["cohortDispersion20"]


def dispersion_panel(frames, sessions, symbols, **kwargs):
    """Keep original readiness/status; a missing cohort never changes membership."""
    if len(symbols) != 12 or len(set(symbols)) != 12 or set(frames) != set(symbols):
        raise ValueError("Exactly twelve known stocks required")
    panel, coverage = base.architecture_panel(frames, sessions, **kwargs)
    values = {}
    for day, group in panel[panel.featureReady].groupby("date"):
        if set(group.symbol) != set(symbols) or len(group) != 12:
            continue
        r = group.set_index("symbol").loc[list(symbols), "return20"].to_numpy(dtype=float)
        if not np.isfinite(r).all():
            continue
        centered = [float(v - r[0]) for v in r]
        mean = math.fsum(centered) / 12
        value = math.sqrt(math.fsum((v - mean) ** 2 for v in centered) / 12)
        if not math.isfinite(value):
            raise ValueError("Nonfinite fixed cohort dispersion")
        values[day] = value
    panel["cohortDispersion20"] = panel.date.map(values).astype(float)
    coverage["candidateFeatures"] = FEATURES
    coverage["dispersionCohort"] = list(symbols)
    coverage["dispersionReadyDates"] = len(values)
    return panel, coverage


def code_hashes():
    return {n: base.sha(Path(__file__).with_name(n)) for n in CODE}


class DispersionRidge:
    """One weighted scaler and shared15 slopes; no stock encoding or selection."""
    def __init__(self, symbols):
        self.symbols = tuple(symbols)
        if len(self.symbols) != 12 or len(set(self.symbols)) != 12:
            raise ValueError("Exactly twelve unique stock identities required")
        self.scaler = StandardScaler()
        self.ridge = Ridge(alpha=SPEC["ridgeAlpha"], fit_intercept=True)

    def design(self, frame):
        if not set(frame.symbol).issubset(self.symbols):
            raise ValueError("Unknown stock identity")
        x = frame[FEATURES].to_numpy(dtype=float)
        if not np.isfinite(x).all():
            raise ValueError("Incomplete fixed cohort or nonfinite feature")
        return self.scaler.transform(x)

    def fit(self, frame, weights):
        if set(frame.symbol) != set(self.symbols):
            raise ValueError("Missing training stock identity")
        x = frame[FEATURES].to_numpy(dtype=float)
        if not np.isfinite(x).all():
            raise ValueError("Incomplete fixed cohort or nonfinite feature")
        self.scaler.fit(x, sample_weight=weights)
        self.ridge.fit(self.design(frame), frame.actual, sample_weight=weights)
        return self

    def predict(self, frame):
        return self.ridge.predict(self.design(frame))


def membership(frames):
    return sorted((part, row.symbol, str(row.date.date()), str(row.entryDate.date()), str(row.labelEnd.date()))
                  for part, frame in frames.items() for row in frame.itertuples())


def fit_dispersion(panel, signal, sessions, symbols, expected=None):
    signal = pd.Timestamp(signal)
    if not set(panel.symbol).issubset(symbols) or panel.duplicated(["date", "symbol"]).any():
        raise ValueError("Unknown/duplicate stock identity")
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
    paths = {"train": train, "calibration": calibration}
    if expected is not None:
        wanted = sorted((r["partition"], r["symbol"], r["session"], r["entryAt"][:10], r["labelEndAt"][:10])
                        for r in expected["memberships"])
        if membership(paths) != wanted:
            raise ValueError("Frozen membership changed before fitting")
    if any(not np.isfinite(f[FEATURES]).all(axis=None) for f in [train, calibration, snapshot]):
        raise ValueError("Incomplete fixed cohort in original membership before fitting")
    weights = 1. / train.groupby("date").symbol.transform("size").to_numpy()
    weights /= weights.mean()
    model = DispersionRidge(symbols)
    with threadpool_limits(limits=2):
        model.fit(train, weights)
        for f in [train, calibration, snapshot]:
            f["rawDispersion"] = model.predict(f)
    bias = float((calibration.actual - calibration.rawDispersion).groupby(calibration.date).mean().mean())
    for f in [train, calibration, snapshot]:
        f["dispersionMean"] = f.rawDispersion + bias
        if not np.isfinite(f[["dispersionMean", "rawDispersion"]]).all(axis=None):
            raise ValueError("Nonfinite unmodified cohort-dispersion forecast")
    lineage = {"status": "fitted", "trainRows": len(train), "trainDates": len(train_dates),
        "calibrationRows": len(calibration), "calibrationDates": len(cal_dates),
        "trainStart": str(train.date.min().date()), "trainLastOutcome": str(train.labelEnd.max().date()),
        "calibrationStart": str(start.date()), "calibrationLastOutcome": str(calibration.labelEnd.max().date()),
        "bias": bias, "features": FEATURES, "spec": SPEC,
        "parameters": {"symbols": list(symbols), "scaleMean": model.scaler.mean_.tolist(),
            "scaleScale": model.scaler.scale_.tolist(), "sharedSlopes": model.ridge.coef_.tolist(),
            "globalIntercept": float(model.ridge.intercept_)},
        "fittedAt": datetime.now(timezone.utc).isoformat()}
    return snapshot, lineage, {"ridge": model, "lineage": lineage}, paths


def capture(out, protocol, original, fit, frames, model_sha):
    marker = lambda value: pd.Timestamp(value).tz_localize("UTC").isoformat()
    records = []
    for part, frame in frames.items():
        for row in frame.itertuples():
            records.append({"partition": part, "symbol": row.symbol, "session": str(row.date.date()),
                "featureAt": marker(row.date), "entryAt": marker(row.entryDate), "labelEndAt": marker(row.labelEnd),
                "featureObservedAt": None, "labelObservedAt": None,
                "raw": float(row.rawDispersion), "calibrated": float(row.dispersionMean), "label": float(row.actual),
                "cost": reference_cost(), "risk": None, "sourceIds": [e["symbol"] for e in original["featureInputs"]],
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


def checked_pin(item):
    path = Path(item["path"])
    if not path.is_absolute() or any(p.is_symlink() for p in [path, *path.parents]):
        raise ValueError("Unsafe pinned path")
    if not path.is_file() or path.stat().st_size > 64 * 1024 * 1024 or base.sha(path) != item["sha256"]:
        raise ValueError("Pinned file changed or exceeds budget")
    return path


def verify(protocol):
    if protocol["spec"] != SPEC or protocol["books"] != list(BOOKS) or protocol["code"] != code_hashes():
        raise ValueError("Frozen cohort-dispersion protocol/code changed")
    frozen = datetime.fromisoformat(protocol["frozenAt"])
    if frozen.tzinfo is None or frozen > datetime.now(timezone.utc):
        raise ValueError("Invalid freeze timestamp")
    design = json.loads(checked_pin(protocol["design"]).read_text())
    plan = json.loads(checked_pin(protocol["plan"]).read_text())
    if (plan["design"] != protocol["design"] or design["referenceProtocols"] != protocol["references"]
            or design["membershipUniverse"] != protocol["symbols"] or design["gates"] != base.SPEC
            or design["spec"]["continuousFeatures"] != FEATURES
            or design["spec"]["estimator"]["alpha"] != SPEC["ridgeAlpha"]
            or design["experimentId"] != SPEC["version"]):
        raise ValueError("Fixed design or ancestry changed")
    if protocol["runtimePython"] != platform.python_version():
        raise ValueError("Frozen Python runtime changed")
    if protocol["runtime"] != {"numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__}:
        raise ValueError("Frozen runtime changed")
    def pins(value):
        if isinstance(value, dict):
            if set(value) >= {"path", "sha256"}:
                yield value
            else:
                for child in value.values():
                    yield from pins(child)
        elif isinstance(value, list):
            for child in value:
                yield from pins(child)
    entries = design.get("referenceJsonArtifacts", []) + design["featureInputs"] + design["executionInputs"] + design.get("parents", []) + list(pins(design.get("primarySources", [])))
    entries += [design["chronology"]["memberships"]]
    if len(entries) > 1024:
        raise ValueError("Pinned file count exceeds budget")
    size = 0
    for e in entries:
        size += checked_pin(e).stat().st_size
    if size > 512 * 1024 * 1024:
        raise ValueError("Pinned input bytes exceed budget")
    # This verifies old code/runtime/engine extraction compatibility, without
    # fitting or loading any model. Never rewrite old manifests to permit reuse.
    recent = load_reference(protocol["references"]["recency"])
    recency.verify(recent[0])
    refs = {k: load_reference(v) for k, v in protocol["references"].items()}
    if recent[0]["references"] != {k: v for k, v in protocol["references"].items() if k not in {"recency", "stockIntercept", "stockSlope"}}:
        raise ValueError("Unmatched recent control ancestry")
    intercepts.verify(refs["stockIntercept"][0])
    if refs["stockIntercept"][0]["references"] != {k: v for k, v in protocol["references"].items() if k not in {"stockIntercept", "stockSlope"}}:
        raise ValueError("Unmatched stock-intercept ancestry")
    slopes.verify(refs["stockSlope"][0])
    if refs["stockSlope"][0]["references"] != {k: v for k, v in protocol["references"].items() if k != "stockSlope"}:
        raise ValueError("Unmatched stock-slope ancestry")
    base.verify_inputs(refs["original"][0])
    return refs


def error_interval(rows, control):
    f = pd.DataFrame(rows)
    daily = f.assign(delta=(f.actual - f[control]) ** 2 - (f.actual - f.dispersionMean) ** 2).groupby("date").delta.mean().to_numpy()
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
    panel, coverage = dispersion_panel(frames, sessions, json.loads(Path(protocol_path).read_text())["symbols"], calendar_source=original["calendarSource"], price_basis=original["priceBasis"])
    atomic_json(out / "coverage.json", coverage)
    atomic_json(out / "dispersion-features.json", [{"date": str(row.date.date()),
        "cohortDispersion20": float(row.cohortDispersion20) if np.isfinite(row.cohortDispersion20) else None}
        for row in panel.drop_duplicates("date").itertuples()])
    old = {name: {(r["date"], r["symbol"]): r for r in json.loads((d / "forecasts.json").read_text())}
           for name, (_, d, _) in refs.items()}
    old_fits = json.loads((old_dir / "fits.json").read_text())
    decisions, forecasts, fits, receipts = {b: [] for b in BOOKS}, [], [], []
    for old_fit, decision in zip(old_fits, json.loads((old_dir / "decisions.json").read_text())["cash"], strict=True):
        signal = pd.Timestamp(decision["signal"])
        protocol = json.loads(Path(protocol_path).read_text())
        design = json.loads(Path(protocol["design"]["path"]).read_text())
        members = json.loads(checked_pin(design["chronology"]["memberships"]).read_text())
        expected = next((g for g in members if g["signal"] == decision["signal"]), None)
        if old_fit["status"] == "fitted" and expected is None:
            raise ValueError("Missing frozen membership before fitting")
        if "schedule" in design["chronology"]:
            declared = next((g for g in design["chronology"]["schedule"] if g["signal"] == decision["signal"]), None)
            if declared is None or any(old_fit.get(k) != v for k, v in declared.items() if k != "entry") or declared["entry"] != decision["entry"]:
                raise ValueError("Frozen signal schedule changed before fitting")
        snapshot, fit, model, paths = fit_dispersion(panel, signal, sessions, protocol["symbols"], expected)
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
            selected = None if snapshot is None else snapshot.assign(mean=snapshot.dispersionMean)
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
                "dispersionMean": float(p["dispersionMean"]) if p else None, "rawDispersion": float(p["rawDispersion"]) if p else None,
                "q10": saved["q10"], "zero": 0., **controls})
    for name, data in [("forecasts", forecasts), ("fits", fits), ("decisions", decisions), ("evidence-receipts", receipts)]:
        atomic_json(out / (name + ".json"), data)
    if len(forecasts) != design["chronology"]["expectedCandidates"] or sum(f["status"] == "fitted" for f in fits) != design["chronology"]["expectedNewFits"]:
        raise ValueError("Frozen candidate/fit count changed")
    scored = [r for r in forecasts if r["actual"] is not None and r["dispersionMean"] is not None]
    if len(scored) != design["chronology"]["expectedScoredRows"] or len({r["date"] for r in scored}) != design["chronology"]["expectedScoredDates"]:
        raise ValueError("Frozen score coverage changed")
    names = ["dispersionMean", "zero"] + [n for mapping in CONTROLS.values() for n in mapping]
    if any(r[n] is None for r in scored for n in names):
        raise ValueError("Missing matched point comparator")
    metrics = {"count": len(scored), "signalDates": len({r["date"] for r in scored}),
        "mse": {n: float(np.mean([(r["actual"] - r[n]) ** 2 for r in scored])) for n in names},
        "mae": {n: float(np.mean([abs(r["actual"] - r[n]) for r in scored])) for n in names}}
    paired = {n: error_interval(scored, n) for n in ["uniformRidge"]}
    predictive = {"same_scored_coverage": len(scored) == previous["predictionMetrics"]["count"],
        "mse_below_all_controls": metrics["mse"]["dispersionMean"] < min(v for k, v in metrics["mse"].items() if k != "dispersionMean"),
        "paired_mse_interval_positive": all(v is not None and v["lower95"] > 0 for v in paired.values())}
    replays = {str(s): {b: replay_rebalanced(raw, d, end, original["lots"], initial=original["capital"], slippage_bps=s)
                       for b, d in decisions.items()} for s in original["slippageScenarios"]}
    atomic_json(out / "replays.json", replays)
    comparison = {s: {**{b: v for _, _, result in refs.values() for b, v in result["comparison"][s].items()},
        **{b: {k: v[k] for k in ["ending", "netReturn", "dailyCloseDrawdown", "fees"]} for b, v in books.items()}}
        for s, books in replays.items()}
    primary = str(original["primarySlippage"])
    old_replays = json.loads((old_dir / "replays.json").read_text())
    intervals = {b: base.excess_interval(replays[primary]["cohort_dispersion_dual"], old_replays[primary][b])
                 for b in ["cash", "ridge_mean_only", "equal_buy_hold"]}
    intercept_replays = json.loads((refs["stockIntercept"][1] / "replays.json").read_text())
    intervals["stock_intercept_dual"] = base.excess_interval(
        replays[primary]["cohort_dispersion_dual"], intercept_replays[primary]["stock_intercept_dual"])
    economic = {}
    for s in ["5.0", "10.0"]:
        candidate = comparison[s]["cohort_dispersion_dual"]
        economic["positive_" + s] = candidate["netReturn"] > 0
        economic["beats_all_old_books_" + s] = all(candidate["ending"] > v["ending"] for b, v in comparison[s].items() if b not in BOOKS)
    economic["drawdown_no_worse"] = comparison[primary]["cohort_dispersion_dual"]["dailyCloseDrawdown"] >= comparison[primary]["ensemble_mean_only"]["dailyCloseDrawdown"]
    economic["excess_intervals_positive"] = all(v is not None and v["lower95"] > 0 for v in intervals.values())
    six_start = previous["sixMonthRestart"]["start"]
    six = {b: replay_rebalanced(raw, [d for d in ds if d["entry"] >= six_start], end, original["lots"],
        initial=original["capital"], slippage_bps=original["primarySlippage"]) for b, ds in decisions.items()}
    atomic_json(out / "restart-replays.json", six)
    replay_receipt = capture_replays(out, protocol_path, original, decisions, replays, six, six_start)
    atomic_json(out / "replay-evidence-receipt.json", replay_receipt)
    return {"version": SPEC["version"], "lineage": lineage, "fitCount": sum(f["status"] == "fitted" for f in fits),
        "replayEvidence": replay_receipt, "predictionMetrics": metrics, "pairedMseImprovement": paired, "predictiveAcceptanceChecks": predictive,
        "predictiveAcceptancePassed": all(predictive.values()), "economicAcceptanceChecks": economic,
        "economicAcceptancePassed": all(economic.values()), "comparison": comparison, "excessIntervals": intervals,
        "sixMonthRestart": {"start": six_start, "books": {**{b: v for _, _, result in refs.values() for b, v in result["sixMonthRestart"]["books"].items()},
            **{b: {k: v[k] for k in ["ending", "netReturn", "fees", "dailyCloseDrawdown"]} for b, v in six.items()}}},
        "limitations": previous["limitations"] + ["One fixed selected12stock cohort-state input; no broad-market, independent-observation or profitable-strategy claim.",
            "Current-survivor stock identity can encode historical luck; overlapping labels are dependent.",
            "Calibration seals use retrospective unknown availability and UTC date markers, never verified exchange event times.",
            "Old uniform-control calibration paths remain unavailable; no baseline was refitted to obtain them."], "livePromotionAllowed": False}


def capture_replays(out, protocol_path, original, decisions, replays, restart, restart_start):
    """Capture supplied NEW replay objects once; never invoke an old replay."""
    protocol = json.loads(Path(protocol_path).read_text())
    sources = [{"id": e["symbol"], "sha256": e["sha256"], "receiptSha256": None,
                "firstSeenAt": None, "availability": "unknown"} for e in original["executionInputs"]]
    records = []
    prices_sha = replay_evidence.digest(original["executionInputs"])
    for kind, scenarios in [("main", replays), ("restart", {str(original["primarySlippage"]): restart})]:
        for scenario, books in scenarios.items():
            for book, replay in books.items():
                ds = decisions[book] if kind == "main" else [d for d in decisions[book] if d["entry"] >= restart_start]
                records.append({"book": book, "scenario": scenario, "kind": kind,
                    "start": ds[0]["entry"], "end": original["end"], "initial": original["capital"],
                    "slippageBps": float(scenario), "lots": original["lots"],
                    # The allocator may return NumPy float weights. Hash the
                    # same finite native values written to decisions.json;
                    # the strict evidence writer deliberately rejects NumPy.
                    "decisionsSha256": replay_evidence.digest(json.loads(json.dumps(ds, allow_nan=False))),
                    "calendarSha256": replay_evidence.digest({"source": original["calendarSource"],
                        "excluded": original["calendarExcluded"],
                        "sessions": [p["date"] for p in replay["curve"] if p["phase"] == "close"]}),
                    "pricesSha256": prices_sha, "replay": replay, "missing": {}})
    expected = [{"book": b, "scenario": str(s), "kind": k}
        for k, scenarios in [("main", original["slippageScenarios"]), ("restart", [original["primarySlippage"]])]
        for s in scenarios for b in BOOKS]
    contract = {"version": replay_evidence.VERSION, "experimentId": SPEC["version"], "mode": "retrospective",
        "protocol": {"sha256": base.sha(protocol_path), "frozenAt": protocol["frozenAt"]},
        "producedAt": datetime.now(timezone.utc).isoformat(), "sources": sources,
        "code": {"replaySha256": base.sha(Path(__file__).with_name("rebalance.py")),
                 "feeSha256": base.sha(Path(__file__).with_name("execution.py"))},
        "cost": {"feeSource": FEE_SOURCE, "feeChecked": FEE_CHECKED,
                 "settlement": "sale-next-session-no-cashflows", "turnover": replay_evidence.TURNOVER},
        "expected": expected}
    archive = out / "replay-evidence"
    archive.mkdir(mode=0o700)
    receipt = replay_evidence.seal(archive, contract, records)
    sealed = replay_evidence.read(archive, receipt["identitySha256"], receipt["payloadSha256"],
                                 as_of=datetime.now(timezone.utc).isoformat())
    if any(row["status"] != "complete" for row in sealed["summary"]):
        raise ValueError("Incomplete new-book replay evidence")
    return receipt


def run(path):
    path = Path(path).absolute()
    if any(p.is_symlink() for p in [path, *path.parents]):
        raise ValueError("Unsafe protocol path")
    protocol = json.loads(path.read_text())
    refs = verify(protocol)
    lineage = {"protocolSha256": base.sha(path), "code": protocol["code"], **{k: refs["original"][2]["lineage"][k] for k in ["numpy", "pandas", "sklearn"]}}
    fingerprint = hashlib.sha256(json.dumps(lineage, sort_keys=True).encode()).hexdigest()
    out = path.parent / "run"
    out.mkdir(mode=0o700, exist_ok=True)
    if out.is_symlink() or stat.S_IMODE(out.stat().st_mode) != 0o700:
        raise ValueError("Private run directory required")
    for p in out.rglob("*"):
        if p.is_symlink() or p.is_file() and stat.S_IMODE(p.stat().st_mode) != 0o600:
            raise ValueError("Unsafe run artifact")
    # One common parent lock serializes this candidate's runs across protocols.
    lock_path = path.parent.parent / ".folio-market-training.lock"
    fd = os.open(lock_path, os.O_WRONLY | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600)
    lock_info = os.fstat(fd)
    if not stat.S_ISREG(lock_info.st_mode) or stat.S_IMODE(lock_info.st_mode) != 0o600 or lock_info.st_uid != os.getuid():
        os.close(fd)
        raise ValueError("Unsafe training lock")
    with os.fdopen(fd, "a") as lock, job_lock(out):
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (out / "completed.json").exists():
            saved = json.loads((out / "completed.json").read_text())
            if saved["fingerprint"] != fingerprint or any(base.sha(out / n) != h for n, h in saved["artifacts"].items()):
                raise ValueError("Immutable cohort-dispersion result changed")
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
