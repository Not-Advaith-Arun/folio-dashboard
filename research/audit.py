"""Bounded, nested research protocol. Paper-inspired adaptations, not replications.

No outer evaluation result may select candidates or tune parameters. The
historical period has been viewed in baseline research, so prospective evidence
is still needed. Experiments remain paper-only regardless of reported returns.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from research.engine import FEATURES, HORIZON, estimators, performance, rank_ic

AUDIT_VERSION = "nested-audit-3"
RICH_FEATURES = [*FEATURES, "return1", "return10", "return120", "relative20", "trend200", "marketTrend", "marketVol"]
CANDIDATES = ["Uniform ensemble", "Validation weighted", "Rich Ridge", "Cross-sectional Ridge", "Risk managed", "Momentum", "Cash"]
REFERENCES = [
    {"title": "Empirical Asset Pricing via Machine Learning", "url": "https://www.nber.org/papers/w25398", "adaptation": "A separate cross-sectional rank-normalized linear challenger; not a replication of the original study."},
    {"title": "FinTSB (2025)", "url": "https://arxiv.org/abs/2502.18834", "adaptation": "Common execution/cost protocol and explicit market conditions."},
    {"title": "Financial time-series benchmark (March 2026)", "url": "https://arxiv.org/abs/2603.01820", "adaptation": "Cost stress, risk-adjusted evaluation and computational restraint."},
    {"title": "Equity portfolio benchmark (June 2026)", "url": "https://arxiv.org/abs/2606.09420", "adaptation": "Regularized linear challenger and portfolio constraints; no assumption complex models win."},
    {"title": "Conformal position sizing, revised October 2026", "url": "https://arxiv.org/abs/2608.01494", "adaptation": "Empirical uncertainty bands and realized coverage are separate from profitability; no Kelly leverage."},
]


def cross_features(frame):
    # Normalize today's cross-section only; never use later trading dates.
    # Market-wide features are identical within a date and carry no rank signal.
    columns = [c for c in RICH_FEATURES if c not in ["market20", "marketTrend", "marketVol"]]
    return frame.groupby("date")[columns].rank(pct=True) * 2 - 1


def fit_models(train):
    models = estimators()
    for model in models.values():
        model.fit(train[FEATURES], train.target)
    ridge = make_pipeline(StandardScaler(), Ridge(alpha=25))
    ridge.fit(train[RICH_FEATURES], train.target)
    cross = make_pipeline(StandardScaler(), Ridge(alpha=25))
    cross.fit(cross_features(train), train.target)
    return models, {"raw": ridge, "cross": cross}


def predict(frame, models, ridge, weights):
    frame = frame.copy()
    for name, model in models.items():
        frame[name] = model.predict(frame[FEATURES])
    frame["Uniform ensemble"] = sum(frame.groupby("date")[n].rank(pct=True) for n in models) / len(models)
    frame["Validation weighted"] = sum(weights[n] * frame.groupby("date")[n].rank(pct=True) for n in models)
    frame["weightedEstimate"] = sum(weights[n] * frame[n] for n in models)
    frame["Rich Ridge"] = ridge["raw"].predict(frame[RICH_FEATURES])
    frame["Cross-sectional Ridge"] = ridge["cross"].predict(cross_features(frame))
    frame["Risk managed"] = frame["Validation weighted"]
    frame["Momentum"] = frame.return20
    return frame


def allocate(day, strategy, top_k):
    if strategy == "Cash":
        return {}, 0.0
    chosen = day.nlargest(top_k, strategy)
    exposure = 1.0
    if strategy == "Risk managed":
        # Fixed before evaluation: long-only, 15% market-vol target, no leverage,
        # 40% per-name cap and a 200-session market trend gate; remainder cash.
        exposure = 0.0 if float(day.marketTrend.iloc[0]) <= 0 else min(1.0, .15 / max(float(day.marketVol.iloc[0]), .01))
        inv_vol = 1 / chosen.volatility20.clip(lower=.005)
        weights = (inv_vol / inv_vol.sum() * exposure).clip(upper=.4)
    else:
        weights = pd.Series(1 / len(chosen), index=chosen.index)
    return {s: float(w) for s, w in zip(chosen.symbol, weights) if w > 0}, float(weights.sum())


def strategy_returns(frame, strategy, top_k, cost_bps, dates=None):
    dates = dates if dates is not None else sorted(frame.date.unique())[::HORIZON]
    returns, exposures = [], []
    for date in dates:
        day = frame[frame.date == date]
        weights, exposure = allocate(day, strategy, top_k)
        actual = day.set_index("symbol").actual
        returns.append(sum(w * float(actual[s]) for s, w in weights.items()) - exposure * cost_bps / 10000)
        exposures.append(exposure)
    return returns, exposures


def validation_score(returns):
    p = performance(returns)
    # Fixed simple utility, not fitted to outer results; cash can legitimately win.
    return p["totalReturn"] - .5 * abs(p["maxDrawdown"])


def development(train, top_k, cost_bps):
    dates = sorted(train.date.unique())
    if len(dates) < 210:
        raise ValueError("Not enough history for nested tuning and calibration.")
    tune_start, selection_start, second_start, calibration_start = [pd.Timestamp(dates[i]) for i in [-147, -105, -63, -21]]
    inner_train = train[train.labelEnd < tune_start]
    tune = train[(train.date >= tune_start) & (train.labelEnd < selection_start)]
    if len(inner_train.date.unique()) < 60:
        raise ValueError("Not enough purged inner training dates.")
    models, ridge = fit_models(inner_train)
    uniform = {n: 1 / len(models) for n in models}
    raw = predict(tune, models, ridge, uniform)
    ics = np.array([rank_ic(raw, n) for n in models])
    probabilities = np.exp(np.clip(ics / .1, -3, 3))
    probabilities /= probabilities.sum()
    # Shrink towards uniform to avoid a noisy short validation window dominating.
    weights = {n: float(.5 / len(models) + .5 * p) for n, p in zip(models, probabilities)}
    # Weight learning is disjoint from strategy selection. Each selection block
    # uses models refitted on outcomes strictly available before that block.
    blocks, candidate_returns, candidate_scores = [], {n: [] for n in CANDIDATES}, {n: [] for n in CANDIDATES}
    for start, end in [(selection_start, second_start), (second_start, calibration_start)]:
        block_train = train[train.labelEnd < start]
        block_test = train[(train.date >= start) & (train.labelEnd < end)]
        block_models, block_ridge = fit_models(block_train)
        predicted = predict(block_test, block_models, block_ridge, weights)
        metrics = []
        for name in CANDIDATES:
            returns, _ = strategy_returns(predicted, name, top_k, cost_bps)
            candidate_returns[name].extend(returns)
            score = validation_score(returns)
            candidate_scores[name].append(score)
            metrics.append({"name": name, "selectionScore": score, **performance(returns)})
        blocks.append({"start": str(start.date()), "end": str(end.date()), "lastTrainingOutcome": str(block_train.labelEnd.max().date()), "lastOutcome": str(block_test.labelEnd.max().date()), "periods": len(returns), "metrics": metrics})
    scores = []
    for name in CANDIDATES:
        scores.append({"name": name, "selectionScore": min(candidate_scores[name]), **performance(candidate_returns[name])})
    selected = max(scores, key=lambda x: (x["selectionScore"], x["name"] == "Cash"))["name"]
    # Refit only before calibration; the calibration block is not training data.
    final_train = train[train.labelEnd < calibration_start]
    final_models, final_ridge = fit_models(final_train)
    calibration = train[train.date >= calibration_start]
    calibration_predictions = predict(calibration, final_models, final_ridge, weights)
    errors = np.abs(calibration_predictions.target - calibration_predictions.weightedEstimate)
    radius = float(np.quantile(errors, .9, method="higher"))
    return final_models, final_ridge, weights, selected, radius, {
        "innerTrainLastOutcome": str(inner_train.labelEnd.max().date()), "tuningStart": str(tune_start.date()), "tuningLastOutcome": str(tune.labelEnd.max().date()), "selectionStart": str(selection_start.date()), "selectionBlocks": blocks, "calibrationStart": str(calibration_start.date()), "finalTrainLastOutcome": str(final_train.labelEnd.max().date()), "calibrationLastOutcome": str(calibration.labelEnd.max().date()), "weights": weights, "selected": selected, "candidates": scores, "intervalRadius": radius,
    }


def moving_block_interval(excess):
    values = np.asarray(excess)
    rng = np.random.default_rng(42)
    block = min(3, len(values))
    samples = []
    for _ in range(400):
        starts = rng.integers(0, max(1, len(values) - block + 1), size=int(np.ceil(len(values) / block)))
        resampled = np.concatenate([values[i:i + block] for i in starts])[:len(values)]
        samples.append(float(resampled.mean()))
    return {"lower": float(np.quantile(samples, .025)), "upper": float(np.quantile(samples, .975)), "blockPeriods": block, "resamples": 400}


def run_audit(panel, top_k, cost_bps, progress):
    all_rows = panel.dropna(subset=RICH_FEATURES)
    mature = all_rows.dropna(subset=["target", "labelEnd"])
    dates = sorted(mature.date.unique())
    test_start = max(240, len(dates) - 252)
    evaluation_dates = dates[test_start:]
    if len(evaluation_dates) < 90:
        raise ValueError("At least 240 rich-feature training dates and 90 evaluation dates are needed.")
    parts, folds = [], []
    for i, fold in enumerate(np.array_split(evaluation_dates, 3)):
        start = pd.Timestamp(fold[0])
        train = mature[(mature.date < start) & (mature.labelEnd < start)]
        test = mature[mature.date.isin(fold)].copy()
        progress(f"Auditing candidates: nested fold {i + 1}/3", 74 + i * 3)
        models, ridge, weights, selected, radius, audit = development(train, top_k, cost_bps)
        predicted = predict(test, models, ridge, weights)
        predicted["selectedStrategy"] = selected
        predicted["intervalRadius"] = radius
        parts.append(predicted)
        folds.append({**audit, "testStart": str(start.date()), "testEnd": str(pd.Timestamp(fold[-1]).date()), "lastAvailableOutcome": str(train.labelEnd.max().date())})
    evaluation = pd.concat(parts).sort_values(["date", "symbol"])
    signals = evaluation_dates[::HORIZON]
    returns = {n: strategy_returns(evaluation, n, top_k, cost_bps, signals)[0] for n in CANDIDATES}
    returns["Equal weight"] = [float(evaluation[evaluation.date == d].actual.mean()) - cost_bps / 10000 for d in signals]
    selected_returns, selected_exposures = [], []
    for date in signals:
        day = evaluation[evaluation.date == date]
        r, exposure = strategy_returns(day, day.selectedStrategy.iloc[0], top_k, cost_bps, [date])
        selected_returns.extend(r)
        selected_exposures.extend(exposure)
    returns["Selected pipeline"] = selected_returns
    levels = {n: 1.0 for n in returns}
    curve = [{"date": str(pd.Timestamp(signals[0]).date()), **levels}]
    for i, date in enumerate(signals):
        for n in returns:
            levels[n] *= 1 + returns[n][i]
        curve.append({"date": str(pd.Timestamp(evaluation[evaluation.date == date].labelEnd.iloc[0]).date()), **levels})
    stress = []
    for bps in [0, 10, 20, 50, 100]:
        stressed = [r + e * (cost_bps - bps) / 10000 for r, e in zip(selected_returns, selected_exposures)]
        stress.append({"costBps": bps, **performance(stressed)})
    excess = np.asarray(selected_returns) - np.asarray(returns["Equal weight"])
    confidence = moving_block_interval(excess)
    interval_coverage = float((np.abs(evaluation.target - evaluation.weightedEstimate) <= evaluation.intervalRadius).mean())
    # Latest selection uses only matured historical outcomes, never the outer score table.
    progress("Selecting the next research strategy from past validation", 85)
    models, ridge, weights, selected, radius, live_audit = development(mature, top_k, cost_bps)
    latest = all_rows[all_rows.date == all_rows.date.max()]
    predictions = predict(latest, models, ridge, weights)
    allocations, exposure = allocate(predictions, selected, top_k)
    proposal = [{"symbol": r.symbol, "estimate": float(r.weightedEstimate), "lower": float(r.weightedEstimate - radius), "upper": float(r.weightedEstimate + radius), "weight": allocations.get(r.symbol, 0.0)} for _, r in predictions.iterrows()]
    metrics = [{"name": n, **performance(values)} for n, values in returns.items()]
    selected_performance = performance(selected_returns)
    benchmark_performance = performance(returns["Equal weight"])
    # Diagnostic gate only; selection has already happened using development data.
    supported = selected_performance["totalReturn"] > benchmark_performance["totalReturn"] and confidence["lower"] > 0
    report = {"version": AUDIT_VERSION, "selectionRule": "Weights learned separately; select by worst of two purged development blocks (net return minus half absolute drawdown), with cash winning ties. Outer results never select models.", "historicalStatus": "Previously viewed historical evaluation; prospective confirmation required.", "folds": folds, "metrics": metrics, "curve": curve, "costStress": stress, "excessMeanInterval": confidence, "intervalCoverage": interval_coverage, "nominalIntervalCoverage": .9, "gate": "Historical evidence supports further paper testing" if supported else "No reliable historical advantage; keep paper-only", "latest": {"strategy": selected, "weights": weights, "exposure": exposure, "cashWeight": max(0.0, 1 - exposure), "allocations": proposal, "development": live_audit}, "references": REFERENCES, "limitations": ["Candidate rules are adaptations of research ideas, not replications of the cited studies.", "The historical evaluation was viewed during baseline work; it is not a fresh sealed holdout.", "Uncertainty bands are empirical residual bands with measured coverage, not distribution-free guarantees for dependent market data.", "Four hundred moving-block resamples describe sampling uncertainty; they do not correct survivorship bias or researcher selection.", "Cash assumes zero interest. Costs charge the full invested exposure each window rather than estimating exact turnover.", "Selected pipeline is frozen within each fold; candidate comparisons are diagnostics and cannot choose a winner using evaluation returns."]}

    return report, {"models": models, "ridge": ridge, "features": RICH_FEATURES, "weights": weights, "selected": selected, "intervalRadius": radius, "version": AUDIT_VERSION}
