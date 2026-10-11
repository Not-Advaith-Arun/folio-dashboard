"""Fixed long-only allocation challenger; not an outer-test-selected strategy.

Separate relative ranking from absolute eligibility and total risk exposure.
Parameters are frozen in POLICY before evaluation. No threshold search, leverage,
interest-bearing cash assumption, or claim to replicate the referenced papers.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from research.engine import performance

POLICY_VERSION = "absolute-risk-1"
POLICY = {"version": POLICY_VERSION, "annualVolTarget": .10, "nameCap": .40,
          "minimumEligible": 2, "ridgeShare": .50, "cashRate": 0,
          "rule": "Positive 60-session return and 200-session trend; positive cohort trend; exposure = eligible breadth × min(1, 10% / cohort volatility). Rank 50% cross-sectional Ridge and 50% 120-session momentum excluding the latest 20 sessions. Inverse-volatility weights capped at 40%; remainder cash."}
REFERENCES = [
    {"title": "Volatility-Managed Portfolios", "url": "https://www.nber.org/papers/w22208", "adaptation": "Reduce exposure during high measured volatility; fixed 10% target, not a replication."},
    {"title": "Time Series Momentum", "url": "https://www.aqr.com/Insights/Research/Journal-Article/Time-Series-Momentum", "adaptation": "Own-price positive trend eligibility separately from relative ranking; different universe and horizons."},
    {"title": "Continuous Cash-Overlay Filters (June 2026)", "url": "https://arxiv.org/abs/2606.09025", "adaptation": "Independent cash/risk layer; no macro/VIX filters or claimed paper replication."},
]


def allocation(day, top_k, use_ridge=True):
    required = ["return60", "return120", "return20", "trend200", "marketTrend", "marketVol", "volatility20"]
    if use_ridge:
        required.append("Cross-sectional Ridge")
    if day.empty or not np.isfinite(day[required].to_numpy()).all():
        return {}, {"exposure": 0., "breadth": 0., "reason": "Incomplete signal features"}
    eligible = (day.return60 > 0) & (day.trend200 > 0)
    breadth = float(eligible.mean())
    state = {"exposure": 0., "breadth": breadth, "reason": "Cohort trend is nonpositive"}
    if float(day.marketTrend.iloc[0]) <= 0:
        return {}, state
    if int(eligible.sum()) < POLICY["minimumEligible"]:
        return {}, {**state, "reason": "Fewer than two eligible stocks"}
    momentum = (1 + day.return120) / (1 + day.return20) - 1
    score = momentum.rank(pct=True)
    if use_ridge:
        score = .5 * score + .5 * day["Cross-sectional Ridge"].rank(pct=True)
    chosen = day.assign(policyScore=score)[eligible].sort_values(["policyScore", "symbol"], ascending=[False, True]).head(top_k)
    exposure = breadth * min(1., POLICY["annualVolTarget"] / max(float(day.marketVol.iloc[0]), .01))
    inverse = 1 / chosen.volatility20.clip(lower=.005)
    weights = (inverse / inverse.sum() * exposure).clip(upper=POLICY["nameCap"])
    # Clipped weight is cash, never redistributed above the cap.
    allocation = {s: float(w) for s, w in zip(chosen.symbol, weights)}
    return allocation, {"exposure": float(weights.sum()), "breadth": breadth, "reason": "Trend eligible; volatility and breadth scaled"}


def evaluate(evaluation, signals, latest, top_k, cost_bps, original_curve):
    returns = {"Absolute-risk challenger": [], "Trend-only ablation": []}
    exposures = []
    for date in signals:
        day = evaluation[evaluation.date == date]
        actual = day.set_index("symbol").actual
        for name, use_ridge in [("Absolute-risk challenger", True), ("Trend-only ablation", False)]:
            weights, state = allocation(day, top_k, use_ridge)
            returns[name].append(sum(w * float(actual[s]) for s, w in weights.items()) - state["exposure"] * cost_bps / 10000)
            if use_ridge:
                exposures.append(state["exposure"])
    curve = [{"date": original_curve[0]["date"], **{n: 1. for n in returns},
              "Original pipeline": 1., "Equal weight": 1.}]
    levels = {n: 1. for n in returns}
    for i, date in enumerate(signals):
        for n in returns:
            levels[n] *= 1 + returns[n][i]
        curve.append({"date": str(pd.Timestamp(evaluation[evaluation.date == date].labelEnd.iloc[0]).date()), **levels,
                      "Original pipeline": original_curve[i + 1]["Selected pipeline"], "Equal weight": original_curve[i + 1]["Equal weight"]})
    for n in ["Original pipeline", "Equal weight"]:
        returns[n] = [curve[i + 1][n] / curve[i][n] - 1 for i in range(len(signals))]
    weights, state = allocation(latest, top_k)
    from research.audit import moving_block_interval
    interval = moving_block_interval(np.array(returns["Absolute-risk challenger"]) - np.array(returns["Equal weight"]))
    return {"version": POLICY_VERSION, "protocol": POLICY, "metrics": [{"name": n, **performance(r)} for n, r in returns.items()],
            "curve": curve, "excessMeanInterval": interval,
            "costStress": [{"costBps": b, **performance([r + e * (cost_bps - b) / 10000 for r, e in zip(returns["Absolute-risk challenger"], exposures)])} for b in [0, 10, 20, 50, 100]],
            "activeWindows": sum(e > 0 for e in exposures), "periods": len(signals),
            "latest": {**state, "cashWeight": 1 - state["exposure"], "allocations": [{"symbol": s, "weight": w} for s, w in weights.items()]},
            "status": "Separate fixed challenger on previously viewed history; collect prospective evidence. Not promoted by historical return.",
            "limitations": ["Risk controls can lose money or miss rebounds; a volatility target is not a loss bound.", "No independent absolute-return forecasting accuracy is claimed; relative Ridge scores only rank eligible stocks.", "Full invested-exposure round-trip cost each window matches the original protocol, not measured turnover.", "Current cohort and revised prices; fractional paper fills; no taxes or cash yield; window-exit drawdown only."],
            "references": REFERENCES}
