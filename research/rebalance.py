"""Retained-position research replay with explicit lots and settled cash.

Targets are supplied, frozen past-signal decisions. This module optimizes no
forecast or parameter and makes no broker orders or fill-availability claims.
"""
from __future__ import annotations

import math
import pandas as pd

from research.execution import charges, validated_calendar, FEE_SOURCE, FEE_CHECKED

REBALANCE_VERSION = "rebalance-replay-1"


def replay_rebalanced(frames, decisions, end, lots, initial=100000., slippage_bps=5., drift_band=0.):
    if not frames or not decisions:
        raise ValueError("Prices and frozen decisions required")
    if not math.isfinite(initial) or initial <= 0 or not math.isfinite(slippage_bps) or not 0 <= slippage_bps <= 100:
        raise ValueError("Invalid capital or slippage")
    if not math.isfinite(drift_band) or not 0 <= drift_band <= .1:
        raise ValueError("Invalid drift band")
    if set(lots) != set(frames) or any(type(v) is not int or v < 1 for v in lots.values()):
        raise ValueError("Explicit positive integer lot for every instrument required")
    start, end = pd.Timestamp(decisions[0]["entry"]), pd.Timestamp(end)
    if start >= end:
        raise ValueError("Final liquidation must follow first entry")
    dates = validated_calendar(frames, start, end)
    events, previous = {}, None
    for decision in decisions:
        signal, entry = map(pd.Timestamp, (decision["signal"], decision["entry"]))
        weights = decision["weights"]
        if not signal < entry < end or entry not in dates or (previous is not None and entry <= previous):
            raise ValueError("Invalid or duplicate decision timing")
        if any(s not in frames or not math.isfinite(w) or w < 0 for s, w in weights.items()) or sum(weights.values()) > 1 + 1e-9:
            raise ValueError("Invalid long-only allocation")
        events[entry], previous = weights, entry
    cash, pending, held = float(initial), 0., {}
    fees, omitted_dividends, suppressed = 0., 0., 0
    orders, shortfalls = [], []
    curve = [{"date": str(start.date()), "phase": "before-entry", "equity": initial,
              "cash": initial, "unsettled": 0., "invested": 0.}]
    slip = slippage_bps / 10000

    def order(d, s, q, side):
        if float(frames[s].loc[d, "volume"]) <= 0:
            raise ValueError("Cannot assume a fill on a zero-volume session")
        price = float(frames[s].loc[d, "open"]) * (1 + slip if side == "buy" else 1 - slip)
        value = q * price
        fee = charges(value, side)
        orders.append({"date": str(d.date()), "symbol": s, "side": side,
                       "units": q, "price": price, "fee": fee})
        return value, fee

    for d in dates:
        cash += pending
        pending = 0.
        for s, q in list(held.items()):
            ratio = float(frames[s].loc[d, "splits"])
            if ratio:
                units = q * ratio
                if abs(units - round(units)) > 1e-8 or round(units) % lots[s]:
                    raise ValueError("Fractional or odd-lot split entitlement requires separate handling")
                held[s] = int(round(units))
            omitted_dividends += held[s] * float(frames[s].loc[d, "dividends"])
        if d in events or d == end:
            weights = {} if d == end else events[d]
            equity = cash + sum(q * float(frames[s].loc[d, "open"]) for s, q in held.items())
            desired = {}
            for s in sorted(set(held) | set(weights)):
                price, lot = float(frames[s].loc[d, "open"]), lots[s]
                target = math.floor(equity * weights.get(s, 0.) / price / lot) * lot
                current = held.get(s, 0)
                if target and current and abs(target - current) * price < drift_band * equity:
                    suppressed += int(target != current)
                    target = current
                desired[s] = target
            # Sales fund next session only. Targets are not retried between decisions.
            for s, target in desired.items():
                q = max(0, held.get(s, 0) - target)
                if q:
                    value, fee = order(d, s, q, "sell")
                    pending += value - fee
                    fees += fee
                    held[s] -= q
            needs = {}
            for s, target in desired.items():
                q = max(0, target - held.get(s, 0))
                if q:
                    value = q * float(frames[s].loc[d, "open"]) * (1 + slip)
                    needs[s] = (q, value + charges(value, "buy"))
            total = sum(v[1] for v in needs.values())
            available = cash
            for s, (wanted, cost) in needs.items():
                budget = min(cost, available * cost / total)
                price, lot = float(frames[s].loc[d, "open"]) * (1 + slip), lots[s]
                q = min(wanted, math.floor(budget / price / lot) * lot)
                while q and q * price + charges(q * price, "buy") > budget + 1e-9:
                    q -= lot
                if q:
                    value, fee = order(d, s, q, "buy")
                    cash -= value + fee
                    fees += fee
                    held[s] = held.get(s, 0) + q
                if q < wanted:
                    shortfalls.append({"date": str(d.date()), "symbol": s, "wantedUnits": wanted,
                                       "filledUnits": q, "reason": "Settled cash, fees or lot affordability"})
            held = {s: q for s, q in held.items() if q}
        invested = sum(q * float(frames[s].loc[d, "close"]) for s, q in held.items())
        if min(cash, pending) < -1e-6:
            raise ValueError("Insolvent paper account")
        curve.append({"date": str(d.date()), "phase": "close", "equity": cash + pending + invested,
                      "cash": cash, "unsettled": pending, "invested": invested})
    peak, drawdown = initial, 0.
    for point in curve:
        peak = max(peak, point["equity"])
        drawdown = min(drawdown, point["equity"] / peak - 1)
    return {"version": REBALANCE_VERSION, "initial": initial, "ending": curve[-1]["equity"],
            "netReturn": curve[-1]["equity"] / initial - 1, "dailyCloseDrawdown": drawdown,
            "fees": fees, "orders": orders, "curve": curve, "unfilledTargets": shortfalls,
            "suppressedAdjustments": suppressed, "driftBand": drift_band, "lots": lots,
            "omittedDividendEntitlements": omitted_dividends, "slippageBpsPerSide": slippage_bps,
            "feeSource": FEE_SOURCE, "feeChecked": FEE_CHECKED,
            "limitations": ["Current published NSE fee scenario; historical/account charges and statutory rounding unverified.",
                            "Supplied fixed lots must be verified for the period; no automatic SME eligibility or odd-lot trades.",
                            "Conservative next-session sale cash availability; unfilled targets wait for next decision.",
                            "Dividends omitted; no interest/tax. Daily closes miss intraday drawdown. Final liquidation at open.",
                            "Fixed-slippage fills remain idealized; no market-depth, circuit, suspension or impact guarantee.",
                            "Previously viewed selected-cohort research; improved execution does not establish predictive alpha."]}
