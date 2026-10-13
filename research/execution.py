"""Whole-share NSE delivery research replay, with cash and daily valuation.

Current published charges are a frozen scenario, not historical contract-note
reconciliation. Raw unadjusted OHLC is required. No broker execution happens.
"""
from __future__ import annotations

import math
import pandas as pd

EXECUTION_VERSION = "delivery-replay-1"
FEE_SOURCE = "https://groww.in/pricing"
FEE_CHECKED = "2026-10-11"


def charges(value, side):
    """Unrounded published standard NSE delivery scenario; conservative DP fee."""
    if side not in {"buy", "sell"} or not math.isfinite(value) or value < 0:
        raise ValueError("Invalid order value or side")
    if value == 0:
        return 0.
    brokerage = max(min(20., value * .001), min(5., value * .025))
    exchange = value * .0000297
    sebi_ipft = value * .000002
    dp = (20. if value >= 100 else 3.5) if side == "sell" else 0.
    gst = .18 * (brokerage + exchange + sebi_ipft + dp)
    return brokerage + exchange + sebi_ipft + dp + gst + value * .001 + (value * .00015 if side == "buy" else 0.)


def validated_calendar(frames, start, end):
    """Strict shared raw-price/action validation for both execution policies."""
    calendar = sorted(set.intersection(*(set(f.index) for f in frames.values())))
    dates = [d for d in calendar if start <= d <= end]
    if not dates or dates[0] != start or dates[-1] != end:
        raise ValueError("Missing execution boundary")
    if any(list(f.loc[start:end].index) != dates for f in frames.values()):
        raise ValueError("Mismatched execution calendars; no silent missing-session fills")
    for symbol, f in frames.items():
        if f.index.has_duplicates or not f.index.is_monotonic_increasing:
            raise ValueError(f"Invalid dates: {symbol}")
        if "splits" not in f or "dividends" not in f:
            raise ValueError("Explicit corporate-action fields required")
        for d in dates:
            row = f.loc[d]
            values = [float(row[k]) for k in ["open", "high", "low", "close"]]
            if not all(math.isfinite(v) and v > 0 for v in values) or row.low > min(row.open, row.close) + 1e-6 or row.high < max(row.open, row.close) - 1e-6:
                raise ValueError(f"Invalid OHLC: {symbol} {d.date()}")
            if not math.isfinite(row.splits) or row.splits < 0 or not math.isfinite(row.dividends) or row.dividends < 0:
                raise ValueError("Invalid corporate action")
            if not math.isfinite(row.volume) or row.volume < 0:
                raise ValueError("Invalid volume")
    return dates


def replay(frames, windows, initial=100000., slippage_bps=5.):
    """Replay disjoint frozen target windows; sale proceeds available next session.

    Windows contain signal/entry/exit and long-only target weights. Liquidate at
    each exit open; on a coincident next entry only previously settled cash may
    be spent. No same-open sale recycling or borrowing. Splits adjust units;
    dividends are omitted conservatively, so this is not a total-return result.
    """
    if not math.isfinite(initial) or initial <= 0 or not math.isfinite(slippage_bps) or not 0 <= slippage_bps <= 100:
        raise ValueError("Invalid capital or slippage")
    if not frames or not windows:
        raise ValueError("Prices and frozen windows required")
    start, end = pd.Timestamp(windows[0]["entry"]), pd.Timestamp(windows[-1]["exit"])
    dates = validated_calendar(frames, start, end)
    entries, exits = {}, set()
    previous_exit = None
    for w in windows:
        signal, entry, exit_date = map(pd.Timestamp, (w["signal"], w["entry"], w["exit"]))
        weights = w["weights"]
        if not signal < entry < exit_date or entry not in dates or exit_date not in dates or (previous_exit is not None and entry < previous_exit):
            raise ValueError("Overlapping windows or invalid execution timing")
        if any(s not in frames or not math.isfinite(v) or v < 0 for s, v in weights.items()) or sum(weights.values()) > 1 + 1e-9:
            raise ValueError("Invalid long-only allocation")
        if entry in entries:
            raise ValueError("Duplicate entry")
        entries[entry] = weights
        exits.add(exit_date)
        previous_exit = exit_date
    cash, pending, held = float(initial), 0., {}
    curve = [{"date": str(start.date()), "phase": "before-entry", "equity": initial, "cash": initial, "unsettled": 0., "invested": 0.}]
    orders, fees, omitted_dividends = [], 0., 0.
    slip = slippage_bps / 10000
    for d in dates:
        cash += pending
        pending = 0.
        for s, q in list(held.items()):
            ratio = float(frames[s].loc[d, "splits"])
            if ratio:
                units = q * ratio
                if abs(units - round(units)) > 1e-8:
                    raise ValueError("Fractional split entitlement needs separate handling")
                held[s] = int(round(units))
            omitted_dividends += held[s] * float(frames[s].loc[d, "dividends"])
        if d in exits:
            for s, q in sorted(held.items()):
                price = float(frames[s].loc[d, "open"]) * (1 - slip)
                value = q * price
                fee = charges(value, "sell")
                pending += value - fee
                fees += fee
                orders.append({"date": str(d.date()), "symbol": s, "side": "sell", "units": q, "price": price, "fee": fee})
            held = {}
        if d in entries:
            equity = cash + pending
            weights = entries[d]
            scale = min(1., cash / (equity * sum(weights.values()))) if equity > 0 and sum(weights.values()) > 0 else 0.
            for s, weight in sorted(weights.items()):
                price = float(frames[s].loc[d, "open"]) * (1 + slip)
                budget = min(cash, equity * weight * scale)
                q = math.floor(budget / price)
                while q > 0 and q * price + charges(q * price, "buy") > budget + 1e-9:
                    q -= 1
                if q:
                    value, fee = q * price, charges(q * price, "buy")
                    cash -= value + fee
                    fees += fee
                    held[s] = q
                    orders.append({"date": str(d.date()), "symbol": s, "side": "buy", "units": q, "price": price, "fee": fee})
        invested = sum(q * float(frames[s].loc[d, "close"]) for s, q in held.items())
        if cash < -1e-6 or pending < -1e-6:
            raise ValueError("Insolvent paper account")
        curve.append({"date": str(d.date()), "phase": "close", "equity": cash + pending + invested, "cash": cash, "unsettled": pending, "invested": invested})
    peak, drawdown = initial, 0.
    for point in curve:
        peak = max(peak, point["equity"])
        drawdown = min(drawdown, point["equity"] / peak - 1)
    return {"version": EXECUTION_VERSION, "initial": initial, "ending": curve[-1]["equity"], "netReturn": curve[-1]["equity"] / initial - 1,
            "dailyCloseDrawdown": drawdown, "fees": fees, "orders": orders, "curve": curve,
            "omittedDividendEntitlements": omitted_dividends, "slippageBpsPerSide": slippage_bps,
            "feeSource": FEE_SOURCE, "feeChecked": FEE_CHECKED,
            "limitations": ["Current published fee scenario, not verified historical charges; statutory rounding and account-specific differences omitted.",
                            "Conservative next-common-session sale cash availability; settlement failures and broker-specific fund reuse excluded.",
                            "Dividends omitted, no interest or capital-gains tax; not total return.",
                            "Daily closes miss intraday drawdown; ideal fills with fixed slippage, no order-book or price-limit availability proof.",
                            "Current-company cohort and revised provider data; exploratory historical replay, no trading edge established."]}
