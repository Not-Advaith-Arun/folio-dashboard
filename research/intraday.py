"""Separate offline five-minute cash-equity research architecture.

No acquisition or broker adapter. Callers must supply reviewed source-use and
dated session evidence. Historical bar availability assumptions do not establish
prospective issuance. Missing exits invalidate a replay, never disappear as losses.
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

VERSION = "intraday-offline-1"
FEATURES = ["return1", "return3", "return6", "volatility12", "range", "volumeRatio", "sessionFraction"]
BAR = pd.Timedelta(minutes=5)
TRAIN_DAYS, TEST_DAYS = 60, 20
BOOKS = ["model", "always", "persistence", "cash"]


def fees(value, side):
    """Frozen Groww/NSE 2026-10-21 intraday scenario, not historical tax proof."""
    if side not in {"buy", "sell"} or not np.isfinite(value) or value < 0:
        raise ValueError("Invalid order value/side")
    if value == 0:
        return 0.
    brokerage = min(max(min(20., .001 * value), 5.), .025 * value)
    taxable = brokerage + value * (.0000297 + .000001 + .000001)
    return float(taxable * 1.18 + value * (.00003 if side == "buy" else .00025))


def reference_hurdle(slippage_bps=10.):
    return (fees(25000., "buy") + fees(25000., "sell")) / 25000. + 2 * slippage_bps / 10000


def build_panel(frames, calendar, provenance, observed_at):
    """Strict complete-session pilot; do not infer a calendar from observed bars.

    Missing instrument-days fail the entire pilot, rather than selecting current
    survivors after outcomes. A future broader unbalanced adapter must version
    historical membership/suspension treatment before changing this contract.
    """
    required = ["source", "sourceUseEvidence", "calendarEvidence", "availabilityBasis"]
    if any(not isinstance(provenance.get(k), str) or not provenance[k].strip() for k in required):
        raise ValueError("Dated source-use/calendar/availability evidence required")
    if provenance.get("priceBasis") != "raw":
        raise ValueError("Raw intraday prices required")
    if (not isinstance(calendar, pd.DatetimeIndex) or calendar.tz is None or calendar.hasnans
            or not calendar.is_unique or not calendar.is_monotonic_increasing or calendar.empty):
        raise ValueError("Explicit ordered timezone-aware bar-start calendar required")
    calendar = calendar.tz_convert("Asia/Kolkata")
    cutoff = pd.Timestamp(observed_at)
    if cutoff.tzinfo is None or cutoff > datetime.now(timezone.utc) or (calendar[-1] + BAR) > cutoff:
        raise ValueError("Future or incomplete bars")
    for _, day in pd.Series(calendar, index=calendar).groupby(calendar.date):
        idx = pd.DatetimeIndex(day)
        if (idx[0].strftime("%H:%M") != "09:15" or idx[-1].strftime("%H:%M") != "15:10"
                or not (idx[1:] - idx[:-1] == BAR).all()):
            raise ValueError("Pilot needs explicit complete 09:15–15:10 five-minute sessions")
    if not frames:
        raise ValueError("Empty instrument universe")
    pieces, hashes = [], {}
    for symbol, original in sorted(frames.items()):
        f = original.copy()
        if (not isinstance(symbol, str) or not symbol or not isinstance(f.index, pd.DatetimeIndex)
                or f.index.tz is None or not f.index.equals(calendar) or not f.columns.is_unique
                or not {"open", "high", "low", "close", "volume", "availableAt"}.issubset(f.columns)):
            raise ValueError("Incomplete/duplicate instrument bars or missing receipts")
        prices = f[["open", "high", "low", "close", "volume"]].apply(pd.to_numeric, errors="coerce")
        p = prices[["open", "high", "low", "close"]]
        if not (np.isfinite(prices).all(axis=1) & p.gt(0).all(axis=1) & prices.volume.ge(0)
                & p.max(axis=1).le(p.high) & p.min(axis=1).ge(p.low)).all():
            raise ValueError("Invalid OHLCV")
        receipts = pd.DatetimeIndex(pd.to_datetime(f.availableAt))
        if receipts.tz is None or receipts.hasnans or (receipts < calendar + BAR).any() or (receipts > cutoff).any():
            raise ValueError("Invalid/future availability timestamps")
        hashes[symbol] = hashlib.sha256(f.to_json(orient="split", date_format="iso").encode()).hexdigest()
        for _, day in prices.groupby(calendar.date):
            x = day.copy()
            idx = x.index
            for lag in [1, 3, 6]:
                x[f"return{lag}"] = x.close.pct_change(lag, fill_method=None)
            x["volatility12"] = x.return1.rolling(12).std()
            x["range"] = (x.high - x.low) / x.close
            x["volumeRatio"] = x.volume / x.volume.rolling(12).mean().replace(0., np.nan)
            x["sessionFraction"] = np.arange(len(x)) / len(x)
            x["symbol"], x["barStart"], x["session"] = symbol, idx, str(idx[0].date())
            x["availableAt"] = receipts[calendar.get_indexer(idx)]
            # A completed signal bar ends at i+1. Skip that full next bar and
            # enter at i+2 open. Exit six bars (30 minutes) after entry.
            dates = pd.Series(idx, index=idx)
            x["entryAt"], x["exitAt"] = dates.shift(-2), dates.shift(-8)
            x["entryOpen"], x["exitOpen"] = x.open.shift(-2), x.open.shift(-8)
            x["entryVolume"], x["exitVolume"] = x.volume.shift(-2), x.volume.shift(-8)
            x["actual"] = x.exitOpen / x.entryOpen - 1
            liquid_path = pd.concat([x.volume.shift(-i).gt(0) for i in range(2, 9)], axis=1).all(axis=1)
            x["actual"] = x.actual.where(liquid_path)
            x["featureReady"] = (np.isfinite(x[FEATURES]).all(axis=1) & x.volume.gt(0)
                                  & (x.availableAt < (x.barStart + 2 * BAR)))
            # Current features stay visible without future labels. End-of-day
            # entry restrictions derive from the calendar, not realized returns.
            x["decisionEligible"] = x.featureReady & x.exitAt.notna()
            pieces.append(x.reset_index(drop=True))
    panel = pd.concat(pieces, ignore_index=True)
    manifest = {"version": VERSION, "provenance": provenance, "inputHashes": hashes,
        "calendarSha256": hashlib.sha256("\n".join(calendar.astype(str)).encode()).hexdigest(),
        "sessions": int(panel.session.nunique()), "rows": len(panel),
        "featureReady": int(panel.featureReady.sum()), "livePromotionAllowed": False,
        "limitations": ["Complete-session pilot is not a historical all-stock universe.",
                        "OHLC cannot verify spread, queue, actual fills, circuits or auction execution.",
                        "Historical availability and current fee assumptions are not prospective evidence."]}
    return panel, manifest


def walk_forward(panel):
    """Two fixed CPU heads; whole test days never share labels with training."""
    days = sorted(panel.session.unique())
    forecasts, fits = [], []
    for start in range(TRAIN_DAYS, len(days), TEST_DAYS):
        test_days = days[start:start + TEST_DAYS]
        first = panel.loc[panel.session.eq(test_days[0]), "barStart"].min()
        train = panel[panel.session.isin(days[start - TRAIN_DAYS:start]) & panel.decisionEligible
                      & (panel.exitAt < first) & np.isfinite(panel.actual)]
        test = panel[panel.session.isin(test_days) & panel.decisionEligible].copy()
        positive = train.actual.gt(reference_hurdle())
        fit = {"testStart": test_days[0], "testEnd": test_days[-1], "trainRows": len(train),
               "trainDays": int(train.session.nunique()), "testRows": len(test)}
        test["prediction"], test["probability"] = np.nan, np.nan
        test["pastMean"] = float(train.actual.mean()) if len(train) else np.nan
        test["pastProbability"] = float(positive.mean()) if len(train) else np.nan
        if train.session.nunique() < TRAIN_DAYS or positive.nunique() < 2 or test.empty:
            fits.append({**fit, "status": "insufficient_history_or_classes"})
            forecasts.append(test)
            continue
        mean = make_pipeline(StandardScaler(), Ridge(alpha=100.))
        probability = make_pipeline(StandardScaler(), LogisticRegression(C=1., max_iter=1000, random_state=42))
        with threadpool_limits(limits=2):
            mean.fit(train[FEATURES], train.actual)
            probability.fit(train[FEATURES], positive.astype(int))
            test["prediction"] = mean.predict(test[FEATURES])
            test["probability"] = probability.predict_proba(test[FEATURES])[:, 1]
        parameters = {}
        for key, model in [("mean", mean), ("probability", probability)]:
            scaler, estimator = model.steps[0][1], model.steps[1][1]
            parameters[key] = {"scaleMean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(),
                               "coefficients": estimator.coef_.tolist(), "intercept": np.asarray(estimator.intercept_).tolist()}
        fits.append({**fit, "status": "fitted", "lastTrainOutcome": str(train.exitAt.max()),
                     "parameters": parameters})
        forecasts.append(test)
    return (pd.concat(forecasts, ignore_index=True) if forecasts else panel.iloc[:0].assign(
        prediction=np.nan, probability=np.nan)), fits


def replay(panel, forecasts, *, book="model", slippage_bps=5., capital=100000.):
    """Sequential whole-share cash book; fail unresolved exits, never force a win.

    Entry participation rejects a fill after a signal, without selecting a new
    instrument using that bar. Exit volume failures invalidate the whole replay.
    Prices include adverse bps; 0bps is a friction sensitivity, not free execution.
    """
    if (book not in BOOKS or not np.isfinite(slippage_bps) or not 0 <= slippage_bps <= 100
            or not np.isfinite(capital) or capital <= 0):
        raise ValueError("Invalid replay configuration")
    cash, available = capital, None
    trades, decisions, curve = [], [], [{"at": None, "equity": capital}]
    indexed = panel.set_index(["symbol", "barStart"])
    if not indexed.index.is_unique or forecasts.duplicated(["symbol", "barStart"]).any():
        raise ValueError("Duplicate replay observations or forecasts")
    for timestamp, group in forecasts.groupby("barStart", sort=True):
        if book == "cash":
            continue
        if available is not None and timestamp < available:
            continue
        candidates = group[group.decisionEligible].copy()
        if book == "model":
            candidates = candidates[(candidates.probability > .5)
                                    & (candidates.prediction > reference_hurdle(slippage_bps))]
            candidates = candidates.sort_values(["prediction", "symbol"], ascending=[False, True])
        else:
            if book == "persistence":
                candidates = candidates[candidates.return1 > 0]
            candidates = candidates.sort_values("symbol")
        if candidates.empty:
            continue
        row = candidates.iloc[0]
        # Forecasts cannot substitute execution bars or change their own timing.
        actual = indexed.loc[(row.symbol, timestamp)]
        if (row.entryAt != actual.entryAt or row.exitAt != actual.exitAt
                or not actual.decisionEligible or actual.availableAt >= actual.entryAt):
            raise ValueError("Forecast timing does not match observed panel")
        buy = float(actual.entryOpen * (1 + slippage_bps / 10000))
        sell = float(actual.exitOpen * (1 - slippage_bps / 10000))
        quantity = min(int(.25 * cash / buy), int(.01 * actual.volume))
        while quantity > 0 and quantity * buy + fees(quantity * buy, "buy") > cash:
            quantity -= 1
        status = "filled"
        if quantity <= 0 or quantity > .01 * actual.entryVolume:
            status = "entry_not_fillable"
        elif book == "model" and row.prediction <= ((fees(quantity * buy, "buy") + fees(quantity * buy, "sell"))
                                                   / (quantity * buy) + 2 * slippage_bps / 10000):
            status = "quantity_cost_gate"
        decisions.append({"at": str(timestamp), "symbol": row.symbol, "quantity": quantity, "status": status})
        if status != "filled":
            continue
        if quantity > .01 * actual.exitVolume:
            raise ValueError("Unresolved exit participation; replay invalid, preserve failure")
        if (actual.exitAt.date() != actual.entryAt.date() or actual.exitAt.strftime("%H:%M") > "15:10"):
            raise ValueError("Overnight or late exit")
        entry_fee, exit_fee = fees(quantity * buy, "buy"), fees(quantity * sell, "sell")
        cash -= quantity * buy + entry_fee
        held = panel[(panel.symbol == row.symbol) & (panel.barStart >= actual.entryAt)
                     & (panel.barStart < actual.exitAt)]
        if len(held) != 6 or held.volume.le(0).any():
            raise ValueError("Unresolved holding path")
        for mark in held.itertuples():
            curve.append({"at": str(mark.barStart + BAR), "equity": float(cash + quantity * mark.close)})
        cash += quantity * sell - exit_fee
        curve.append({"at": str(actual.exitAt), "equity": float(cash)})
        trades.append({"symbol": row.symbol, "signalAt": str(timestamp), "entryAt": str(actual.entryAt),
            "exitAt": str(actual.exitAt), "quantity": quantity, "entryPrice": buy, "exitPrice": sell,
            "fees": entry_fee + exit_fee, "pnl": quantity * (sell - buy) - entry_fee - exit_fee})
        available = actual.exitAt + BAR
    values = np.array([p["equity"] for p in curve])
    return {"book": book, "slippageBps": slippage_bps, "capital": capital, "endingCapital": float(cash),
            "netReturn": float(cash / capital - 1), "maximumDrawdown": float((1 - values / np.maximum.accumulate(values)).max()),
            "trades": trades, "decisions": decisions, "curve": curve, "livePromotionAllowed": False}


def study(frames, calendar, provenance, observed_at):
    """Complete offline pilot, all fixed books/costs and failed coverage visible.

    Actual market use requires a separately frozen source/input/model protocol.
    Persist returned artifacts privately with the experiment's immutable manifest.
    Synthetic tests establish mechanics only. No promotion rule is inferred here.
    """
    panel, manifest = build_panel(frames, calendar, provenance, observed_at)
    forecasts, fits = walk_forward(panel)
    scored = forecasts.dropna(subset=["actual", "prediction", "probability"])
    quality = {"scoredRows": len(scored), "testDays": int(forecasts.session.nunique()),
               "unscoredRows": len(forecasts) - len(scored)}
    if len(scored):
        event = scored.actual.gt(reference_hurdle()).astype(float)
        quality["mae"] = {key: float((scored.actual - value).abs().mean()) for key, value in
                          [("model", scored.prediction), ("zero", 0.), ("pastMean", scored.pastMean)]}
        quality["brier"] = {key: float(((event - value) ** 2).mean()) for key, value in
                            [("model", scored.probability), ("neutral", .5), ("pastPrevalence", scored.pastProbability)]}
    comparison = {str(bps): {book: replay(panel, forecasts, book=book, slippage_bps=bps)
                              for book in BOOKS} for bps in [0., 5., 10.]}
    return {"version": VERSION, "manifest": manifest, "fits": fits, "quality": quality,
            "comparison": comparison, "livePromotionAllowed": False}, forecasts
