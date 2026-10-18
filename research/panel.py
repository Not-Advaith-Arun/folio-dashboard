"""Explicit-session, per-security research inputs; never execution eligibility.

Keep unavailable outcomes beside ready features. Dropping them before choosing
historical trades would select securities using future data availability. The
legacy balanced-cohort engine remains separate until its benchmark, selection
and paper-ledger semantics are versioned for an unbalanced universe.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd

PANEL_VERSION = "session-panel-1"
HISTORY = 200
HORIZON = 5
PRICE_COLUMNS = ["open", "high", "low", "close", "volume"]
STOCK_FEATURES = ["return1", "return5", "return10", "return20", "return60", "return120",
                  "volatility20", "range", "volumeRatio", "trend200"]


def daily_index(index):
    return (isinstance(index, pd.DatetimeIndex) and index.tz is None
            and not index.hasnans and index.is_unique and index.is_monotonic_increasing
            and index.equals(index.normalize()))


def session_panel(frames, sessions, *, calendar_source, price_basis, horizon=HORIZON):
    """Return all security/session rows and a JSON-safe coverage manifest.

    Sessions must be supplied independently of the requested securities, through
    the observation cutoff. Neither business-day generation nor the union of
    provider bars is automatically a verified exchange calendar. The caller must
    disclose its source. Price adjustment/identity correctness is not established
    by structural checks, so investment eligibility is always false.
    """
    if type(horizon) is not int or not 1 <= horizon <= 252:
        raise ValueError("Holding horizon must be 1–252 explicit sessions.")
    if not frames or not all(isinstance(s, str) and s.strip() for s in frames):
        raise ValueError("Nonempty security identifiers required.")
    if not daily_index(sessions) or len(sessions) == 0:
        raise ValueError("Explicit sorted unique timezone-free daily sessions required.")
    if not all(isinstance(s, str) and s.strip() for s in [calendar_source, price_basis]):
        raise ValueError("Calendar source and price basis must be disclosed.")
    dates = pd.Series(sessions, index=sessions)
    entry, end = dates.shift(-1), dates.shift(-(horizon + 1))
    panels, coverage = [], []
    for symbol, source in sorted(frames.items()):
        if not isinstance(source, pd.DataFrame):
            raise ValueError(f"DataFrame required for {symbol}.")
        source_hash = hashlib.sha256(source.to_json(orient="split", date_format="iso",
                                                     double_precision=15).encode()).hexdigest()
        reason = None
        if not daily_index(source.index):
            reason = "Invalid, duplicate or unordered daily session index"
        elif not source.index.isin(sessions).all():
            reason = "Source observations outside supplied calendar"
        elif not source.columns.is_unique or not set(PRICE_COLUMNS).issubset(source.columns):
            reason = "Missing or duplicate OHLCV columns"
        if reason:
            f = pd.DataFrame(np.nan, index=sessions, columns=PRICE_COLUMNS)
            present = pd.Series(False, index=sessions)
        else:
            present = pd.Series(sessions.isin(source.index), index=sessions)
            f = source[PRICE_COLUMNS].apply(pd.to_numeric, errors="coerce").astype(float).reindex(sessions)
        prices = f[["open", "high", "low", "close"]]
        valid = (np.isfinite(f).all(axis=1) & (prices > 0).all(axis=1) & (f.volume >= 0)
                 & (f.high >= prices.max(axis=1)) & (f.low <= prices.min(axis=1)))
        invalid = present & ~valid
        # Retain original source/hash and dated exclusions; never repair prices.
        f = f.where(valid, np.nan)
        x = pd.DataFrame({"symbol": symbol, "date": sessions}, index=sessions)
        for column in PRICE_COLUMNS:
            x[column] = f[column]
        for lag in [1, 5, 10, 20, 60, 120]:
            x[f"return{lag}"] = f.close.pct_change(lag, fill_method=None)
        x["volatility20"] = x.return1.rolling(20, min_periods=20).std()
        x["range"] = (f.high - f.low) / f.close
        x["volumeRatio"] = f.volume / f.volume.rolling(20, min_periods=20).mean().replace(0, np.nan)
        x["trend200"] = f.close / f.close.rolling(HISTORY, min_periods=HISTORY).mean() - 1
        x[STOCK_FEATURES] = x[STOCK_FEATURES].replace([np.inf, -np.inf], np.nan)
        continuous = valid.rolling(HISTORY, min_periods=HISTORY).sum().eq(HISTORY)
        ready = continuous & (f.volume > 0) & x[STOCK_FEATURES].notna().all(axis=1)
        x["featureStatus"] = np.select(
            [~present, invalid, f.volume.eq(0), ~continuous, ~ready],
            ["missing_observation", "invalid_ohlcv", "zero_volume", "insufficient_contiguous_history", "nonfinite_features"],
            default="ready")
        if reason:
            x["featureStatus"] = "quarantined_security"
        x["featureReady"] = ready
        x.loc[~ready, STOCK_FEATURES] = np.nan
        # Fixed session positions: never advance an unavailable entry/exit to the
        # next observed bar or shorten the holding horizon by dropping dates.
        future_valid = valid & f.volume.gt(0)
        observed_path = pd.concat([future_valid.shift(-i, fill_value=False)
                                  for i in range(1, horizon + 2)], axis=1).all(axis=1)
        actual = f.open.shift(-(horizon + 1)) / f.open.shift(-1) - 1
        finite_outcome = np.isfinite(actual)
        label_ready = valid & end.notna() & observed_path & finite_outcome
        x["entryDate"], x["labelEnd"] = entry, end
        x["labelStatus"] = np.select([~valid, end.isna(), ~observed_path, ~finite_outcome],
                                     ["no_signal_observation", "pending_future_sessions", "unavailable_future_path", "nonfinite_outcome"],
                                     default="observed")
        x["actual"] = actual.where(label_ready)
        x["investmentEligible"] = False
        panels.append(x.reset_index(drop=True))
        coverage.append({
            "symbol": symbol, "inputRows": len(source), "sourceFrameSha256": source_hash,
            "securityQuarantine": reason, "validObservations": int(valid.sum()),
            "missingSessions": int((~present).sum()) if reason is None else None,
            "invalidRows": [str(d.date()) for d in sessions[invalid]],
            "featureReadyRows": int(ready.sum()), "labeledReadyRows": int((ready & label_ready).sum()),
            "readyWithoutOutcome": int((ready & ~label_ready).sum()),
            "latestFeatureStatus": str(x.featureStatus.iloc[-1]),
            "featureStatusCounts": {str(k): int(v) for k, v in x.featureStatus.value_counts().items()},
            "labelStatusCounts": {str(k): int(v) for k, v in x.labelStatus.value_counts().items()},
            "investmentEligible": False,
        })
    panel = pd.concat(panels, ignore_index=True)
    manifest = {
        "version": PANEL_VERSION, "calendarSource": calendar_source, "priceBasis": price_basis,
        "calendarSha256": hashlib.sha256("\n".join(sessions.strftime("%Y-%m-%d")).encode()).hexdigest(),
        "start": str(sessions[0].date()), "asOf": str(sessions[-1].date()), "sessions": len(sessions),
        "historySessions": HISTORY, "holdingSessions": horizon, "features": STOCK_FEATURES.copy(),
        "securities": coverage, "rows": len(panel), "featureReadyRows": int(panel.featureReady.sum()),
        "limitations": ["Research-feature availability is not sufficient training history or investment eligibility.",
                        "Calendar, historical membership, adjustment/actions, lots and execution are not verified here.",
                        "No cohort-relative target or benchmark; never filter candidates by future label availability.",
                        "Missing sessions include unknown pre-listing/post-delisting periods; causes are not inferred.",
                        "Current source revisions do not prove information was available at historical decisions."],
    }
    return panel, manifest


def panel_payload(panel, manifest):
    """Portable local research artifact; NaN/NaT are JSON null, never prices."""
    rows = panel.copy()
    for name in ["date", "entryDate", "labelEnd"]:
        rows[name] = rows[name].dt.strftime("%Y-%m-%d")
    return {"coverage": manifest, "rows": json.loads(rows.to_json(orient="records", double_precision=15))}
