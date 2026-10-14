"""Versioned, evidence-bound data reconciliation. Never alter original inputs.

This removes reviewed closures from a supplied observation calendar; it does
not certify the other dates as exchange sessions or reconcile price adjustments.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from datetime import date, datetime
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

import pandas as pd

from research.panel import PRICE_COLUMNS, daily_index
from research.universe import daily_snapshot

CORRECTION_VERSION = "calendar-exclusion-1"
BAR_AUDIT_VERSION = "source-reconciliation-1"


def _instant(value):
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("Correction timestamps require an explicit timezone.")
    return parsed


def _day(value):
    parsed = date.fromisoformat(value)
    if parsed.isoformat() != value:
        raise ValueError("Closure dates require YYYY-MM-DD.")
    return pd.Timestamp(parsed)


def _frame_hash(frame):
    # Canonical frame hash is separate from the caller's original file hash.
    content = frame.to_json(orient="split", date_format="iso", double_precision=15)
    return hashlib.sha256(content.encode()).hexdigest()


def _calendar_hash(sessions):
    return hashlib.sha256("\n".join(sessions.strftime("%Y-%m-%d")).encode()).hexdigest()


def exclude_closed_sessions(frames, sessions, notices, *, market, market_timezone, knowledge_as_of, sources):
    """Derive corrected frames/calendar plus a private exclusion ledger.

    Each notice is a manually reviewed closure with market, closedSession,
    publishedDate, observedAt, sourceUrl, sourceSha256, sourceKind and reason.
    `sources` contains the corresponding captured evidence bytes, keyed by hash.
    A captured excerpt is explicitly distinguished from an original document.
    observedAt is actual acquisition, never backdated to publication. Caller
    declares that all frames belong to `market`; identity remains unaudited.
    """
    cutoff = _instant(knowledge_as_of)
    local_zone = ZoneInfo(market_timezone)
    if not frames or not daily_index(sessions) or not len(sessions) or not market:
        raise ValueError("Nonempty frames, market and explicit daily calendar required.")
    reviewed, closed = [], set()
    for notice in notices:
        if notice["market"] != market:
            raise ValueError("Closure market does not match the input market.")
        day, published = _day(notice["closedSession"]), _day(notice["publishedDate"])
        observed = _instant(notice["observedAt"])
        if observed > cutoff:
            raise ValueError("Evidence was observed after the correction knowledge cutoff.")
        # Date-only publication is metadata, not an intraday availability claim.
        if published.date() > observed.astimezone(local_zone).date() or published > day:
            raise ValueError("Invalid closure publication chronology.")
        if day in closed:
            raise ValueError("Duplicate closure; resolve notice precedence explicitly.")
        url = urlparse(notice["sourceUrl"])
        if url.scheme != "https" or not url.netloc or not notice["reason"].strip():
            raise ValueError("Source URL and reviewed closure reason required.")
        if notice["sourceKind"] not in {"original-document", "captured-excerpt"}:
            raise ValueError("Explicit source kind required.")
        evidence = sources.get(notice["sourceSha256"])
        if not isinstance(evidence, bytes) or hashlib.sha256(evidence).hexdigest() != notice["sourceSha256"]:
            raise ValueError("Closure source hash does not match captured evidence.")
        closed.add(day)
        reviewed.append({**notice, "status": "removed" if day in sessions else "already_absent"})
    calendar = sessions[~sessions.isin(closed)].copy()
    if not len(calendar):
        raise ValueError("Closure correction would remove the entire calendar.")
    outputs, changes = {}, []
    for symbol, frame in sorted(frames.items()):
        if not daily_index(frame.index) or not frame.index.isin(sessions).all() or not frame.columns.is_unique:
            raise ValueError(f"Invalid source calendar/columns for {symbol}.")
        mask = frame.index.isin(closed)
        corrected = frame.loc[~mask].copy(deep=True)
        outputs[symbol] = corrected
        changes.append({
            "symbol": symbol, "inputRows": len(frame), "outputRows": len(corrected),
            "inputFrameSha256": _frame_hash(frame), "outputFrameSha256": _frame_hash(corrected),
            "excludedRows": [{"session": str(d.date()), "values": json.loads(frame.loc[d].to_json(double_precision=15))}
                             for d in frame.index[mask]],
        })
    return outputs, calendar, {
        "version": CORRECTION_VERSION, "market": market, "marketTimezone": market_timezone,
        "knowledgeAsOf": cutoff.isoformat(),
        "inputCalendarSha256": _calendar_hash(sessions), "outputCalendarSha256": _calendar_hash(calendar),
        "inputSessions": len(sessions), "outputSessions": len(calendar),
        "notices": reviewed, "securities": changes, "calendarCompletelyVerified": False,
        "limitations": ["Only supplied reviewed closures are reconciled; other sessions remain unverified.",
                        "Acquired-now corrections do not establish historical first-seen availability.",
                        "Market scope is caller-declared; historical identity/actions/lots are not reconciled.",
                        "No bar values are repaired; original files and earlier predictions/results must be preserved."],
    }


def quarantine_bhavcopy_conflicts(frames, symbol_map, source, source_bytes, *,
                                 knowledge_as_of, price_basis, shape_tolerance=0.001):
    """Compare one supplied NSE CM daily file; quarantine, never repair, conflicts.

    Explicit symbol/series mappings are caller-reviewed candidates, not historical
    identity proof. A common OHLC scale is only a compatibility diagnostic. Volume
    differences stay unresolved because adjustment/action bases are unverified.
    Missing official rows stay visible and do not imply zero trading activity.
    Acquired-now source revisions cannot establish historical first-seen evidence.
    """
    cutoff, observed = _instant(knowledge_as_of), _instant(source["observedAt"])
    day = _day(source["tradingDate"])
    if observed > cutoff:
        raise ValueError("Evidence was observed after the reconciliation knowledge cutoff.")
    if source["market"] != "NSE:CM":
        raise ValueError("Full bhavcopy comparison requires NSE:CM market scope.")
    url = urlparse(source["url"])
    if url.scheme != "https" or not url.netloc or not price_basis.strip():
        raise ValueError("Source URL and provider price basis required.")
    if not isinstance(source_bytes, bytes) or hashlib.sha256(source_bytes).hexdigest() != source["sha256"]:
        raise ValueError("Exchange source hash does not match captured bytes.")
    if not math.isfinite(shape_tolerance) or not 0 <= shape_tolerance < 1:
        raise ValueError("Finite relative shape tolerance in [0, 1) required.")
    if not frames or set(frames) != set(symbol_map):
        raise ValueError("An explicit symbol/series mapping is required for every frame.")
    keys = [(m["symbol"], m["series"]) for m in symbol_map.values()]
    if any(not all(isinstance(v, str) and v.strip() for v in key) for key in keys) or len(keys) != len(set(keys)):
        raise ValueError("Nonempty, unambiguous symbol/series mappings required.")
    # Empty registry intentionally avoids presenting current-master ISIN joins
    # as historical identity evidence. Duplicate/date/structural checks still run.
    official = daily_snapshot(source_bytes.decode("utf-8-sig"), {"securities": []},
                              source["tradingDate"], source["observedAt"])
    bars = {(r["symbol"], r["series"]): r for r in official["bars"]}
    invalid = {(r["symbol"], r["series"]): r for r in official["quarantined"]}
    outputs, records = {}, []
    for symbol, frame in sorted(frames.items()):
        if (not daily_index(frame.index) or not frame.columns.is_unique
                or not set(PRICE_COLUMNS).issubset(frame.columns)):
            raise ValueError(f"Valid daily index and OHLCV columns required for {symbol}.")
        mapping = symbol_map[symbol]
        key = (mapping["symbol"], mapping["series"])
        derived = frame.copy(deep=True)
        record = {"symbol": symbol, "lookup": dict(mapping), "date": source["tradingDate"],
                  "inputFrameSha256": _frame_hash(frame), "quarantined": False,
                  "investmentEligible": False, "reasons": []}
        if day not in frame.index:
            record["status"] = "provider_missing"
        else:
            record["originalRow"] = json.loads(frame.loc[day].to_json(double_precision=15))
            if key not in bars:
                record["status"] = "official_invalid" if key in invalid else "official_missing"
                record["officialProblem"] = invalid.get(key)
            else:
                bar = bars[key]
                record["officialRow"] = bar.copy()
                values = pd.to_numeric(frame.loc[day, PRICE_COLUMNS], errors="coerce")
                o, h, l, c, v = (float(values[field]) for field in PRICE_COLUMNS)
                if (not all(math.isfinite(x) for x in (o, h, l, c, v))
                        or min(o, h, l, c) <= 0 or not l <= min(o, c) <= max(o, c) <= h
                        or v < 0 or not v.is_integer()):
                    record["reasons"].append("provider_invalid")
                else:
                    # Same-day normalization cancels one uniform price scale;
                    # no factor is applied to either source or inferred as valid.
                    shape_error = max(abs((float(values[k]) / c) / (bar[k] / bar["close"]) - 1)
                                      for k in PRICE_COLUMNS[:4])
                    record.update({"shapeMaxRelativeError": shape_error,
                                   "closeScaleDiagnostic": c / bar["close"],
                                   "sameVolume": int(v) == bar["volume"]})
                    if shape_error > shape_tolerance:
                        record["reasons"].append("ohlc_shape_conflict")
                    if int(v) != bar["volume"]:
                        record["reasons"].append("zero_volume_conflict" if v == 0 or bar["volume"] == 0
                                                 else "volume_basis_unresolved")
                record["quarantined"] = bool(record["reasons"])
                record["status"] = "quarantined_conflict" if record["quarantined"] else "compatible_unverified"
                if record["quarantined"]:
                    # Keep the session position. NaN invalidates rolling history
                    # and exact future paths in session_panel without date shifts.
                    derived = derived.astype({k: float for k in PRICE_COLUMNS})
                    derived.loc[day, PRICE_COLUMNS] = float("nan")
        record["outputFrameSha256"] = _frame_hash(derived)
        outputs[symbol] = derived
        records.append(record)
    return outputs, {
        "version": BAR_AUDIT_VERSION, "source": dict(source), "knowledgeAsOf": cutoff.isoformat(),
        "providerPriceBasis": price_basis, "shapeRelativeTolerance": shape_tolerance,
        "officialCoverage": official["coverage"], "comparisons": records,
        "statusCounts": dict(Counter(r["status"] for r in records)),
        "quarantinedRows": sum(r["quarantined"] for r in records),
        "limitations": ["Symbol/series mappings are not historical ISIN or corporate-action verification.",
                        "Common-scale compatibility does not establish correct adjusted or executable prices.",
                        "Volume differences remain unresolved; no split/dividend repair is inferred.",
                        "Unchecked dates and missing/invalid official rows are not certified or replaced.",
                        "Current acquisition is not historical first-seen availability; preserve prior results.",
                        "Quarantine masks values only in derived frames; the calendar and originals are retained."],
    }
