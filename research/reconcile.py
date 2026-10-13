"""Versioned, evidence-bound session exclusions. Never alter original inputs.

This removes reviewed closures from a supplied observation calendar; it does
not certify the other dates as exchange sessions or reconcile price adjustments.
"""
from __future__ import annotations

import hashlib
import json
from datetime import date, datetime
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

import pandas as pd

from research.panel import daily_index

CORRECTION_VERSION = "calendar-exclusion-1"


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
