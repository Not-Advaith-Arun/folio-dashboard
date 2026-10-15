"""Assemble local, source-bound daily histories without a current-master join.

These are unadjusted observations, not verified continuous investment histories.
The explicit date grid must stay separate from a certified exchange calendar.
"""
from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from datetime import date, datetime
from zoneinfo import ZoneInfo

import pandas as pd

from research.dated_universe import udiff_snapshot
from research.panel import PRICE_COLUMNS

VERSION = "dated-history-1"
MAX_GRID_ROWS = 2_000_000


def _key(row):
    # Do not merge settlement series or sessions, even for the same ISIN.
    parts = [row[k] for k in ["exchange", "segment", "isin", "series", "session"]]
    if any(":" in p for p in parts):
        raise ValueError("Ambiguous history key component")
    return ":".join(parts)


def udiff_histories(archives, requested_dates, *, knowledge_as_of, calendar_source):
    """Return sparse OHLCV frames and a complete instrument/date audit ledger.

    Each archive supplies payload bytes, filename, tradingDate, observedAt,
    sha256 and sourceUrl from its acquisition record. Reparse original bytes;
    never trust edited snapshots or silently select among daily revisions.
    Missing files are explicit. The caller must disclose how dates were chosen.
    Current acquisition does not prove historical publication availability.
    """
    cutoff = datetime.fromisoformat(knowledge_as_of.replace("Z", "+00:00"))
    if cutoff.tzinfo is None:
        raise ValueError("Timezone-aware knowledge cutoff required")
    days = list(requested_dates)
    if (not days or days != sorted(set(days))
            or any(date.fromisoformat(d).isoformat() != d for d in days)):
        raise ValueError("Sorted unique explicit ISO dates required")
    if date.fromisoformat(days[-1]) > cutoff.astimezone(ZoneInfo("Asia/Kolkata")).date():
        raise ValueError("Requested dates exceed knowledge cutoff")
    if not isinstance(calendar_source, str) or not calendar_source.strip():
        raise ValueError("Date-grid source must be disclosed")
    snapshots = {}
    for source in archives:
        day = source["tradingDate"]
        if day not in days or day in snapshots:
            raise ValueError("Outside requested dates or duplicate/revised daily file")
        if hashlib.sha256(source["payload"]).hexdigest() != source["sha256"]:
            raise ValueError("Archive hash differs from acquisition record")
        snapshot = udiff_snapshot(source["payload"], source["filename"], day, source["observedAt"])
        if datetime.fromisoformat(source["observedAt"].replace("Z", "+00:00")) > cutoff:
            raise ValueError("Source acquisition exceeds knowledge cutoff")
        if not isinstance(source["sourceUrl"], str) or not source["sourceUrl"].strip():
            raise ValueError("Reviewed acquisition source URL required")
        snapshots[day] = {**snapshot, "sourceUrl": source["sourceUrl"]}
    if not snapshots:
        raise ValueError("At least one acquired daily file required")

    observations, sources = defaultdict(dict), []
    alias_isins, isin_series = defaultdict(set), defaultdict(set)
    for day, snapshot in sorted(snapshots.items()):
        sources.append({k: snapshot[k] for k in [
            "version", "tradingDate", "observedAt", "sourceUrl", "sourceFilename",
            "sourceSha256", "csvSha256", "coverage", "quarantined"]})
        bars = {_key(row): row for row in snapshot["bars"]}
        for row in snapshot["identities"]:
            key = _key(row)
            if day in observations[key]:
                raise ValueError("Ambiguous same-day ISIN/series/session history")
            observations[key][day] = (row, bars.get(key))
            alias_isins[(row["symbol"], row["series"], row["session"])].add(row["isin"])
            isin_series[row["isin"]].add(row["series"])
        if len(observations) * len(days) > MAX_GRID_ROWS:
            raise ValueError("History grid exceeds bounded output size")

    frames, histories, changes = {}, [], []
    status_counts = Counter()
    tracked = ["symbol", "instrumentId", "instrumentType", "reportedBoardLot", "reportedBoardLotRaw"]
    for key, observed in sorted(observations.items()):
        records, price_rows, price_dates, previous = [], [], [], None
        for day in days:
            snapshot = snapshots.get(day)
            record = {"date": day, "investmentEligible": False,
                      "sourceSha256": snapshot["sourceSha256"] if snapshot else None}
            if day in observed:
                identity, bar = observed[day]
                record.update(identity)
                record["status"] = "observed" if bar is not None else "quarantined_price"
                price_rows.append({c: bar[c] for c in PRICE_COLUMNS} if bar else {})
                price_dates.append(day)
                if previous:
                    delta = {f: {"before": previous[f], "after": identity[f]}
                             for f in tracked if previous[f] != identity[f]}
                    if delta:
                        changes.append({"key": key, "previousObservedDate": previous["date"],
                                        "observedDate": day, "changes": delta})
                previous = identity
            else:
                record["status"] = "not_observed_in_file" if snapshot else "missing_file"
            records.append(record)
            status_counts[record["status"]] += 1
        frame = pd.DataFrame(price_rows, index=pd.DatetimeIndex(price_dates), columns=PRICE_COLUMNS)
        frames[key] = frame
        histories.append({"key": key, "investmentEligible": False, "records": records,
                          "sourceFrameSha256": hashlib.sha256(frame.to_json(
                              orient="split", date_format="iso", double_precision=15).encode()).hexdigest()})
    ledger = {
        "version": VERSION, "knowledgeAsOf": knowledge_as_of, "calendarSource": calendar_source,
        "requestedDates": days, "dateGridSha256": hashlib.sha256("\n".join(days).encode()).hexdigest(),
        "missingFileDates": [d for d in days if d not in snapshots],
        "sources": sources, "histories": histories, "observedChanges": changes,
        "aliasIdentityConflicts": [{"symbol": s, "series": r, "session": t, "isins": sorted(ids)}
                                   for (s, r, t), ids in sorted(alias_isins.items()) if len(ids) > 1],
        "multipleSeriesIsins": [{"isin": isin, "series": sorted(series)}
                                for isin, series in sorted(isin_series.items()) if len(series) > 1],
        "coverage": {"files": len(sources), "requestedDates": len(days), "histories": len(histories),
                     "uniqueIsins": len(isin_series), "gridRows": sum(status_counts.values()),
                     "statusCounts": dict(sorted(status_counts.items())),
                     "datedIdentities": sum(len(o) for o in observations.values()),
                     "observedChanges": len(changes)},
        "limitations": [
            "Requested dates and observed files do not certify a complete exchange calendar or universe.",
            "Absent records are unknown, not verified listing, delisting, suspension or zero trading.",
            "Same ISIN/series/session is an observation key, not proof of corporate-action continuity.",
            "Observed alias/ID/lot changes are not effective-date action or execution-rule evidence.",
            "Prices are unadjusted; unsupported price rows and unknown instrument classes remain explicit.",
            "Acquisition cutoff is current research knowledge, not historical first-seen publication.",
            "No current-master filter, filled prices, inferred actions, training or investment eligibility.",
        ],
    }
    return frames, ledger
