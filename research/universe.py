"""Exchange-file discovery registry; membership is observed now, not historical.

No market data is fabricated and no listing is an investment recommendation.
SME lots, surveillance and liquidity require separate dated verification.
"""
from __future__ import annotations

import csv
import hashlib
import io
import math
import re
from datetime import datetime
from collections import Counter

UNIVERSE_VERSION = "exchange-discovery-1"


def rows(text):
    if len(text.encode()) > 3_000_000:
        raise ValueError("Exchange file exceeds bounded input size")
    reader = csv.DictReader(io.StringIO(text.lstrip("\ufeff")))
    if not reader.fieldnames:
        raise ValueError("Missing exchange CSV header")
    reader.fieldnames = [s.strip().upper().replace("_", " ") for s in reader.fieldnames]
    return [{k: str(v or "").strip() for k, v in row.items() if k} for row in reader]


def identity(row, index=False):
    symbol = row.get("SYMBOL", "")
    isin = row.get("ISIN CODE" if index else "ISIN NUMBER", "")
    if not re.fullmatch(r"[A-Z0-9&.\-]{1,40}", symbol) or not re.fullmatch(r"IN[A-Z0-9]{10}", isin):
        raise ValueError("Invalid exchange symbol or ISIN")
    return symbol, isin


def build_registry(main_csv, sme_csv, index_files, observed_at):
    """Import complete supplied current files; fail on malformed identities."""
    observed = datetime.fromisoformat(observed_at.replace("Z", "+00:00"))
    if observed.tzinfo is None:
        raise ValueError("Timezone-aware observation timestamp required")
    securities, sources = {}, []
    for segment, text in [("main", main_csv), ("sme", sme_csv)]:
        records = rows(text)
        if not records or not {"SYMBOL", "NAME OF COMPANY", "SERIES", "ISIN NUMBER"}.issubset(records[0]):
            raise ValueError(f"Missing {segment} exchange columns")
        sources.append({"name": segment, "sha256": hashlib.sha256(text.encode()).hexdigest(), "rows": len(records)})
        seen = set()
        for row in records:
            symbol, isin = identity(row)
            key = (symbol, row["SERIES"])
            if key in seen:
                raise ValueError("Duplicate symbol/series in exchange file")
            seen.add(key)
            name = row["NAME OF COMPANY"]
            if not name:
                raise ValueError("Missing company name")
            lot = row.get("MARKET LOT", "")
            if lot and (not lot.isdecimal() or int(lot) < 1):
                raise ValueError("Invalid exchange market lot")
            listing_date = None
            value = row.get("DATE OF LISTING", "")
            for fmt in ["%d-%b-%Y", "%d-%b-%y"]:
                try:
                    listing_date = datetime.strptime(value, fmt).date().isoformat()
                    break
                except ValueError:
                    pass
            security = securities.setdefault(isin, {"isin": isin, "name": name, "listings": [], "groups": [], "industries": [], "flags": [], "priceHistoryStatus": "not audited", "modelStatus": "not evaluated", "investmentEligible": False})
            security["listings"].append({"exchange": "NSE", "symbol": symbol, "series": row["SERIES"], "segment": segment, "marketLot": int(lot) if lot else None, "listingDate": listing_date})
            if segment == "sme":
                security["flags"].append("SME execution and lot verification required")
            if not lot:
                security["flags"].append("Market lot unknown")
            if row["SERIES"] != "EQ":
                security["flags"].append("Non-EQ series; separate trading rules required")
            if listing_date is None:
                security["flags"].append("Listing date unverified")
            elif listing_date > observed.date().isoformat():
                security["flags"].append("Listing begins after observation date")
    unmatched = []
    for name, text in sorted(index_files.items()):
        records = rows(text)
        if not records or not {"SYMBOL", "ISIN CODE", "INDUSTRY"}.issubset(records[0]):
            raise ValueError("Missing index membership columns")
        sources.append({"name": name, "sha256": hashlib.sha256(text.encode()).hexdigest(), "rows": len(records)})
        seen = set()
        for row in records:
            try:
                symbol, isin = identity(row, index=True)
            except ValueError:
                # Official index snapshots can contain temporary dummy securities.
                # Preserve the row and exclusion; never invent a tradeable ticker.
                unmatched.append({"group": name, "symbol": row.get("SYMBOL", ""),
                                  "isin": row.get("ISIN CODE", ""), "reason": "Nonstandard identifier; index placeholder or invalid row"})
                continue
            if isin in seen:
                raise ValueError("Duplicate ISIN in index membership file")
            seen.add(isin)
            security = securities.get(isin)
            if security is None:
                unmatched.append({"group": name, "symbol": symbol, "isin": isin, "reason": "No matching exchange-master ISIN"})
                continue
            security["groups"].append(name)
            if row["INDUSTRY"]:
                security["industries"].append(row["INDUSTRY"])
            if symbol not in {l["symbol"] for l in security["listings"]}:
                security["flags"].append("Index/exchange symbol mismatch")
    for security in securities.values():
        for field in ["groups", "industries", "flags"]:
            security[field] = sorted(set(security[field]))
        if len(security["industries"]) > 1:
            security["flags"].append("Industry classification conflict")
        if len({l["segment"] for l in security["listings"]}) > 1:
            security["flags"].append("Main/SME overlap needs migration reconciliation")
    data = sorted(securities.values(), key=lambda s: s["isin"])
    return {"version": UNIVERSE_VERSION, "observedAt": observed_at, "historicalMembershipVerified": False,
            "coverage": {"exchange": "NSE only", "uniqueSecurities": len(data), "sourceRows": sources,
                         "industryTagged": sum(bool(s["industries"]) for s in data),
                         "groupCounts": dict(Counter(g for s in data for g in s["groups"])),
                         "unmatchedIndexRows": unmatched, "evaluatedSecurities": 0,
                         "investmentEligibleSecurities": 0}, "securities": data,
            "limitations": ["Current observed snapshots are for discovery; not point-in-time historical universe or AMFI cap classification.",
                            "No prices, liquidity/spreads, surveillance, execution availability or model quality verified by registry import.",
                            "SME lots absent from this master remain unknown; ordinary one-share replay must not be used.",
                            "BSE-only securities and historical suspended/delisted securities are not covered.",
                            "Index membership may change; actual file row counts retained without forcing the index name's count."]}


def daily_snapshot(text, registry, trading_date, observed_at):
    """Audit a downloaded full bhavcopy; symbol joins are discovery-only.

    One day of traded volume cannot establish liquidity or investment eligibility.
    Preserve unmatched instruments and quarantined bars rather than inventing data.
    """
    expected = datetime.strptime(trading_date, "%Y-%m-%d").date()
    observed = datetime.fromisoformat(observed_at.replace("Z", "+00:00"))
    if observed.tzinfo is None or expected > observed.date():
        raise ValueError("Dated observation must follow the trading date")
    records = rows(text)
    columns = {"SYMBOL", "SERIES", "DATE1", "OPEN PRICE", "HIGH PRICE", "LOW PRICE", "CLOSE PRICE", "TTL TRD QNTY", "TURNOVER LACS"}
    if not records or not columns.issubset(records[0]):
        raise ValueError("Missing full bhavcopy columns")
    lookup = {}
    for security in registry["securities"]:
        for listing in security["listings"]:
            key = (listing["symbol"], listing["series"])
            if key in lookup and lookup[key] != security["isin"]:
                raise ValueError("Ambiguous current exchange identity")
            lookup[key] = security["isin"]
    seen, audited, quarantined, unmatched = set(), [], [], []
    for row in records:
        key = (row["SYMBOL"], row["SERIES"])
        if key in seen:
            raise ValueError("Duplicate symbol/series in daily file")
        seen.add(key)
        if datetime.strptime(row["DATE1"], "%d-%b-%Y").date() != expected:
            raise ValueError("Daily file contains a different trading date")
        try:
            prices = {field: float(row[f"{field.upper()} PRICE"]) for field in ("open", "high", "low", "close")}
            volume, turnover = float(row["TTL TRD QNTY"]), float(row["TURNOVER LACS"]) * 100000
            if not all(math.isfinite(v) for v in [*prices.values(), volume, turnover]):
                raise ValueError("Nonfinite daily value")
            if min(prices.values()) <= 0 or not prices["low"] <= min(prices["open"], prices["close"]) <= max(prices["open"], prices["close"]) <= prices["high"]:
                raise ValueError("Invalid daily OHLC")
            if volume < 0 or not volume.is_integer() or turnover < 0:
                raise ValueError("Invalid daily volume/turnover")
        except (ValueError, TypeError) as error:
            quarantined.append({"symbol": key[0], "series": key[1], "reason": str(error), "sourceRow": row})
            continue
        isin = lookup.get(key)
        if isin is None:
            unmatched.append({"symbol": key[0], "series": key[1], "reason": "No exact current-master symbol/series match"})
        audited.append({"symbol": key[0], "series": key[1], "isin": isin, "date": trading_date,
                        **prices, "volume": int(volume), "turnoverInr": turnover})
    matched = {r["isin"] for r in audited if r["isin"]}
    return {"version": UNIVERSE_VERSION, "tradingDate": trading_date, "observedAt": observed_at,
            "sourceSha256": hashlib.sha256(text.encode()).hexdigest(),
            "coverage": {"sourceRows": len(records), "auditedRows": len(audited), "matchedSecurities": len(matched),
                         "missingRegistrySecurities": len(registry["securities"]) - len(matched),
                         "matchedGroupCounts": dict(Counter(g for s in registry["securities"] if s["isin"] in matched for g in s["groups"])),
                         "quarantinedRows": len(quarantined), "unmatchedRows": len(unmatched)},
            "bars": audited, "quarantined": quarantined, "unmatched": unmatched,
            "limitations": ["Single-day discovery snapshot; no historical universe, liquidity, fill or model eligibility established.",
                            "Daily file lacks ISIN: joins use exact symbols/series in the current master and are not historical identity proof.",
                            "Missing rows may reflect another series, no trade, suspension, separate SME file or other cause; never inferred as zero returns.",
                            "Price bars are not corporate-action-adjusted total returns. No spreads or intraday executable quotes included."]}
