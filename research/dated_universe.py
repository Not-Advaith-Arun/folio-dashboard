"""Bounded official daily identity snapshots, independent of today's survivors.

Raw files remain local. Dated identities and reported lots do not establish a
complete historical universe, corporate-action chain, fills or eligibility.
"""
from __future__ import annotations

import csv
import hashlib
import io
import math
import re
import zipfile
from collections import Counter
from datetime import date, datetime
from zoneinfo import ZoneInfo

VERSION = "dated-universe-1"
MAX_ZIP = 3_000_000
MAX_CSV = 10_000_000


def _valid_isin(value):
    if not re.fullmatch(r"IN[A-Z0-9]{9}[0-9]", value):
        return False
    digits = "".join(str(int(c, 36)) for c in value)
    return sum((2 * int(c) // 10 + 2 * int(c) % 10) if i % 2 else int(c)
               for i, c in enumerate(reversed(digits))) % 10 == 0


def udiff_snapshot(payload, source_filename, trading_date, observed_at):
    """Read one final NSE CM ZIP without extraction or a current-master join.

    Require explicit requested date, filename and actual acquisition timestamp.
    The caller supplies bytes downloaded from a reviewed official source. Hashes
    preserve identity of those bytes; they are not authenticity signatures.
    """
    expected = date.fromisoformat(trading_date)
    observed = datetime.fromisoformat(observed_at.replace("Z", "+00:00"))
    if expected.isoformat() != trading_date or observed.tzinfo is None:
        raise ValueError("ISO date and timezone-aware acquisition required")
    if expected > observed.astimezone(ZoneInfo("Asia/Kolkata")).date():
        raise ValueError("Acquisition cannot precede the trading date")
    member_name = f"BhavCopy_NSE_CM_0_0_0_{expected:%Y%m%d}_F_0000.csv"
    if source_filename != member_name + ".zip":
        raise ValueError("UDiFF filename/date mismatch")
    if not isinstance(payload, bytes) or not 0 < len(payload) <= MAX_ZIP:
        raise ValueError("UDiFF compressed file exceeds bounded input size")
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        members = archive.infolist()
        if len(members) != 1 or members[0].filename != member_name:
            raise ValueError("Exactly one matching UDiFF CSV member required")
        info = members[0]
        if (info.flag_bits & 1 or info.file_size > MAX_CSV
                or info.compress_type not in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}):
            raise ValueError("Unsupported or oversized UDiFF archive member")
        with archive.open(info) as handle:
            content = handle.read(MAX_CSV + 1)
        if len(content) > MAX_CSV:
            raise ValueError("UDiFF CSV exceeds bounded input size")
    reader = csv.DictReader(io.StringIO(content.decode("utf-8-sig")))
    required = {"TradDt", "BizDt", "Sgmt", "Src", "FinInstrmTp", "FinInstrmId", "ISIN",
                "TckrSymb", "SctySrs", "OpnPric", "HghPric", "LwPric", "ClsPric",
                "TtlTradgVol", "TtlTrfVal", "SsnId", "NewBrdLotQty"}
    if (not reader.fieldnames or len(reader.fieldnames) != len(set(reader.fieldnames))
            or not required.issubset(reader.fieldnames)):
        raise ValueError("Missing or duplicate UDiFF columns")
    bars, identities, quarantined, ids, aliases = [], [], [], set(), set()
    total = 0
    for row in reader:
        total += 1
        if total > 50000:
            raise ValueError("UDiFF row limit exceeded")
        if None in row or any(v is None for v in row.values()):
            raise ValueError("Malformed UDiFF row length")
        row = {k: v.strip() for k, v in row.items()}
        if row["TradDt"] != trading_date or row["BizDt"] != trading_date:
            raise ValueError("UDiFF row contains a different trading/business date")
        if row["Sgmt"] != "CM" or row["Src"] != "NSE":
            raise ValueError("UDiFF row has wrong exchange/segment scope")
        key = (row["FinInstrmId"], row["SsnId"])
        alias = (row["TckrSymb"], row["SctySrs"], row["SsnId"])
        if key in ids or alias in aliases:
            raise ValueError("Duplicate or ambiguous UDiFF daily identity")
        ids.add(key)
        aliases.add(alias)
        try:
            if (not _valid_isin(row["ISIN"]) or not row["FinInstrmId"].isdecimal()
                    or int(row["FinInstrmId"]) < 1 or not all(row[k] for k in ["TckrSymb", "SctySrs", "SsnId", "FinInstrmTp"])):
                raise ValueError("Invalid dated identity or ISIN checksum")
        except ValueError as error:
            quarantined.append({"reason": str(error), "sourceRow": row})
            continue
        lot = row["NewBrdLotQty"]
        reported_lot = int(lot) if lot.isdecimal() and int(lot) > 0 else None
        identity = {"date": trading_date, "exchange": "NSE", "segment": "CM",
                    "instrumentId": row["FinInstrmId"], "isin": row["ISIN"],
                    "symbol": row["TckrSymb"], "series": row["SctySrs"], "session": row["SsnId"],
                    "instrumentType": row["FinInstrmTp"], "name": row.get("FinInstrmNm", ""),
                    "reportedBoardLot": reported_lot, "reportedBoardLotRaw": lot,
                    "priceStatus": "quarantined", "investmentEligible": False}
        # A missing/invalid traded price must not erase the dated identity.
        identities.append(identity)
        try:
            prices = {k: float(row[col]) for k, col in
                      [("open", "OpnPric"), ("high", "HghPric"), ("low", "LwPric"), ("close", "ClsPric")]}
            turnover = float(row["TtlTrfVal"])
            if (not all(math.isfinite(v) for v in [*prices.values(), turnover])
                    or min(prices.values()) <= 0 or turnover < 0
                    or not prices["low"] <= min(prices["open"], prices["close"])
                    <= max(prices["open"], prices["close"]) <= prices["high"]):
                raise ValueError("Invalid daily OHLC/value")
            if not row["TtlTradgVol"].isdecimal():
                raise ValueError("Invalid integer daily volume")
        except ValueError as error:
            quarantined.append({"reason": str(error), "sourceRow": row})
            continue
        identity["priceStatus"] = "structurally_valid"
        bars.append({**identity,
                     **prices, "volume": int(row["TtlTradgVol"]), "reportedTradeValue": turnover,
                     })
    if not total:
        raise ValueError("Empty UDiFF file")
    return {"version": VERSION, "tradingDate": trading_date, "observedAt": observed_at,
            "sourceFilename": source_filename, "sourceSha256": hashlib.sha256(payload).hexdigest(),
            "csvSha256": hashlib.sha256(content).hexdigest(), "identities": identities,
            "bars": bars, "quarantined": quarantined,
            "coverage": {"sourceRows": total, "auditedRows": len(bars), "quarantinedRows": len(quarantined),
                         "datedIdentities": len(identities), "uniqueIsins": len({r["isin"] for r in identities}),
                         "pricedUniqueIsins": len({r["isin"] for r in bars}),
                         "seriesCounts": dict(Counter(r["series"] for r in identities)),
                         "instrumentTypeCounts": dict(Counter(r["instrumentType"] for r in identities)),
                         "missingReportedLots": sum(r["reportedBoardLot"] is None for r in identities)},
            "limitations": ["Identities are observed in the dated file, not inferred from today's registry.",
                            "Absence is not delisting, suspension, no trading or complete universe membership proof.",
                            "Instrument type STK is not proof of ordinary stock classification; other instruments remain.",
                            "Reported board lot is retained, not verified as sufficient execution/affordability eligibility.",
                            "Raw daily prices are not corporate-action-adjusted returns; no repairs or fills are inferred.",
                            "Current acquisition is not historical first-seen availability or verified intraday publication."]}
