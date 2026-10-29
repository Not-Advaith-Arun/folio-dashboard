"""Optional offline replay capture for future experiments; no numerical adapter.

Retain the returned hash externally. Caller provenance and supplied prices are
not authenticated. Missing older telemetry stays missing; no replay is invoked.
"""
from contextlib import contextmanager
from datetime import date, datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import secrets
import stat

VERSION = "replay-evidence-1"
MAX_BYTES, MAX_RECORDS, MAX_ROWS, MAX_TOTAL = 16 * 1024 * 1024, 128, 20000, 100000
REASONS = {"not_recorded", "unavailable", "not_applicable", "pending"}
TURNOVER = "gross executed buy+sell notional / initial equity; not halved or annualized"
CONTRACT = set("version experimentId mode protocol producedAt sources code cost expected".split())
RECORD = set("book scenario kind start end initial slippageBps lots decisionsSha256 calendarSha256 pricesSha256 replay missing".split())
REPLAY = set("version initial ending netReturn dailyCloseDrawdown fees orders curve unfilledTargets suppressedAdjustments driftBand lots omittedDividendEntitlements slippageBpsPerSide feeSource feeChecked limitations".split())


def _now():
    return datetime.now(timezone.utc)


def _fields(value, keys):
    if type(value) is not dict or set(value) != set(keys):
        raise ValueError("Missing/unknown evidence fields")


def _name(value):
    if type(value) is not str or not re.fullmatch(r"[A-Za-z0-9_.:-]{1,80}", value):
        raise ValueError("Invalid identifier")
    return value


def _hash(value):
    if type(value) is not str or not re.fullmatch(r"[a-f0-9]{64}", value):
        raise ValueError("Invalid hash")
    return value


def _time(value):
    if type(value) is not str:
        raise ValueError("Aware timestamp required")
    t = datetime.fromisoformat(value)
    if t.tzinfo is None or t.utcoffset() is None:
        raise ValueError("Aware timestamp required")
    return t.astimezone(timezone.utc)


def _day(value):
    if type(value) is not str or date.fromisoformat(value).isoformat() != value:
        raise ValueError("Explicit ISO civil day required")
    return value


def _num(value, *, minimum=None):
    try:
        valid = type(value) in {int, float} and math.isfinite(value)
    except OverflowError:
        valid = False
    if not valid or minimum is not None and value < minimum:
        raise ValueError("Invalid finite number")
    return value


def _integer(value, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError("Whole quantity/count required")
    return value


def _close(a, b):
    _num(a); _num(b)
    if abs(a - b) > 1e-6 + 1e-10 * max(abs(a), abs(b)):
        raise ValueError("Replay accounting conflict")


def _bytes(value):
    nodes = 0
    def safe(v, depth=0):
        nonlocal nodes
        nodes += 1
        if depth > 24 or nodes > 1000000:
            raise ValueError("JSON structural budget exceeded")
        if type(v) is dict:
            for k, child in v.items():
                if type(k) is not str or len(k) > 4096:
                    raise ValueError("Invalid JSON key")
                safe(child, depth + 1)
        elif type(v) is list:
            for child in v:
                safe(child, depth + 1)
        elif type(v) is str:
            if len(v) > 4096:
                raise ValueError("JSON string budget exceeded")
        elif v is not None and type(v) is not bool:
            _num(v)
    safe(value)
    body = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    if len(body) > MAX_BYTES:
        raise ValueError("JSON byte budget exceeded")
    return body


def digest(value):
    """Bounded deterministic local Python JSON hash, not RFC8785."""
    return hashlib.sha256(_bytes(value)).hexdigest()


def _pairs(pairs):
    out = {}
    for k, v in pairs:
        if k in out:
            raise ValueError("Duplicate JSON key")
        out[k] = v
    return out


def _constant(value):
    raise ValueError("Nonfinite JSON token")


def _lots(value):
    if type(value) is not dict or not value or len(value) > 256:
        raise ValueError("Explicit lot mapping required")
    for symbol, lot in value.items():
        _name(symbol); _integer(lot, 1)


def _key(record):
    for k in ["book", "scenario"]:
        _name(record[k])
    if record["kind"] not in {"main", "restart"}:
        raise ValueError("Invalid replay kind")
    return record["book"], record["scenario"], record["kind"]


def _replay(record):
    replay = record["replay"]
    derived = {"orderCount": None, "grossNotional": None, "turnover": None,
               "feeSum": None, "unfilledCount": None, "cashCheck": "unavailable"}
    missing = {"lots"} if record["lots"] is None else set()
    if replay is None:
        return derived, missing | {"replay"}, [], 0
    if type(replay) is not dict:
        raise ValueError("Replay object or explicit missing required")
    missing |= {"replay." + k for k in REPLAY if replay.get(k) is None}
    known = lambda k: k in replay and replay[k] is not None
    for k in ["initial", "ending", "fees", "omittedDividendEntitlements", "driftBand", "slippageBpsPerSide"]:
        if known(k):
            _num(replay[k], minimum=0)
    for k in ["netReturn", "dailyCloseDrawdown"]:
        if known(k):
            _num(replay[k])
    if known("version") and replay["version"] != "rebalance-replay-1":
        raise ValueError("Unsupported numerical replay contract")
    if known("initial"):
        _close(replay["initial"], record["initial"])
    if known("ending") and known("netReturn"):
        _close(replay["netReturn"], replay["ending"] / record["initial"] - 1)
    if known("dailyCloseDrawdown") and not -1 <= replay["dailyCloseDrawdown"] <= 0:
        raise ValueError("Invalid drawdown")
    if known("driftBand") and replay["driftBand"] > .1:
        raise ValueError("Invalid drift convention")
    if known("suppressedAdjustments"):
        _integer(replay["suppressedAdjustments"])
    if known("slippageBpsPerSide"):
        _close(replay["slippageBpsPerSide"], record["slippageBps"])
    if known("lots"):
        _lots(replay["lots"])
        if record["lots"] is not None and replay["lots"] != record["lots"]:
            raise ValueError("Conflicting lots")
    if known("limitations") and (type(replay["limitations"]) is not list or any(type(v) is not str for v in replay["limitations"])):
        raise ValueError("Explicit limitations required")
    lots, start, end, by_date, total = record["lots"], record["start"], record["end"], {}, 0
    def instrument(row):
        _name(row["symbol"])
        if not start <= _day(row["date"]) <= end or lots is not None and row["symbol"] not in lots:
            raise ValueError("Order/unfilled identity outside contract")
    for field in ["orders", "curve", "unfilledTargets"]:
        if known(field):
            if type(replay[field]) is not list or len(replay[field]) > MAX_ROWS:
                raise ValueError("Replay row budget exceeded")
            total += len(replay[field])
    if known("orders"):
        seen, fee_sum, notional, last_date = set(), 0., 0., start
        for order in replay["orders"]:
            _fields(order, {"date", "symbol", "side", "units", "price", "fee"})
            instrument(order)
            if order["side"] not in {"buy", "sell"} or order["date"] < last_date:
                raise ValueError("Invalid order side/chronology")
            last_date = order["date"]
            key = (order["date"], order["symbol"], order["side"])
            if key in seen:
                raise ValueError("Duplicate replay order")
            seen.add(key)
            units = _integer(order["units"], 1)
            if lots is not None and units % lots[order["symbol"]]:
                raise ValueError("Non-lot quantity")
            price, fee = _num(order["price"], minimum=0), _num(order["fee"], minimum=0)
            if price == 0:
                raise ValueError("Positive execution price required")
            value = _num(units * price)
            notional = _num(notional + value); fee_sum = _num(fee_sum + fee)
            by_date.setdefault(order["date"], []).append(order)
        if known("fees"):
            _close(fee_sum, replay["fees"])
        derived.update(orderCount=len(seen), grossNotional=notional, turnover=_num(notional / record["initial"]), feeSum=fee_sum)
    if known("unfilledTargets"):
        for item in replay["unfilledTargets"]:
            _fields(item, {"date", "symbol", "wantedUnits", "filledUnits", "reason"})
            instrument(item)
            wanted, filled = _integer(item["wantedUnits"], 1), _integer(item["filledUnits"])
            if filled >= wanted or type(item["reason"]) is not str or not item["reason"]:
                raise ValueError("Invalid unfilled state")
            if lots is not None and (wanted % lots[item["symbol"]] or filled % lots[item["symbol"]]):
                raise ValueError("Non-lot unfilled quantity")
        derived["unfilledCount"] = len(replay["unfilledTargets"])
    if known("curve"):
        curve = replay["curve"]
        if len(curve) < 2:
            raise ValueError("Incomplete equity curve")
        previous, peak, dd, dates = None, record["initial"], 0., set()
        for i, point in enumerate(curve):
            _fields(point, {"date", "phase", "equity", "cash", "unsettled", "invested"})
            day = _day(point["date"])
            for k in ["equity", "cash", "unsettled", "invested"]:
                _num(point[k], minimum=0)
            _close(point["equity"], point["cash"] + point["unsettled"] + point["invested"])
            if i == 0:
                if day != start or point["phase"] != "before-entry":
                    raise ValueError("Explicit before-entry point required")
                for k, value in {"cash": record["initial"], "equity": record["initial"], "unsettled": 0, "invested": 0}.items():
                    _close(point[k], value)
            else:
                if point["phase"] != "close" or not start <= day <= end or day in dates or i > 1 and day <= previous["date"]:
                    raise ValueError("Invalid close curve chronology")
                dates.add(day)
                if known("orders"):
                    orders = by_date.get(day, [])
                    buys = sum(o["units"] * o["price"] + o["fee"] for o in orders if o["side"] == "buy")
                    sells = sum(o["units"] * o["price"] - o["fee"] for o in orders if o["side"] == "sell")
                    _close(point["cash"], previous["cash"] + previous["unsettled"] - buys)
                    _close(point["unsettled"], sells)
            peak = max(peak, point["equity"]); dd = min(dd, point["equity"] / peak - 1)
            previous = point
        if curve[-1]["date"] != end:
            raise ValueError("Missing final curve endpoint")
        _close(curve[-1]["invested"], 0)
        if known("ending"):
            _close(curve[-1]["equity"], replay["ending"])
        if known("dailyCloseDrawdown"):
            _close(dd, replay["dailyCloseDrawdown"])
        if known("orders"):
            if set(by_date) - dates:
                raise ValueError("Order missing from curve")
            derived["cashCheck"] = "supplied-convention-consistent"
    return derived, missing, sorted(set(replay) - REPLAY), total


def _validate(contract, records, recorded):
    _fields(contract, CONTRACT)
    if contract["version"] != VERSION or contract["mode"] not in {"synthetic", "retrospective"}:
        raise ValueError("Unsupported replay evidence version/mode")
    _name(contract["experimentId"])
    _fields(contract["protocol"], {"sha256", "frozenAt"}); _hash(contract["protocol"]["sha256"])
    produced = _time(contract["producedAt"])
    if not _time(contract["protocol"]["frozenAt"]) <= produced <= recorded:
        raise ValueError("Invalid protocol/production/seal chronology")
    _fields(contract["code"], {"replaySha256", "feeSha256"})
    for v in contract["code"].values():
        _hash(v)
    cost = contract["cost"]
    _fields(cost, {"feeSource", "feeChecked", "settlement", "turnover"})
    if type(cost["feeSource"]) is not str or not cost["feeSource"] or cost["settlement"] != "sale-next-session-no-cashflows" or cost["turnover"] != TURNOVER:
        raise ValueError("Explicit fixed cost/settlement/turnover convention required")
    _day(cost["feeChecked"])
    sources = contract["sources"]
    if type(sources) is not list or not 1 <= len(sources) <= 64:
        raise ValueError("Invalid source budget")
    seen = set()
    for source in sources:
        _fields(source, {"id", "sha256", "firstSeenAt", "receiptSha256", "availability"})
        _name(source["id"]); _hash(source["sha256"])
        if source["id"] in seen:
            raise ValueError("Duplicate source identity")
        seen.add(source["id"])
        if source["availability"] == "known":
            _hash(source["receiptSha256"])
            if _time(source["firstSeenAt"]) > produced:
                raise ValueError("Source first seen after production")
        elif source["availability"] != "unknown" or source["firstSeenAt"] is not None or source["receiptSha256"] is not None:
            raise ValueError("Unknown first-seen must remain explicit")
    expected = contract["expected"]
    if type(expected) is not list or not 1 <= len(expected) <= MAX_RECORDS:
        raise ValueError("Explicit bounded expected books required")
    for item in expected:
        _fields(item, {"book", "scenario", "kind"})
    keys = [_key(v) for v in expected]
    if len(set(keys)) != len(keys) or type(records) is not list or len(records) != len(keys) or sorted(_key(r) for r in records) != sorted(keys):
        raise ValueError("Missing/duplicate/unexpected book scenario")
    summaries, rows = [], 0
    for record in records:
        _fields(record, RECORD)
        if not _day(record["start"]) < _day(record["end"]) <= produced.date().isoformat():
            raise ValueError("Invalid replay range/production date")
        if _num(record["initial"], minimum=0) == 0 or not 0 <= _num(record["slippageBps"]) <= 100:
            raise ValueError("Invalid initial equity/cost scenario")
        for k in ["decisionsSha256", "calendarSha256", "pricesSha256"]:
            _hash(record[k])
        if record["lots"] is not None:
            _lots(record["lots"])
        derived, missing, extras, count = _replay(record); rows += count
        _fields(record["missing"], missing)
        if any(type(v) is not str or v not in REASONS for v in record["missing"].values()):
            raise ValueError("Missing reason required")
        replay = record["replay"]
        if replay is not None:
            for k in ["feeSource", "feeChecked"]:
                if replay.get(k) is not None and replay[k] != cost[k]:
                    raise ValueError("Conflicting cost lineage")
        summaries.append({"book": record["book"], "scenario": record["scenario"], "kind": record["kind"],
            "status": "incomplete" if missing else "complete", "missing": sorted(missing), "unverifiedExtraFields": extras, **derived})
    if rows > MAX_TOTAL:
        raise ValueError("Total replay row budget exceeded")
    return sorted(summaries, key=lambda r: _key(r))


@contextmanager
def _directory(root):
    path = Path(root).absolute()
    if ".." in path.parts or path == Path("/"):
        raise ValueError("Invalid evidence directory")
    fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in path.parts[1:]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd); fd = child
        info = os.fstat(fd)
        if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o700:
            raise ValueError("Owner-private 0700 directory required")
        yield fd
    finally:
        os.close(fd)


def _load(fd, identity, expected=None):
    handle = os.open(_hash(identity) + ".json", os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=fd)
    with os.fdopen(handle, "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o600 or info.st_size > MAX_BYTES:
            raise ValueError("Bounded owner-private regular 0600 file required")
        body = stream.read(MAX_BYTES + 1)
    if len(body) > MAX_BYTES:
        raise ValueError("Byte budget exceeded")
    payload = json.loads(body, object_pairs_hook=_pairs, parse_constant=_constant)
    if _bytes(payload) != body or expected is not None and digest(payload) != _hash(expected):
        raise ValueError("Noncanonical/external hash mismatch")
    _fields(payload, {"version", "recordedAt", "writerSha256", "contract", "records", "summary", "livePromotionAllowed", "tradabilityVerified", "provenanceClaimsVerified"})
    if payload["version"] != VERSION or any(payload[k] is not False for k in ["livePromotionAllowed", "tradabilityVerified", "provenanceClaimsVerified"]):
        raise ValueError("Invalid capability claim")
    _hash(payload["writerSha256"])
    recorded = _time(payload["recordedAt"])
    if recorded > _now() or _identity(payload["contract"]) != identity or _validate(payload["contract"], payload["records"], recorded) != payload["summary"]:
        raise ValueError("Evidence chronology/identity/accounting mismatch")
    return payload


def _identity(contract):
    return digest({"experimentId": contract["experimentId"], "protocol": contract["protocol"]["sha256"],
        "expected": sorted(contract["expected"], key=_key)})


def seal(root, contract, records):
    """Capture supplied replay objects once. Existing private root is required."""
    document = json.loads(_bytes({"contract": contract, "records": records}), object_pairs_hook=_pairs, parse_constant=_constant)
    contract, records = document["contract"], document["records"]
    now = _now(); summary = _validate(contract, records, now); identity = _identity(contract)
    payload = {"version": VERSION, "recordedAt": now.isoformat(), "writerSha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "contract": contract, "records": records, "summary": summary,
        "livePromotionAllowed": False, "tradabilityVerified": False, "provenanceClaimsVerified": False}
    body = _bytes(payload); reused = False
    with _directory(root) as fd:
        temporary = ".pending-" + secrets.token_hex(16)
        handle = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=fd)
        try:
            with os.fdopen(handle, "wb") as stream:
                os.fchmod(stream.fileno(), 0o600); stream.write(body); stream.flush(); os.fsync(stream.fileno())
            try:
                os.link(temporary, identity + ".json", src_dir_fd=fd, dst_dir_fd=fd, follow_symlinks=False)
            except FileExistsError:
                previous = _load(fd, identity)
                if any(previous[k] != payload[k] for k in ["contract", "records", "writerSha256"]):
                    raise ValueError("Immutable replay evidence conflict")
                payload = previous; reused = True
            os.fsync(fd)
        finally:
            os.unlink(temporary, dir_fd=fd)
    return {"identitySha256": identity, "payloadSha256": digest(payload), "recordedAt": payload["recordedAt"], "reused": reused}


def read(root, identity_sha256, payload_sha256, *, as_of):
    """Read only against an external pin and cutoff at/after actual first seal."""
    cutoff = _time(as_of)
    if cutoff > _now():
        raise ValueError("Future read cutoff")
    with _directory(root) as fd:
        payload = _load(fd, identity_sha256, _hash(payload_sha256))
    if max(_time(payload["recordedAt"]), _time(payload["contract"]["producedAt"])) > cutoff:
        raise ValueError("Evidence not sealed at cutoff")
    return payload
