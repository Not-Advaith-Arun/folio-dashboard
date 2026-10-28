"""Optional offline reviewed-calendar bridge; no engine, data or outcome adapter.

Exports are local RESEARCH schedules, never market-occurrence, tradability or
authenticated provenance evidence. A reviewer can make false declarations; this
module checks their consistency and exact pins, not their external truth.
"""
from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import stat
from zoneinfo import ZoneInfo

VERSION = "calendar-evidence-1"
MAX_BYTES, MAX_DAYS, MAX_SOURCES, MAX_SYMBOLS = 2 * 1024 * 1024, 366, 64, 256
FLAGS = {"livePromotionAllowed": False, "tradabilityVerified": False,
         "provenanceClaimsVerified": False, "marketOccurrenceVerified": False}


def _now():
    return datetime.now(timezone.utc)


def _fields(value, keys):
    if type(value) is not dict or set(value) != set(keys.split()):
        raise ValueError("Missing or unknown calendar fields")


def _name(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_.:-]{1,80}", value):
        raise ValueError("Invalid calendar identifier")
    return value


def _hash(value):
    if not isinstance(value, str) or not re.fullmatch(r"[a-f0-9]{64}", value):
        raise ValueError("Invalid external hash")
    return value


def _instant(value):
    if not isinstance(value, str):
        raise ValueError("Timezone-aware calendar timestamp required")
    instant = datetime.fromisoformat(value)
    if instant.tzinfo is None or instant.utcoffset() is None:
        raise ValueError("Timezone-aware calendar timestamp required")
    return instant.astimezone(timezone.utc)


def _day(value):
    if not isinstance(value, str) or date.fromisoformat(value).isoformat() != value:
        raise ValueError("ISO calendar date required")
    return date.fromisoformat(value)


def _bytes(value):
    body = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    if len(body) > MAX_BYTES:
        raise ValueError("Calendar byte budget exceeded")
    return body


def digest(value):
    """Deterministic local Python JSON hash; not an RFC8785 implementation."""
    return hashlib.sha256(_bytes(value)).hexdigest()


def _pairs(items):
    out = {}
    for key, value in items:
        if key in out:
            raise ValueError("Duplicate JSON key")
        out[key] = value
    return out


def _constant(value):
    raise ValueError("Nonfinite JSON value")


@contextmanager
def _directory(root):
    # Open every component relative to an already-open descriptor. No ancestor
    # symlink can redirect a check followed by a later path-based open.
    path = Path(root).absolute()
    if ".." in path.parts or path == Path("/"):
        raise ValueError("Invalid calendar directory")
    fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
    try:
        for component in path.parts[1:]:
            child = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = child
        info = os.fstat(fd)
        if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o700:
            raise ValueError("Owner-private 0700 calendar directory required")
        yield fd
    finally:
        os.close(fd)


def _read(fd, name):
    file = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=fd)
    with os.fdopen(file, "rb") as stream:
        info = os.fstat(stream.fileno())
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                or stat.S_IMODE(info.st_mode) != 0o600 or info.st_size > MAX_BYTES):
            raise ValueError("Bounded owner-private regular calendar file required")
        body = stream.read(MAX_BYTES + 1)
    if len(body) > MAX_BYTES:
        raise ValueError("Calendar byte budget exceeded")
    value = json.loads(body, object_pairs_hook=_pairs, parse_constant=_constant)
    if body != _bytes(value):
        raise ValueError("Noncanonical/tampered calendar bytes")
    return value


def seal(root, identity, document):
    """Atomically publish a caller-selected immutable identity; root must exist.

    Sealing stores declarations only, and does not approve them for export.
    Reuse preserves the first actual seal timestamp and refuses changed content.
    """
    name = _name(identity) + ".json"
    # Bound/deep-copy before publication so concurrent caller mutation is inert.
    document = json.loads(_bytes(document), object_pairs_hook=_pairs, parse_constant=_constant)
    payload = {"version": VERSION, "recordedAt": _now().isoformat(), "document": document}
    body = _bytes(payload)
    temporary = ".pending-" + secrets.token_hex(16)
    with _directory(root) as fd:
        file = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=fd)
        try:
            with os.fdopen(file, "wb") as stream:
                stream.write(body)
                stream.flush()
                os.fsync(stream.fileno())
            try:
                os.link(temporary, name, src_dir_fd=fd, dst_dir_fd=fd, follow_symlinks=False)
            except FileExistsError:
                previous = _read(fd, name)
                _fields(previous, "version recordedAt document")
                if previous["version"] != VERSION or _bytes(previous["document"]) != _bytes(document):
                    raise ValueError("Immutable calendar conflict")
                if _instant(previous["recordedAt"]) > _now():
                    raise ValueError("Future calendar seal")
                payload = previous
            os.fsync(fd)
        finally:
            os.unlink(temporary, dir_fd=fd)
    return {"identity": identity, "payloadSha256": digest(payload), "recordedAt": payload["recordedAt"]}


def read(root, identity, expected_hash, *, as_of):
    cutoff = _instant(as_of)
    if cutoff > _now():
        raise ValueError("Future calendar read cutoff")
    _hash(expected_hash)
    with _directory(root) as fd:
        payload = _read(fd, _name(identity) + ".json")
    if digest(payload) != expected_hash:
        raise ValueError("Calendar external hash mismatch")
    _fields(payload, "version recordedAt document")
    if payload["version"] != VERSION or _instant(payload["recordedAt"]) > cutoff:
        raise ValueError("Calendar not sealed at cutoff")
    return payload


def _refs(value, sources, *, required=True):
    if (type(value) is not list or any(not isinstance(key, str) for key in value)
            or len(value) != len(set(value)) or any(key not in sources for key in value)
            or required and not value):
        raise ValueError("Missing/duplicate calendar source references")


def _minute(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-2][0-9]:[0-5][0-9]", value):
        raise ValueError("HH:MM phase required")
    hour, minute = map(int, value.split(":"))
    if hour >= 24:
        raise ValueError("Invalid phase time")
    return hour * 60 + minute


def _bundle(bundle, recorded):
    _fields(bundle, "version calendarId revision mode timezone coverageStart coverageEnd informationCutoffAt protocolSha256 sources days")
    if bundle["version"] != VERSION or bundle["mode"] not in {"research", "synthetic"} or bundle["timezone"] != "Asia/Kolkata":
        raise ValueError("Unsupported calendar version/mode/timezone; provisional facts cannot export")
    _name(bundle["calendarId"])
    _name(bundle["revision"])
    _hash(bundle["protocolSha256"])
    cutoff = _instant(bundle["informationCutoffAt"])
    if cutoff > recorded:
        raise ValueError("Future calendar information cutoff")
    first, last = _day(bundle["coverageStart"]), _day(bundle["coverageEnd"])
    count = (last - first).days + 1
    if not 1 <= count <= MAX_DAYS:
        raise ValueError("Calendar coverage budget")
    if type(bundle["sources"]) is not list or not 1 <= len(bundle["sources"]) <= MAX_SOURCES:
        raise ValueError("Calendar source budget")
    sources = {}
    for source in bundle["sources"]:
        _fields(source, "id url kind contentSha256 receiptSha256 issueDate receivedAt revisionStatus")
        key = _name(source["id"])
        if key in sources:
            raise ValueError("Duplicate source identity")
        if (not isinstance(source["url"], str) or not source["url"].startswith("https://")
                or source["kind"] not in {"original", "normalized", "excerpt", "synthetic"}
                or source["revisionStatus"] not in {"reviewed", "unknown", "conflict"}):
            raise ValueError("Invalid source provenance")
        _hash(source["contentSha256"])
        _hash(source["receiptSha256"])
        received = _instant(source["receivedAt"])
        if _day(source["issueDate"]) > received.astimezone(ZoneInfo(bundle["timezone"])).date() or received > cutoff:
            raise ValueError("Source received after information cutoff")
        sources[key] = source
    days = bundle["days"]
    if (type(days) is not list or len(days) != count or any(type(d) is not dict for d in days)
            or [d.get("date") for d in days] != [(first + timedelta(days=i)).isoformat() for i in range(count)]):
        raise ValueError("Explicit complete ordered civil-day coverage required")
    for day in days:
        _fields(day, "date cashStatus cashSourceIds t0Status t0SourceIds settlementStatus settlementSourceIds phases eligibility")
        if day["cashStatus"] not in {"regular", "special", "closed", "unknown", "conflict"}:
            raise ValueError("Invalid normal-cash state")
        _refs(day["cashSourceIds"], sources, required=day["cashStatus"] not in {"unknown", "conflict"})
        for axis in ["t0", "settlement"]:
            if day[axis + "Status"] not in {"open", "closed", "unknown", "conflict"}:
                raise ValueError("Invalid separate settlement/T0 state")
            _refs(day[axis + "SourceIds"], sources, required=day[axis + "Status"] not in {"unknown", "conflict"})
        phases = day["phases"]
        if type(phases) is not list or len(phases) > 32:
            raise ValueError("Phase budget")
        identities = set()
        for phase in phases:
            _fields(phase, "class kind start end sourceIds")
            if phase["class"] not in {"ALL_CASH", "CAS", "NON_CAS", "IPO_RELISTED", "ILLIQUID"} or phase["kind"] not in {"continuous", "auction", "preopen", "transition", "postclose"}:
                raise ValueError("Unknown phase/class")
            identity = tuple(phase[k] for k in ["class", "kind", "start", "end"])
            if identity in identities or _minute(phase["start"]) >= _minute(phase["end"]):
                raise ValueError("Duplicate/invalid phase")
            identities.add(identity)
            _refs(phase["sourceIds"], sources)
        for session_class in {p["class"] for p in phases}:
            spans = sorted((_minute(p["start"]), _minute(p["end"])) for p in phases
                           if p["class"] == session_class and p["kind"] == "continuous")
            if any(a[1] > b[0] for a, b in zip(spans, spans[1:])):
                raise ValueError("Conflicting continuous phase intervals")
        if day["cashStatus"] == "closed" and phases:
            raise ValueError("Closed normal cash day has contradictory phases")
        if type(day["eligibility"]) is not dict or len(day["eligibility"]) > MAX_SYMBOLS:
            raise ValueError("Eligibility budget")
        for symbol, entry in day["eligibility"].items():
            _name(symbol)
            _fields(entry, "class sourceIds")
            if entry["class"] not in {"CAS", "NON_CAS", "unknown"}:
                raise ValueError("Invalid eligibility class")
            _refs(entry["sourceIds"], sources, required=entry["class"] != "unknown")
    return sources


def export_calendar(root, bundle_id, bundle_hash, review_id, review_hash, *, as_of, target="daily", symbols=()):
    """Explicitly compile a pinned reviewed RESEARCH schedule, without publishing.

    A separate review is mandatory; receipt times never become historical data
    availability. Unknown settlement/T0 states remain visible, not cash closures.
    """
    payload = read(root, bundle_id, bundle_hash, as_of=as_of)
    review_payload = read(root, review_id, review_hash, as_of=as_of)
    bundle, review = payload["document"], review_payload["document"]
    sources = _bundle(bundle, _instant(payload["recordedAt"]))
    _fields(review, "bundleSha256 reviewedAt reviewer reviewEvidenceSha256 coverageComplete exceptionsComplete cashRevisionsResolved sourceVisualReviewComplete exceptionsReviewedThrough")
    if review["bundleSha256"] != bundle_hash:
        raise ValueError("Review bound to different bundle/revision")
    _name(review["reviewer"])
    _hash(review["reviewEvidenceSha256"])
    checked = _instant(review["reviewedAt"])
    if not (_instant(payload["recordedAt"]) <= checked <= _instant(review_payload["recordedAt"])):
        raise ValueError("Review chronology")
    if not (_instant(bundle["informationCutoffAt"]) <= _instant(review["exceptionsReviewedThrough"]) <= checked):
        raise ValueError("Incomplete exception review boundary")
    if any(review[k] is not True for k in ["coverageComplete", "exceptionsComplete", "cashRevisionsResolved", "sourceVisualReviewComplete"]):
        raise ValueError("Incomplete reviewed calendar")
    expected_kind = "synthetic" if bundle["mode"] == "synthetic" else "original"
    if any(s["kind"] != expected_kind or s["revisionStatus"] != "reviewed" for s in sources.values()):
        raise ValueError("Unqualified original/source revision evidence")
    if target not in {"daily", "intraday_pilot"}:
        raise ValueError("Unsupported calendar export target")
    if target == "intraday_pilot" and (not symbols or len(symbols) > MAX_SYMBOLS or len(set(symbols)) != len(symbols)):
        raise ValueError("Explicit unique intraday instrument universe required")
    for symbol in symbols:
        _name(symbol)
    sessions, bars = [], []
    for day in bundle["days"]:
        status = day["cashStatus"]
        if status in {"unknown", "conflict"}:
            raise ValueError("Unknown/conflicting cash session stays pending")
        if status == "closed":
            continue
        continuous = [p for p in day["phases"] if p["kind"] == "continuous"
                      and p["class"] in {"ALL_CASH", "CAS", "NON_CAS"}]
        if not continuous:
            raise ValueError("Missing continuous cash-session evidence")
        sessions.append(day["date"])
        if target == "intraday_pilot":
            if status != "regular":
                raise ValueError("Special session incompatible with fixed regular-session pilot")
            for symbol in symbols:
                entry = day["eligibility"].get(symbol)
                if entry is None or entry["class"] == "unknown":
                    raise ValueError("Unknown dated CAS/instrument eligibility")
                candidates = [p for p in continuous if p["class"] in {"ALL_CASH", entry["class"]}]
                if len(candidates) != 1 or _minute(candidates[0]["start"]) != 555 or _minute(candidates[0]["end"]) < 915:
                    raise ValueError("Ambiguous/incomplete regular-session phase coverage")
            start = datetime.fromisoformat(day["date"] + "T09:15:00").replace(tzinfo=ZoneInfo("Asia/Kolkata"))
            bars.extend((start + timedelta(minutes=5*i)).isoformat() for i in range(72))
    if not sessions:
        raise ValueError("No explicit cash sessions in reviewed coverage")
    return {"version": VERSION, "source": "Pinned local reviewed RESEARCH schedule; external truth unauthenticated",
            "sourceSha256": digest({"bundle": bundle_hash, "review": review_hash}),
            "coverageStart": bundle["coverageStart"], "coverageEnd": bundle["coverageEnd"],
            "sessions": sessions, "barStarts": bars, "sessionTimezone": bundle["timezone"],
            "target": target, "mode": bundle["mode"],
            "bundleSha256": bundle_hash, "reviewSha256": review_hash,
            "informationCutoffAt": bundle["informationCutoffAt"],
            "historicalAvailability": "unknown; receipt/review/seal times are local evidence only",
            "days": bundle["days"], **FLAGS}
