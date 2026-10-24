"""Optional local evidence capture for newly declared experiments; no fitting.

Records are private, bounded and immutable per experiment/protocol/model/signal.
Callers must preserve the returned payload hash outside this directory for reads.
Local seals and caller-supplied provenance are not trusted external timestamps.
This POSIX writer does not integrate with an engine or resolve paper outcomes.
"""
from __future__ import annotations

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
from zoneinfo import ZoneInfo

VERSION = "calibration-evidence-1"
MAX_ROWS, MAX_BYTES, MAX_SOURCES = 20000, 16 * 1024 * 1024, 64
REASONS = {"not_recorded", "unavailable", "not_applicable", "pending"}
PARTITIONS = {"train", "calibration", "forecast"}
CONTRACT = {"version", "experimentId", "mode", "sessionTimezone", "signalAt", "issuedAt",
            "protocol", "model", "target", "sources", "expectedRows"}
ROW = {"partition", "symbol", "session", "featureAt", "entryAt", "labelEndAt",
       "featureObservedAt", "labelObservedAt", "raw", "calibrated", "label",
       "cost", "risk", "sourceIds", "missing"}
NULLABLE = {"featureObservedAt", "labelObservedAt", "raw", "calibrated", "label", "cost", "risk"}


def _now():
    return datetime.now(timezone.utc)


def _bytes(value):
    # Deterministic local Python JSON encoding, not an RFC8785 implementation.
    return json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()


def _digest(value):
    return hashlib.sha256(_bytes(value)).hexdigest()


def _fields(value, fields):
    if type(value) is not dict or set(value) != fields:
        raise ValueError("Missing or unknown evidence fields")


def _hash(value):
    if not isinstance(value, str) or not re.fullmatch(r"[a-f0-9]{64}", value):
        raise ValueError("Invalid evidence hash")
    return value


def _name(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_.:-]{1,80}", value):
        raise ValueError("Invalid evidence identifier")
    return value


def _time(value):
    if not isinstance(value, str):
        raise ValueError("Timezone-aware timestamp required")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError("Invalid evidence timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("Timezone-aware timestamp required")
    return parsed.astimezone(timezone.utc)


def _number(value, *, nonnegative=False):
    if type(value) not in {int, float} or not math.isfinite(value) or nonnegative and value < 0:
        raise ValueError("Finite evidence number required")
    return float(value)


def _validate(contract, records, recorded):
    _fields(contract, CONTRACT)
    if contract["version"] != VERSION or not isinstance(contract["mode"], str) or contract["mode"] not in {"retrospective", "prospective", "synthetic"}:
        raise ValueError("Unsupported evidence mode/version")
    _name(contract["experimentId"])
    try:
        zone = ZoneInfo(contract["sessionTimezone"])
    except (ValueError, TypeError, KeyError) as exc:
        raise ValueError("Invalid session timezone") from exc
    signal, issued = _time(contract["signalAt"]), _time(contract["issuedAt"])
    _fields(contract["protocol"], {"sha256", "frozenAt"})
    _fields(contract["model"], {"sha256", "fittedAt"})
    _hash(contract["protocol"]["sha256"])
    _hash(contract["model"]["sha256"])
    frozen, fitted = _time(contract["protocol"]["frozenAt"]), _time(contract["model"]["fittedAt"])
    if not (frozen <= fitted <= issued <= recorded and signal <= issued):
        raise ValueError("Invalid declaration/model/issuance chronology")
    target = contract["target"]
    _fields(target, {"name", "functional", "score", "quantile", "unit"})
    _name(target["name"])
    scores = {"mean": {"squared_error"}, "median": {"absolute_error"},
              "quantile": {"pinball"}, "point_proxy": {"squared_error", "absolute_error"}}
    if (target["unit"] != "return_fraction" or not isinstance(target["functional"], str)
            or not isinstance(target["score"], str) or target["score"] not in scores.get(target["functional"], set())):
        raise ValueError("Target functional and score do not match")
    if target["functional"] == "quantile":
        if not 0 < _number(target["quantile"]) < 1:
            raise ValueError("Invalid target quantile")
    elif target["quantile"] is not None:
        raise ValueError("Unexpected target quantile")
    sources = contract["sources"]
    if type(sources) is not list or not 1 <= len(sources) <= MAX_SOURCES:
        raise ValueError("Source budget exceeded or missing sources")
    source_ids, normalized_sources = set(), []
    prospective = contract["mode"] == "prospective"
    for source in sources:
        _fields(source, {"id", "sha256", "receiptSha256", "firstSeenAt", "availability"})
        name = _name(source["id"])
        _hash(source["sha256"])
        if name in source_ids:
            raise ValueError("Duplicate source")
        source_ids.add(name)
        normalized = dict(source)
        if source["availability"] == "known":
            first = _time(source["firstSeenAt"])
            _hash(source["receiptSha256"])
            if first > recorded or prospective and first > signal:
                raise ValueError("Source first seen after cutoff")
            normalized["firstSeenAt"] = first.isoformat()
        elif source["availability"] != "unknown" or source["firstSeenAt"] is not None or source["receiptSha256"] is not None:
            raise ValueError("Unknown source availability must be explicit")
        elif prospective:
            raise ValueError("Prospective source availability unknown")
        normalized_sources.append(normalized)
    if prospective and frozen > signal:
        raise ValueError("Prospective protocol declared after signal")
    expected = contract["expectedRows"]
    _fields(expected, PARTITIONS)
    if any(type(n) is not int or n < 0 for n in expected.values()) or not expected["train"] or not expected["calibration"]:
        raise ValueError("Explicit positive training/calibration counts required")
    if type(records) is not list or not 1 <= len(records) <= MAX_ROWS or sum(expected.values()) != len(records):
        raise ValueError("Row budget/count mismatch")
    rows, seen, dates = [], set(), {p: set() for p in PARTITIONS}
    for source in records:
        _fields(source, ROW)
        row = dict(source)
        part = row["partition"]
        if not isinstance(part, str) or part not in PARTITIONS:
            raise ValueError("Invalid partition")
        _name(row["symbol"])
        feature, entry, end = (_time(row[k]) for k in ["featureAt", "entryAt", "labelEndAt"])
        if row["session"] != feature.astimezone(zone).date().isoformat():
            raise ValueError("Session/date-group mismatch")
        date.fromisoformat(row["session"])
        key = (part, row["session"], row["symbol"])
        if key in seen:
            raise ValueError("Duplicate evidence row")
        seen.add(key)
        dates[part].add(row["session"])
        if not feature < entry < end or feature > signal or part == "forecast" and feature != signal:
            raise ValueError("Invalid feature/label/signal chronology")
        ids = row["sourceIds"]
        if type(ids) is not list or not ids or any(type(i) is not str or i not in source_ids for i in ids) or len(set(ids)) != len(ids):
            raise ValueError("Invalid row source membership")
        _fields(row["missing"], {k for k in NULLABLE if row[k] is None})
        if any(not isinstance(reason, str) or reason not in REASONS for reason in row["missing"].values()):
            raise ValueError("Invalid missing-evidence reason")
        for k in ["raw", "calibrated", "label", "cost", "risk"]:
            if row[k] is not None:
                row[k] = _number(row[k], nonnegative=k in {"cost", "risk"})
        for k in ["featureObservedAt", "labelObservedAt"]:
            if row[k] is not None:
                observed = _time(row[k])
                if observed > recorded or observed > (issued if part == "forecast" else fitted):
                    raise ValueError("Observation unavailable to model/issuance")
                if k == "featureObservedAt" and prospective and observed > feature:
                    raise ValueError("Feature first seen after historical cutoff")
                if k == "labelObservedAt" and observed < end:
                    raise ValueError("Outcome observation precedes label end")
                row[k] = observed.isoformat()
            elif prospective and (part != "forecast" or k == "featureObservedAt"):
                raise ValueError("Prospective row availability unknown")
        if part == "forecast":
            if row["label"] is not None or row["labelObservedAt"] is not None:
                raise ValueError("Forecast outcome must remain separate/pending")
        elif row["label"] is None or end >= signal or end > fitted:
            raise ValueError("Training/calibration outcome not mature")
        row.update(featureAt=feature.isoformat(), entryAt=entry.isoformat(), labelEndAt=end.isoformat(), sourceIds=sorted(ids))
        row["costMargin"] = None if row["calibrated"] is None or row["cost"] is None else _number(row["calibrated"] - row["cost"])
        row["riskMargin"] = None if row["costMargin"] is None or row["risk"] is None else _number(row["costMargin"] - row["risk"])
        rows.append(row)
    if any(dates[a] & dates[b] for a, b in [("train", "calibration"), ("train", "forecast"), ("calibration", "forecast")]):
        raise ValueError("Whole-date partition overlap")
    if any(sum(r["partition"] == p for r in rows) != expected[p] for p in PARTITIONS):
        raise ValueError("Partition membership count mismatch")
    calibration_start = min(_time(r["featureAt"]) for r in rows if r["partition"] == "calibration")
    for row in rows:
        if row["partition"] == "forecast":
            continue
        cutoff = calibration_start if row["partition"] == "train" else signal
        if _time(row["labelEndAt"]) >= cutoff:
            raise ValueError("Purged label overlaps next partition")
        if prospective and _time(row["labelObservedAt"]) >= cutoff:
            raise ValueError("Label first seen after partition cutoff")
    normalized_contract = {**contract, "signalAt": signal.isoformat(), "issuedAt": issued.isoformat(),
        "protocol": {**contract["protocol"], "frozenAt": frozen.isoformat()},
        "model": {**contract["model"], "fittedAt": fitted.isoformat()},
        "sources": sorted(normalized_sources, key=lambda s: s["id"])}
    rows.sort(key=lambda r: (r["partition"], r["session"], r["symbol"]))
    summary = {p: {"rows": expected[p], "dates": len(dates[p]),
        "membershipSha256": _digest([(r["symbol"], r["session"]) for r in rows if r["partition"] == p]),
        "missingPredictions": sum(r["raw"] is None or r["calibrated"] is None for r in rows if r["partition"] == p)} for p in sorted(PARTITIONS)}
    return normalized_contract, rows, summary


def _identity(contract):
    return _digest({"experimentId": contract["experimentId"], "signalAt": contract["signalAt"],
        "protocolSha256": contract["protocol"]["sha256"], "modelSha256": contract["model"]["sha256"]})


@contextmanager
def _directory(root, *, create=False):
    root = Path(root)
    if create:
        root.mkdir(mode=0o700, exist_ok=True)
    fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        info = os.fstat(fd)
        if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o700:
            raise ValueError("Evidence directory must be owner-private 0700")
        yield fd
    finally:
        os.close(fd)


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


def _load(fd, identity, expected=None):
    handle = os.open(_hash(identity) + ".json", os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=fd)
    with os.fdopen(handle, "rb") as f:
        info = os.fstat(f.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o600 or info.st_size > MAX_BYTES:
            raise ValueError("Evidence file must be bounded owner-private regular 0600")
        data = f.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise ValueError("Evidence byte budget exceeded")
    wrapper = json.loads(data, object_pairs_hook=_unique)
    _fields(wrapper, {"payload", "sha256"})
    payload = wrapper["payload"]
    actual = _digest(payload)
    if wrapper["sha256"] != actual or expected is not None and actual != _hash(expected):
        raise ValueError("Evidence seal/hash mismatch")
    _fields(payload, {"version", "recordedAt", "writerSha256", "contract", "rows", "summary",
                      "livePromotionAllowed", "provenanceClaimsVerified"})
    if payload["version"] != VERSION or payload["livePromotionAllowed"] is not False or payload["provenanceClaimsVerified"] is not False:
        raise ValueError("Invalid evidence capability claim")
    _hash(payload["writerSha256"])
    recorded = _time(payload["recordedAt"])
    if recorded > _now():
        raise ValueError("Record timestamp is in the future")
    original_rows = payload["rows"]
    if type(original_rows) is not list or len(original_rows) > MAX_ROWS:
        raise ValueError("Invalid evidence row budget")
    for row in original_rows:
        _fields(row, ROW | {"costMargin", "riskMargin"})
    contract, rows, summary = _validate(payload["contract"], [{k: r[k] for k in ROW} for r in original_rows], recorded)
    if contract != payload["contract"] or rows != original_rows or summary != payload["summary"] or _identity(contract) != identity:
        raise ValueError("Evidence identity/normalization mismatch")
    return payload, actual


def _request(payload):
    return _digest({k: payload[k] for k in ["contract", "rows", "writerSha256"]})


def write_evidence(root, contract, records):
    """Seal future-experiment evidence once; identical retries retain first time.

    Do not supply credentials or account data. Hash fields are provenance claims,
    not verification of provider truth. No model is fitted/loaded or promoted.
    """
    now = _now()
    normalized, rows, summary = _validate(contract, records, now)
    identity = _identity(normalized)
    payload = {"version": VERSION, "recordedAt": now.isoformat(),
        "writerSha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "contract": normalized, "rows": rows, "summary": summary,
        "livePromotionAllowed": False, "provenanceClaimsVerified": False}
    digest = _digest(payload)
    encoded = _bytes({"sha256": digest, "payload": payload})
    if len(encoded) > MAX_BYTES:
        raise ValueError("Evidence byte budget exceeded")
    reused = False
    with _directory(root, create=True) as fd:
        def previous():
            saved, previous_hash = _load(fd, identity)
            if _request(saved) != _request(payload):
                raise ValueError("Immutable evidence conflict")
            return saved, previous_hash
        try:
            payload, digest = previous()
            reused = True
        except FileNotFoundError:
            temporary = ".pending-" + secrets.token_hex(16)
            handle = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=fd)
            try:
                with os.fdopen(handle, "wb") as f:
                    os.fchmod(f.fileno(), 0o600)
                    f.write(encoded)
                    f.flush()
                    os.fsync(f.fileno())
                try:
                    os.link(temporary, identity + ".json", src_dir_fd=fd, dst_dir_fd=fd, follow_symlinks=False)
                    os.fsync(fd)
                except FileExistsError:
                    payload, digest = previous()
                    reused = True
            finally:
                os.unlink(temporary, dir_fd=fd)
    return {"identitySha256": identity, "payloadSha256": digest,
            "recordedAt": payload["recordedAt"], "reused": reused}


def read_evidence(root, identity_sha256, payload_sha256, *, as_of):
    """Read against an externally retained hash and actual record-time cutoff."""
    cutoff = _time(as_of)
    if cutoff > _now():
        raise ValueError("Read cutoff cannot be in the future")
    _hash(payload_sha256)
    with _directory(root) as fd:
        payload, _ = _load(fd, identity_sha256, payload_sha256)
    if max(_time(payload["recordedAt"]), _time(payload["contract"]["issuedAt"])) > cutoff:
        raise ValueError("Evidence was not recorded by requested cutoff")
    return payload
