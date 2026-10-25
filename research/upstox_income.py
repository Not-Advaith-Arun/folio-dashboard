"""Two fixed free income-statement GETs and private first-seen evidence.

No historical availability, derived features, model fitting or broker methods.
python -m research.upstox_income /private/path/PLAN.json
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import secrets
import signal
import stat
import time

import requests

VERSION = "upstox-income-snapshot-1"
INSTRUMENTS = [{"symbol": "RELIANCE", "isin": "INE002A01018"},
               {"symbol": "INFY", "isin": "INE009A01021"}]
QUERY = {"type": "consolidated", "time_period": "quarterly", "fs": "false"}
LIMITS = {"maximumRequests": 2, "maximumResponseBytes": 262144,
          "maximumBytes": 524288, "timeoutSeconds": 15, "maximumElapsedSeconds": 60}
TOKEN_PATH = Path.home() / ".config/folio/upstox-analytics-token"
FILE_CAP = 2097152
GATES = {k: False for k in ["rawPriceVerified", "volumeUnitsVerified",
    "productionCalendarVerified", "bonusTradabilityVerified",
    "historicalAvailabilityVerified", "trainingApproved", "livePromotionAllowed"]}


class SourceError(ValueError):
    """Only fixed, credential-free error messages may escape this module."""


def now():
    return datetime.now(timezone.utc).isoformat()


def timestamp(value):
    v = datetime.fromisoformat(value)
    if v.tzinfo is None or v.utcoffset() is None:
        raise SourceError("Timezone-aware evidence time required")
    return v.astimezone(timezone.utc)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def encoded(value):
    return json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()


def unique(pairs):
    result = {}
    for k, v in pairs:
        if k in result:
            raise SourceError("Duplicate JSON object key")
        result[k] = v
    return result


def decoded(body):
    try:
        return json.loads(body, object_pairs_hook=unique,
                          parse_constant=lambda _: (_ for _ in ()).throw(SourceError("Nonfinite JSON")))
    except (ValueError, UnicodeError, RecursionError):
        raise SourceError("Malformed JSON evidence") from None


@contextmanager
def directory(path, *, create=False):
    """Walk every component without following symlinks; hold the leaf dirfd."""
    path = Path(os.path.abspath(path))
    fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
    try:
        for index, part in enumerate(path.parts[1:]):
            if create and index == len(path.parts) - 2:
                try:
                    os.mkdir(part, 0o700, dir_fd=fd)
                except FileExistsError:
                    pass
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = child
        info = os.fstat(fd)
        if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o700:
            raise SourceError("Evidence directory must be owner-private 0700")
        yield fd
    finally:
        os.close(fd)


def read(fd, name, cap=FILE_CAP):
    if Path(name).name != name or name in {".", ".."}:
        raise SourceError("Unsafe evidence filename")
    handle = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=fd)
    with os.fdopen(handle, "rb") as f:
        info = os.fstat(f.fileno())
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                or stat.S_IMODE(info.st_mode) != 0o600 or info.st_size > cap):
            raise SourceError("Evidence must be bounded owner-private regular 0600")
        body = f.read(cap + 1)
    if len(body) > cap:
        raise SourceError("Evidence byte budget exceeded")
    return body


def publish(fd, name, body):
    if Path(name).name != name or len(body) > FILE_CAP:
        raise SourceError("Unsafe or oversized evidence publication")
    temporary = ".pending-" + secrets.token_hex(16)
    handle = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                     0o600, dir_fd=fd)
    try:
        with os.fdopen(handle, "wb") as f:
            f.write(body)
            f.flush()
            os.fsync(f.fileno())
        os.link(temporary, name, src_dir_fd=fd, dst_dir_fd=fd, follow_symlinks=False)
        os.fsync(fd)
    finally:
        os.unlink(temporary, dir_fd=fd)


def credential(path):
    p = Path(os.path.abspath(path))
    if Path(__file__).resolve().parents[1] in p.parents:
        raise SourceError("Credential must be outside repository")
    with directory(p.parent) as fd:
        value = read(fd, p.name, 16384).decode("ascii").strip()
    if not re.fullmatch(r"[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+", value):
        raise SourceError("Invalid private credential format")
    return value


def url_for(instrument):
    return ("https://api.upstox.com/v2/fundamentals/" + instrument["isin"]
            + "/income-statement?type=consolidated&time_period=quarterly&fs=false")


@contextmanager
def wall_limit(seconds):
    # Socket read timeouts alone do not bound a trickling response's wall time.
    def expired(*_):
        raise SourceError("Income source wall-time budget exhausted")
    previous = signal.signal(signal.SIGALRM, expired)
    timer = signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, *timer)
        signal.signal(signal.SIGALRM, previous)


class ReadOnlyIncome:
    def __init__(self, token, plan, transport=None):
        if any(type(plan.get(k)) is not int or not 0 < plan[k] <= v for k, v in LIMITS.items()):
            raise SourceError("Invalid income request budgets")
        self._token, self.plan = token, plan
        self.transport = transport if transport is not None else requests.Session()
        self.transport.trust_env = False
        self.allowed = {url_for(i) for i in INSTRUMENTS}
        self.seen, self.calls, self.bytes, self.last_status, self.halted = set(), 0, 0, None, False
        self.deadline = time.monotonic() + plan["maximumElapsedSeconds"]

    def get(self, url):
        if url not in self.allowed:
            raise SourceError("Endpoint outside exact income allowlist")
        remaining = self.deadline - time.monotonic()
        if self.halted or url in self.seen or self.calls >= self.plan["maximumRequests"] or remaining <= 0:
            raise SourceError("Income request/time budget exhausted or client stopped")
        self.seen.add(url)
        self.calls += 1
        self.last_status = None
        requested = now()
        try:
            with wall_limit(min(remaining, self.plan["timeoutSeconds"])):
                with self.transport.get(url, headers={"Accept": "application/json",
                        "Accept-Encoding": "identity", "Authorization": "Bearer " + self._token},
                        stream=True, allow_redirects=False,
                        timeout=min(remaining, self.plan["timeoutSeconds"]), verify=True) as response:
                    self.last_status = int(response.status_code)
                    chunks, size = [], 0
                    # One-byte yielded chunks let even reduced test/continuation
                    # budgets stop at the cap, without reading a lookahead block.
                    for chunk in response.iter_content(1):
                        size += len(chunk)
                        self.bytes += len(chunk)
                        if size >= self.plan["maximumResponseBytes"] or self.bytes >= self.plan["maximumBytes"]:
                            raise SourceError("Income source byte budget exceeded")
                        chunks.append(chunk)
                    body = b"".join(chunks)
                    # Detect both literal and JSON-escaped credential echoes before any persistence.
                    text = body.decode("utf-8", errors="replace")
                    echoed = self._token in text
                    # Scan strings even in malformed/duplicate-key JSON: a normal
                    # object parser may discard a credential-bearing duplicate.
                    for match in re.finditer(r'"(?:[^"\\]|\\.)*"', text):
                        try:
                            echoed |= self._token in json.loads(match.group())
                        except ValueError:
                            pass
                    if echoed:
                        raise SourceError("Credential echo rejected before evidence publication")
                    self.halted = self.last_status != 200
                    return body, {"url": url, "requestedAt": requested, "receivedAt": now(),
                        "httpStatus": self.last_status, "responseBytes": len(body),
                        "contentType": "application/json" if response.headers.get("Content-Type", "").split(";")[0].strip() == "application/json" else "other_or_missing"}
        except Exception:
            self.halted = True
            # Never expose server-controlled exception strings, URLs or headers.
            raise SourceError("Income source stopped; no retry or credential logging") from None


def normalize(body, receipt):
    payload = decoded(body)
    data = payload.get("data") if type(payload) is dict else None
    if (type(payload) is not dict or payload.get("status") != "success" or type(data) is not dict
            or data.get("type") != "consolidated" or data.get("time_period") != "quarterly"
            or data.get("units_in") != "crore" or type(data.get("income_statement")) is not list
            or len(data["income_statement"]) > 64):
        raise SourceError("Income envelope/type/frequency/unit mismatch; raw response quarantined")
    rows, identities = [], {}
    for group_index, group in enumerate(data["income_statement"]):
        if type(group) is not dict or not isinstance(group.get("category"), str) or type(group.get("history")) is not list:
            raise SourceError("Malformed income category/history; raw response quarantined")
        for index, source in enumerate(group["history"]):
            if type(source) is not dict:
                raise SourceError("Malformed income row; raw response quarantined")
            period = source.get("period")
            value = source.get("value")
            status = ("missing" if "value" not in source else "null" if value is None
                      else "finite" if type(value) in {int, float} and math.isfinite(value) else "invalid")
            identity = (group["category"], period if type(period) is str else None)
            row = {"sourceGroupIndex": group_index, "sourceRowIndex": index,
                "category": group["category"], "periodLabel": period,
                "periodStatus": "unparsed_claim" if isinstance(period, str) and period.strip() else "missing_or_invalid",
                "sourceRow": source, "valueStatus": status, "duplicateStatus": "unique",
                "firstSeenAt": receipt["receivedAt"], "issuerPublicationAt": None,
                "providerAvailableAt": None, "revisionAt": None, "restatementStatus": "unknown",
                "periodStartAt": None, "periodEndAt": None, "quarterVsYTD": "unknown", **GATES}
            rows.append(row)
            identities.setdefault(identity, []).append(row)
            if len(rows) > 4096:
                raise SourceError("Income row budget exceeded; raw response quarantined")
    for duplicates in identities.values():
        if len(duplicates) > 1:
            distinct = {encoded(row["sourceRow"]) for row in duplicates}
            for row in duplicates:
                row["duplicateStatus"] = "conflicting" if len(distinct) > 1 else "duplicate"
    present = {g["category"] for g in data["income_statement"]}
    return {"rows": rows, "missingCategories": sorted({"revenue", "operating_profit", "net_profit"} - present),
        "unknownCategories": sorted(present - {"revenue", "operating_profit", "net_profit"}),
        "unexpectedFullStatement": "full_statement" in data,
        "originalRetainsAllUnknownFields": True, "historicalAvailability": "unqualified", **GATES}


def snapshot(root, instrument, body, receipt, lineage, *, previous=None):
    """Seal one content version; identical reuse preserves its actual first receipt.

    A previous link is explicit and offline: this function never polls a source.
    Changes are observations, not proof of an issuer restatement or its date.
    """
    if instrument not in INSTRUMENTS or receipt["url"] != url_for(instrument):
        raise SourceError("Income identity mismatch")
    if not timestamp(receipt["requestedAt"]) <= timestamp(receipt["receivedAt"]) <= timestamp(now()):
        raise SourceError("Invalid source receipt chronology")
    identity = {"instrument": instrument, "url": url_for(instrument), "query": QUERY}
    raw_hash = digest(body)
    key = digest(encoded({"identity": identity, "rawSha256": raw_hash, "lineage": lineage}))
    with directory(root, create=True) as fd:
        try:
            saved = decoded(read(fd, key + ".json"))
        except FileNotFoundError:
            saved = None
        if saved is not None:
            if timestamp(receipt["receivedAt"]) < timestamp(saved["receipt"]["receivedAt"]):
                raise SourceError("Receipt predates first-seen snapshot")
            return read_snapshot(root, key, digest(encoded(saved)), as_of=now())
        normalized = normalize(body, receipt)
        if previous is not None:
            old = read_snapshot(root, previous["key"], previous["sealSha256"], as_of=receipt["receivedAt"])
            if old["identity"] != identity:
                raise SourceError("Previous snapshot identity mismatch")
        sealed = now()
        value = {"version": VERSION, "key": key, "identity": identity, "lineage": lineage,
            "rawSha256": raw_hash, "receipt": receipt, "normalized": normalized,
            "normalizedAt": sealed, "sealedAt": sealed, "usableAt": sealed,
            "previous": previous, "changeMeaning": "snapshot_change_only" if previous else "initial_observation",
            "provenanceClaimsVerified": False, **GATES}
        try:
            publish(fd, raw_hash + ".raw", body)
        except FileExistsError:
            if read(fd, raw_hash + ".raw") != body:
                raise SourceError("Immutable raw conflict")
        publish(fd, key + ".json", encoded(value))
        return value


def read_snapshot(root, key, seal_sha256, *, as_of):
    if any(not re.fullmatch(r"[0-9a-f]{64}", h) for h in [key, seal_sha256]):
        raise SourceError("Invalid external snapshot hash")
    cutoff = timestamp(as_of)
    if cutoff > timestamp(now()):
        raise SourceError("Future read cutoff")
    with directory(root) as fd:
        body = read(fd, key + ".json")
        if digest(body) != seal_sha256:
            raise SourceError("Snapshot seal mismatch")
        value = decoded(body)
        computed = digest(encoded({"identity": value["identity"], "rawSha256": value["rawSha256"], "lineage": value["lineage"]}))
        if (value["version"] != VERSION or value["key"] != key or computed != key
                or digest(read(fd, value["rawSha256"] + ".raw")) != value["rawSha256"]
                or any(value[k] is not False for k in GATES)
                or not timestamp(value["receipt"]["requestedAt"]) <= timestamp(value["receipt"]["receivedAt"])
                <= timestamp(value["normalizedAt"]) <= timestamp(value["sealedAt"]) == timestamp(value["usableAt"]) <= cutoff):
            raise SourceError("Snapshot integrity or time-bound read failure")
        return value


def run(plan_path, *, token_path=TOKEN_PATH, transport=None):
    path = Path(os.path.abspath(plan_path))
    if Path(__file__).resolve().parents[1] in path.parents:
        raise SourceError("Private plan must be outside repository")
    with directory(path.parent) as plan_fd:
        plan_body = read(plan_fd, path.name)
        plan = decoded(plan_body)
        if (plan.get("version") != VERSION or plan.get("instruments") != INSTRUMENTS or plan.get("query") != QUERY
                or timestamp(plan["frozenAt"]) > timestamp(now())
                or any(type(plan.get(k)) is not int or not 0 < plan[k] <= v for k, v in LIMITS.items())
                or plan["maximumRequests"] != 2):
            raise SourceError("Invalid fixed income pilot plan")
        def verify_sources():
            if read(plan_fd, path.name) != plan_body or not plan["sources"]:
                raise SourceError("Frozen plan mismatch")
            for source in plan["sources"]:
                p = Path(source["path"])
                with directory(p.parent) as fd:
                    if digest(read(fd, p.name)) != source["sha256"]:
                        raise SourceError("Frozen source mismatch")
        verify_sources()
        lineage = {"planSha256": digest(plan_body), "codeSha256": digest(Path(__file__).read_bytes()),
            "sources": plan["sources"]}
        fingerprint = digest(encoded(lineage))
        out = path.parent / "run"
        with directory(out, create=True) as fd:
            handle = os.open(".lock", os.O_WRONLY | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600, dir_fd=fd)
            with os.fdopen(handle, "wb") as lock:
                info = os.fstat(lock.fileno())
                if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o600:
                    raise SourceError("Income lock must be private regular 0600")
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                try:
                    complete = decoded(read(fd, "completed.json"))
                except FileNotFoundError:
                    complete = None
                if complete is not None:
                    if complete["fingerprint"] != fingerprint:
                        raise SourceError("Immutable income lineage mismatch")
                    for name, h in complete["artifacts"].items():
                        if digest(read(fd, name)) != h:
                            raise SourceError("Immutable income artifact mismatch")
                    for ref in decoded(read(fd, "results.json"))["snapshots"]:
                        read_snapshot(out / "snapshots", ref["key"], ref["sealSha256"], as_of=now())
                    return {**decoded(read(fd, "results.json")), "cacheHit": True}
                try:
                    read(fd, "started.json")
                except FileNotFoundError:
                    pass
                else:
                    raise SourceError("Prior attempt retained; no automatic retry")
                publish(fd, "started.json", encoded({"fingerprint": fingerprint, "lineage": lineage, "startedAt": now()}))
                client, refs = None, []
                try:
                    client = ReadOnlyIncome(credential(token_path), plan, transport)
                    for instrument in INSTRUMENTS:
                        body, receipt = client.get(url_for(instrument))
                        # Keep original non-success/malformed data privately before quarantining.
                        name = instrument["symbol"]
                        publish(fd, name + ".raw", body)
                        publish(fd, name + "-receipt.json", encoded(receipt))
                        if receipt["httpStatus"] != 200 or receipt["contentType"] != "application/json":
                            raise SourceError("Income HTTP/content failure; no retry")
                        value = snapshot(out / "snapshots", instrument, body, receipt, lineage)
                        refs.append({"symbol": name, "key": value["key"], "sealSha256": digest(encoded(value)),
                                     "firstSeenAt": value["receipt"]["receivedAt"], "usableAt": value["usableAt"]})
                    verify_sources()
                    result = {"version": VERSION, "lineage": lineage, "requests": client.calls,
                        "responseBytes": client.bytes, "snapshots": refs, "cashSpent": 0, "modelsTrained": 0,
                        "historicalFinancialFeaturesQualified": False, "cacheHit": False, **GATES}
                    publish(fd, "results.json", encoded(result))
                    names = ["started.json", "results.json"] + [i["symbol"] + suffix
                        for i in INSTRUMENTS for suffix in [".raw", "-receipt.json"]]
                    publish(fd, "completed.json", encoded({"fingerprint": fingerprint,
                        "artifacts": {n: digest(read(fd, n)) for n in names}, "snapshots": refs}))
                    return result
                except Exception:
                    publish(fd, "failure.json", encoded({"failedAt": now(), "requests": client.calls if client else 0,
                        "responseBytes": client.bytes if client else 0, "httpStatus": client.last_status if client else None,
                        "partialSnapshots": refs, "error": "Income pilot stopped; no automatic retry", **GATES}))
                    raise SourceError("Income pilot stopped; inspect private failure; no automatic retry") from None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    result = run(parser.parse_args().plan)
    print(json.dumps({k: result[k] for k in ["requests", "responseBytes", "cashSpent", "modelsTrained", "cacheHit"]}))
