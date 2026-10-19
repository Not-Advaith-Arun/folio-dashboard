"""Local, append-only prospective evidence. No legacy record is retrofitted.

Seals detect accidental alteration; they are not third-party trusted timestamps.
Resolved returns remain provisional adjusted-price research, never broker fills.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import date, datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from zoneinfo import ZoneInfo

VERSION = "prospective-ledger-1"
IST = ZoneInfo("Asia/Kolkata")
NEWS_FIELDS = ["id", "symbol", "title", "url", "source", "publishedAt"]


def instant(value):
    value = datetime.fromisoformat(value)
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Evidence timestamp must include timezone")
    return value.astimezone(timezone.utc)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                     separators=(",", ":")).encode()).hexdigest()


def evidence_root(root):
    path = Path(root) / VERSION
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    return path


@contextmanager
def job_lock(root):
    with (evidence_root(root) / ".lock").open("a") as lock:
        os.chmod(lock.name, 0o600)
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def read_seal(path, expected=None):
    wrapper = json.loads(Path(path).read_bytes())
    actual = digest(wrapper["payload"])
    if wrapper["sha256"] != actual or expected is not None and actual != expected:
        raise ValueError("Evidence seal mismatch")
    return wrapper["payload"]


def seal(path, payload, *, first_wins=False):
    """Publish complete bytes exclusively; a crash cannot expose a partial seal."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if path.exists():
        previous = read_seal(path)
        if not first_wins and digest(previous) != digest(payload):
            raise ValueError("Immutable evidence conflict")
        return previous
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as f:
            temporary = f.name
            os.chmod(temporary, 0o600)
            json.dump({"sha256": digest(payload), "payload": payload}, f,
                      allow_nan=False, sort_keys=True, separators=(",", ":"))
            f.flush()
            os.fsync(f.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            previous = read_seal(path)
            if not first_wins and digest(previous) != digest(payload):
                raise ValueError("Immutable evidence conflict")
            return previous
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        return payload
    finally:
        if temporary is not None:
            os.unlink(temporary)


def put_blob(root, payload):
    key = digest(payload)
    seal(evidence_root(root) / "blobs" / f"{key}.json", payload)
    return key


def blob(root, key):
    if not isinstance(key, str) or not re.fullmatch(r"[a-f0-9]{64}", key):
        raise ValueError("Invalid evidence reference")
    return read_seal(evidence_root(root) / "blobs" / f"{key}.json", key)


def observe_prices(root, frames, observed_at):
    cutoff = instant(observed_at).astimezone(IST).date()
    bars = {}
    for symbol, frame in frames.items():
        if (not frame.index.is_unique or not frame.index.is_monotonic_increasing
                or len({d.date() for d in frame.index}) != len(frame)):
            raise ValueError("Price observations require unique ordered sessions")
        rows = []
        for session, row in frame.iterrows():
            day = session.date()
            if day >= cutoff:
                raise ValueError("Incomplete or future daily price observation")
            prices = {k: float(row[k]) for k in ["open", "high", "low", "close", "volume"]}
            if any(not math.isfinite(v) or v < 0 or k != "volume" and v == 0 for k, v in prices.items()):
                raise ValueError("Invalid observed price")
            rows.append({"date": day.isoformat(), **prices})
        bars[symbol] = rows
    prices = put_blob(root, {"bars": bars, "basis": "provider-adjusted OHLCV; actions unverified"})
    observation = {"observedAt": observed_at, "pricesSha256": prices,
                   "source": "Local observation of supplied provider/cache data; not original publication time"}
    return put_blob(root, observation)


def capture_news(root, articles, observed_at):
    cutoff = instant(observed_at)
    captured = []
    for article in articles:
        content = {k: article[k] for k in NEWS_FIELDS}
        if (any(not isinstance(v, str) or not v for v in content.values())
                or not content["url"].startswith("https://")
                or instant(content["publishedAt"]) > cutoff
                or instant(article.get("firstSeenAt", observed_at)) > cutoff):
            raise ValueError("Unavailable news input")
        key = digest(content)
        receipt = seal(evidence_root(root) / "news" / f"{key}.json",
                       {"content": content, "firstSeenAt": observed_at}, first_wins=True)
        if receipt["content"] != content or instant(receipt["firstSeenAt"]) > cutoff:
            raise ValueError("Invalid news receipt")
        captured.append({**content, "firstSeenAt": receipt["firstSeenAt"], "contentSha256": key})
    return captured


def validate_news(root, articles, cutoff):
    seen = set()
    for article in articles:
        content = {k: article[k] for k in NEWS_FIELDS}
        key = digest(content)
        receipt = read_seal(evidence_root(root) / "news" / f"{key}.json")
        identity = (article["symbol"], key)
        if (identity in seen or article["contentSha256"] != key or receipt["content"] != content
                or receipt["firstSeenAt"] != article["firstSeenAt"]
                or instant(article["publishedAt"]) > cutoff or instant(article["firstSeenAt"]) > cutoff
                or not math.isfinite(article["sentiment"]) or not -1 <= article["sentiment"] <= 1):
            raise ValueError("Invalid scored news evidence")
        seen.add(identity)


def issue(root, forecast, *, rankings, features, articles, observation, model_hashes,
          analysis_key, text_revision, data_fingerprint, feature_cutoff):
    issued = instant(forecast["createdAt"])
    cutoff = instant(feature_cutoff)
    observed = blob(root, observation)
    if instant(observed["observedAt"]) > cutoff or cutoff > issued:
        raise ValueError("Price input observed after issuance")
    price_data = blob(root, observed["pricesSha256"])
    if (sorted(price_data["bars"]) != forecast["universe"]
            or sorted(r["symbol"] for r in rankings) != forecast["universe"]
            or sorted(r["symbol"] for r in features) != forecast["universe"]
            or date.fromisoformat(forecast["signalDate"]) < issued.astimezone(IST).date()):
        raise ValueError("Forecast universe or timing mismatch")
    validate_news(root, articles, cutoff)
    payload = {"version": VERSION, "forecast": forecast, "rankings": rankings,
        "features": features, "articles": articles, "observationSha256": observation,
        "modelHashes": model_hashes, "analysisKey": analysis_key, "textModelRevision": text_revision,
        "dataFingerprint": data_fingerprint, "featureCutoffAt": feature_cutoff,
        "entryRule": "first declared session after issuance/signal day",
        "ledgerCodeSha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "tradabilityVerified": False}
    if not re.fullmatch(r"[a-f0-9]{64}", analysis_key):
        raise ValueError("Invalid analysis reference")
    analysis_path = Path(root) / "analyses-v2" / f"{analysis_key}.json"
    payload["analysisSha256"] = hashlib.sha256(analysis_path.read_bytes()).hexdigest()
    validate_issue(root, payload)
    return seal(evidence_root(root) / "issues" / f"{digest(forecast['id'])}.json", payload, first_wins=True)


def validate_issue(root, record):
    f = record["forecast"]
    issued = instant(f["createdAt"])
    observation = blob(root, record["observationSha256"])
    prices = blob(root, observation["pricesSha256"])
    cutoff = instant(record["featureCutoffAt"])
    if instant(observation["observedAt"]) > cutoff or cutoff > issued:
        raise ValueError("Invalid issued observation time")
    if any(date.fromisoformat(r["date"]) >= issued.astimezone(IST).date()
           for rows in prices["bars"].values() for r in rows):
        raise ValueError("Issued evidence contains future bars")
    validate_news(root, record["articles"], cutoff)
    for feature in record["features"]:
        if date.fromisoformat(feature["date"]) >= cutoff.astimezone(IST).date():
            raise ValueError("Invalid feature availability date")
    key = record["analysisKey"]
    if not re.fullmatch(r"[a-f0-9]{64}", key):
        raise ValueError("Invalid analysis reference")
    raw = (Path(root) / "analyses-v2" / f"{key}.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != record["analysisSha256"]:
        raise ValueError("Issued analysis integrity failure")
    for model_id, expected in record["modelHashes"].items():
        if not re.fullmatch(r"[a-f0-9]{12}", model_id):
            raise ValueError("Invalid issued model identity")
        if hashlib.sha256((Path(root) / "models" / f"{model_id}.joblib").read_bytes()).hexdigest() != expected:
            raise ValueError("Issued model integrity failure")


def resolve(root, record, observation, calendar, horizon):
    """Resolve only explicit full session windows; missing evidence stays pending."""
    validate_issue(root, record)
    observed = blob(root, observation)
    now = instant(observed["observedAt"])
    key = digest(record)
    out = evidence_root(root) / "outcomes" / f"{key}.json"
    if out.exists():
        saved = read_seal(out)
        if saved["issueSha256"] != key:
            raise ValueError("Outcome lineage mismatch")
        if instant(saved["resolvedAt"]) > now:
            return None, "Saved outcome was not available at this observation cutoff"
        observation_record = blob(root, saved["observationSha256"])
        blob(root, observation_record["pricesSha256"])
        blob(root, saved["calendarSha256"])
        return saved, None
    f = record["forecast"]
    if now <= instant(f["createdAt"]):
        return None, "Awaiting a later price observation"
    if calendar is None:
        return None, "A reviewed session calendar is required to resolve outcomes"
    sessions = calendar["sessions"]
    if (not sessions or sorted(set(sessions)) != sessions
            or not calendar["source"] or not re.fullmatch(r"[a-f0-9]{64}", calendar["sourceSha256"])):
        raise ValueError("Invalid calendar evidence")
    days = [date.fromisoformat(d) for d in sessions]
    if (date.fromisoformat(calendar["coverageStart"]) > days[0]
            or date.fromisoformat(calendar["coverageEnd"]) < days[-1]):
        raise ValueError("Calendar sessions fall outside declared coverage")
    start = max(date.fromisoformat(f["signalDate"]), instant(f["createdAt"]).astimezone(IST).date())
    if date.fromisoformat(calendar["coverageStart"]) > start:
        return None, "Calendar does not cover the issuance boundary"
    eligible = [d for d in days if d > start]
    if len(eligible) < horizon + 1:
        return None, "Calendar does not cover the full outcome window"
    window = eligible[:horizon + 1]
    if (window[-1] >= now.astimezone(IST).date()
            or date.fromisoformat(calendar["coverageEnd"]) < window[-1]):
        return None, "Outcome window is not yet observed"
    prices = blob(root, observed["pricesSha256"])
    values = {}
    for symbol in f["universe"]:
        by_day = {r["date"]: r for r in prices["bars"].get(symbol, [])}
        if any(d.isoformat() not in by_day for d in window):
            return None, "Missing a required security/session observation"
        opens = [by_day[d.isoformat()]["open"] for d in window]
        if any(not math.isfinite(v) or v <= 0 for v in opens):
            raise ValueError("Invalid outcome prices")
        values[symbol] = opens[-1] / opens[0] - 1
    cost = f["costBps"] / 10000
    average = lambda picks: sum(values[s] for s in picks) / len(picks) - cost
    result = {"status": "resolved", "entryDate": window[0].isoformat(), "exitDate": window[-1].isoformat(),
              "priceReturn": average(f["pricePicks"]), "newsReturn": average(f["newsPicks"]),
              "benchmarkReturn": average(f["universe"])}
    for prefix in ["audit", "policy"]:
        if prefix + "Weights" in f:
            result[prefix + "Return"] = sum(w * values[s] for s, w in f[prefix + "Weights"].items()) - f[prefix + "Exposure"] * cost
    outcome = {"issueSha256": key, "observationSha256": observation,
               "calendarSha256": put_blob(root, calendar), "resolvedAt": observed["observedAt"],
               "result": result, "individualReturns": values, "tradabilityVerified": False}
    return seal(out, outcome), None


def ledger_view(root, observation, calendar, horizon):
    rows, reasons = [], set()
    for path in sorted((evidence_root(root) / "issues").glob("*.json")):
        record = read_seal(path)
        if path.stem != digest(record["forecast"]["id"]):
            raise ValueError("Forecast identity seal mismatch")
        outcome, reason = resolve(root, record, observation, calendar, horizon)
        rows.append({**record["forecast"], **(outcome["result"] if outcome else {})})
        if reason:
            reasons.add(reason)
    return sorted(rows, key=lambda r: r["createdAt"]), sorted(reasons)
