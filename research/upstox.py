"""Budgeted free read-only historical source pilot; never a broker adapter.

python -m research.upstox /private/path/PLAN.json
Only the official public NSE catalogue and V3 historical-candle GET endpoints
exist in this client. Credentials never enter artifacts, URLs or exception text.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import re
from datetime import date, datetime, timezone
from pathlib import Path
from urllib.parse import quote, urlsplit

import numpy as np
import pandas as pd
import requests

from research.architecture import sha
from research.engine import atomic_json
from research.prospective import job_lock

CATALOGUE = "https://assets.upstox.com/market-quote/instruments/exchange/NSE.json.gz"
HISTORY = "https://api.upstox.com/v3/historical-candle/"
TOKEN_PATH = Path.home() / ".config/folio/upstox-analytics-token"
LIMITS = {"maximumRequests": 10, "maximumBytes": 15000000,
          "maximumResponseBytes": 5000000, "timeoutSeconds": 20}


def credential(path):
    path = Path(path)
    if (path.is_symlink() or path.parent.is_symlink() or path.stat().st_uid != os.getuid()
            or path.stat().st_mode & 0o077 or path.parent.stat().st_mode & 0o077
            or Path(__file__).resolve().parents[1] in path.resolve().parents):
        raise ValueError("Credential must be owner-private and outside repository storage")
    value = path.read_text().strip()
    if len(value.split(".")) != 3 or not value.isascii() or any(c.isspace() for c in value):
        raise ValueError("Invalid credential format")
    return value


class ReadOnlyHistory:
    def __init__(self, token, plan, transport=None):
        self.token, self.plan = token, plan
        self.transport = transport or requests.Session()
        self.transport.trust_env = False
        self.calls, self.bytes = 0, 0
        self.allowed = {CATALOGUE}
        for instrument in plan["instruments"]:
            key = instrument["key"]
            if not re.fullmatch(r"NSE_EQ\|[A-Z0-9]{12}|NSE_INDEX\|[A-Za-z0-9 ]+", key):
                raise ValueError("Invalid cash/index instrument key")
            for window in plan["windows"]:
                unit, interval = window["unit"], window["interval"]
                start, end = date.fromisoformat(window["from"]), date.fromisoformat(window["to"])
                if (unit, interval) not in {("days", 1), ("minutes", 5)} or start > end or end >= date.today():
                    raise ValueError("Invalid completed historical window")
                self.allowed.add(HISTORY + quote(key, safe="") + f"/{unit}/{interval}/{end}/{start}")

    def get(self, url):
        parsed = urlsplit(url)
        if (url not in self.allowed or url != CATALOGUE and not (url.startswith(HISTORY) and parsed.scheme == "https"
                and parsed.netloc == "api.upstox.com" and not parsed.query and not parsed.fragment)):
            raise ValueError("Endpoint is outside the free historical-data allowlist")
        if self.calls >= self.plan["maximumRequests"]:
            raise ValueError("Request budget exhausted")
        self.calls += 1
        headers = {"Accept": "application/json"}
        if url.startswith(HISTORY):
            headers["Authorization"] = "Bearer " + self.token
        try:
            with self.transport.get(url, headers=headers, stream=True, allow_redirects=False,
                                    timeout=self.plan["timeoutSeconds"]) as response:
                if response.status_code != 200:
                    raise ValueError(f"Historical source HTTP {int(response.status_code)}; no retry")
                chunks, size = [], 0
                for chunk in response.iter_content(65536):
                    size += len(chunk)
                    self.bytes += len(chunk)
                    if size > self.plan["maximumResponseBytes"] or self.bytes > self.plan["maximumBytes"]:
                        raise ValueError("Historical source byte budget exceeded")
                    chunks.append(chunk)
                return b"".join(chunks)
        except requests.RequestException:
            raise ValueError("Historical source transport failed; no retry or credential logging") from None


def audit_candles(payload, window):
    if payload.get("status") != "success":
        raise ValueError("Historical response did not report success")
    candles = payload.get("data", {}).get("candles")
    if not isinstance(candles, list) or not candles:
        raise ValueError("Historical source returned no candles")
    if any(not isinstance(row, list) or len(row) != 7 for row in candles):
        raise ValueError("Unexpected candle schema")
    dates = pd.DatetimeIndex(pd.to_datetime([row[0] for row in candles]))
    numbers = np.asarray([row[1:6] for row in candles], dtype=float)
    if dates.tz is None or dates.hasnans or dates.has_duplicates or not np.isfinite(numbers).all():
        raise ValueError("Invalid candle timestamps or values")
    dates = dates.tz_convert("Asia/Kolkata")
    if (min(dates.date).isoformat() < window["from"] or max(dates.date).isoformat() > window["to"]
            or dates.max() >= pd.Timestamp.now(tz="Asia/Kolkata").normalize()):
        raise ValueError("Candles outside fixed completed historical dates")
    o, h, low, c, volume = numbers.T
    if not ((np.minimum.reduce([o, h, low, c]) > 0) & (volume >= 0)
            & (h >= np.maximum(o, c)) & (low <= np.minimum(o, c)) & (h >= low)).all():
        raise ValueError("Invalid source OHLCV")
    sorted_dates = dates.sort_values()
    counts = pd.Series(1, index=sorted_dates).groupby(sorted_dates.strftime("%Y-%m-%d")).sum()
    return {"rows": len(candles), "first": str(dates.min()), "last": str(dates.max()),
            "sessionCounts": {str(k): int(v) for k, v in counts.items()},
            "zeroVolumeBars": int((volume == 0).sum()), "priceBasisVerifiedRaw": False,
            "calendarVerified": False, "trainingApproved": False}


def run(plan_path, *, token_path=TOKEN_PATH, transport=None):
    path = Path(plan_path).resolve()
    plan = json.loads(path.read_text())
    frozen = datetime.fromisoformat(plan["frozenAt"])
    if (plan["version"] != "upstox-source-pilot-1" or frozen.tzinfo is None
            or frozen > datetime.now(timezone.utc)
            or any(type(plan.get(k)) is not int or not 0 < plan[k] <= limit for k, limit in LIMITS.items())
            or len(plan["instruments"]) != 3 or len(plan["windows"]) != 3):
        raise ValueError("Invalid fixed source-pilot plan")
    source = plan["sourceManifest"]
    if sha(source["path"]) != source["sha256"]:
        raise ValueError("Source-use evidence hash mismatch")
    source_snapshots = json.loads(Path(source["path"]).read_text())
    if any(e.get("status") != 200 or sha(e["path"]) != e["sha256"] for e in source_snapshots):
        raise ValueError("Source-use snapshot mismatch")
    lineage = {"planSha256": sha(path), "codeSha256": sha(__file__)}
    fingerprint = hashlib.sha256(json.dumps(lineage, sort_keys=True).encode()).hexdigest()
    out = path.parent / "run"
    out.mkdir(mode=0o700, exist_ok=True)
    with job_lock(out):
        if (out / "completed.json").exists():
            saved = json.loads((out / "completed.json").read_text())
            if saved["fingerprint"] != fingerprint or any(sha(out / f) != h for f, h in saved["artifacts"].items()):
                raise ValueError("Source-pilot cache/lineage integrity failure")
            return {**json.loads((out / "results.json").read_text()), "cacheHit": True}
        if (out / "started.json").exists():
            raise ValueError("Prior attempt retained; inspect before a frozen continuation")
        client = ReadOnlyHistory(credential(token_path), plan, transport)
        atomic_json(out / "started.json", {**lineage, "fingerprint": fingerprint})
        artifacts, observations = [], []
        try:
            data = client.get(CATALOGUE)
            catalogue_path = out / "catalogue.json.gz"
            fd = os.open(catalogue_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            with os.fdopen(fd, "wb") as f:
                f.write(data)
            artifacts.append(catalogue_path.name)
            with gzip.GzipFile(fileobj=io.BytesIO(data)) as f:
                expanded = f.read(50000001)
            if len(expanded) > 50000000:
                raise ValueError("Expanded catalogue exceeds pilot budget")
            catalogue = json.loads(expanded)
            for instrument in plan["instruments"]:
                key = instrument["key"]
                matches = [row for row in catalogue if row.get("instrument_key") == key]
                if len(matches) != 1 or not key.startswith(("NSE_EQ|", "NSE_INDEX|")):
                    raise ValueError("Declared instrument not uniquely present in current catalogue")
                for index, window in enumerate(plan["windows"]):
                    if (window["unit"], window["interval"]) not in {("days", 1), ("minutes", 5)}:
                        raise ValueError("Pilot supports only daily/five-minute historical bars")
                    url = (HISTORY + quote(key, safe="") + f"/{window['unit']}/{window['interval']}/"
                           + window["to"] + "/" + window["from"])
                    payload = json.loads(client.get(url))
                    name = f"{instrument['symbol']}-{index}.json"
                    if Path(name).name != name:
                        raise ValueError("Unsafe instrument filename")
                    atomic_json(out / name, payload)
                    artifacts.append(name)
                    observation = audit_candles(payload, window)
                    observations.append({"symbol": instrument["symbol"], "instrumentKey": key,
                        "window": window, "file": name, "catalogueMatch": matches[0], **observation})
            result = {"lineage": lineage, "requests": client.calls, "responseBytes": client.bytes,
                      "observations": observations, "priceBasis": plan["priceBasis"], "cashSpent": 0,
                      "modelsTrained": 0, "livePromotionAllowed": False, "cacheHit": False}
            if (sha(path) != lineage["planSha256"] or sha(source["path"]) != source["sha256"]
                    or any(sha(e["path"]) != e["sha256"] for e in source_snapshots)):
                raise ValueError("Source evidence changed during pilot")
            atomic_json(out / "results.json", result)
            artifacts.append("results.json")
            atomic_json(out / "completed.json", {"fingerprint": fingerprint,
                        "artifacts": {name: sha(out / name) for name in artifacts}})
            return result
        except Exception as error:
            # Never serialize request headers, exception strings, or credentials.
            atomic_json(out / "failure.json", {"errorType": type(error).__name__, "requests": client.calls,
                        "responseBytes": client.bytes, "partialArtifacts": artifacts})
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    result = run(parser.parse_args().plan)
    print(json.dumps({k: result[k] for k in ["requests", "responseBytes", "cashSpent", "modelsTrained", "cacheHit"]}))
