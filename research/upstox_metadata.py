"""Bounded free corporate-actions pilot; no account, order or price endpoints.

python -m research.upstox_metadata /private/path/PLAN.json
Event dates are provider claims, never historical feature-availability evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

import requests

from research.architecture import sha
from research.engine import atomic_json
from research.prospective import job_lock
from research.upstox import TOKEN_PATH, credential

VERSION = "upstox-corporate-actions-1"
INSTRUMENTS = {"RELIANCE": "INE002A01018", "INFY": "INE009A01021"}
BASE = "https://api.upstox.com/v2/fundamentals/"
LIMITS = {"maximumRequests": 2, "maximumBytes": 2000000,
          "maximumResponseBytes": 1000000, "timeoutSeconds": 20}
GATES = {"rawPriceVerified": False, "volumeUnitsVerified": False,
         "productionCalendarVerified": False, "bonusTradabilityVerified": False,
         "historicalAvailabilityVerified": False, "trainingApproved": False,
         "livePromotionAllowed": False}
DATE_LABELS = {"announcement date", "record date", "ex dividend date", "ex bonus date",
               "ex split date", "allotment date"}


def now():
    return datetime.now(timezone.utc).isoformat()


class SourceError(ValueError):
    """Only internally generated, credential-free messages may escape transport."""


class ReadOnlyActions:
    def __init__(self, token, plan, transport=None):
        self._token, self.plan = token, plan
        self.allowed = {BASE + isin + "/corporate-actions" for isin in INSTRUMENTS.values()}
        self.transport = transport if transport is not None else requests.Session()
        self.transport.trust_env = False
        self.calls, self.bytes, self.last_status = 0, 0, None

    def get(self, url):
        if url not in self.allowed:
            raise SourceError("Endpoint outside fixed corporate-actions allowlist")
        if self.calls >= self.plan["maximumRequests"]:
            raise SourceError("Request budget exhausted")
        self.calls += 1
        requested = now()
        self.last_status = None
        try:
            with self.transport.get(url, headers={"Accept": "application/json",
                    "Authorization": "Bearer " + self._token}, stream=True, allow_redirects=False,
                    timeout=self.plan["timeoutSeconds"]) as response:
                self.last_status = int(response.status_code)
                if self.last_status != 200:
                    raise SourceError("Metadata source HTTP failure; no retry")
                chunks, size = [], 0
                for chunk in response.iter_content(65536):
                    size += len(chunk)
                    self.bytes += len(chunk)
                    if size > self.plan["maximumResponseBytes"] or self.bytes > self.plan["maximumBytes"]:
                        raise SourceError("Metadata source byte budget exceeded")
                    chunks.append(chunk)
                body = b"".join(chunks)
                try:
                    decoded = json.loads(body)
                except (ValueError, UnicodeError):
                    decoded = None
                if self._token.encode() in body or self._token in json.dumps(decoded, ensure_ascii=False):
                    raise SourceError("Source echoed credential; response not retained")
                return body, {"url": url, "requestedAt": requested, "receivedAt": now(),
                              "httpStatus": self.last_status, "responseBytes": len(body)}
        except SourceError:
            raise
        except Exception:
            # Requests exceptions can contain headers or other server-controlled
            # values; neither their message nor their chained traceback survives.
            raise SourceError("Metadata source transport failure; no retry") from None


def claimed_date(value):
    if not isinstance(value, str):
        return None
    for fmt in ["%Y-%m-%d", "%d %b %Y"]:
        try:
            return datetime.strptime(value, fmt).date().isoformat()
        except ValueError:
            pass
    return None


def normalize(payload, receipt):
    if not isinstance(payload, dict) or payload.get("status") != "success" or not isinstance(payload.get("data"), list):
        raise SourceError("Unexpected metadata success/data schema")
    records = []
    for index, event in enumerate(payload["data"]):
        if (not isinstance(event, dict) or not isinstance(event.get("name"), str) or not event["name"].strip()
                or not isinstance(event.get("event_details"), list)
                or any(event.get(k) is not None and not isinstance(event[k], str) for k in ["expiry_date", "ratio"])
                or (event.get("amount") is not None and (type(event["amount"]) not in [int, float]
                    or not math.isfinite(event["amount"])))):
            raise SourceError("Malformed corporate event")
        details = []
        for detail in event["event_details"]:
            if (not isinstance(detail, dict) or not isinstance(detail.get("name"), str)
                    or not isinstance(detail.get("value"), str)):
                raise SourceError("Malformed event detail")
            label = detail["name"].strip().casefold()
            details.append({"source": detail, "recognizedDateLabel": label in DATE_LABELS,
                            "claimedDate": claimed_date(detail["value"]) if label in DATE_LABELS else None})
        records.append({"sourceIndex": index, "sourceEvent": event, "details": details,
                        "claimedEffectiveDate": claimed_date(event.get("expiry_date")),
                        "firstSeenAt": receipt["receivedAt"], **GATES})
    return records


def compare(records, facts):
    comparisons = []
    for fact in facts:
        candidates = []
        for record in records:
            event = record["sourceEvent"]
            if event["name"].strip().casefold() != fact["action"].casefold():
                continue
            dates = lambda label: [d["claimedDate"] for d in record["details"]
                                  if d["source"]["name"].strip().casefold() == label]
            record_dates, allotment = dates("record date"), dates("allotment date")
            agrees = lambda values, expected: (all(v == expected for v in values if v is not None)
                                               if any(v is not None for v in values) else None)
            candidates.append({"sourceIndex": record["sourceIndex"],
                "ratioMatches": event["ratio"] == fact["ratio"] if event.get("ratio") is not None else None,
                "recordDateClaims": record_dates,
                "recordDateMatches": agrees(record_dates, fact["recordDate"]),
                "allotmentDateClaims": allotment,
                "allotmentDateMatches": agrees(allotment, fact["allotmentDate"])})
        comparisons.append({"reviewedFact": fact, "status": "candidates_retained" if candidates else "missing_action",
                            "candidates": candidates, "completeHistoryVerified": False, **GATES})
    return comparisons


def run(plan_path, *, token_path=TOKEN_PATH, transport=None):
    path = Path(plan_path).resolve()
    plan_digest = sha(path)
    plan = json.loads(path.read_text())
    frozen = datetime.fromisoformat(plan["frozenAt"])
    if (plan["version"] != VERSION or frozen.tzinfo is None or frozen > datetime.now(timezone.utc)
            or len(plan["instruments"]) != 2
            or {i["symbol"]: i["isin"] for i in plan["instruments"]} != INSTRUMENTS
            or any(type(plan.get(k)) is not int or not 0 < plan[k] <= v for k, v in LIMITS.items())):
        raise SourceError("Invalid fixed metadata pilot")
    if path.parent.stat().st_mode & 0o077 or Path(__file__).resolve().parents[1] in path.parents:
        raise SourceError("Evidence directory must be private and outside repository")
    def verify_sources():
        if sha(path) != plan_digest or not plan["sources"] or any(sha(e["path"]) != e["sha256"] for e in plan["sources"]):
            raise SourceError("Frozen plan/source evidence mismatch")
    verify_sources()
    dependencies = [Path(__file__).with_name(n) for n in
                    ["upstox_metadata.py", "upstox.py", "architecture.py", "engine.py", "prospective.py"]]
    lineage = {"planSha256": plan_digest, "dependencies": {str(p): sha(p) for p in dependencies}}
    fingerprint = hashlib.sha256(json.dumps(lineage, sort_keys=True).encode()).hexdigest()
    out = path.parent / "run"
    out.mkdir(mode=0o700, exist_ok=True)
    if out.is_symlink() or out.stat().st_mode & 0o077:
        raise SourceError("Output directory must be private")
    with job_lock(out):
        complete = out / "completed.json"
        if complete.exists():
            saved = json.loads(complete.read_text())
            if (saved["fingerprint"] != fingerprint or "results.json" not in saved["artifacts"]
                    or any(Path(n).name != n or sha(out / n) != h for n, h in saved["artifacts"].items())):
                raise SourceError("Immutable metadata cache integrity failure")
            return {**json.loads((out / "results.json").read_text()), "cacheHit": True}
        if (out / "started.json").exists():
            raise SourceError("Prior attempt retained; no automatic retry")
        atomic_json(out / "started.json", {**lineage, "fingerprint": fingerprint, "startedAt": now()})
        client, artifacts = None, ["started.json"]
        try:
            client = ReadOnlyActions(credential(token_path), plan, transport)
            observations = []
            for instrument in plan["instruments"]:
                url = BASE + instrument["isin"] + "/corporate-actions"
                body, receipt = client.get(url)
                name = instrument["symbol"] + ".json"
                with os.fdopen(os.open(out / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb") as f:
                    f.write(body)
                artifacts.append(name)
                receipt.update(instrument=instrument, file=name, sha256=sha(out / name))
                atomic_json(out / (instrument["symbol"] + "-receipt.json"), receipt)
                artifacts.append(instrument["symbol"] + "-receipt.json")
                records = normalize(json.loads(body), receipt)
                facts = [f for f in plan["reviewedFacts"] if f["symbol"] == instrument["symbol"]]
                observations.append({**receipt, "records": records, "comparisons": compare(records, facts), **GATES})
            verify_sources()
            result = {"version": VERSION, "lineage": lineage, "requests": client.calls,
                "responseBytes": client.bytes, "observations": observations, "cacheHit": False,
                "cashSpent": 0, "modelsTrained": 0, **GATES}
            atomic_json(out / "results.json", result)
            artifacts.append("results.json")
            atomic_json(complete, {"fingerprint": fingerprint, "artifacts": {n: sha(out / n) for n in artifacts}})
            return result
        except Exception as error:
            atomic_json(out / "failure.json", {"errorType": type(error).__name__,
                "requests": client.calls if client else 0, "responseBytes": client.bytes if client else 0,
                "httpStatus": client.last_status if client else None, "partialArtifacts": artifacts,
                "failedAt": now(), **GATES})
            raise SourceError("Metadata pilot stopped; inspect private failure record, no automatic retry") from None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    result = run(parser.parse_args().plan)
    print(json.dumps({k: result[k] for k in ["requests", "responseBytes", "cashSpent", "modelsTrained", "cacheHit"]}))
