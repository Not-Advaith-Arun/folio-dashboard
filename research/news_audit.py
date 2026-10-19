"""Offline admission audit for legacy news/forecast exports; never issues trades.

Publication time is not availability time. Reconstruct only the exact issuance
snapshot, and never upgrade mutable legacy evidence into a sealed prediction.
Run: python -m research.news_audit /private/path/protocol.json
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import date, datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
from zoneinfo import ZoneInfo

VERSION = "news-timing-audit-1"
IST = ZoneInfo("Asia/Kolkata")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def instant(value):
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("Timezone required")
    return parsed.astimezone(timezone.utc)


def number(value):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise ValueError("Finite number required")
    return float(value)


def article_issues(article, cutoff):
    issues = []
    try:
        if any(not isinstance(article[k], str) or not article[k] for k in
               ["id", "symbol", "title", "url", "source", "label"]):
            raise ValueError("Missing article identity")
        published, seen = instant(article["publishedAt"]), instant(article["firstSeenAt"])
        if published > cutoff:
            issues.append("publication_after_cutoff")
        if seen > cutoff:
            issues.append("first_seen_after_cutoff")
        if not -1 <= number(article["sentiment"]) <= 1:
            raise ValueError("Invalid sentiment")
    except (KeyError, TypeError, ValueError, OverflowError):
        issues.append("invalid_article")
    return issues


def audit_bundle(ledger, snapshots, reports, first_seen, as_of):
    """Read-only legacy audit. No prices are read and no outcomes are resolved."""
    cutoff = instant(as_of)
    registry_issues, snapshot_rows = [], []
    versions = defaultdict(set)
    for name, snapshot in snapshots.items():
        issues = []
        try:
            recorded = instant(snapshot["recordedAt"])
            if recorded > cutoff:
                issues.append("snapshot_after_audit")
            articles = snapshot["articles"]
            for article in articles:
                issues.extend(article_issues(article, recorded))
                content = {k: article.get(k) for k in ["title", "url", "source", "publishedAt"]}
                versions[article.get("id")].add(digest(content))
                key = article.get("id")
                if key not in first_seen:
                    registry_issues.append("missing_registry_id")
                else:
                    try:
                        if instant(first_seen[key]) != instant(article["firstSeenAt"]):
                            registry_issues.append("registry_timestamp_conflict")
                    except (TypeError, ValueError, KeyError):
                        registry_issues.append("invalid_registry_timestamp")
        except (TypeError, ValueError, KeyError):
            issues.append("invalid_snapshot")
        snapshot_rows.append({"file": name, "articleCount": len(snapshot.get("articles", [])),
                              "issues": dict(Counter(issues)), "contentSha256": digest(snapshot)})

    rows = []
    ids = Counter(f.get("id") for f in ledger)
    for forecast in ledger:
        issues, predictions = [], []
        exact_reports, exact_news, later = [], [], []
        issued = None
        try:
            issued = instant(forecast["createdAt"])
            if issued > cutoff:
                issues.append("issuance_after_audit")
            if date.fromisoformat(forecast["signalDate"]) < issued.astimezone(IST).date():
                issues.append("signal_before_actual_issuance_day")
            if ids[forecast["id"]] != 1:
                issues.append("duplicate_forecast_id")
            for name, report in reports.items():
                if report.get("id") != forecast["id"]:
                    continue
                generated = instant(report["generatedAt"])
                if generated == issued:
                    exact_reports.append((name, report))
                elif generated > issued:
                    later.append(name)
            # Exact timestamp and full content, never filename/nearest refresh.
            for name, snapshot in snapshots.items():
                if instant(snapshot["recordedAt"]) == issued:
                    exact_news.append((name, snapshot))
        except (KeyError, TypeError, ValueError):
            issues.append("invalid_issuance_or_evidence_time")
        if len(exact_reports) != 1:
            issues.append("missing_or_ambiguous_issuance_report")
        report_hash = news_hash = None
        if len(exact_reports) == 1 and issued is not None:
            _, report = exact_reports[0]
            report_hash = digest(report)
            try:
                articles = report["articles"]
                matching = [(name, s) for name, s in exact_news
                            if digest(s["articles"]) == digest(articles)
                            and s["revision"] == report["textModelRevision"]]
                if articles and len(matching) != 1:
                    issues.append("missing_or_ambiguous_exact_news_snapshot")
                elif articles:
                    news_hash = digest(matching[0][1])
                if bool(articles) != forecast["newsEnabled"]:
                    issues.append("news_enabled_mismatch")
                if report["modelId"] != forecast["modelId"]:
                    issues.append("model_mismatch")
                if not report.get("dataFingerprint") or not report.get("textModelRevision"):
                    issues.append("missing_model_lineage")
                symbols = forecast["universe"]
                rankings = report["rankings"]
                if (not symbols or len(set(symbols)) != len(symbols)
                        or sorted(r["symbol"] for r in rankings) != sorted(symbols)):
                    issues.append("ranking_universe_mismatch")
                keys = [(a["symbol"], a["id"]) for a in articles]
                if len(set(keys)) != len(keys):
                    issues.append("duplicate_article_per_symbol")
                for article in articles:
                    issues.extend(article_issues(article, issued))
                    if article["symbol"] not in symbols:
                        issues.append("article_outside_universe")
                # Existing freshness-v1 arithmetic, with availability admission
                # enforced independently. Invalid/late rows are not silently fixed.
                for row in rankings:
                    selected = [a for a in articles if a["symbol"] == row["symbol"]]
                    if any(article_issues(a, issued) for a in selected):
                        continue
                    weights = [math.exp(-(issued - instant(a["publishedAt"])).total_seconds()
                                        / 172800) for a in selected]
                    news = sum(number(a["sentiment"]) * w for a, w in zip(selected, weights)) / max(1., sum(weights))
                    score = number(row["priceScore"]) + .15 * news
                    if (not math.isclose(number(row["newsScore"]), news, abs_tol=1e-12, rel_tol=0)
                            or not math.isclose(number(row["score"]), score, abs_tol=1e-12, rel_tol=0)
                            or row["headlines"] != len(selected)):
                        issues.append("overlay_feature_mismatch")
                    predictions.append({"symbol": row["symbol"], "priceScore": row["priceScore"],
                        "reconstructedNewsScore": news, "reconstructedScore": score,
                        "articleHashes": [digest(a) for a in selected]})
                top_k = report["config"]["topK"]
                if type(top_k) is not int or not 1 <= top_k <= len(symbols):
                    raise ValueError("Invalid topK")
                for key, field in [("pricePicks", "priceScore"), ("newsPicks", "score")]:
                    picks = [r["symbol"] for r in sorted(rankings, key=lambda r: number(r[field]), reverse=True)[:top_k]]
                    if picks != forecast[key]:
                        issues.append(key + "_mismatch")
            except (KeyError, ValueError, TypeError, OverflowError):
                issues.append("invalid_issuance_payload")
        outcome_issues = []
        if forecast.get("status") == "resolved":
            try:
                entry, exit_day = date.fromisoformat(forecast["entryDate"]), date.fromisoformat(forecast["exitDate"])
                if issued is None or entry <= issued.astimezone(IST).date():
                    outcome_issues.append("entry_not_after_actual_issuance_day")
                if exit_day <= entry or exit_day > cutoff.astimezone(IST).date():
                    outcome_issues.append("invalid_or_future_exit")
            except (KeyError, TypeError, ValueError):
                outcome_issues.append("invalid_outcome_dates")
            outcome_issues.append("missing_observed_price_lineage")
        elif forecast.get("status") != "pending":
            outcome_issues.append("unknown_outcome_status")
        rows.append({"id": forecast.get("id"), "createdAt": forecast.get("createdAt"),
            "originalStatus": forecast.get("status"), "forecastContentSha256": digest(forecast),
            "issuanceReportSha256": report_hash, "issuanceNewsSha256": news_hash,
            "laterRefreshFiles": later, "featureIssues": dict(Counter(issues)),
            "reconstructionPassed": not issues, "outcomeIssues": outcome_issues,
            "prospectiveEligible": False, "admissionBlockers": ["legacy_missing_issuance_seal"],
            "predictions": predictions})
    return {"version": VERSION, "auditAsOf": as_of, "forecasts": rows,
        "snapshots": snapshot_rows, "summary": {"forecasts": len(rows),
            "reconstructable": sum(r["reconstructionPassed"] for r in rows),
            "verifiedProspective": 0, "pending": sum(f.get("status") == "pending" for f in ledger),
            "newsSnapshots": len(snapshots), "runExports": len(reports),
            "laterRefreshes": sum(len(r["laterRefreshFiles"]) for r in rows),
            "articleIdsWithContentRevisions": sum(len(v) > 1 for v in versions.values()),
            "registryIssues": dict(Counter(registry_issues)),
            "featureIssues": dict(Counter(k for r in rows for k in r["featureIssues"]))},
        "limitations": ["Hashes were recorded at audit time, not proven at original issuance.",
            "Legacy reconstruction is not prospective validation or tradable profit.",
            "No original ledger/news/model/price file was changed; outcomes remain untouched.",
            "Standalone admission audit; production writer/resolver integration remains pending."]}


def write_once(path, value):
    # Exclusive creation preserves failures too; never replace partial evidence.
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as stream:
        json.dump(value, stream, sort_keys=True, allow_nan=False, separators=(",", ":"))
        stream.flush()
        os.fsync(stream.fileno())


def run_audit(protocol_path):
    protocol_path = Path(protocol_path).resolve()
    protocol = json.loads(protocol_path.read_text())
    if protocol["version"] != VERSION:
        raise ValueError("Unsupported audit protocol")
    instant(protocol["frozenAt"])
    instant(protocol["auditAsOf"])
    if sha(Path(__file__).with_name("engine.py")) != protocol["engineSha256"]:
        raise ValueError("Source engine changed; freeze a new protocol")
    loaded = defaultdict(dict)
    paths = set()
    for entry in protocol["inputs"]:
        path = Path(entry["path"]).resolve()
        if path in paths or entry["role"] not in {"ledger", "news", "report", "first_seen"}:
            raise ValueError("Duplicate input or invalid role")
        paths.add(path)
        # Hash the exact bytes parsed, avoiding a hash/read race.
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != entry["sha256"]:
            raise ValueError("Input hash mismatch")
        loaded[entry["role"]][str(path)] = json.loads(raw)
    if len(loaded["ledger"]) != 1 or len(loaded["first_seen"]) != 1:
        raise ValueError("Exactly one ledger and first-seen registry required")
    out = Path(protocol["outputDir"]).resolve()
    if out == protocol_path.parent or any(p == out or p.is_relative_to(out) for p in paths):
        raise ValueError("Output must be separate from inputs/protocol")
    out.mkdir(parents=True, exist_ok=True, mode=0o700)
    fingerprint = digest({"protocol": sha(protocol_path), "code": sha(__file__)})
    with (out / ".lock").open("a") as lock:
        os.chmod(out / ".lock", 0o600)
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        completed = out / "completed.json"
        if completed.exists():
            saved = json.loads(completed.read_text())
            if saved["fingerprint"] != fingerprint or sha(out / "audit.json") != saved["auditSha256"]:
                raise ValueError("Completed audit lineage/hash mismatch")
            return json.loads((out / "audit.json").read_text())
        if (out / "started.json").exists():
            raise ValueError("Incomplete audit preserved; use a new output directory")
        write_once(out / "started.json", {"fingerprint": fingerprint, "protocolSha256": sha(protocol_path)})
        report = audit_bundle(next(iter(loaded["ledger"].values())), loaded["news"],
                              loaded["report"], next(iter(loaded["first_seen"].values())), protocol["auditAsOf"])
        report["lineage"] = {"protocolSha256": sha(protocol_path), "codeSha256": sha(__file__),
                             "inputs": protocol["inputs"], "parentCommit": protocol["parentCommit"]}
        write_once(out / "audit.json", report)
        write_once(completed, {"fingerprint": fingerprint, "auditSha256": sha(out / "audit.json")})
        return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("protocol", type=Path)
    print(json.dumps(run_audit(parser.parse_args().protocol)["summary"], indent=2))
