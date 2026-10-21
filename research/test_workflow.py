"""Spanning workflows: capital/discovery, session data, research, API security.

Synthetic prices and sentiment are explicit offline fixtures, never production
fallbacks. Calculation timing and access control warrant these critical checks.
"""
import json
import hashlib
import io
import tempfile
import time
import unittest
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from research.engine import HORIZON, run_research
from research import service

SYMBOLS = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "ITC.NS"]
CONFIG = {"symbols": SYMBOLS, "names": {}, "years": 3, "topK": 2, "costBps": 20, "includeNews": True}


def fixture_loader(extended=False):
    today = datetime.now(ZoneInfo("Asia/Kolkata")).date()
    dates = pd.bdate_range(end=today - timedelta(days=1), periods=650)
    if extended:
        dates = dates.append(pd.bdate_range(start=today + timedelta(days=1), periods=12))
    frames = {}
    for i, symbol in enumerate(SYMBOLS):
        rng = np.random.default_rng(100 + i)
        close = 100 * np.exp(np.cumsum(rng.normal(.0003, .012, len(dates))))
        frames[symbol] = pd.DataFrame({"open": close * 1.001, "high": close * 1.01, "low": close * .99, "close": close, "volume": rng.integers(10000, 20000, len(dates))}, index=dates)
    return lambda s, _years, _root: frames[s].copy()


def fixture_news(symbol, _name, _root):
    now = datetime.now(timezone.utc).isoformat()
    return [{"id": symbol, "symbol": symbol, "title": "Offline fixture headline", "url": "https://example.com/fixture", "source": "Offline test fixture", "publishedAt": now, "firstSeenAt": now}]


def fixture_scores(articles):
    return [{**a, "sentiment": .6 if a["symbol"] == "ITC.NS" else -.1, "label": "positive" if a["symbol"] == "ITC.NS" else "negative"} for a in articles]


class ResearchWorkflow(unittest.TestCase):
    def test_news_snapshot_feature_forecast_admission_audit(self):
        # A complete frozen export -> feature -> forecast -> audit workflow.
        # Adversarial timing/provenance cases are critical financial boundaries;
        # no estimator or network is needed to verify them.
        from copy import deepcopy
        from research.news_audit import audit_bundle, run_audit, sha, VERSION
        issued = "2026-10-11T20:00:00+00:00"  # Already October 12 in India.
        articles = [{"id": s, "symbol": s, "title": "Offline fixture", "source": "Fixture",
                     "url": "https://example.com/fixture", "publishedAt": issued,
                     "firstSeenAt": issued, "sentiment": score, "label": label}
                    for s, score, label in [("A", .6, "positive"), ("B", -.4, "negative")]]
        forecast = {"id": "fixture-2026-10-12", "createdAt": issued, "signalDate": "2026-10-12",
            "status": "pending", "modelId": "fixture-model", "universe": ["A", "B"],
            "newsEnabled": True, "pricePicks": ["A"], "newsPicks": ["A"], "costBps": 20}
        report = {"id": forecast["id"], "generatedAt": issued, "modelId": forecast["modelId"],
            "dataFingerprint": "fixture-only", "textModelRevision": "fixture-text-model",
            "config": {"topK": 1}, "articles": articles, "rankings": [
                {"symbol": "A", "priceScore": .75, "newsScore": .6, "score": .84, "headlines": 1},
                {"symbol": "B", "priceScore": .25, "newsScore": -.4, "score": .19, "headlines": 1}]}
        snapshot = {"recordedAt": issued, "revision": "fixture-text-model", "articles": articles}
        seen = {a["id"]: issued for a in articles}
        as_of = "2026-10-19T01:00:00Z"
        def check(f=forecast, s=snapshot, r=report):
            return audit_bundle([f], {"news": s}, {"run": r}, seen, as_of)["forecasts"][0]
        clean = check()
        self.assertTrue(clean["reconstructionPassed"])
        self.assertFalse(clean["prospectiveEligible"])
        self.assertEqual(len(clean["predictions"]), 2)
        # Publication backfills do not excuse first-seen availability after issuance.
        for key, value in [("firstSeenAt", "2026-10-12T01:00:00Z"),
                           ("publishedAt", "2026-10-12T01:00:00Z"),
                           ("firstSeenAt", "2026-10-11T20:00:00")]:
            changed = deepcopy(report)
            changed["articles"][0][key] = value
            self.assertFalse(check(r=changed)["reconstructionPassed"])
        changed = deepcopy(snapshot)
        changed["articles"][0]["title"] = "Revised same-ID content"
        self.assertIn("missing_or_ambiguous_exact_news_snapshot", check(s=changed)["featureIssues"])
        changed = deepcopy(report)
        changed["rankings"][0]["score"] = 99
        self.assertIn("overlay_feature_mismatch", check(r=changed)["featureIssues"])
        self.assertIn("newsPicks_mismatch", check(f={**forecast, "newsPicks": ["B"]})["featureIssues"])
        self.assertIn("signal_before_actual_issuance_day", check(f={**forecast, "signalDate": "2026-10-11"})["featureIssues"])
        invalid_outcome = check(f={**forecast, "status": "resolved", "entryDate": "2026-10-12", "exitDate": "2026-10-20"})
        self.assertIn("entry_not_after_actual_issuance_day", invalid_outcome["outcomeIssues"])
        self.assertIn("invalid_or_future_exit", invalid_outcome["outcomeIssues"])
        self.assertFalse(invalid_outcome["prospectiveEligible"])
        later = deepcopy(report)
        later["generatedAt"] = "2026-10-12T02:00:00Z"
        later["rankings"][0]["score"] = 999
        joined = audit_bundle([forecast], {"news": snapshot}, {"original": report, "later": later}, seen, as_of)
        self.assertEqual(joined["forecasts"][0]["predictions"], clean["predictions"])
        self.assertEqual(joined["summary"]["laterRefreshes"], 1)
        self.assertIn("missing_or_ambiguous_issuance_report", check(r=later)["featureIssues"])
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            # Legacy evidence is now preserved, even if future bars are supplied.
            from research.engine import resolve_ledger
            ledger_path = root / "forecasts.json"
            ledger_path.write_text(json.dumps([forecast]))
            ledger_hash = sha(ledger_path)
            dates = pd.bdate_range("2026-10-13", periods=HORIZON + 1)
            frames = {s: pd.DataFrame({"open": np.arange(len(dates)) + 100.}, index=dates)
                      for s in ["A", "B"]}
            premature = resolve_ledger(root, frames)[0]
            self.assertEqual(premature["status"], "pending")
            self.assertEqual(sha(ledger_path), ledger_hash)
            inputs = []
            for role, payload in [("ledger", [forecast]), ("news", snapshot),
                                  ("report", report), ("first_seen", seen)]:
                path = root / (role + ".json")
                path.write_text(json.dumps(payload))
                inputs.append({"role": role, "path": str(path), "sha256": sha(path)})
            protocol = root / "protocol.json"
            protocol.write_text(json.dumps({"version": VERSION, "frozenAt": as_of,
                "auditAsOf": as_of, "engineSha256": sha(Path(__file__).with_name("engine.py")),
                "parentCommit": "offline-fixture", "inputs": inputs, "outputDir": str(root / "run")}))
            result = run_audit(protocol)
            self.assertEqual(result["summary"]["reconstructable"], 1)
            self.assertEqual(result["summary"]["verifiedProspective"], 0)
            with patch("research.news_audit.audit_bundle", side_effect=AssertionError("Unexpected repeat audit")):
                self.assertEqual(run_audit(protocol), result)
            self.assertTrue(all(sha(i["path"]) == i["sha256"] for i in inputs))
            for path in (root / "run").iterdir():
                self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            # Input and result edits fail closed, rather than recomputing clean evidence.
            source = Path(inputs[1]["path"])
            original = source.read_bytes()
            source.write_text("{}")
            with self.assertRaisesRegex(ValueError, "Input hash"):
                run_audit(protocol)
            source.write_bytes(original)
            (root / "run/audit.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "lineage/hash"):
                run_audit(protocol)

    def test_architecture_training_allocation_replay_and_immutable_export(self):
        # One spanning model workflow covers financial timing, future-data
        # selection, private artifacts and unchanged-run reuse, not tree internals.
        from research.architecture import (SPEC, BOOKS, architecture_panel, fit_at,
                                           allocate, run_experiment, sha)
        from research.normalized import (TARGET, BOOKS as NORMALIZED_BOOKS, normalize_panel,
                                         fit_scaled, run_experiment as run_normalized)
        from research.payoff import (SPEC as PAYOFF_SPEC, BOOKS as PAYOFF_BOOKS, OUTPUTS,
                                     fit_payoff, reference_cost, code_hashes,
                                     run_experiment as run_payoff)
        dates = pd.bdate_range("2020-01-01", periods=960)
        frames = {}
        for i in range(5):
            rng = np.random.default_rng(400 + i)
            close = 100 * np.exp(np.cumsum(rng.normal(.0003, .012, len(dates))))
            frames[f"FIXTURE{i}"] = pd.DataFrame({"open": close * 1.001,
                "high": close * (1 + rng.uniform(.005, .02, len(dates))),
                "low": close * (1 - rng.uniform(.005, .02, len(dates))), "close": close,
                "volume": rng.integers(50000, 100000, len(dates)), "splits": 0., "dividends": 0.}, index=dates)
        adjusted = {s: f.copy() for s, f in frames.items()}
        adjusted["FIXTURE0"].loc[dates[650], ["open", "high", "low", "close", "volume"]] = np.nan
        build = lambda f: architecture_panel(f, dates, calendar_source="Explicit synthetic test sessions",
                                             price_basis="Synthetic fixture only")
        panel, coverage = build(adjusted)
        self.assertEqual(coverage["holdingSessions"], 20)
        self.assertFalse(panel.investmentEligible.any())
        signal = dates[730]
        before, lineage, _ = fit_at(panel, signal)
        self.assertEqual(lineage["status"], "fitted")
        self.assertLess(lineage["trainLastOutcome"], lineage["calibrationStart"])
        self.assertLess(lineage["calibrationLastOutcome"], str(signal.date()))
        self.assertNotIn("FIXTURE0", before.symbol.tolist())
        self.assertEqual(before.iloc[0].labelEnd, dates[751])
        changed = {s: f.copy() for s, f in adjusted.items()}
        for f in changed.values():
            f.loc[dates[731]:, ["open", "high", "low", "close"]] *= 2
        changed["FIXTURE1"] = changed["FIXTURE1"].drop(dates[731])
        future, _ = build(changed)
        after, later_lineage, _ = fit_at(future, signal)
        columns = ["symbol", "mean", "ridge", "tree", "q10"]
        pd.testing.assert_frame_equal(before[columns], after[columns])
        self.assertEqual(lineage, later_lineage)
        self.assertTrue(pd.isna(after[after.symbol == "FIXTURE1"].iloc[0].actual))
        scaled_panel = normalize_panel(panel)
        ready = panel.featureReady
        np.testing.assert_allclose((scaled_panel.actual * scaled_panel.targetScale)[ready], panel.actual[ready], equal_nan=True)
        self.assertTrue((scaled_panel.loc[ready, "targetScale"] >= .01).all())
        scaled_before, scaled_lineage, _ = fit_scaled(panel, signal)
        scaled_after, later_scaled_lineage, _ = fit_scaled(future, signal)
        pd.testing.assert_frame_equal(scaled_before[columns + ["targetScale"]], scaled_after[columns + ["targetScale"]])
        self.assertEqual(scaled_lineage, later_scaled_lineage)
        np.testing.assert_allclose(scaled_before.actual, panel.loc[scaled_before.index, "actual"], equal_nan=True)
        payoff_before, payoff_lineage, _ = fit_payoff(panel, signal)
        payoff_after, later_payoff_lineage, _ = fit_payoff(future, signal)
        # Critical economic/timing boundary: conditional gain/loss estimates
        # and the calibrated event probability must not see the future path.
        payoff_columns = [k for k in OUTPUTS if k != "q10"]
        pd.testing.assert_frame_equal(payoff_before[payoff_columns], payoff_after[payoff_columns])
        self.assertEqual(payoff_lineage, later_payoff_lineage)
        np.testing.assert_allclose(payoff_before.uncalibratedNetProxy,
            payoff_before.proxyProbability * payoff_before.gain -
            (1 - payoff_before.proxyProbability) * payoff_before.loss)
        np.testing.assert_allclose(payoff_before["mean"] - reference_cost(), payoff_before.netProxy)
        self.assertTrue(payoff_before.proxyProbability.between(0, 1).all())
        self.assertTrue((payoff_before[["gain", "loss"]] >= 0).all().all())
        for key in ["trainLastOutcome", "calibrationStart", "calibrationLastOutcome"]:
            self.assertEqual(payoff_lineage[key], lineage[key])
        payoff_before["q10"], payoff_after["q10"] = before.q10, after.q10
        for allocator in PAYOFF_BOOKS.values():
            self.assertEqual(allocate(payoff_before, allocator), allocate(payoff_after, allocator))
        for book in ["dual_head", "ensemble_mean_only", "ridge_mean_only", "trend"]:
            weights = allocate(before, book)
            self.assertEqual(weights, allocate(after, book))
            self.assertLessEqual(sum(weights.values()), 1)
            self.assertTrue(all(0 < w <= .25 for w in weights.values()))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol = {"architecture": SPEC, "books": BOOKS, "signalStride": 20,
                "frozenAt": datetime.now(timezone.utc).isoformat(), "calendarExcluded": [],
                "calendarSource": "Synthetic dates", "priceBasis": "Synthetic prices",
                "start": str(dates[700].date()), "end": str(dates[950].date()),
                "lots": dict.fromkeys(frames, 1), "capital": 100000.,
                "slippageScenarios": [0., 5., 10.], "primarySlippage": 5., "evidenceLimits": ["Fixture only"]}
            for key, source in [("featureInputs", adjusted), ("executionInputs", frames)]:
                protocol[key] = []
                for s, frame in source.items():
                    p = root / f"{key}-{s}.json"
                    rows = frame.reset_index(names="date")
                    rows["date"] = rows.date.dt.strftime("%Y-%m-%d")
                    p.write_text(rows.to_json(orient="records"))
                    protocol[key].append({"path": str(p), "sha256": sha(p), "symbol": s})
            p = root / "protocol.json"
            p.write_text(json.dumps(protocol))
            result = run_experiment(p)
            self.assertGreater(result["fitCount"], 0)
            self.assertFalse(result["livePromotionAllowed"])
            self.assertFalse(result["cacheHit"])
            self.assertEqual(set(result["comparison"]["5.0"]), set(BOOKS))
            self.assertEqual(result["comparison"]["5.0"]["cash"]["ending"], 100000.)
            forecasts = json.loads((root / "run/forecasts.json").read_text())
            self.assertEqual(len(forecasts), result["scheduledDecisions"] * len(frames))
            self.assertTrue(any(r["mean"] is not None and r["actual"] is None for r in forecasts))
            for artifact in (root / "run").iterdir():
                self.assertEqual(artifact.stat().st_mode & 0o777, 0o600)
            with patch("research.architecture.fit_at", side_effect=AssertionError("Unchanged run refitted")):
                self.assertTrue(run_experiment(p)["cacheHit"])
            # Reuse the frozen original books; change only target units. Verify
            # a different observed result cannot silently replace that parent.
            original_completion = root / "run/completed.json"
            original_hash = sha(original_completion)
            normalized_root = root / "normalized"
            normalized_root.mkdir()
            normalized_protocol = normalized_root / "protocol.json"
            normalized_protocol.write_text(json.dumps({"target": TARGET, "books": list(NORMALIZED_BOOKS),
                "frozenAt": datetime.now(timezone.utc).isoformat(),
                "referenceProtocol": {"path": str(p), "sha256": sha(p)},
                "referenceCompleted": {"path": str(original_completion), "sha256": original_hash}}))
            normalized_result = run_normalized(normalized_protocol)
            self.assertFalse(normalized_result["cacheHit"])
            self.assertFalse(normalized_result["livePromotionAllowed"])
            self.assertEqual(normalized_result["fitCount"], result["fitCount"])
            self.assertEqual(normalized_result["predictionMetrics"]["count"], result["predictionMetrics"]["count"])
            for slip, books in result["comparison"].items():
                for book, expected in books.items():
                    self.assertEqual(normalized_result["comparison"][slip][book], expected)
            self.assertEqual(original_hash, sha(original_completion))
            normalized_predictions = json.loads((normalized_root / "run/forecasts.json").read_text())
            self.assertEqual(len(normalized_predictions), len(forecasts))
            for artifact in (normalized_root / "run").iterdir():
                self.assertEqual(artifact.stat().st_mode & 0o777, 0o600)
            with patch("research.normalized.fit_scaled", side_effect=AssertionError("Repeated normalized fit")):
                self.assertTrue(run_normalized(normalized_protocol)["cacheHit"])
            payoff_root = root / "payoff"
            payoff_root.mkdir()
            plan = payoff_root / "PLAN.json"
            plan.write_text(json.dumps({"fixture": True, "spec": PAYOFF_SPEC}))
            payoff_protocol = payoff_root / "protocol.json"
            completion = normalized_root / "run/completed.json"
            payoff_protocol.write_text(json.dumps({
                "frozenAt": datetime.now(timezone.utc).isoformat(), "spec": PAYOFF_SPEC,
                "books": list(PAYOFF_BOOKS), "code": code_hashes(),
                "plan": {"path": str(plan), "sha256": sha(plan)},
                "references": {
                    "original": {"protocol": {"path": str(p), "sha256": sha(p)},
                                 "completed": {"path": str(original_completion), "sha256": original_hash}},
                    "normalized": {"protocol": {"path": str(normalized_protocol), "sha256": sha(normalized_protocol)},
                                   "completed": {"path": str(completion), "sha256": sha(completion)}}}}))
            with patch("pickle.load", side_effect=AssertionError("Baseline pickle loaded")), \
                 patch("research.architecture.fit_at", side_effect=AssertionError("Baseline refitted")):
                payoff_result = run_payoff(payoff_protocol)
            self.assertFalse(payoff_result["cacheHit"])
            self.assertFalse(payoff_result["livePromotionAllowed"])
            self.assertEqual(payoff_result["fitCount"], result["fitCount"])
            self.assertTrue(payoff_result["predictiveAcceptanceChecks"]["unchanged_downside"])
            self.assertTrue(payoff_result["predictiveAcceptanceChecks"]["same_scored_coverage"])
            for slip, books in normalized_result["comparison"].items():
                for book, expected in books.items():
                    self.assertEqual(payoff_result["comparison"][slip][book], expected)
            payoff_rows = json.loads((payoff_root / "run/forecasts.json").read_text())
            self.assertTrue(any(r["mean"] is not None and r["actual"] is None for r in payoff_rows))
            for artifact in (payoff_root / "run").iterdir():
                self.assertEqual(artifact.stat().st_mode & 0o777, 0o600)
            with patch("research.payoff.fit_payoff", side_effect=AssertionError("Repeated payoff fit")):
                self.assertTrue(run_payoff(payoff_protocol)["cacheHit"])
            artifact = payoff_root / "run/decisions.json"
            saved = artifact.read_bytes()
            artifact.write_text("{}")
            with self.assertRaisesRegex(ValueError, "Completed artifact hash"):
                run_payoff(payoff_protocol)
            artifact.write_bytes(saved)
            # Reject incomplete conditional targets rather than silently
            # replacing a blocked model with a profitable-looking fallback.
            one_class = panel.copy()
            one_class["actual"] = .05
            blocked, reason, models = fit_payoff(one_class, signal)
            self.assertIsNone(blocked)
            self.assertIsNone(models)
            self.assertEqual(reason["status"], "insufficient_payoff_classes")
            forecasts_path = root / "run/forecasts.json"
            original = forecasts_path.read_bytes()
            forecasts_path.write_text("[]")
            with self.assertRaisesRegex(ValueError, "artifact hash"):
                run_experiment(p)
            with self.assertRaisesRegex(ValueError, "Reference artifact hash"):
                run_normalized(normalized_protocol)
            with self.assertRaisesRegex(ValueError, "Reference artifact hash"):
                run_payoff(payoff_protocol)
            forecasts_path.write_bytes(original)
            source_path = Path(protocol["featureInputs"][0]["path"])
            source_path.write_text("[]")
            with self.assertRaisesRegex(ValueError, "Source hash"):
                run_experiment(p)
            with self.assertRaisesRegex(ValueError, "Source hash"):
                run_payoff(payoff_protocol)

    def test_session_data_readiness_leakage_and_export(self):
        # Missing-data selection and label timing are critical financial
        # boundaries. Test source -> readiness/labels -> local artifact together.
        from research.engine import feature_panel, atomic_json
        from research.panel import session_panel, panel_payload, STOCK_FEATURES
        source = fixture_loader()
        frames = {s: source(s, 3, None).iloc[:460] for s in SYMBOLS[:2]}
        sessions = next(iter(frames.values())).index
        build = lambda f: session_panel(f, sessions, calendar_source="Explicit offline fixture sessions",
                                         price_basis="Synthetic adjusted research prices")
        base, coverage = build(frames)
        self.assertEqual(coverage["featureReadyRows"], (460 - 199) * 2)
        legacy = feature_panel(frames)[0].set_index(["symbol", "date"])
        ready = base[base.featureReady].set_index(["symbol", "date"])
        np.testing.assert_allclose(ready[STOCK_FEATURES], legacy.loc[ready.index, STOCK_FEATURES])
        short = {**frames, "NEW": frames[SYMBOLS[0]].iloc[-80:]}
        with_new, _ = build(short)
        pd.testing.assert_frame_equal(base, with_new[with_new.symbol != "NEW"].reset_index(drop=True))
        self.assertFalse(with_new[with_new.symbol == "NEW"].featureReady.any())
        missing = {**frames, SYMBOLS[1]: frames[SYMBOLS[1]].drop(sessions[230])}
        sparse, sparse_coverage = build(missing)
        pd.testing.assert_frame_equal(base[base.symbol == SYMBOLS[0]], sparse[sparse.symbol == SYMBOLS[0]])
        b = sparse[sparse.symbol == SYMBOLS[1]].set_index("date")
        self.assertTrue(b.loc[sessions[229], "featureReady"])
        self.assertEqual(b.loc[sessions[229], "labelStatus"], "unavailable_future_path")
        self.assertEqual(b.loc[sessions[229], "entryDate"], sessions[230])
        self.assertEqual(b.loc[sessions[229], "labelEnd"], sessions[235])
        self.assertTrue(pd.isna(b.loc[sessions[229], "actual"]))
        self.assertTrue(pd.isna(b.loc[sessions[230], "open"]))
        self.assertFalse(b.loc[sessions[231:430], "featureReady"].any())
        self.assertTrue(b.loc[sessions[430], "featureReady"])
        self.assertTrue(b.iloc[-1].featureReady)
        self.assertEqual(b.iloc[-1].labelStatus, "pending_future_sessions")
        changed = {s: f.copy() for s, f in frames.items()}
        changed[SYMBOLS[0]].loc[sessions[300]:, ["open", "high", "low", "close"]] *= 1.5
        future, _ = build(changed)
        cols = ["date", "symbol", *STOCK_FEATURES, "featureReady", "featureStatus"]
        pd.testing.assert_frame_equal(base.loc[base.date < sessions[300], cols],
                                      future.loc[future.date < sessions[300], cols])
        damaged = {s: f.copy() for s, f in frames.items()}
        damaged[SYMBOLS[1]].loc[sessions[250], "high"] = 1
        damaged[SYMBOLS[1]].loc[sessions[450], "volume"] = 0
        damaged["DUPLICATE"] = pd.concat([frames[SYMBOLS[0]], frames[SYMBOLS[0]].iloc[-1:]])
        quarantine, details = build(damaged)
        statuses = {s["symbol"]: s for s in details["securities"]}
        self.assertEqual(statuses[SYMBOLS[1]]["invalidRows"], [str(sessions[250].date())])
        self.assertIsNotNone(statuses["DUPLICATE"]["securityQuarantine"])
        self.assertEqual(statuses["DUPLICATE"]["featureReadyRows"], 0)
        q = quarantine[quarantine.symbol == SYMBOLS[1]].set_index("date")
        self.assertEqual(q.loc[sessions[450], "featureStatus"], "zero_volume")
        self.assertEqual(q.loc[sessions[449], "labelStatus"], "unavailable_future_path")
        self.assertFalse(q.loc[sessions[449], "featureReady"])
        self.assertFalse(quarantine.investmentEligible.any())
        # Reviewed calendar corrections derive new inputs; first-seen source
        # knowledge and original bars must survive through panel/export.
        from research.reconcile import exclude_closed_sessions
        evidence = b"Offline synthetic closure notice; not an exchange circular"
        digest = hashlib.sha256(evidence).hexdigest()
        observed = datetime.now(timezone.utc)
        notice = {"market": "NSE:CM", "closedSession": str(sessions[230].date()),
                  "publishedDate": str((sessions[230] - timedelta(days=3)).date()),
                  "observedAt": observed.isoformat(), "sourceUrl": "https://example.com/fixture",
                  "sourceSha256": digest, "sourceKind": "captured-excerpt", "reason": "Offline holiday fixture"}
        kwargs = {"market": "NSE:CM", "market_timezone": "Asia/Kolkata",
                  "knowledge_as_of": observed.isoformat(), "sources": {digest: evidence}}
        originals = {s: f.copy(deep=True) for s, f in frames.items()}
        corrected, calendar, ledger = exclude_closed_sessions(frames, sessions, [notice], **kwargs)
        self.assertNotIn(sessions[230], calendar)
        self.assertEqual(len(calendar), len(sessions) - 1)
        for symbol in frames:
            pd.testing.assert_frame_equal(originals[symbol], frames[symbol])
            pd.testing.assert_frame_equal(corrected[symbol], originals[symbol].drop(sessions[230]))
        self.assertEqual(sum(len(s["excludedRows"]) for s in ledger["securities"]), 2)
        again, same_calendar, repeated = exclude_closed_sessions(corrected, calendar, [notice], **kwargs)
        self.assertTrue(same_calendar.equals(calendar))
        self.assertEqual(repeated["notices"][0]["status"], "already_absent")
        for symbol in corrected:
            pd.testing.assert_frame_equal(corrected[symbol], again[symbol])
        resolved_panel, _ = session_panel(corrected, calendar, calendar_source="Reviewed fixture closure",
                                           price_basis="Synthetic adjusted research prices")
        previous = resolved_panel[(resolved_panel.symbol == SYMBOLS[0]) & (resolved_panel.date == sessions[229])].iloc[0]
        self.assertEqual(previous.entryDate, sessions[231])
        self.assertEqual(previous.labelEnd, sessions[236])
        self.assertEqual(previous.labelStatus, "observed")
        for bad_notice, bad_kwargs, message in [
            (notice, {**kwargs, "sources": {digest: b"tampered"}}, "source hash"),
            (notice, {**kwargs, "knowledge_as_of": (observed - timedelta(seconds=1)).isoformat()}, "knowledge cutoff"),
            ({**notice, "market": "NYSE"}, kwargs, "input market"),
        ]:
            with self.assertRaisesRegex(ValueError, message):
                exclude_closed_sessions(frames, sessions, [bad_notice], **bad_kwargs)
        with self.assertRaisesRegex(ValueError, "Duplicate closure"):
            exclude_closed_sessions(frames, sessions, [notice, notice], **kwargs)
        # Official raw/adjusted reconciliation flows through quarantine, rolling
        # readiness, fixed label endpoints and the same private artifact export.
        from research.reconcile import quarantine_bhavcopy_conflicts
        day = sessions[240]
        compared = {s: f.copy() for s, f in frames.items()}
        compared[SYMBOLS[1]].loc[day, "volume"] = 0
        compared.update({"NO_OFFICIAL": frames[SYMBOLS[0]].copy(),
                         "NO_PROVIDER": frames[SYMBOLS[0]].drop(day),
                         "BAD_OFFICIAL": frames[SYMBOLS[0]].copy()})
        mappings = {s: {"symbol": name, "series": "EQ"} for s, name in zip(
            compared, ["ALPHA", "BETA", "ABSENT", "EMPTY", "INVALID"])}
        lines = ["SYMBOL,SERIES,DATE1,OPEN_PRICE,HIGH_PRICE,LOW_PRICE,CLOSE_PRICE,TTL_TRD_QNTY,TURNOVER_LACS"]
        for symbol, name in [(SYMBOLS[0], "ALPHA"), (SYMBOLS[1], "BETA")]:
            row = frames[symbol].loc[day]
            # A uniform factor must not be mistaken for proof of bad prices.
            values = [row[k] * 2 for k in ["open", "high", "low", "close"]]
            lines.append(",".join(map(str, [name, "EQ", day.strftime("%d-%b-%Y"), *values, int(row.volume), 10])))
        lines.append(f"INVALID,EQ,{day:%d-%b-%Y},100,90,80,95,1000,10")
        raw_csv = "\n".join(lines).encode()
        daily_source = {"market": "NSE:CM", "tradingDate": str(day.date()),
                        "observedAt": observed.isoformat(), "url": "https://example.com/offline-bhavcopy",
                        "sha256": hashlib.sha256(raw_csv).hexdigest()}
        daily_kwargs = {"knowledge_as_of": observed.isoformat(), "price_basis": "Offline adjusted fixture"}
        before = {s: f.copy() for s, f in compared.items()}
        isolated, audit = quarantine_bhavcopy_conflicts(compared, mappings, daily_source, raw_csv, **daily_kwargs)
        self.assertEqual(audit["quarantinedRows"], 1)
        statuses = {r["symbol"]: r for r in audit["comparisons"]}
        self.assertEqual(statuses[SYMBOLS[0]]["status"], "compatible_unverified")
        self.assertEqual(statuses[SYMBOLS[0]]["closeScaleDiagnostic"], 0.5)
        self.assertEqual(statuses[SYMBOLS[1]]["reasons"], ["zero_volume_conflict"])
        for symbol, status in [("NO_OFFICIAL", "official_missing"), ("NO_PROVIDER", "provider_missing"),
                               ("BAD_OFFICIAL", "official_invalid")]:
            self.assertEqual(statuses[symbol]["status"], status)
        for symbol in compared:
            pd.testing.assert_frame_equal(compared[symbol], before[symbol])
            if symbol != SYMBOLS[1]:
                pd.testing.assert_frame_equal(isolated[symbol], before[symbol])
        isolated_panel, _ = build(isolated)
        b = isolated_panel[isolated_panel.symbol == SYMBOLS[1]].set_index("date")
        self.assertTrue(b.loc[day, ["open", "high", "low", "close", "volume"]].isna().all())
        self.assertFalse(b.loc[sessions[240:440], "featureReady"].any())
        self.assertTrue(b.loc[sessions[440], "featureReady"])
        self.assertTrue(b.loc[sessions[239], "featureReady"])
        self.assertEqual(b.loc[sessions[239], "labelStatus"], "unavailable_future_path")
        self.assertEqual(b.loc[sessions[239], "entryDate"], day)
        self.assertEqual(b.loc[sessions[239], "labelEnd"], sessions[245])
        pd.testing.assert_frame_equal(base[base.symbol == SYMBOLS[0]].reset_index(drop=True),
                                      isolated_panel[isolated_panel.symbol == SYMBOLS[0]].reset_index(drop=True))
        distorted = {s: f.copy() for s, f in compared.items()}
        distorted[SYMBOLS[0]].loc[day, "high"] *= 1.02
        _, shape_audit = quarantine_bhavcopy_conflicts(distorted, mappings, daily_source, raw_csv, **daily_kwargs)
        shape_record = next(r for r in shape_audit["comparisons"] if r["symbol"] == SYMBOLS[0])
        self.assertIn("ohlc_shape_conflict", shape_record["reasons"])
        for bad_bytes, options, message in [
            (raw_csv + b"tampered", daily_kwargs, "source hash"),
            (raw_csv, {**daily_kwargs, "knowledge_as_of": (observed - timedelta(seconds=1)).isoformat()}, "knowledge cutoff"),
        ]:
            with self.assertRaisesRegex(ValueError, message):
                quarantine_bhavcopy_conflicts(compared, mappings, daily_source, bad_bytes, **options)
        with self.assertRaisesRegex(ValueError, "unambiguous"):
            quarantine_bhavcopy_conflicts(compared, {**mappings, SYMBOLS[1]: mappings[SYMBOLS[0]]},
                                         daily_source, raw_csv, **daily_kwargs)
        with self.assertRaisesRegex(ValueError, "Explicit sorted unique"):
            session_panel(frames, sessions[::-1], calendar_source="fixture", price_basis="fixture")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "session-panel.json"
            atomic_json(path, panel_payload(sparse, sparse_coverage))
            artifact = json.loads(path.read_text())
            self.assertEqual(len(artifact["rows"]), len(sparse))
            missing_bar = next(r for r in artifact["rows"] if r["symbol"] == SYMBOLS[1] and r["date"] == str(sessions[230].date()))
            self.assertIsNone(missing_bar["open"])
            self.assertIsNone(missing_bar["actual"])
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            json.dumps(artifact, allow_nan=False)
            atomic_json(Path(directory) / "correction-ledger.json", ledger)
            self.assertEqual(json.loads((Path(directory) / "correction-ledger.json").read_text()), ledger)
            atomic_json(Path(directory) / "source-audit.json", audit)
            self.assertEqual(json.loads((Path(directory) / "source-audit.json").read_text()), audit)

    def test_whole_share_cash_cost_split_and_daily_replay(self):
        # A spanning execution workflow protects critical money identities,
        # corporate actions and settlement; it is not a narrow estimator test.
        from research.execution import replay, charges
        dates = pd.bdate_range("2025-01-02", periods=5)
        frame = pd.DataFrame({"open": [100., 50., 55., 56., 57.],
                              "close": [100., 50., 55., 56., 57.],
                              "high": [101., 51., 56., 57., 58.],
                              "low": [99., 49., 54., 55., 56.],
                              "volume": 10000, "splits": [0., 2., 0., 0., 0.],
                              "dividends": [0., 1., 0., 0., 0.]}, index=dates)
        window = {"signal": "2025-01-01", "entry": str(dates[0].date()),
                  "exit": str(dates[2].date()), "weights": {"A": 1.}}
        second = {"signal": str(dates[1].date()), "entry": str(dates[2].date()),
                  "exit": str(dates[4].date()), "weights": {"A": 1.}}
        result = replay({"A": frame}, [window, second], 1000., 0.)
        self.assertEqual([(o["side"], o["units"]) for o in result["orders"]],
                         [("buy", 9), ("sell", 18), ("buy", 1), ("sell", 1)])
        self.assertAlmostEqual(charges(10000., "buy"), 23.67406)
        self.assertEqual(result["omittedDividendEntitlements"], 18.)
        self.assertTrue(all(o["units"] == int(o["units"]) for o in result["orders"]))
        for point in result["curve"]:
            self.assertAlmostEqual(point["equity"], point["cash"] + point["unsettled"] + point["invested"])
            self.assertGreaterEqual(point["cash"], 0.)
        inflow = sum(o["price"] * o["units"] * (1 if o["side"] == "sell" else -1) for o in result["orders"])
        self.assertAlmostEqual(result["ending"], 1000. + inflow - result["fees"])
        changed = frame.copy()
        changed.loc[dates[-1], ["open", "close", "high", "low"]] *= 2
        future = replay({"A": changed}, [window, second], 1000., 0.)
        self.assertEqual(result["orders"][:3], future["orders"][:3])
        unaffordable = replay({"A": frame}, [window], 50., 0.)
        self.assertEqual(unaffordable["orders"], [])
        self.assertEqual(unaffordable["ending"], 50.)
        bad = frame.copy(); bad.loc[dates[0], "high"] = 1
        with self.assertRaisesRegex(ValueError, "Invalid OHLC"):
            replay({"A": bad}, [window], 1000.)
        with self.assertRaisesRegex(ValueError, "Overlapping"):
            replay({"A": frame}, [window, window], 1000.)
        from research.rebalance import replay_rebalanced
        flat = frame.copy()
        flat[["open", "high", "low", "close"]] = 100.
        flat[["splits", "dividends"]] = 0.
        decisions = [{"signal": w["signal"], "entry": w["entry"], "weights": w["weights"]} for w in [window, second]]
        retained = replay_rebalanced({"A": flat}, decisions, dates[-1], {"A": 1}, 1000., 0.)
        self.assertEqual([(o["side"], o["units"]) for o in retained["orders"]], [("buy", 9), ("sell", 9)])
        halves = [{**w, "weights": {"A": .5}} for w in [window, second]]
        half_retained = replay_rebalanced({"A": flat}, halves, dates[-1], {"A": 1}, 10000., 0.)
        self.assertLess(half_retained["fees"], replay({"A": flat}, halves, 10000., 0.)["fees"])
        self.assertAlmostEqual(retained["ending"], 1000. - retained["fees"])
        rotation = [decisions[0], {**decisions[1], "weights": {"B": 1.}}]
        rotated = replay_rebalanced({"A": flat, "B": flat}, rotation, dates[-1], {"A": 1, "B": 1}, 1000., 0.)
        self.assertFalse(any(o["symbol"] == "B" for o in rotated["orders"]))
        self.assertEqual(rotated["unfilledTargets"][-1]["filledUnits"], 0)
        self.assertGreater(rotated["curve"][3]["unsettled"], 0)
        lots = replay_rebalanced({"A": flat}, decisions, dates[-1], {"A": 10}, 2000., 0.)
        self.assertTrue(all(o["units"] % 10 == 0 for o in lots["orders"]))
        unaffordable_lot = replay_rebalanced({"A": flat}, decisions, dates[-1], {"A": 10}, 1000., 0.)
        self.assertEqual(unaffordable_lot["orders"], [])
        tiny_change = [decisions[0], {**decisions[1], "weights": {"A": .91}}]
        banded = replay_rebalanced({"A": flat}, tiny_change, dates[-1], {"A": 1}, 10000., 0., .1)
        self.assertGreater(banded["suppressedAdjustments"], 0)
        self.assertEqual(len(banded["orders"]), 2)  # Final liquidation ignores drift band.
        split_retained = replay_rebalanced({"A": frame}, decisions, dates[-1], {"A": 1}, 1000., 0.)
        changed_retained = replay_rebalanced({"A": changed}, decisions, dates[-1], {"A": 1}, 1000., 0.)
        self.assertEqual(split_retained["orders"][:-1], changed_retained["orders"][:-1])
        self.assertEqual(split_retained["omittedDividendEntitlements"], 18.)
        for point in split_retained["curve"]:
            self.assertAlmostEqual(point["equity"], point["cash"] + point["unsettled"] + point["invested"])
        with self.assertRaisesRegex(ValueError, "Explicit positive integer lot"):
            replay_rebalanced({"A": flat}, decisions, dates[-1], {"A": None})
        with self.assertRaisesRegex(ValueError, "Invalid or duplicate decision timing"):
            replay_rebalanced({"A": flat}, [decisions[0], decisions[0]], dates[-1], {"A": 1})
        no_trades = flat.copy(); no_trades.loc[dates[0], "volume"] = 0
        with self.assertRaisesRegex(ValueError, "zero-volume"):
            replay_rebalanced({"A": no_trades}, decisions, dates[-1], {"A": 1})
        # Discovery must not silently promote SME/dummy index securities to the
        # ordinary one-share execution workflow. These are explicit offline CSVs.
        from research.universe import build_registry, daily_snapshot, sme_band_snapshot
        main = "SYMBOL,NAME OF COMPANY,SERIES,DATE OF LISTING,MARKET LOT,ISIN NUMBER\nA,Alpha,EQ,01-JAN-2020,1,INE000A01001\n"
        sme = "SYMBOL,NAME_OF_COMPANY,SERIES,DATE_OF_LISTING,ISIN_NUMBER\nB,Beta,SM,01-Jan-24,INE000B01001\n"
        index = "Company Name,Industry,Symbol,Series,ISIN Code\nAlpha,Capital Goods,A,EQ,INE000A01001\nDummy,Dummy,DUMMY,EQ,DUM00000001\n"
        registry = build_registry(main, sme, {"Microcap fixture": index}, "2025-01-01T00:00:00Z")
        self.assertEqual(registry["coverage"]["uniqueSecurities"], 2)
        self.assertEqual(registry["coverage"]["industryTagged"], 1)
        self.assertEqual(len(registry["coverage"]["unmatchedIndexRows"]), 1)
        self.assertFalse(registry["historicalMembershipVerified"])
        self.assertTrue(all(not s["investmentEligible"] for s in registry["securities"]))
        beta = next(s for s in registry["securities"] if s["name"] == "Beta")
        self.assertIn("Market lot unknown", beta["flags"])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            build_registry(main + main.splitlines()[1] + "\n", sme, {}, "2025-01-01T00:00:00Z")
        daily = "SYMBOL,SERIES,DATE1,OPEN_PRICE,HIGH_PRICE,LOW_PRICE,CLOSE_PRICE,TTL_TRD_QNTY,TURNOVER_LACS\nA,EQ,31-Dec-2024,100,105,95,102,100,0.1\nB,SM,31-Dec-2024,100,1,95,102,100,0.1\nC,EQ,31-Dec-2024,100,105,95,102,100,0.1\n"
        snapshot = daily_snapshot(daily, registry, "2024-12-31", "2025-01-01T00:00:00Z")
        self.assertEqual(snapshot["coverage"]["matchedSecurities"], 1)
        self.assertEqual(snapshot["coverage"]["quarantinedRows"], 1)
        self.assertEqual(snapshot["coverage"]["unmatchedRows"], 1)
        self.assertEqual(snapshot["bars"][0]["turnoverInr"], 10000)
        self.assertFalse(registry["securities"][0]["investmentEligible"])
        with self.assertRaisesRegex(ValueError, "different trading date"):
            daily_snapshot(daily, registry, "2024-12-30", "2025-01-01T00:00:00Z")
        with self.assertRaisesRegex(ValueError, "follow the trading date"):
            daily_snapshot(daily, registry, "2025-01-02", "2025-01-01T00:00:00Z")
        band_csv = "Symbol,Series,Name,Band,Remarks\nB,SM,Beta,5,-\nB,ST,Beta,2,Restriction\n"
        bands = sme_band_snapshot(band_csv, registry, "sme_bands_complete_31122024.csv", "2024-12-31", "2025-01-01T00:00:00Z")
        self.assertEqual(bands["coverage"]["matchedSecurities"], 1)
        self.assertEqual(bands["coverage"]["unmatchedRows"], 1)
        self.assertEqual(bands["bands"][1]["remarks"], "Restriction")
        self.assertIsNone(beta["listings"][0]["marketLot"])
        self.assertFalse(beta["investmentEligible"])
        with self.assertRaisesRegex(ValueError, "filename/date mismatch"):
            sme_band_snapshot(band_csv, registry, "sme_bands_complete_01012025.csv", "2024-12-31", "2025-01-01T00:00:00Z")
        with self.assertRaisesRegex(ValueError, "Future"):
            sme_band_snapshot(band_csv, registry, "sme_bands_complete_02012025.csv", "2025-01-02", "2025-01-01T00:00:00Z")
        with self.assertRaisesRegex(ValueError, "Invalid SME band percent"):
            sme_band_snapshot(band_csv.replace(",5,", ",nan,"), registry, "sme_bands_complete_31122024.csv", "2024-12-31", "2025-01-01T00:00:00Z")
        # Dated exchange identity must survive independently of current survivors.
        # ZIP/date/scope controls protect the source-to-private-artifact workflow.
        from research.dated_universe import udiff_snapshot
        from research.engine import atomic_json
        member = "BhavCopy_NSE_CM_0_0_0_20241231_F_0000.csv"
        header = "TradDt,BizDt,Sgmt,Src,FinInstrmTp,FinInstrmId,ISIN,TckrSymb,SctySrs,OpnPric,HghPric,LwPric,ClsPric,TtlTradgVol,TtlTrfVal,SsnId,NewBrdLotQty\n"
        good = "2024-12-31,2024-12-31,CM,NSE,STK,1,INE009A01021,A,EQ,100,105,95,102,100,10200,F1,\n"
        historical = "2024-12-31,2024-12-31,CM,NSE,STK,2,INE002A01018,C,SM,100,105,95,102,2000,204000,F1,1000\n"
        bad_price = "2024-12-31,2024-12-31,CM,NSE,STK,3,IN0020200104,D,EQ,100,1,95,102,100,10200,F1,1\n"
        bad_isin = "2024-12-31,2024-12-31,CM,NSE,STK,4,INE009A01020,E,EQ,100,105,95,102,100,10200,F1,1\n"
        def zipped(body, name=member):
            stream = io.BytesIO()
            with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                archive.writestr(name, body)
            return stream.getvalue()
        payload = zipped(header + good + historical + bad_price + bad_isin)
        dated = udiff_snapshot(payload, member + ".zip", "2024-12-31", "2025-01-01T00:00:00Z")
        self.assertEqual(dated["coverage"]["auditedRows"], 2)
        self.assertEqual(dated["coverage"]["quarantinedRows"], 2)
        self.assertEqual(dated["coverage"]["datedIdentities"], 3)
        self.assertEqual(next(r for r in dated["identities"] if r["symbol"] == "D")["priceStatus"], "quarantined")
        self.assertEqual(dated["coverage"]["missingReportedLots"], 1)
        self.assertEqual(dated["sourceSha256"], hashlib.sha256(payload).hexdigest())
        old = next(r for r in dated["bars"] if r["symbol"] == "C")
        self.assertEqual(old["isin"], "INE002A01018")
        self.assertEqual(old["reportedBoardLot"], 1000)
        self.assertNotIn("C", {l["symbol"] for s in registry["securities"] for l in s["listings"]})
        self.assertTrue(all(not r["investmentEligible"] for r in dated["bars"]))
        for data, name, observed, message in [
            (payload, member.replace("20241231", "20241230") + ".zip", "2025-01-01T00:00:00Z", "filename/date"),
            (payload, member + ".zip", "2024-12-30T00:00:00Z", "precede"),
            (zipped(header + good, "../" + member), member + ".zip", "2025-01-01T00:00:00Z", "matching"),
            (zipped("0" * 10_000_001), member + ".zip", "2025-01-01T00:00:00Z", "oversized"),
            (zipped(header + good + good), member + ".zip", "2025-01-01T00:00:00Z", "Duplicate"),
            (zipped((header + good).replace(",NSE,", ",BSE,")), member + ".zip", "2025-01-01T00:00:00Z", "scope"),
            (zipped((header + good).replace("2024-12-31", "2024-12-30")), member + ".zip", "2025-01-01T00:00:00Z", "different"),
        ]:
            with self.assertRaisesRegex(ValueError, message):
                udiff_snapshot(data, name, "2024-12-31", observed)
        # Source archives -> dated histories -> explicit-session readiness ->
        # private export. Identity/timing/gap distinctions are critical to avoid
        # survivorship and false continuity in later financial calculations.
        from research.dated_history import udiff_histories
        from research.panel import session_panel, panel_payload
        first_archive = {"payload": payload, "filename": member + ".zip", "tradingDate": "2024-12-31",
                         "observedAt": "2025-01-01T00:00:00Z", "sha256": hashlib.sha256(payload).hexdigest(),
                         "sourceUrl": "https://example.com/offline-fixture"}
        next_member = member.replace("20241231", "20250102")
        next_body = (header + good.replace(",STK,1,", ",STK,10,").replace(",A,EQ,", ",AA,EQ,")
                     .replace(",F1,\n", ",F1,5\n") + historical.replace(",C,SM,", ",C,ST,")
                     + bad_price.replace(",D,EQ,", ",A,EQ,").replace(",100,1,95,", ",100,105,95,"))
        next_bytes = zipped(next_body.replace("2024-12-31", "2025-01-02"), next_member)
        next_archive = {**first_archive, "payload": next_bytes, "filename": next_member + ".zip",
                        "tradingDate": "2025-01-02", "observedAt": "2025-01-03T00:00:00Z",
                        "sha256": hashlib.sha256(next_bytes).hexdigest()}
        days = ["2024-12-31", "2025-01-01", "2025-01-02"]
        build_history = lambda sources: udiff_histories(sources, days, knowledge_as_of="2025-01-03T00:00:00Z",
                                                      calendar_source="Explicit offline fixture dates")
        histories, ledger = build_history([next_archive, first_archive])
        again_frames, again_ledger = build_history([first_archive, next_archive])
        self.assertEqual(ledger, again_ledger)
        for key, frame in histories.items():
            pd.testing.assert_frame_equal(frame, again_frames[key])
        self.assertEqual(ledger["missingFileDates"], ["2025-01-01"])
        self.assertEqual(ledger["coverage"]["statusCounts"], {
            "observed": 5, "quarantined_price": 1, "missing_file": 4, "not_observed_in_file": 2})
        self.assertEqual(ledger["coverage"]["histories"], 4)
        self.assertEqual(ledger["aliasIdentityConflicts"][0]["symbol"], "A")
        self.assertEqual(ledger["multipleSeriesIsins"], [{"isin": "INE002A01018", "series": ["SM", "ST"]}])
        self.assertTrue(any(c["changes"].get("reportedBoardLot") == {"before": None, "after": 5}
                            for c in ledger["observedChanges"]))
        self.assertEqual(ledger["sources"][0]["quarantined"], dated["quarantined"])
        panel, coverage = session_panel(histories, pd.DatetimeIndex(days),
                                        calendar_source=ledger["calendarSource"], price_basis="Unadjusted offline fixture")
        self.assertEqual(coverage["featureReadyRows"], 0)
        self.assertEqual(len(panel), 12)
        self.assertFalse(panel.investmentEligible.any())
        self.assertEqual(panel.featureStatus.eq("invalid_ohlcv").sum(), 1)
        self.assertEqual(panel.featureStatus.eq("missing_observation").sum(), 6)
        self.assertTrue(panel.loc[panel.date.eq("2025-01-01"), "close"].isna().all())
        ambiguous_bytes = zipped(header + good + good.replace(",STK,1,", ",STK,9,").replace(",A,EQ,", ",Z,EQ,"))
        for sources, message in [
            ([first_archive, first_archive], "duplicate/revised"),
            ([{**first_archive, "sha256": "0" * 64}], "hash differs"),
            ([{**first_archive, "observedAt": "2025-01-04T00:00:00Z"}], "knowledge cutoff"),
            ([{**first_archive, "tradingDate": "2024-12-30"}], "Outside requested"),
            ([{**first_archive, "payload": ambiguous_bytes, "sha256": hashlib.sha256(ambiguous_bytes).hexdigest()}], "Ambiguous same-day"),
        ]:
            with self.assertRaisesRegex(ValueError, message):
                build_history(sources)
        # Reviewed official facts may explain a symbol change but cannot confer
        # price continuity, resolve lots/IDs, fill gaps or backdate availability.
        from research.reconcile import reconcile_symbol_changes
        history_bytes = json.dumps(ledger, sort_keys=True).encode()
        history_hash = hashlib.sha256(history_bytes).hexdigest()
        excerpt = b"Offline reviewed symbol-change fixture, not an exchange notice."
        excerpt_hash = hashlib.sha256(excerpt).hexdigest()
        notice = {"market": "NSE:CM", "oldSymbol": "A", "newSymbol": "AA", "circularId": "fixture-1",
                  "publishedDate": "2024-12-30", "effectiveDate": "2025-01-02", "observedAt": "2025-01-03T00:00:00Z",
                  "sourceKind": "captured-excerpt", "sourceUrl": "https://example.com/offline-fixture",
                  "sourceSha256": excerpt_hash}
        audit_aliases = lambda notes: reconcile_symbol_changes(
            history_bytes, history_hash, notes, {excerpt_hash: excerpt}, knowledge_as_of="2025-01-03T00:00:00Z")
        evidence = audit_aliases([notice, {**notice, "oldSymbol": "X", "newSymbol": "Y", "circularId": "unused"}])
        self.assertEqual(evidence["coverage"]["statusCounts"], {"supported": 1, "unresolved": 1})
        self.assertEqual(evidence["coverage"]["unreviewedOtherChanges"], 1)
        self.assertEqual(len(evidence["unusedNotices"]), 1)
        self.assertTrue(next(r for r in evidence["changes"] if r["status"] == "supported")["effectiveDateObserved"])
        self.assertTrue(all(not r["priceContinuityVerified"] and not r["investmentEligible"] for r in evidence["changes"]))
        gap_evidence = audit_aliases([{**notice, "effectiveDate": "2025-01-01"}])
        self.assertFalse(next(r for r in gap_evidence["changes"] if r["status"] == "supported")["effectiveDateObserved"])
        self.assertIn("date_conflict", audit_aliases([{**notice, "effectiveDate": "2024-12-31"}])["coverage"]["statusCounts"])
        for notes, message in [
            ([notice, notice], "duplicate"),
            ([{**notice, "sourceSha256": "0" * 64}], "hash"),
            ([{**notice, "observedAt": "2025-01-04T00:00:00Z"}], "chronology"),
            ([{**notice, "publishedDate": "2025-01-03"}], "chronology"),
            ([{**notice, "market": "BSE:CM"}], "market"),
            ([{**notice, "sourceKind": "unknown"}], "source kind"),
        ]:
            with self.assertRaisesRegex(ValueError, message):
                audit_aliases(notes)
        with self.assertRaisesRegex(ValueError, "History hash"):
            reconcile_symbol_changes(history_bytes + b" ", history_hash, [], {}, knowledge_as_of="2025-01-03T00:00:00Z")
        with self.assertRaisesRegex(ValueError, "History knowledge"):
            reconcile_symbol_changes(history_bytes, history_hash, [], {}, knowledge_as_of="2025-01-02T00:00:00Z")
        self.assertEqual(json.dumps(ledger, sort_keys=True).encode(), history_bytes)
        # Reference-close equality explains a series-specific observation; it
        # must not turn a quarantined T0 bar into an ordinary executable close.
        from research.reconcile import audit_t0_reference_closes
        t0 = good.replace(",STK,1,", ",STK,7,").replace(",EQ,", ",T0,")
        t0 = t0.replace(",100,105,95,102,", ",110,110,110,102.0000,")
        def audit_series(body):
            data = zipped(header + body)
            source = {**first_archive, "payload": data, "sha256": hashlib.sha256(data).hexdigest()}
            return audit_t0_reference_closes(source, knowledge_as_of="2025-01-03T00:00:00Z"), source
        series_evidence, series_source = audit_series(good + t0)
        record = series_evidence["records"][0]
        self.assertEqual(record["status"], "matched_reference_close")
        self.assertEqual(record["identity"]["priceStatus"], "quarantined")
        self.assertFalse(record["ordinaryPriceContractSupported"])
        self.assertFalse(record["investmentEligible"] or record["historicalRuleApplicabilityVerified"])
        self.assertEqual(len(series_evidence["sourceQuarantines"]), 1)
        for body, status in [
            (good + t0.replace("102.0000", "102.0000000000000001"), "close_mismatch"),
            (good.replace(",EQ,", ",BE,") + t0, "reference_missing"),
            (good.replace(",F1,", ",F2,") + t0, "reference_missing"),
            (good + good.replace(",STK,1,", ",STK,8,").replace(",A,EQ,", ",Z,EQ,") + t0,
             "reference_ambiguous"),
            (good.replace(",A,EQ,", ",Z,EQ,") + t0, "symbol_conflict"),
            (good.replace(",100,105,95,", ",100,1,95,") + t0, "reference_quarantined"),
            (good + t0.replace("102.0000", "NaN"), "invalid_close"),
        ]:
            self.assertEqual(audit_series(body)[0]["records"][0]["status"], status)
        with self.assertRaisesRegex(ValueError, "hash differs"):
            audit_t0_reference_closes({**series_source, "sha256": "0" * 64}, knowledge_as_of="2025-01-03T00:00:00Z")
        with self.assertRaisesRegex(ValueError, "knowledge cutoff"):
            audit_t0_reference_closes(series_source, knowledge_as_of="2024-12-31T00:00:00Z")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dated-universe.json"
            atomic_json(path, dated)
            self.assertEqual(json.loads(path.read_text()), dated)
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            artifact = {"history": ledger, "panel": panel_payload(panel, coverage),
                        "aliasEvidence": evidence, "seriesEvidence": series_evidence}
            history_path = Path(directory) / "dated-history.json"
            atomic_json(history_path, artifact)
            self.assertEqual(json.loads(history_path.read_text()), artifact)
            self.assertEqual(history_path.stat().st_mode & 0o777, 0o600)

    def test_train_export_repeat_and_resolve_future_outcomes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            from research import prospective, price_analysis
            run = lambda loader: run_research(CONFIG, root, loader=loader, news_loader=fixture_news, scorer=fixture_scores)
            first = run(fixture_loader())
            issue_path = next((root / prospective.VERSION / "issues").glob("*.json"))
            issue_bytes = issue_path.read_bytes()
            issue = prospective.read_seal(issue_path)
            self.assertEqual(len(issue["features"]), 4)
            self.assertTrue(all("date" in r and "actual" not in r for r in issue["features"]))
            self.assertTrue(all(a["contentSha256"] for a in issue["articles"]))
            self.assertFalse((root / "forecasts.json").exists())
            self.assertEqual(len(first["rankings"]), 4)
            self.assertEqual(len(first["backtest"]["metrics"]), 6)
            self.assertEqual(len(first["articles"]), 4)
            for fold in first["backtest"]["folds"]:
                self.assertLess(fold["lastTrainingOutcome"], fold["testStart"])
                self.assertLess(fold["trainEnd"], fold["testStart"])
            audit = first["audit"]
            self.assertEqual(len(audit["metrics"]), 9)
            self.assertEqual(len(audit["costStress"]), 5)
            policy = audit["policy"]
            self.assertEqual(policy["version"], "absolute-risk-1")
            self.assertEqual(len(policy["metrics"]), 4)
            self.assertEqual(policy["periods"] + 1, len(policy["curve"]))
            self.assertLessEqual(policy["activeWindows"], policy["periods"])
            for m in policy["metrics"]:
                self.assertAlmostEqual(policy["curve"][-1][m["name"]] - 1, m["totalReturn"])
            self.assertAlmostEqual(sum(a["weight"] for a in policy["latest"]["allocations"]), policy["latest"]["exposure"])
            self.assertTrue(all(a["weight"] <= .4 for a in policy["latest"]["allocations"]))
            self.assertAlmostEqual(policy["latest"]["exposure"] + policy["latest"]["cashWeight"], 1)
            self.assertAlmostEqual(sum(a["weight"] for a in audit["latest"]["allocations"]), audit["latest"]["exposure"])
            for fold in audit["folds"]:
                self.assertLess(fold["innerTrainLastOutcome"], fold["tuningStart"])
                self.assertLess(fold["tuningLastOutcome"], fold["calibrationStart"])
                self.assertLess(fold["finalTrainLastOutcome"], fold["calibrationStart"])
                self.assertLess(fold["calibrationLastOutcome"], fold["testStart"])
                self.assertAlmostEqual(sum(fold["weights"].values()), 1)
                self.assertLess(fold["tuningLastOutcome"], fold["selectionStart"])
                self.assertEqual(len(fold["selectionBlocks"]), 2)
                for block in fold["selectionBlocks"]:
                    self.assertLess(block["lastTrainingOutcome"], block["start"])
                    self.assertLess(block["lastOutcome"], block["end"])
                    self.assertGreater(block["periods"], 0)
                for candidate in fold["candidates"]:
                    scores = [next(m["selectionScore"] for m in b["metrics"] if m["name"] == candidate["name"]) for b in fold["selectionBlocks"]]
                    self.assertEqual(candidate["selectionScore"], min(scores))
            # Perturb future evaluation prices: the first fold's development choices
            # must remain identical. This guards the central financial leakage risk.
            source = fixture_loader()
            boundary = pd.Timestamp(audit["folds"][0]["testStart"])
            def perturbed(symbol, years, root):
                frame = source(symbol, years, root)
                mask = frame.index >= boundary
                frame.loc[mask, ["open", "high", "low", "close"]] *= np.linspace(1, 1.2, int(mask.sum()))[:, None]
                return frame
            changed = run(perturbed)
            for field in ["selected", "weights", "candidates", "intervalRadius", "selectionBlocks"]:
                self.assertEqual(changed["audit"]["folds"][0][field], audit["folds"][0][field])
            # Critical financial property: future outcome labels cannot affect sizing;
            # relative scores cannot force investment through a negative market gate.
            from research.policy import allocation
            from research.engine import feature_panel
            panel = feature_panel({s: source(s, 3, root) for s in SYMBOLS})[0]
            day = panel[panel.date == panel.date.max()].copy()
            day["Cross-sectional Ridge"] = np.arange(len(day))
            day["return60"] = .1; day["trend200"] = .1; day["marketTrend"] = .1
            before = allocation(day, 2)
            self.assertGreater(before[1]["exposure"], 0)
            day["actual"] = 1000; day["target"] = -1000
            self.assertEqual(allocation(day, 2), before)
            day["marketTrend"] = -1
            self.assertEqual(allocation(day, 2)[0], {})
            self.assertEqual(allocation(day, 2)[1]["exposure"], 0)
            # The absolute entry hypothesis calibrates on a disjoint past block;
            # current/future outcomes must not enter either fitting stage.
            from research.entry import predict_entry
            panel["grossReturn"] = panel.actual
            signal = panel.date.max()
            entry, lineage = predict_entry(panel, signal)
            self.assertLess(lineage["lastTrainingOutcome"], lineage["calibrationStart"])
            self.assertLess(lineage["lastCalibrationOutcome"], lineage["signal"])
            self.assertTrue(entry.profitProbability.between(0, 1).all())
            future_panel = panel.copy()
            future_panel.loc[future_panel.date >= signal, "grossReturn"] = 1000
            changed_entry, _ = predict_entry(future_panel, signal)
            np.testing.assert_array_equal(entry.profitProbability, changed_entry.profitProbability)
            np.testing.assert_array_equal(entry.grossEstimate, changed_entry.grossEstimate)
            exported = json.loads((root / "latest.json").read_text())
            self.assertEqual(exported, changed)
            self.assertNotEqual(changed["dataFingerprint"], first["dataFingerprint"])
            self.assertEqual(first["forecasts"][0]["status"], "pending")
            # Repeating unchanged data skips fitting, but still refreshes news and
            # resolves the ledger. A revised future price must invalidate reuse.
            with patch("research.price_analysis.fit_analysis", side_effect=AssertionError("Unexpected retraining")):
                second = run(fixture_loader())
            self.assertEqual(second["backtest"], first["backtest"])
            self.assertEqual(second["audit"], first["audit"])
            self.assertEqual(issue_path.read_bytes(), issue_bytes)
            # Bookkeeping changes do not affect training identity; price logic does.
            training_hash = price_analysis.training_code_hash()
            with patch.object(prospective, "VERSION", "offline-bookkeeping-fixture"):
                self.assertEqual(price_analysis.training_code_hash(), training_hash)
            with patch("research.engine.HORIZON", HORIZON + 1):
                self.assertNotEqual(price_analysis.training_code_hash(), training_hash)
            cache = next(p for p in (root / "analyses-v2").glob("*.json") if json.loads(p.read_text())["data"]["fingerprint"] == first["dataFingerprint"])
            cache_bytes = cache.read_bytes()
            damaged = json.loads(cache.read_text())
            damaged["data"]["backtest"]["metrics"][0]["totalReturn"] = 123
            cache.write_text(json.dumps(damaged))
            with self.assertRaisesRegex(ValueError, "cache integrity"):
                run(fixture_loader())
            cache.write_bytes(cache_bytes)
            self.assertEqual(len(second["forecasts"]), 1)
            self.assertEqual(second["forecasts"][0]["createdAt"], first["forecasts"][0]["createdAt"])
            longer = run_research({**CONFIG, "years": 5}, root, loader=fixture_loader(), news_loader=fixture_news, scorer=fixture_scores)
            self.assertNotEqual(longer["id"], first["id"])
            self.assertEqual(len(longer["forecasts"]), 1)
            # A future observation cannot enter training or resolution under today's clock.
            with patch("research.price_analysis.fit_analysis", side_effect=AssertionError("Unexpected fitting")):
                with self.assertRaisesRegex(ValueError, "future daily price"):
                    run(fixture_loader(extended=True))
            source = fixture_loader(extended=True)
            frames = {s: source(s, 3, root) for s in SYMBOLS}
            last = max(frames[SYMBOLS[0]].index).date()
            future_clock = datetime.combine(last + timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc).isoformat()
            observation = prospective.observe_prices(root, frames, future_clock)
            pending, reason = prospective.resolve(root, issue, observation, None, HORIZON)
            self.assertIsNone(pending)
            self.assertIn("calendar", reason)
            sessions = [d.date().isoformat() for d in frames[SYMBOLS[0]].index]
            calendar = {"sessions": sessions, "coverageStart": sessions[0], "coverageEnd": sessions[-1],
                        "source": "Explicit offline fixture calendar", "sourceSha256": hashlib.sha256(b"fixture").hexdigest()}
            gap_frames = {s: f.copy() for s, f in frames.items()}
            future_days = [d for d in frames[SYMBOLS[0]].index if d.date().isoformat() > first["forecasts"][0]["signalDate"]]
            gap_frames[SYMBOLS[0]] = gap_frames[SYMBOLS[0]].drop(future_days[1])
            gap_observation = prospective.observe_prices(root, gap_frames, future_clock)
            self.assertIsNone(prospective.resolve(root, issue, gap_observation, calendar, HORIZON)[0])
            outcome, reason = prospective.resolve(root, issue, observation, calendar, HORIZON)
            self.assertIsNone(reason)
            old = {**issue["forecast"], **outcome["result"]}
            self.assertEqual(old["status"], "resolved")
            self.assertGreater(old["entryDate"], old["signalDate"])
            dates = pd.bdate_range(old["entryDate"], old["exitDate"])
            self.assertEqual(len(dates), HORIZON + 1)
            source = fixture_loader(extended=True)
            actual = np.mean([source(s, 3, root).loc[old["exitDate"], "open"] / source(s, 3, root).loc[old["entryDate"], "open"] - 1 for s in old["pricePicks"]]) - .002
            self.assertAlmostEqual(old["priceReturn"], actual)
            audit_actual = sum(w * (source(s, 3, root).loc[old["exitDate"], "open"] / source(s, 3, root).loc[old["entryDate"], "open"] - 1) for s, w in old["auditWeights"].items()) - old["auditExposure"] * .002
            self.assertAlmostEqual(old["auditReturn"], audit_actual)
            policy_actual = sum(w * (source(s, 3, root).loc[old["exitDate"], "open"] / source(s, 3, root).loc[old["entryDate"], "open"] - 1) for s, w in old["policyWeights"].items()) - old["policyExposure"] * .002
            self.assertAlmostEqual(old["policyReturn"], policy_actual)
            self.assertTrue(any("overlap" in w for w in second["warnings"]))
            self.assertEqual(issue_path.read_bytes(), issue_bytes)
            revised = {s: f * 2 for s, f in frames.items()}
            later_observation = prospective.observe_prices(root, revised, future_clock)
            self.assertEqual(prospective.resolve(root, issue, later_observation, calendar, HORIZON)[0], outcome)
            self.assertIsNone(prospective.resolve(root, issue, issue["observationSha256"], calendar, HORIZON)[0])
            late_article = {**issue["articles"][0], "firstSeenAt": future_clock}
            with self.assertRaisesRegex(ValueError, "Unavailable news"):
                prospective.capture_news(root, [late_article], issue["featureCutoffAt"])
            receipt = dict(issue["articles"][0])
            revised_content = {**receipt, "title": "A later revised offline headline"}
            captured = prospective.capture_news(root, [revised_content], future_clock)[0]
            self.assertNotEqual(captured["contentSha256"], receipt["contentSha256"])
            self.assertEqual(captured["firstSeenAt"], future_clock)
            for path in (root / prospective.VERSION).rglob("*.json"):
                self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            issue_path.write_text("{}")
            with self.assertRaises((ValueError, KeyError)):
                prospective.ledger_view(root, observation, calendar, HORIZON)

    def test_local_access_invalid_input_and_full_job(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            def fixture_run(config, directory, progress):
                return run_research(config, directory, progress, loader=fixture_loader(), news_loader=fixture_news, scorer=fixture_scores)
            with patch.object(service, "ROOT", root), patch.object(service, "run_research", fixture_run), TestClient(service.app, base_url="http://127.0.0.1") as client:
                self.assertEqual(client.get("/report").status_code, 403)
                self.assertEqual(client.get("/session", headers={"Origin": "https://evil.example"}).status_code, 403)
                self.assertEqual(client.get("/session", headers={"Origin": service.ORIGINS[0], "Host": "evil.example"}).status_code, 400)
                origin = {"Origin": service.ORIGINS[0]}
                token = client.get("/session", headers=origin).json()["token"]
                headers = {**origin, "X-Folio-Token": token}
                self.assertEqual(client.post("/run", headers=origin, json=CONFIG).status_code, 401)
                self.assertEqual(client.post("/run", headers=headers, json={**CONFIG, "symbols": ["http://evil.example"] * 4}).status_code, 422)
                self.assertEqual(client.post("/run", headers=headers, json=CONFIG).status_code, 202)
                self.assertEqual(client.post("/run", headers=headers, json=CONFIG).status_code, 409)
                for _ in range(200):
                    status = client.get("/status", headers=headers).json()
                    if status["status"] != "running":
                        break
                    time.sleep(.1)
                self.assertEqual(status["status"], "complete", status)
                report = client.get("/report", headers=headers).json()["report"]
                self.assertEqual(len(report["rankings"]), 4)
                self.assertEqual(report["forecasts"][0]["status"], "pending")


if __name__ == "__main__":
    unittest.main()
