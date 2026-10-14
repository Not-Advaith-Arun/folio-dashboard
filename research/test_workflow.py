"""Spanning workflows: capital/discovery, session data, research, API security.

Synthetic prices and sentiment are explicit offline fixtures, never production
fallbacks. Calculation timing and access control warrant these critical checks.
"""
import json
import hashlib
import tempfile
import time
import unittest
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

    def test_train_export_repeat_and_resolve_future_outcomes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            run = lambda loader: run_research(CONFIG, root, loader=loader, news_loader=fixture_news, scorer=fixture_scores)
            first = run(fixture_loader())
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
            with patch("research.engine.walk_forward", side_effect=AssertionError("Unexpected retraining")):
                second = run(fixture_loader())
            self.assertEqual(second["backtest"], first["backtest"])
            self.assertEqual(second["audit"], first["audit"])
            cache = next(p for p in (root / "analyses").glob("*.json") if json.loads(p.read_text())["data"]["fingerprint"] == first["dataFingerprint"])
            damaged = json.loads(cache.read_text())
            damaged["data"]["backtest"]["metrics"][0]["totalReturn"] = 123
            cache.write_text(json.dumps(damaged))
            recovered = run(fixture_loader())
            self.assertEqual(recovered["backtest"], first["backtest"])
            self.assertEqual(len(second["forecasts"]), 1)
            self.assertEqual(second["forecasts"][0]["createdAt"], first["forecasts"][0]["createdAt"])
            longer = run_research({**CONFIG, "years": 5}, root, loader=fixture_loader(), news_loader=fixture_news, scorer=fixture_scores)
            self.assertNotEqual(longer["id"], first["id"])
            self.assertEqual(len(longer["forecasts"]), 1)
            resolved = run(fixture_loader(extended=True))
            old = next(f for f in resolved["forecasts"] if f["id"] == first["id"])
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
            self.assertTrue(any("overlap" in w for w in resolved["warnings"]))

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
