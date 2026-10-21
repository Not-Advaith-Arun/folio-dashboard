"""Retained source -> session reconciliation -> sealed audit workflow, offline.

Critical boundaries: auction bars cannot become continuous fills, a reviewed
holiday is not a full calendar, and matching bars cannot certify raw prices.
"""
import json
import tempfile
import unittest
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from research import upstox_audit as target
from research.engine import atomic_json


class RetainedAuditWorkflow(unittest.TestCase):
    def test_source_to_reconciliation_cache_and_failure_boundaries(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parent = root / "parent"
            parent.mkdir(mode=0o700)
            atomic_json(parent / "PLAN.json", {"fixture": True})
            source = parent / "run"
            source.mkdir(mode=0o700)
            dates = pd.date_range("2026-09-21 09:15", "2026-09-21 15:25", freq="5min", tz="Asia/Kolkata")
            payloads, observations = {}, []
            for symbol, key in [("A", "NSE_EQ|FIXTURE"), ("INDEX", "NSE_INDEX|Fixture")]:
                for unit in ["days", "minutes"]:
                    if unit == "days":
                        rows = [[d + "T00:00:00+05:30", 100, 101, 99, 100, 7500 if symbol == "A" else 0, 0]
                                for d in ["2026-09-21", "2026-10-19"]]
                    else:
                        rows = [[d.isoformat(), 100, 101, 99, 100,
                                 0 if symbol == "INDEX" or d.strftime("%H:%M") in ["15:15", "15:20"] else 100, 0]
                                for d in dates]
                    name = f"{symbol}-{unit}.json"
                    payloads[name] = {"data": {"candles": rows}}
                    observations.append({"symbol": symbol, "instrumentKey": key, "window": {"unit": unit},
                                         "file": name, "catalogueMatch": {"cas_eligible": symbol == "A"}})
                    atomic_json(source / name, payloads[name])
            atomic_json(source / "results.json", {"observations": observations})
            (source / "catalogue.json.gz").write_bytes(b"fixture catalogue, never loaded")
            atomic_json(source / "completed.json", {"artifacts": {p.name: target.sha(p) for p in source.iterdir()}})
            rules = {"prefixStart": "09:15", "prefixEndExclusive": "15:15", "barMinutes": 5,
                     "calendarFact": {"date": "2026-10-20", "status": "closed", "source": "Synthetic fixture only"}}
            plan = {"version": target.VERSION, "frozenAt": datetime.now(timezone.utc).isoformat(), "rules": rules,
                    "parentPlan": {"path": str(parent / "PLAN.json"), "sha256": target.sha(parent / "PLAN.json")},
                    "inputs": [{"path": str(p), "sha256": target.sha(p)} for p in source.glob("*.json")]}
            plan_path = root / "PLAN.json"
            atomic_json(plan_path, plan)
            with patch("requests.Session.get", side_effect=AssertionError("Audit requested market data")), \
                 patch("research.upstox.credential", side_effect=AssertionError("Audit read a credential")):
                result = target.run(plan_path)
            self.assertFalse(result["trainingApproved"])
            self.assertFalse(result["rawPriceVerified"])
            self.assertFalse(result["productionCalendarVerified"])
            self.assertEqual([s["prefixRows"] for s in result["sessions"]], [72, 72])
            self.assertTrue(all(not s["missingPrefixSlots"] for s in result["sessions"]))
            stock = result["sessions"][0]
            self.assertEqual(stock["zeroVolumeTimes"], ["15:15", "15:20"])
            self.assertEqual(stock["prefixZeroVolumeRows"], 0)
            self.assertTrue(stock["allStockZerosAfterPrefix"])
            comparison = result["reconciliations"][0]
            self.assertEqual(comparison["dailyVolume"], 7500)
            self.assertEqual(comparison["minuteVolume"], 7300)
            self.assertNotEqual(comparison["dailyToMinuteVolumeRatio"], 1.)
            self.assertTrue(all(x["endDateGapConsistentWithReviewedHoliday"] for x in result["calendar"]["observations"]))
            self.assertTrue(any(x["status"] == "benchmark_not_trade_candidate" for x in result["retainedExceptions"]))
            with patch("research.upstox_audit.audit", side_effect=AssertionError("Repeated audit")):
                self.assertTrue(target.run(plan_path)["cacheHit"])
            missing = deepcopy(payloads)
            missing["A-minutes.json"]["data"]["candles"].pop(1)
            incomplete = target.audit(observations, missing, rules)
            self.assertEqual(len(incomplete["sessions"][0]["missingPrefixSlots"]), 1)
            self.assertFalse(incomplete["trainingApproved"])
            duplicate = deepcopy(payloads)
            duplicate["A-minutes.json"]["data"]["candles"].append(duplicate["A-minutes.json"]["data"]["candles"][0])
            with self.assertRaisesRegex(ValueError, "Unique"):
                target.audit(observations, duplicate, rules)
            zero = deepcopy(payloads)
            zero["A-minutes.json"]["data"]["candles"][3][5] = 0
            altered = target.audit(observations, zero, rules)
            self.assertFalse(altered["sessions"][0]["allStockZerosAfterPrefix"])
            self.assertTrue(any(x["status"] == "unresolved_zero_volume" for x in altered["retainedExceptions"]))
            for file in (root / "run").glob("*.json"):
                self.assertEqual(file.stat().st_mode & 0o777, 0o600)
            (root / "run/audit.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "Immutable"):
                target.run(plan_path)
            (source / "A-days.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "source hash"):
                target.run(plan_path)


if __name__ == "__main__":
    unittest.main()
