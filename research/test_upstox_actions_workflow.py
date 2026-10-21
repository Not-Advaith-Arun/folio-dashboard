"""Issuer evidence -> fixed bonus diagnostics -> sealed offline audit.

One spanning workflow covers the critical unit/calculation/lineage boundaries;
agreement must never produce executable prices or immediate bonus liquidity.
"""
import json
import tempfile
import unittest
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from research import upstox_actions as target
from research.engine import atomic_json


class ActionAuditWorkflow(unittest.TestCase):
    def test_evidence_to_diagnostics_and_immutable_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parent = root / "parent"
            parent.mkdir(mode=0o700)
            cache = parent / "run"
            cache.mkdir(mode=0o700)
            atomic_json(parent / "PLAN.json", {"fixture": True})
            payload = {"data": {"candles": [
                ["2024-01-02T00:00:00+05:30", 50, 55, 45, 52, 300, 0],
                ["2024-01-03T00:00:00+05:30", 52, 60, 50, 58, 500, 0],
                ["2024-10-25T00:00:00+05:30", 75, 80, 70, 76, 1000, 0],
                ["2024-10-28T00:00:00+05:30", 77, 80, 75, 78, 1800, 0]]}}
            atomic_json(cache / "A.json", payload)
            atomic_json(cache / "results.json", {"observations": [{"file": "A.json",
                "instrumentKey": "NSE_EQ|FIXTURE", "window": {"unit": "days"}}]})
            (cache / "catalogue.gz").write_bytes(b"never used as historical membership")
            atomic_json(cache / "completed.json", {"artifacts": {p.name: target.sha(p) for p in cache.iterdir()}})
            source = root / "issuer.json"
            atomic_json(source, {"fixtureOnly": True})
            rules = {"action": {"type": "bonus", "bonusNumerator": 1, "bonusDenominator": 1,
                "totalShareFactor": 2, "recordDate": "2024-10-28", "diagnosticBoundaryDate": "2024-10-28",
                "sourceIds": ["issuer"], "bonusCreditDate": None, "bonusTradableDate": None},
                "anchor": {"month": "2024-01", "exchange": "NSE", "high": 120, "low": 90,
                           "volume": 800, "sourceId": "issuer"},
                "window": {"start": "2024-10-21", "end": "2024-11-01"},
                "priceMultipliers": [1, 2], "volumeMultipliers": [1, 0.5, 2],
                "priceAbsoluteTolerance": 0.01, "volumeAbsoluteTolerance": 0}
            entry = lambda p: {"path": str(p), "sha256": target.sha(p)}
            plan = {"version": target.VERSION, "frozenAt": datetime.now(timezone.utc).isoformat(),
                "parentPlan": entry(parent / "PLAN.json"), "parentCompletion": entry(cache / "completed.json"),
                "sources": [{"id": "issuer", **entry(source)}], "rules": rules,
                "candleFile": "A.json", "instrumentKey": "NSE_EQ|FIXTURE"}
            path = root / "PLAN.json"
            atomic_json(path, plan)
            with patch("requests.Session.get", side_effect=AssertionError("Network request")), \
                 patch("research.upstox.credential", side_effect=AssertionError("Credential read")):
                result = target.run(path)
            self.assertEqual([x["agreesWithinTolerance"] for x in result["priceComparisons"]], [False, True])
            self.assertEqual([x["agreesExactly"] for x in result["volumeComparisons"]], [True, False, False])
            self.assertEqual(result["boundaryDiagnostic"]["declaredFactorPreClose"], 152)
            self.assertAlmostEqual(result["boundaryDiagnostic"]["declaredFactorPostToPrePriceRatio"], 77 / 152)
            self.assertEqual(result["eventWindowProviderBars"][0]["close"], 76)
            self.assertFalse(result["boundaryDiagnostic"]["netReturnEvaluated"])
            self.assertTrue(all(result[k] is False for k in target.GATES))
            self.assertIsNone(result["action"]["bonusTradableDate"])
            with patch.object(target, "audit", side_effect=AssertionError("Repeated evaluation")):
                self.assertTrue(target.run(path)["cacheHit"])
            missing = deepcopy(payload)
            missing["data"]["candles"] = missing["data"]["candles"][2:3]
            pending = target.audit(missing, rules)
            self.assertEqual(pending["anchorStatus"], "missing_anchor_month")
            self.assertEqual(pending["boundaryDiagnostic"]["status"], "missing_side")
            # Changing one observed day must preserve the disagreement, not find
            # a new factor or repair prices/volumes to match an aggregate.
            incomplete = deepcopy(payload)
            incomplete["data"]["candles"].pop(0)
            mismatch = target.audit(incomplete, rules)
            self.assertFalse(any(x["agreesExactly"] for x in mismatch["volumeComparisons"]))
            self.assertFalse(mismatch["productionCalendarVerified"])
            rounded = deepcopy(rules)
            rounded["anchor"].update(high=120.05, low=90.05)
            self.assertFalse(any(x["agreesWithinTolerance"] for x in
                                 target.audit(payload, rounded)["priceComparisons"]))
            for field, value in [(1, float("nan")), (2, 1), (5, -1), (5, 0.5)]:
                bad = deepcopy(payload)
                bad["data"]["candles"][0][field] = value
                with self.assertRaisesRegex(ValueError, "Invalid daily"):
                    target.audit(bad, rules)
            duplicate = deepcopy(payload)
            duplicate["data"]["candles"].append(duplicate["data"]["candles"][0])
            with self.assertRaisesRegex(ValueError, "Unique"):
                target.audit(duplicate, rules)
            invented = deepcopy(rules)
            invented["action"]["bonusTradableDate"] = "2024-10-28"
            with self.assertRaisesRegex(ValueError, "Credit/listing"):
                target.audit(payload, invented)
            tuned = deepcopy(rules)
            tuned["priceMultipliers"].append(2.01)
            with self.assertRaisesRegex(ValueError, "frozen"):
                target.audit(payload, tuned)
            self.assertEqual((root / "run").stat().st_mode & 0o777, 0o700)
            for p in (root / "run").glob("*.json"):
                self.assertEqual(p.stat().st_mode & 0o777, 0o600)
            sealed_source = (cache / "A.json").read_bytes()
            (cache / "A.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "Parent artifact"):
                target.run(path)
            (cache / "A.json").write_bytes(sealed_source)
            (root / "run/audit.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "Immutable"):
                target.run(path)
            source.write_text("tampered")
            with self.assertRaisesRegex(ValueError, "Source hash"):
                target.run(path)


if __name__ == "__main__":
    unittest.main()
