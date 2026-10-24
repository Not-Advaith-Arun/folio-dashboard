"""Synthetic export/read/retry/hostile-boundary workflow, without market fitting.

The timing, no-clobber, path and external-hash checks protect evidence consumers;
none of this fixture is an investment result or real prospective prediction.
"""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from research import calibration_evidence as e


def fixture():
    contract = {"version": e.VERSION, "experimentId": "synthetic-calibration-example",
        "mode": "prospective", "sessionTimezone": "Asia/Kolkata",
        "signalAt": "2026-01-10T09:00:00+00:00", "issuedAt": "2026-01-10T09:01:00+00:00",
        "protocol": {"sha256": "a" * 64, "frozenAt": "2026-01-01T09:00:00+00:00"},
        "model": {"sha256": "b" * 64, "fittedAt": "2026-01-10T08:00:00+00:00"},
        "target": {"name": "synthetic-20-session-return", "functional": "mean", "score": "squared_error",
                   "quantile": None, "unit": "return_fraction"},
        "sources": [{"id": "synthetic-prices", "sha256": "c" * 64, "receiptSha256": "d" * 64,
                     "firstSeenAt": "2026-01-10T08:30:00+00:00", "availability": "known"}],
        "expectedRows": {"train": 2, "calibration": 2, "forecast": 2}}
    rows = []
    for part, day, end in [("train", "02", "03"), ("calibration", "05", "06"), ("forecast", "10", "12")]:
        for symbol in ["A", "B"]:
            row = {"partition": part, "symbol": symbol, "session": "2026-01-" + day,
                "featureAt": f"2026-01-{day}T09:00:00+00:00",
                "entryAt": f"2026-01-{day}T09:05:00+00:00", "labelEndAt": f"2026-01-{end}T09:05:00+00:00",
                "featureObservedAt": f"2026-01-{day}T08:59:00+00:00",
                "labelObservedAt": f"2026-01-{end}T10:00:00+00:00",
                "raw": .04, "calibrated": .01, "label": -.02, "cost": .007, "risk": .02,
                "sourceIds": ["synthetic-prices"], "missing": {}}
            if part == "train":
                for k in ["raw", "calibrated", "cost", "risk"]:
                    row[k] = None
                    row["missing"][k] = "not_recorded"
            if part == "forecast":
                for k in ["label", "labelObservedAt"]:
                    row[k] = None
                    row["missing"][k] = "pending"
            rows.append(row)
    return contract, rows


class EvidenceWorkflow(unittest.TestCase):
    def test_capture_read_and_fail_closed_boundaries(self):
        clock = {"now": datetime(2026, 1, 10, 9, 2, tzinfo=timezone.utc)}
        contract, rows = fixture()
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(e, "_now", side_effect=lambda: clock["now"]), \
             patch("socket.socket.connect", side_effect=AssertionError("Network forbidden")), \
             patch("pickle.load", side_effect=AssertionError("Model loading forbidden")), \
             patch("pickle.loads", side_effect=AssertionError("Model loading forbidden")):
            base = Path(tmp)
            root = base / "evidence"
            first = e.write_evidence(root, contract, rows)
            self.assertFalse(first["reused"])
            self.assertEqual(root.stat().st_mode & 0o777, 0o700)
            target = root / (first["identitySha256"] + ".json")
            self.assertEqual(target.stat().st_mode & 0o777, 0o600)

            def read(as_of=None, expected=None):
                return e.read_evidence(root, first["identitySha256"], expected or first["payloadSha256"],
                    as_of=as_of or clock["now"].isoformat())

            sealed = read()
            self.assertFalse(sealed["livePromotionAllowed"])
            self.assertFalse(sealed["provenanceClaimsVerified"])
            self.assertEqual(sealed["summary"]["train"]["missingPredictions"], 2)
            self.assertEqual(sealed["summary"]["calibration"]["rows"], 2)
            self.assertAlmostEqual(sealed["rows"][0]["costMargin"], .003)
            self.assertAlmostEqual(sealed["rows"][0]["riskMargin"], -.017)
            with self.assertRaisesRegex(ValueError, "not recorded"):
                read((clock["now"] - timedelta(seconds=1)).isoformat())
            with self.assertRaisesRegex(ValueError, "future"):
                read((clock["now"] + timedelta(seconds=1)).isoformat())
            with self.assertRaisesRegex(ValueError, "Timezone-aware"):
                read("2026-01-10T09:02:00")
            clock["now"] += timedelta(minutes=5)
            again = e.write_evidence(root, deepcopy(contract), list(reversed(deepcopy(rows))))
            self.assertTrue(again["reused"])
            self.assertEqual(again["recordedAt"], first["recordedAt"])
            self.assertEqual(again["payloadSha256"], first["payloadSha256"])
            changed = deepcopy(rows)
            changed[2]["calibrated"] = .02
            with self.assertRaisesRegex(ValueError, "Immutable"):
                e.write_evidence(root, contract, changed)
            for mutate in [lambda c: c.update(mode="retrospective"),
                           lambda c: c["model"].update(fittedAt="2026-01-10T07:00:00+00:00"),
                           lambda c: c["protocol"].update(frozenAt="2025-12-31T09:00:00+00:00")]:
                revised = deepcopy(contract)
                mutate(revised)
                with self.assertRaisesRegex(ValueError, "Immutable"):
                    e.write_evidence(root, revised, rows)

            # Unknown historical availability remains explicit, never upgraded.
            retrospective, missing = deepcopy(contract), deepcopy(rows)
            retrospective["mode"] = "retrospective"
            retrospective["sources"][0].update(availability="unknown", firstSeenAt=None, receiptSha256=None)
            for row in missing:
                for k in ["featureObservedAt", "labelObservedAt"]:
                    if row[k] is not None:
                        row[k] = None
                        row["missing"][k] = "not_recorded"
            unknown = e.write_evidence(base / "unknown", retrospective, missing)
            payload = e.read_evidence(base / "unknown", unknown["identitySha256"], unknown["payloadSha256"], as_of=clock["now"].isoformat())
            self.assertEqual(payload["contract"]["sources"][0]["availability"], "unknown")
            retrospective["mode"] = "prospective"
            with self.assertRaisesRegex(ValueError, "availability unknown"):
                e.write_evidence(base / "invalid", retrospective, missing)

            invalid = []
            c, r = deepcopy(contract), deepcopy(rows)
            c["target"]["score"] = "absolute_error"
            invalid.append((c, r))
            c, r = deepcopy(contract), deepcopy(rows)
            c["sources"][0]["firstSeenAt"] = "2026-01-10T09:00:01+00:00"
            invalid.append((c, r))
            c, r = deepcopy(contract), deepcopy(rows)
            r[2]["featureObservedAt"] = "2026-01-05T09:00:01+00:00"
            invalid.append((c, r))
            c, r = deepcopy(contract), deepcopy(rows)
            r[0]["labelEndAt"] = r[2]["featureAt"]
            r[0]["labelObservedAt"] = r[2]["featureAt"]
            invalid.append((c, r))
            c, r = deepcopy(contract), deepcopy(rows)
            r[0].update(session="2026-01-05", featureAt="2026-01-05T08:00:00+00:00",
                entryAt="2026-01-05T08:01:00+00:00", labelEndAt="2026-01-05T08:30:00+00:00",
                featureObservedAt="2026-01-05T07:59:00+00:00", labelObservedAt="2026-01-05T08:31:00+00:00")
            invalid.append((c, r))  # Same local session split across partitions.
            for key, value in [("raw", float("nan")), ("cost", True), ("label", .2),
                               ("sourceIds", ["not-a-source"]), ("missing", {}),
                               ("featureObservedAt", "2026-01-10T09:03:00+00:00")]:
                c, r = deepcopy(contract), deepcopy(rows)
                r[4][key] = value
                invalid.append((c, r))
            c, r = deepcopy(contract), deepcopy(rows)
            c["expectedRows"]["forecast"] = 3
            invalid.append((c, r))
            c, r = deepcopy(contract), deepcopy(rows)
            c["issuedAt"] = (clock["now"] + timedelta(seconds=1)).isoformat()
            invalid.append((c, r))
            c, r = deepcopy(contract), deepcopy(rows)
            r[1] = deepcopy(r[0])
            invalid.append((c, r))
            for c, r in invalid:
                with self.subTest(invalidCase=invalid.index((c, r))), self.assertRaises(ValueError):
                    e.write_evidence(base / "invalid", c, r)
            self.assertFalse((base / "invalid").exists())
            for cap, value in [("MAX_ROWS", 5), ("MAX_BYTES", 1000), ("MAX_SOURCES", 0)]:
                with patch.object(e, cap, value), self.assertRaises(ValueError):
                    e.write_evidence(base / "budget", contract, rows)
            self.assertFalse((base / "budget").exists())

            original = target.read_bytes()
            wrapper = json.loads(original)
            wrapper["payload"]["rows"][0]["calibrated"] = .99
            target.write_text(json.dumps(wrapper))
            with self.assertRaisesRegex(ValueError, "seal/hash"):
                read()
            wrapper["sha256"] = e._digest(wrapper["payload"])
            target.write_text(json.dumps(wrapper))
            with self.assertRaisesRegex(ValueError, "seal/hash"):
                read()  # Local rehash still fails the external pinned digest.
            target.write_bytes(original)
            target.chmod(0o644)
            with self.assertRaisesRegex(ValueError, "0600"):
                read()
            target.chmod(0o600)
            read()
            insecure = base / "insecure"
            insecure.mkdir(mode=0o755)
            insecure.chmod(0o755)
            with self.assertRaisesRegex(ValueError, "0700"):
                e.write_evidence(insecure, contract, rows)
            self.assertEqual(insecure.stat().st_mode & 0o777, 0o755)
            linked = base / "linked"
            linked.symlink_to(root, target_is_directory=True)
            with self.assertRaises(OSError):
                e.write_evidence(linked, contract, rows)
            hostile = base / "hostile"
            hostile.mkdir(mode=0o700)
            outside = base / "outside"
            outside.write_text("must stay unchanged")
            (hostile / target.name).symlink_to(outside)
            with self.assertRaises(OSError):
                e.write_evidence(hostile, contract, rows)
            self.assertEqual(outside.read_text(), "must stay unchanged")
            with self.assertRaises(FileNotFoundError):
                e.read_evidence(base / "absent", first["identitySha256"], first["payloadSha256"], as_of=clock["now"].isoformat())
            self.assertFalse((base / "absent").exists())

            # Atomic publication never exposes partial JSON; same-content races reuse.
            with patch.object(e.os, "link", side_effect=OSError("Injected pre-publication failure")):
                with self.assertRaisesRegex(OSError, "Injected"):
                    e.write_evidence(base / "interrupted", contract, rows)
            self.assertEqual(list((base / "interrupted").iterdir()), [])
            race = base / "race"
            with ThreadPoolExecutor(max_workers=2) as pool:
                winners = list(pool.map(lambda _: e.write_evidence(race, deepcopy(contract), deepcopy(rows)), range(2)))
            self.assertEqual(len({r["payloadSha256"] for r in winners}), 1)
            self.assertEqual(sum(not r["reused"] for r in winners), 1)
            self.assertEqual(len(list(race.iterdir())), 1)


if __name__ == "__main__":
    unittest.main()
