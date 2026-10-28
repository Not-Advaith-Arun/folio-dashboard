"""One synthetic pinned review/export/refusal workflow, zero market fits.

Critical JSON/private-path/concurrency boundaries belong here because the old
consumers cannot test this optional evidence interface. No live adapter is used.
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

from research import calendar_evidence as e


def fixture():
    source = {"id": "synthetic-not-an-exchange-document", "url": "https://example.invalid/synthetic",
              "kind": "synthetic", "contentSha256": "a"*64, "receiptSha256": "b"*64,
              "issueDate": "2026-01-01", "receivedAt": "2026-01-02T00:00:00Z", "revisionStatus": "reviewed"}
    days = []
    for i, status in enumerate(["special", "regular", "closed"]):
        days.append({"date": f"2026-02-0{i+1}", "cashStatus": status, "cashSourceIds": [source["id"]],
            "t0Status": "closed", "t0SourceIds": [source["id"]],
            "settlementStatus": "closed" if i == 0 else "unknown",
            "settlementSourceIds": [source["id"]] if i == 0 else [],
            "phases": [] if status == "closed" else [
                {"class": "CAS", "kind": "continuous", "start": "14:20" if i == 0 else "09:15",
                 "end": "15:15", "sourceIds": [source["id"]]},
                {"class": "NON_CAS", "kind": "continuous", "start": "14:20" if i == 0 else "09:15",
                 "end": "15:30", "sourceIds": [source["id"]]}],
            "eligibility": {} if status == "closed" else {
                "SYN_A": {"class": "CAS", "sourceIds": [source["id"]]},
                "SYN_B": {"class": "NON_CAS", "sourceIds": [source["id"]]}}})
    return {"version": e.VERSION, "calendarId": "synthetic-only", "revision": "r1", "mode": "synthetic",
            "timezone": "Asia/Kolkata", "coverageStart": "2026-02-01", "coverageEnd": "2026-02-03",
            "informationCutoffAt": "2026-01-03T00:00:00Z", "protocolSha256": "c"*64,
            "sources": [source], "days": days}


class CalendarWorkflow(unittest.TestCase):
    def test_source_to_review_export_and_refusals(self):
        clock = {"now": datetime(2026, 2, 10, tzinfo=timezone.utc)}
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(e, "_now", side_effect=lambda: clock["now"]), \
             patch("socket.socket.connect", side_effect=AssertionError("network forbidden")), \
             patch("pickle.load", side_effect=AssertionError("model load forbidden")), \
             patch("pickle.loads", side_effect=AssertionError("model load forbidden")):
            root = Path(tmp) / "calendar"
            root.mkdir(mode=0o700)
            serial = 0

            def prepare(bundle, modify_review=None):
                nonlocal serial
                serial += 1
                bid, rid = f"bundle-{serial}", f"review-{serial}"
                bp = e.seal(root, bid, bundle)
                review = {"bundleSha256": bp["payloadSha256"], "reviewedAt": clock["now"].isoformat(),
                    "reviewer": "synthetic-reviewer", "reviewEvidenceSha256": "d"*64,
                    "coverageComplete": True, "exceptionsComplete": True, "cashRevisionsResolved": True,
                    "sourceVisualReviewComplete": True, "exceptionsReviewedThrough": clock["now"].isoformat()}
                if modify_review:
                    modify_review(review)
                rp = e.seal(root, rid, review)
                return bid, bp["payloadSha256"], rid, rp["payloadSha256"]

            def export(ids, **kwargs):
                return e.export_calendar(root, *ids, as_of=clock["now"].isoformat(), **kwargs)

            bundle = fixture()
            ids = prepare(bundle)
            daily = export(ids)
            self.assertEqual(daily["sessions"], ["2026-02-01", "2026-02-02"])
            self.assertEqual(daily["days"][0]["t0Status"], "closed")
            self.assertEqual(daily["days"][0]["settlementStatus"], "closed")
            self.assertTrue(all(daily[k] is False for k in e.FLAGS))
            self.assertEqual(daily["mode"], "synthetic")
            # A supported special Sunday is not forced into the regular pilot.
            with self.assertRaisesRegex(ValueError, "Special session"):
                export(ids, target="intraday_pilot", symbols=["SYN_A", "SYN_B"])

            regular = deepcopy(bundle)
            regular.update(coverageStart="2026-02-02", coverageEnd="2026-02-02", days=[regular["days"][1]])
            regular_ids = prepare(regular)
            intraday = export(regular_ids, target="intraday_pilot", symbols=["SYN_A", "SYN_B"])
            self.assertEqual(len(intraday["barStarts"]), 72)
            self.assertTrue(intraday["barStarts"][0].endswith("09:15:00+05:30"))
            self.assertTrue(intraday["barStarts"][-1].endswith("15:10:00+05:30"))
            # Exercise the unchanged real consumer through panel creation only.
            import numpy as np
            import pandas as pd
            from research import intraday as pilot
            calendar = pd.DatetimeIndex(intraday["barStarts"]).tz_convert(intraday["sessionTimezone"])
            prices = 100 + np.arange(72) * .01
            frame = pd.DataFrame({"open": prices, "high": prices + .1, "low": prices - .1,
                "close": prices + .01, "volume": 1000 + np.arange(72),
                "availableAt": calendar + pd.Timedelta(minutes=5)}, index=calendar)
            provenance = {"source": "SYNTHETIC ONLY", "sourceUseEvidence": "synthetic",
                "calendarEvidence": intraday["sourceSha256"], "availabilityBasis": "synthetic",
                "priceBasis": "raw"}
            with patch.object(pilot, "walk_forward", side_effect=AssertionError("fit forbidden")):
                panel, manifest = pilot.build_panel({"SYN_A": frame, "SYN_B": frame.copy()},
                                                    calendar, provenance, clock["now"].isoformat())
            self.assertEqual(len(panel), 144)
            self.assertFalse(manifest["livePromotionAllowed"])

            # Receipt's UTC day can precede the issuer's local India date.
            boundary = deepcopy(bundle)
            boundary["sources"][0].update(issueDate="2026-01-02", receivedAt="2026-01-01T20:00:00Z")
            self.assertEqual(export(prepare(boundary))["sessions"], daily["sessions"])

            changed = []
            for mutate in [lambda b: b["days"].pop(),
                           lambda b: b["days"][0].update(cashStatus="conflict"),
                           lambda b: b["days"][0].update(cashStatus="unknown"),
                           lambda b: b["sources"][0].update(kind="normalized"),
                           lambda b: b["sources"][0].update(revisionStatus="unknown"),
                           lambda b: b["sources"][0].update(receivedAt="2026-02-20T00:00:00Z"),
                           lambda b: b.update(mode="research"),
                           lambda b: b["days"][0].update(cashSourceIds=[]),
                           lambda b: b["days"][0]["phases"][0].update(start="16:00"),
                           lambda b: b["days"][2].update(phases=deepcopy(bundle["days"][1]["phases"]))]:
                b = deepcopy(bundle)
                mutate(b)
                changed.append(b)
            for b in changed:
                with self.assertRaises(ValueError):
                    export(prepare(b))
            for mutate in [lambda r: r.update(coverageComplete=False),
                           lambda r: r.update(sourceVisualReviewComplete=False),
                           lambda r: r.update(exceptionsComplete=1),
                           lambda r: r.update(cashRevisionsResolved=False),
                           lambda r: r.update(bundleSha256="e"*64),
                           lambda r: r.update(reviewedAt="2026-01-01T00:00:00Z"),
                           lambda r: r.update(exceptionsReviewedThrough="2026-01-01T00:00:00Z")]:
                with self.assertRaises(ValueError):
                    export(prepare(bundle, mutate))
            for mutate in [lambda b: b["days"][0]["eligibility"].pop("SYN_A"),
                           lambda b: b["days"][0]["eligibility"]["SYN_A"].update(**{"class": "unknown"}),
                           lambda b: b["days"][0]["phases"][0].update(end="15:10"),
                           lambda b: b["days"][0]["phases"].append(deepcopy(b["days"][0]["phases"][0]))]:
                b = deepcopy(regular)
                mutate(b)
                with self.assertRaises(ValueError):
                    export(prepare(b), target="intraday_pilot", symbols=["SYN_A", "SYN_B"])

            # Byte-immutable reuse must distinguish JSON true from numeric 1.
            first = e.seal(root, "immutable", {"checked": True})
            path = root / "immutable.json"
            original = path.read_bytes()
            clock["now"] += timedelta(seconds=1)
            self.assertEqual(e.seal(root, "immutable", {"checked": True}), first)
            self.assertEqual(path.read_bytes(), original)
            with self.assertRaisesRegex(ValueError, "Immutable"):
                e.seal(root, "immutable", {"checked": 1})
            with ThreadPoolExecutor(max_workers=3) as pool:
                copies = list(pool.map(lambda _: e.seal(root, "concurrent", {"same": "document"}), range(3)))
            self.assertTrue(all(copy == copies[0] for copy in copies))
            with patch.object(e.os, "link", side_effect=OSError("synthetic pre-publication failure")):
                with self.assertRaises(OSError):
                    e.seal(root, "not-published", {})
            self.assertFalse((root / "not-published.json").exists())
            self.assertFalse(list(root.glob(".pending-*")))
            with self.assertRaises(ValueError):
                e.read(root, "immutable", first["payloadSha256"], as_of="2026-01-01T00:00:00Z")
            with self.assertRaises(ValueError):
                e.read(root, "immutable", first["payloadSha256"], as_of=(clock["now"]+timedelta(seconds=1)).isoformat())
            for bad in [original+b' ', b'{"version":1,"version":2}', b'{"x":NaN}']:
                path.write_bytes(bad)
                with self.assertRaises(ValueError):
                    e.read(root, "immutable", first["payloadSha256"], as_of=clock["now"].isoformat())
            path.write_bytes(original)
            path.chmod(0o644)
            with self.assertRaises(ValueError):
                e.read(root, "immutable", first["payloadSha256"], as_of=clock["now"].isoformat())
            path.chmod(0o600)
            (root / "link.json").symlink_to(path)
            with self.assertRaises(OSError):
                e.read(root, "link", first["payloadSha256"], as_of=clock["now"].isoformat())
            ancestor = Path(tmp) / "redirect"
            ancestor.symlink_to(root, target_is_directory=True)
            with self.assertRaises(OSError):
                e.seal(ancestor, "bad", {})
            (root / "fifo.json").unlink(missing_ok=True)
            os.mkfifo(root / "fifo.json", 0o600)
            with self.assertRaises(ValueError):
                e.read(root, "fifo", "a"*64, as_of=clock["now"].isoformat())
            with self.assertRaises(ValueError):
                e.seal(root, "oversized", {"value": "a"*e.MAX_BYTES})
            # Reusing the external pins needs no export or market evaluation.
            with patch.object(e, "export_calendar", side_effect=AssertionError("recompile forbidden")):
                self.assertEqual(e.read(root, ids[0], ids[1], as_of=clock["now"].isoformat())["document"], bundle)


if __name__ == "__main__":
    unittest.main()
