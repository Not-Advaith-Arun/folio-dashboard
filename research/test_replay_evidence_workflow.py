"""Synthetic unchanged main/restart replay -> immutable evidence -> refusal."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
from research import replay_evidence as evidence
from research.rebalance import replay_rebalanced
from research.execution import FEE_SOURCE, FEE_CHECKED


def fixture():
    sessions = pd.bdate_range("2024-01-02", periods=8)
    frames = {s: pd.DataFrame({"open": [price + i for i in range(8)],
        "close": [price + i + .5 for i in range(8)], "high": [price + i + 1 for i in range(8)],
        "low": [price + i - 1 for i in range(8)], "volume": 100000,
        "splits": 0., "dividends": 0.}, index=sessions) for s, price in [("A", 100.), ("B", 50.)]}
    decisions = [{"signal": str(sessions[i].date()), "entry": str(sessions[i+1].date()),
        "weights": {s: 1.}} for i, s in [(0, "A"), (2, "B"), (4, "A")]]
    now = datetime.now(timezone.utc)
    contract = {"version": evidence.VERSION, "experimentId": "synthetic-replay-only", "mode": "synthetic",
        "protocol": {"sha256": "a"*64, "frozenAt": (now-timedelta(seconds=1)).isoformat()},
        "producedAt": now.isoformat(), "sources": [{"id": "synthetic", "sha256": "b"*64,
            "firstSeenAt": None, "receiptSha256": None, "availability": "unknown"}],
        "code": {"replaySha256": "c"*64, "feeSha256": "d"*64},
        "cost": {"feeSource": FEE_SOURCE, "feeChecked": FEE_CHECKED,
            "settlement": "sale-next-session-no-cashflows", "turnover": evidence.TURNOVER}, "expected": []}
    records = []
    for kind, ds in [("main", decisions), ("restart", decisions[1:])]:
        replay = replay_rebalanced(frames, ds, sessions[-1], {"A": 1, "B": 1}, initial=1000., slippage_bps=5.)
        replay["unknownProviderField"] = {"value": None, "meaning": "unverified"}
        records.append({"book": "candidate", "scenario": "5bps", "kind": kind,
            "start": ds[0]["entry"], "end": str(sessions[-1].date()), "initial": 1000., "slippageBps": 5.,
            "lots": {"A": 1, "B": 1}, "decisionsSha256": evidence.digest(ds), "calendarSha256": "e"*64,
            "pricesSha256": "f"*64, "replay": replay, "missing": {}})
    # A summary-only legacy fixture is synthetic, never a historical backfill.
    legacy = {**deepcopy(records[1]), "book": "missing-control", "lots": None,
        "replay": {"initial": 1000., "ending": 1000., "netReturn": 0., "fees": 0., "dailyCloseDrawdown": 0.}}
    legacy["missing"] = {"lots": "not_recorded", **{"replay."+k: "not_recorded" for k in evidence.REPLAY if k not in legacy["replay"]}}
    records.append(legacy)
    contract["expected"] = [{k: r[k] for k in ["book", "scenario", "kind"]} for r in records]
    return contract, records


class ReplayEvidenceWorkflow(unittest.TestCase):
    def test_synthetic_main_restart_capture_accounting_missing_and_security(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); archive = root / "archive"; archive.mkdir(mode=0o700)
            contract, records = fixture()
            original = deepcopy(records)
            # From here, capture/read/reuse must never invoke a numerical replay,
            # fitting, deserialization or network operation.
            with patch("research.rebalance.replay_rebalanced", side_effect=AssertionError("Replay")), \
                 patch("pickle.load", side_effect=AssertionError("Model load")), \
                 patch("pickle.loads", side_effect=AssertionError("Model load")), \
                 patch("socket.socket.connect", side_effect=AssertionError("Network")):
                with ThreadPoolExecutor(max_workers=4) as pool:
                    receipts = list(pool.map(lambda _: evidence.seal(archive, contract, records), range(4)))
                first = receipts[0]
                self.assertEqual(len({r["payloadSha256"] for r in receipts}), 1)
                self.assertEqual(len({r["recordedAt"] for r in receipts}), 1)
                self.assertEqual(sum(not r["reused"] for r in receipts), 1)
                self.assertEqual(original, records)
                cutoff = datetime.now(timezone.utc).isoformat()
                read = lambda: evidence.read(archive, first["identitySha256"], first["payloadSha256"], as_of=cutoff)
                payload = read()
                self.assertEqual(payload["records"], records)
                self.assertFalse(payload["livePromotionAllowed"])
                self.assertFalse(payload["tradabilityVerified"])
                self.assertFalse(payload["provenanceClaimsVerified"])
                summaries = {(s["book"], s["kind"]): s for s in payload["summary"]}
                for record in records[:2]:
                    summary = summaries[(record["book"], record["kind"])]
                    self.assertEqual(summary["status"], "complete")
                    self.assertEqual(summary["unverifiedExtraFields"], ["unknownProviderField"])
                    self.assertEqual(summary["cashCheck"], "supplied-convention-consistent")
                    self.assertGreater(summary["unfilledCount"], 0)
                    actual = sum(o["price"]*o["units"] for o in record["replay"]["orders"])/1000.
                    self.assertAlmostEqual(actual, summary["turnover"])
                    self.assertEqual(record["replay"]["curve"][0]["cash"], 1000.)
                missing = summaries[("missing-control", "restart")]
                self.assertEqual(missing["status"], "incomplete")
                self.assertIsNone(missing["orderCount"])
                self.assertIsNone(missing["turnover"])
                self.assertEqual(missing["cashCheck"], "unavailable")
                self.assertNotEqual(records[0]["replay"]["ending"], records[1]["replay"]["ending"])
                self.assertTrue(evidence.seal(archive, contract, records)["reused"])
                changed = deepcopy(records); changed[0]["replay"]["unknownProviderField"] = "changed"
                with self.assertRaisesRegex(ValueError, "Immutable"):
                    evidence.seal(archive, contract, changed)
                with self.assertRaisesRegex(ValueError, "Missing/duplicate"):
                    evidence.seal(archive, contract, records[:-1])
                with self.assertRaises(ValueError):
                    evidence.read(archive, first["identitySha256"], first["payloadSha256"], as_of=(datetime.fromisoformat(first["recordedAt"])-timedelta(microseconds=1)).isoformat())
                with self.assertRaises(ValueError):
                    evidence.read(archive, first["identitySha256"], first["payloadSha256"], as_of=(datetime.now(timezone.utc)+timedelta(days=1)).isoformat())
                # Critical accounting/security calculations belong here because
                # valid JSON and an ordinary read cannot reveal these failures.
                mutations = [lambda r: r[0]["replay"].update(fees=r[0]["replay"]["fees"]+.01),
                    lambda r: r[0]["replay"]["orders"][0].update(units=1.5),
                    lambda r: r[0]["replay"]["orders"][0].update(fee=True),
                    lambda r: r[0]["replay"]["orders"][0].update(price=float("inf")),
                    lambda r: r[0]["replay"]["curve"][1].update(cash=999.),
                    lambda r: r[0]["replay"]["curve"].pop(),
                    lambda r: r[0].update(lots={"A": 2, "B": 2}),
                    lambda r: r[0]["replay"].update(feeChecked="2024-01-01"),
                    lambda r: r[2].update(missing={}),
                    lambda r: r[0]["replay"]["orders"].append(deepcopy(r[0]["replay"]["orders"][0]))]
                for mutation in mutations:
                    bad = deepcopy(records); mutation(bad)
                    with self.assertRaises((ValueError, TypeError)):
                        evidence.seal(archive, contract, bad)
                future = deepcopy(contract); future["producedAt"] = (datetime.now(timezone.utc)+timedelta(days=1)).isoformat()
                with self.assertRaises(ValueError): evidence.seal(archive, future, records)
                future = deepcopy(contract);future["sources"][0].update(availability="known",firstSeenAt=(datetime.now(timezone.utc)+timedelta(days=1)).isoformat(),receiptSha256="a"*64)
                with self.assertRaises(ValueError): evidence.seal(archive, future, records)
                absent = deepcopy(records);absent[2]["replay"]=None;absent[2]["missing"]={"replay":"unavailable","lots":"not_recorded"}
                new_contract = deepcopy(contract);new_contract["protocol"]["sha256"]="b"*64
                missing_receipt = evidence.seal(archive,new_contract,absent)
                self.assertNotEqual(first["identitySha256"],missing_receipt["identitySha256"])
                evidence._close(1.,1.+.5e-6)
                with self.assertRaises(ValueError):evidence._close(1.,1.+2e-6)
                with self.assertRaises(ValueError):evidence._close(1.,float("inf"))
                with patch.object(evidence,"MAX_BYTES",100):
                    with self.assertRaises(ValueError):read()
                    with self.assertRaises(ValueError):evidence.seal(archive,contract,records)
                with patch.object(evidence,"MAX_TOTAL",1):
                    with self.assertRaises(ValueError):evidence.seal(archive,contract,records)
                with patch.object(evidence.os,"link",side_effect=OSError("Injected publication failure")):
                    with self.assertRaises(OSError):evidence.seal(archive,contract,records)
                self.assertFalse(list(archive.glob(".pending-*")))
                path=archive/(first["identitySha256"]+".json");body=path.read_bytes()
                for bad_bytes in [b' '+body,body[:-1]+b',"version":"duplicate"}',body.replace(b'1000.0',b'NaN',1),body.replace(b'synthetic-replay-only',b'tampered-replay-only')]:
                    path.write_bytes(bad_bytes)
                    with self.assertRaises(ValueError):read()
                path.write_bytes(body);path.chmod(0o644)
                with self.assertRaises(ValueError):read()
                path.chmod(0o600);backup=root/'backup';backup.write_bytes(body);path.unlink();path.symlink_to(backup)
                with self.assertRaises(OSError):read()
                path.unlink();os.mkfifo(path,0o600)
                with self.assertRaises(ValueError):read()
                path.unlink();path.write_bytes(body);path.chmod(0o600)
                link=root/'link';link.symlink_to(archive,target_is_directory=True)
                with self.assertRaises(OSError):evidence.read(link,first["identitySha256"],first["payloadSha256"],as_of=cutoff)
                archive.chmod(0o755)
                with self.assertRaises(ValueError):read()
                archive.chmod(0o700)
                self.assertEqual(payload,read())
                self.assertEqual(path.stat().st_mode & 0o777,0o600)


if __name__ == "__main__":
    unittest.main()
