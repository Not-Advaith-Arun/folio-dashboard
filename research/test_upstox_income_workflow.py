"""Synthetic source -> private first-seen versions -> bounded read/cache.

One spanning workflow covers endpoint/credential/time/budget hard boundaries,
malformed or partial responses, immutable versions and hostile local paths.
No network, real credentials, model fit or financial outcomes are used.
"""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from research import upstox_income as target


class Response:
    def __init__(self, body, status):
        self.body, self.status_code = body, status
        self.headers = {"Content-Type": "application/json; charset=utf-8"}

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def iter_content(self, size):
        for i in range(0, len(self.body), size):
            yield self.body[i:i + size]


class Transport:
    def __init__(self, body, status=200, error=None):
        self.body, self.status, self.error, self.calls = body, status, error, []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        if self.error:
            raise self.error
        return Response(self.body, self.status)


class IncomeWorkflow(unittest.TestCase):
    def test_source_first_seen_versions_cache_and_security(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            token = root / "credential"
            token.write_text("synthetic.private.credential")
            token.chmod(0o600)
            source = root / "source.html"
            source.write_text("SYNTHETIC source contract")
            source.chmod(0o600)
            plan = {"version": target.VERSION, "frozenAt": target.now(),
                "instruments": target.INSTRUMENTS, "query": target.QUERY, **target.LIMITS,
                "sources": [{"path": str(source), "sha256": target.digest(source.read_bytes())}]}
            payload = {"status": "success", "unknownTop": {"retain": True}, "data": {
                "type": "consolidated", "time_period": "quarterly", "units_in": "crore",
                "extraData": [1, 2], "income_statement": [
                    {"category": "revenue", "unknownGroup": True, "history": [
                        {"period": "Mar 2024", "value": 100, "unknownRow": "retained"},
                        {"period": "Mar 2024", "value": 101},
                        {"period": "Jun 2024"}, {"period": "Sep 2024", "value": None},
                        {"period": "Dec 2024", "value": "not_numeric"}]},
                    {"category": "new_category", "history": []}]}}
            body = target.encoded(payload)
            def prepare(name, edits=None):
                folder = root / name
                folder.mkdir(mode=0o700)
                p = folder / "PLAN.json"
                p.write_bytes(target.encoded({**plan, **(edits or {})}))
                p.chmod(0o600)
                return p
            path = prepare("success")
            transport = Transport(body)
            result = target.run(path, token_path=token, transport=transport)
            self.assertEqual((result["requests"], result["cashSpent"], result["modelsTrained"]), (2, 0, 0))
            self.assertEqual(result["responseBytes"], 2 * len(body))
            ref = result["snapshots"][0]
            out = path.parent / "run"
            seals = out / "snapshots"
            first = target.read_snapshot(seals, ref["key"], ref["sealSha256"], as_of=target.now())
            rows = first["normalized"]["rows"]
            self.assertEqual([r["valueStatus"] for r in rows], ["finite", "finite", "missing", "null", "invalid"])
            self.assertEqual([r["duplicateStatus"] for r in rows[:2]], ["conflicting", "conflicting"])
            self.assertEqual(first["normalized"]["missingCategories"], ["net_profit", "operating_profit"])
            self.assertEqual(first["normalized"]["unknownCategories"], ["new_category"])
            self.assertEqual((out / "RELIANCE.raw").read_bytes(), body)
            self.assertEqual(rows[0]["sourceRow"]["unknownRow"], "retained")
            self.assertEqual(rows[0]["firstSeenAt"], first["receipt"]["receivedAt"])
            self.assertGreater(rows[0]["firstSeenAt"], "2024-03-31")
            self.assertIsNone(rows[0]["issuerPublicationAt"])
            self.assertIsNone(rows[0]["periodEndAt"])
            self.assertTrue(all(first[k] is False for k in target.GATES))
            for url, kwargs in transport.calls:
                self.assertIn(url, {target.url_for(i) for i in target.INSTRUMENTS})
                self.assertFalse(kwargs["allow_redirects"])
                self.assertTrue(kwargs["verify"])
                self.assertEqual(kwargs["timeout"], 15)
            self.assertFalse(transport.trust_env)
            with patch.object(target, "credential", side_effect=AssertionError("credential read")), \
                 patch.object(target, "normalize", side_effect=AssertionError("recompute")):
                cache_transport = Transport(body, error=AssertionError("network"))
                cached = target.run(path, transport=cache_transport)
            self.assertTrue(cached["cacheHit"])
            self.assertEqual(cached["snapshots"], result["snapshots"])
            self.assertEqual(cache_transport.calls, [])
            before = (target.timestamp(first["sealedAt"]) - timedelta(microseconds=1)).isoformat()
            with self.assertRaisesRegex(target.SourceError, "time-bound"):
                target.read_snapshot(seals, ref["key"], ref["sealSha256"], as_of=before)
            for invalid_hash in ["../x", "0" * 64]:
                with self.assertRaises((target.SourceError, FileNotFoundError)):
                    target.read_snapshot(seals, ref["key"], invalid_hash, as_of=target.now())
            receipt = {**first["receipt"], "requestedAt": target.now(), "receivedAt": target.now()}
            reused = target.snapshot(seals, target.INSTRUMENTS[0], body, receipt, first["lineage"])
            self.assertEqual(reused, first)
            changed = deepcopy(payload)
            changed["data"]["income_statement"][0]["history"][0]["value"] = 90
            later = target.snapshot(seals, target.INSTRUMENTS[0], target.encoded(changed), receipt,
                                    first["lineage"], previous={"key": ref["key"], "sealSha256": ref["sealSha256"]})
            self.assertEqual(later["previous"]["sealSha256"], ref["sealSha256"])
            self.assertEqual(later["changeMeaning"], "snapshot_change_only")
            self.assertEqual(later["normalized"]["rows"][0]["restatementStatus"], "unknown")
            self.assertEqual(target.read_snapshot(seals, ref["key"], ref["sealSha256"], as_of=target.now()), first)
            # Independent client refuses every unlisted URL and duplicate request.
            client_transport = Transport(body)
            client = target.ReadOnlyIncome(token.read_text(), plan, client_transport)
            for url in ["https://evil.example/", "https://api.upstox.com/v2/orders",
                        target.url_for(target.INSTRUMENTS[0]) + "&from=2024-01-01",
                        target.url_for(target.INSTRUMENTS[0]).replace("consolidated", "standalone")]:
                with self.assertRaisesRegex(target.SourceError, "allowlist"):
                    client.get(url)
            self.assertEqual(client.calls, 0)
            client.get(target.url_for(target.INSTRUMENTS[0]))
            with self.assertRaisesRegex(target.SourceError, "budget"):
                client.get(target.url_for(target.INSTRUMENTS[0]))
            self.assertEqual(len(client_transport.calls), 1)
            cases = [("redirect", body, 302, None, None), ("auth", body, 401, None, None),
                ("plan", body, 403, None, None), ("rate", body, 429, None, None),
                ("malformed", b"not json", 200, None, None),
                ("duplicate_key", b'{"status":"error","status":"success","data":{}}', 200, None, None),
                ("nonfinite", body.replace(b'100', b'NaN'), 200, None, None),
                ("wrong_units", body.replace(b'crore', b'rupees'), 200, None, None),
                ("wrong_frequency", body.replace(b'quarterly', b'yearly'), 200, None, None),
                ("byte_budget", body, 200, {"maximumResponseBytes": 1}, None),
                ("total_bytes", body, 200, {"maximumBytes": 1}, None),
                ("echo", target.encoded({"error": token.read_text()}), 200, None, None),
                ("escaped_duplicate_echo", b'{"x":"synthetic\\u002eprivate.credential","x":"safe"}', 200, None, None),
                ("transport", body, 200, None, RuntimeError(token.read_text()))]
            for name, bad, status, edits, error in cases:
                failed = prepare(name, edits)
                source_transport = Transport(bad, status, error)
                with self.assertRaisesRegex(target.SourceError, "pilot stopped"):
                    target.run(failed, token_path=token, transport=source_transport)
                failure = target.decoded((failed.parent / "run/failure.json").read_bytes())
                self.assertEqual(failure["requests"], 1)
                self.assertEqual(len(source_transport.calls), 1)
                with self.assertRaisesRegex(target.SourceError, "Prior attempt"):
                    target.run(failed, token_path=token, transport=source_transport)
                self.assertEqual(len(source_transport.calls), 1)
            # Real wall-time expiry, exhausted deadlines, private/no-follow input/output.
            with self.assertRaises(target.SourceError):
                with target.wall_limit(0.001):
                    import time
                    time.sleep(0.01)
            client.deadline = 0
            with self.assertRaisesRegex(target.SourceError, "budget"):
                client.get(target.url_for(target.INSTRUMENTS[1]))
            token.chmod(0o644)
            with self.assertRaisesRegex(target.SourceError, "owner-private"):
                target.credential(token)
            token.chmod(0o600)
            link = root / "linked-token"
            link.symlink_to(token)
            with self.assertRaises(OSError):
                target.credential(link)
            symlink_root = root / "linked-dir"
            symlink_root.symlink_to(seals, target_is_directory=True)
            with self.assertRaises(OSError):
                target.read_snapshot(symlink_root, ref["key"], ref["sealSha256"], as_of=target.now())
            for artifact in root.rglob("run/*"):
                if artifact.is_file():
                    self.assertEqual(artifact.stat().st_mode & 0o777, 0o600)
                    self.assertNotIn(token.read_bytes(), artifact.read_bytes())
            seal_file = seals / (ref["key"] + ".json")
            seal_file.write_bytes(seal_file.read_bytes() + b" ")
            with self.assertRaisesRegex(target.SourceError, "seal mismatch"):
                target.run(path, transport=Transport(body))
            source.write_text("tampered")
            with self.assertRaisesRegex(target.SourceError, "source mismatch"):
                target.run(path, transport=Transport(body))


if __name__ == "__main__":
    unittest.main()
