"""Bounded actions source -> first-seen records -> immutable private cache.

One spanning workflow verifies credential/endpoint/failure boundaries, retained
unknown fields and date claims that must never become historical availability.
No real credentials or network calls are used.
"""
import json
import tempfile
import unittest
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import requests

from research import upstox_metadata as target
from research.test_upstox_workflow import Response


class Transport:
    def __init__(self, body, status=200, error=None):
        self.body, self.status, self.error, self.calls = body, status, error, []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        if self.error:
            raise self.error
        return Response(self.body, self.status)


class MetadataWorkflow(unittest.TestCase):
    def test_source_security_first_seen_cache_and_failure_boundaries(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            token = root / "token"
            token.write_text("synthetic.private.credential")
            token.chmod(0o600)
            evidence = root / "source.json"
            evidence.write_text('{"fixture":true}')
            payload = {"status": "success", "extraTop": "retained in original", "data": [{
                "name": "Bonus", "ratio": "1:1", "amount": None, "expiry_date": "28 Oct 2024",
                "unknownEvent": {"x": 1}, "event_details": [
                    {"name": "Record date", "value": "28 Oct 2024"},
                    {"name": "Record date", "value": "2024-10-28", "extra": True},
                    {"name": "Announcement date", "value": "05 Sep 2024"},
                    {"name": "New future label", "value": "2027-01-01"},
                    {"name": "Allotment date", "value": "unparsed date"}]}]}
            facts = [{"symbol": "RELIANCE", "action": "Bonus", "ratio": "1:1",
                      "recordDate": "2024-10-28", "allotmentDate": "2024-10-29"}]
            plan = {"version": target.VERSION, "frozenAt": datetime.now(timezone.utc).isoformat(),
                    **target.LIMITS, "instruments": [{"symbol": s, "isin": i} for s, i in target.INSTRUMENTS.items()],
                    "sources": [{"path": str(evidence), "sha256": target.sha(evidence)}], "reviewedFacts": facts}
            def prepare(name, edits=None):
                folder = root / name
                folder.mkdir(mode=0o700)
                path = folder / "PLAN.json"
                path.write_text(json.dumps({**plan, **(edits or {})}))
                return path
            body = json.dumps(payload).encode()
            transport = Transport(body)
            path = prepare("success")
            result = target.run(path, token_path=token, transport=transport)
            self.assertEqual(result["requests"], 2)
            self.assertEqual(result["cashSpent"], 0)
            self.assertEqual(result["modelsTrained"], 0)
            first = result["observations"][0]
            record = first["records"][0]
            self.assertEqual(record["firstSeenAt"], first["receivedAt"])
            self.assertGreater(record["firstSeenAt"], "2024-10-29")
            self.assertEqual(len(record["details"]), 5)
            self.assertEqual(record["sourceEvent"]["unknownEvent"], {"x": 1})
            self.assertFalse(record["details"][3]["recognizedDateLabel"])
            self.assertIsNone(record["details"][4]["claimedDate"])
            self.assertTrue(all(record[k] is False for k in target.GATES))
            comparison = first["comparisons"][0]["candidates"][0]
            self.assertTrue(comparison["ratioMatches"])
            self.assertTrue(comparison["recordDateMatches"])
            self.assertIsNone(comparison["allotmentDateMatches"])
            original = path.parent / "run/RELIANCE.json"
            self.assertEqual(original.read_bytes(), body)
            for url, kwargs in transport.calls:
                self.assertIn(url, {target.BASE + i + "/corporate-actions" for i in target.INSTRUMENTS.values()})
                self.assertFalse(kwargs["allow_redirects"])
                self.assertEqual(kwargs["timeout"], 20)
            with patch.object(target, "credential", side_effect=AssertionError("Cached token read")):
                cached = target.run(path, transport=Transport(b"no network"))
            self.assertTrue(cached["cacheHit"])
            self.assertEqual(cached["observations"][0]["receivedAt"], first["receivedAt"])
            client = target.ReadOnlyActions(token.read_text(), plan, Transport(body))
            for url in ["https://evil.example/", "https://api.upstox.com/v2/orders",
                        target.BASE + "INE002A01018/corporate-actions?token=x",
                        target.BASE + "../user", target.BASE + "INE000000000/corporate-actions"]:
                with self.assertRaisesRegex(target.SourceError, "allowlist"):
                    client.get(url)
            self.assertEqual(client.calls, 0)
            conflict = deepcopy(payload)
            conflict["data"][0]["ratio"] = "2:1"
            conflict["data"][0]["event_details"][0]["value"] = "2024-10-27"
            case = target.run(prepare("conflict"), token_path=token,
                              transport=Transport(json.dumps(conflict).encode()))
            self.assertFalse(case["observations"][0]["comparisons"][0]["candidates"][0]["ratioMatches"])
            self.assertFalse(case["observations"][0]["comparisons"][0]["candidates"][0]["recordDateMatches"])
            empty = target.run(prepare("empty"), token_path=token,
                               transport=Transport(b'{"status":"success","data":[]}'))
            self.assertEqual(empty["observations"][0]["comparisons"][0]["status"], "missing_action")
            cases = [("redirect", Transport(body, 302), None), ("unauthorized", Transport(body, 401), None),
                     ("limited", Transport(body, 429), None), ("too_large", Transport(body), {"maximumResponseBytes": 1}),
                     ("malformed", Transport(b'{"status":"success","data":{}}'), None),
                     ("invalid_json", Transport(b"not json"), None),
                     ("transport", Transport(body, error=requests.ConnectionError(token.read_text())), None),
                     ("echo", Transport(json.dumps({"leak": token.read_text()}).encode()), None),
                     ("escaped_echo", Transport(b'{"leak":"synthetic\\u002eprivate.credential"}'), None)]
            for name, source, edits in cases:
                failed = prepare(name, edits)
                with self.assertRaisesRegex(target.SourceError, "pilot stopped"):
                    target.run(failed, token_path=token, transport=source)
                failure = json.loads((failed.parent / "run/failure.json").read_text())
                self.assertEqual(failure["requests"], 1)
                self.assertEqual(len(source.calls), 1)
                with self.assertRaisesRegex(target.SourceError, "Prior attempt"):
                    target.run(failed, token_path=token, transport=source)
            for artifact in root.rglob("run/*.json"):
                self.assertEqual(artifact.stat().st_mode & 0o777, 0o600)
                self.assertNotIn(token.read_bytes(), artifact.read_bytes())
            original.write_bytes(body + b" ")
            with self.assertRaisesRegex(target.SourceError, "integrity"):
                target.run(path, token_path=token, transport=transport)
            evidence.write_text("tampered")
            with self.assertRaisesRegex(target.SourceError, "evidence mismatch"):
                target.run(path, token_path=token, transport=transport)


if __name__ == "__main__":
    unittest.main()
