"""Source -> bounded GET acquisition -> audit -> immutable reuse spanning test.

All transport/credentials are synthetic. Endpoint and credential checks are
critical security boundaries; this test never contacts a broker or market API.
"""
import gzip
import json
import os
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from research import upstox


class Response:
    def __init__(self, body, status=200):
        self.body, self.status_code = body, status

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def iter_content(self, _size):
        yield self.body


class Transport:
    def __init__(self, catalogue):
        self.catalogue, self.calls, self.status = catalogue, [], 200

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        if url == upstox.CATALOGUE:
            return Response(gzip.compress(json.dumps(self.catalogue).encode()), self.status)
        return Response(json.dumps({"status": "success", "data": {"candles": [
            ["2024-01-08T09:15:00+05:30", 100, 101, 99, 100.5, 10000, 0]]}}).encode(), self.status)


class UpstoxWorkflow(unittest.TestCase):
    def test_private_readonly_source_acquisition_and_failure_boundaries(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            token = root / "token"
            token.write_text("synthetic.private.credential")
            token.chmod(0o600)
            evidence = root / "evidence.json"
            evidence.write_text("{}")
            sources = root / "sources.json"
            sources.write_text(json.dumps([dict(path=str(evidence), sha256=upstox.sha(evidence), status=200)]))
            instruments = [dict(symbol=s, key=k) for s, k in [
                ("A", "NSE_EQ|INE009A01021"), ("B", "NSE_EQ|INE002A01018"), ("INDEX", "NSE_INDEX|Nifty 50")]]
            plan = dict(version="upstox-source-pilot-1", frozenAt=datetime.now(timezone.utc).isoformat(),
                **upstox.LIMITS, instruments=instruments, sourceManifest=dict(path=str(sources), sha256=upstox.sha(sources)),
                priceBasis="Synthetic fixture only", windows=[dict(unit="minutes", interval=5,
                    **{"from": "2024-01-08", "to": "2024-01-12"}) for _ in range(3)])
            path = root / "PLAN.json"
            path.write_text(json.dumps(plan))
            transport = Transport([dict(instrument_key=i["key"], lot_size=1) for i in instruments])
            result = upstox.run(path, token_path=token, transport=transport)
            self.assertEqual(result["requests"], 10)
            self.assertEqual(result["cashSpent"], 0)
            self.assertEqual(result["modelsTrained"], 0)
            self.assertEqual(len(result["observations"]), 9)
            self.assertTrue(all(not o["trainingApproved"] for o in result["observations"]))
            for url, kwargs in transport.calls:
                self.assertFalse(kwargs["allow_redirects"])
                self.assertEqual("Authorization" in kwargs["headers"], url.startswith(upstox.HISTORY))
            self.assertTrue(upstox.run(path, token_path=token, transport=transport)["cacheHit"])
            self.assertEqual(len(transport.calls), 10)
            for artifact in (root / "run").glob("*"):
                if artifact.is_file():
                    self.assertEqual(artifact.stat().st_mode & 0o777, 0o600)
                    self.assertNotIn(token.read_bytes(), artifact.read_bytes())
            client = upstox.ReadOnlyHistory(token.read_text(), plan, transport)
            for url in ["https://evil.example/", "https://api.upstox.com/v2/orders",
                        upstox.HISTORY + "../../user", upstox.HISTORY + "x?token=secret"]:
                with self.assertRaisesRegex(ValueError, "allowlist"):
                    client.get(url)
            self.assertEqual(client.calls, 0)
            transport.status = 302
            with self.assertRaisesRegex(ValueError, "HTTP 302"):
                client.get(upstox.CATALOGUE)
            transport.status = 200
            small = upstox.ReadOnlyHistory(token.read_text(), {**plan, "maximumResponseBytes": 1}, transport)
            with self.assertRaisesRegex(ValueError, "byte budget"):
                small.get(upstox.CATALOGUE)
            token.chmod(0o644)
            with self.assertRaisesRegex(ValueError, "owner-private"):
                upstox.credential(token)
            token.chmod(0o600)
            payload = root / "run/A-0.json"
            payload.write_text(payload.read_text() + " ")
            with self.assertRaisesRegex(ValueError, "integrity"):
                upstox.run(path, token_path=token, transport=transport)
            # A failed source attempt is retained, never auto-retried or logged
            # with credential-bearing request headers or exception text.
            failed = root / "failed"
            failed.mkdir(mode=0o700)
            failed_plan = failed / "PLAN.json"
            failed_plan.write_text(json.dumps(plan))
            transport.status = 401
            with self.assertRaisesRegex(ValueError, "HTTP 401"):
                upstox.run(failed_plan, token_path=token, transport=transport)
            failure = failed / "run/failure.json"
            self.assertEqual(json.loads(failure.read_text())["requests"], 1)
            self.assertNotIn(token.read_text(), failure.read_text())
            with self.assertRaisesRegex(ValueError, "Prior attempt"):
                upstox.run(failed_plan, token_path=token, transport=transport)


if __name__ == "__main__":
    unittest.main()
