"""Two spanning workflows: dated research/outcomes and local API security.

Synthetic prices and sentiment are explicit offline fixtures, never production
fallbacks. Calculation timing and access control warrant these critical checks.
"""
import json
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
            self.assertAlmostEqual(sum(a["weight"] for a in audit["latest"]["allocations"]), audit["latest"]["exposure"])
            for fold in audit["folds"]:
                self.assertLess(fold["innerTrainLastOutcome"], fold["tuningStart"])
                self.assertLess(fold["tuningLastOutcome"], fold["calibrationStart"])
                self.assertLess(fold["finalTrainLastOutcome"], fold["calibrationStart"])
                self.assertLess(fold["calibrationLastOutcome"], fold["testStart"])
                self.assertAlmostEqual(sum(fold["weights"].values()), 1)
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
            for field in ["selected", "weights", "candidates", "intervalRadius"]:
                self.assertEqual(changed["audit"]["folds"][0][field], audit["folds"][0][field])
            exported = json.loads((root / "latest.json").read_text())
            self.assertEqual(exported, changed)
            self.assertEqual(first["forecasts"][0]["status"], "pending")
            second = run(fixture_loader())
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
