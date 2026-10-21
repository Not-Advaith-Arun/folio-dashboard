"""Spanning multi-horizon workflow for critical finance/time/evidence boundaries.

Synthetic data is never production fallback or evidence of market profitability.
"""
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from research import horizons
from research.panel import session_panel


class DurationWorkflow(unittest.TestCase):
    def test_frozen_multihorizon_prediction_workflow(self):
        dates = pd.bdate_range("2020-01-01", periods=1300)
        frames = {}
        for i in range(4):
            rng = np.random.default_rng(i + 100)
            close = 100 * np.exp(np.cumsum(rng.normal(.0002, .01, len(dates))))
            frames[str(i)] = pd.DataFrame(dict(open=close, high=close * 1.01,
                low=close * .99, close=close, volume=np.full(len(dates), 100000.)), index=dates)
        plan = dict(version="long-horizon-study-1", frozenAt=datetime.now(timezone.utc).isoformat(),
                    spec=horizons.SPEC, start="2022-01-01", end=str(dates[-1].date()), parents=[],
                    calendarExcluded=[], calendarSource="Synthetic weekday fixture, not exchange evidence",
                    priceBasis="Synthetic prices", limitations=["Fixture only"])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            entries = []
            for symbol, frame in frames.items():
                path = root / f"{symbol}.json"
                frame.rename_axis("date").reset_index().assign(date=dates.strftime("%Y-%m-%d")).to_json(path, orient="records")
                entries.append(dict(path=str(path), sha256=horizons.sha(path), symbol=symbol))
            plan["featureInputs"] = entries
            path = root / "PLAN.json"
            path.write_text(json.dumps(plan))
            result = horizons.run(path)
            self.assertGreater(result["fitCount"], 20)
            self.assertFalse(result["livePromotionAllowed"])
            self.assertFalse(result["portfolioProfitEvaluated"])
            records = json.loads((root / "run/forecasts.json").read_text())
            self.assertEqual({r["horizon"] for r in records}, {20, 63, 126, 252})
            self.assertTrue(any(r["prediction"] is not None and r["actual"] is None for r in records))
            fits = json.loads((root / "run/fits.json").read_text())
            for fit in fits:
                if fit["status"] == "fitted":
                    self.assertLess(fit["lastTrainOutcome"], fit["signal"])
            signal = dates[900]
            changed = {s: f.copy() for s, f in frames.items()}
            for f in changed.values():
                f.loc[f.index > signal, ["open", "high", "low", "close"]] *= 10
            for horizon in horizons.SPEC["horizons"]:
                def predict(source):
                    panel, _ = session_panel(source, dates, horizon=horizon,
                        calendar_source="Fixture", price_basis="Synthetic")
                    return horizons.fit_snapshot(panel, signal)[0]
                before, after = predict(frames), predict(changed)
                np.testing.assert_allclose(before.prediction, after.prediction, rtol=0, atol=0)
            with patch("research.horizons.fit_snapshot", side_effect=AssertionError("Repeated fit")):
                self.assertTrue(horizons.run(path)["cacheHit"])
            artifact = root / "run/forecasts.json"
            artifact.write_text(artifact.read_text() + " ")
            with self.assertRaisesRegex(ValueError, "Artifact hash"):
                horizons.run(path)
            for artifact in (root / "run").glob("*.json"):
                self.assertEqual(artifact.stat().st_mode & 0o777, 0o600)


if __name__ == "__main__":
    unittest.main()
