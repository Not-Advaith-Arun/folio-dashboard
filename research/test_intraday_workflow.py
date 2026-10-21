"""Spanning intraday workflow for critical finance/time/evidence boundaries.

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

from research import intraday


class IntradayWorkflow(unittest.TestCase):
    def test_intraday_source_training_execution_workflow(self):
        days = pd.bdate_range("2023-01-02", periods=85)
        calendar = pd.DatetimeIndex([bar for day in days for bar in
            pd.date_range(str(day.date()) + " 09:15", str(day.date()) + " 15:10", freq="5min", tz="Asia/Kolkata")])
        frames = {}
        for i, symbol in enumerate(["A", "B"]):
            rng = np.random.default_rng(99 + i)
            close = 100 * np.exp(np.cumsum(rng.normal(.00003, .003, len(calendar))))
            frames[symbol] = pd.DataFrame(dict(open=close, high=close * 1.01, low=close * .99,
                close=close, volume=np.full(len(calendar), 100000.), availableAt=calendar + intraday.BAR), index=calendar)
        provenance = dict(source="Synthetic", sourceUseEvidence="Fixture-only", calendarEvidence="Synthetic weekdays",
                          availabilityBasis="Synthetic bar-close receipt", priceBasis="raw")
        cutoff = calendar[-1] + intraday.BAR
        result, forecasts = intraday.study(frames, calendar, provenance, cutoff)
        panel, manifest = intraday.build_panel(frames, calendar, provenance, cutoff)
        fits = result["fits"]
        self.assertGreater(result["quality"]["scoredRows"], 0)
        self.assertEqual(set(result["comparison"]), {"0.0", "5.0", "10.0"})
        self.assertFalse(manifest["livePromotionAllowed"])
        self.assertEqual(len(fits), 2)
        self.assertTrue(all(f["status"] == "fitted" for f in fits))
        self.assertTrue(all(pd.Timestamp(f["lastTrainOutcome"]).date().isoformat() < f["testStart"] for f in fits))
        self.assertTrue((forecasts.entryAt - forecasts.barStart == 2 * intraday.BAR).all())
        self.assertTrue((forecasts.exitAt - forecasts.entryAt == 6 * intraday.BAR).all())
        for book in intraday.BOOKS:
            runs = [intraday.replay(panel, forecasts, book=book, slippage_bps=bps) for bps in [0, 5, 10]]
            for result in runs:
                self.assertAlmostEqual(result["endingCapital"], 100000 + sum(t["pnl"] for t in result["trades"]))
                for trade in result["trades"]:
                    self.assertEqual(pd.Timestamp(trade["entryAt"]).date(), pd.Timestamp(trade["exitAt"]).date())
                    self.assertLessEqual(pd.Timestamp(trade["exitAt"]).strftime("%H:%M"), "15:10")
                    self.assertGreater(trade["fees"], 0)
            if book == "always":
                self.assertGreater(len(runs[0]["trades"]), 0)
                self.assertLess(runs[2]["endingCapital"], runs[0]["endingCapital"])
        # Exact official-fee arithmetic is a critical financial calculation.
        self.assertAlmostEqual(intraday.fees(25000, "buy"), (20 + .7925) * 1.18 + .75)
        self.assertAlmostEqual(intraday.fees(25000, "sell"), (20 + .7925) * 1.18 + 6.25)
        # Later outcomes cannot change first-block features, fitting or predictions.
        changed = {s: f.copy() for s, f in frames.items()}
        boundary = pd.Timestamp(fits[0]["testStart"], tz="Asia/Kolkata")
        for f in changed.values():
            f.loc[f.index >= boundary, ["open", "high", "low", "close"]] *= 2
        changed_panel, _ = intraday.build_panel(changed, calendar, provenance, cutoff)
        changed_forecasts, changed_fits = intraday.walk_forward(changed_panel)
        self.assertEqual(fits[0]["parameters"], changed_fits[0]["parameters"])
        first = forecasts.session.eq(fits[0]["testStart"])
        np.testing.assert_allclose(forecasts.loc[first, "prediction"], changed_forecasts.loc[first, "prediction"], atol=1e-12)
        missing = {**frames, "A": frames["A"].iloc[:-1]}
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            intraday.build_panel(missing, calendar, provenance, cutoff)
        with self.assertRaisesRegex(ValueError, "Future or incomplete"):
            intraday.build_panel(frames, calendar, provenance, cutoff - 2 * intraday.BAR)
        with self.assertRaisesRegex(ValueError, "evidence required"):
            intraday.build_panel(frames, calendar, {}, cutoff)
        late = {s: f.copy() for s, f in frames.items()}
        bar = calendar[20]
        late["A"].loc[bar, "availableAt"] = bar + 3 * intraday.BAR
        late_panel, _ = intraday.build_panel(late, calendar, provenance, cutoff)
        self.assertFalse(late_panel[(late_panel.symbol == "A") & late_panel.barStart.eq(bar)].featureReady.item())
        # An exit failure invalidates the book; it is never silently excluded.
        broken = panel.copy()
        first_row = forecasts.iloc[0]
        mask = broken.symbol.eq(first_row.symbol) & broken.barStart.eq(first_row.barStart)
        broken.loc[mask, "exitVolume"] = 0
        with self.assertRaisesRegex(ValueError, "Unresolved exit"):
            intraday.replay(broken, forecasts.iloc[:1], book="always")
        altered = forecasts.iloc[:1].copy()
        altered["entryAt"] += intraday.BAR
        with self.assertRaisesRegex(ValueError, "Forecast timing"):
            intraday.replay(panel, altered, book="always")


if __name__ == "__main__":
    unittest.main()
