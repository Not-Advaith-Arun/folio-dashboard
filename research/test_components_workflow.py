"""Synthetic source -> purged component fit -> unchanged replay -> sealed cache.

Checks exact log identity, no future-feature leakage and preservation of missing
outcomes/candidates; none of this fixture is market or prospective evidence.
"""
import json
import tempfile
import unittest
from pathlib import Path
from datetime import datetime, timezone
from unittest.mock import patch

import numpy as np
import pandas as pd

from research import architecture as base, components as target
from research.engine import atomic_json


class ComponentWorkflow(unittest.TestCase):
    def test_component_source_fit_replay_and_immutable_boundaries(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sessions = pd.bdate_range("2021-01-04", periods=820)
            rng = np.random.default_rng(19)
            frames, feature_inputs, raw_inputs = {}, [], []
            for i, symbol in enumerate(["A", "B", "C", "D"]):
                n = rng.normal(.0002, .004, len(sessions))
                d = rng.normal(-.0001, .006, len(sessions))
                opens, closes, previous = [], [], 100. + i * 20
                for overnight, intraday in zip(n, d):
                    opening = previous * np.exp(overnight)
                    previous = opening * np.exp(intraday)
                    opens.append(opening)
                    closes.append(previous)
                f = pd.DataFrame({"open": opens, "close": closes, "volume": rng.integers(50000, 200000, len(sessions))}, index=sessions)
                f["high"] = f[["open", "close"]].max(axis=1) * 1.005
                f["low"] = f[["open", "close"]].min(axis=1) * .995
                f["splits"] = 0.
                f["dividends"] = 0.
                frames[symbol] = f
                p = root / (symbol + ".json")
                f.reset_index(names="date").assign(date=lambda x: x.date.dt.strftime("%Y-%m-%d")).to_json(p, orient="records", double_precision=15)
                entry = {"symbol": symbol, "path": str(p), "sha256": base.sha(p)}
                feature_inputs.append(entry)
                raw_inputs.append(entry)
            original_root = root / "original"
            original_root.mkdir(mode=0o700)
            compat = {"previous": target.code_hashes()["engine.py"], "current": target.code_hashes()["engine.py"]}
            parent = {"architecture": base.SPEC, "books": base.BOOKS, "frozenAt": datetime.now(timezone.utc).isoformat(),
                "featureInputs": feature_inputs, "executionInputs": raw_inputs, "calendarExcluded": [],
                "start": str(sessions[-45].date()), "end": str(sessions[-1].date()), "calendarSource": "Synthetic weekdays only",
                "priceBasis": "Synthetic prices only", "capital": 100000., "slippageScenarios": [0., 5., 10.],
                "primarySlippage": 5., "signalStride": 20, "lots": dict.fromkeys(frames, 1), "evidenceLimits": ["Synthetic fixture"],
                "engineCompatibility": compat}
            pp = original_root / "protocol.json"
            atomic_json(pp, parent)
            original = base.run_experiment(pp)
            ref = {"protocol": {"path": str(pp), "sha256": base.sha(pp)}, "completed": {
                "path": str(original_root / "run/completed.json"), "sha256": base.sha(original_root / "run/completed.json")}}
            references = dict.fromkeys(["original", "normalized", "payoff"], ref)
            folder = root / "candidate"
            folder.mkdir(mode=0o700)
            plan = folder / "PLAN.json"
            atomic_json(plan, {"references": references, "books": list(target.BOOKS), "fixtureOnly": True})
            protocol = folder / "protocol.json"
            atomic_json(protocol, {"frozenAt": datetime.now(timezone.utc).isoformat(), "spec": target.SPEC,
                "books": list(target.BOOKS), "code": target.code_hashes(), "references": references,
                "plan": {"path": str(plan), "sha256": base.sha(plan)}, "engineCompatibility": compat})
            with patch("research.architecture.fit_at", side_effect=AssertionError("Reference refit")), \
                 patch("pickle.load", side_effect=AssertionError("Old pickle loaded")):
                result = target.run(protocol)
            self.assertEqual(result["fitCount"], original["fitCount"])
            self.assertEqual(result["estimatorsFitted"], 3 * result["fitCount"])
            self.assertTrue(result["predictiveAcceptanceChecks"]["same_scored_coverage"])
            self.assertFalse(result["livePromotionAllowed"])
            for slip in ["0.0", "5.0", "10.0"]:
                for b, summary in original["comparison"][slip].items():
                    self.assertEqual(result["comparison"][slip][b], summary)
            panel, _ = target.component_panel(frames, sessions, calendar_source="Synthetic", price_basis="Synthetic")
            signal = sessions[-40]
            first, fit, _ = target.fit_components(panel, signal)
            self.assertLess(fit["trainLastOutcome"], fit["calibrationStart"])
            self.assertLess(fit["calibrationLastOutcome"], str(signal.date()))
            changed = {s: f.copy() for s, f in frames.items()}
            for f in changed.values():
                f.loc[f.index > signal, ["open", "close", "high", "low"]] *= 1.5
                f.loc[f.index > signal, "volume"] *= 3
            changed["A"] = changed["A"].drop(sessions[-10])
            altered, _ = target.component_panel(changed, sessions, calendar_source="Synthetic", price_basis="Synthetic")
            second, _, _ = target.fit_components(altered, signal)
            for key in ["componentMean", "directMean", "nightLog", "dayLog"]:
                np.testing.assert_array_equal(first[key], second[key])
            last, _, _ = target.fit_components(panel, sessions[-1])
            self.assertEqual(len(last), 4)
            self.assertTrue(last.actual.isna().all())
            rows = json.loads((folder / "run/forecasts.json").read_text())
            self.assertTrue(any(r["componentMean"] is not None and r["actual"] is None for r in rows))
            for f in (folder / "run").glob("*"):
                if f.is_file():
                    self.assertEqual(f.stat().st_mode & 0o777, 0o600)
            with patch.object(target, "evaluate", side_effect=AssertionError("Repeated fit/replay")):
                self.assertTrue(target.run(protocol)["cacheHit"])
            (folder / "run/decisions.json").write_text("[]")
            with self.assertRaisesRegex(ValueError, "Immutable"):
                target.run(protocol)


if __name__ == "__main__":
    unittest.main()
