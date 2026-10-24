"""Synthetic weighted fit -> calibration seal -> replay -> immutable reuse."""
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from research import architecture as base, components, recency
from research.calibration_evidence import read_evidence
from research.engine import atomic_json


class RecencyWorkflow(unittest.TestCase):
    def test_weighted_fit_evidence_and_unchanged_boundaries(self):
        # Critical calculation: each whole date has declared exponential mass,
        # independent of its stock count; Ridge/scaler share mean-one weights.
        dates = pd.bdate_range("2021-01-04", periods=253)
        sample = pd.DataFrame({"date": [dates[0], dates[126], dates[126], dates[252]], "symbol": list("ABCD")})
        w, ess = recency.recency_weights(sample, dates)
        masses = pd.Series(w).groupby(sample.date).sum().to_numpy()
        np.testing.assert_allclose(masses / masses[-1], [.25, .5, 1.], atol=1e-14)
        self.assertAlmostEqual(w.mean(), 1.)
        self.assertAlmostEqual(w[1], w[2])
        self.assertLess(ess, 3)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sessions = pd.bdate_range("2021-01-04", periods=820)
            rng, frames, inputs = np.random.default_rng(19), {}, []
            for i, symbol in enumerate("ABCD"):
                close = (100 + i * 20) * np.exp(np.cumsum(rng.normal(.0002, .008, len(sessions))))
                opening = close * np.exp(rng.normal(0, .002, len(sessions)))
                f = pd.DataFrame({"open": opening, "close": close, "volume": rng.integers(50000, 200000, len(sessions)),
                    "high": np.maximum(opening, close) * 1.005, "low": np.minimum(opening, close) * .995,
                    "splits": 0., "dividends": 0.}, index=sessions)
                frames[symbol] = f
                path = root / (symbol + ".json")
                f.reset_index(names="date").assign(date=lambda x: x.date.dt.strftime("%Y-%m-%d")).to_json(path, orient="records", double_precision=15)
                inputs.append({"symbol": symbol, "path": str(path), "sha256": base.sha(path)})
            original = root / "original"
            original.mkdir(mode=0o700)
            compatibility = {"previous": base.sha(Path(base.__file__).with_name("engine.py")),
                             "current": base.sha(Path(base.__file__).with_name("engine.py"))}
            parent = {"architecture": base.SPEC, "books": base.BOOKS, "frozenAt": datetime.now(timezone.utc).isoformat(),
                "featureInputs": inputs, "executionInputs": inputs, "calendarExcluded": [],
                "start": str(sessions[-45].date()), "end": str(sessions[-1].date()), "calendarSource": "Synthetic weekdays only",
                "priceBasis": "Synthetic only", "capital": 100000., "slippageScenarios": [0., 5., 10.],
                "primarySlippage": 5., "signalStride": 20, "lots": dict.fromkeys(frames, 1),
                "evidenceLimits": ["Synthetic only"], "engineCompatibility": compatibility}
            pp = original / "protocol.json"
            atomic_json(pp, parent)
            baseline = base.run_experiment(pp)

            def reference(folder):
                return {"protocol": {"path": str(folder / "protocol.json"), "sha256": base.sha(folder / "protocol.json")},
                    "completed": {"path": str(folder / "run/completed.json"), "sha256": base.sha(folder / "run/completed.json")}}

            refs = dict.fromkeys(["original", "normalized", "payoff"], reference(original))
            comp = root / "component"
            comp.mkdir(mode=0o700)
            atomic_json(comp / "PLAN.json", {"references": refs, "books": list(components.BOOKS)})
            atomic_json(comp / "protocol.json", {"frozenAt": datetime.now(timezone.utc).isoformat(), "spec": components.SPEC,
                "books": list(components.BOOKS), "code": components.code_hashes(), "references": refs,
                "plan": {"path": str(comp / "PLAN.json"), "sha256": base.sha(comp / "PLAN.json")}, "engineCompatibility": compatibility})
            component = components.run(comp / "protocol.json")
            refs = {**refs, "component": reference(comp)}
            candidate = root / "candidate"
            candidate.mkdir(mode=0o700)
            atomic_json(candidate / "PLAN.json", {"references": refs, "books": list(recency.BOOKS), "spec": recency.SPEC})
            protocol = candidate / "protocol.json"
            atomic_json(protocol, {"frozenAt": datetime.now(timezone.utc).isoformat(), "spec": recency.SPEC,
                "books": list(recency.BOOKS), "code": recency.code_hashes(), "references": refs,
                "plan": {"path": str(candidate / "PLAN.json"), "sha256": base.sha(candidate / "PLAN.json")}})
            with patch("research.architecture.fit_at", side_effect=AssertionError("Old control refit")), \
                 patch("research.components.fit_components", side_effect=AssertionError("Old component refit")), \
                 patch("pickle.load", side_effect=AssertionError("Old model loaded")), \
                 patch("socket.socket.connect", side_effect=AssertionError("Network")):
                result = recency.run(protocol)
            self.assertEqual(result["fitCount"], baseline["fitCount"])
            self.assertTrue(result["predictiveAcceptanceChecks"]["same_scored_coverage"])
            self.assertFalse(result["livePromotionAllowed"])
            for slip in ["0.0", "5.0", "10.0"]:
                for book, summary in component["comparison"][slip].items():
                    self.assertEqual(result["comparison"][slip][book], summary)
            receipts = json.loads((candidate / "run/evidence-receipts.json").read_text())
            fits = {f["signal"]: f for f in json.loads((candidate / "run/fits.json").read_text())}
            for receipt in receipts:
                evidence = read_evidence(candidate / "run/calibration" / receipt["signal"], receipt["identitySha256"],
                    receipt["payloadSha256"], as_of=datetime.now(timezone.utc).isoformat())
                self.assertEqual(evidence["contract"]["mode"], "retrospective")
                self.assertTrue(all(s["availability"] == "unknown" for s in evidence["contract"]["sources"]))
                self.assertEqual(evidence["summary"]["calibration"]["missingPredictions"], 0)
                self.assertEqual(evidence["summary"]["train"]["missingPredictions"], 0)
                self.assertEqual(evidence["summary"]["calibration"]["rows"], fits[receipt["signal"]]["calibrationRows"])
                self.assertEqual(evidence["summary"]["forecast"]["rows"], 0)
                cal = [r for r in evidence["rows"] if r["partition"] == "calibration"]
                residual = pd.DataFrame({"date": [r["session"] for r in cal], "error": [r["label"] - r["calibrated"] for r in cal]})
                self.assertAlmostEqual(residual.groupby("date").error.mean().mean(), 0.)
            panel, _ = base.architecture_panel(frames, sessions, calendar_source="Synthetic", price_basis="Synthetic")
            signal = sessions[-40]
            first, fit, _, _ = recency.fit_weighted(panel, signal, sessions)
            self.assertLess(fit["trainLastOutcome"], fit["calibrationStart"])
            self.assertLess(fit["calibrationLastOutcome"], str(signal.date()))
            self.assertLess(fit["dateWeightEffectiveSize"], fit["trainDates"])
            changed = {s: f.copy() for s, f in frames.items()}
            for f in changed.values():
                f.loc[f.index > signal, ["open", "high", "low", "close"]] *= 2
                f.loc[f.index > signal, "volume"] *= 3
            altered, _ = base.architecture_panel(changed, sessions, calendar_source="Synthetic", price_basis="Synthetic")
            second, _, _, _ = recency.fit_weighted(altered, signal, sessions)
            np.testing.assert_array_equal(first.weightedMean, second.weightedMean)
            last, _, _, _ = recency.fit_weighted(panel, sessions[-1], sessions)
            self.assertTrue(last.actual.isna().all())
            self.assertEqual(len(last), 4)
            for p in (candidate / "run").rglob("*"):
                if p.is_file():
                    self.assertEqual(p.stat().st_mode & 0o777, 0o600)
            with patch.object(recency, "evaluate", side_effect=AssertionError("Refit/replay")):
                self.assertTrue(recency.run(protocol)["cacheHit"])
            seal = next((candidate / "run/calibration").rglob("*.json"))
            seal.write_text("{}")
            with self.assertRaisesRegex(ValueError, "Immutable"):
                recency.run(protocol)


if __name__ == "__main__":
    unittest.main()
