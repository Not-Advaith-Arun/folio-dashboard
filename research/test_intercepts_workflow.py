"""Synthetic weighted fit -> calibration seal -> replay -> immutable reuse."""
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from research import architecture as base, components, recency, intercepts as target
from research.calibration_evidence import read_evidence
from research.engine import atomic_json


class StockInterceptWorkflow(unittest.TestCase):
    def test_source_fit_evidence_replay_and_immutable_boundaries(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sessions = pd.bdate_range("2021-01-04", periods=820)
            rng, frames, inputs = np.random.default_rng(19), {}, []
            for i, symbol in enumerate("ABCDEFGHIJKL"):
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
            recent = root / "recent"
            recent.mkdir(mode=0o700)
            atomic_json(recent / "PLAN.json", {"references": refs, "books": list(recency.BOOKS), "spec": recency.SPEC})
            recent_protocol = {"frozenAt": datetime.now(timezone.utc).isoformat(), "spec": recency.SPEC,
                "books": list(recency.BOOKS), "code": recency.code_hashes(), "references": refs,
                "plan": {"path": str(recent / "PLAN.json"), "sha256": base.sha(recent / "PLAN.json")}}
            atomic_json(recent / "protocol.json", recent_protocol)
            recent_result = recency.run(recent / "protocol.json")
            refs = {**refs, "recency": reference(recent)}
            # Entirely synthetic contract; no market cache is opened by this test.
            baseline_rows = json.loads((original / "run/forecasts.json").read_text())
            design = {"referenceProtocols": refs, "membershipUniverse": list(frames), "gates": base.SPEC,
                "spec": {"continuousFeatures": base.FEATURES, "estimator": {"alpha": 100.}},
                "experimentId": target.SPEC["version"], "featureInputs": inputs, "executionInputs": inputs,
                "chronology": {"expectedCandidates": len(baseline_rows), "expectedNewFits": baseline["fitCount"],
                    "expectedScoredRows": baseline["predictionMetrics"]["count"],
                    "expectedScoredDates": baseline["predictionMetrics"]["signalDates"]}}
            atomic_json(root / "memberships.json", [])
            design["chronology"]["memberships"] = {"path": str(root / "memberships.json"), "sha256": base.sha(root / "memberships.json")}
            atomic_json(root / "DESIGN.json", design)
            design_pin = {"path": str(root / "DESIGN.json"), "sha256": base.sha(root / "DESIGN.json")}
            candidate = root / "candidate"
            candidate.mkdir(mode=0o700)
            atomic_json(candidate / "PLAN.json", {"design": design_pin})
            protocol = candidate / "protocol.json"
            atomic_json(protocol, {"frozenAt": datetime.now(timezone.utc).isoformat(), "spec": target.SPEC,
                "books": list(target.BOOKS), "code": target.code_hashes(), "references": refs,
                "symbols": list(frames), "design": design_pin,
                "runtime": {k: baseline["lineage"][k] for k in ["numpy", "pandas", "sklearn"]},
                "plan": {"path": str(candidate / "PLAN.json"), "sha256": base.sha(candidate / "PLAN.json")}})
            with patch("research.architecture.fit_at", side_effect=AssertionError("Old control refit")), \
                 patch("research.components.fit_components", side_effect=AssertionError("Old component refit")), \
                 patch("research.recency.fit_weighted", side_effect=AssertionError("Old recency refit")), \
                 patch("pickle.load", side_effect=AssertionError("Old model loaded")), \
                 patch("socket.socket.connect", side_effect=AssertionError("Network")):
                result = target.run(protocol)
            self.assertEqual(result["fitCount"], baseline["fitCount"])
            self.assertTrue(result["predictiveAcceptanceChecks"]["same_scored_coverage"])
            self.assertFalse(result["livePromotionAllowed"])
            for slip in ["0.0", "5.0", "10.0"]:
                for book, summary in recent_result["comparison"][slip].items():
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
            first, fit, _, _ = target.fit_intercepts(panel, signal, sessions, list(frames))
            self.assertLess(fit["trainLastOutcome"], fit["calibrationStart"])
            self.assertLess(fit["calibrationLastOutcome"], str(signal.date()))
            changed = {s: f.copy() for s, f in frames.items()}
            for f in changed.values():
                f.loc[f.index > signal, ["open", "high", "low", "close"]] *= 2
                f.loc[f.index > signal, "volume"] *= 3
            altered, _ = base.architecture_panel(changed, sessions, calendar_source="Synthetic", price_basis="Synthetic")
            second, _, _, _ = target.fit_intercepts(altered, signal, sessions, list(frames))
            np.testing.assert_array_equal(first.interceptMean, second.interceptMean)
            last, _, _, _ = target.fit_intercepts(panel, sessions[-1], sessions, list(frames))
            self.assertTrue(last.actual.isna().all())
            self.assertEqual(len(last), 12)
            # Hard calculation/design boundaries warrant direct checks inside
            # this spanning workflow: no dummy scaling or reference-stock bias.
            _, _, models, paths = target.fit_intercepts(panel, signal, sessions, list(frames))
            model = models["ridge"]
            train = paths["train"]
            w = 1. / train.groupby("date").symbol.transform("size").to_numpy()
            w /= w.mean()
            x = model.design(train)
            np.testing.assert_array_equal(x[:, 14:].sum(axis=1), np.ones(len(train)))
            self.assertTrue(set(np.unique(x[:, 14:])) <= {0., 1.})
            self.assertAlmostEqual(float(model.ridge.coef_[14:].sum()), 0., places=10)
            residual = train.actual.to_numpy() - model.predict(train)
            np.testing.assert_allclose(x.T @ (w * residual), 100 * model.ridge.coef_, atol=1e-8)
            self.assertAlmostEqual(float(w @ residual), 0., places=8)
            permuted, _, _, _ = target.fit_intercepts(panel, signal, sessions, list(reversed(frames)))
            np.testing.assert_allclose(first.interceptMean, permuted.interceptMean, atol=1e-12)
            bad = panel.copy();bad.loc[bad.symbol == "A", "symbol"] = "UNKNOWN"
            with self.assertRaisesRegex(ValueError, "Unknown"):
                target.fit_intercepts(bad, signal, sessions, list(frames))
            with self.assertRaisesRegex(ValueError, "Missing training"):
                target.fit_intercepts(panel[panel.symbol != "A"], signal, sessions, list(frames))
            with self.assertRaisesRegex(ValueError, "Frozen membership"):
                target.fit_intercepts(panel, signal, sessions, list(frames), {"memberships": []})
            constant = train.copy();constant[base.FEATURES[0]] = 1.
            constant_model = target.StockInterceptRidge(list(frames)).fit(constant, w)
            self.assertEqual(constant_model.scaler.scale_[0], 1.)
            receipt = receipts[0]
            with self.assertRaises(ValueError):
                read_evidence(candidate / "run/calibration" / receipt["signal"], receipt["identitySha256"],
                              receipt["payloadSha256"], as_of="2000-01-01T00:00:00Z")
            for p in (candidate / "run").rglob("*"):
                if p.is_file():
                    self.assertEqual(p.stat().st_mode & 0o777, 0o600)
            with patch.object(target, "evaluate", side_effect=AssertionError("Refit/replay")):
                self.assertTrue(target.run(protocol)["cacheHit"])
            # Symlink/nonprivate paths and incomplete attempts fail closed.
            link = root / "protocol-link.json";link.symlink_to(protocol)
            with self.assertRaisesRegex(ValueError, "Unsafe"):
                target.run(link)
            (candidate / "run/started.json").chmod(0o644)
            with self.assertRaisesRegex(ValueError, "Unsafe"):
                target.run(protocol)
            (candidate / "run/started.json").chmod(0o600)
            seal = next((candidate / "run/calibration").rglob("*.json"))
            seal.write_text("{}")
            with self.assertRaisesRegex(ValueError, "Immutable"):
                target.run(protocol)


if __name__ == "__main__":
    unittest.main()
