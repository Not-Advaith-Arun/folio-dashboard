"""Synthetic source -> actual15feature model -> immutable calibration/replay."""
from copy import deepcopy
from datetime import datetime, timezone
import json
import platform
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from research import architecture as base, components, recency, intercepts, slopes, dispersion, skewness as target
from research.calibration_evidence import read_evidence
from research.engine import atomic_json


class SkewnessWorkflow(unittest.TestCase):
    def test_source_fit_calibration_main_restart_and_immutable_boundaries(self):
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
            symbols = list(frames)
            original = root / "original"; original.mkdir(mode=0o700)
            compatibility = {"previous": base.sha(Path(base.__file__).with_name("engine.py")),
                             "current": base.sha(Path(base.__file__).with_name("engine.py"))}
            parent = {"architecture": base.SPEC, "books": base.BOOKS, "frozenAt": datetime.now(timezone.utc).isoformat(),
                "featureInputs": inputs, "executionInputs": inputs, "calendarExcluded": [],
                "start": str(sessions[-45].date()), "end": str(sessions[-1].date()), "calendarSource": "Synthetic weekdays only",
                "priceBasis": "Synthetic only", "capital": 100000., "slippageScenarios": [0., 5., 10.],
                "primarySlippage": 5., "signalStride": 20, "lots": dict.fromkeys(frames, 1),
                "evidenceLimits": ["Synthetic only"], "engineCompatibility": compatibility}
            atomic_json(original / "protocol.json", parent)
            baseline = base.run_experiment(original / "protocol.json")
            def reference(folder):
                return {"protocol": {"path": str(folder / "protocol.json"), "sha256": base.sha(folder / "protocol.json")},
                    "completed": {"path": str(folder / "run/completed.json"), "sha256": base.sha(folder / "run/completed.json")}}
            refs = dict.fromkeys(["original", "normalized", "payoff"], reference(original))
            comp = root / "component"; comp.mkdir(mode=0o700)
            atomic_json(comp / "PLAN.json", {"references": refs, "books": list(components.BOOKS)})
            atomic_json(comp / "protocol.json", {"frozenAt": datetime.now(timezone.utc).isoformat(), "spec": components.SPEC,
                "books": list(components.BOOKS), "code": components.code_hashes(), "references": refs,
                "plan": {"path": str(comp / "PLAN.json"), "sha256": base.sha(comp / "PLAN.json")}, "engineCompatibility": compatibility})
            components.run(comp / "protocol.json")
            refs = {**refs, "component": reference(comp)}
            recent = root / "recent"; recent.mkdir(mode=0o700)
            atomic_json(recent / "PLAN.json", {"references": refs, "books": list(recency.BOOKS), "spec": recency.SPEC})
            atomic_json(recent / "protocol.json", {"frozenAt": datetime.now(timezone.utc).isoformat(), "spec": recency.SPEC,
                "books": list(recency.BOOKS), "code": recency.code_hashes(), "references": refs,
                "plan": {"path": str(recent / "PLAN.json"), "sha256": base.sha(recent / "PLAN.json")}})
            recency.run(recent / "protocol.json")
            refs = {**refs, "recency": reference(recent)}
            def candidate(folder, module, reference_set, membership_file, feature_names):
                folder.mkdir(mode=0o700)
                design = {"referenceProtocols": reference_set, "membershipUniverse": symbols, "gates": base.SPEC,
                    "spec": {"continuousFeatures": feature_names, "estimator": {"alpha": 100.}},
                    "experimentId": module.SPEC["version"], "featureInputs": inputs, "executionInputs": inputs,
                    "dispersionRecoveryReference": recovery_reference if module is target else {},
                    "chronology": {"expectedCandidates": len(json.loads((original / "run/forecasts.json").read_text())),
                        "expectedNewFits": baseline["fitCount"], "expectedScoredRows": baseline["predictionMetrics"]["count"],
                        "expectedScoredDates": baseline["predictionMetrics"]["signalDates"],
                        "memberships": {"path": str(membership_file), "sha256": base.sha(membership_file)}}}
                atomic_json(folder / "DESIGN.json", design)
                design_pin = {"path": str(folder / "DESIGN.json"), "sha256": base.sha(folder / "DESIGN.json")}
                atomic_json(folder / "PLAN.json", {"design": design_pin})
                atomic_json(folder / "protocol.json", {"frozenAt": datetime.now(timezone.utc).isoformat(),
                    "spec": module.SPEC, "books": list(module.BOOKS), "code": module.code_hashes(), "references": reference_set,
                    "runtimePython": platform.python_version(), "symbols": symbols, "design": design_pin,
                    "runtime": {k: baseline["lineage"][k] for k in ["numpy", "pandas", "sklearn"]},
                    "plan": {"path": str(folder / "PLAN.json"), "sha256": base.sha(folder / "PLAN.json")}})
                return folder / "protocol.json"
            empty = root / "empty-memberships.json"; atomic_json(empty, [])
            nested = root / "intercepts"
            intercepts.run(candidate(nested, intercepts, refs, empty, base.FEATURES))
            members = []
            for receipt in json.loads((nested / "run/evidence-receipts.json").read_text()):
                e = read_evidence(nested / "run/calibration" / receipt["signal"], receipt["identitySha256"],
                    receipt["payloadSha256"], as_of=datetime.now(timezone.utc).isoformat())
                members.append({"signal": receipt["signal"], "memberships": e["rows"]})
            member_file = root / "candidate-memberships.json"; atomic_json(member_file, members)
            refs = {**refs, "stockIntercept": reference(nested)}
            slope = root / "slopes"
            old_result = slopes.run(candidate(slope, slopes, refs, member_file, base.FEATURES))
            refs = {**refs, "stockSlope": reference(slope)}
            prior = root / "dispersion"
            dispersion.run(candidate(prior, dispersion, refs, member_file, dispersion.FEATURES))
            old_result = json.loads((prior / "run/results.json").read_text())
            (prior / "run/completed.json").unlink() # SYNTHETIC interrupted-archive fixture only.
            def pin(path):
                return {"path": str(path), "sha256": base.sha(path)}
            for name in ["dispersion.py", "test_dispersion_workflow.py"]:
                (prior / ("attempt-1-" + name)).write_bytes(Path(dispersion.__file__).with_name(name).read_bytes())
            recovery = prior / "post-fit-recovery"; recovery.mkdir(mode=0o700)
            atomic_json(recovery / "results.json", old_result)
            manifest = {str(p.relative_to(prior)): base.sha(p) for p in prior.rglob("*")
                        if p.is_file() and "post-fit-recovery" not in p.parts}
            atomic_json(prior / "FAILED-ATTEMPT-MANIFEST.json", manifest)
            atomic_json(prior / "RECOVERY-PROTOCOL.json", {"marketProtocol": pin(prior / "protocol.json"),
                "failedAttemptManifest": pin(prior / "FAILED-ATTEMPT-MANIFEST.json")})
            atomic_json(prior / "RECOVERY-BINDING.json", {
                "recoveryProtocolSha256": base.sha(prior / "RECOVERY-PROTOCOL.json"),
                "originalProtocolSha256": base.sha(prior / "protocol.json")})
            atomic_json(recovery / "completed.json", {"artifacts": {"results.json": base.sha(recovery / "results.json")},
                "originalAttemptRemainsInterrupted": True,
                "recoveryProtocolSha256": base.sha(prior / "RECOVERY-PROTOCOL.json")})
            atomic_json(prior / "COMPLETED.json", {"files": {str(p.relative_to(prior)): base.sha(p)
                for p in prior.rglob("*") if p.is_file()}})
            recovery_reference = {"overall": pin(prior / "COMPLETED.json"),
                "failedAttemptManifest": pin(prior / "FAILED-ATTEMPT-MANIFEST.json"),
                "attemptProtocol": pin(prior / "protocol.json"), "recoveryProtocol": pin(prior / "RECOVERY-PROTOCOL.json"),
                "recoveryBinding": pin(prior / "RECOVERY-BINDING.json"), "completed": pin(recovery / "completed.json"),
                "result": pin(recovery / "results.json"),
                "attemptArtifacts": [{"path": str(prior/n), "sha256": h} for n,h in manifest.items()],
                "recoveredArtifacts": [pin(recovery / "results.json")]}
            folder = root / "candidate"
            protocol = candidate(folder, target, refs, member_file, target.FEATURES)
            # Ancestor creation is wholly synthetic. From here, no old fitter,
            # model deserializer or network may run, including during reuse.
            with patch.object(base, "fit_at", side_effect=AssertionError("Old fit")), \
                 patch.object(components, "fit_components", side_effect=AssertionError("Old fit")), \
                 patch.object(recency, "fit_weighted", side_effect=AssertionError("Old fit")), \
                 patch.object(intercepts, "fit_intercepts", side_effect=AssertionError("Old fit")), \
                 patch.object(slopes, "fit_slopes", side_effect=AssertionError("Old fit")), \
                 patch.object(dispersion, "fit_dispersion", side_effect=AssertionError("Old fit")), \
                 patch("pickle.load", side_effect=AssertionError("Model load")), \
                 patch("pickle.loads", side_effect=AssertionError("Model load")), \
                 patch("socket.socket.connect", side_effect=AssertionError("Network")):
                result = target.run(protocol)
            self.assertEqual(result["fitCount"], baseline["fitCount"])
            self.assertTrue(result["predictiveAcceptanceChecks"]["same_scored_coverage"])
            for slip in ["0.0", "5.0", "10.0"]:
                for book, value in old_result["comparison"][slip].items():
                    self.assertEqual(result["comparison"][slip][book], value)
            receipts = json.loads((folder / "run/evidence-receipts.json").read_text())
            for receipt in receipts:
                e = read_evidence(folder / "run/calibration" / receipt["signal"], receipt["identitySha256"],
                    receipt["payloadSha256"], as_of=datetime.now(timezone.utc).isoformat())
                self.assertTrue(all(s["availability"] == "unknown" for s in e["contract"]["sources"]))
                self.assertEqual(e["summary"]["calibration"]["missingPredictions"], 0)
                self.assertEqual(e["summary"]["train"]["missingPredictions"], 0)
                self.assertEqual(e["summary"]["forecast"]["rows"], 0)
            panel, _ = target.skewness_panel(frames, sessions, symbols, calendar_source="Synthetic", price_basis="Synthetic")
            old_panel, _ = base.architecture_panel(frames, sessions, calendar_source="Synthetic", price_basis="Synthetic")
            pd.testing.assert_frame_equal(panel.drop(columns="stockSkewness60"), old_panel)
            # Critical arithmetic, independently known empirical moments.
            values = [-.01] * 40 + [.02] * 20
            self.assertAlmostEqual(target.population_skewness(values), 1 / np.sqrt(2), places=13)
            self.assertEqual(target.population_skewness([-.01, .01] * 30), 0.)
            self.assertTrue(np.isnan(target.population_skewness([.1] * 60)))
            self.assertTrue(np.isnan(target.population_skewness([.1] * 59)))
            self.assertTrue(np.isnan(target.population_skewness([np.nan] * 60)))
            self.assertAlmostEqual(target.population_skewness([-v for v in values]), -1 / np.sqrt(2), places=13)
            linear = {}
            for k, symbol in enumerate(symbols):
                c = [100. + k]
                for i in range(len(sessions) - 1):
                    c.append(c[-1] * (1 + values[i % 60]))
                c = np.asarray(c)
                linear[symbol] = pd.DataFrame({"open": c, "close": c, "high": c+1, "low": c-1,
                    "volume": 100.+np.arange(len(c))}, index=sessions)
            lp, _ = target.skewness_panel(linear, sessions, symbols, calendar_source="Synthetic", price_basis="Synthetic")
            np.testing.assert_allclose(lp[lp.date==sessions[250]].stockSkewness60, 1 / np.sqrt(2), rtol=1e-12, atol=1e-14)
            self.assertTrue(lp[lp.date==sessions[198]].stockSkewness60.isna().all())
            self.assertTrue(lp[lp.date==sessions[199]].stockSkewness60.notna().all())
            flat = {s: f.copy() for s,f in linear.items()}
            for f in flat.values():
                f.loc[:, ["open", "close", "high", "low"]] = [100., 100., 101., 99.]
            fp, _ = target.skewness_panel(flat, sessions, symbols, calendar_source="Synthetic", price_basis="Synthetic")
            self.assertTrue(fp[fp.featureReady].stockSkewness60.isna().all())
            # Whole candidate preflight fails before ANY model fitting.
            candidate_design = json.loads((folder / "DESIGN.json").read_text())
            with patch.object(target.SkewnessRidge, "fit", side_effect=AssertionError("Premature fit")):
                with self.assertRaisesRegex(ValueError, "before ANY fit"):
                    target.preflight_memberships(fp, candidate_design, symbols)
            signal = sessions[-40]
            first, fit, models, paths = target.fit_skewness(panel, signal, sessions, symbols)
            self.assertLess(fit["trainLastOutcome"], fit["calibrationStart"])
            self.assertLess(fit["calibrationLastOutcome"], str(signal.date()))
            train = paths["train"]; model = models["ridge"]
            weights = 1./train.groupby("date").symbol.transform("size").to_numpy(); weights /= weights.mean()
            x = model.design(train); residual = train.actual.to_numpy()-model.predict(train)
            self.assertEqual(x.shape[1], 15)
            np.testing.assert_allclose(model.scaler.mean_, np.average(train[target.FEATURES], axis=0, weights=weights))
            np.testing.assert_allclose(x.T@(weights*residual), 100*model.ridge.coef_, atol=1e-8)
            self.assertAlmostEqual(float(weights@residual), 0., places=8)
            # Nonuniform weights verify the scaler itself, while the actual
            # chronological workflow above retains original equal-date weights.
            nw = np.linspace(.5, 1.5, len(train)); m = target.SkewnessRidge(symbols).fit(train, nw)
            np.testing.assert_allclose(m.scaler.mean_, np.average(train[target.FEATURES], axis=0, weights=nw))
            constant = train.copy();constant["stockSkewness60"]=0.
            self.assertEqual(target.SkewnessRidge(symbols).fit(constant,weights).scaler.scale_[-1],1.)
            changed = {s:f.copy() for s,f in frames.items()}
            for f in changed.values():f.loc[f.index>signal,["open","close","high","low"]]*=2
            altered,_=target.skewness_panel(changed,sessions,symbols,calendar_source="Synthetic",price_basis="Synthetic")
            second,_,_,_=target.fit_skewness(altered,signal,sessions,symbols)
            np.testing.assert_array_equal(first.skewnessMean,second.skewnessMean)
            altered=panel.copy();altered.loc[altered.labelEnd>=signal,"actual"]=12345.
            second,_,_,_=target.fit_skewness(altered,signal,sessions,symbols)
            np.testing.assert_array_equal(first.skewnessMean,second.skewnessMean)
            last,_,_,_=target.fit_skewness(panel,sessions[-1],sessions,symbols)
            self.assertEqual(len(last),12);self.assertTrue(last.actual.isna().all())
            permuted,_=target.skewness_panel(frames,sessions,symbols[::-1],calendar_source="Synthetic",price_basis="Synthetic")
            np.testing.assert_allclose(panel.stockSkewness60,permuted.stockSkewness60,rtol=1e-12,atol=1e-14)
            broken=panel.copy();broken.loc[(broken.date==train.date.min())&(broken.symbol==symbols[0]),"stockSkewness60"]=np.nan
            with patch.object(target.SkewnessRidge,"fit",side_effect=AssertionError("Fitted before gate")):
                with self.assertRaisesRegex(ValueError,"Unavailable historical skewness"):target.fit_skewness(broken,signal,sessions,symbols)
                with self.assertRaisesRegex(ValueError,"Frozen membership"):target.fit_skewness(panel,signal,sessions,symbols,{"memberships":[]})
            missing={s:f.copy() for s,f in frames.items()};missing[symbols[0]]=missing[symbols[0]].drop(signal)
            mp,_=target.skewness_panel(missing,sessions,symbols,calendar_source="Synthetic",price_basis="Synthetic")
            self.assertTrue(mp[(mp.date==signal)&(mp.symbol==symbols[0])].stockSkewness60.isna().all())
            self.assertTrue(mp[(mp.date==signal)&(mp.symbol!=symbols[0])].stockSkewness60.notna().all())
            with self.assertRaises(ValueError):target.SkewnessRidge(symbols).fit(train[train.symbol!=symbols[0]],weights[:len(train[train.symbol!=symbols[0]])])
            bad=panel.copy();bad.loc[bad.symbol==symbols[0],"symbol"]="UNKNOWN"
            with self.assertRaises(ValueError):target.fit_skewness(bad,signal,sessions,symbols)
            with patch.object(dispersion, "run", side_effect=AssertionError("Old executor")), \
                 patch("pickle.load", side_effect=AssertionError("Model load")), \
                 patch("socket.socket.connect", side_effect=AssertionError("Network")):
                target.load_dispersion_recovery(recovery_reference, refs)
                marker = prior / "run/completed.json"; marker.write_text("{}")
                with self.assertRaisesRegex(ValueError, "remain interrupted"):
                    target.load_dispersion_recovery(recovery_reference, refs)
                marker.unlink()
                corrupted = deepcopy(recovery_reference); corrupted["recoveredArtifacts"] = []
                with self.assertRaises(ValueError): target.load_dispersion_recovery(corrupted, refs)
                sealed_result = recovery / "results.json"; saved = sealed_result.read_bytes()
                sealed_result.write_bytes(saved + b" ")
                with self.assertRaises(ValueError): target.load_dispersion_recovery(recovery_reference, refs)
                sealed_result.write_bytes(saved)
                unsafe = deepcopy(recovery_reference); unsafe["result"]["path"] = str(prior / "run/results.json")
                with self.assertRaises(ValueError): target.load_dispersion_recovery(unsafe, refs)
            rr=result["replayEvidence"]; archive=folder/"run/replay-evidence"
            payload=target.replay_evidence.read(archive,rr["identitySha256"],rr["payloadSha256"],as_of=datetime.now(timezone.utc).isoformat())
            self.assertEqual(len(payload["records"]),8)
            self.assertTrue(all(v["status"]=="complete" for v in payload["summary"]))
            self.assertFalse(payload["livePromotionAllowed"])
            # Independent synthetic rotation forces nonzero orders/fees,
            # unsettled sale cash, unfilled orders and different restart equity.
            from research.test_replay_evidence_workflow import fixture
            contract,records=fixture();proof=root/"rotation-proof";proof.mkdir(mode=0o700)
            sealed=target.replay_evidence.seal(proof,contract,records)
            evidence=target.replay_evidence.read(proof,sealed["identitySha256"],sealed["payloadSha256"],as_of=datetime.now(timezone.utc).isoformat())
            self.assertEqual(evidence["summary"][0]["status"],"complete")
            self.assertNotEqual(records[0]["replay"]["ending"],records[1]["replay"]["ending"])
            self.assertGreater(len(records[0]["replay"]["orders"]),0)
            self.assertTrue(any(v["unsettled"]>0 for v in records[0]["replay"]["curve"]))
            self.assertGreater(evidence["summary"][0]["turnover"],0)
            # Exercise the actual adapter with nonempty NumPy allocator
            # weights, rather than relying on model books that may hold cash.
            rotation = root/"rotation-adapter";rotation.mkdir(mode=0o700)
            rotation_parent = {**parent,"executionInputs":[inputs[0],inputs[1]],
                "lots":records[0]["lots"],"end":records[0]["end"],"capital":1000.}
            ds=[{"signal": "2024-01-02","entry":records[0]["start"],"weights":{"A":np.float64(1.)}}]
            rotation_decisions={b:ds for b in target.BOOKS}
            main={str(s):{b:{**deepcopy(records[0]["replay"]),"slippageBpsPerSide":s}
                         for b in target.BOOKS} for s in [0.,5.,10.]}
            # Keep scenario convention exact: reuse the actual5bps rotation
            # for the adapter proof, with an explicit single5bps contract.
            rotation_parent["slippageScenarios"]=[5.]
            main={"5.0":main["5.0"]}
            restart={b:deepcopy(records[0]["replay"]) for b in target.BOOKS}
            adapter=target.capture_replays(rotation,protocol,rotation_parent,rotation_decisions,main,restart,ds[0]["entry"])
            native=target.replay_evidence.read(rotation/"replay-evidence",adapter["identitySha256"],adapter["payloadSha256"],as_of=datetime.now(timezone.utc).isoformat())
            self.assertTrue(all(v["status"]=="complete" for v in native["summary"]))
            bad=deepcopy(records);bad[0]["replay"]["fees"]+=1
            with self.assertRaises(ValueError):target.replay_evidence.seal(proof,contract,bad)
            with self.assertRaises(ValueError):target.replay_evidence.read(archive,rr["identitySha256"],rr["payloadSha256"],as_of="2000-01-01T00:00:00Z")
            with patch.object(target,"evaluate",side_effect=AssertionError("No repeat fit/replay")):
                self.assertTrue(target.run(protocol)["cacheHit"])
            completed=folder/"run/completed.json";body=completed.read_bytes();completed.unlink()
            with patch.object(target,"evaluate",side_effect=AssertionError("No interrupted refit")):
                with self.assertRaisesRegex(ValueError,"Incomplete attempt"):target.run(protocol)
            completed.write_bytes(body)
            link=root/"protocol-link";link.symlink_to(protocol)
            with self.assertRaises(ValueError):target.run(link)
            seal=archive/(rr["identitySha256"]+".json");body=seal.read_bytes();seal.write_bytes(body+b" ")
            with self.assertRaises(ValueError):target.run(protocol)
            seal.write_bytes(body)
            for path in (folder/"run").rglob("*"):
                if path.is_file():self.assertEqual(path.stat().st_mode&0o777,0o600)


if __name__ == "__main__":unittest.main()
