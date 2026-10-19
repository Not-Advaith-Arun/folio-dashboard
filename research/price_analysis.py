"""Price-only analysis identity and reviewed legacy-cache reuse.

Cache reads never unpickle models. News/ledger implementation is deliberately
outside training identity; estimator/features/audit/policy/runtime remain inside.
"""
from __future__ import annotations
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile

import joblib
import numpy as np
import pandas as pd
import sklearn

# Populated only after source-equivalence review of the frozen predecessor.
LEGACY_IMPLEMENTATIONS = {
    "c05c178bfcfe70fb99cc4b7374694fc2b69a7e7a8f66f9284472acd7e9c09638":
        "ac4194180d145993354d94514ae23914b4288fe9fc3fc11cc5ebf45bcd1fb29c",
}


def training_code_hash():
    from research import engine
    source = ast.parse(Path(engine.__file__).read_text())
    names = {"feature_panel", "estimators", "ranked_blend", "rank_ic", "performance", "walk_forward"}
    nodes = [n for n in source.body if isinstance(n, ast.FunctionDef) and n.name in names]
    imports = [n for n in source.body if isinstance(n, ast.ImportFrom) and (n.module or "").startswith("sklearn")]
    own = ast.parse(Path(__file__).read_text())
    fit = next(n for n in own.body if isinstance(n, ast.FunctionDef) and n.name == "fit_analysis")
    payload = json.dumps({"functions": [ast.dump(n, include_attributes=False) for n in nodes + imports + [fit]],
                          "features": engine.FEATURES, "horizon": engine.HORIZON, "version": engine.VERSION}, sort_keys=True).encode()
    return hashlib.sha256(payload + Path(__file__).with_name("audit.py").read_bytes()
                          + Path(__file__).with_name("policy.py").read_bytes()).hexdigest()


def analysis_key(history_hash, code_hash, config, top_k):
    from research.engine import VERSION
    from research.audit import AUDIT_VERSION
    return hashlib.sha256(json.dumps({"history": history_hash, "code": code_hash,
        "years": config["years"], "topK": top_k, "costBps": config["costBps"], "version": VERSION,
        "audit": AUDIT_VERSION, "libraries": [np.__version__, pd.__version__, sklearn.__version__]}, sort_keys=True).encode()).hexdigest()


def model_hashes(root, analysis):
    hashes = {}
    for name in ["modelId", "auditId"]:
        model_id = analysis[name]
        if not re.fullmatch(r"[0-9a-f]{12}", model_id):
            raise ValueError("Invalid cached model identity")
        path = root / "models" / f"{model_id}.joblib"
        hashes[model_id] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def fit_analysis(config, root, frames, panel, top_k, code_hash, progress):
    from research.engine import (FEATURES, VERSION, atomic_json, estimators,
                                 ranked_blend, walk_forward)
    mature = panel.dropna(subset=["target", "labelEnd"])
    backtest, mature = walk_forward(panel, top_k, config["costBps"], progress)
    progress("Fitting final models on matured outcomes", 73)
    latest_date = panel.date.max()
    latest = panel[panel.date == latest_date].copy()
    fingerprint = hashlib.sha256(pd.util.hash_pandas_object(mature[["date", "symbol", *FEATURES, "target"]], index=False).values.tobytes()).hexdigest()
    model_id = hashlib.sha256(json.dumps({"version": VERSION, "symbols": sorted(frames), "data": fingerprint, "features": FEATURES, "code": code_hash}, sort_keys=True).encode()).hexdigest()[:12]
    fitted = estimators()
    for name, model in fitted.items():
        model.fit(mature[FEATURES], mature.target)
        latest[name] = model.predict(latest[FEATURES])
    model_dir = root / "models"
    model_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    with tempfile.NamedTemporaryFile(dir=model_dir, delete=False) as f:
        model_temp = f.name
    joblib.dump({"models": fitted, "features": FEATURES, "dataFingerprint": fingerprint, "version": VERSION}, model_temp)
    os.chmod(model_temp, 0o600)
    os.replace(model_temp, model_dir / f"{model_id}.joblib")
    latest["Ensemble"] = ranked_blend(latest, list(estimators()))
    from research.audit import run_audit, AUDIT_VERSION, RICH_FEATURES
    audit, audit_models = run_audit(panel, top_k, config["costBps"], progress)
    audit_fingerprint = hashlib.sha256(pd.util.hash_pandas_object(mature[["date", "symbol", *RICH_FEATURES, "target"]], index=False).values.tobytes()).hexdigest()
    audit_id = hashlib.sha256(json.dumps({"version": AUDIT_VERSION, "data": audit_fingerprint, "topK": top_k, "costBps": config["costBps"], "code": code_hash}, sort_keys=True).encode()).hexdigest()[:12]
    audit["modelId"] = audit_id
    audit["dataFingerprint"] = audit_fingerprint
    with tempfile.NamedTemporaryFile(dir=model_dir, delete=False) as f:
        audit_temp = f.name
    joblib.dump({**audit_models, "dataFingerprint": audit_fingerprint}, audit_temp)
    os.chmod(audit_temp, 0o600)
    os.replace(audit_temp, model_dir / f"{audit_id}.joblib")
    analysis = {"backtest": backtest, "audit": audit, "modelId": model_id, "auditId": audit_id, "fingerprint": fingerprint, "latest": latest[["symbol", "close", "Ensemble", *estimators()]].to_dict("records")}
    return analysis


def get_analysis(config, root, frames, progress):
    from research.engine import feature_panel, atomic_json
    panel, _ = feature_panel(frames)
    top_k = min(config["topK"], len(frames))
    code_hash = training_code_hash()
    history_hash = hashlib.sha256(pd.util.hash_pandas_object(panel.sort_values(["date", "symbol"]), index=False).values.tobytes()).hexdigest()
    key = analysis_key(history_hash, code_hash, config, top_k)
    path = root / "analyses-v2" / f"{key}.json"
    legacy = LEGACY_IMPLEMENTATIONS.get(code_hash)
    old_key = analysis_key(history_hash, legacy, config, top_k) if legacy else None
    old_path = root / "analyses" / f"{old_key}.json" if old_key else None
    candidate = path if path.exists() else old_path if old_path and old_path.exists() else None
    if candidate:
        raw = candidate.read_bytes()
        cached = json.loads(raw)
        expected = key if candidate == path else old_key
        analysis = cached["data"]
        checksum = hashlib.sha256(json.dumps(analysis, sort_keys=True).encode()).hexdigest()
        if cached["key"] != expected or cached["checksum"] != checksum:
            raise ValueError("Analysis cache integrity failure; original evidence preserved")
        hashes = model_hashes(root, analysis)
        if candidate == path and hashes != cached["modelHashes"]:
            raise ValueError("Model artifact integrity failure")
        if candidate != path:
            # Adopt exact cached values, never relabel them as a newly fitted model.
            atomic_json(path, {"key": key, "checksum": checksum, "data": analysis, "modelHashes": hashes,
                               "legacy": {"key": old_key, "analysisSha256": hashlib.sha256(raw).hexdigest(),
                                          "implementationSha256": legacy}})
        progress("Reusing unchanged price analysis; refreshing news and forward outcomes", 85)
    else:
        analysis = fit_analysis(config, root, frames, panel, top_k, code_hash, progress)
        hashes = model_hashes(root, analysis)
        atomic_json(path, {"key": key, "checksum": hashlib.sha256(json.dumps(analysis, sort_keys=True).encode()).hexdigest(),
                           "data": analysis, "modelHashes": hashes})
    return analysis, panel, key, hashes
