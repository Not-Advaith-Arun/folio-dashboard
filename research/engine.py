"""Free, local research pipeline. Forecasts are experiments, never broker orders."""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.parse import urlencode
from zoneinfo import ZoneInfo

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np
import pandas as pd
import requests
import joblib
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

VERSION = "folio-research-1"
TEXT_REVISION = "4556d13015211d73dccd3fdd39d39232506f3e43"
HORIZON = 5
FEATURES = ["return5", "return20", "return60", "volatility20", "range", "volumeRatio", "market20"]
DEFAULT_SYMBOLS = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS", "ITC.NS", "LT.NS", "SBIN.NS", "BHARTIARTL.NS", "HINDUNILVR.NS", "SUNPHARMA.NS", "MARUTI.NS"]
NAMES = {"RELIANCE.NS": "Reliance Industries", "TCS.NS": "Tata Consultancy Services", "HDFCBANK.NS": "HDFC Bank", "ICICIBANK.NS": "ICICI Bank", "INFY.NS": "Infosys", "ITC.NS": "ITC Limited", "LT.NS": "Larsen Toubro", "SBIN.NS": "State Bank of India", "BHARTIARTL.NS": "Bharti Airtel", "HINDUNILVR.NS": "Hindustan Unilever", "SUNPHARMA.NS": "Sun Pharma", "MARUTI.NS": "Maruti Suzuki"}
_sentiment = None


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as f:
        json.dump(value, f, allow_nan=False, separators=(",", ":"))
        temp = f.name
    os.chmod(temp, 0o600)
    os.replace(temp, path)


def load_json(path: Path, default):
    return json.loads(path.read_text()) if path.exists() else default


def market_history(symbol, years, root: Path):
    """Bounded personal-use public data; retain the downloaded observation."""
    import yfinance as yf
    today = datetime.now(ZoneInfo("Asia/Kolkata")).date()
    cache = root / "prices" / f"{symbol}-{years}-{today}.json"
    if cache.exists():
        frame = pd.DataFrame(load_json(cache, []))
        frame["date"] = pd.to_datetime(frame["date"])
        return frame.set_index("date")
    frame = yf.Ticker(symbol).history(start=str(today - timedelta(days=365 * years + 100)), end=str(today), auto_adjust=True, actions=False, timeout=15)
    if frame.empty:
        raise ValueError(f"No price history for {symbol}.")
    frame.index = pd.DatetimeIndex(frame.index).tz_localize(None).normalize()
    frame = frame.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
    frame = frame[~frame.index.duplicated()].sort_index()
    frame = frame[(frame[["open", "high", "low", "close"]] > 0).all(axis=1) & (frame.volume >= 0)].dropna()
    if len(frame) < 400:
        raise ValueError(f"{symbol} needs at least 400 daily observations; found {len(frame)}.")
    frame.index.name = "date"
    records = frame.reset_index().assign(date=lambda x: x.date.dt.strftime("%Y-%m-%d")).to_dict("records")
    atomic_json(cache, records)
    return frame


def feature_panel(frames):
    # A complete common calendar avoids forward-filled prices and fictitious fills.
    dates = sorted(set.intersection(*(set(f.index) for f in frames.values())))
    if len(dates) < 400:
        raise ValueError("Less than 400 common trading dates. Choose stocks with longer overlapping histories.")
    opens = pd.DataFrame({s: f.loc[dates, "open"] for s, f in frames.items()}, index=dates)
    closes = pd.DataFrame({s: f.loc[dates, "close"] for s, f in frames.items()}, index=dates)
    realized = opens.shift(-(HORIZON + 1)) / opens.shift(-1) - 1
    benchmark = realized.mean(axis=1)
    market20 = closes.pct_change(20).mean(axis=1)
    market_returns = closes.pct_change().mean(axis=1)
    market_vol = market_returns.rolling(20).std() * np.sqrt(252)
    market_trend = (closes / closes.rolling(200).mean() - 1).mean(axis=1)
    rows = []
    for symbol, source in frames.items():
        f = source.loc[dates]
        r = f.close.pct_change()
        x = pd.DataFrame({"date": dates, "symbol": symbol})
        for days in [5, 20, 60]:
            x[f"return{days}"] = f.close.pct_change(days).to_numpy()
        x["volatility20"] = r.rolling(20).std().to_numpy()
        x["range"] = ((f.high - f.low) / f.close).to_numpy()
        x["volumeRatio"] = (f.volume / f.volume.rolling(20).mean().replace(0, np.nan)).to_numpy()
        x["market20"] = market20.to_numpy()
        x["actual"] = realized[symbol].to_numpy()
        x["target"] = (realized[symbol] - benchmark).to_numpy()
        x["labelEnd"] = pd.Series(dates).shift(-(HORIZON + 1)).to_numpy()
        x["close"] = f.close.to_numpy()
        for days in [1, 10, 120]:
            x[f"return{days}"] = f.close.pct_change(days).to_numpy()
        x["relative20"] = x.return20 - x.market20
        x["trend200"] = (f.close / f.close.rolling(200).mean() - 1).to_numpy()
        x["marketTrend"] = market_trend.to_numpy()
        x["marketVol"] = market_vol.to_numpy()
        rows.append(x)
    panel = pd.concat(rows, ignore_index=True).replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURES)
    return panel, opens


def estimators():
    return {
        "Elastic Net": make_pipeline(StandardScaler(), ElasticNet(alpha=0.0005, l1_ratio=0.2, max_iter=2000)),
        "Random Forest": RandomForestRegressor(n_estimators=80, max_depth=6, min_samples_leaf=15, random_state=42, n_jobs=2),
        # Automatic early stopping would reserve random rows, inappropriate here.
        "Gradient Boosting": HistGradientBoostingRegressor(max_iter=100, max_leaf_nodes=12, l2_regularization=2, early_stopping=False, random_state=42),
    }


def ranked_blend(frame, columns):
    ranks = [frame.groupby("date")[c].rank(pct=True) for c in columns]
    return sum(ranks) / len(ranks)


def rank_ic(frame, column):
    correlations = []
    for _, g in frame.groupby("date"):
        a, b = g[column].rank().to_numpy(), g.target.rank().to_numpy()
        if np.std(a) > 0 and np.std(b) > 0:
            correlations.append(float(np.corrcoef(a, b)[0, 1]))
    return float(np.mean(correlations)) if correlations else 0.0


def performance(returns):
    equity = np.cumprod(1 + np.array(returns, dtype=float))
    peaks = np.maximum.accumulate(np.r_[1.0, equity])
    drawdown = np.r_[1.0, equity] / peaks - 1
    years = len(returns) * HORIZON / 252
    volatility = np.std(returns, ddof=1) if len(returns) > 1 else 0
    return {"totalReturn": float(equity[-1] - 1) if len(equity) else 0, "cagr": float(equity[-1] ** (1 / years) - 1) if years and len(equity) else 0, "maxDrawdown": float(drawdown.min()), "sharpe": float(np.mean(returns) / volatility * np.sqrt(252 / HORIZON)) if volatility else 0}


def walk_forward(panel, top_k, cost_bps, progress):
    mature = panel.dropna(subset=["target", "labelEnd"])
    dates = sorted(mature.date.unique())
    test_start = max(240, len(dates) - 252)
    test_dates = dates[test_start:]
    if len(test_dates) < 90:
        raise ValueError("At least 240 training and 90 evaluation dates are required.")
    parts, audits = [], []
    for i, fold in enumerate(np.array_split(test_dates, 3)):
        start = pd.Timestamp(fold[0])
        train = mature[(mature.date < start) & (mature.labelEnd < start)]
        test = mature[mature.date.isin(fold)].copy()
        progress(f"Training chronological fold {i + 1}/3", 35 + i * 12)
        for name, model in estimators().items():
            model.fit(train[FEATURES], train.target)
            test[name] = model.predict(test[FEATURES])
        test["Ensemble"] = ranked_blend(test, list(estimators()))
        parts.append(test)
        audits.append({"trainStart": str(train.date.min().date()), "trainEnd": str(train.date.max().date()), "lastTrainingOutcome": str(train.labelEnd.max().date()), "testStart": str(start.date()), "testEnd": str(pd.Timestamp(fold[-1]).date()), "trainingRows": len(train), "testRows": len(test)})
    oof = pd.concat(parts).sort_values(["date", "symbol"])
    columns = ["Ensemble", "Equal weight", "Momentum", *estimators()]
    returns = {c: [] for c in columns}
    curve = [{"date": str(pd.Timestamp(test_dates[0]).date()), **{c: 1.0 for c in columns}}]
    levels = {c: 1.0 for c in columns}
    # Consecutive five-session holding windows, executed at the following open.
    for date in test_dates[::HORIZON]:
        g = oof[oof.date == date]
        for c in columns:
            selected = g if c == "Equal weight" else g.nlargest(top_k, "return20" if c == "Momentum" else c)
            r = float(selected.actual.mean()) - cost_bps / 10000
            returns[c].append(r)
            levels[c] *= 1 + r
        curve.append({"date": str(pd.Timestamp(g.labelEnd.iloc[0]).date()), **levels})
    metrics = [{"name": c, **performance(returns[c]), "rankIC": rank_ic(oof, c) if c in estimators() or c == "Ensemble" else None} for c in columns]
    return {"folds": audits, "curve": curve, "metrics": metrics, "periods": len(returns["Ensemble"]), "start": str(pd.Timestamp(test_dates[0]).date()), "end": curve[-1]["date"]}, mature


def news_headlines(symbol, name, root):
    query = f'"{name or NAMES.get(symbol, symbol.split(".")[0])}" India stock when:7d'
    url = "https://news.google.com/rss/search?" + urlencode({"q": query, "hl": "en-IN", "gl": "IN", "ceid": "IN:en"})
    response = requests.get(url, timeout=12, headers={"User-Agent": "FolioPersonalResearch/1.0"})
    response.raise_for_status()
    if len(response.content) > 2_000_000:
        raise ValueError("News feed is too large.")
    now = utc_now()
    seen = load_json(root / "news-seen.json", {})
    articles = []
    for item in ET.fromstring(response.content).findall(".//item")[:12]:
        title = (item.findtext("title") or "").strip()[:700]
        link = item.findtext("link") or ""
        if not title or not link.startswith("https://"):
            continue
        key = hashlib.sha256(re.sub(r"\W+", " ", title.lower()).encode()).hexdigest()
        if key in {a["id"] for a in articles}:
            continue
        try:
            published = parsedate_to_datetime(item.findtext("pubDate") or "").astimezone(timezone.utc).isoformat()
            if published > now:
                continue
        except (ValueError, TypeError):
            continue
        seen.setdefault(key, now)
        articles.append({"id": key, "symbol": symbol, "title": title, "url": link, "source": item.findtext("source") or "Google News", "publishedAt": published, "firstSeenAt": seen[key]})
    atomic_json(root / "news-seen.json", seen)
    return articles[:6]


def sentiment_scores(articles):
    global _sentiment
    if not articles:
        return []
    if _sentiment is None:
        import torch
        from transformers import pipeline
        torch.set_num_threads(2)
        _sentiment = pipeline("text-classification", model="ProsusAI/finbert", revision=TEXT_REVISION, device=-1)
    results = _sentiment([a["title"] for a in articles], truncation=True, max_length=128, batch_size=8, top_k=None)
    for article, scores in zip(articles, results):
        values = {s["label"].lower(): float(s["score"]) for s in scores}
        article["sentiment"] = values.get("positive", 0) - values.get("negative", 0)
        article["label"] = max(values, key=values.get)
    return articles


def resolve_ledger(root, frames):
    ledger = load_json(root / "forecasts.json", [])
    for forecast in ledger:
        if forecast["status"] != "pending":
            continue
        symbols = forecast["universe"]
        if any(s not in frames for s in symbols):
            continue
        dates = sorted(set.intersection(*(set(frames[s].index) for s in symbols)))
        eligible = [d for d in dates if d.date().isoformat() > forecast["signalDate"]]
        if len(eligible) < HORIZON + 1:
            continue
        entry, exit_date = eligible[0], eligible[HORIZON]
        actual = {s: float(frames[s].loc[exit_date, "open"] / frames[s].loc[entry, "open"] - 1) for s in symbols}
        cost = forecast["costBps"] / 10000
        forecast.update(status="resolved", entryDate=str(entry.date()), exitDate=str(exit_date.date()), priceReturn=float(np.mean([actual[s] for s in forecast["pricePicks"]])) - cost, newsReturn=float(np.mean([actual[s] for s in forecast["newsPicks"]])) - cost, benchmarkReturn=float(np.mean(list(actual.values()))) - cost)
        if "auditWeights" in forecast:
            forecast["auditReturn"] = sum(w * actual[s] for s, w in forecast["auditWeights"].items()) - forecast["auditExposure"] * cost

    return ledger


def run_research(config, root: Path, progress=lambda *_: None, loader=market_history, news_loader=news_headlines, scorer=sentiment_scores):
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    frames, warnings = {}, []
    symbols = config["symbols"]
    for i, symbol in enumerate(symbols):
        progress(f"Fetching {symbol} ({i + 1}/{len(symbols)})", 3 + int(i / len(symbols) * 25))
        try:
            frames[symbol] = loader(symbol, config["years"], root)
        except Exception as e:
            warnings.append(f"{symbol}: price history unavailable ({type(e).__name__}).")
    if len(frames) < 4:
        raise ValueError("At least four stocks with usable daily history are needed. Public data may be temporarily unavailable; retry later.")
    top_k = min(config["topK"], len(frames))
    panel, _ = feature_panel(frames)
    backtest, mature = walk_forward(panel, top_k, config["costBps"], progress)
    progress("Fitting final models on matured outcomes", 73)
    latest_date = panel.date.max()
    latest = panel[panel.date == latest_date].copy()
    fingerprint = hashlib.sha256(pd.util.hash_pandas_object(mature[["date", "symbol", *FEATURES, "target"]], index=False).values.tobytes()).hexdigest()
    model_id = hashlib.sha256(json.dumps({"version": VERSION, "symbols": sorted(frames), "data": fingerprint, "features": FEATURES}, sort_keys=True).encode()).hexdigest()[:12]
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
    audit_id = hashlib.sha256(json.dumps({"version": AUDIT_VERSION, "data": audit_fingerprint, "topK": top_k, "costBps": config["costBps"]}, sort_keys=True).encode()).hexdigest()[:12]
    audit["modelId"] = audit_id
    audit["dataFingerprint"] = audit_fingerprint
    with tempfile.NamedTemporaryFile(dir=model_dir, delete=False) as f:
        audit_temp = f.name
    joblib.dump({**audit_models, "dataFingerprint": audit_fingerprint}, audit_temp)
    os.chmod(audit_temp, 0o600)
    os.replace(audit_temp, model_dir / f"{audit_id}.joblib")
    articles = []
    if config["includeNews"]:
        for i, symbol in enumerate(frames):
            progress(f"Reading headlines for {symbol}", 86 + int(i / len(frames) * 7))
            try:
                articles.extend(news_loader(symbol, config.get("names", {}).get(symbol), root))
            except Exception:
                warnings.append(f"{symbol}: news unavailable; no news adjustment applied.")
        progress("Scoring headlines locally with FinBERT", 94)
        try:
            articles = scorer(articles)
        except Exception as e:
            warnings.append(f"FinBERT unavailable ({type(e).__name__}); news overlay disabled, price models still available.")
            articles = []
    now = utc_now()
    rankings = []
    for _, row in latest.iterrows():
        headlines = [a for a in articles if a["symbol"] == row.symbol]
        # Publication freshness is only an overlay; first-seen timestamps govern future training.
        weights = [np.exp(-max(0, (datetime.fromisoformat(now) - datetime.fromisoformat(a["publishedAt"])).total_seconds()) / 172800) for a in headlines]
        news = float(np.dot([a["sentiment"] for a in headlines], weights) / max(1.0, sum(weights))) if weights else 0.0
        base = float(row.Ensemble)
        models = {n: float(row[n]) for n in estimators()}
        rankings.append({"symbol": row.symbol, "name": config.get("names", {}).get(row.symbol) or NAMES.get(row.symbol, row.symbol), "price": float(row.close), "priceScore": base, "newsScore": news, "score": base + 0.15 * news, "headlines": len(headlines), "modelSpread": float(np.std(list(models.values()))), "models": models})
    rankings.sort(key=lambda r: r["score"], reverse=True)
    price_picks = [r["symbol"] for r in sorted(rankings, key=lambda r: r["priceScore"], reverse=True)[:top_k]]
    news_picks = [r["symbol"] for r in rankings[:top_k]]
    ledger = resolve_ledger(root, frames)
    signal_date = max(str(latest_date.date()), datetime.now(ZoneInfo("Asia/Kolkata")).date().isoformat())
    experiment = hashlib.sha256(json.dumps({"symbols": sorted(frames), "topK": top_k, "costBps": config["costBps"], "includeNews": config["includeNews"], "version": VERSION, "overlay": "freshness-v1", "audit": AUDIT_VERSION}, sort_keys=True).encode()).hexdigest()[:12]
    forecast_id = f"{experiment}-{signal_date}"
    if not any(p["id"] == forecast_id for p in ledger):
        ledger.append({"id": forecast_id, "createdAt": now, "signalDate": signal_date, "modelId": model_id, "experiment": experiment, "universe": sorted(frames), "pricePicks": price_picks, "newsPicks": news_picks, "costBps": config["costBps"], "status": "pending", "newsEnabled": bool(articles), "auditModelId": audit_id, "auditStrategy": audit["latest"]["strategy"], "auditWeights": {a["symbol"]: a["weight"] for a in audit["latest"]["allocations"]}, "auditExposure": audit["latest"]["exposure"]})
    ledger = ledger[-500:]
    atomic_json(root / "forecasts.json", ledger)
    if articles:
        atomic_json(root / "news" / f"{forecast_id}-{now[:19].replace(':', '-')}.json", {"recordedAt": now, "model": "ProsusAI/finbert", "revision": TEXT_REVISION, "articles": articles})
    stale_days = (datetime.now(ZoneInfo("Asia/Kolkata")).date() - latest_date.date()).days
    if stale_days > 4:
        warnings.append(f"Latest common market data is {stale_days} calendar days old. Treat rankings as stale.")
    warnings.extend(["Historical evaluation uses today's selected stock cohort; survivorship bias remains.", "News overlay is experimental and excluded from the historical price backtest. Sentiment is not a return probability.", "Adjusted public prices can be revised. Backtest is a research simulation, not an execution record.", "Five-session sampled drawdown misses losses inside each holding period. Cost is a configurable assumption, not a verified broker charge.", "Paper forecasts may overlap across days; their individual returns must not be compounded as an executable portfolio."])
    report = {"version": 1, "engineVersion": VERSION, "id": forecast_id, "modelId": model_id, "dataFingerprint": fingerprint, "textModelRevision": TEXT_REVISION, "generatedAt": now, "marketAsOf": str(latest_date.date()), "source": "Yahoo Finance via yfinance · personal research, adjusted daily OHLCV", "config": {**config, "symbols": list(frames), "topK": top_k}, "coverage": {"requested": len(symbols), "loaded": len(frames), "rows": len(panel), "trainingEnd": str(mature.date.max().date()), "newsArticles": len(articles)}, "rankings": rankings, "backtest": backtest, "audit": audit, "articles": articles, "forecasts": [f for f in ledger if f["experiment"] == experiment][-100:], "warnings": warnings}
    atomic_json(root / "latest.json", report)
    atomic_json(root / "runs" / f"{forecast_id}-{now[:19].replace(':', '-')}.json", report)
    progress("Research complete", 100)
    return report
