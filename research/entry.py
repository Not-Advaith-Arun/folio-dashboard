"""Fixed exploratory absolute-return entry gate, independently dated calibration.

This is a price-only hypothesis, not a meta-model or profitable signal. Input
grossReturn must come from raw executable opens and split-adjusted share counts,
not a relative-return label or sentiment score.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from research.engine import FEATURES

ENTRY_VERSION = "absolute-entry-1"
ENTRY_PROTOCOL = {"version": ENTRY_VERSION, "grossHurdle": .005,
                  "probabilityThreshold": .60, "calibrationDates": 63,
                  "maximumTrainingYears": 3, "ridgeAlpha": 100.,
                  "logisticC": .1, "calibratorC": 1.,
                  "description": "Gross return above 50 bps and calibrated probability above 60%; costs are measured separately by execution replay. Not an account-specific net-profit probability."}


def predict_entry(panel, signal):
    signal = pd.Timestamp(signal)
    day = panel[panel.date == signal].copy()
    mature = panel[(panel.date < signal) & (panel.labelEnd < signal)].dropna(subset=["grossReturn", "labelEnd", *FEATURES])
    dates = sorted(mature.date.unique())
    if len(dates) < 252 + ENTRY_PROTOCOL["calibrationDates"] or day.empty or not np.isfinite(day[FEATURES].to_numpy()).all():
        raise ValueError("Insufficient past data or incomplete entry features")
    calibration_start = pd.Timestamp(dates[-ENTRY_PROTOCOL["calibrationDates"]])
    train = mature[(mature.date < calibration_start) & (mature.labelEnd < calibration_start) &
                   (mature.date >= signal - pd.DateOffset(years=ENTRY_PROTOCOL["maximumTrainingYears"]))]
    calibration = mature[mature.date >= calibration_start]
    y = (train.grossReturn > ENTRY_PROTOCOL["grossHurdle"]).astype(int)
    yc = (calibration.grossReturn > ENTRY_PROTOCOL["grossHurdle"]).astype(int)
    if train.date.nunique() < 252 or y.nunique() < 2 or yc.nunique() < 2:
        raise ValueError("Insufficient training dates or both outcome classes")
    model = make_pipeline(StandardScaler(), LogisticRegression(C=.1, max_iter=1000, random_state=42))
    model.fit(train[FEATURES], y)
    # The sigmoid calibrator sees only predictions on a later, disjoint block.
    calibrator = LogisticRegression(C=1., max_iter=1000, random_state=42)
    calibrator.fit(model.decision_function(calibration[FEATURES]).reshape(-1, 1), yc)
    regression = make_pipeline(StandardScaler(), Ridge(alpha=100.))
    regression.fit(train[FEATURES], train.grossReturn)
    day["profitProbability"] = calibrator.predict_proba(model.decision_function(day[FEATURES]).reshape(-1, 1))[:, 1]
    day["grossEstimate"] = regression.predict(day[FEATURES])
    day["entryAccepted"] = (day.profitProbability > .60) & (day.grossEstimate > .005)
    lineage = {"version": ENTRY_VERSION, "signal": str(signal.date()),
               "lastTrainingOutcome": str(train.labelEnd.max().date()),
               "calibrationStart": str(calibration_start.date()),
               "lastCalibrationOutcome": str(calibration.labelEnd.max().date()),
               "trainingRows": len(train), "calibrationRows": len(calibration),
               "pastPrevalence": float(yc.mean())}
    return day, lineage
