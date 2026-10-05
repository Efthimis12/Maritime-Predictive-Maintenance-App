"""
features.py
------------
Feature engineering for the maritime predictive-maintenance pipeline.
Adds rolling-window statistics (mean, std, slope) per engine unit, which
are commonly the most predictive features in condition-based maintenance
because they capture *trends* rather than single noisy readings.
"""

import numpy as np
import pandas as pd

SENSOR_COLS = [
    "exhaust_temp_C",
    "coolant_temp_C",
    "lube_oil_pressure_bar",
    "vibration_rms_mm_s",
    "rpm",
    "fuel_rate_kg_h",
    "oil_particle_count",
    "hull_speed_knots",
]

WINDOW = 10


def _rolling_slope(series: pd.Series, window: int) -> pd.Series:
    """Simple linear-trend slope over a rolling window."""
    x = np.arange(window)

    def slope(y):
        if len(y) < window:
            return np.nan
        return np.polyfit(x, y, 1)[0]

    return series.rolling(window).apply(slope, raw=True)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["unit_id", "cycle"]).copy()
    grouped = df.groupby("unit_id")

    for col in SENSOR_COLS:
        df[f"{col}_roll_mean"] = grouped[col].transform(
            lambda s: s.rolling(WINDOW, min_periods=1).mean()
        )
        df[f"{col}_roll_std"] = grouped[col].transform(
            lambda s: s.rolling(WINDOW, min_periods=1).std()
        )
        df[f"{col}_slope"] = grouped[col].transform(
            lambda s: _rolling_slope(s, WINDOW)
        )

    df = df.fillna(0)
    return df


def build_feature_matrix(df):
    df_feat = add_features(df)
    feature_cols = [c for c in df_feat.columns
                     if c not in ("unit_id", "cycle", "RUL", "failure_within_30cy")]
    X = df_feat[feature_cols]
    y = df_feat["failure_within_30cy"] if "failure_within_30cy" in df_feat.columns else None
    groups = df_feat["unit_id"]
    return X, y, groups, feature_cols