"""
predict.py
----------
Command-line inference tool.

Usage:
    python predict.py --input path/to/sensor_readings.csv --output risk_scores.csv

Expects a CSV with the same columns produced by generate_data.py
(minus RUL / failure_within_30cy, which are not required for inference).
"""

import argparse
from pathlib import Path

import joblib
import pandas as pd

from features import build_feature_matrix

ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description="Predict maintenance risk from sensor data.")
    parser.add_argument("--input", required=True, help="CSV file with sensor readings")
    parser.add_argument("--output", default="risk_scores.csv", help="Where to write predictions")
    parser.add_argument("--threshold", type=float, default=0.5, help="Alert threshold")
    args = parser.parse_args()

    bundle = joblib.load(ROOT / "models" / "best_model.joblib")
    model, feature_cols = bundle["model"], bundle["feature_cols"]

    df = pd.read_csv(args.input)
    X, _, _, _ = build_feature_matrix(df)
    X = X[feature_cols]

    proba = model.predict_proba(X)[:, 1]
    df["risk_score"] = pd.Series(proba, index=X.index)
    df["alert"] = df["risk_score"] >= args.threshold

    df.to_csv(args.output, index=False)
    n_alerts = int(df["alert"].sum())
    print(f"Wrote {len(df)} predictions to {args.output} ({n_alerts} above threshold {args.threshold}).")


if __name__ == "__main__":
    main()
