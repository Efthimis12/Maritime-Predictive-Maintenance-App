"""
train.py
--------
Trains and evaluates predictive-maintenance classifiers on the synthetic
maritime engine dataset. Splits by engine unit (GroupShuffleSplit) so
that no cycles from the same physical engine leak between train and
test sets -- this mirrors how a model would really be deployed (trained
on a fleet, tested on unseen vessels).

Models compared:
  - Logistic Regression (baseline, interpretable)
  - Random Forest
  - Gradient Boosting

Outputs:
  - models/best_model.joblib
  - figures/confusion_matrix.png
  - figures/feature_importance.png
  - figures/roc_curve.png
  - outputs/metrics.json
"""

import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (GroupShuffleSplit, GroupKFold)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.base import clone

from features import build_feature_matrix

ROOT = Path(__file__).resolve().parent.parent


def load_data():
    df = pd.read_csv(ROOT / "data" / "maritime_sensor_data.csv")
    return df

def cross_validate_model(model_name, model_template, X, y, groups, n_splits=5):
    gkf = GroupKFold(n_splits=n_splits)
    fold_metrics = {"precision": [], "recall": [], "f1": [], "roc_auc": []}
    for train_idx, test_idx in gkf.split(X, y, groups):
        model = clone(model_template)
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
        if model_name == "gradient_boosting":
            sw = compute_sample_weight(class_weight="balanced", y=y_tr)
            model.fit(X_tr, y_tr, sample_weight=sw)
        else:
            model.fit(X_tr, y_tr)
        proba = model.predict_proba(X_te)[:, 1]
        preds = (proba >= 0.5).astype(int)
        fold_metrics["precision"].append(precision_score(y_te, preds, zero_division=0))
        fold_metrics["recall"].append(recall_score(y_te, preds, zero_division=0))
        fold_metrics["f1"].append(f1_score(y_te, preds, zero_division=0))
        fold_metrics["roc_auc"].append(roc_auc_score(y_te, proba))
    return {k: (float(np.mean(v)), float(np.std(v))) for k, v in fold_metrics.items()}


def main():
    df = load_data()
    X, y, groups, feature_cols = build_feature_matrix(df)

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    candidates = {
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]),
        "random_forest": RandomForestClassifier(
            n_estimators=300, max_depth=8, min_samples_leaf=3,
            class_weight="balanced", random_state=42, n_jobs=-1,
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=250, max_depth=3, learning_rate=0.08, random_state=42,
        ),
    }

    print("=== 5-fold GroupKFold cross-validation (robustness check) ===")
    cv_results = {}
    for name, model_template in candidates.items():
        cv_results[name] = cross_validate_model(name, model_template, X, y, groups, n_splits=5)
        p, r, f, a = cv_results[name]["precision"], cv_results[name]["recall"], cv_results[name]["f1"], cv_results[name]["roc_auc"]
        print(f"{name:20s} precision={p[0]:.3f}+/-{p[1]:.3f}  recall={r[0]:.3f}+/-{r[1]:.3f} "
              f"f1={f[0]:.3f}+/-{f[1]:.3f}  roc_auc={a[0]:.3f}+/-{a[1]:.3f}")
    print()

    results = {}
    fitted = {}
    for name, model in candidates.items():
        if name == "gradient_boosting":
            sw = compute_sample_weight(class_weight="balanced", y=y_train)
            model.fit(X_train, y_train, sample_weight=sw)
        else:
            model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
        preds = (proba >= 0.5).astype(int)
        results[name] = {
            "precision": precision_score(y_test, preds),
            "recall": recall_score(y_test, preds),
            "f1": f1_score(y_test, preds),
            "roc_auc": roc_auc_score(y_test, proba),
        }
        fitted[name] = model

    best_name = max(results, key=lambda k: results[k]["roc_auc"])
    best_model = fitted[best_name]
    best_proba = best_model.predict_proba(X_test)[:, 1]
    best_preds = (best_proba >= 0.5).astype(int)

    print("=== Model comparison (ROC-AUC) ===")
    for name, r in results.items():
        print(f"{name:20s} precision={r['precision']:.3f}  recall={r['recall']:.3f} "
              f"f1={r['f1']:.3f}  roc_auc={r['roc_auc']:.3f}")
    print(f"\nBest model: {best_name}")
    print(classification_report(y_test, best_preds, target_names=["Healthy", "At-risk"]))

    # --- Save model ---
    (ROOT / "models").mkdir(exist_ok=True)
    joblib.dump({"model": best_model, "feature_cols": feature_cols, "name": best_name},
                ROOT / "models" / "best_model.joblib")

    # --- Save metrics ---
    (ROOT / "outputs").mkdir(exist_ok=True)
    with open(ROOT / "outputs" / "metrics.json", "w") as f:
         json.dump({"results": results, "cv_results": cv_results, "best_model": best_name,
                "n_train": int(len(X_train)), "n_test": int(len(X_test)),
                "n_units": int(groups.nunique())}, f, indent=2)

    # --- Figures ---
    figs_dir = ROOT / "figures"
    figs_dir.mkdir(exist_ok=True)

    cm = confusion_matrix(y_test, best_preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Healthy", "At-risk"])
    fig, ax = plt.subplots(figsize=(5, 4.5))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion Matrix ({best_name})")
    fig.tight_layout()
    fig.savefig(figs_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4.5))
    RocCurveDisplay.from_predictions(y_test, best_proba, ax=ax)
    ax.set_title(f"ROC Curve ({best_name})")
    fig.tight_layout()
    fig.savefig(figs_dir / "roc_curve.png", dpi=150)
    plt.close(fig)

    if hasattr(best_model, "feature_importances_"):
        importances = pd.Series(best_model.feature_importances_, index=feature_cols)
        top = importances.sort_values(ascending=False).head(12)
        fig, ax = plt.subplots(figsize=(6, 5))
        top.iloc[::-1].plot.barh(ax=ax, color="#1f6f8b")
        ax.set_title("Top 12 Feature Importances")
        ax.set_xlabel("Importance")
        fig.tight_layout()
        fig.savefig(figs_dir / "feature_importance.png", dpi=150)
        plt.close(fig)

    print("\nSaved model, metrics, and figures.")


if __name__ == "__main__":
    main()
