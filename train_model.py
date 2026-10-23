"""
train_model.py
==============
Trains the FINAL Risk Alert Classifier and writes the artifacts used by the
Streamlit app:

    models/risk_model.joblib   -> fitted sklearn Pipeline (preprocessing + model)
    models/model_meta.json     -> metrics, threshold, ROC points, feature
                                  importances, UI option lists, etc.

Usage
-----
    python train_model.py                       # re-train from data/*.csv
    python train_model.py --data path/to.csv    # custom dataset location

The notebook (Part H) calls ``export_artifacts`` with the model it selected, so
the notebook and this script always write the same file format.  The app also
calls ``train_and_save`` automatically if the saved model cannot be loaded
(e.g. different scikit-learn version on the deployment server).

FINAL MODEL
-----------
Logistic Regression (no resampling).  It was selected in the notebook because
it produced the fewest false negatives (and no false positives) on the hold-out
set, is the most interpretable model and is cheap to serve.  See Part G of the
notebook for the full comparison.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

import risk_utils as ru

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(HERE, "data", "Risk_Alert_Classifier_Dataset_4600.csv")
MODEL_DIR = os.path.join(HERE, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "risk_model.joblib")
META_PATH = os.path.join(MODEL_DIR, "model_meta.json")

RANDOM_STATE = 42
FINAL_MODEL_NAME = "Logistic Regression"


def build_final_estimator():
    """Estimator selected in the notebook (Part G)."""
    return LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)


def _importances(model, names):
    """Tree models -> impurity importance; linear models -> |coefficient|."""
    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_, dtype=float)
        kind = "impurity importance"
    elif hasattr(model, "coef_"):
        values = np.abs(np.asarray(model.coef_)).ravel()
        kind = "|standardised coefficient|"
    else:
        return [], "n/a"
    order = np.argsort(values)[::-1]
    return [{"feature": names[i], "importance": float(values[i])} for i in order], kind


def export_artifacts(pipeline, X_train, y_train, X_test, y_test, model_name,
                     threshold=0.5, out_dir=MODEL_DIR):
    """Evaluate ``pipeline`` on the hold-out set and write model + metadata."""
    os.makedirs(out_dir, exist_ok=True)

    proba = pipeline.predict_proba(X_test)[:, 1]
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
    fpr, tpr, _ = roc_curve(y_test, proba)
    keep = np.unique(np.linspace(0, len(fpr) - 1, 150).astype(int))

    names = ru.get_feature_names(pipeline.named_steps["preprocess"])
    importances, imp_kind = _importances(pipeline.named_steps["model"], names)

    numeric_info = {}
    for col in ru.RAW_NUMERIC_FEATURES:
        s = pd.to_numeric(X_train[col], errors="coerce")
        numeric_info[col] = {
            "min": float(s.min()), "max": float(s.max()),
            "median": float(s.median()),
        }

    meta = {
        "model_name": model_name,
        "trained_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
        "sklearn_version": sklearn.__version__,
        "threshold": float(threshold),
        "metrics": {
            "accuracy": float(accuracy_score(y_test, pred)),
            "precision": float(precision_score(y_test, pred, zero_division=0)),
            "recall": float(recall_score(y_test, pred, zero_division=0)),
            "f1": float(f1_score(y_test, pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, proba)),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        },
        "roc": {"fpr": fpr[keep].tolist(), "tpr": tpr[keep].tolist()},
        "importance_kind": imp_kind,
        "feature_importance": importances,
        "options": {c: sorted(X_train[c].dropna().unique().tolist())
                    for c in ru.CATEGORICAL_FEATURES},
        "numeric_info": numeric_info,
        "date_range": [str(pd.to_datetime(X_train[ru.DATE_COL]).min().date()),
                       str(pd.to_datetime(X_train[ru.DATE_COL]).max().date())],
        "n_train": int(len(y_train)), "n_test": int(len(y_test)),
        "high_risk_share_train": float(np.mean(y_train)),
    }

    joblib.dump(pipeline, os.path.join(out_dir, "risk_model.joblib"), compress=3)
    with open(os.path.join(out_dir, "model_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    return meta


def train_and_save(data_path=DATA_PATH, out_dir=MODEL_DIR, verbose=True):
    """Full training run for the final model (same split as the notebook)."""
    df = ru.load_dataset(data_path)
    X, y = ru.split_xy(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )

    # 1) threshold from out-of-fold predictions on TRAINING data only
    oof = ru.oof_probabilities(X_train, y_train, build_final_estimator(), n_splits=5,
                               random_state=RANDOM_STATE)
    threshold, _, _ = ru.pick_threshold(y_train, oof, beta=2.0)

    # 2) fit the final pipeline on the full training split
    pre = ru.build_preprocessor()
    Xt = pre.fit_transform(X_train)
    clf = build_final_estimator().fit(Xt, y_train)
    pipeline = Pipeline([("preprocess", pre), ("model", clf)])

    # 3) evaluate on the untouched test split and write artifacts
    meta = export_artifacts(pipeline, X_train, y_train, X_test, y_test,
                            FINAL_MODEL_NAME, threshold, out_dir)
    if verbose:
        m = meta["metrics"]
        print(f"Saved {FINAL_MODEL_NAME} -> {out_dir}")
        print(f"threshold={meta['threshold']:.2f}  recall={m['recall']:.3f}  "
              f"precision={m['precision']:.3f}  f1={m['f1']:.3f}  auc={m['roc_auc']:.3f}  "
              f"FN={m['fn']}  FP={m['fp']}")
    return pipeline, meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the final Risk Alert Classifier")
    parser.add_argument("--data", default=DATA_PATH, help="path to the CSV dataset")
    args = parser.parse_args()
    sys.exit(0 if train_and_save(args.data) else 1)
