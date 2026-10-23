"""
risk_utils.py
=============
Shared helpers for the Risk Alert Classifier project.

Used by:
    * Risk_Alert_Classifier.ipynb  (analysis + model building)
    * train_model.py               (re-trains the final model, writes models/)
    * app.py                       (Streamlit web app)

Keeping the preprocessing in ONE place guarantees that the notebook, the saved
model and the web app all transform raw customer data in exactly the same way.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# --------------------------------------------------------------------------- #
# Column definitions
# --------------------------------------------------------------------------- #
TARGET = "risk_status"                      # 0 = Low Risk, 1 = High Risk
ID_COL = "customer_id"                      # identifier -> never a feature
DATE_COL = "last_transaction_date"          # converted to "days since last txn"

# Fixed reference date (latest date in the dataset) so that the engineered
# "recency" feature is reproducible at training time AND at prediction time.
REFERENCE_DATE = pd.Timestamp("2025-12-01")

CATEGORICAL_FEATURES = ["gender", "region", "employment_type"]

RAW_NUMERIC_FEATURES = [
    "age",
    "annual_income_inr",
    "credit_score",
    "credit_utilization_ratio",
    "missed_payments_12m",
    "avg_late_payment_days",
    "monthly_transaction_count",
    "monthly_spend_inr",
    "cash_advance_count_6m",
    "complaints_last_6m",
    "failed_login_attempts_3m",
    "account_tenure_months",
    "debt_balance_inr",
]
ENGINEERED_FEATURES = ["days_since_last_txn", "debt_to_income_ratio"]
NUMERIC_FEATURES = RAW_NUMERIC_FEATURES + ENGINEERED_FEATURES

# Columns a raw customer record is expected to contain (target / id optional)
RAW_INPUT_COLUMNS = CATEGORICAL_FEATURES + RAW_NUMERIC_FEATURES + [DATE_COL]


# --------------------------------------------------------------------------- #
# Feature engineering transformer
# --------------------------------------------------------------------------- #
class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Turns a raw customer DataFrame into the model-ready column layout.

    * drops ``customer_id`` and ``risk_status`` if present (no leakage / no ID)
    * ``last_transaction_date``  ->  ``days_since_last_txn``
    * ``debt_balance_inr / annual_income_inr``  ->  ``debt_to_income_ratio``
    * leaves missing values as NaN (they are imputed later, inside the pipeline,
      using statistics learned from the TRAINING data only)
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()

        # make sure every expected raw column exists (missing -> NaN)
        for col in RAW_INPUT_COLUMNS:
            if col not in X.columns:
                X[col] = np.nan

        # recency feature
        dates = pd.to_datetime(X[DATE_COL], errors="coerce")
        X["days_since_last_txn"] = (REFERENCE_DATE - dates).dt.days.astype(float)

        # leverage feature (guard against zero / missing income)
        income = pd.to_numeric(X["annual_income_inr"], errors="coerce")
        debt = pd.to_numeric(X["debt_balance_inr"], errors="coerce")
        ratio = debt / income.where(income > 0)
        X["debt_to_income_ratio"] = ratio.replace([np.inf, -np.inf], np.nan)

        # consistent dtypes
        for col in CATEGORICAL_FEATURES:
            X[col] = X[col].astype(object).where(X[col].notna(), np.nan)
        for col in NUMERIC_FEATURES:
            X[col] = pd.to_numeric(X[col], errors="coerce").astype(float)

        return X[CATEGORICAL_FEATURES + NUMERIC_FEATURES]


# --------------------------------------------------------------------------- #
# Full preprocessing pipeline
# --------------------------------------------------------------------------- #
def build_preprocessor(n_neighbors: int = 5) -> Pipeline:
    """
    raw DataFrame
        -> FeatureEngineer
        -> ColumnTransformer
              numeric     : StandardScaler -> KNNImputer(k)      (multivariate imputation)
              categorical : SimpleImputer(most_frequent) -> OneHotEncoder

    The numeric columns are scaled BEFORE the KNN imputer because KNN is
    distance based - without scaling, large-valued columns such as income
    would dominate the distance calculation.
    """
    numeric_pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("knn_imputer", KNNImputer(n_neighbors=n_neighbors)),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    column_transformer = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, NUMERIC_FEATURES),
            ("cat", categorical_pipe, CATEGORICAL_FEATURES),
        ]
    )
    return Pipeline(
        steps=[("features", FeatureEngineer()), ("transform", column_transformer)]
    )


def get_feature_names(preprocessor: Pipeline) -> list[str]:
    """Readable column names of the matrix produced by ``build_preprocessor``."""
    names = preprocessor.named_steps["transform"].get_feature_names_out()
    return [n.split("__", 1)[1] for n in names]


# --------------------------------------------------------------------------- #
# Data helpers
# --------------------------------------------------------------------------- #
def load_dataset(path: str) -> pd.DataFrame:
    """Read the CSV and return the raw DataFrame."""
    return pd.read_csv(path)


def split_xy(df: pd.DataFrame):
    """Return (X, y). X keeps the raw columns (customer_id dropped)."""
    y = df[TARGET].astype(int)
    X = df.drop(columns=[TARGET, ID_COL], errors="ignore")
    return X, y


# --------------------------------------------------------------------------- #
# Leak-free cross-validation helpers
# --------------------------------------------------------------------------- #
import copy

from sklearn.base import clone
from sklearn.metrics import fbeta_score
from sklearn.model_selection import StratifiedKFold


def oof_probabilities(X, y, estimator, sampler=None, n_splits=5, random_state=42):
    """
    Out-of-fold P(high risk) for every row of the TRAINING data.

    For every fold the preprocessor (incl. KNN imputer) and the optional
    resampler are fitted on the fold's training part only, so nothing from the
    validation part leaks into the model.  Used for (a) cross-validated scores
    and (b) choosing the decision threshold without touching the test set.
    """
    y = np.asarray(y)
    oof = np.zeros(len(y), dtype=float)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for tr_idx, va_idx in skf.split(X, y):
        pre = build_preprocessor()
        X_tr = pre.fit_transform(X.iloc[tr_idx])
        X_va = pre.transform(X.iloc[va_idx])
        y_tr = y[tr_idx]
        if sampler is not None:
            X_tr, y_tr = copy.deepcopy(sampler).fit_resample(X_tr, y_tr)
        model = clone(estimator).fit(X_tr, y_tr)
        oof[va_idx] = model.predict_proba(X_va)[:, 1]
    return oof


def pick_threshold(y_true, proba, beta: float = 2.0):
    """
    Threshold that maximises F-beta (beta=2 weights recall twice as much as
    precision -> fits the 'minimise false negatives' business goal).
    If many thresholds tie, the middle of the plateau is returned (most robust).
    """
    grid = np.round(np.arange(0.05, 0.951, 0.01), 2)
    scores = np.array(
        [fbeta_score(y_true, (proba >= t).astype(int), beta=beta, zero_division=0) for t in grid]
    )
    best = grid[scores >= scores.max() - 1e-9]
    return float(np.median(best)), grid, scores
