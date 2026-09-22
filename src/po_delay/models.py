"""Baseline and candidate models, plus evaluation metrics.

Models
------
- ``heuristic_baseline_scores``: a simple rule a planner might already use today
  (rush order + below-median supplier reliability -> flagged), used as the naive floor.
- ``build_logistic_pipeline``: one-hot encoded Logistic Regression — interpretable
  statistical baseline.
- ``build_random_forest_pipeline``: one-hot encoded Random Forest.
- ``build_lightgbm``: gradient boosting using native categorical support, tuned via
  the rolling-origin CV splits in ``validation.py``.

Metrics
-------
ROC-AUC and PR-AUC for ranking quality (the latter matters more under class
imbalance), Brier score for calibration, and precision@k for the buyer-facing
"top-k riskiest open POs" use case.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from po_delay.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES


def _preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("num", StandardScaler(), NUMERIC_FEATURES),
        ]
    )


def build_logistic_pipeline(**kwargs) -> Pipeline:
    params = {"max_iter": 1000, "class_weight": "balanced"} | kwargs
    return Pipeline(steps=[("preprocess", _preprocessor()), ("clf", LogisticRegression(**params))])


def build_random_forest_pipeline(**kwargs) -> Pipeline:
    params = {
        "n_estimators": 300,
        "max_depth": 8,
        "min_samples_leaf": 20,
        "class_weight": "balanced_subsample",
        "random_state": 42,
        "n_jobs": -1,
    } | kwargs
    return Pipeline(
        steps=[("preprocess", _preprocessor()), ("clf", RandomForestClassifier(**params))]
    )


def build_lightgbm(**kwargs) -> LGBMClassifier:
    params = {
        "n_estimators": 500,
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": 30,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "class_weight": "balanced",
        "random_state": 42,
        "verbosity": -1,
    } | kwargs
    return LGBMClassifier(**params)


def fit_lightgbm_with_early_stopping(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    **kwargs,
) -> LGBMClassifier:
    import lightgbm as lgb

    model = build_lightgbm(**kwargs)
    model.fit(
        X_train,
        y_train,
        eval_X=X_val,
        eval_y=y_val,
        eval_metric="average_precision",
        categorical_feature=CATEGORICAL_FEATURES,
        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
    )
    return model


def heuristic_baseline_scores(X: pd.DataFrame) -> np.ndarray:
    """A simple planner heuristic: flag rush orders from historically unreliable
    or brand-new suppliers. Returns pseudo-probabilities in {0.1, 0.5, 0.9} so it can
    be scored with the same ranking/calibration metrics as the ML models."""
    is_rush = X["is_rush_order"].astype(bool)
    unreliable = X["supplier_on_time_rate_trailing365"] < 0.75
    new_supplier = X["is_new_supplier"].astype(bool)

    scores = np.full(len(X), 0.1)
    scores[unreliable.to_numpy() & ~is_rush.to_numpy()] = 0.5
    scores[is_rush.to_numpy() | new_supplier.to_numpy()] = 0.5
    scores[is_rush.to_numpy() & unreliable.to_numpy()] = 0.9
    return scores


def precision_at_k(y_true: np.ndarray, y_score: np.ndarray, k_frac: float) -> float:
    n = len(y_true)
    k = max(1, int(np.ceil(n * k_frac)))
    top_k_idx = np.argsort(-y_score)[:k]
    return float(np.mean(np.asarray(y_true)[top_k_idx]))


def evaluate(y_true: pd.Series | np.ndarray, y_proba: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true)
    metrics = {
        "roc_auc": roc_auc_score(y_true, y_proba),
        "pr_auc": average_precision_score(y_true, y_proba),
        "brier_score": brier_score_loss(y_true, y_proba),
        "precision_at_5pct": precision_at_k(y_true, y_proba, 0.05),
        "precision_at_10pct": precision_at_k(y_true, y_proba, 0.10),
        "precision_at_20pct": precision_at_k(y_true, y_proba, 0.20),
        "base_rate": float(y_true.mean()),
    }
    return metrics
