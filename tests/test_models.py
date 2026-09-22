import numpy as np

from po_delay.features import build_features
from po_delay.models import (
    build_logistic_pipeline,
    build_random_forest_pipeline,
    evaluate,
    fit_lightgbm_with_early_stopping,
    heuristic_baseline_scores,
    precision_at_k,
)
from po_delay.validation import time_based_split


def test_precision_at_k_perfect_ranking():
    y_true = np.array([0, 0, 0, 1, 1])
    y_score = np.array([0.1, 0.2, 0.3, 0.9, 0.95])
    assert precision_at_k(y_true, y_score, k_frac=0.4) == 1.0


def test_evaluate_returns_expected_keys(small_dataset):
    X, y = build_features(small_dataset)
    scores = heuristic_baseline_scores(X)
    metrics = evaluate(y, scores)
    expected_keys = {
        "roc_auc",
        "pr_auc",
        "brier_score",
        "precision_at_5pct",
        "precision_at_10pct",
        "precision_at_20pct",
        "base_rate",
    }
    assert expected_keys == set(metrics.keys())
    assert 0.0 <= metrics["roc_auc"] <= 1.0
    assert 0.0 <= metrics["pr_auc"] <= 1.0


def test_heuristic_beats_random_guessing(small_dataset):
    X, y = build_features(small_dataset)
    scores = heuristic_baseline_scores(X)
    metrics = evaluate(y, scores)
    assert metrics["roc_auc"] > 0.5


def test_logistic_pipeline_fits_and_predicts(small_dataset):
    X, y = build_features(small_dataset)
    split = time_based_split(small_dataset)
    pipe = build_logistic_pipeline(max_iter=200)
    pipe.fit(X.iloc[split.train_idx], y.iloc[split.train_idx])
    proba = pipe.predict_proba(X.iloc[split.test_idx])[:, 1]
    assert proba.shape[0] == len(split.test_idx)
    assert (proba >= 0).all() and (proba <= 1).all()


def test_random_forest_pipeline_fits_and_predicts(small_dataset):
    X, y = build_features(small_dataset)
    split = time_based_split(small_dataset)
    pipe = build_random_forest_pipeline(n_estimators=50)
    pipe.fit(X.iloc[split.train_idx], y.iloc[split.train_idx])
    proba = pipe.predict_proba(X.iloc[split.test_idx])[:, 1]
    assert proba.shape[0] == len(split.test_idx)


def test_lightgbm_fits_with_early_stopping(small_dataset):
    X, y = build_features(small_dataset)
    split = time_based_split(small_dataset)
    model = fit_lightgbm_with_early_stopping(
        X.iloc[split.train_idx],
        y.iloc[split.train_idx],
        X.iloc[split.val_idx],
        y.iloc[split.val_idx],
        n_estimators=100,
    )
    proba = model.predict_proba(X.iloc[split.test_idx])[:, 1]
    metrics = evaluate(y.iloc[split.test_idx], proba)
    assert 0.0 <= metrics["roc_auc"] <= 1.0
