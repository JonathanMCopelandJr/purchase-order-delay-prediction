import numpy as np

from po_delay.explain import compute_shap_values, explain_instance, global_importance, slice_metrics
from po_delay.features import build_features
from po_delay.models import fit_lightgbm_with_early_stopping
from po_delay.validation import time_based_split


def test_shap_values_shape_matches_features(small_dataset):
    X, y = build_features(small_dataset)
    split = time_based_split(small_dataset, train_frac=0.6, val_frac=0.2)
    model = fit_lightgbm_with_early_stopping(
        X.iloc[split.train_idx],
        y.iloc[split.train_idx],
        X.iloc[split.val_idx],
        y.iloc[split.val_idx],
        n_estimators=50,
    )
    X_test = X.iloc[split.test_idx]
    shap_values = compute_shap_values(model, X_test)
    assert shap_values.shape == X_test.shape


def test_global_importance_sums_positive_and_sorted(small_dataset):
    X, y = build_features(small_dataset)
    split = time_based_split(small_dataset, train_frac=0.6, val_frac=0.2)
    model = fit_lightgbm_with_early_stopping(
        X.iloc[split.train_idx],
        y.iloc[split.train_idx],
        X.iloc[split.val_idx],
        y.iloc[split.val_idx],
        n_estimators=50,
    )
    X_test = X.iloc[split.test_idx]
    shap_values = compute_shap_values(model, X_test)
    importance = global_importance(shap_values, list(X_test.columns))

    assert set(importance["feature"]) == set(X_test.columns)
    assert (importance["mean_abs_shap"] >= 0).all()
    assert importance["mean_abs_shap"].is_monotonic_decreasing


def test_explain_instance_returns_top_n(small_dataset):
    X, y = build_features(small_dataset)
    split = time_based_split(small_dataset, train_frac=0.6, val_frac=0.2)
    model = fit_lightgbm_with_early_stopping(
        X.iloc[split.train_idx],
        y.iloc[split.train_idx],
        X.iloc[split.val_idx],
        y.iloc[split.val_idx],
        n_estimators=50,
    )
    X_test = X.iloc[split.test_idx]
    shap_values = compute_shap_values(model, X_test)
    top = explain_instance(shap_values, list(X_test.columns), row_idx=0, top_n=3)
    assert len(top) == 3
    assert set(top.columns) == {"feature", "shap_value"}


def test_slice_metrics_covers_all_slice_values(small_dataset):
    X, y = build_features(small_dataset)
    rng = np.random.default_rng(0)
    fake_proba = rng.uniform(size=len(y))
    table = slice_metrics(X, y.to_numpy(), fake_proba, "category")
    assert set(table["slice"]) == set(X["category"].unique())
    assert (table["n"] > 0).all()
