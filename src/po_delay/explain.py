"""Model explainability and error-analysis helpers.

Global/local explanations use SHAP's ``TreeExplainer`` against the LightGBM model
(the project's primary candidate). Error-analysis slicing works with any model's
predicted probabilities and is used to check for systematic weaknesses (e.g., poor
performance on new suppliers or a specific category).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

with warnings.catch_warnings():
    # shap's import chain (via tqdm.autonotebook) warns that ipywidgets/IProgress
    # isn't installed; harmless here since none of our usage needs a progress bar.
    warnings.filterwarnings("ignore", message="IProgress not found")
    # shap's bundled colormap uses a matplotlib Colormap API that is pending
    # deprecation in newer matplotlib; not something this project's code controls.
    warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
    import shap

from po_delay.models import evaluate


def compute_shap_values(model, X: pd.DataFrame) -> np.ndarray:
    """Return SHAP values for a tree-based binary classifier (e.g., LightGBM)."""
    explainer = shap.TreeExplainer(model)
    with warnings.catch_warnings():
        # Expected and already handled below (list -> positive-class array);
        # not a sign of misuse, so no need to surface it to callers.
        warnings.filterwarnings("ignore", message="LightGBM binary classifier with TreeExplainer")
        shap_values = explainer.shap_values(X)
    # LightGBM binary classifiers may return a single array or a [class0, class1] list
    # depending on version; normalize to the positive-class contribution matrix.
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    return shap_values


def global_importance(shap_values: np.ndarray, feature_names: list[str]) -> pd.DataFrame:
    """Mean absolute SHAP value per feature, sorted descending."""
    mean_abs = np.abs(shap_values).mean(axis=0)
    return (
        pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )


def explain_instance(
    shap_values: np.ndarray, feature_names: list[str], row_idx: int, top_n: int = 5
) -> pd.DataFrame:
    """Top contributing features (by |SHAP|) for a single PO, for a buyer-facing
    "why is this order flagged" explanation."""
    row = shap_values[row_idx]
    order = np.argsort(-np.abs(row))[:top_n]
    return pd.DataFrame(
        {
            "feature": [feature_names[i] for i in order],
            "shap_value": row[order],
        }
    )


def slice_metrics(
    df: pd.DataFrame,
    y_true: np.ndarray,
    y_proba: np.ndarray,
    slice_col: str,
    min_count: int = 20,
) -> pd.DataFrame:
    """Per-slice (e.g., per category, per region) performance for error analysis.

    Slices with fewer than ``min_count`` rows are still reported but flagged, since
    their metrics are noisy.
    """
    work = pd.DataFrame({slice_col: df[slice_col].to_numpy(), "y_true": y_true, "y_proba": y_proba})
    rows = []
    for value, group in work.groupby(slice_col, observed=True):
        n = len(group)
        if group["y_true"].nunique() < 2:
            rows.append(
                {
                    "slice": value,
                    "n": n,
                    "low_sample": n < min_count,
                    "roc_auc": np.nan,
                    "pr_auc": np.nan,
                    "brier_score": np.nan,
                    "base_rate": float(group["y_true"].mean()),
                }
            )
            continue
        m = evaluate(group["y_true"], group["y_proba"])
        rows.append(
            {
                "slice": value,
                "n": n,
                "low_sample": n < min_count,
                "roc_auc": m["roc_auc"],
                "pr_auc": m["pr_auc"],
                "brier_score": m["brier_score"],
                "base_rate": m["base_rate"],
            }
        )
    return pd.DataFrame(rows).sort_values("n", ascending=False).reset_index(drop=True)
