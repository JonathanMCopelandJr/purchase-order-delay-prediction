"""End-to-end reproducible pipeline: generate data -> features -> time-based split ->
train baseline + candidate models -> evaluate -> explain -> decision rules -> save
charts and a metrics report.

This is the scripted equivalent of the notebooks under notebooks/, useful for CI and
for reproducing headline numbers with a single command:

    PYTHONPATH=src python scripts/run_pipeline.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from po_delay.data_generation import GeneratorParams, generate  # noqa: E402
from po_delay.decision_rules import build_decision_table, sweep_thresholds  # noqa: E402
from po_delay.explain import compute_shap_values, global_importance, slice_metrics  # noqa: E402
from po_delay.features import build_features  # noqa: E402
from po_delay.models import (  # noqa: E402
    build_logistic_pipeline,
    build_random_forest_pipeline,
    evaluate,
    fit_lightgbm_with_early_stopping,
    heuristic_baseline_scores,
)
from po_delay.validation import assert_no_time_leakage, time_based_split  # noqa: E402

FIGURES_DIR = ROOT / "reports" / "figures"
REPORTS_DIR = ROOT / "reports"


def main(n_orders: int = 40_000, n_suppliers: int = 150, seed: int = 42) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Generating synthetic dataset (n_orders={n_orders}, seed={seed})...")
    df = generate(GeneratorParams(n_orders=n_orders, n_suppliers=n_suppliers, seed=seed))
    (ROOT / "data" / "raw").mkdir(parents=True, exist_ok=True)
    df.to_csv(ROOT / "data" / "raw" / "po_lines.csv", index=False)

    X, y = build_features(df)
    split = time_based_split(df)
    assert_no_time_leakage(df, split)

    X_train, y_train = X.iloc[split.train_idx], y.iloc[split.train_idx]
    X_val, y_val = X.iloc[split.val_idx], y.iloc[split.val_idx]
    X_test, y_test = X.iloc[split.test_idx], y.iloc[split.test_idx]
    print(f"Train/val/test sizes: {len(X_train)}/{len(X_val)}/{len(X_test)}")

    results: dict[str, dict[str, float]] = {}

    print("Scoring heuristic baseline...")
    heuristic_scores = heuristic_baseline_scores(X_test)
    results["heuristic_baseline"] = evaluate(y_test, heuristic_scores)

    print("Training logistic regression...")
    logit = build_logistic_pipeline()
    logit.fit(X_train, y_train)
    logit_proba = logit.predict_proba(X_test)[:, 1]
    results["logistic_regression"] = evaluate(y_test, logit_proba)

    print("Training random forest...")
    rf = build_random_forest_pipeline()
    rf.fit(X_train, y_train)
    rf_proba = rf.predict_proba(X_test)[:, 1]
    results["random_forest"] = evaluate(y_test, rf_proba)

    print("Training LightGBM (with early stopping on validation set)...")
    lgbm = fit_lightgbm_with_early_stopping(X_train, y_train, X_val, y_val)
    lgbm_proba = lgbm.predict_proba(X_test)[:, 1]
    results["lightgbm"] = evaluate(y_test, lgbm_proba)

    print("\nModel comparison (test set):")
    for name, metrics in results.items():
        print(
            f"  {name:20s} roc_auc={metrics['roc_auc']:.3f}  pr_auc={metrics['pr_auc']:.3f}  "
            f"brier={metrics['brier_score']:.3f}  p@10%={metrics['precision_at_10pct']:.3f}"
        )

    with open(REPORTS_DIR / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    # --- Charts -----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(y_test, lgbm_proba, name="LightGBM", ax=ax)
    RocCurveDisplay.from_predictions(y_test, rf_proba, name="Random Forest", ax=ax)
    RocCurveDisplay.from_predictions(y_test, logit_proba, name="Logistic Regression", ax=ax)
    RocCurveDisplay.from_predictions(y_test, heuristic_scores, name="Heuristic baseline", ax=ax)
    ax.set_title("ROC curve — held-out test period")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "roc_curve.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    PrecisionRecallDisplay.from_predictions(y_test, lgbm_proba, name="LightGBM", ax=ax)
    PrecisionRecallDisplay.from_predictions(y_test, rf_proba, name="Random Forest", ax=ax)
    PrecisionRecallDisplay.from_predictions(y_test, logit_proba, name="Logistic Regression", ax=ax)
    PrecisionRecallDisplay.from_predictions(
        y_test, heuristic_scores, name="Heuristic baseline", ax=ax
    )
    ax.set_title("Precision-Recall curve — held-out test period")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "pr_curve.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    frac_pos, mean_pred = calibration_curve(y_test, lgbm_proba, n_bins=10)
    ax.plot(mean_pred, frac_pos, marker="o", label="LightGBM")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfectly calibrated")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed late rate")
    ax.set_title("Calibration curve — LightGBM")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "calibration_curve.png", dpi=150)
    plt.close(fig)

    print("Computing SHAP explanations for LightGBM...")
    lgbm_booster = lgbm  # LGBMClassifier is directly usable by shap.TreeExplainer
    shap_values = compute_shap_values(lgbm_booster, X_test)
    importance = global_importance(shap_values, list(X_test.columns))
    importance.to_csv(REPORTS_DIR / "feature_importance.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 6))
    top = importance.head(15).iloc[::-1]
    ax.barh(top["feature"], top["mean_abs_shap"], color="#2b6cb0")
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Global feature importance — LightGBM")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "shap_importance.png", dpi=150)
    plt.close(fig)

    print("Running error-analysis slices...")
    for slice_col in ["category", "destination_region"]:
        table = slice_metrics(X_test.assign(**{}), y_test.to_numpy(), lgbm_proba, slice_col)
        table.to_csv(REPORTS_DIR / f"error_analysis_{slice_col}.csv", index=False)

    print("Building decision table and cost-threshold sweep...")
    decision_table = build_decision_table(lgbm_proba)
    decision_table.to_csv(REPORTS_DIR / "decision_table_sample.csv", index=False)

    cost_grid = sweep_thresholds(y_test.to_numpy(), lgbm_proba)
    best_row = cost_grid.loc[cost_grid["expected_cost"].idxmin()]
    print(
        f"Illustrative cost-minimizing threshold: {best_row['threshold']:.2f} "
        f"(expected_cost={best_row['expected_cost']:.0f}) — synthetic costs, not real figures."
    )

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(cost_grid["threshold"], cost_grid["expected_cost"])
    ax.axvline(
        best_row["threshold"], color="red", linestyle="--", label="Cost-minimizing threshold"
    )
    ax.set_xlabel("Flagging threshold")
    ax.set_ylabel("Illustrative expected cost (synthetic)")
    ax.set_title("Cost-sensitive threshold sweep (synthetic costs)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "cost_threshold_sweep.png", dpi=150)
    plt.close(fig)

    print(f"\nDone. Figures written to {FIGURES_DIR}, reports written to {REPORTS_DIR}.")


if __name__ == "__main__":
    main()
