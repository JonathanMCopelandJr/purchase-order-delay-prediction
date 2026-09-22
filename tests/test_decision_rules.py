import numpy as np
import pandas as pd

from po_delay.decision_rules import (
    assign_risk_tier,
    build_decision_table,
    expected_cost,
    sweep_thresholds,
)


def test_assign_risk_tier_boundaries():
    probs = pd.Series([0.0, 0.05, 0.10, 0.29, 0.30, 0.59, 0.60, 0.99, 1.0])
    tiers = assign_risk_tier(probs)
    assert list(tiers) == [
        "Low",
        "Low",
        "Medium",
        "Medium",
        "High",
        "High",
        "Critical",
        "Critical",
        "Critical",
    ]


def test_build_decision_table_has_action_for_every_row():
    probs = np.array([0.02, 0.5, 0.8])
    table = build_decision_table(probs)
    assert len(table) == 3
    assert table["recommended_action"].notna().all()
    assert set(table["risk_tier"]) <= {"Low", "Medium", "High", "Critical"}


def test_expected_cost_zero_when_perfectly_classified():
    y_true = np.array([0, 0, 1, 1])
    y_proba = np.array([0.0, 0.0, 1.0, 1.0])
    assert expected_cost(y_true, y_proba, threshold=0.5) == 0.0


def test_expected_cost_penalizes_false_negatives_more_by_default():
    y_true = np.array([1])
    y_proba_missed = np.array([0.0])  # false negative
    y_true_fp = np.array([0])
    y_proba_fp = np.array([1.0])  # false positive
    fn_cost = expected_cost(y_true, y_proba_missed, threshold=0.5)
    fp_cost = expected_cost(y_true_fp, y_proba_fp, threshold=0.5)
    assert fn_cost > fp_cost


def test_sweep_thresholds_returns_full_grid():
    y_true = np.array([0, 1, 0, 1, 0])
    y_proba = np.array([0.1, 0.9, 0.4, 0.6, 0.2])
    grid = sweep_thresholds(y_true, y_proba)
    assert len(grid) == 99
    assert {"threshold", "expected_cost"} == set(grid.columns)
