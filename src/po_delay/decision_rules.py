"""Translate model risk scores into buyer/planner risk tiers and actions.

The tier thresholds and the illustrative cost figures in ``config.py`` are examples
for demonstrating a cost-sensitive decision framework — they are not derived from any
real business data and should be re-fit to an organization's actual costs before use.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from po_delay import config


def assign_risk_tier(probabilities: pd.Series | np.ndarray) -> pd.Series:
    """Map calibrated late-probabilities to a risk tier name."""
    probs = pd.Series(probabilities)
    tier = pd.Series(index=probs.index, dtype="object")
    for t in config.RISK_TIERS:
        mask = (probs >= t.lower) & (probs < t.upper)
        tier[mask] = t.name
    return tier


def recommended_action(tier_names: pd.Series) -> pd.Series:
    action_by_tier = {t.name: t.action for t in config.RISK_TIERS}
    return tier_names.map(action_by_tier)


def build_decision_table(probabilities: pd.Series | np.ndarray) -> pd.DataFrame:
    """Given per-PO late-probabilities, return tier + recommended action per row."""
    probs = pd.Series(probabilities, name="risk_probability").reset_index(drop=True)
    tier = assign_risk_tier(probs)
    action = recommended_action(tier)
    return pd.DataFrame(
        {"risk_probability": probs, "risk_tier": tier, "recommended_action": action}
    )


def expected_cost(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float,
    cost_fn: float = config.ILLUSTRATIVE_COST_FALSE_NEGATIVE,
    cost_fp: float = config.ILLUSTRATIVE_COST_FALSE_POSITIVE,
) -> float:
    """Illustrative total cost if POs with probability >= threshold are flagged for
    intervention. Uses synthetic cost assumptions to demonstrate threshold tuning —
    not a claim about real expediting or stockout costs."""
    y_true = np.asarray(y_true)
    flagged = y_proba >= threshold
    false_negatives = np.sum((~flagged) & (y_true == 1))
    false_positives = np.sum(flagged & (y_true == 0))
    return float(false_negatives * cost_fn + false_positives * cost_fp)


def sweep_thresholds(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> pd.DataFrame:
    """Compute illustrative expected cost across a grid of thresholds, to show how
    the flagging threshold should be tuned to relative cost rather than picked
    arbitrarily (e.g., a fixed 0.5 cutoff)."""
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)
    rows = [
        {"threshold": t, "expected_cost": expected_cost(y_true, y_proba, t)} for t in thresholds
    ]
    return pd.DataFrame(rows)
