"""Time-based validation utilities.

Because PO delay risk drifts over time (seasonality, supplier mix changes), we split
strictly by date rather than by random shuffling: everything in the test set must be
chronologically after everything in train/validation. This mirrors how the model would
actually be used in production (train on history, predict on the future).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TimeSplit:
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray


def time_based_split(
    df: pd.DataFrame,
    date_col: str = "ack_date",
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> TimeSplit:
    """Split a dataframe into chronological train/val/test index arrays.

    Rows are ordered by ``date_col`` and cut at the ``train_frac`` / ``val_frac``
    quantiles, so ``max(train dates) <= min(val dates) <= max(val dates) <= min(test
    dates)`` holds by construction.
    """
    if not (0 < train_frac < 1) or not (0 < val_frac < 1) or train_frac + val_frac >= 1:
        raise ValueError("train_frac and val_frac must be in (0, 1) and sum to < 1")

    dates = pd.to_datetime(df[date_col])
    order = np.argsort(dates.to_numpy(), kind="stable")
    n = len(order)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    train_idx = order[:n_train]
    val_idx = order[n_train : n_train + n_val]
    test_idx = order[n_train + n_val :]
    return TimeSplit(train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)


def assert_no_time_leakage(df: pd.DataFrame, split: TimeSplit, date_col: str = "ack_date") -> None:
    """Raise if any split boundary is violated (test dates before train dates, etc.)."""
    dates = pd.to_datetime(df[date_col])
    max_train = dates.iloc[split.train_idx].max()
    min_val = dates.iloc[split.val_idx].min()
    max_val = dates.iloc[split.val_idx].max()
    min_test = dates.iloc[split.test_idx].min()

    if max_train > min_val:
        raise ValueError(
            f"Time leakage: max(train {date_col})={max_train} > min(val {date_col})={min_val}"
        )
    if max_val > min_test:
        raise ValueError(
            f"Time leakage: max(val {date_col})={max_val} > min(test {date_col})={min_test}"
        )


def rolling_origin_splits(
    df: pd.DataFrame,
    date_col: str = "ack_date",
    n_splits: int = 5,
    min_train_frac: float = 0.4,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Expanding-window CV splits for hyperparameter tuning within the training set.

    Fold ``k`` trains on the earliest ``min_train_frac + k * step`` fraction of rows
    and validates on the next slice — never on data chronologically before it.
    """
    dates = pd.to_datetime(df[date_col])
    order = np.argsort(dates.to_numpy(), kind="stable")
    n = len(order)
    start = int(n * min_train_frac)
    remaining = n - start
    step = remaining // (n_splits + 1)
    if step <= 0:
        raise ValueError("Not enough rows for the requested number of rolling-origin splits")

    splits = []
    for k in range(n_splits):
        train_end = start + k * step
        val_end = train_end + step
        train_idx = order[:train_end]
        val_idx = order[train_end:val_end]
        splits.append((train_idx, val_idx))
    return splits
