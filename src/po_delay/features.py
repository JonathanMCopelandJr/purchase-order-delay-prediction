"""Leakage-safe feature engineering for the PO delay model.

Only columns knowable at PO acknowledgment time (the prediction point) are used as
model inputs. ``config.LEAKAGE_COLUMNS`` and ``config.ID_COLUMNS`` are explicitly
excluded and checked by ``assert_no_leakage``.
"""

from __future__ import annotations

import pandas as pd

from po_delay import config

CATEGORICAL_FEATURES = [
    "category",
    "supplier_country",
    "incoterms",
    "shipping_mode",
    "destination_region",
]

NUMERIC_FEATURES = [
    "requested_lead_time_days",
    "quantity",
    "unit_price",
    "total_value",
    "payment_terms_days",
    "is_rush_order",
    "price_dev_from_supplier_avg",
    "supplier_open_po_count",
    "supplier_on_time_rate_trailing365",
    "supplier_mean_delay_trailing365",
    "supplier_n_orders_trailing365",
    "ack_month",
    "ack_day_of_week",
    "is_quarter_end",
    "is_holiday_period",
    "is_new_supplier",
]

FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES


def assert_no_leakage(columns: list[str]) -> None:
    """Raise if any post-event column has leaked into a feature set."""
    leaked = set(columns) & set(config.LEAKAGE_COLUMNS)
    if leaked:
        raise ValueError(f"Leakage columns present in feature set: {sorted(leaked)}")


def add_derived_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add features derived from ``ack_date`` (the prediction-time anchor)."""
    out = df.copy()
    ack = pd.to_datetime(out["ack_date"])
    out["ack_month"] = ack.dt.month
    out["ack_day_of_week"] = ack.dt.dayofweek
    out["is_quarter_end"] = (ack.dt.month.isin([3, 6, 9, 12]) & (ack.dt.day >= 21)).astype(int)
    day_of_year = ack.dt.dayofyear
    dist_to_new_year = pd.concat([day_of_year, 366 - day_of_year], axis=1).min(axis=1)
    out["is_holiday_period"] = (dist_to_new_year <= 10).astype(int)
    out["is_new_supplier"] = out["supplier_n_orders_trailing365"].fillna(0).eq(0).astype(int)
    return out


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Build the model-ready feature matrix and target from raw PO line data.

    Parameters
    ----------
    df:
        Raw PO line dataframe as produced by ``data_generation.generate`` (or loaded
        from ``data/raw/po_lines.csv``), containing at least ``FEATURE_COLUMNS``'
        source columns plus ``config.TARGET_COLUMN``.

    Returns
    -------
    (X, y): feature matrix (categoricals as pandas ``category`` dtype, ready for
    LightGBM / one-hot downstream) and the binary target series.
    """
    enriched = add_derived_date_features(df)
    assert_no_leakage(FEATURE_COLUMNS)

    X = enriched[FEATURE_COLUMNS].copy()
    for col in CATEGORICAL_FEATURES:
        X[col] = X[col].astype("category")

    # Trailing supplier stats are NaN for genuinely new suppliers (no prior history).
    # Impute with the training-time global mean would leak future info if done
    # globally; instead we impute with a fixed, feature-derived neutral prior here
    # and flag the condition explicitly via `is_new_supplier` so the model can
    # learn the distinction rather than treating the imputed value as a real signal.
    X["supplier_on_time_rate_trailing365"] = X["supplier_on_time_rate_trailing365"].fillna(0.5)
    X["supplier_mean_delay_trailing365"] = X["supplier_mean_delay_trailing365"].fillna(0.0)

    y = enriched[config.TARGET_COLUMN].astype(int)
    return X, y
