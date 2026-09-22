"""Synthetic purchase-order line-item data generator.

Every value produced here is synthetic. The generative process is designed to mimic
realistic supply-chain structure (heterogeneous supplier reliability, category effects,
seasonality, rush-order risk, supplier congestion) without representing any real
company, supplier, or transaction.

The generator is deterministic given a seed, so the resulting dataset is fully
reproducible.

Design note on leakage safety: the "true" per-order delay probability is driven by each
supplier's hidden reliability parameter plus category/seasonal/rush/congestion effects.
Trailing supplier statistics (on-time rate, mean delay) are computed *after* delay
outcomes are generated, using only other orders whose ``actual_delivery_date`` precedes
the current order's acknowledgment date. This mirrors what a real buyer would actually
know at prediction time and keeps the generator itself leakage-safe by construction.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd

from po_delay import config


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


@dataclass
class GeneratorParams:
    n_orders: int = config.N_ORDERS_DEFAULT
    n_suppliers: int = config.N_SUPPLIERS_DEFAULT
    seed: int = config.RANDOM_SEED
    date_range_years: int = config.DATE_RANGE_YEARS


def _make_suppliers(rng: np.random.Generator, n_suppliers: int) -> pd.DataFrame:
    categories = list(config.CATEGORIES.keys())
    supplier_ids = [f"SUP-{i:04d}" for i in range(n_suppliers)]
    # Beta(6, 2) skews toward reliable suppliers with a long unreliable tail.
    reliability_raw = rng.beta(6, 2, size=n_suppliers)
    base_on_time_rate = 0.50 + 0.48 * reliability_raw  # in [0.50, 0.98]
    delay_scale = rng.gamma(shape=2.0, scale=3.0, size=n_suppliers) + 1.0
    primary_category = rng.choice(categories, size=n_suppliers)
    country = rng.choice(config.SUPPLIER_COUNTRIES, size=n_suppliers)
    return pd.DataFrame(
        {
            "supplier_id": supplier_ids,
            "supplier_country": country,
            "primary_category": primary_category,
            "base_on_time_rate": base_on_time_rate,
            "delay_scale": delay_scale,
        }
    )


def _assign_orders_to_suppliers(
    rng: np.random.Generator, n_orders: int, suppliers: pd.DataFrame
) -> pd.DataFrame:
    categories = list(config.CATEGORIES.keys())
    order_category = rng.choice(categories, size=n_orders)

    # For each order, sample a supplier whose primary_category matches, falling back
    # to any supplier if a category has none (shouldn't happen with n_suppliers>=150).
    supplier_by_category = {
        cat: suppliers.index[suppliers["primary_category"] == cat].to_numpy() for cat in categories
    }
    chosen_idx = np.empty(n_orders, dtype=int)
    for cat in categories:
        mask = order_category == cat
        pool = supplier_by_category[cat]
        if len(pool) == 0:
            pool = suppliers.index.to_numpy()
        chosen_idx[mask] = rng.choice(pool, size=mask.sum())

    orders = suppliers.iloc[chosen_idx].reset_index(drop=True)
    orders["category"] = order_category
    return orders


def generate(params: GeneratorParams | None = None) -> pd.DataFrame:
    """Generate a synthetic PO line-item dataset.

    Returns a DataFrame with the schema documented in ``data/README.md``.
    """
    params = params or GeneratorParams()
    rng = np.random.default_rng(params.seed)

    suppliers = _make_suppliers(rng, params.n_suppliers)
    orders = _assign_orders_to_suppliers(rng, params.n_orders, suppliers)
    n = len(orders)

    # --- Order dates ---------------------------------------------------
    start = pd.Timestamp("2023-01-01")
    span_days = 365 * params.date_range_years
    order_offsets = rng.integers(0, span_days, size=n)
    order_date = start + pd.to_timedelta(order_offsets, unit="D")
    ack_offset = rng.integers(0, 4, size=n)  # 0-3 days to acknowledge
    ack_date = order_date + pd.to_timedelta(ack_offset, unit="D")

    # --- Category-driven fields -----------------------------------------
    cat_lead = orders["category"].map(lambda c: config.CATEGORIES[c]["lead_time_days"]).to_numpy()
    cat_risk = orders["category"].map(lambda c: config.CATEGORIES[c]["risk_multiplier"]).to_numpy()
    price_lo = orders["category"].map(lambda c: config.CATEGORIES[c]["price_range"][0]).to_numpy()
    price_hi = orders["category"].map(lambda c: config.CATEGORIES[c]["price_range"][1]).to_numpy()

    lead_time_factor = rng.normal(1.0, 0.25, size=n).clip(0.35, 2.0)
    requested_lead_time_days = np.maximum(1, np.round(cat_lead * lead_time_factor)).astype(int)
    is_rush_order = (lead_time_factor < 0.65).astype(int)

    promised_date = ack_date + pd.to_timedelta(requested_lead_time_days, unit="D")

    quantity = rng.integers(1, 500, size=n)
    unit_price = rng.uniform(price_lo, price_hi) * rng.lognormal(0, 0.15, size=n)
    total_value = quantity * unit_price

    shipping_mode = rng.choice(config.SHIPPING_MODES, size=n)
    incoterms = rng.choice(config.INCOTERMS, size=n)
    destination_region = rng.choice(config.DESTINATION_REGIONS, size=n)
    payment_terms_days = rng.choice(config.PAYMENT_TERMS_DAYS, size=n)

    # --- Structural congestion: how many other POs from the same supplier ---
    # --- are "open" (promised, not yet due) around this order's ack date. ---
    df = orders.copy()
    df["order_date"] = order_date
    df["ack_date"] = ack_date
    df["promised_date"] = promised_date
    df["po_line_id"] = [f"PO-{i:07d}" for i in range(n)]

    supplier_open_po_count = np.zeros(n, dtype=int)
    for _, idx in df.groupby("supplier_id").groups.items():
        idx = np.asarray(idx)
        acks = ack_date.to_numpy()[idx]
        proms = promised_date.to_numpy()[idx]
        order = np.argsort(acks)
        acks_sorted = acks[order]
        proms_sorted = proms[order]
        counts = np.zeros(len(idx), dtype=int)
        for j in range(len(idx)):
            this_ack = acks_sorted[j]
            open_mask = (acks_sorted < this_ack) & (proms_sorted >= this_ack)
            counts[j] = open_mask.sum()
        supplier_open_po_count[idx[order]] = counts

    # --- Seasonality -----------------------------------------------------
    month = ack_date.month
    day = ack_date.day
    is_quarter_end = np.isin(month, [3, 6, 9, 12]) & (day >= 21)
    day_of_year = ack_date.dayofyear.to_numpy()
    dist_to_new_year = np.minimum(day_of_year, 366 - day_of_year)
    is_holiday_period = dist_to_new_year <= 10

    # --- True delay-risk model -------------------------------------------
    base_on_time = df["base_on_time_rate"].to_numpy()
    base_late_logit = -np.log(base_on_time / (1 - base_on_time))  # higher = more late risk

    logit_p_late = (
        base_late_logit
        + 0.55 * (cat_risk - 1.0)
        + 0.9 * is_rush_order
        + 0.35 * is_quarter_end
        + 0.45 * is_holiday_period
        + 0.06 * np.minimum(supplier_open_po_count, 10)
        + rng.normal(0, 0.5, size=n)
    )
    p_late = _sigmoid(logit_p_late)
    is_late = (rng.uniform(size=n) < p_late).astype(int)

    delay_scale = df["delay_scale"].to_numpy()
    late_severity = rng.gamma(shape=2.0, scale=delay_scale * cat_risk, size=n)
    late_days = np.maximum(1, np.round(late_severity)).astype(int)
    early_days = -rng.integers(0, np.maximum(1, requested_lead_time_days // 4) + 1, size=n)
    delay_days = np.where(is_late == 1, late_days, early_days)

    actual_delivery_date = promised_date + pd.to_timedelta(delay_days, unit="D")

    df["requested_lead_time_days"] = requested_lead_time_days
    df["is_rush_order"] = is_rush_order
    df["quantity"] = quantity
    df["unit_price"] = np.round(unit_price, 2)
    df["total_value"] = np.round(total_value, 2)
    df["shipping_mode"] = shipping_mode
    df["incoterms"] = incoterms
    df["destination_region"] = destination_region
    df["payment_terms_days"] = payment_terms_days
    df["supplier_open_po_count"] = supplier_open_po_count
    df["actual_delivery_date"] = actual_delivery_date
    df["delay_days"] = delay_days
    df["is_late"] = is_late

    # --- Trailing supplier stats (computed strictly from prior, already- ---
    # --- delivered orders relative to this order's ack_date) ---------------
    on_time_rate_trailing = np.full(n, np.nan)
    mean_delay_trailing = np.full(n, np.nan)
    n_orders_trailing = np.zeros(n, dtype=int)
    price_dev_from_supplier_avg = np.zeros(n)

    for _, idx in df.groupby("supplier_id").groups.items():
        idx = np.asarray(idx)
        acks = df["ack_date"].to_numpy()[idx]
        delivered = df["actual_delivery_date"].to_numpy()[idx]
        late_flags = df["is_late"].to_numpy()[idx]
        delays = df["delay_days"].to_numpy()[idx]
        prices = df["unit_price"].to_numpy()[idx]

        deliv_order = np.argsort(delivered)
        delivered_sorted = delivered[deliv_order]
        late_sorted = late_flags[deliv_order]
        delay_sorted = delays[deliv_order]
        cum_late = np.cumsum(late_sorted)
        cum_delay = np.cumsum(delay_sorted)

        window_start = acks - np.timedelta64(365, "D")
        lo = np.searchsorted(delivered_sorted, window_start, side="left")
        hi = np.searchsorted(delivered_sorted, acks, side="left")

        trailing_n = hi - lo
        trailing_late = np.where(hi > 0, cum_late[np.clip(hi - 1, 0, None)], 0) - np.where(
            lo > 0, cum_late[np.clip(lo - 1, 0, None)], 0
        )
        trailing_delay = np.where(hi > 0, cum_delay[np.clip(hi - 1, 0, None)], 0) - np.where(
            lo > 0, cum_delay[np.clip(lo - 1, 0, None)], 0
        )

        with np.errstate(invalid="ignore", divide="ignore"):
            on_time_rate = np.where(
                trailing_n > 0, 1.0 - trailing_late / np.maximum(trailing_n, 1), np.nan
            )
            mean_delay = np.where(
                trailing_n > 0, trailing_delay / np.maximum(trailing_n, 1), np.nan
            )

        on_time_rate_trailing[idx] = on_time_rate
        mean_delay_trailing[idx] = mean_delay
        n_orders_trailing[idx] = trailing_n

        # Price deviation vs. supplier's historical average, using orders
        # placed (order_date) strictly before this one — no outcome needed.
        order_dates = df["order_date"].to_numpy()[idx]
        price_order = np.argsort(order_dates)
        order_dates_sorted = order_dates[price_order]
        prices_sorted = prices[price_order]
        cum_price_sum = np.cumsum(prices_sorted)

        pos = np.searchsorted(order_dates_sorted, order_dates, side="left")
        prior_n = pos
        prior_sum = np.where(pos > 0, cum_price_sum[np.clip(pos - 1, 0, None)], 0.0)
        prior_mean = np.where(prior_n > 0, prior_sum / np.maximum(prior_n, 1), prices)
        std_price = prices.std() if prices.std() > 0 else 1.0
        price_dev_from_supplier_avg[idx] = (prices - prior_mean) / std_price

    df["supplier_on_time_rate_trailing365"] = on_time_rate_trailing
    df["supplier_mean_delay_trailing365"] = mean_delay_trailing
    df["supplier_n_orders_trailing365"] = n_orders_trailing
    df["price_dev_from_supplier_avg"] = np.round(price_dev_from_supplier_avg, 3)

    ordered_cols = [
        "po_line_id",
        "supplier_id",
        "supplier_country",
        "category",
        "order_date",
        "ack_date",
        "promised_date",
        "requested_lead_time_days",
        "quantity",
        "unit_price",
        "total_value",
        "incoterms",
        "shipping_mode",
        "destination_region",
        "payment_terms_days",
        "is_rush_order",
        "price_dev_from_supplier_avg",
        "supplier_open_po_count",
        "supplier_on_time_rate_trailing365",
        "supplier_mean_delay_trailing365",
        "supplier_n_orders_trailing365",
        "actual_delivery_date",
        "delay_days",
        "is_late",
    ]
    out = df[ordered_cols].sort_values("order_date").reset_index(drop=True)
    for col in ["order_date", "ack_date", "promised_date", "actual_delivery_date"]:
        out[col] = pd.to_datetime(out[col]).dt.strftime("%Y-%m-%d")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-orders", type=int, default=config.N_ORDERS_DEFAULT)
    parser.add_argument("--n-suppliers", type=int, default=config.N_SUPPLIERS_DEFAULT)
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    parser.add_argument("--out", type=str, default="data/raw/po_lines.csv")
    args = parser.parse_args()

    params = GeneratorParams(n_orders=args.n_orders, n_suppliers=args.n_suppliers, seed=args.seed)
    df = generate(params)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df):,} rows to {args.out}")
    print(f"Late rate: {df['is_late'].mean():.3f}")


if __name__ == "__main__":
    main()
