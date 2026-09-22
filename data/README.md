# Data

All data in this project is **synthetically generated**. No real supplier names, employer data,
or confidential business information is used anywhere in this repository.

## Generation

Run:

```bash
python -m po_delay.data_generation --n-orders 40000 --seed 42 --out data/raw/po_lines.csv
```

The generator (`src/po_delay/data_generation.py`) is fully deterministic given a seed. It
simulates purchase order line items over a 3-year window with:

- Heterogeneous supplier reliability (base on-time rate per supplier, drawn from a Beta
  distribution so most suppliers cluster around reliable behavior with a long tail of
  chronically-late suppliers)
- Category-level baseline lead times and delay variance (commodity vs. fabricated/custom parts)
- Seasonal effects: quarter-end order surges and holiday-period congestion increase delay risk
- Rush/expedited orders (short requested lead time relative to category norm) carry elevated risk
- Supplier concurrent workload (number of open POs) as a congestion proxy
- Realistic noise: the true delay probability is never fully determined by observed features,
  so the prediction problem is not trivially separable

## Files

- `raw/po_lines.csv` — one row per PO line item, output of the generator. Not committed to git
  (regenerate locally); a `.gitkeep` placeholder keeps the directory tracked.
- `processed/` — feature-engineered train/validation/test splits produced by
  `src/po_delay/features.py` and `src/po_delay/validation.py`. Also not committed.

## Schema: `raw/po_lines.csv`

| Column | Type | Description | Known at prediction time? |
|---|---|---|---|
| `po_line_id` | str | Unique PO line identifier | — |
| `supplier_id` | str | Supplier identifier | yes |
| `supplier_country` | str | Supplier home region/country code | yes |
| `category` | str | Material/item category | yes |
| `order_date` | date (`YYYY-MM-DD`) | Date PO line was created | yes |
| `ack_date` | date (`YYYY-MM-DD`) | Date supplier acknowledged the order (prediction point) | yes |
| `promised_date` | date (`YYYY-MM-DD`) | Promised delivery date as of acknowledgment | yes |
| `requested_lead_time_days` | int | Days between order and promised delivery | yes |
| `quantity` | int | Order quantity | yes |
| `unit_price` | float | Unit price | yes |
| `total_value` | float | `quantity * unit_price` | yes |
| `incoterms` | str | Incoterms code | yes |
| `shipping_mode` | str | Air / Sea / Road / Rail | yes |
| `destination_region` | str | Receiving plant region | yes |
| `payment_terms_days` | int | Net payment terms | yes |
| `is_rush_order` | int (0/1) | Requested lead time well below category norm | yes |
| `price_dev_from_supplier_avg` | float | Unit price z-score vs. supplier's historical average | yes |
| `supplier_open_po_count` | int | Supplier's concurrently open POs at acknowledgment time | yes |
| `supplier_on_time_rate_trailing365` | float | Supplier's on-time rate over prior 365 days (NaN if no history) | yes |
| `supplier_mean_delay_trailing365` | float | Supplier's mean delay days over prior 365 days | yes |
| `supplier_n_orders_trailing365` | int | Number of prior orders used for the trailing stats | yes |
| `actual_delivery_date` | date (`YYYY-MM-DD`) | **Ground truth** — actual delivery date | **no (label source, post-event)** |
| `delay_days` | int | `actual_delivery_date - promised_date` | **no (label source, post-event)** |
| `is_late` | int (0/1) | Primary target: `delay_days > 0` | **no (target)** |

`actual_delivery_date`, `delay_days`, and `is_late` are only known after the fact and must never
be used as model inputs — `src/po_delay/features.py` and `tests/test_leakage.py` enforce this.
