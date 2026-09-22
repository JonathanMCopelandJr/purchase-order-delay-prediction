-- Leakage-safe feature queries for the PO delay model.
--
-- These mirror the Python logic in src/po_delay/features.py and
-- src/po_delay/data_generation.py's trailing-stat computation, expressed in SQL for
-- a warehouse setting. The guiding rule throughout: a feature for PO X may only use
-- rows from OTHER POs whose outcome (delivery) was already known as of X's ack_date.

-- 1. Supplier trailing reliability, as of each PO's acknowledgment date.
-- Uses a self-join restricted to prior orders that were delivered before this
-- order's ack_date, within a 365-day trailing window - never a future order,
-- and never an order that hasn't been delivered yet.
WITH supplier_trailing AS (
    SELECT
        cur.po_line_id,
        cur.supplier_id,
        cur.ack_date,
        COUNT(hist.po_line_id)                                   AS supplier_n_orders_trailing365,
        AVG(CASE WHEN hist.is_late = 0 THEN 1.0 ELSE 0.0 END)    AS supplier_on_time_rate_trailing365,
        AVG(hist.delay_days * 1.0)                                AS supplier_mean_delay_trailing365
    FROM po_lines AS cur
    LEFT JOIN po_lines AS hist
        ON hist.supplier_id = cur.supplier_id
       AND hist.po_line_id <> cur.po_line_id
       AND hist.actual_delivery_date IS NOT NULL
       AND hist.actual_delivery_date < cur.ack_date
       AND hist.actual_delivery_date >= DATE(cur.ack_date, '-365 days')
    GROUP BY cur.po_line_id, cur.supplier_id, cur.ack_date
),

-- 2. Supplier concurrent workload: other POs from the same supplier that are
-- "open" (already promised, not yet due) as of this order's ack_date. This uses
-- only order timing (order_date/ack_date/promised_date), never outcomes, so it is
-- safe to use even for the very first order from a supplier.
supplier_congestion AS (
    SELECT
        cur.po_line_id,
        COUNT(other.po_line_id) AS supplier_open_po_count
    FROM po_lines AS cur
    LEFT JOIN po_lines AS other
        ON other.supplier_id = cur.supplier_id
       AND other.po_line_id <> cur.po_line_id
       AND other.ack_date < cur.ack_date
       AND other.promised_date >= cur.ack_date
    GROUP BY cur.po_line_id
),

-- 3. Price deviation vs. the supplier's own historical average unit price,
-- using only orders PLACED (order_date) strictly before this one.
supplier_price_history AS (
    SELECT
        cur.po_line_id,
        cur.unit_price,
        AVG(hist.unit_price) AS supplier_avg_price_to_date
    FROM po_lines AS cur
    LEFT JOIN po_lines AS hist
        ON hist.supplier_id = cur.supplier_id
       AND hist.po_line_id <> cur.po_line_id
       AND hist.order_date < cur.order_date
    GROUP BY cur.po_line_id, cur.unit_price
)

-- 4. Assembled leakage-safe feature view for scoring OPEN purchase orders
-- (i.e., WHERE actual_delivery_date IS NULL — the model's real production input).
SELECT
    p.po_line_id,
    p.supplier_id,
    p.category,
    p.requested_lead_time_days,
    p.quantity,
    p.unit_price,
    p.total_value,
    p.incoterms,
    p.shipping_mode,
    p.destination_region,
    p.payment_terms_days,
    p.is_rush_order,
    st.supplier_on_time_rate_trailing365,
    st.supplier_mean_delay_trailing365,
    st.supplier_n_orders_trailing365,
    sc.supplier_open_po_count,
    sph.supplier_avg_price_to_date
FROM po_lines AS p
LEFT JOIN supplier_trailing AS st ON st.po_line_id = p.po_line_id
LEFT JOIN supplier_congestion AS sc ON sc.po_line_id = p.po_line_id
LEFT JOIN supplier_price_history AS sph ON sph.po_line_id = p.po_line_id
WHERE p.actual_delivery_date IS NULL;  -- only open orders are scored in production
