-- Schema for the synthetic purchase-order delay dataset.
-- Mirrors the columns produced by src/po_delay/data_generation.py and documented
-- in data/README.md. Written against standard SQL (tested against SQLite/Postgres
-- syntax); adjust types as needed for your target engine.

CREATE TABLE IF NOT EXISTS suppliers (
    supplier_id            TEXT PRIMARY KEY,
    supplier_country       TEXT NOT NULL,
    primary_category       TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS po_lines (
    po_line_id                          TEXT PRIMARY KEY,
    supplier_id                         TEXT NOT NULL REFERENCES suppliers (supplier_id),
    category                            TEXT NOT NULL,
    order_date                          DATE NOT NULL,
    ack_date                            DATE NOT NULL,
    promised_date                       DATE NOT NULL,
    requested_lead_time_days            INTEGER NOT NULL,
    quantity                            INTEGER NOT NULL,
    unit_price                          NUMERIC(12, 2) NOT NULL,
    total_value                         NUMERIC(14, 2) NOT NULL,
    incoterms                           TEXT NOT NULL,
    shipping_mode                       TEXT NOT NULL,
    destination_region                  TEXT NOT NULL,
    payment_terms_days                  INTEGER NOT NULL,
    is_rush_order                       INTEGER NOT NULL CHECK (is_rush_order IN (0, 1)),
    -- Post-event / label columns. Never join these back into a feature view used
    -- for scoring open (undelivered) POs — see sql/feature_queries.sql.
    actual_delivery_date                DATE,
    delay_days                          INTEGER,
    is_late                             INTEGER CHECK (is_late IN (0, 1))
);

CREATE INDEX IF NOT EXISTS idx_po_lines_supplier_ack ON po_lines (supplier_id, ack_date);
CREATE INDEX IF NOT EXISTS idx_po_lines_ack_date ON po_lines (ack_date);
