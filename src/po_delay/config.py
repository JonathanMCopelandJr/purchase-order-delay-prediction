"""Shared constants and configuration for the synthetic PO delay pipeline.

All values here describe a synthetic simulation, not any real supplier, employer, or
business relationship.
"""

from __future__ import annotations

from dataclasses import dataclass

RANDOM_SEED = 42

# --- Categories -------------------------------------------------------------
# lead_time_days: typical requested lead time norm for the category
# risk_multiplier: relative multiplier on delay-risk logit for the category
# price_range: (low, high) unit price band
CATEGORIES: dict[str, dict] = {
    "Electronics": {"lead_time_days": 21, "risk_multiplier": 0.9, "price_range": (5, 500)},
    "Raw Materials": {"lead_time_days": 14, "risk_multiplier": 1.0, "price_range": (1, 50)},
    "Packaging": {"lead_time_days": 10, "risk_multiplier": 0.8, "price_range": (0.5, 20)},
    "Custom Fabrication": {"lead_time_days": 45, "risk_multiplier": 1.6, "price_range": (50, 2000)},
    "MRO": {"lead_time_days": 7, "risk_multiplier": 0.7, "price_range": (5, 300)},
    "Chemicals": {"lead_time_days": 18, "risk_multiplier": 1.2, "price_range": (2, 100)},
}

SHIPPING_MODES = ["Road", "Sea", "Air", "Rail"]
INCOTERMS = ["EXW", "FOB", "CIF", "DDP"]
DESTINATION_REGIONS = ["NA-East", "NA-West", "EU-West", "EU-East", "APAC", "LATAM"]
PAYMENT_TERMS_DAYS = [30, 45, 60, 90]
SUPPLIER_COUNTRIES = ["US", "MX", "DE", "PL", "CN", "VN", "IN", "BR"]

N_SUPPLIERS_DEFAULT = 150
N_ORDERS_DEFAULT = 40_000
DATE_RANGE_YEARS = 3

# Columns known only after delivery — must never be used as model features.
LEAKAGE_COLUMNS = ["actual_delivery_date", "delay_days", "is_late"]

# Columns that identify the row but carry no predictive signal on their own.
ID_COLUMNS = ["po_line_id", "supplier_id"]

TARGET_COLUMN = "is_late"

# --- Risk tiers / decision rules --------------------------------------------


@dataclass(frozen=True)
class RiskTier:
    name: str
    lower: float  # inclusive
    upper: float  # exclusive (1.0 for the top tier)
    action: str


RISK_TIERS: list[RiskTier] = [
    RiskTier("Low", 0.0, 0.10, "No action — monitor via standard reporting."),
    RiskTier("Medium", 0.10, 0.30, "Automated supplier check-in email."),
    RiskTier(
        "High",
        0.30,
        0.60,
        "Buyer follow-up call with supplier; evaluate expedite shipping quote.",
    ),
    RiskTier(
        "Critical",
        0.60,
        1.0 + 1e-9,
        "Escalate: evaluate alternate sourcing / expedite fees; "
        "alert downstream planning to adjust schedule or safety stock.",
    ),
]

# Illustrative, synthetic cost assumptions used only to demonstrate
# cost-sensitive threshold tuning. These are NOT real business figures.
ILLUSTRATIVE_COST_FALSE_NEGATIVE = 5_000.0  # cost of an unflagged late PO (e.g., stockout)
ILLUSTRATIVE_COST_FALSE_POSITIVE = 300.0  # cost of unnecessary expediting on a flagged PO
