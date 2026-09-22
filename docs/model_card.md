# Model Card — Purchase Order Delay Risk

## Overview

Predicts the probability that an individual purchase order (PO) line item will arrive
after its promised delivery date, using only information available at supplier
acknowledgment time. Intended to help buyers/planners prioritize follow-up on the
riskiest open orders — not to make procurement decisions autonomously.

**All data is synthetic.** See `data/README.md` for the generation methodology. Numbers
below reflect a single run of `scripts/run_pipeline.py` with `seed=42`,
`n_orders=40,000`; they describe model behavior on simulated data, not real-world
performance.

## Intended use

- Ranking currently-open POs by delay risk for buyer triage.
- Feeding the risk tiers in `src/po_delay/decision_rules.py` to trigger check-ins,
  escalations, or planning adjustments.

## Out of scope

- Fully automated supplier actions (cancellations, penalty clauses) without human
  review.
- Any claim of real-world accuracy — this model has never seen real procurement data.

## Data

Synthetic PO line items, 2023–2025 (3-year window), ~40K rows, 150 suppliers, 6
categories. Base late rate ≈ 21%. Generation process documented in `data/README.md`.

## Features

See `src/po_delay/features.py`. All features are computed from information available
at PO acknowledgment time — supplier trailing reliability stats, structural
congestion, order characteristics, and calendar/seasonality. Leakage is enforced by
`assert_no_leakage` and covered by `tests/test_leakage.py`.

## Validation

Chronological train/validation/test split (70/15/15) via
`src/po_delay/validation.time_based_split` — the test set is entirely after the
training window in time, mirroring production use. `assert_no_time_leakage` checks
this holds. Hyperparameter tuning uses expanding-window rolling-origin CV
(`rolling_origin_splits`), not random k-fold.

## Results (single run, `seed=42`, test set n=6,000)

| Model | ROC-AUC | PR-AUC | Brier score | Precision@10% |
|---|---|---|---|---|
| Heuristic baseline (planner rule) | 0.623 | 0.287 | 0.176 | 0.383 |
| Logistic Regression (scaled features) | 0.676 | 0.372 | 0.222 | 0.455 |
| Random Forest | 0.674 | 0.365 | 0.227 | 0.438 |
| **LightGBM** | **0.679** | **0.375** | 0.228 | **0.458** |

Full numbers: `reports/metrics.json`. Charts: `reports/figures/`.

**LightGBM is the selected candidate** — best ranking quality (ROC-AUC, PR-AUC,
precision@10%) among the ML models, and a real (if modest) lift over the heuristic
baseline. Note that Logistic Regression's numbers only became competitive with the
tree models after adding `StandardScaler` to its numeric preprocessing
(`src/po_delay/models.py`); an earlier iteration without scaling scored a materially
worse 0.609 ROC-AUC, purely a preprocessing artifact rather than a real capability
gap. This is left in the model card as a reminder that baseline comparisons are only
fair once every model gets appropriate preprocessing.

### Known limitation: calibration

LightGBM and Random Forest were trained with `class_weight="balanced"` to improve
ranking under the ~21% positive rate. This measurably **hurts calibration** — both
have a *worse* Brier score than the heuristic baseline, meaning their raw predicted
probabilities are not directly trustworthy as "the probability this PO is late." The
risk-tier thresholds in `decision_rules.py` are illustrative for this reason. Before
using this in the cost-sensitive threshold sweep or as an at-a-glance percentage
shown to a buyer, refit calibration (e.g.,
`sklearn.calibration.CalibratedClassifierCV` with isotonic scaling) on the validation
split, evaluate with the calibration curve in `reports/figures/calibration_curve.png`,
and only then finalize threshold cutoffs. This is deliberately left as documented
future work rather than silently swept under a good-looking AUC number.

## Explainability

`src/po_delay/explain.py` computes SHAP TreeExplainer values for LightGBM. Global
importance (`reports/feature_importance.csv`,
`reports/figures/shap_importance.png`) is dominated by
`supplier_on_time_rate_trailing365`, followed by `is_rush_order` and seasonal
(`is_quarter_end`) and congestion (`supplier_open_po_count`) features — consistent
with how the synthetic generator was designed, which is a useful sanity check that the
model is learning the intended structure rather than spurious correlations.

## Error analysis

Per-category and per-region slices (`reports/error_analysis_*.csv`) show the model
performs unevenly across segments — e.g., ROC-AUC varies from ~0.62 to ~0.69 by
category, and "Custom Fabrication" has both the highest late rate and the highest
PR-AUC, since higher variance in that category gives the model more signal to exploit.
Segments with a low sample size are flagged in the output (`low_sample` column) so
their metrics aren't over-trusted.

## Decision framework

`src/po_delay/decision_rules.py` maps a probability to a risk tier (Low / Medium /
High / Critical) and a suggested buyer action, and includes an illustrative
cost-sensitive threshold sweep (`sweep_thresholds`) using placeholder cost figures
(`config.ILLUSTRATIVE_COST_FALSE_NEGATIVE`, `..._FALSE_POSITIVE`). **These costs are
synthetic examples of the methodology, not real business figures**, and must be
replaced with an organization's actual cost estimates before use.

## Limitations

- Synthetic data encodes the author's assumptions about what drives delay risk; a
  real deployment would very likely find different feature relationships.
- No live feedback loop, no A/B test, no evidence of real-world business impact.
- Calibration issue noted above — do not read raw probabilities as literal
  percentages without recalibration.
- Cold-start suppliers (no order history) get a neutral imputed trailing-reliability
  value and are separately flagged via `is_new_supplier`; the model has limited signal
  for genuinely new suppliers.

## Safe portfolio claims

This project demonstrates: an end-to-end, leakage-safe ML pipeline; chronological
validation appropriate to a forecasting-style problem; baseline-to-candidate model
comparison; SHAP-based explainability and slice-based error analysis; translation of
model output into a documented business decision framework; and a tested, CI-checked
codebase. It does **not** demonstrate real-world business impact, real supplier data,
or production deployment experience.
