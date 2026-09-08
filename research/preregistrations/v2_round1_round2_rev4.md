# Brazil-RV v2 Round 1 / Round 2 registration rev 4

Status: frozen before any rev-4 score. Rev 3 is void for future research decisions,
but its sealed artifacts remain immutable engineering evidence. This registration keeps
the rev-3 folds, purges, clock, score heads, K/buffer/caps/expiry/settlement rules,
costs, seeds, source tiers, and protected-window rules except where explicitly
superseded below. No official-validation or permanently spent test row may be read.

## Purpose and immutable population

The rev-3 readout showed that learned scores retained nonlinear volatility, beta, and
momentum tilts and that nominal dollar neutrality did not imply market neutrality.
Rev 4 therefore changes both the neutral target and the executable book. The immutable
store remains the sealed `8021e42` build; the neutral target remains a deterministic
virtual view over its hash-bound arrays. Evaluation windows are F1 2023-07-03 through
2023-12-29, F2 2024-01-02 through 2024-06-28, and F3 2024-07-01 through 2024-12-30,
with the existing 55-session selection windows and ten-session purges.

## Target

For each date and each horizon in 1, 2, 3, 5, and 10 sessions, clip the legacy scaled
return to [-5, 5]. On names satisfying the existing target and characteristic masks,
fit float64 OLS with ten equal-count current `yang_zhang_vol_20` decile dummies, five
equal-count current `beta_60` quintile dummies, and linear rank-Gaussianized
`log_volume_mean_20`. There is no separate intercept: the full dummy blocks absorb it.
The target is the tie-aware cross-sectional midrank of the residual. At 20--39 names,
use and flag the rev-3 intercept plus three linear rank-Gaussianized risks; below 20,
the target is invalid. Values exposed by the store are float32. A score that is any
function only of the volatility decile must have absolute target Spearman below 0.01;
inverse-volatility-20 must have absolute neutral IC below 0.02 in every fold.

## Constructed headline book

The headline is `cost_4_borrow_lending_v1`. Short entries require a lending rate or
balance observed in the prior 20 sessions. They pay the last observed annual rate with
a 2% floor. Exits are never blocked. `comparator_lending_unconstructed` uses the same
borrow information without volatility balancing or the hedge;
`comparator_uniform_borrow` reproduces the rev-3 uniform-borrow, unconstructed,
unhedged book.

Each side allocates its effective K equally across five current
`yang_zhang_vol_20` quintiles. Remainders go to middle quintiles first. Candidates are
consumed in score order within a quintile; exhausted or unshortable strata spill unused
quota to the best remaining candidates overall. Existing holdings are not exited merely
because their quintile changes. Each side's mean realized occupancy must be within two
slots of mean quota in every quintile.

A separate BOVA11 hedge uses ISIN `BRBOVACTF003`, ticker `BOVA11`, security spec `CI`,
market type 10, and the continuous B3 ETF BDI classification 14. The supplied BDI 02
classification was corrected before scores: raw COTAHIST shows BDI 14 throughout all
registered 2023--2024 sessions, while 02 is only a brief 2019 classification. The
close series is a separately hash-bound, development-only, calendar-aligned artifact
and must reject 2025/2026 rows. Each session the ledger computes signed ex-ante beta
over held and pending equity notional and targets `-beta_net * NAV` in BOVA11. It trades
at the same close only when required change exceeds 5% NAV, costs 4 bps per side, and a
short hedge pays its observed lending rate or the 2% floor. Missing prints carry the
hedge. It is outside K, the equity name cap, and equity gross; gross with and without
the hedge and ex-ante and realized post-hedge beta are reported. Every control must
have absolute realized post-hedge beta at most 0.30 in at least two folds.

## Decisions and reduced rounds

Candidates are ranked first by pooled nonlinear-neutral target IC. A candidate whose
constructed-headline net excess is negative with its entire 95% interval below zero is
ineligible. The eligible IC leader is designated unless another candidate has a paired
constructed-economics interval wholly above zero versus it and that candidate's paired
IC interval versus it includes zero. Exact ties use the registered order only for
determinism.

Acceptance is CPU-only and contains exactly the five controls on F1/F2/F3 plus one
F1 `b_intraday` GBDT: sixteen books. It enforces the existing D1--D5, 1.5--2.25 equity
gross, stale/unresolved, source-tier, native-fast, and legacy-scaled-target identity
gates, plus the inverse-volatility, volatility-occupancy, BOVA identity/calendar, and
post-hedge-beta gates above. Stop at the first failure.

Round 1 is CPU-only: the five controls and only the `b_intraday` GBDT on all three
folds. The prior ladder and data-span preview are not rerun. Round 2 is not authorized
until Gabriel gives an explicit go after reviewing Round 1. When authorized, it runs
the existing disposable one-epoch Arm-A/F1/seed-11 smoke, Stage P for seeds 11/29/47,
then only Arms A and B across F1/F2/F3 and those seeds. Arm C is dropped. The parent
comparison uses Arm B, the Round-1 GBDT, and their fixed equal-rank ensemble. There is
no deployment change.

Every run uses a fresh commit-bound immutable root, exact source paths and hashes,
false official-validation/test access flags, operational logs, and a complete hash
inventory. Never retry a completed score. A score-free operational repair requires a
new clean commit and fresh root and must not change this contract.

## Machine-readable protocol

<!-- BRAZIL_RV_V2_PROTOCOL_JSON_BEGIN -->
```json
{
  "borrow_cells": [
    "uniform",
    "lending_sidecar_v1"
  ],
  "bova11": {
    "bdi_code": "14",
    "cost_bps_per_side": 4.0,
    "isin": "BRBOVACTF003",
    "market_type": 10,
    "rebalance_threshold_fraction_nav": 0.05,
    "security_spec": "CI",
    "short_borrow_floor": 0.02,
    "supplied_bdi_02_corrected_from_raw_cotahist": true,
    "ticker": "BOVA11"
  },
  "candidate_decision_rule": {
    "economics_override": "paired_constructed_economics_interval_above_zero_and_paired_ic_interval_includes_zero",
    "ineligible": "constructed_headline_net_excess_negative_with_95_interval_below_zero",
    "primary": "pooled_primary_neutral_target_ic"
  },
  "comparators": [
    "comparator_lending_unconstructed",
    "comparator_uniform_borrow"
  ],
  "cross_fit": "none",
  "evaluation_window_per_fold": {
    "F1": {"end": "2023-12-29", "start": "2023-07-03"},
    "F2": {"end": "2024-06-28", "start": "2024-01-02"},
    "F3": {"end": "2024-12-30", "start": "2024-07-01"}
  },
  "headline_cell": {
    "ledger": {
      "annual_borrow_rate": 0.02,
      "annual_debit_spread": 0.0,
      "annual_sessions": 252,
      "beta_hedge": true,
      "borrow_source": "lending_sidecar_v1",
      "buffer_per_side": 30,
      "cost_bps_per_side": 4.0,
      "entry_expiry_sessions": 3,
      "gross_target": 2.0,
      "hedge_annual_borrow_rate": 0.02,
      "hedge_cost_bps_per_side": 4.0,
      "hedge_rebalance_threshold_nav": 0.05,
      "ineligible_hold_sessions": 5,
      "initial_capital_brl": 1.0,
      "k_per_side": 30,
      "lot_size": null,
      "planned_absolute_net_cap": 0.2,
      "planned_gross_cap": 2.25,
      "planned_name_weight_cap": 0.05,
      "settlement_economics_unresolved_fraction_nav": 0.15,
      "settlement_grace_sessions": 10,
      "settlement_haircut": 0.3,
      "short_proceeds_remuneration": 0.0,
      "volatility_balanced_entries": true,
      "volatility_group_count": 5
    },
    "signal": "tie_aware_rank_average_D1_D2_D3_D5",
    "signal_horizons_sessions": [1, 2, 3, 5]
  },
  "headline_cell_name": "cost_4_borrow_lending_v1",
  "inverse_volatility_neutral_ic_absolute_bound": 0.02,
  "models_per_fold_seed": 1,
  "primary_population_rule": {
    "daily_aggregation": "equal_mean_of_all_primary_horizons_when_all_defined",
    "horizons_sessions": [1, 2, 3, 5],
    "minimum_cross_section_names": 20,
    "per_horizon_metric": "tie_aware_spearman",
    "requirements": [
      "active_at_entry",
      "finite_target_scale_sigma_greater_than_1e-8",
      "valid_and_finite_neutral_target_on_every_primary_horizon",
      "valid_and_finite_neutralization_characteristics",
      "valid_and_finite_score_on_every_primary_horizon"
    ],
    "target": "target_primary_neutral"
  },
  "purge_sessions": {"fit_to_selection": 10, "selection_to_evaluation": 10},
  "realized_beta_label_threshold": 0.3,
  "round1": {
    "controls": [
      "reversal_5",
      "reversal_21",
      "momentum_12_1",
      "reversal_5_momentum_12_1_blend",
      "inverse_volatility_20"
    ],
    "cpu_only": true,
    "data_span_preview": false,
    "gbdt_rungs": ["b_intraday"]
  },
  "round2": {
    "arms": ["A_fine_only", "B_pretrain_finetune"],
    "dropped_arm": "C_joint_decay_756",
    "parent_comparison_network_arm": "B_pretrain_finetune",
    "requires_explicit_go_after_round1": true
  },
  "schema": "BRAZIL_RV_V2_REGISTRATION_PROTOCOL_V2",
  "selection_sessions": 55,
  "source_tier_labels": {
    "action_terms_source": "inferred_cotahist_dismes_v1",
    "schedule_source": "reconstructed_v1"
  },
  "target_neutralization": {
    "clip": 5.0,
    "fallback_below_40_names": "rev3_intercept_plus_three_linear_risks",
    "intercept": "absorbed_by_full_dummy_blocks",
    "method": "ols_10_vol_dummies_5_beta_dummies_linear_rank_gauss_log_adv",
    "minimum_names": 20,
    "nonlinear_minimum_names": 40,
    "target_validity_array": "target_primary_neutral_valid",
    "target_value_array": "target_primary_neutral"
  }
}
```
<!-- BRAZIL_RV_V2_PROTOCOL_JSON_END -->
