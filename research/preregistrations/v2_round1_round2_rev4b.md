# Brazil-RV v2 Round 1 / Round 2 registration rev 4b

Status: frozen before any rev-4b score. Rev 4 is void for future research
decisions, while all of its artifacts remain immutable engineering evidence.
All folds, purges, clocks, targets, controls, caps, settlement rules, beta hedge,
protected-window rules, and decision rules remain as registered in rev 4 except
for the construction and borrow contracts stated here. No official-validation
or permanently spent test row may be read.

## Construction

On each decision session, eligible names are assigned to five equal-count
`yang_zhang_vol_20` quintiles. Scores are ranked independently inside each
quintile. Each side's K and retention buffer are divided across quintiles, with
remainders assigned Q3, Q2, Q4, Q1, Q5. If a quintile has fewer than four times
its combined quota and buffer, both widths are scaled down proportionally and
the scaled quota sum is that session's effective side size. Long and short bands
are the corresponding top and bottom names within each quintile and must be
disjoint.

Retention uses the held name's current quintile. Entries first fill deficits
within each quintile and then fill any remaining side deficit from the best
remaining side-band names across quintiles. Spilled entries are counted by
source quintile. The whole-book planned gross and net caps include BOVA11 hedge
exposure; the acceptance hard gross interval continues to describe the equity
book excluding the hedge. Per side, the session-mean absolute deviation of
quintile occupancy from quota must not exceed two slots.

## Borrow and lending archive

The direct, hash-bound `lending_archive_v2_2009_202412` extends the existing
sealed observations with the 127 sealed B3 chapter-05 PDFs from 2024-07-01
through 2024-12-30. Its old overlap must be identical and its rows are exposed
only on the next B3 session. The store lending feature is not rebuilt and no
registered candidate consumes it.

Every equity short pays the last observed name-level annual taker rate plus
0.25% annual registration fee, without a floor. When a name has no observed rate
in the prior 60 sessions, it receives that decision day's cross-sectional 75th
percentile of recent observed rates and is labelled imputed. The imputed share
of held short notional is reported. A short BOVA11 hedge continues to pay 2%
because the lending archive has no BOVA11 observation.

`borrow_balance` is the headline and permits a positive published open balance
or a trade observed in the prior 60 sessions. `borrow_open` permits every name
once a causal same-day cross-sectional rate exists. Before the first such rate,
it remains unavailable rather than future-backfilling. `borrow_strict` is the
trade-only prior-20-session comparator. This resolves the prose/table conflict
in the supplied rev-4b note in favor of its explicit required test: a t-1
balance without a trade is balance-shortable and not strict-shortable.
`comparator_uniform_borrow` remains the un-screened flat-2% continuity cell.

The F1 window begins before the first parseable rate table (trade date
2023-07-10, available 2023-07-11); the preceding sessions therefore remain
honestly unavailable in `borrow_open`. No 2025/2026 lending row is admissible.

## Acceptance and rounds

Acceptance is CPU-only and evaluates exactly the five registered controls on
F1/F2/F3 plus one F1 `b_intraday` GBDT: sixteen books. It retains the prior
sanity, D1-D5, target-null, source, native-fast, BOVA11, realized-beta, and
equity-gross gates, adds the construction occupancy gate above, and compares
legacy identity only to the fixed, hash-bound rev-3 Round-1 root recorded in the
protocol below.

If acceptance passes, Round 1 is CPU-only: the five controls and only the
`b_intraday` GBDT on all three folds. It reports all borrow cells, beta,
quintile occupancy, spill counts, imputed-rate share, and lending coverage. Stop
after the sealed Round-1 readout. Round 2 is not authorized until Gabriel gives
an explicit later go. It retains only Arms A and B and the existing Stage-P and
R2.2 contracts; there is no deployment change.

Every run uses a fresh clean commit-bound immutable root, hash-bound sources,
false official-validation/test access flags, preserved operational logs, and a
complete inventory. Never retry a completed score. A score-free operational
repair requires a new clean commit and fresh root without changing this frozen
research contract.

## Machine-readable protocol

<!-- BRAZIL_RV_V2_PROTOCOL_JSON_BEGIN -->
```json
{
  "acceptance_legacy_identity": {
    "inventory_sha256": "b5084c817fc478c2759d962af12e282c292cade6f8e715aebed290ba5500ce08",
    "result_sha256": "ca39f11340f13956dca11da7fb6b3a8fa0a38f8a79cb558387d81aff26bf57a0",
    "root": "D:\\quant-data\\b3\\processed\\model_runs\\v2_round1_81fe0cb_20260907T023339Z"
  },
  "borrow_availability": {
    "borrow_balance": "positive_published_open_balance_or_lending_trade_observed_in_prior_60_sessions",
    "borrow_open": "all_names_when_a_causal_cross_sectional_rate_exists",
    "borrow_strict": "lending_trade_observed_in_prior_20_sessions",
    "pre_first_causal_rate": "unavailable_without_future_backfill"
  },
  "borrow_cells": [
    "borrow_strict",
    "borrow_balance",
    "borrow_open",
    "uniform"
  ],
  "borrow_rate": {
    "equity_rate_floor": null,
    "hedge_short_rate_floor": 0.02,
    "missing_rate_imputation": "same_day_cross_sectional_observed_rate_75th_percentile",
    "observed_rate_lookback_sessions": 60,
    "registration_fee_annual": 0.0025
  },
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
    "borrow_strict",
    "borrow_open",
    "comparator_uniform_borrow"
  ],
  "construction": {
    "fill_order": "within_quintile_then_global_band_spill",
    "occupancy_mean_absolute_deviation_limit_slots": 2.0,
    "quota_remainder_order": [3, 2, 4, 1, 5],
    "rank_within_stratum": true,
    "retention_uses_current_quintile": true,
    "small_stratum_scaling_threshold_multiple": 4,
    "volatility_strata": "five_equal_count_yang_zhang_vol_20_quintiles"
  },
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
      "borrow_registration_fee": 0.0025,
      "borrow_source": "borrow_balance",
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
  "headline_cell_name": "borrow_balance",
  "inverse_volatility_neutral_ic_absolute_bound": 0.02,
  "lending_archive": {
    "availability_lag_sessions": 1,
    "last_source_session": "2024-12-30",
    "source_label": "lending_archive_v2_2009_202412",
    "store_lending_feature_rebuilt": false
  },
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
  "schema": "BRAZIL_RV_V2_REGISTRATION_PROTOCOL_V3",
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
