# Rev 4f amendment — registered before replay

The following rev-4e registration is retained verbatim as the base contract.
The amendment after it overrides its ledger provisions for new rev-4f roots.
The base JSON describes the historical rev-4e protocol, not executable rev-4f settings.
No rev-4f replay has been started.

# Brazil-RV v2 Round 1 / Round 2 registration rev 4e

Status: frozen before any rev-4e ledger replay or Round-2 score. Rev 4e changes
only two ledger conventions: the headline remunerates equity and BOVA11 short-
proceeds collateral at CDI, and the equity borrower registration fee is 20% of
the contract rate with a 2.5-bps annual floor and 70-bps annual cap. The
otherwise identical sterile-proceeds cell remains a labelled comparator.

All score panels in the Round-1 replay are hash-bound artifacts from the sealed
rev-4d root. No model is fitted and no score is recomputed. Every non-ledger
evaluation field must remain bit-identical. `b_intraday` remains the only GBDT
rung and parent.

Round 2 is authorized on exactly one GH200. It runs the disposable one-epoch
Arm-A/F1/seed-11 compile smoke without a score directory, Stage P for seeds
11/29/47, Arms A and B on F1/F2/F3, and the fixed Arm-B/GBDT/equal-weight-rank
ensemble comparison. Round 3, every 2025 row, store rebuilds, and deployment
changes remain forbidden.

## Machine-readable protocol

<!-- BRAZIL_RV_V2_PROTOCOL_JSON_BEGIN -->
```json
{"acceptance_legacy_identity":{"inventory_sha256":"b5084c817fc478c2759d962af12e282c292cade6f8e715aebed290ba5500ce08","result_sha256":"ca39f11340f13956dca11da7fb6b3a8fa0a38f8a79cb558387d81aff26bf57a0","root":"D:\\quant-data\\b3\\processed\\model_runs\\v2_round1_81fe0cb_20260907T023339Z"},"borrow_availability":{"borrow_balance":"positive_published_open_balance_or_lending_trade_observed_in_prior_60_sessions","borrow_open":"all_names_when_a_causal_cross_sectional_rate_exists","borrow_strict":"lending_trade_observed_in_prior_20_sessions","pre_first_rate_borrow":"placeholder_0.02"},"borrow_cells":["borrow_strict","borrow_balance","borrow_open","uniform"],"borrow_rate":{"equity_rate_floor":null,"hedge_short_rate_floor":0.02,"missing_rate_imputation":"same_day_cross_sectional_observed_rate_75th_percentile","observed_rate_lookback_sessions":60,"registration_fee":{"annual_cap":0.007,"annual_floor":0.00025,"fraction_of_contract_rate":0.2}},"bova11":{"bdi_code":"14","cost_bps_per_side":4.0,"hedge_notional_cap_nav":0.6,"isin":"BRBOVACTF003","market_type":10,"rebalance_threshold_fraction_nav":0.05,"security_spec":"CI","short_borrow_floor":0.02,"supplied_bdi_02_corrected_from_raw_cotahist":true,"ticker":"BOVA11"},"candidate_decision_rule":{"economics_override":"paired_constructed_economics_interval_above_zero_and_paired_ic_interval_includes_zero","ineligible":"constructed_headline_net_excess_negative_with_95_interval_below_zero","primary":"pooled_primary_neutral_target_ic"},"comparators":["borrow_strict","borrow_open","comparator_sterile_proceeds","comparator_uniform_borrow"],"construction":{"caps_scope":"equity_only","fill_order":"within_quintile_then_global_band_spill","occupancy_mean_absolute_deviation_limit_slots":2.0,"quota_remainder_order":[3,2,4,1,5],"rank_within_stratum":true,"retention_uses_current_quintile":true,"small_stratum_scaling_threshold_multiple":2,"volatility_strata":"five_equal_count_yang_zhang_vol_20_quintiles"},"cross_fit":"none","evaluation_window_per_fold":{"F1":{"end":"2023-12-29","start":"2023-07-03"},"F2":{"end":"2024-06-28","start":"2024-01-02"},"F3":{"end":"2024-12-30","start":"2024-07-01"}},"headline_cell":{"ledger":{"annual_borrow_rate":0.02,"annual_debit_spread":0.0,"annual_sessions":252,"beta_hedge":true,"borrow_registration_fee_cap":0.007,"borrow_registration_fee_floor":0.00025,"borrow_registration_fee_fraction":0.2,"borrow_source":"borrow_balance","buffer_per_side":30,"cost_bps_per_side":4.0,"entry_expiry_sessions":3,"gross_target":2.0,"hedge_annual_borrow_rate":0.02,"hedge_cost_bps_per_side":4.0,"hedge_notional_cap_nav":0.6,"hedge_rebalance_threshold_nav":0.05,"ineligible_hold_sessions":5,"initial_capital_brl":1.0,"k_per_side":30,"lot_size":null,"planned_absolute_net_cap":0.2,"planned_gross_cap":2.25,"planned_name_weight_cap":0.05,"settlement_economics_unresolved_fraction_nav":0.15,"settlement_grace_sessions":10,"settlement_haircut":0.3,"short_proceeds_remuneration":1.0,"small_stratum_scaling_threshold_multiple":2,"volatility_balanced_entries":true,"volatility_group_count":5},"signal":"tie_aware_rank_average_D1_D2_D3_D5","signal_horizons_sessions":[1,2,3,5]},"headline_cell_name":"borrow_balance","inverse_volatility_neutral_ic_absolute_bound":0.02,"lending_archive":{"availability_lag_sessions":1,"last_source_session":"2024-12-30","source_label":"lending_archive_v2_2009_202412","store_lending_feature_rebuilt":false},"models_per_fold_seed":1,"primary_population_rule":{"daily_aggregation":"equal_mean_of_all_primary_horizons_when_all_defined","horizons_sessions":[1,2,3,5],"minimum_cross_section_names":20,"per_horizon_metric":"tie_aware_spearman","requirements":["active_at_entry","finite_target_scale_sigma_greater_than_1e-8","valid_and_finite_neutral_target_on_every_primary_horizon","valid_and_finite_neutralization_characteristics","valid_and_finite_score_on_every_primary_horizon"],"target":"target_primary_neutral"},"purge_sessions":{"fit_to_selection":10,"selection_to_evaluation":10},"realized_beta_label_threshold":0.3,"round1":{"controls":["reversal_5","reversal_21","momentum_12_1","reversal_5_momentum_12_1_blend","inverse_volatility_20"],"cpu_only":true,"data_span_preview":false,"gbdt_rungs":["b_intraday"]},"round2":{"arms":["A_fine_only","B_pretrain_finetune"],"dropped_arm":"C_joint_decay_756","parent_comparison_network_arm":"B_pretrain_finetune","requires_explicit_go_after_round1":true},"schema":"BRAZIL_RV_V2_REGISTRATION_PROTOCOL_V4E","selection_sessions":55,"source_tier_labels":{"action_terms_source":"inferred_cotahist_dismes_v1","schedule_source":"reconstructed_v1"},"target_neutralization":{"clip":5.0,"fallback_below_40_names":"rev3_intercept_plus_three_linear_risks","intercept":"absorbed_by_full_dummy_blocks","method":"ols_10_vol_dummies_5_beta_dummies_linear_rank_gauss_log_adv","minimum_names":20,"nonlinear_minimum_names":40,"target_validity_array":"target_primary_neutral_valid","target_value_array":"target_primary_neutral"}}
```
<!-- BRAZIL_RV_V2_PROTOCOL_JSON_END -->

## Rev-4f ledger amendment

### 1.1 Economic beta against the hedge instrument

A hash-bound derived sidecar, not a store rebuild: for each name and session t, the OLS
slope of the name's shareholder-wealth daily return on the BOVA11 close-to-close return
over the 60 completed sessions ending t−1, requiring ≥ 40 valid pairs, Blume-adjusted
`β_hedge = 0.67·β̂ + 0.33`, clipped to [−1, 3], with a validity mask and the BOVA11
series' manifest hash bound into it. The ledger takes `hedge_beta` and `hedge_beta_valid`
in place of the transformed `beta_60`; a held or pending name without a valid hedge beta
uses the last valid value up to 20 sessions old, else 1.0, and the session is counted
under `hedge_beta_fallback_sessions`. The realized-beta diagnostic gains a second slope
against BOVA11's own return (the benchmark the hedge trades); the equal-weight-universe
slope stays as it is. `slow_valid` is respected wherever a slow feature is lifted for a
diagnostic (the exposure summary too).

Tests: two names with true betas 1.0 and 1.2 and notionals ∓1 give a hedge target of
−0.20 NAV with the raw beta and −0.194 with the Blume-adjusted one; the transformed
`beta_60` is no longer accepted by the ledger (a type/name guard); the sidecar refuses
2025/2026 dates; hedge-beta distribution by year is reported (expect mass on 0.5–1.5).

### 1.2 Hedge decided at 15:45, executed at the close

Move the hedge sizing into the decision block: `planned_equity` from `shares × marks_{t−1}`
plus pending and today's planned entries at their reference prices, minus today's
planned exits; hedge target = −β_net × NAV_{t−1}; the hedge order is an `IntendedOrder`
with purpose `hedge`, decision session t, reference price BOVA11 close_{t−1}; it fills at
BOVA11 close_t when BOVA11 prints, with the 4 bps cost, and appears in the fills,
cancellations and the intended-versus-realized reconciliation. Fill imbalance on the
equity side is corrected by the next session's hedge decision. The 5% NAV rebalance
threshold and the 0.6 cap apply to the decision-time target.

Test (same-day future mutation): hold everything fixed and change only a later equity
close; every intended order including the hedge must be bit-identical; only fills and
marks may change.

### 1.3 Decision-known actions for decisions, retrospective actions for accounting

Split the action path. **Decision path** (order construction, reference prices, entry
eligibility, cap checks): uses the decision-known alignment — an inferred action dated
session t is unknown at 15:45 of t; the name is `unresolved_action` for new entries on t
(already the 4e state), pending entries for it are cancelled, exits of held positions
remain allowed, reference prices are not converted. **Accounting path** (shares, marks,
claims, settlement): the retrospective conversion for held positions is booked after the
session's fills, exactly as the terms are realized, never before. Report the count of
sessions on which the two alignments differ for a held or pending name.

Test (joined builder → evaluator → ledger): change only the event-day close so that the
inferred cash term changes; every intended order dated that day must be bit-identical;
the held position's converted shares/marks after the close may change. Keep the existing
4e tests (uncertainty blocks entries, exits fill on any print, terminal liquidation).

### 1.4 Pending entries follow the signal

At each decision, cancel a pending (unfilled) entry whose name is no longer inside its
side's retention band in its current quintile (`entries_to_cancel` at line 1856 only
covers held names). Reason `band_exit`. Test: a name enters the long band, its order is
pending, the next session it ranks in the middle of the quintile; the order is cancelled
and no fill occurs. Bit-identical with the switch off.

### 1.5 Borrow accrual and the cost grid

Daily borrow charge = `(1 + rate)^(1/252) − 1` × short value (B3's compounded 252-day
convention), for equity rates, fees and the hedge placeholder (lines 1660–1686).
Replace the `cost_{2,4,7}_borrow_{…}` grid, which is built on the pre-rev-4 strategy,
with a headline-preserving grid: cost per side ∈ {2, 4, 7} bps × borrow cell ∈ {balance,
strict, open} on the constructed, hedged book; keep one labelled `legacy_strategy_
comparator` if it is still useful for continuity, else drop it.


### Arithmetic correction and unresolved interpretation

The specified Blume formula gives -0.134 NAV, not -0.194, for a short
notional of 1 at beta 1.0 and a long notional of 1 at beta 1.2. The formula governs.

The retrospective inferred event flag itself can depend on the later close.
The requested decision-known action path must pass event-classification as well as
cash-term mutation tests. No replay proceeds until the decision-time uncertainty
contract is resolved and the joined test passes. Existing verified-action accounting
fixtures remain required. No later-close action terms may resize an intended order.

Masking diagnostic slow features changes exposure diagnostics and their input hashes.
That requested correction conflicts with literal equality of every non-ledger field.
Any replay must expose this conflict and stop rather than silently exempt changed fields.

All expectations in section 1.6 of the pass-5 response are reported expectations,
not tuning targets. 2025/2026 data and paid GPU work remain out of scope for this stage.

