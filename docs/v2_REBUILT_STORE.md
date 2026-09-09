# Repaired development store and input audits

The new store covers 3,717 sessions through 2024-12-30 and 933 ISINs. Its measured full-build peak is 6.7059 GiB, below the unchanged 8-GiB limit. This report covers the input audits; the separate 16-book acceptance and Round 1' results follow.

The rebuild includes repaired intraday scalars and their masks, ages and support; M1-internal to-close endpoints; lending archive v2; and oddlot re-derived through the final development decision. Sparse archive gaps remain unknown. The return-consistency tolerance remains 0.005.

Root: `D:\quant-data\b3\processed\v2_daily_store_3d67624_20260909T100323Z`

Manifest SHA-256: `db4f751d47133a6611733739ad9bfc15721452218365e338fec61a6b608bea64`

All arrays outside the registered changes match the authorized canonical slice exactly. Target comparisons censor unavailable horizon endpoints before decoding; the corrected comparison reader decodes only granted payload rows. Historical integrity-only scans of later rows are disclosed in [the reader correction](v2_store_comparison_stop_evidence.json). The complete per-array comparison is retained in the evidence JSON.

| Array | Registered change | Comparison | Different cells |
| --- | --- | --- | --- |
| action_cash_per_prior_share | False | exact | 0 |
| action_has_action | False | exact | 0 |
| action_payment_session | False | exact | 0 |
| action_session_resolved | False | exact | 0 |
| action_shares_per_prior_share | False | exact | 0 |
| action_successor_index | False | exact | 0 |
| active | False | exact | 0 |
| activity_valid | False | exact | 0 |
| ambiguous_action_mask | False | exact | 0 |
| audit_eventual_survives_to_final_year | True | different | 397719 |
| common_state_diagnostic_valid | False | exact | 0 |
| common_state_diagnostic_values | False | exact | 0 |
| detected_cash_event_mask | False | exact | 0 |
| detected_event_mask | False | exact | 0 |
| detected_split_mask | False | exact | 0 |
| distribution_change_mask | False | exact | 0 |
| distribution_number | False | exact | 0 |
| fast_last_price_age_minutes | False | exact | 0 |
| fast_last_price_age_valid | False | exact | 0 |
| fast_patch_mask | False | exact | 0 |
| fast_patch_valid | False | exact | 0 |
| fast_patch_values | False | exact | 0 |
| fast_present | False | exact | 0 |
| inferred_action_c1_mask | False | exact | 0 |
| inferred_action_large_move_no_action_mask | False | exact | 0 |
| inferred_action_u1_mask | False | exact | 0 |
| inferred_action_u2_mask | False | exact | 0 |
| intraday_age_sessions | True | different | 995285 |
| intraday_boundary_lagged_mask | True | exact | 0 |
| intraday_boundary_sameday_mask | True | exact | 0 |
| intraday_support_fraction | True | added |  |
| intraday_unit_or_unresolved_boundary_mask | True | different | 150044 |
| intraday_valid | True | different | 479006 |
| intraday_values | True | different | 748410 |
| m1_cotahist_close_consistent_mask | True | removed |  |
| m1_cotahist_return_consistent_mask | True | added |  |
| observed | False | exact | 0 |
| price_jump_anomaly_mask | False | exact | 0 |
| prior_reference_close | False | exact | 0 |
| quantity | False | exact | 0 |
| raw_close | False | exact | 0 |
| raw_high | False | exact | 0 |
| raw_low | False | exact | 0 |
| raw_open | False | exact | 0 |
| shareholder_wealth_close | False | exact | 0 |
| shareholder_wealth_high | False | exact | 0 |
| shareholder_wealth_low | False | exact | 0 |
| shareholder_wealth_open | False | exact | 0 |
| shareholder_wealth_valid | False | exact | 0 |
| sidecar_lending_age_sessions | True | shape_or_dtype_changed |  |
| sidecar_lending_valid | True | shape_or_dtype_changed |  |
| sidecar_lending_values | True | shape_or_dtype_changed |  |
| sidecar_oddlot_age_sessions | True | different | 840078 |
| sidecar_oddlot_valid | True | different | 841218 |
| sidecar_oddlot_values | True | different | 839848 |
| slow_age_sessions | False | exact | 0 |
| slow_timestep_valid | False | exact | 0 |
| slow_valid | False | exact | 0 |
| slow_values | False | exact | 0 |
| source_session_complete | False | exact | 0 |
| target_normalized_cross_section_valid | False | exact | 0 |
| target_normalized_residual | False | exact | 0 |
| target_price_midrank | False | exact | 0 |
| target_price_simple_return | False | exact | 0 |
| target_price_valid | False | exact | 0 |
| target_primary | False | exact | 0 |
| target_scale_sigma | False | exact | 0 |
| target_shareholder_midrank | False | exact | 0 |
| target_shareholder_simple_return | False | exact | 0 |
| target_shareholder_valid | False | exact | 0 |
| target_terminal_loss | False | exact | 0 |
| target_terminal_wealth | False | exact | 0 |
| target_to_close | True | different | 89199 |
| target_to_close_normalized_residual | True | different | 88615 |
| target_to_close_raw_log_return | True | different | 83960 |
| target_to_close_valid | True | different | 84843 |
| target_valid | False | exact | 0 |
| trade_count | False | exact | 0 |
| trade_observed | False | exact | 0 |
| volume_brl | False | exact | 0 |

The independent native-fast audit re-derives seven channels for 20 names over 20 sessions from the raw M1 archive. Feature, patch and age masks match exactly; value errors remain within its frozen tolerance. All publication-lag masks reproduce independently, daily-archive D+1 violations are zero, and all binding survivorship strata pass.

The table below uses the full causal active universe for named features and sessions for global diagnostics. Native-fast coverage means an active name-day with any valid patch for a channel; it is distinct from the archive-only table restricted to the 158 configured M1 names.

| Family | Features | Unit | 2024 mean feature coverage | Any feature present |
| --- | --- | --- | --- | --- |
| common_state_diagnostic | 3 | session_with_valid_diagnostic | 100.00% | 100.00% |
| intraday | 20 | active_name_day_with_valid_feature | 41.32% | 69.52% |
| native_fast | 7 | active_name_day_with_any_valid_patch | 69.51% | 70.71% |
| sidecar_lending | 5 | active_name_day_with_valid_feature | 63.59% | 70.44% |
| sidecar_oddlot | 2 | active_name_day_with_valid_feature | 99.90% | 99.90% |
| slow | 32 | active_name_day_with_valid_feature | 98.32% | 100.00% |

[Every family and feature by month](v2_monthly_feature_coverage.json) and [the complete input evidence](v2_rebuilt_store_evidence.json) are available for review. Provider invariance and same-decision causality are covered by the joined builder fixture; production input/audit arrays remain hash-bound.

The prior memory stop, voluntarily stopped slow lookup build, and Windows Update interruption remain preserved as separate evidence. Their partial arrays were not reused. The canonical raw sources and prior sealed store remain unchanged.
