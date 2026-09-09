# Round 1' on the repaired store

The five controls and the paired `a_slow`/`b_intraday` GBDTs completed on F1-F3. All 21 panels pass the registered D1-D5 and headline book bounds. The parent remains `b_intraday` as registered; this comparison does not reopen the rung ladder.

The repaired intraday inputs do not demonstrate a GBDT improvement here. The paired neutral-IC difference is -0.004836 [-0.009128, 0.001347], and net excess differs by -0.874 [-5.439, 7.458] bps/day. Both intervals include zero. `b_intraday` has lower IC point estimates in F1 and F3 and a small increase in F2; its economic differences vary strongly by fold. Improved data validity is not evidence of improved predictive performance.

This comparison adds intraday scalars and `fast_present` to the slow GBDT. It does not evaluate the native TCN, which is tested separately by the registered Round-3 neural ablation. The five-seed GBDTs use all five heads, up to 3,000 boosting rounds and 100-round early stopping, with no setting changed after these results.

The book uses the R3.1 policy selected in sample: theta 1, D3/D5/D10, equal notional and buffer 9 per quintile. Intervals use the registered 10,000-replication, 20-session bootstrap preserving fold boundaries. Net excess is bps/day against all-cash CDI.

| Candidate | Neutral IC [95%] | Net excess [95%] | Persistence 1 | Persistence 5 | Shareholder spread / holding session |
| --- | --- | --- | --- | --- | --- |
| inverse_volatility_20 | 0.004378 [0.001539, 0.007621] | -2.573 [-6.921, 3.897] | 0.994 | 0.965 | 14.306 |
| momentum_12_1 | 0.021473 [0.010222, 0.034446] | 3.190 [-1.374, 12.316] | 0.994 | 0.971 | 20.082 |
| reversal_21 | -0.010957 [-0.025548, -0.006817] | -5.330 [-15.143, -2.345] | 0.938 | 0.731 | -2.491 |
| reversal_5 | -0.003795 [-0.015977, 0.003713] | -4.951 [-10.879, -0.387] | 0.751 | 0.011 | -3.299 |
| reversal_5_momentum_12_1_blend | 0.013463 [0.000585, 0.022574] | 1.448 [-4.792, 9.753] | 0.858 | 0.435 | 5.249 |
| a_slow | 0.015301 [0.007902, 0.023404] | -0.048 [-5.429, 5.733] | 0.763 | 0.589 | 7.860 |
| b_intraday | 0.010465 [0.002507, 0.020830] | -0.922 [-4.158, 6.068] | 0.759 | 0.597 | 5.388 |

The paired intraday-scalar contribution is measured on each statistic's common supported population:

| Statistic | b_intraday minus a_slow [95%] |
| --- | --- |
| headline_net_excess_bps | -0.874 [-5.439, 7.458] |
| persistence_1 | -0.004 [-0.009, -0.001] |
| persistence_5 | 0.008 [-0.002, 0.015] |
| price_return_rank_ic | -0.007200 [-0.011619, -0.000751] |
| primary_neutral_target_ic | -0.004836 [-0.009128, 0.001347] |
| shareholder_rank_ic | -0.008137 [-0.012564, -0.002060] |
| shareholder_return_spread_bps_per_holding_session | -2.471 [-6.468, 1.945] |

| Fold | a_slow IC | b_intraday IC | Paired IC [95%] | Paired net [95%] |
| --- | --- | --- | --- | --- |
| F1 | 0.009788 | 0.001824 | -0.007964 [-0.015813, 0.001717] | -7.024 [-14.768, 3.212] |
| F2 | 0.007656 | 0.008646 | 0.000991 [-0.007025, 0.008184] | 0.058 [-7.485, 4.609] |
| F3 | 0.028136 | 0.020667 | -0.007470 [-0.015983, 0.005231] | 4.221 [-5.237, 26.649] |

[The evidence JSON](v2_round1_prime_evidence.json) includes every scenario summary, fold economics decomposition, realized BOVA11 beta, traded-signal metrics, and artifact hashes. Unresolved economics remain explicitly labelled and excluded according to the frozen support rules.

Root: `D:\quant-data\b3\processed\model_runs\v2_round1_prime_615ae41_20260909T105303Z`

| Artifact | SHA-256 |
| --- | --- |
| frozen_design.json | f3f056798e9e79ac57da90d2592669e57bb28549f8a34b1aa2978daadf22a259 |
| round1_result.json | fb4a6ae1b82e31fbe777757a1e33fd84c8748b07bcc9bf23145bf25c34f871fa |
| access_audit.json | a26114dc816572d63d19715631f6aa3ffcb709dda9901c696a7979bd1eda27f4 |
| artifact_inventory.json | c97646eafce7abb78c63826e007ffb4a78c2cdc15954a42bfdedb37b615e2a9a |

These are development-fold research results under reconstructed calendar and inferred corporate-action terms. This round reads only the store ending in 2024. Historical integrity-only scans of later old-store rows are disclosed in [the reader correction](v2_store_comparison_stop_evidence.json); no held-out performance evaluation was run. Deployment is unchanged.
