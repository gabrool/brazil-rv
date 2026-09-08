# Pass-5 rev4f: completed Round 1 and stopped Round 2

Round 1 completed under `a25ae450b37d9e3b669197d488d97de5c27692f8`: 18 sealed panels, no score or model recomputation, all protected evaluation fields bit-identical. The approved six-field lending amendment is active. Round 2 stopped before its first new evaluation on one unregistered input hash; see [the stop evidence](v2_PASS5_ROUND2_STOP.md). No execution sweep, new store or fit ran.

## Pooled Round-1 economics

Intervals are the registered 10,000-replication, 20-session moving-block bootstrap preserving fold boundaries. Net excess is bps/day versus all-cash CDI. IC is unchanged. The existing `economics_unresolved` rule excludes the whole F3 fold for inverse volatility and the reversal/momentum blend: their pooled economics use 248 sessions, while the other four candidates use 375. Those two F3 labels remain unchanged; terminal-settlement incidence is 16.16% and 16.15% respectively, above the separate 15% incidence label.

| Candidate | Neutral IC | Rev4e net | Rev4f net | Rev4f 95% interval |
| --- | --- | --- | --- | --- |
| inverse_volatility_20 | 0.004378 | -5.130 | -5.327 | [-11.063, 0.485] |
| momentum_12_1 | 0.021473 | 1.665 | 3.219 | [-1.913, 12.443] |
| reversal_21 | -0.010957 | -4.290 | -5.852 | [-15.635, -2.986] |
| reversal_5 | -0.003795 | -6.410 | -5.048 | [-10.825, -0.621] |
| reversal_5_momentum_12_1_blend | 0.013463 | -0.969 | -1.054 | [-6.935, 6.898] |
| b_intraday | 0.019991 | -1.007 | 0.076 | [-3.604, 7.257] |

Momentum remains the strongest Round-1 control on headline economics; the GBDT moves from -1.007 to +0.076 bps/day. Both intervals include zero. The unchanged Round-1 rule retains b_intraday. Round-2 B−A, ensemble−GBDT, ensemble−B and the Round-2 designation cannot yet be assessed under rev4f.

## Fold-by-fold headline comparison

Arrows show rev4e → rev4f. Gross and turnover are NAV fractions. F3 net figures for inverse volatility and the reversal/momentum blend are descriptive ledger arithmetic with `economics_unresolved=true`, not supported economics estimates; they are excluded from the pooled results above. Their raw IC remains reportable.

| Candidate / fold | Neutral IC | Net bps/day | Mean gross | Daily turnover |
| --- | --- | --- | --- | --- |
| inverse_volatility_20 / F1 | -0.000044 | -3.676 → -6.859 | 1.971 → 1.979 | 0.952 → 0.908 |
| inverse_volatility_20 / F2 | 0.010931 | -6.584 → -3.795 | 1.963 → 1.961 | 1.020 → 0.958 |
| inverse_volatility_20 / F3 | 0.002300 | -15.182 → -14.435 | 1.999 → 1.999 | 0.972 → 0.921 |
| momentum_12_1 / F1 | 0.015184 | 2.536 → 1.592 | 1.914 → 1.903 | 0.139 → 0.112 |
| momentum_12_1 / F2 | 0.008565 | -2.665 → -1.284 | 1.869 → 1.873 | 0.132 → 0.113 |
| momentum_12_1 / F3 | 0.040198 | 5.043 → 9.204 | 1.986 → 1.974 | 0.129 → 0.111 |
| reversal_21 / F1 | -0.010994 | 1.186 → -1.752 | 1.962 → 1.960 | 0.479 → 0.437 |
| reversal_21 / F2 | -0.013154 | -9.688 → -9.500 | 1.981 → 1.979 | 0.471 → 0.436 |
| reversal_21 / F3 | -0.008777 | -4.364 → -6.292 | 1.978 → 1.981 | 0.446 → 0.412 |
| reversal_5 / F1 | 0.003615 | -0.632 → 1.245 | 1.951 → 1.942 | 1.187 → 1.116 |
| reversal_5 / F2 | -0.004120 | -5.871 → -5.998 | 1.972 → 1.962 | 1.171 → 1.086 |
| reversal_5 / F3 | -0.010707 | -12.578 → -10.265 | 1.979 → 1.978 | 1.164 → 1.090 |
| reversal_5_momentum_12_1_blend / F1 | 0.012568 | 1.317 → 0.756 | 1.982 → 1.978 | 0.724 → 0.664 |
| reversal_5_momentum_12_1_blend / F2 | 0.005390 | -3.255 → -2.864 | 1.965 → 1.972 | 0.712 → 0.634 |
| reversal_5_momentum_12_1_blend / F3 | 0.022212 | -4.043 → -0.085 | 1.992 → 1.987 | 0.759 → 0.707 |
| b_intraday / F1 | 0.010955 | 0.063 → 1.941 | 1.955 → 1.949 | 0.558 → 0.492 |
| b_intraday / F2 | 0.008978 | -0.951 → -0.911 | 1.938 → 1.945 | 0.679 → 0.617 |
| b_intraday / F3 | 0.039546 | -2.106 → -0.780 | 1.987 → 1.980 | 0.708 → 0.656 |

## Hedge and realized beta

Hedge size is mean absolute realized signed notional divided by prior NAV. Beta is OLS with an intercept against the stated daily benchmark. Rev4e BOVA11 slopes below are newly calculated for this report from immutable rev4e ledger daily returns and the same hash-bound BOVA11 series; they were not fields of the original evaluation.

| Candidate / fold | Mean abs hedge / NAV | Capped sessions | Beta vs BOVA11 | Beta vs universe |
| --- | --- | --- | --- | --- |
| inverse_volatility_20 / F1 | 0.430 → 0.175 | 22 → 0 | 0.161 → -0.099 | 0.029 → -0.159 |
| inverse_volatility_20 / F2 | 0.352 → 0.136 | 12 → 0 | 0.101 → -0.091 | 0.001 → -0.124 |
| inverse_volatility_20 / F3 | 0.397 → 0.205 | 12 → 0 | 0.128 → -0.073 | -0.023 → -0.144 |
| momentum_12_1 / F1 | 0.304 → 0.098 | 2 → 0 | 0.043 → -0.127 | 0.002 → -0.108 |
| momentum_12_1 / F2 | 0.316 → 0.083 | 0 → 0 | 0.293 → 0.045 | 0.168 → 0.021 |
| momentum_12_1 / F3 | 0.534 → 0.216 | 60 → 0 | 0.171 → -0.181 | -0.035 → -0.234 |
| reversal_21 / F1 | 0.416 → 0.179 | 33 → 3 | -0.070 → 0.161 | -0.078 → 0.089 |
| reversal_21 / F2 | 0.309 → 0.093 | 27 → 0 | -0.218 → 0.030 | -0.152 → 0.014 |
| reversal_21 / F3 | 0.362 → 0.144 | 32 → 0 | -0.228 → 0.055 | -0.092 → 0.078 |
| reversal_5 / F1 | 0.341 → 0.153 | 31 → 0 | -0.044 → 0.033 | -0.029 → 0.026 |
| reversal_5 / F2 | 0.276 → 0.109 | 10 → 0 | -0.229 → -0.095 | -0.104 → -0.021 |
| reversal_5 / F3 | 0.320 → 0.175 | 27 → 0 | -0.039 → 0.062 | -0.022 → 0.023 |
| reversal_5_momentum_12_1_blend / F1 | 0.329 → 0.195 | 26 → 0 | 0.028 → -0.001 | -0.033 → -0.052 |
| reversal_5_momentum_12_1_blend / F2 | 0.210 → 0.137 | 1 → 0 | 0.070 → 0.045 | 0.028 → 0.022 |
| reversal_5_momentum_12_1_blend / F3 | 0.405 → 0.262 | 41 → 4 | -0.033 → -0.137 | -0.133 → -0.194 |
| b_intraday / F1 | 0.153 → 0.074 | 0 → 0 | -0.037 → -0.048 | -0.037 → -0.044 |
| b_intraday / F2 | 0.134 → 0.072 | 0 → 0 | -0.177 → -0.095 | -0.084 → -0.038 |
| b_intraday / F3 | 0.138 → 0.079 | 0 → 0 | -0.263 → -0.214 | -0.205 → -0.164 |

| Candidate / fold | Hedge P&L bps/day | Hedge trading cost | Hedge borrow cost | Quintile occupancy deviation L / S | Mean stale/unresolved % |
| --- | --- | --- | --- | --- | --- |
| inverse_volatility_20 / F1 | 7.137 → 2.772 | 0.311 → 0.121 | 0.002 → 0.001 | 0.255 → 0.255 / 0.477 → 0.479 | 0.540 → 0.545 |
| inverse_volatility_20 / F2 | -2.623 → -1.101 | 0.385 → 0.132 | 0.002 → 0.001 | 0.113 → 0.110 / 0.513 → 0.513 | 0.234 → 0.220 |
| inverse_volatility_20 / F3 | 1.552 → 0.032 | 0.380 → 0.176 | 0.000 → 0.000 | 0.217 → 0.217 / 0.299 → 0.299 | 1.153 → 1.142 |
| momentum_12_1 / F1 | 3.403 → 0.357 | 0.145 → 0.043 | 0.004 → 0.006 | 0.939 → 0.927 / 0.744 → 0.721 | 0.463 → 0.460 |
| momentum_12_1 / F2 | -1.815 → -0.389 | 0.118 → 0.036 | 0.002 → 0.007 | 1.124 → 1.092 / 0.787 → 0.787 | 0.000 → 0.000 |
| momentum_12_1 / F3 | -0.930 → 0.842 | 0.091 → 0.041 | 0.000 → 0.000 | 1.373 → 1.231 / 0.712 → 0.759 | 0.669 → 0.444 |
| reversal_21 / F1 | 5.829 → 3.517 | 0.280 → 0.113 | 0.206 → 0.039 | 0.513 → 0.519 / 0.656 → 0.660 | 0.558 → 0.516 |
| reversal_21 / F2 | -1.362 → -0.903 | 0.295 → 0.151 | 0.230 → 0.048 | 0.626 → 0.626 / 0.489 → 0.473 | 0.108 → 0.116 |
| reversal_21 / F3 | 3.117 → 0.513 | 0.273 → 0.138 | 0.214 → 0.039 | 0.641 → 0.628 / 0.373 → 0.356 | 0.918 → 0.931 |
| reversal_5 / F1 | 1.470 → 1.912 | 0.658 → 0.371 | 0.127 → 0.033 | 0.229 → 0.229 / 0.540 → 0.547 | 0.169 → 0.199 |
| reversal_5 / F2 | 0.618 → 0.725 | 0.650 → 0.298 | 0.159 → 0.042 | 0.194 → 0.221 / 0.298 → 0.361 | 0.458 → 0.551 |
| reversal_5 / F3 | -1.419 → -0.136 | 0.682 → 0.384 | 0.135 → 0.050 | 0.304 → 0.302 / 0.227 → 0.230 | 0.612 → 0.666 |
| reversal_5_momentum_12_1_blend / F1 | 4.595 → 3.383 | 0.395 → 0.160 | 0.043 → 0.001 | 0.405 → 0.406 / 0.316 → 0.316 | 0.240 → 0.238 |
| reversal_5_momentum_12_1_blend / F2 | -0.175 → -0.407 | 0.415 → 0.111 | 0.024 → 0.000 | 0.340 → 0.331 / 0.305 → 0.302 | 0.157 → 0.236 |
| reversal_5_momentum_12_1_blend / F3 | -3.518 → -0.696 | 0.424 → 0.216 | 0.015 → 0.001 | 0.413 → 0.413 / 0.236 → 0.236 | 1.136 → 1.148 |
| b_intraday / F1 | -1.597 → 0.214 | 0.369 → 0.105 | 0.048 → 0.011 | 0.315 → 0.316 / 0.479 → 0.495 | 0.284 → 0.245 |
| b_intraday / F2 | -0.278 → -0.343 | 0.330 → 0.077 | 0.055 → 0.006 | 0.315 → 0.326 / 0.348 → 0.342 | 0.000 → 0.000 |
| b_intraday / F3 | -0.134 → -0.624 | 0.369 → 0.168 | 0.057 → 0.023 | 0.217 → 0.228 / 0.279 → 0.280 | 0.209 → 0.208 |

## Gates and advance expectations

- D1–D5 are zero in all reported scenarios; all 18 headline books satisfy mean gross [1.5, 2.25], mean stale/unresolved <2%, and mean absolute quintile occupancy deviation ≤2 slots.
- Mean absolute hedge size fell by at least half in 11/18 panels; cap sessions fell from 336 to 7.
- Realized BOVA11 beta is inside ±0.15 in 15/18 panels.
- The expected F1 decline occurred in 4/6 panels; the expected F2/F3 increase occurred in 10/12 panels. These expectations have no gate or selection weight.
- The machine-readable evidence retains all headline fields, same-day-action counts, exposure summaries, notional distributions, costs, financing, occupancy and source/replay hashes. Full registered before/after diagnostics also remain in the sealed Round-1 result.

The pre-existing zero-buffer structural sensitivity remains underdeployed in several control panels. It changes the headline construction and is not a proposed R3.1 cell (whose b_q values are 3/6/9). The established acceptance code applies its gross/occupancy limits to headline books. Its reference-bound departures are disclosed here; no zero-buffer policy is selected or accepted.

| Candidate / fold | Scenario | Mean gross before → after | Short occupancy deviation before → after |
| --- | --- | --- | --- |
| inverse_volatility_20 / F1 | sensitivity_buffer_0 | 1.320 → 1.330 | 2.597 → 2.556 |
| inverse_volatility_20 / F2 | sensitivity_buffer_0 | 1.260 → 1.261 | 2.776 → 2.784 |
| inverse_volatility_20 / F3 | sensitivity_buffer_0 | 1.367 → 1.368 | 2.526 → 2.531 |
| reversal_21 / F1 | sensitivity_buffer_0 | 1.504 → 1.495 | 2.156 → 2.166 |
| reversal_21 / F2 | sensitivity_buffer_0 | 1.498 → 1.491 | 2.185 → 2.187 |
| reversal_5 / F1 | sensitivity_buffer_0 | 1.520 → 1.511 | 2.032 → 2.031 |
| reversal_5 / F2 | sensitivity_buffer_0 | 1.501 → 1.479 | 2.100 → 2.158 |

## Immutable roots and hashes

**Completed Round 1:** `D:\quant-data\b3\processed\model_runs\v2_round1_rev4f_a25ae45_20260908T233251Z`

| Artifact | SHA-256 |
| --- | --- |
| frozen_design.json | `571af91e44fb19514dbb3bade6bce7bcf62403e5c2b973acd841efe31f84ad76` |
| access_audit.json | `337593cee3fa5180b6bf6a3c32af2bc0e0cafc7c547915188e8a41814060144f` |
| artifact_inventory.json | `9940b01076677f19ea8dbf1cca0af2991e2899494e491c92d666d0ceb0c7aafb` |
| round1_result.json | `415aae05aed3df1a21f20d87fbe2b06827df0fec317af58b4bf0f8a84ae35839` |

**Stopped Round 2:** `D:\quant-data\b3\processed\model_runs\v2_round2_rev4f_a25ae45_20260908T233834Z`

| Artifact | SHA-256 |
| --- | --- |
| frozen_design.json | `c90cf848b93aa168951eb3e036f69ca9327901397bf970958156a69f1c301798` |
| access_audit.json | `87f7474f81c55f8d2733d7bae4b26021ce527ac2a8dfdf0768fcfdf6dde421a2` |
| artifact_inventory.json | `9d90d85129368f571eb6daf43185eb35aad945e90e1e46f63df36248898d7ef6` |

[Machine-readable before/after evidence](v2_pass5_rebaseline_evidence.json). No 2025/2026 consumer payload, official validation, test, deployment or paid instance was accessed.
