# Completed Round-2 rev4f rebaseline

All 12 sealed A/B/ensemble/GBDT panels completed under `564efc8`, with no training or score recomputation. Every protected report field is bit-identical. The separately approved history-age provenance hash is reported before/after; its decode and quality strata are unchanged. All headline gates and D1–D5 checks passed. Both earlier stopped roots remain immutable.

## Pooled levels and unchanged conclusions

Intervals use the registered 10,000-replication, 20-session moving-block bootstrap preserving fold boundaries. All four candidates have 375 economics sessions. Net excess is bps/day against all-cash CDI.

| Candidate | Neutral IC (unchanged) | Rev4e net [95%] | Rev4f net [95%] |
| --- | --- | --- | --- |
| arm_A | 0.019564 [0.011607, 0.029864] | 1.476 [-3.102, 9.557] | 2.179 [-1.857, 10.268] |
| arm_B | 0.024777 [0.015940, 0.034967] | 2.404 [-0.521, 9.039] | 3.022 [0.144, 9.977] |
| ensemble | 0.026531 [0.018022, 0.036904] | 1.226 [-1.691, 8.364] | 2.638 [-0.151, 9.861] |
| gbdt | 0.019991 [0.011608, 0.029269] | -1.007 [-5.109, 6.343] | 0.076 [-3.604, 7.257] |

Arm B remains the selected network arm. The ensemble remains the registered overall designation because it leads on neutral IC and no supported economics override fired. Arm B's corrected level interval is now slightly above zero; the ensemble's interval still spans zero. Neither B−A nor the ensemble's paired economic advantage over GBDT is established by these intervals. The B/ensemble/GBDT ordering is unchanged; the corrected Round-1 momentum control remains ahead of B on its point estimate (3.219 versus 3.022 bps/day).

| Paired contrast | IC delta [95%], unchanged | Rev4e net delta [95%] | Rev4f net delta [95%] |
| --- | --- | --- | --- |
| B_minus_A | 0.005212 [-0.000195, 0.010310] | 0.928 [-4.635, 6.354] | 0.844 [-5.051, 6.387] |
| ensemble_minus_gbdt | 0.006540 [0.002281, 0.011748] | 2.233 [-2.336, 7.707] | 2.561 [-1.358, 7.435] |
| ensemble_minus_network | 0.001754 [-0.003446, 0.006363] | -1.178 [-5.685, 4.085] | -0.385 [-4.985, 4.652] |
| network_minus_gbdt | 0.004786 [-0.003333, 0.014622] | 3.411 [-3.386, 10.615] | 2.946 [-2.818, 9.397] |

## Fold-by-fold economics and hedge

Arrows are rev4e → rev4f. Hedge is mean absolute realized hedge notional / prior NAV. Rev4e BOVA11 slopes are calculated for this report using immutable old daily net returns and the same hash-bound BOVA11 series.

| Candidate / fold | Neutral IC | Net bps/day | Mean gross | Mean abs hedge | Cap sessions | Beta vs BOVA11 | Beta vs universe |
| --- | --- | --- | --- | --- | --- | --- | --- |
| arm_A / F1 | 0.003348 | -3.725 → -2.734 | 1.958 → 1.946 | 0.303 → 0.075 | 6 → 0 | 0.019 → -0.187 | -0.047 → -0.180 |
| arm_A / F2 | 0.021309 | 0.861 → 1.139 | 1.907 → 1.901 | 0.266 → 0.123 | 2 → 0 | 0.087 → -0.043 | 0.058 → -0.026 |
| arm_A / F3 | 0.033680 | 7.154 → 7.991 | 1.950 → 1.944 | 0.429 → 0.137 | 26 → 0 | 0.155 → -0.161 | -0.018 → -0.199 |
| arm_B / F1 | 0.016891 | 0.700 → 0.016 | 1.942 → 1.936 | 0.247 → 0.108 | 3 → 0 | -0.100 → -0.188 | -0.121 → -0.190 |
| arm_B / F2 | 0.021353 | -1.734 → -1.885 | 1.937 → 1.929 | 0.239 → 0.108 | 2 → 0 | 0.162 → 0.016 | 0.085 → -0.008 |
| arm_B / F3 | 0.035807 | 8.109 → 10.749 | 1.917 → 1.910 | 0.404 → 0.213 | 29 → 0 | 0.006 → -0.214 | -0.070 → -0.223 |
| ensemble / F1 | 0.016401 | 4.357 → 5.828 | 1.941 → 1.928 | 0.151 → 0.073 | 0 → 0 | -0.171 → -0.156 | -0.122 → -0.116 |
| ensemble / F2 | 0.016849 | -2.244 → -1.536 | 1.945 → 1.952 | 0.151 → 0.063 | 0 → 0 | -0.030 → -0.079 | 0.018 → -0.015 |
| ensemble / F3 | 0.045854 | 1.557 → 3.598 | 1.963 → 1.959 | 0.293 → 0.138 | 5 → 0 | -0.053 → -0.178 | -0.138 → -0.210 |
| gbdt / F1 | 0.010955 | 0.063 → 1.941 | 1.955 → 1.949 | 0.153 → 0.074 | 0 → 0 | -0.037 → -0.048 | -0.037 → -0.044 |
| gbdt / F2 | 0.008978 | -0.951 → -0.911 | 1.938 → 1.945 | 0.134 → 0.072 | 0 → 0 | -0.177 → -0.095 | -0.084 → -0.038 |
| gbdt / F3 | 0.039546 | -2.106 → -0.780 | 1.987 → 1.980 | 0.138 → 0.079 | 0 → 0 | -0.263 → -0.214 | -0.205 → -0.164 |

| Candidate / fold | Hedge P&L bps/day | Hedge trading cost | Hedge borrow cost | Occupancy deviation long / short | Mean stale/unresolved % |
| --- | --- | --- | --- | --- | --- |
| arm_A / F1 | 2.385 → 0.169 | 0.264 → 0.125 | 0.002 → 0.006 | 0.444 → 0.397 / 0.494 → 0.500 | 0.000 → 0.000 |
| arm_A / F2 | 0.002 → 0.236 | 0.335 → 0.130 | 0.017 → 0.003 | 0.579 → 0.568 / 0.513 → 0.523 | 0.213 → 0.211 |
| arm_A / F3 | -0.347 → -0.009 | 0.227 → 0.139 | 0.000 → 0.004 | 0.688 → 0.630 / 0.391 → 0.391 | 0.000 → 0.000 |
| arm_B / F1 | 1.795 → 1.811 | 0.488 → 0.215 | 0.045 → 0.015 | 0.323 → 0.344 / 0.390 → 0.406 | 0.494 → 0.500 |
| arm_B / F2 | -0.972 → -0.680 | 0.534 → 0.241 | 0.011 → 0.010 | 0.318 → 0.348 / 0.647 → 0.645 | 0.000 → 0.000 |
| arm_B / F3 | -1.365 → 0.635 | 0.395 → 0.289 | 0.001 → 0.004 | 0.348 → 0.337 / 0.554 → 0.499 | 0.148 → 0.138 |
| ensemble / F1 | -1.282 → -0.989 | 0.410 → 0.117 | 0.063 → 0.029 | 0.242 → 0.247 / 0.400 → 0.385 | 0.253 → 0.255 |
| ensemble / F2 | 0.200 → 0.109 | 0.379 → 0.090 | 0.024 → 0.005 | 0.298 → 0.305 / 0.431 → 0.440 | 0.000 → 0.000 |
| ensemble / F3 | -0.389 → 1.559 | 0.451 → 0.227 | 0.019 → 0.010 | 0.217 → 0.230 / 0.302 → 0.310 | 0.218 → 0.203 |
| gbdt / F1 | -1.597 → 0.214 | 0.369 → 0.105 | 0.048 → 0.011 | 0.315 → 0.316 / 0.479 → 0.495 | 0.284 → 0.245 |
| gbdt / F2 | -0.278 → -0.343 | 0.330 → 0.077 | 0.055 → 0.006 | 0.315 → 0.326 / 0.348 → 0.342 | 0.000 → 0.000 |
| gbdt / F3 | -0.134 → -0.624 | 0.369 → 0.168 | 0.057 → 0.023 | 0.217 → 0.228 / 0.279 → 0.280 | 0.209 → 0.208 |

## Gates, expectations and provenance

- All D1–D5 signatures are zero, including cost/borrow sensitivities and the D5-only diagnostic. Every headline mean gross is in [1.5,2.25], mean stale/unresolved inventory is below 2%, and mean quintile occupancy deviation is at most 2 slots.
- Mean hedge size at least halved in 9/12 panels; cap sessions changed 73 → 0; BOVA11 beta is within ±0.15 in 5/12.
- F1 economics fell in 1/4 panels; F2/F3 economics rose in 7/8. These were advance expectations, not gates.
- Complete occupancy, same-day-action counts, notional distributions, cost/financing decomposition and exposure summaries are in the machine-readable evidence. All registered recomputed diagnostics are also compared in the sealed result.
- The result's `sources` map retains original score-source provenance. The active replay registration and unrounded decoding contract are in the new frozen design; beta bindings are recorded there and in evaluation inputs. The original sealed root and model checkpoints are unchanged.

| Candidate / fold | Sealed history-age hash | Host history-age hash |
| --- | --- | --- |
| arm_A / F1 | `1fd83a52eaae80a03c30716edac17cdc907f7cb79efdaae5c9a06d9563356411` | `ee3db7b9ea2188912418b96f406107ec31fb736862642c6d70dda85a955878b1` |
| arm_A / F2 | `b39666f41506291758daa83ab17f4fd481220f6c8dd8c60d354ab04b19ac514e` | `85e549f58305093cd2fb9c9dbc2cf5d3e3b631eb670b248d3fa634f624d5deab` |
| arm_A / F3 | `c2540a7a12cc2ccf489fea75ad70bc6a92e2be7c8c4b2f9424e9d8b9fc05b53f` | `4bfbbf10901dd83887f74590ac6a311ac27a5780fe27c24ddb5a25dfd5cc85dd` |
| arm_B / F1 | `1fd83a52eaae80a03c30716edac17cdc907f7cb79efdaae5c9a06d9563356411` | `ee3db7b9ea2188912418b96f406107ec31fb736862642c6d70dda85a955878b1` |
| arm_B / F2 | `b39666f41506291758daa83ab17f4fd481220f6c8dd8c60d354ab04b19ac514e` | `85e549f58305093cd2fb9c9dbc2cf5d3e3b631eb670b248d3fa634f624d5deab` |
| arm_B / F3 | `c2540a7a12cc2ccf489fea75ad70bc6a92e2be7c8c4b2f9424e9d8b9fc05b53f` | `4bfbbf10901dd83887f74590ac6a311ac27a5780fe27c24ddb5a25dfd5cc85dd` |
| ensemble / F1 | `1fd83a52eaae80a03c30716edac17cdc907f7cb79efdaae5c9a06d9563356411` | `ee3db7b9ea2188912418b96f406107ec31fb736862642c6d70dda85a955878b1` |
| ensemble / F2 | `b39666f41506291758daa83ab17f4fd481220f6c8dd8c60d354ab04b19ac514e` | `85e549f58305093cd2fb9c9dbc2cf5d3e3b631eb670b248d3fa634f624d5deab` |
| ensemble / F3 | `c2540a7a12cc2ccf489fea75ad70bc6a92e2be7c8c4b2f9424e9d8b9fc05b53f` | `4bfbbf10901dd83887f74590ac6a311ac27a5780fe27c24ddb5a25dfd5cc85dd` |
| gbdt / F1 | `1fd83a52eaae80a03c30716edac17cdc907f7cb79efdaae5c9a06d9563356411` | `ee3db7b9ea2188912418b96f406107ec31fb736862642c6d70dda85a955878b1` |
| gbdt / F2 | `b39666f41506291758daa83ab17f4fd481220f6c8dd8c60d354ab04b19ac514e` | `85e549f58305093cd2fb9c9dbc2cf5d3e3b631eb670b248d3fa634f624d5deab` |
| gbdt / F3 | `c2540a7a12cc2ccf489fea75ad70bc6a92e2be7c8c4b2f9424e9d8b9fc05b53f` | `4bfbbf10901dd83887f74590ac6a311ac27a5780fe27c24ddb5a25dfd5cc85dd` |

Root: `D:\quant-data\b3\processed\model_runs\v2_round2_rev4f_564efc8_20260909T000352Z`

| Artifact | SHA-256 |
| --- | --- |
| frozen_design.json | `e4469624a9658a9df3a2d566290e36532cf4a1a5f0024551836f1f6a1812e562` |
| round2_result.json | `808369c99b14eebcf3d6f55d67b888c8a2e4cb536965f21adae9c17d282d25a6` |
| access_audit.json | `df95c131b734ac50dcaaf9be482554cbfde246f7c8a5c3ef522de4a7134197b8` |
| artifact_inventory.json | `0f8b3bb6116c0fe285671fd77666fa50ea9690d8ca707a70d3ba3c5dd69767f6` |

[Full machine-readable evidence](v2_pass5_round2_rebaseline_evidence.json). [Round-1 report](v2_PASS5_REBASELINE.md). The next authorized step is the CPU execution sweep; paid Round 3 still requires an explicit go after the data rebuild, acceptance and Round 1'. Official validation/test access and deployment remain unchanged.
