# Round 3 on the repaired store

The registered rule retains six-seed **Arm B** and designates the network over the GBDT and ensemble, with no economics override. Its neutral IC is **0.025078 [0.016108, 0.035333]** and headline net excess is **4.436 [1.307, 11.653] bps/day**. All four development criteria for the future 2025 read are met. This completes the registered pass; the official validation read is not spent, and Round 4 requires its own registration.

The native fast stream has not earned its place on this evidence: B3-minus-fast-off pooled IC is -0.000200 [-0.001068, 0.000718], D1 is 0.000083 [-0.000624, 0.000871], and net excess is 0.075 [-1.666, 2.292] bps/day. All five per-horizon IC intervals include zero. Shareholder spread is lower with fast enabled by 1.297 [0.450, 2.360] bps per holding session, while persistence increases. This is a finding, not authorization to alter the frozen parent after seeing results.

The six-seed IC advantage over three seeds is only 0.000039 [-0.000489, 0.000564]; its net difference is -1.078 [-3.620, 1.049] bps/day. The registered point-estimate rule still retains six, but the extension has not established an improvement. Seed dispersion decreases in F1, is nearly unchanged in F2, and increases in F3. B6 beats GBDT on paired neutral IC, but its paired net interval includes zero. Neither paired neutral IC nor net excess establishes superiority to momentum. The prior old-store ensemble designation is preserved as historical evidence; the rebuilt-store comparison designates the network.

The two smoke trajectories, six fresh Stage-P seeds, 18 Arm-B fits and nine matched fast-off fits completed. All 24 aggregate/comparator panels pass the registered D1-D5 and headline book bounds. Old-store checkpoints were not reused. The complete artifact copy is sealed and hash-verified.

The book uses the R3.1 execution policy selected in sample: theta 1, D3/D5/D10, equal notional and buffer 9 per quintile. Intervals use the registered fold-preserving 10,000-replication, 20-session bootstrap. Net excess is bps/day against all-cash CDI.

| Candidate | Neutral IC [95%] | Net excess [95%] | Persistence 1 | Persistence 5 | Shareholder spread / holding session |
| --- | --- | --- | --- | --- | --- |
| B3 | 0.025039 [0.016119, 0.035400] | 5.515 [2.561, 13.169] | 0.872 | 0.776 | 20.981 |
| B6 | 0.025078 [0.016108, 0.035333] | 4.436 [1.307, 11.653] | 0.873 | 0.777 | 21.462 |
| fast_off | 0.025239 [0.016407, 0.035492] | 5.440 [2.136, 13.222] | 0.867 | 0.774 | 22.278 |
| gbdt | 0.010465 [0.002507, 0.020830] | -0.922 [-4.158, 6.068] | 0.759 | 0.597 | 5.388 |
| momentum | 0.021473 [0.010222, 0.034446] | 3.190 [-1.374, 12.316] | 0.994 | 0.971 | 20.082 |
| ensemble_B3 | 0.021342 [0.012131, 0.032605] | 3.104 [0.558, 9.228] | 0.815 | 0.697 | 15.404 |
| ensemble_B6 | 0.021385 [0.012172, 0.032702] | 1.975 [-1.174, 8.350] | 0.814 | 0.696 | 15.574 |
| ensemble_fast_off | 0.021709 [0.012551, 0.032960] | 2.713 [-0.590, 8.691] | 0.810 | 0.693 | 16.017 |

| Paired comparison | Neutral IC delta [95%] | Net excess delta [95%] | Spread delta [95%] |
| --- | --- | --- | --- |
| B3_minus_fast_off | -0.000200 [-0.001068, 0.000718] | 0.075 [-1.666, 2.292] | -1.297 [-2.360, -0.450] |
| B6_minus_B3 | 0.000039 [-0.000489, 0.000564] | -1.078 [-3.620, 1.049] | 0.481 [-0.783, 1.540] |
| B6_minus_gbdt | 0.014613 [0.006431, 0.022503] | 5.359 [-0.959, 12.007] | 16.073 [3.735, 24.793] |
| B6_minus_momentum | 0.003802 [-0.003846, 0.012434] | 1.246 [-3.776, 5.961] | 1.623 [-7.972, 8.785] |
| ensemble_B6_minus_B6 | -0.003693 [-0.007681, 0.001121] | -2.462 [-7.012, 1.478] | -5.888 [-10.887, 2.408] |
| ensemble_B6_minus_gbdt | 0.010920 [0.006613, 0.015724] | 2.897 [-1.345, 6.707] | 10.186 [4.198, 15.634] |
| ensemble_B6_minus_momentum | -0.000393 [-0.009593, 0.010762] | -1.215 [-8.706, 5.408] | -4.864 [-13.891, 3.875] |

The fast-off fits share the matching fresh Stage-P checkpoints with B3. They force effective fast presence to zero and bypass the TCN; the repaired intraday scalars remain present. Positive B3-minus-fast-off deltas favor the native stream.

| Horizon | B3 minus fast-off IC [95%] |
| --- | --- |
| D1 | 0.000083 [-0.000624, 0.000871] |
| D2 | -0.000253 [-0.000984, 0.000431] |
| D3 | -0.000459 [-0.001472, 0.000583] |
| D5 | -0.000172 [-0.001528, 0.001261] |
| D10 | -0.000244 [-0.003043, 0.002855] |

| Fold | Active name-days with native fast | 3-seed IC dispersion | 6-seed IC dispersion |
| --- | --- | --- | --- |
| F1 | 0.5812 | 0.001366 | 0.001224 |
| F2 | 0.5886 | 0.001196 | 0.001224 |
| F3 | 0.6117 | 0.000981 | 0.002820 |

The registered seed rule keeps **B6**: six-seed IC is 0.025078 and three-seed IC is 0.025039. The six-seed network/GBDT/ensemble comparison designates `network` under the registered IC-first rule; no economics override applied.

| 2025 read-once criterion | Observed | Minimum | Met |
| --- | --- | --- | --- |
| headline_net_bps | 4.436429 | 3.0 | True |
| headline_net_lower_95 | 1.306909 | -2.0 | True |
| neutral_IC_lower_95 | 0.016108 | 0.015 | True |
| sterile_minus_headline_bps | -3.415339 | -5.0 | True |

The development rule is met. This pass does not exercise it and reads only the store ending in 2024. Historical integrity-only scans of later old-store rows are disclosed in [the reader correction](v2_store_comparison_stop_evidence.json); no held-out performance evaluation was run.

[The evidence JSON](v2_round3_evidence.json) includes every scenario summary, economics decomposition, realized BOVA11 beta, traded-signal metrics, per-fold fast/pool gate activations and per-epoch branch gradient norms. Unsupported economics remain explicitly labelled under the frozen rules. Detailed order/fill/action and daily-state audit rows stay in the complete sealed root, with exact source paths and hashes in the GitHub evidence.


| B6 fold | Neutral IC [95%] | Net excess bps/day [95%] |
| --- | --- | --- |
| F1 | 0.017007 [0.002195, 0.030895] | 6.243158 [-2.106415, 15.608021] |
| F2 | 0.021719 [0.008955, 0.038168] | -0.615841 [-6.960274, 11.302348] |
| F3 | 0.036228 [0.015178, 0.055104] | 7.605302 [1.487745, 18.786271] |


Pooled primary IC has 360 supported sessions out of 375; headline economics supports all 375 for all eight candidates. F2 B6 economics is negative at the point estimate, so pooled profitability is not uniform across folds. The evidence retains scenario-specific unresolved flags rather than extending headline support to every stress result. Comparator populations remain their own registered populations; paired metrics use common supported dates. The momentum score mask is not replaced with the network mask.


Gate means below pool channels across six Arm-B seeds within each fold. They are sigmoid outputs, not fractions of explained performance; a nonzero gate or gradient does not establish useful incremental information. Native fast is observed on 58.1%-61.2% of active name-days.

| Fold | Effective fast presence | Fast gate mean | Pool gate mean |
| --- | --- | --- | --- |
| F1 | 0 | 0.126756 | 0.300416 |
| F1 | 1 | 0.126385 | 0.301310 |
| F2 | 0 | 0.129707 | 0.312871 |
| F2 | 1 | 0.137083 | 0.313808 |
| F3 | 0 | 0.126439 | 0.309329 |
| F3 | 1 | 0.124334 | 0.309829 |


Branch L2 means use second-SAM gradients before global clipping, weighted by updates over all completed fine-tune epochs and seeds. Fast-off encoder gradients are exactly zero in every recorded epoch. Every branch, seed, epoch and gate-state diagnostic is included in the evidence JSON.

| Arm / fold | Fast encoder L2 mean | Slow encoder L2 mean | Trunk L2 mean |
| --- | --- | --- | --- |
| arm_B/F1 | 0.014463 | 0.018018 | 0.004957 |
| arm_B/F2 | 0.014168 | 0.017976 | 0.005647 |
| arm_B/F3 | 0.011562 | 0.016669 | 0.004213 |
| fast_off/F1 | 0.000000 | 0.016563 | 0.004586 |
| fast_off/F2 | 0.000000 | 0.016045 | 0.004345 |
| fast_off/F3 | 0.000000 | 0.015617 | 0.004086 |


The GH200 ran up to six trajectories concurrently, each with eight CPU threads. GPU statistics below include compilation and gaps within each phase and come from 15-second samples, so peaks are sampled observations. Both serial one-epoch smokes completed with exactly two compiled graphs each and no score output. A nonfatal PyTorch eager-backward compilation warning appeared in smoke; both graph checks and every subsequent trajectory completed successfully with the registered FP32 settings.

| Phase | Minutes | Mean GPU utilization | Median GPU utilization | Peak GPU MiB |
| --- | --- | --- | --- | --- |
| smoke | 12.37 | 3.4% | 0.0% | 11176 |
| p | 36.66 | 66.5% | 98.0% | 52813 |
| main | 31.75 | 69.7% | 97.0% | 68605 |
| finalize | 4.51 | 0.0% | 0.0% | 3 |


The complete sealed root contains **732 physical files / 603,721,484 bytes**, including checkpoint, score, diagnostic, inventory self-files and closed operational logs. The host copy was SHA-256 and exact-file-set verified at 2026-09-09 15:11:07 UTC. The source remains on the attached persistent filesystem. [Operations evidence](v2_round3_operations.json) records hardware, phase timings, transport hashes and instance closure. The five-minute monitor is paused. No research artifact is modified after sealing.

Root: `D:\quant-data\b3\processed\model_runs\v2_round3_e49aacd_20260909T133152Z`

| Artifact | SHA-256 |
| --- | --- |
| frozen_design.json | c9f84509649e316fb2da3f4f314249917c01ac174c08e188bca70178d85de324 |
| round3_result.json | ae2e4d7b05569282ba1756801375d935ddaaf36477e56fc7fe5b592044ae208b |
| access_audit.json | 4f3874ea1eaff1917e93bbebf5590fcb70212f61f2d2a5898f983ec3ef55b27c |
| artifact_inventory.json | 939c9b2c1e1366bf7cd62e1c040ba78ea29960b87f00e372008435f5b56ab829 |
| economics_detail.json | 1e93be4ce96026ac22e082d4c37a3dd574a413982fe3a402b9a0e34dd736be82 |

These are development-fold research results under reconstructed calendar and inferred corporate-action terms. Deployment is unchanged.
