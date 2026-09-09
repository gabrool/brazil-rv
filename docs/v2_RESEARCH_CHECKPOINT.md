# Research checkpoint implementation and CPU acceptance

Subsequent authorization: [Round-4 A1–A3](v2_ROUND4.md) accepts this sealed checkpoint
and authorizes amended paid sessions after revision/freeze. The figures below retain
their original resolved-fold economics definition; the A1 replay adds full-calendar
economics separately and preserves these source artifacts.

Status: **the pre-GPU checkpoint is complete**. The protocol and graph changes are
implemented; the CPU re-baseline and diagnostics are accepted and sealed.
[Round 4 is registered](../research/preregistrations/v2_round4.md), with a fresh frozen
root and six-job smoke plan ready for review. No GPU training has started; paid compute
requires Gabriel's renewed go.

The implementation is on GitHub in `d2f7de4`, following the protocol and CDI commits.
The completed CPU root is
`D:/quant-data/b3/processed/model_runs/v2_research_checkpoint_d2f7de4_20260909`,
fitted from an isolated clean checkout at that commit. Diagnostics used `f995c33`. All 70 copied controls
passed exact source-contract and file-inventory verification at freeze.

## Governing contract

The accepted contract is [v2_research_checkpoint.md](../research/preregistrations/v2_research_checkpoint.md)
and its [source-bound protocol JSON](../research/preregistrations/v2_research_checkpoint.json),
committed in `34e2140`, with the CDI source extension in `f5f387e`.
There are 1,738 evaluation sessions in fourteen half-year folds, 2018H1â€“2024H2.
Stage P ends 2016-06-30; expanding fine-tuning starts 2016-07-18. F12/F13/F14 preserve
the old evaluation/selection/purge dates, with earlier expanding fit starts.

The new headline and Stage-F checkpoint selection use the equal daily mean of D3/D5/D10
neutral Spearman IC on their common support. The former four-head statistic is retained
as `legacy_primary_ic_1235`. Stage P keeps uniform loss and its prior internal selection
metric. The default loss preserves the previous mean operation bit-for-bit, including
gradients. Historical B6 and the repaired source store remain unchanged.

## Implemented Round-4 paths

| Path | Change relative to fast_off | Stage P |
| --- | --- | --- |
| fast_off | Current twenty intraday scalars; native fast disabled without reading M1 | Fresh baseline graph |
| S0 | No current scalar values, masks, ages, fast presence, current projection or fast parameters | Fresh graph |
| H | Fine-tuning head weights `(0.25,0.25,1,1,1)/3.5` | Reuse same-seed parent |
| P | Fine-tuning persistence coefficient 0.1 | Reuse same-seed parent |
| L | Existing lending slow sidecar, canonical transforms/validity/ages | Fresh graph |
| C | Three stored causal common-state values before the fusion projection/trunk; invalid values zeroed | Fresh graph |

The original native-fast and v1 paths are retained. No new architecture, optimizer,
learning-rate, execution sweep, or blend candidate is introduced. The all-invalid L
test preserves every shared parent batch field, including labels and masks; extra
sidecar channels are explicitly absent. It does not claim identical predictions from
two different-width networks.

`brazil_rv.v2.round4` prepares staged plans: six one-epoch smokes, 12 Stage-P runs,
42 parent fine-tunes, 210 arm fine-tunes, and nine old-window selection-only fits.
The runner requires parent acceptance before the arm phase. It uses six concurrent
training trajectories after the serial smokes. Confirmation uses seeds 61/79/97 on
both sides, adding 42 fine-tunes per configuration and three P runs per needed graph.
Screening cannot promote. S0 wins ties subject to nonnegative constructed-book economics;
IC leads designation, with the registered paired-interval economics exception.
No 2025-read bar has been approved.

## CPU evidence and performance correction

All 70 control/fold cells passed the unchanged engineering gates, including the 2020
folds: D1â€“D5, equity gross 1.5â€“2.25, stale inventory, occupancy, insolvency, null-control
absolute IC below .02, and control absolute IC below .10. No gate was waived or amended.

The [control readout](v2_checkpoint_controls.json) includes every fold and confidence
intervals. Fourteen-fold primary IC is -0.000493 for reversal-5, -0.013047 for
reversal-21, +0.025134 for momentum, +0.018014 for the existing diagnostic blend,
and +0.003434 for the inverse-volatility null. Momentum's primary interval is
[0.012752, 0.034368]; its old-window primary is 0.032833 and its old-window legacy
headline is 0.021473. The blend remains a control, never a promotion candidate.

Engineering acceptance does not make every fold's economics resolved. Momentum has
unresolved terminal economics in F4 (2019H2, remaining hedge) and F7 (2021H1, a position
without a terminal print). The existing rule excludes these whole folds from economics
pooling, leaving 1,490 of 1,738 sessions. Its +2.718 bps/day [-1.203, 6.903] therefore
describes resolved-fold economics, not the full fourteen-fold return. The old three
windows are fully resolved at +3.190 bps/day. No missing price is invented and no
valuation rule is changed. Every candidate's economics coverage is reported separately.
Turnover includes initial/final book trades; the unchanged non-circular block bootstrap
underweights those boundary spikes, so its interval need not contain the whole-panel
turnover mean. Turnover is a readout, not a gate.

The first attempt stopped before scoring because the original CDI file began in 2021.
The replacement derived file adds the official BCB prefix while retaining all original
overlapping observations exactly; see [CDI evidence](v2_checkpoint_cdi_evidence.json).

A subsequent read-only profile found that the first GBDT fit had completed but was
spending its time explaining the full panel, including inactive securities. That
attempt is preserved at `v2_research_checkpoint_cdi_20260909`, with its profiling stop
and 70 accepted controls. No GBDT cell had been published. Its measured process peak
was 1,777,201,152 bytes. This is separate from the source-store build peak of 6.706 GiB.

The restart moves TreeSHAP out of model fitting. The diagnostic sample is at most
4,096 active evaluation rows per fold, evenly spaced in date/security order and
identical across the compared rungs. Mean absolute contributions are averaged across
the five heads and five seeds; value and age channels are grouped per intraday field.
Top-15 tables and sample hashes are retained. Importance has zero selection weight.
Training, checkpoint selection, predictions, and ledger gates are unchanged.

The new freeze may copy only the 70 already accepted controls after exact agreement
on sources, folds, protocol, policy and registration, followed by byte-for-byte
inventory verification. Their original manifests remain untouched and their source
root and acceptance hashes are recorded in the new frozen design. GBDTs are refit.

## Completed CPU results

All **126 cells passed**: 70 controls and 56 GBDT fold cells, comprising 1,400
individual GBDT head/seed fits. The fitting process completed on 2026-09-09 at
20:25 UTC after 7,209.8 seconds (2 hours); its measured peak RSS was 4.134 GiB.
The separate paired/TreeSHAP pass completed at approximately 20:42 UTC.
Accepted cell hashes and identical per-fold TreeSHAP sample indices were checked
before sealing. No hard gate changed.

The sealed root contains **2,956 files / 2,460,356,168 bytes**. Its inventory is
`36aa5f132828aacf700ce728f1d202942c5b11ca9f4b8f9be04df9e8e59da917`.
The access audit passes, with official validation and test access both false.
Historical integrity-only scans of later old-store rows remain disclosed; these
flags describe the current bounded consumer run, not a retraction of that history.

Exact, complete machine-readable snapshots: [CPU result](v2_checkpoint_cpu_result.json),
[all candidate/fold readouts, paired intervals and TreeSHAP](v2_checkpoint_cpu_diagnostics.json),
and [acceptance, resources and seal evidence](v2_checkpoint_cpu_evidence.json).
Their adjacent SHA-256 files bind the published bytes. Daily population audits,
models and ledgers remain in the sealed local root.

### Fourteen-fold headline

Intervals below are the registered 95% fold-preserving moving-block intervals.
Primary IC has 1,598 defined sessions out of 1,738; legacy IC has 1,668. The final
ten sessions of each fold lack common D10 outcomes, rather than being scored as zero.

| Candidate | D3/D5/D10 primary IC [95%] | Legacy D1/D2/D3/D5 IC | Resolved-fold net excess bps/day [95%] | Economics sessions |
| --- | --- | --- | --- | --- |
| a_slow | 0.017440 [0.009235, 0.023769] | 0.017836 | 4.488 [0.587, 8.427] | 1487 / 1,738 |
| b_intraday | 0.015539 [0.007121, 0.021590] | 0.015748 | 2.977 [-0.719, 7.071] | 1487 / 1,738 |
| c_lending | 0.015015 [0.006470, 0.021351] | 0.015075 | 3.622 [0.082, 7.911] | 1487 / 1,738 |
| b_intraday_legacy12 | 0.015923 [0.007648, 0.022345] | 0.016456 | 3.045 [-0.759, 7.028] | 1487 / 1,738 |
| momentum_12_1 | 0.025134 [0.012752, 0.034368] | 0.019797 | 2.718 [-1.203, 6.903] | 1490 / 1,738 |

All four GBDTs exclude F4 (2019H2) and F8 (2021H2) under the existing unresolved
terminal-economics rule. F4 has 3.6645% NAV of unresolved inventory and a remaining
hedge; F8 has 3.3322% NAV and an unresolved terminal settlement flag. Momentum instead
excludes F4 and F7. Consequently, subtracting the GBDT and momentum headline means
would compare different calendars and is not a paired economics advantage. The
GBDT-to-GBDT paired rows below share 1,487 resolved sessions. These exclusions do not
remove IC, persistence or turnover observations. Full per-fold coverage is in the JSON.

### Same fourteen-fold paired comparisons

| Left minus right | Primary IC delta [95%] | Legacy IC delta [95%] | Net excess delta bps/day [95%] |
| --- | --- | --- | --- |
| b_intraday minus a_slow | -0.001900 [-0.004465, 0.000085] | -0.002088 [-0.003644, -0.000391] | -1.512 [-3.424, 0.667] |
| c_lending minus a_slow | -0.002425 [-0.005263, -0.000132] | -0.002761 [-0.004481, -0.001145] | -0.866 [-2.615, 1.400] |
| c_lending minus b_intraday | -0.000524 [-0.001608, 0.000658] | -0.000673 [-0.001329, -0.000249] | 0.646 [-0.420, 1.919] |
| b_intraday_legacy12 minus a_slow | -0.001517 [-0.003694, 0.000460] | -0.001380 [-0.003017, 0.000035] | -1.443 [-3.173, 0.293] |
| b_intraday_legacy12 minus b_intraday | 0.000383 [-0.000982, 0.002154] | 0.000708 [-0.000499, 0.001606] | 0.069 [-1.536, 1.364] |
| b_intraday_legacy12 minus c_lending | 0.000908 [-0.000952, 0.003092] | 0.001381 [0.000135, 0.002506] | -0.577 [-2.460, 0.668] |

The full intraday rung has a negative point delta versus slow-only, but its new-primary
interval narrowly includes zero. Lending is worse than slow-only on this primary
interval; adding lending to the intraday rung has an interval spanning zero. Restricting
intraday inputs to the old twelve recovers only +0.000383 primary IC versus all twenty,
also with an interval spanning zero. This does **not** isolate the earlier repaired-store
loss to the eight newly populated fields: the twelve retained fields still have a
negative point delta versus slow-only. The longer expanding fits also differ from the
historical three-fold experiment, so this is not an exact causal replay of that repair.
The data do not establish that intraday information is generally useless, or determine
the neural S0 result. GBDTs and TreeSHAP have no role in choosing Round-4 arms or parent.

### Former Round-3 evaluation windows

These are F12/F13/F14 (2023H2, 2024H1, 2024H2), with earlier expanding training starts.
All 375 economics sessions are resolved; primary has 345 defined sessions.

| Candidate | Primary IC [95%] | Legacy IC | Net excess bps/day [95%] |
| --- | --- | --- | --- |
| a_slow | 0.025687 [0.013750, 0.038309] | 0.021125 | 7.210 [4.184, 13.175] |
| b_intraday | 0.020223 [0.004875, 0.034178] | 0.016592 | 3.525 [-0.058, 9.335] |
| c_lending | 0.019288 [0.003304, 0.035128] | 0.014348 | 2.853 [0.510, 9.657] |
| b_intraday_legacy12 | 0.020911 [0.005304, 0.034410] | 0.017978 | 2.073 [-1.666, 7.538] |
| momentum_12_1 | 0.032833 [0.015061, 0.048008] | 0.021473 | 3.190 [-1.374, 12.316] |

### Horizon, persistence and lending coverage readouts

The horizon ICs below preserve their historical per-head population; their average
need not equal the new common-population primary. Turnover is daily fraction of NAV,
including fold-boundary opening/closing trades; its interval limitation is stated above.

| Candidate | D1 IC | D2 IC | D3 IC | D5 IC | D10 IC | Persistence 1 | Persistence 5 | Turnover |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| a_slow | 0.015978 | 0.017778 | 0.018416 | 0.019172 | 0.015569 | 0.689725 | 0.564091 | 0.288447 |
| b_intraday | 0.014731 | 0.014772 | 0.015674 | 0.017814 | 0.013744 | 0.688745 | 0.556797 | 0.293642 |
| c_lending | 0.013781 | 0.014393 | 0.015751 | 0.016374 | 0.013629 | 0.689870 | 0.556818 | 0.292191 |
| b_intraday_legacy12 | 0.014581 | 0.016407 | 0.016323 | 0.018511 | 0.014385 | 0.689281 | 0.557690 | 0.287247 |
| momentum_12_1 | 0.015663 | 0.017798 | 0.020494 | 0.025235 | 0.030412 | 0.994412 | 0.975581 | 0.084729 |

The observed-rate era includes any session with an available observed rate (369
resolved sessions here); it can still include imputed rates in the actual book. The
strict subset requires positive opening short exposure and no imputed or placeholder
opening short notional (249 sessions). These are labelled conditional readouts, not
evidence of historical locates or a new candidate selection criterion.

| Candidate | Observed-rate era net bps/day [95%] | Strict held-short subset net bps/day [95%] |
| --- | --- | --- |
| a_slow | 6.933 [4.046, 13.182] | 6.738 [2.655, 12.662] |
| b_intraday | 3.247 [-0.108, 9.266] | 2.684 [-1.534, 9.542] |
| c_lending | 2.740 [0.421, 9.657] | 5.330 [2.259, 13.364] |
| b_intraday_legacy12 | 1.712 [-1.984, 7.605] | 1.247 [-3.996, 7.022] |
| momentum_12_1 | 3.088 [-1.536, 12.321] | 3.052 [-3.857, 14.352] |

### Per-fold primary IC

| Fold | Half-year | a_slow | b_intraday | c_lending | legacy12 | momentum |
| --- | --- | --- | --- | --- | --- | --- |
| F1 | 2018H1 | -0.002917 | -0.002917 | -0.002917 | -0.002917 | 0.002977 |
| F2 | 2018H2 | 0.002282 | 0.002282 | 0.002282 | 0.002282 | 0.023979 |
| F3 | 2019H1 | 0.009321 | 0.009321 | 0.009321 | 0.009321 | 0.031860 |
| F4 | 2019H2 | 0.016014 | 0.016014 | 0.016014 | 0.016014 | 0.015320 |
| F5 | 2020H1 | 0.042799 | 0.042799 | 0.042799 | 0.042799 | 0.064688 |
| F6 | 2020H2 | 0.005337 | 0.005337 | 0.005337 | 0.005337 | -0.009494 |
| F7 | 2021H1 | 0.012630 | 0.012630 | 0.012630 | 0.012630 | 0.013795 |
| F8 | 2021H2 | 0.010718 | 0.010718 | 0.010718 | 0.010718 | 0.051020 |
| F9 | 2022H1 | 0.010195 | 0.004275 | 0.004275 | 0.005953 | 0.038005 |
| F10 | 2022H2 | 0.029248 | 0.026360 | 0.026360 | 0.032373 | 0.012838 |
| F11 | 2023H1 | 0.030893 | 0.029647 | 0.025127 | 0.025144 | 0.008999 |
| F12 | 2023H2 | 0.012790 | 0.013859 | 0.005112 | 0.013345 | 0.024222 |
| F13 | 2024H1 | 0.005620 | 0.007187 | 0.008671 | 0.003268 | 0.012215 |
| F14 | 2024H2 | 0.057806 | 0.039126 | 0.043445 | 0.045473 | 0.061313 |

F1–F8 give identical GBDT results. The later validity-only A2 audit finds current
intraday support during F8 beginning 2021-07-19, although its earlier fit had none;
F1–F7 have no such evaluation input. The full-calendar mean includes long periods
without incremental inputs.
Lending and intraday rungs also match through F10. Per-fold intervals and all metrics
remain in the evidence rather than treating these repeated early values as extra
independent evidence about incremental information.

### Intraday TreeSHAP

Each cell shows field and mean absolute contribution, grouped across its value and age
channels and averaged across five heads and five seeds. The pooled value weights each
fold by its sampled active rows. It includes missingness/age behavior, and is not an
estimate of incremental alpha or a direction of effect. Legacy12 has only twelve
eligible fields. Exact sample counts, hashes and per-fold top lists are retained.

| Rank | b_intraday | c_lending | legacy12 |
| --- | --- | --- | --- |
| 1 | overnight_return_sum_20 (0.001016) | overnight_return_sum_20 (0.001164) | intraday_return_sum_20 (0.000988) |
| 2 | intraday_return_sum_20 (0.000965) | intraday_return_sum_20 (0.000994) | realized_vol_5m_5 (0.000973) |
| 3 | realized_vol_5m_5 (0.000771) | realized_vol_5m_5 (0.000837) | realized_vol_5m_20 (0.000794) |
| 4 | roll_spread_20 (0.000665) | roll_spread_20 (0.000666) | roll_spread_20 (0.000760) |
| 5 | volume_1545_relative_median_20 (0.000613) | realized_vol_5m_20 (0.000654) | realized_skew_5m_20 (0.000689) |
| 6 | realized_vol_5m_20 (0.000579) | volume_1545_relative_median_20 (0.000622) | corwin_schultz_spread_20 (0.000509) |
| 7 | overnight_minus_intraday_mean_20 (0.000566) | realized_skew_5m_20 (0.000568) | intraday_return_sum_5 (0.000489) |
| 8 | realized_skew_5m_20 (0.000564) | overnight_minus_intraday_mean_20 (0.000540) | intraday_range_1545 (0.000469) |
| 9 | corwin_schultz_spread_20 (0.000387) | corwin_schultz_spread_20 (0.000417) | intraday_return_1545 (0.000466) |
| 10 | realized_vol_5m_1 (0.000350) | intraday_return_sum_5 (0.000381) | last_hour_volume_share_lag1 (0.000454) |
| 11 | last_hour_volume_share_lag1 (0.000345) | intraday_range_1545 (0.000364) | realized_vol_5m_1 (0.000444) |
| 12 | intraday_return_sum_5 (0.000342) | realized_vol_5m_1 (0.000345) | vwap_deviation_1545 (0.000359) |
| 13 | intraday_range_1545 (0.000316) | last_hour_volume_share_lag1 (0.000334) | — |
| 14 | overnight_return_sum_5 (0.000303) | overnight_return_sum_5 (0.000330) | — |
| 15 | vwap_deviation_1545 (0.000278) | vwap_deviation_1545 (0.000287) | — |

The B6 momentum diagnostic reported below is complete. The corresponding fourteen-fold neural
parent diagnostic and the nine selection-only fine-tunes require the later authorized
GPU run. CPU results do not promote a GBDT or change the predeclared neural comparison.

## Readouts and next operations

The CPU readout code reports all folds, individual folds and the former three windows;
primary/legacy/per-horizon IC, persistence, spread, turnover and economics; observed-rate
era economics and the stricter held-short subset without imputed or placeholder rates.
The pooled economics remains labelled: balance data starts in March 2022 and observed
rate availability in July 2023. Earlier lending information includes placeholders.

The separate paired GBDT readout compares `a_slow`, `b_intraday`, `c_lending` and
`b_intraday_legacy12`. Momentum diagnostics on B6 and the future re-baselined parent
include daily rank correlation, per-day rank-regression residual IC, and attribution
of the unchanged four-head decile spread to extreme momentum quintiles. Missing
momentum remains an explicit unattributed group. This spread attribution is not a
constructed-book P&L attribution. No diagnostic contributes to selection.

The [historical B6 diagnostic](v2_checkpoint_b6_diagnostics.json) has completed after
verification of the original Round-3 inventory. Its old-window primary is 0.033588,
legacy primary 0.025078, and momentum-residual primary IC 0.014644. Average daily
composite rank correlation with momentum is 0.758935 (daily SD 0.051451); extreme
momentum quintiles account for 86.47% of the unchanged decile-spread contribution.
These are descriptive figures without selection weight or a refit. The new parent
diagnostic and the selection-only paired experiment remain pending GPU training.

Targeted tests cover loss value/gradient equivalence, head support, calendar/access,
model/data/scoring paths, actual S0/C Stage-P training and transfer, H/P checkpoint
reuse, invalid sidecars, daily diagnostic causality and spread conservation, plan
counts and seed pairing, and promotion eligibility. Ruff and the targeted suites pass.

CPU results and their sealed evidence were published in `bd8ce6b` before registering
`v2_round4.md` in `1cb6425`. The fresh root and smoke plan are prepared below; GPU smoke
and training remain untested until an authorized paid session. Launch still requires
Gabriel's renewed go. Use the existing
operational `.txt` handoff as the authority for Lambda paths and launch procedures.
Every paid session must end with a sealed host copy, hash verification, termination,
and two provider reads.

The overlay work starts after the GPU launch. Event-day inferred-action flags are
retrospective annotations, never intraday foreknowledge; lower-exposure comparators
are calibrated before evaluation. Meaningful-capital execution assumptions precede
implementability claims or the read; deployment evidence is deferred. No 2025/2026
consumer access, overlay execution, deployment change, or paid launch is authorized
by this checkpoint.

## Round-4 preparation ready for review

The fresh preparation root is
`D:/quant-data/b3/processed/model_runs/v2_round4_1cb6425_20260909T205015Z`,
frozen at clean commit `1cb642514648af3652f94530fdef2f3868d08b78` after verifying the
complete CPU inventory and accepted cells. Its frozen-design SHA-256 is
`4ac1408fc76060c0e08699d7142679a2180b68c81af3fcba2d866f84586bf923`;
the smoke-plan SHA-256 is
`be5b3447b6fcdcb3799330ec792aeb8653ab0ea2fc4841d8dc5eda39cfc962b7`.

Review the [full frozen design](v2_round4_frozen_design.json),
[six-job smoke plan](v2_round4_smoke_plan.json), and
[prelaunch verification and canonical root](v2_round4_prelaunch.json).
The registration's Git bytes match its frozen digest, with LF endings pinned for the
Windows/Linux handoff. Every smoke command was parsed against the actual training CLI
and checked for its arm, fold, seed, one epoch, CUDA device, feature/loss flags, absence
of reusable score output and model contract. This verifies the plan, not CUDA execution.
There are no smoke, P or F outputs in the new root.

This is a host review plan containing Windows paths. After renewed go, use the frozen
commit and the operational `.txt` handoff to resolve and verify sources on the actual
compute host, freeze a fresh execution root and generate its host-specific commands.
Preserve this preparation root and do not rewrite its frozen paths in place. The latest
main branch additionally publishes these review artifacts; the frozen commit identifies
the exact training implementation. Generate later stages only after their preceding
smokes, checkpoints and parent acceptance exist.

The immediate decision is whether to authorize that paid session. Confirmation is
mandatory before any promotion; the proposed 2025-read and overlay-adoption bars still
need Gabriel's decision. The completed CPU work does not authorize either read or overlay.
