# Round 4: execution and development evidence

Status: Round 4 and its fixed A4/A4.1 seed audit are complete. All original
three-seed fits and full screening are retained; no further training ran.
Working research parent: **S0**; research designation: **S0**.
P fails one unchanged seed-sensitivity occupancy bound and is excluded from
all final choices. All required roots are sealed and fully recovered locally.
The GH200 is terminated, with its exact ID confirmed absent twice. The development-only limitations
and post-result amendments below are part of the conclusion.

The controlling [registration](../research/preregistrations/v2_round4.md) and
[protocol JSON](../research/preregistrations/v2_round4.json) retain all six configurations,
the original screening seeds, training settings and selected execution policy.
The old six-seed requirement is superseded by A4 for this round. The fixed diagnostic
checks all three leave-one-seed-out panels, all six arms and all fourteen folds.
Unstable choices remain inconclusive; no six-seed or independent-replication claim
is made. The amendment was made after training, not before original scores existed.
The prior [CPU checkpoint report](v2_RESEARCH_CHECKPOINT.md) is historical context;
its resolved-fold-only economics will be retained as a secondary replay readout.

## Amendments fixed before scores

A1 settles unpriced equity residuals at the final boundary without waiting out the
ten-session grace, and closes a terminal hedge at its print or labelled last mark.
Full-calendar economics becomes primary; uncertainty, residual notional, 30% adverse
settlement haircut and the 15%-NAV settlement label remain visible. The replay verifies
every non-ledger field and exact resolved-fold economics continuity. Earlier registered
books remain reproducible with their prior boundary convention.

The F4 hedge failure is a source gap: BOVA11 has only 34 observed sessions in 2019H2;
its last mark is 96.15 on 2019-08-16, 92 sessions before the terminal date. The terminal
order was correctly submitted but expired without a print. Settlement closes that
residual and charges costs; it does not recover missing hedge returns or establish
historical executability. See [source evidence](v2_round4_amendment_source_evidence.json).

A2 measures consumed validity before scores. S0's informative subset is F8–F14,
beginning 2021-07-19, correcting the proposed F7 start. L's is F9–F14, beginning
2022-03-23 in the lagged slow window. H/P/C use all folds. Every pair receives its
informative-subset readouts beside all-fold results; S0's tie decision uses F8–F14 at
screening and confirmation. No target, score or selected result determined these subsets.

A3 extends momentum correlation, residual primary IC and extreme-momentum spread
attribution to every arm, with per-fold momentum IC and arm deltas. These remain
descriptive and carry no selection weight. Continuous-book walk-forward is registered
as future work and is not implemented here.

## Verification and execution order

Targeted checks cover terminal no-print settlement and haircut, printed/missing hedge
closes, unchanged pre-terminal accounting, full-calendar pooling despite unresolved
labels, exact resolved-fold continuity, protected non-ledger identity, causal input
coverage and whole-fold subset intervals, plus S0's subset-based decision. Existing
ledger/evaluator/research diagnostics and staged-plan tests pass; Ruff passes.

After the amended commit: launch through the approved `.txt` handoff, verify sources
and freeze a fresh compute-host root, run six serial smokes and twelve Stage-P jobs.
Run the 126-panel settlement replay on the local CPU concurrently; parent acceptance
requires its bound completed result. Then parent fits/acceptance, arms and selection-only
fits, screening, the registered confirmation roster, and final evaluation/report.
Measure runtime after smokes/P and split into recoverable sessions when appropriate.
Each paid session ends with a sealed verified host copy, exact-ID termination and two
provider reads. Detailed results and operational evidence will be recorded here.

## Completed CPU settlement replay

The complete [replay result](v2_round4_cpu_replay_result.json) records all 126 source
and replay evaluation hashes, their protected-field identity, every candidate/fold,
all six GBDT pairs and comparisons with momentum. No score or model was refit.
All registered engineering gates pass. Exact dictionary comparisons reproduce each
candidate's sealed resolved-fold pooled and former-three-window economics, including
bootstrap intervals and observation counts. Every amended book and economics pair
uses all 1,738 calendar sessions. Undefined IC sessions retain their existing masks.

| Candidate | Full-calendar excess bps/day [95% interval] | Uncertain folds |
| --- | ---: | --- |
| a_slow | 4.616 [1.276, 8.208] | F4, F8 |
| b_intraday | 3.322 [0.189, 7.023] | F4, F8 |
| c_lending | 3.875 [0.826, 7.720] | F4, F8 |
| b_intraday_legacy12 | 3.381 [0.156, 6.977] | F4, F8 |
| momentum_12_1 | 2.627 [-1.630, 6.551] | F4, F7 |

The b_intraday-minus-a_slow paired excess is -1.293 [-2.929, 0.570] bps/day;
a_slow-minus-momentum is 1.989 [-1.262, 6.010]. Their protected primary-IC deltas
remain -0.001900 [-0.004465, 0.000085] and -0.007004 [-0.014422, 0.002757].
These CPU comparisons remain context, with zero neural-arm selection weight.
The F4 source gap and all settlement uncertainties described above still apply.

Host replay root: `D:/quant-data/b3/processed/model_runs/v2_round4_cpu_replay_3d1e95f_20260909`.
Result SHA-256: `17d43a0ab9c2818605e87804cc6fe8a67063d33aba8b12b9f0d03235246c5f68`.
Inventory SHA-256: `1259dc9544996f626fd6d8bea67ee12afdbef0c1e4d16b493039dded5a719746`.
The closed log, access audit and 1,799 inventoried files total 2,173,424,257 bytes;
the two inventory self-files are additional. The root was verified before transport.

## Session 1: accepted parent checkpoint

Approved launcher instance `482f0aea79aa459bbd4c0466e6d3607e` completed bootstrap
at 21:52:52 UTC on 2026-09-09, on clean commit `3d1e95f`.
Execution root: `v2_round4_3d1e95f_20260909T220000Z` under the attached NFS model-runs
directory. Its frozen-design SHA-256 is
`b1965d5a6c87f9f8adf997a099e385618e42eecbfbcd80a0de55ce652211546b`.
The source CPU/CDI transport hash verified before extraction; freeze reverified the
complete sealed CPU inventory, source bindings and informative-fold payload.
Serial smokes started at 22:04:41 UTC; their wall time includes first compilation.
The GPU logger records utilization, memory and power every fifteen seconds.
All twelve P runs and 42 parent fits subsequently passed their checkpoint/graph
checks. All fourteen aggregate parent books passed unchanged acceptance gates.
This session ends before arms, with complete host recovery preceding exact-ID closure.


## Parent result and uncertainty

The [complete parent readout](v2_round4_parent_result.json) contains every horizon,
fold, former Round-3 window, persistence/spread metric, lending-coverage readout and
settlement label. [Operational evidence](v2_round4_parent_operations.json) binds
all 42 fit manifests, fourteen accepted books, timings and the verified host copy.
These are screening-seed parent levels, without arm selection or six-seed confirmation.
A2/A3 paired and momentum diagnostics remain due at full screening and confirmation.

| Pooled parent metric | Estimate [95% interval] |
| --- | ---: |
| Primary D3/D5/D10 IC | 0.023481 [0.013620, 0.031649] |
| Legacy D1/D2/D3/D5 IC | 0.022294 [0.017002, 0.028210] |
| Full-calendar net excess, bps/day | 4.200572 [-0.491423, 7.828492] |
| Resolved-fold-only net, bps/day | 4.243828 [-0.931452, 8.616697] |
| Persistence 1 | 0.793029 [0.791247, 0.799133] |
| Persistence 5 | 0.701717 [0.697041, 0.708214] |
| Four-head score spread, bps/holding session | 11.279949 [4.965271, 18.874427] |
| Turnover / NAV | 0.319259 [0.275579, 0.299777] |

Primary IC has 1,598 defined sessions; economics has all 1,738. Turnover includes
fold-boundary trades; the unchanged non-circular bootstrap underweights those spikes,
so its interval lies below the whole-panel turnover mean. No estimator was changed.

F4, F8 and F9 retain `economics_unresolved` labels. F4 uses last-mark hedge settlement
against the disclosed 92-session BOVA11 gap; total labelled settlement is 16.09% of
terminal NAV and triggers the 15% label. F8/F9 report terminal unpriced-equity fractions
of 3.51%/1.89%. The adverse settlement haircut reduces terminal NAV by 0.05471,
0.03318 and 0.02853 respectively (initial NAV units), without excluding these folds.
The secondary resolved-fold pool has 1,363 sessions. Positive net point estimates
do not establish positive net excess with 95% confidence or historical executability.

| Fold | Primary IC | Net excess bps/day | Uncertain economics label |
| --- | ---: | ---: | --- |
| F1 | 0.019438 | 14.188 | no |
| F2 | 0.010322 | -7.977 | no |
| F3 | 0.028163 | 0.381 | no |
| F4 | 0.030088 | 7.953 | yes |
| F5 | 0.064379 | 20.058 | no |
| F6 | -0.009113 | 3.048 | no |
| F7 | 0.017480 | 6.022 | no |
| F8 | 0.032764 | 8.528 | yes |
| F9 | 0.007636 | -4.450 | yes |
| F10 | 0.024872 | -1.634 | no |
| F11 | 0.017604 | -1.613 | no |
| F12 | 0.015426 | 7.164 | no |
| F13 | 0.020673 | 0.188 | no |
| F14 | 0.048760 | 6.884 | no |

## Measured session budget and recovery

Serial smokes took 28.57 minutes. Twelve P runs took 40.71 minutes / 97 total epochs,
with 79.97% mean GPU utilization and 54,485 MiB peak memory. Parent fine-tuning
took 111.01 minutes / 447 total epochs, with
90.63% mean GPU utilization and 52,903 MiB peak memory. Parent acceptance
then took 115 seconds. All model and training settings remained registered.

The original budget, before A4, projected 210 arm fits at about 9.25 GPU hours and
confirmation F to 3.70–11.10 hours depending on the registered roster. These are
planning proxies: arm-specific patience can differ, and selection-only fits, fresh
confirmation P, CPU readouts and transport add time. Continue in fresh sealed
sessions, reusing verified P/parent artifacts and preserving their source inventories.

The closed complete root contains 2,900 physical files /
2,708,863,375 bytes. Host recovery verified every file, hash and exact file set at
2026-09-10T01:32:46.639853+00:00. Host root: `D:\quant-data\b3\processed\model_runs\v2_round4_3d1e95f_20260909T220000Z`.
Parent result SHA-256: `fcdc338d54f8626b8c448710243e135366ad746a13c71c848823e9f370661b40`.
Inventory SHA-256: `c5e836d4945ea85227dfbfcd11b1b79066fe7d34ca6d1f9e9ce7d81252acedd3`.
Exact instance `482f0aea79aa459bbd4c0466e6d3607e` was terminated after verified recovery
and clean local/GitHub/instance match at `a0ce839`. Provider inventory confirmed
its absence at 2026-09-10T01:37:28.6759002Z and 2026-09-10T01:37:38.9778036Z.
This closing documentation records later provider observations. No sealed artifact
was modified. The remaining arms and confirmation are still authorized for fresh sessions.

## Second session: arms and screening

Exact instance `248ec61615fa4c7ba2a58a32ff8fc813` was launched through the approved
launcher and bootstrapped from clean main `e51e0ea`, then checked out the unchanged
frozen training commit `3d1e95f`. Its fresh root is
`/lambda/nfs/brazil-rv-east3/quant-data/b3/processed/model_runs/v2_round4_arms_3d1e95f_20260910T015000Z`.
The fresh freeze exactly matches the original design SHA-256
`b1965d5a6c87f9f8adf997a099e385618e42eecbfbcd80a0de55ce652211546b`.

After verifying the complete sealed parent root, the session copied 2,848 files /
2,707,447,895 bytes of smoke, P, parent and CPU-replay artifacts unchanged. The exact
copied file set and hashes passed; `operations/reuse_provenance.json` has SHA-256
`ff8054f887abc5c1c21250e2b299d1a6bc582ae2e6fa242c8c86e6b454666bd2`.
Source manifests retain their original paths and hashes, with the immutable source
root available on the same persistent filesystem. Completed trajectories are reused.

The driver started at 2026-09-10T01:58:35.317926+00:00, sequencing 210 arm fits,
nine selection-only fits and full screening evaluation. The arm plan has SHA-256
`d2e6370604abce6012b021cbc825d418d37303a2541e0a0d0e8a476dbf9a4187`
and six concurrent trajectories. All arm and selection trajectories completed. This session
ends after screening with complete recovery, verification and exact-instance
termination. A4 now replaces the former planned fresh confirmation session with
a CPU-only saved-score audit on this instance before final recovery and closure.

## Completed three-seed screening

All 87 original screening/selection books pass the unchanged engineering gates.
Arms took 9.04 hours, selection-only fits 38.54 minutes, and full screening readouts
27.48 minutes. [The complete summary projection](v2_round4_screening_report.json)
retains all pooled/per-fold readouts and pairs; only repetitive daily momentum
arrays are omitted from GitHub and remain in the sealed, verified full source.

All six configurations have nonnegative net point estimates. S0 is the provisional
IC leader and provisional working parent; no economics override qualifies.
The final A4 decision below supersedes this original provisional screening label.
The first complete-score audit stopped at P/F14 with seeds 11/47: long-quintile
occupancy deviation was 2.061417 against the unchanged 2.0 limit; D1–D5 were zero.
[A4.1](../research/preregistrations/v2_round4_budget_amendment.md) now disqualifies
an arm with an isolated non-baseline occupancy failure from all final-choice panels.
The failed report remains failed. Baseline, D1–D5, other risk and implementation
errors remain global stops. The continuation reuses sealed completed evaluations
in a fresh root and completes the unaffected cells. This continuation rule was
adopted after observing the failure. [Exact evidence](v2_round4_seed_balance_stop.json)
includes all four P/F14 panels and their hashes.

| Arm | Primary IC [95%] | Net excess bps/day [95%] | Persistence 1 | Turnover/NAV |
| --- | --- | --- | ---: | ---: |
| fast_off | 0.023481 [0.013620, 0.031649] | 4.201 [-0.491, 7.828] | 0.793029 | 0.319259 |
| S0 | 0.026591 [0.016880, 0.035140] | 4.936 [1.174, 8.989] | 0.782725 | 0.299263 |
| H | 0.023822 [0.013856, 0.032094] | 4.556 [0.201, 8.654] | 0.800255 | 0.294837 |
| P | 0.022907 [0.011834, 0.031855] | 2.852 [-1.242, 6.282] | 0.981127 | 0.117978 |
| L | 0.023737 [0.013873, 0.032420] | 4.785 [0.688, 8.998] | 0.808814 | 0.278204 |
| C | 0.023992 [0.014375, 0.032796] | 3.857 [-0.445, 7.751] | 0.788803 | 0.315491 |

| Arm minus fast_off | All-fold IC delta [95%] | All-fold net delta bps/day [95%] | Informative-fold IC delta [95%] |
| --- | --- | --- | --- |
| S0 | 0.003110 [0.001522, 0.005129] | 0.735 [-0.234, 3.130] | 0.006113 [0.002838, 0.009972] |
| H | 0.000341 [-0.000055, 0.000684] | 0.356 [-0.332, 1.891] | 0.000341 [-0.000055, 0.000684] |
| P | -0.000574 [-0.002773, 0.001134] | -1.348 [-3.374, 1.225] | -0.000574 [-0.002773, 0.001134] |
| L | 0.000256 [-0.000255, 0.001350] | 0.585 [-0.408, 2.721] | 0.001553 [0.000543, 0.003041] |
| C | 0.000511 [-0.000111, 0.001931] | -0.344 [-1.402, 1.360] | 0.000511 [-0.000111, 0.001931] |

S0 uses F8–F14 and L uses F9–F14 as declared from input validity; H/P/C use all
fourteen folds. Full paired tables include both members’ informative subsets.

| Arm | Momentum rank correlation mean | SD | Momentum-residual primary IC | Extreme-momentum spread share |
| --- | ---: | ---: | ---: | ---: |
| fast_off | 0.689061 | 0.106642 | 0.012130 | 70.58% |
| S0 | 0.726328 | 0.104797 | 0.015414 | 70.24% |
| H | 0.692605 | 0.106719 | 0.012736 | 72.68% |
| P | 0.760176 | 0.091832 | 0.008861 | 82.03% |
| L | 0.700587 | 0.109470 | 0.011734 | 70.16% |
| C | 0.688694 | 0.098953 | 0.012245 | 67.38% |

S0 raises both momentum correlation and residual IC relative to fast_off. These
are descriptive score diagnostics, not an independent alpha or implementability
test. P produces much more persistent scores and lower turnover, alongside lower
primary IC and net point estimates. None of these diagnostics changes the rule.

| Fold | Momentum IC | fast_off | S0 | H | P | L | C |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| F1 | 0.002977 | 0.019438 | 0.019428 | 0.019096 | 0.007312 | 0.021954 | 0.017773 |
| F2 | 0.023979 | 0.010322 | 0.009558 | 0.010830 | 0.022135 | 0.004355 | 0.008650 |
| F3 | 0.031860 | 0.028163 | 0.028237 | 0.028102 | 0.024897 | 0.027697 | 0.028165 |
| F4 | 0.015320 | 0.030088 | 0.032060 | 0.030621 | 0.030530 | 0.031529 | 0.031414 |
| F5 | 0.064688 | 0.064379 | 0.065423 | 0.064197 | 0.060865 | 0.063692 | 0.063457 |
| F6 | -0.009494 | -0.009113 | -0.009774 | -0.009039 | -0.017700 | -0.009883 | -0.010610 |
| F7 | 0.013795 | 0.017480 | 0.016263 | 0.018340 | 0.015505 | 0.016541 | 0.016929 |
| F8 | 0.051020 | 0.032764 | 0.033842 | 0.034552 | 0.042709 | 0.031721 | 0.032401 |
| F9 | 0.038005 | 0.007636 | 0.033771 | 0.008933 | 0.028765 | 0.015915 | 0.018327 |
| F10 | 0.012838 | 0.024872 | 0.025454 | 0.024942 | 0.022492 | 0.024535 | 0.025046 |
| F11 | 0.008999 | 0.017604 | 0.020567 | 0.017441 | 0.012104 | 0.018656 | 0.018595 |
| F12 | 0.024222 | 0.015426 | 0.018913 | 0.015958 | 0.016835 | 0.017862 | 0.016301 |
| F13 | 0.012215 | 0.020673 | 0.023747 | 0.020196 | 0.014223 | 0.018987 | 0.020507 |
| F14 | 0.061313 | 0.048760 | 0.054383 | 0.049081 | 0.040072 | 0.048415 | 0.048641 |

The full original screening root contains 7,711 files / 5,111,515,026 bytes,
verified on the host at 2026-09-10T12:19:54.065976+00:00.
Inventory SHA-256: `a4c9d098124515594c27acc970b0f3805adc3a36ebbed74e9ddf5be82b284c76`.
Full result SHA-256: `6ab60e7a3c37d53563c5fd3e140a0f001ce7b43dd9aa5b9a6ca92b5e281bdaee`.

The first A4 audit attempt stopped before any book when the new caller omitted
the canonical forecast reader’s required schema/date/security bindings. The
reader was kept intact; the caller was corrected in `eb5e9ca`, 31 targeted tests
passed, and the audit restarted in a fresh root. [Stop evidence](v2_round4_seed_audit_stop.json)
binds the preserved failed attempt. No trained model or original screening result changed.

A subsequent source inspection caught a reporting adapter mismatch before the
continuation could make a decision: `promotion_trace` consumes each comparison's
pooled metrics, whereas the audit supplied the enclosing report. The audit was
interrupted after copying 246 completed evaluations, with zero new evaluations.
Its partial root and interruption logs are preserved and sealed. The corrected
adapter is shared by the omission and final decisions. Forty-two targeted tests
pass, including actual economics-override and excluded-arm selection through the
nested-report adapter; it also reproduces the original screening choice on the
published full report. No score, book, threshold or selection rule changes.

## Final A4 development decision

Working research parent: **S0**. Research designation: **S0**.
Both choices are stable across all four fixed panels.
The audit accepts 251 of 252 leave-one-out books and retains one rejected book.
P is excluded from every final choice after its isolated occupancy failure; its
reported metrics remain descriptive. The original three-seed headline is unchanged.
This is the fixed A4/A4.1 development decision, with post-result amendments disclosed;
it is not six-seed confirmation or independent replication. The overlapping panels
do not provide new independent confidence levels. No 2025/2026 or deployment read ran.

[The machine-readable audit](v2_round4_seed_audit_report.json) retains all omission
summary metrics, pairs, informative subsets, fold points, individual-seed IC and
matched individual-seed IC deltas. Raw daily arrays and population audits remain
in the complete verified host copy.

| Panel | Eligible arms after engineering exclusions | IC leader | Economics override | Provisional parent |
| --- | --- | --- | --- | --- |
| omit 11 | fast_off, S0, H, L, C | S0 | none | S0 |
| omit 29 | fast_off, S0, H, L, C | S0 | none | S0 |
| omit 47 | fast_off, S0, H, L, C | S0 | none | S0 |
| full 11/29/47 | fast_off, S0, H, L, C | S0 | none | S0 |

| Omitted seed | Arm | Primary IC [95%] | Net excess bps/day [95%] |
| --- | --- | --- | --- |
| 11 | fast_off | 0.023383 [0.013599, 0.031539] | 4.938 [0.652, 8.801] |
| 11 | S0 | 0.026464 [0.016822, 0.035002] | 5.194 [1.363, 9.422] |
| 11 | H | 0.023745 [0.013767, 0.031948] | 4.140 [0.008, 8.266] |
| 11 | P | 0.023117 [0.012158, 0.032059] | 2.617 [-1.697, 6.339] |
| 11 | L | 0.023853 [0.013976, 0.032572] | 4.401 [0.034, 8.556] |
| 11 | C | 0.023818 [0.014231, 0.032682] | 3.998 [-0.302, 7.804] |
| 29 | fast_off | 0.023321 [0.013513, 0.031419] | 3.399 [-1.126, 7.202] |
| 29 | S0 | 0.026512 [0.016772, 0.035000] | 5.638 [1.591, 9.622] |
| 29 | H | 0.023688 [0.013767, 0.031870] | 4.022 [-0.371, 8.018] |
| 29 | P | 0.022907 [0.011800, 0.031858] | 3.384 [-0.692, 7.100] |
| 29 | L | 0.023874 [0.013973, 0.032698] | 5.192 [0.749, 9.066] |
| 29 | C | 0.022955 [0.013463, 0.031651] | 4.440 [0.457, 8.387] |
| 47 | fast_off | 0.023654 [0.013829, 0.031955] | 5.118 [1.066, 9.362] |
| 47 | S0 | 0.026649 [0.016906, 0.035275] | 4.599 [0.837, 8.723] |
| 47 | H | 0.024062 [0.014141, 0.032435] | 4.557 [0.133, 8.715] |
| 47 | P | 0.022461 [0.011407, 0.031365] | 2.848 [-1.452, 6.508] |
| 47 | L | 0.023295 [0.013614, 0.031809] | 4.104 [0.077, 8.245] |
| 47 | C | 0.025006 [0.015308, 0.033780] | 4.731 [0.385, 8.800] |

| Omitted seed | S0 minus fast_off IC [95%] | Net delta bps/day [95%] | Informative IC [95%] |
| --- | --- | --- | --- |
| 11 | 0.003081 [0.001557, 0.005107] | 0.256 [-1.041, 2.442] | 0.005187 [0.002042, 0.008827] |
| 29 | 0.003191 [0.001421, 0.005329] | 2.239 [0.718, 4.445] | 0.007038 [0.003563, 0.011247] |
| 47 | 0.002995 [0.001306, 0.005010] | -0.519 [-2.518, 1.673] | 0.006164 [0.002676, 0.010046] |

S0's paired IC improvement remains positive in every omission, including its
informative subset. Its incremental net return is less stable: omitting seed 47
changes the net point difference to negative, and two of the three net intervals
span zero. The stable model choice does not establish a stable economic advantage.

| Individual seed | fast_off IC [95%] | S0 IC [95%] | Matched S0 minus fast_off IC [95%] |
| --- | --- | --- | --- |
| 11 | 0.023409 [0.013718, 0.031643] | 0.026696 [0.016807, 0.035380] | 0.003286 [0.000941, 0.005737] |
| 29 | 0.023570 [0.013728, 0.031891] | 0.026563 [0.016959, 0.035195] | 0.002993 [0.001471, 0.005066] |
| 47 | 0.023012 [0.013263, 0.030941] | 0.026223 [0.016596, 0.034538] | 0.003211 [0.001554, 0.005367] |

Every source fit, all fourteen folds, the original 20-epoch/patience-three
contract, execution/cost assumptions and 10,000-draw intervals are retained.
The saved-score audit trains zero models. It replaces the blanket extra
three-seed stage; evidence about unseen initializations is correspondingly
weaker. Extra seeds are reserved for a separately registered consequential
or close decision, without choosing seeds until a desired answer appears.

Complete host recovery covers the original screening, the failed occupancy
audit (including the first loader stop), the preemptively interrupted reporting
attempt, and the final audit. Each exact file set and SHA-256 verified before
instance closure. See [recovery evidence](v2_round4_final_operations.json).
Report commit `b6a114465920c3bad2b35420ab085b15d2b6a42b` matched clean local
main, GitHub and the compute host before termination. Exact instance
`248ec61615fa4c7ba2a58a32ff8fc813` was confirmed absent at
2026-09-10T13:08:42.2241632Z and 2026-09-10T13:08:52.5076574Z.
The closing documentation records these later provider observations; no sealed
artifact changed and no adjacent instance was touched. The monitor is paused.
