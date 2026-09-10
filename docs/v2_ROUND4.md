# Round 4: A1–A3 execution record

Status: A1–A3 is committed in `3d1e95f`. The 126-cell CPU replay and the three-seed
neural parent on all fourteen folds are complete, accepted, sealed and recovered.
This is the first session checkpoint; arms and confirmation remain outstanding.
The original CPU root and historical B6 panels remain immutable. No arm comparison
or promotion is made here. Holdout consumers and deployment remain outside this work.

The controlling [registration](../research/preregistrations/v2_round4.md) and
[protocol JSON](../research/preregistrations/v2_round4.json) retain all six configurations,
screening/confirmation seeds, training settings and the selected execution policy.
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

At the measured parent rate, 210 arm fits project to about 9.25 GPU hours and
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
