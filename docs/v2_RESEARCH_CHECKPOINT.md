# Research checkpoint implementation and CPU acceptance

Status: the protocol is committed; the graph changes and staged Round-4 runner are
implemented. The CPU GBDT re-baseline is still running. Round 4 is not yet registered
or launched. This document will receive the completed CPU readout before registration.

The implementation is on GitHub in `d2f7de4`, following the protocol and CDI commits.
The active CPU root is
`D:/quant-data/b3/processed/model_runs/v2_research_checkpoint_d2f7de4_20260909`,
running from an isolated clean checkout at that commit. All 70 copied controls
passed exact source-contract and file-inventory verification at freeze.

## Governing contract

The accepted contract is [v2_research_checkpoint.md](../research/preregistrations/v2_research_checkpoint.md)
and its [source-bound protocol JSON](../research/preregistrations/v2_research_checkpoint.json),
committed in `34e2140`, with the CDI source extension in `f5f387e`.
There are 1,738 evaluation sessions in fourteen half-year folds, 2018H1–2024H2.
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
folds: D1–D5, equity gross 1.5–2.25, stale inventory, occupancy, insolvency, null-control
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

After CPU completion: produce the paired/TreeSHAP diagnostics, verify and seal the
root, publish the completed CPU report, then register `v2_round4.md`. The staged plans
are prepared for that registration; GPU smoke and training remain untested until an
authorized paid session. Launch still requires Gabriel's renewed go. Use the existing
operational `.txt` handoff as the authority for Lambda paths and launch procedures.
Every paid session must end with a sealed host copy, hash verification, termination,
and two provider reads.

The overlay work starts after the GPU launch. Event-day inferred-action flags are
retrospective annotations, never intraday foreknowledge; lower-exposure comparators
are calibrated before evaluation. Meaningful-capital execution assumptions precede
implementability claims or the read; deployment evidence is deferred. No 2025/2026
consumer access, overlay execution, deployment change, or paid launch is authorized
by this checkpoint.
