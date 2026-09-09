# Research checkpoint: fourteen development folds

Registered from Gabriel's `v2_research_checkpoint.md`, before any new fold scores.
This replaces the development protocol for future research. Immutable Round-3 roots,
scores and reports remain historical evidence. This registration authorizes CPU
preparation and re-baselining only. Paid compute needs Gabriel's renewed go. It does
not authorize a 2025 read, any 2026 consumer, deployment, or the future research agenda.

## Metric and selection

`primary_ic` is the session-pooled daily equal mean of D3, D5 and D10 neutral-target
Spearman correlations, using one common population with active membership and valid,
finite neutral targets and supported finite scores on all three heads. Each day needs
at least 20 names and three defined correlations. Neither D1/D2 validity nor their
scores restrict this population. Paired comparisons intersect both candidates' score
support before computing either IC. Undefined days remain missing on the calendar;
they are never replaced by zero or averaged over fewer heads.

`legacy_primary_ic_1235` retains the former rev-4 neutral-target metric and its exact
four-head outcome/scale population. Existing per-horizon, price/shareholder, traded
composite, persistence and spread definitions stay unchanged. Evaluation schema V17
identifies the new headline. Session pooling and 20-session moving-block uncertainty
retain the existing conventions (10,000 replications, bootstrap seed 20260815 in
pooled research comparisons; no block crosses a fold boundary).

Stage F selects raw Patience checkpoints on D3/D5/D10. Stage P retains uniform loss
and its previous four-head internal selection metric, isolating the requested Stage-F
selection change. H alone changes fine-tuning loss weights; the default still calls
`head_losses.mean()` exactly. A paired selection-only diagnostic refits the parent
using four-head Stage-F selection on F12/F13/F14, reusing the same new Stage-P seed
checkpoints. This adds nine fine-tunes; it does not refit historical fast-on B6.

## Calendar and sources

Stage P spans 2010-01-04 through 2016-06-30: 1,607 exchange sessions and approximately
6.5 years in total, including internal selection and the unchanged 70-session embargo.
It is a single pretraining per seed/graph, not per fold. Fine-tuning starts 2016-07-18,
with eleven actual exchange sessions between P and F. F1 fits 286 sessions through
2017-09-06. Every fold expands its fit from that same start and then uses exactly
10 purge, 55 selection, 10 purge, and a half-year evaluation window.

The accompanying [protocol JSON](v2_research_checkpoint.json) records all exact dates,
counts and source identities. There are 1,738 evaluation sessions in 2018H1–2024H2.
F12/F13/F14 exactly reproduce the old F1/F2/F3 evaluation, selection and purge dates.
Their fit windows necessarily start earlier; “exact reproduction” does not claim
unchanged training dates or unchanged predictions. The repaired store is bounded at
2024-12-30 and remains immutable. The existing inferred-action/calendar source labels
remain. Historical integrity-only scans of later old-store rows stay disclosed; this
checkpoint does not consume any new 2025/2026 payload.

Source manifests correct a coverage claim in the supplied instructions: lending
balances begin in March 2022, not at the start of development; observed rate source
records begin 2023-07-10 and first become available 2023-07-11. Earlier availability
and pricing include labelled placeholders. Rates are decimals. Economics report all
folds, the old three evaluation windows, the observed-rate era (still allowing labelled
imputation), and a stricter subset with positive short exposure and no imputed or
placeholder opening short notional. None is presented as evidence of historical locates.
The heuristic sqrt(1738/375) precision gain is not a confidence-interval guarantee.

Before the first scored fold, the CPU startup found that the old CDI extension starts
2021-08-16. A BCB SGS-12 prefix through 2021-08-15 now extends the derived series back
to 2016-07-18. Every old-extension date/value is bitwise unchanged; all new development
sessions align, and the Experiment-52 overlap proof still passes. The amended JSON
binds the new file, with source/hash evidence in `docs/v2_checkpoint_cdi_evidence.json`.
The unscored `v2_research_checkpoint_34e2140_20260909` attempt is retained with its
startup log; the restart uses a fresh root. This changes no rate assumptions or gates.

## CPU checkpoint and stops

Run the five existing controls on each fold, then a_slow and b_intraday GBDT, then the
c_lending and b_intraday_legacy12 diagnostics. All GBDTs use the unchanged five-seed,
five-head settings and expanding Stage-F fit windows, with no added pretraining data
or decay. There are 1,400 individual head/seed fits, not 1,400 five-head ensembles.
The legacy12 set comes from nonzero old transformed coverage in
`docs/v2_INTRADAY_COVERAGE.md`, with its original names/order, ages and fast_present.
TreeSHAP uses LightGBM's native contributions in bounded chunks, with evaluation-only
importance explicitly diagnostic and no role in selection.

Persist each cell before checking unchanged baseline-book D1–D5 zero limits,
gross 1.5–2.25, stale inventory below .02, insolvency and quintile occupancy limits.
The inverse-volatility null must have absolute new primary IC below .02 on every fold;
control absolute IC must remain below .10. A failure stops the run, with its exact
cell/report preserved. No runtime waiver or gate tuning is allowed. A justified change
needs a separately recorded amendment and affected cells in a fresh root. The user's
earlier standing authority to choose the recommended resolution is retained for
implementation decisions; it does not override the renewed-go requirement for paid
compute or authorize a holdout read. The store/build measured 8-GiB invariant remains.

## Parent and confirmation

The paired parent is fast_off, retaining the current twenty intraday scalar inputs.
Screening seeds are 11/29/47; confirmation seeds are 61/79/97. Any proposed promotion,
including S0 becoming the parent, first requires the six-seed paired panel. Both sides
of a confirmation comparison need the same six seeds, same folds and common support.
No screening result alone changes the default.

## Promotion

IC ranks candidates first. Negative constructed-book economics makes a candidate
ineligible; a positive paired economics interval may override the IC leader only if
their paired IC interval contains zero. Persistence and turnover are reported, not
gated. For the default simplification decision, S0 wins ties unless its paired primary
IC upper 95% bound is below zero, subject to the same nonnegative-economics eligibility
and six-seed confirmation. If S0 fails eligibility, retain fast_off and report that
reason explicitly; do not silently override the economics rule. No blend is a candidate.
All in-sample execution-selection labels remain. Record unsuccessful arms equally.
Gabriel has not yet set the proposed 2025-read bar or overlay-adoption thresholds.

## Round-4 preparation and later work

After committing this checkpoint, register Round 4 separately: parent plus S0/H/P/L/C,
no other architecture, SAM, learning-rate or transfer changes. Parent/S0/L/C each need
fresh P for three screening seeds: **12 Stage-P runs**, not the supplied 3–9 estimate.
There are 252 screening fine-tunes plus the nine selection-only diagnostics. Confirmation
adds 42 fine-tunes per confirmed configuration and three P runs for each needed graph.
The supplied fixed dollar/hour estimate is unverified and is not a compute commitment.
If S0 becomes the parent, future transferred hypotheses require fourteen folds per
seed (42 screening fits), not the old nine-fit count.

The overlay implementation and its CPU replay start only after the Round-4 GPU launch.
Two proposed overlay statements need causal interpretation: inferred event-day action
flags can be retrospective coverage annotations, never information used to cancel
that day's decisions; the static lower-exposure scale must be calibrated on fit/selection
history and frozen before evaluation, with ex-post matched-gross attribution separately
labelled. No retrospective performance claim may be called a causal overlay test.
Meaningful-capital execution assumptions are required before implementability claims
or a read, while contractual event evidence and prospective validation precede deployment.
