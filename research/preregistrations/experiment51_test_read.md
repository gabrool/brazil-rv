# Experiment 51 — the held-out test read

Preregistration instructions for a coding model in the Brazil-RV repository.
Freeze this section in `EXPERIMENT_LOG.md` (full text under
`research/preregistrations/`) before any test-period target or prediction is
opened. This is the program's FIRST and ONLY access to the held-out test
split — a single measurement event, authorized by this registration alone.
After it, the test period is spent: no future experiment may read it again,
and any hypothesis this read generates can only be validated on new forward
data.

## Purpose and explicit scope limit

Measure, once, the out-of-sample predictive skill (IC) of the deployed
recipe on the sealed test period, and its staleness structure. **This read
measures model skill only. No execution metrics — no Sharpe, no costs, no
turnover, no portfolio simulation — are computed in this event; the
execution layer is a separate later program.** No model, recipe, weighting,
or deployment change may result from this read; its outputs are
calibration.

## The measured object (frozen; nothing else may be evaluated)

The deployed Experiment-45 recipe, exactly as declared: the ten retained
member checkpoints (selected states), store-v2 34-field inputs, uniform
tie-aware rank averaging across the ten members. Before any test date is
opened, hash-verify every checkpoint and manifest against the Experiment-45
deployed-recipe declaration. No retraining, no re-selection, no new
trajectory — inference only. No comparator arm exists in this event.

## Data

The sealed test split: 2025-07-07 → 2026-07-17, 259 dates, from the
immutable store. Targets constructed by the standard pipeline under the
standard availability/mutation audits. Test dates sit ~12–25 months after
the training window's end (2024-06-28) — that distance is itself one of the
measurements.

## Measurements (all predeclared; no additions after any number is seen)

1. **The number:** mean per-date primary IC over the 259 test dates —
   per-date cross-sectional Spearman averaged over decisions, dates, and
   the three horizons, exactly as the official metric — with block-5 and
   block-10 bootstrap intervals (10,000 reps) on the daily series.
2. **Per-horizon ICs** (30/60/120m) with intervals.
3. **Staleness profile:** quarterly IC means across the test year; the
   slope of daily IC on days-since-train-end; the H1-vs-H2 paired
   difference with a block-10 bootstrap interval. This is the measurement
   that decides whether a live deployment should retrain first.
4. **Period-difficulty context:** cross-sectional dispersion, per-name
   vol levels, and active-universe size per quarter, reported alongside the
   same statistics for the official-validation year — so a level difference
   can be attributed to period difficulty versus skill before anyone
   reaches for a story.
5. **TOD guardrail table**; monthly IC series for the record; member ICs,
   member correlation, and ensemble-vs-mean-member gain (informational).

## Predeclared expectations and interpretation (IC terms only)

Recorded before the read: official-validation IC of the deployed recipe is
0.043719 across five access events of selection pressure; the honest point
expectation for the test year at comparable difficulty is ≈ 0.041–0.043,
with period-difficulty variation of ±0.003 entirely plausible (fold windows
have ranged 0.038–0.051 on the same recipe family).

- **Band A — test IC ≥ 0.040:** official-period skill confirmed out of
  sample; the selection-pressure discount was small.
- **Band B — 0.035 to 0.040:** skill real but discounted; consult the
  staleness slope and difficulty context (measurements 3–4) before
  attribution.
- **Band C — 0.030 to 0.035:** material degradation; a written attribution
  analysis (staleness vs regime vs selection bias) is required before any
  live planning, using only measurements 3–4 — no new test analyses.
- **Band D — < 0.030:** the official-period edge did not generalize;
  program-level reassessment.
- **Staleness rule:** if the H1-vs-H2 difference is negative with its
  interval excluding zero, a retrain-before-live policy is indicated; if
  flat (as the Experiment-15 validation-year profile was), the static
  recipe stands and this read's number is conservative relative to a
  freshly retrained live model.

The bands are interpretation discipline, not decisions: deployment sizing,
execution design, and go-live are later programs that will take this
read's outputs as inputs.

## Accounting and hygiene

This event's ledger entry records `test_accessed=true` — the first and
only such entry; every prior artifact's `test_accessed=false` remains
truthful history. One immutable read root: this registration; checkpoint
verification records; every test prediction archive; all analyses;
the interpretation-band statement quoting this section verbatim; SHA-256
for every artifact; result-log commit and push. Nothing deployed is
touched; no checkpoint cleanup applies to deployed members. Inference and
analysis only — negligible GPU; terminate any paid instance afterward and
verify absent twice.

## Explicit non-goals

No execution metrics of any kind (per the program owner's instruction —
Sharpe, costs, turnover, and books come later); no candidate comparisons;
no retraining or fine-tuning; no recipe or deployment change; no second
test read ever under any registration; no post-score analysis additions;
no store changes. A surprise here generates hypotheses for forward data
only.

## Honest priors, recorded in advance

Band B is the modal expectation: the fold→official record shows evidence
shrinks at each boundary, and this is one more boundary — plus a year of
staleness the official read never carried. A Band-A result would say the
program's governance priced its own selection bias correctly; a Band-C
result would most likely be staleness or regime, and measurements 3–4 are
predeclared precisely so that question gets answered by numbers already in
the read rather than by a new dip into spent data.
