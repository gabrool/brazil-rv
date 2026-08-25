# Experiment 47 — final bounded HPO + structural sweep

Preregistration-style instructions for a coding model in the Brazil-RV
repository. Freeze this section in `EXPERIMENT_LOG.md` (full text under
`research/preregistrations/`) before any trajectory exists. Discovery-only:
official validation and the held-out test are NOT accessed. Nothing here
changes the deployed Experiment-45 ten-seed recipe. This is the program's
one and only hyperparameter/architecture sweep: when it completes, the HPO
and architecture axes are closed permanently for this generation, whatever
the outcome. The active GH200 runs this program and is terminated and
verified absent afterward (the standing keep-alive override ends here).

## Shared contract

- All configs train on the store-v2 34-field specification, canonical
  optimizer/sampler/RNG contract except where a config's row says otherwise,
  20-epoch trajectories, fold monitors only.
- Decision comparator for Stage 2: the retained store-v2 3-seed fold
  comparator (Experiment-41 Stage-C, cross-fitted Raw Patience-3, folds
  C/A/B). Stage-1 comparisons are config-vs-C0 at matched protocol.
- Readouts: bidirectional odd/even cross-fitted Raw Patience-3 primary;
  final EMA-0.995 secondary; paired daily deltas; block-5/10 bootstraps;
  horizon/TOD guardrails.
- Every trajectory in both stages archives predictions in both states and is
  declared eligible for a future ensemble-pool registration
  (hyperparameter-jittered member family). Nothing is rescored as a pool
  here.

## The sixteen configs (frozen; no additions, substitutions, or edits)

Incumbent values: lr 3e-4, T 0.50, depth 6 (k3, dilations 1/2/4/8/16/32),
patch 5, wd 0.01, dropout = incumbent contract value, SAM ρ 0.125.
Only deviations from incumbent are listed. "Track" fixes each config's
Stage-2 gate in advance: SIMP = strictly simplifying (removes wd, dropout,
or blocks; no other change except ρ); VAL = value change.

| id | deviations from incumbent | receptive field check | track |
|---|---|---|---|
| C0 | none (control) | 127 ≥ 69 | — |
| S1 | wd 0, dropout 0 | 127 | SIMP |
| S2 | wd 0, dropout 0, ρ 0.25 | 127 | SIMP |
| S3 | wd 0, dropout 0, ρ 0.50 | 127 | SIMP |
| P1 | dropout 0 | 127 | SIMP |
| P2 | wd 0, ρ 0.25 | 127 | SIMP |
| R1 | depth 4, dilations 1/4/16/32 | 107 ≥ 69 | SIMP |
| R2 | depth 8: incumbent six blocks + two extra dilation-1 blocks | 131 | VAL |
| R3 | kernel 7 dense, dilations 1/1/2/2/4/4 | 85 ≥ 69 | VAL |
| R4 | patch 10 (35 tokens/session) | 127 ≥ 35 | VAL |
| R5 | lr 5e-4 | 127 | VAL |
| R6 | lr 2e-4 | 127 | VAL |
| R7 | T 1.00 | 127 | VAL |
| R8 | T 1.00, lr 5e-4, ρ 0.25 | 127 | VAL |
| R9 | kernel 7 dense, depth 4, dilations 1/2/4/8, lr 5e-4 | 91 ≥ 69 | VAL |
| R10 | patch 10, depth 4, dilations 1/4/8/16, T 1.00, wd 0.02 | 59 ≥ 35 | VAL |

Notes fixed in advance: width 64, head, fusion, loss family, batch, and
epoch budget are not swept; parameter-count changes from depth/kernel are
recorded and accepted; per-config temperature floor honors the recorded
prior evidence that T < 0.50 underperforms (no sub-0.50 cells). The
SAM-only arms (S1–S3) test the registered hypothesis that SAM subsumes
wd/dropout; the ρ escalation in S2/S3 is deliberate, since removing weight
decay plausibly requires a larger perturbation radius.

## Stage 1 — screen (seed 29 × folds B and C; 32 trajectories ≈ 6 GPU-h)

One trajectory per config per fold, fold-monitor Patience-3, cross-fitted
readout. Comparison: paired daily deltas vs C0's matched stage-1 runs on
each fold's selection window.

Advancement (frozen):

- **VAL configs** advance only if their delta is positive on BOTH folds.
- **SIMP configs** advance if their delta is ≥ −0.0005 on both folds
  (parity is their success condition; their adoption case is simplification).
- Cap: at most 3 configs advance. Slot priority: (1) the best VAL qualifier
  by two-fold mean, (2) the best SIMP qualifier by two-fold mean, (3) the
  next best qualifier of either track. If no config qualifies, the sweep
  ends here: the incumbent recipe is confirmed as the settled optimum at
  this measurement scale, and Stage 2 is skipped entirely.
- Stage-1 multiplicity is acknowledged: 15 single-seed comparisons imply an
  expected maximum spurious delta of roughly +0.003–0.005; stage-1 wins are
  screening evidence only, and Stage 2 exists to burn them down.

## Stage 2 — confirmation (≤ 3 configs × 9 trajectories ≈ 5 GPU-h)

Each advancing config: seeds 11/29/47 × folds C/A/B, standard readout,
paired against the store-v2 3-seed fold comparator.

Gates (frozen, per track):

- **VAL**: superiority — three-fold mean ≥ +0.001 AND every fold ≥ 0, with
  supporting intervals. A passer earns an arm in a future official read;
  nothing is adopted from fold evidence.
- **SIMP**: non-inferiority — three-fold mean ≥ 0 AND no fold < −0.0005.
  A passer becomes the **training-recipe specification for future members
  and the next generation** (the store-v2 precedent: spec adoption, not
  deployment). The deployed ten-seed recipe is untouched; any deployment
  change would ride a future read on its own results. If multiple SIMP
  configs pass, the simplest wins (fewest retained components; ties by
  higher mean).
- Ties or parity for VAL configs go to the incumbent — value churn at
  parity is not adopted.

## Reporting and hygiene

One immutable program root: this registration; all 32 + ≤27 trajectory
manifests and histories; both-state prediction archives for every
trajectory; stage-1 screen table (config × fold deltas); stage-2 full
analyses with intervals and guardrails; gate verdicts quoting rules
verbatim; the pool-eligibility declaration for all archived members;
SHA-256 for every artifact; `official_validation_accessed=false`,
`test_accessed=false`. Deletion-first cleanup after decisions are recorded:
retain all prediction archives, manifests, and analyses; retain epoch-20
and honest-selection checkpoints per trajectory (the Experiment-46
retention pattern); remove other epoch checkpoints under the reviewed
inventory procedure. Then terminate the paid instance and verify it absent
twice — no further keep-alive applies.

## Explicit non-goals

No 17th config, no second sweep round, no width/head/loss/batch axes, no
per-config seed extension beyond the frozen stages, no ensemble scoring of
archived members, no official or test access, no deployment change, no
post-score edits. A surprise here that suggests a new hypothesis goes to a
future registration.

## Honest priors, recorded in advance

Direct expected value ±0.001 — every adjacent axis the program has measured
has nulled, and this sweep exists to close the last inherited-default gap
honestly rather than to find treasure. The most likely positive outcome is
a SIMP adoption (SAM-only or no-dropout at parity), which would simplify
the recipe without moving IC. The guaranteed value is the 32-to-59-member
archive of hyperparameter-jittered trajectories — the raw material the
hyper-deep-ensembles evidence says is the best member-manufacturing axis —
and the permanent, evidence-based closure of the architecture question.

## Frozen implementation resolutions

These operational resolutions remove ambiguity before any trajectory or score
exists and do not add a candidate or change a gate:

- Patch-10 cells causally left-pad only the oldest edge of each available
  prefix to a multiple of ten minutes. They never consume the next five-minute
  interval. The 345-minute global history is likewise left-padded by five
  minutes, producing exactly 35 tokens.
- "Supporting intervals" for the VAL Stage-2 gate means that both pooled
  paired-daily block-5 and block-10 bootstrap lower 95% bounds are strictly
  above zero, matching the established Experiment-46 implementation.
- "Fewest retained components" ranks SIMP passers by residual-block count
  plus one for nonzero weight decay plus one for nonzero dropout. The frozen
  tie-break remains the higher three-fold mean.
- The cap slots are literal: best VAL, best SIMP, then the best remaining
  qualifier. An empty track-specific slot is not backfilled beyond the single
  third slot.
- Stage-2 trajectories are new, separately archived trajectories even where
  seed 29 on Fold B/C duplicates a Stage-1 protocol. No score or prediction is
  reused across stages.
