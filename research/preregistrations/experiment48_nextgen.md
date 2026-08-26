# Experiment 48 — next-generation spec: R1+T combination and the 15-minute head

Preregistration-style instructions for a coding model in the Brazil-RV
repository. Freeze this section in `EXPERIMENT_LOG.md` (full text under
`research/preregistrations/`) before any score exists. Discovery-only:
official validation and the held-out test are NOT accessed anywhere —
Part A runs on discovery-fold archives, not official data. Nothing here
changes the deployed Experiment-45 ten-seed recipe. Purpose: finalize the
next-generation training-recipe specification (base architecture ×
temperature × horizon set) so the endgame — a possible validation read
event 5 and the single held-out test read — is run on a settled candidate.

Standing context: Experiment 47 adopted R1 (depth 4, dilations 1/4/16/32,
store-v2 fields) as the training-recipe spec via the simplification track,
and confirmed T=1.00 as a real ≈+0.0005 effect (both pooled intervals above
zero) on the depth-6 base — refused there under the value-churn rule, whose
materiality bar is now recognized as conflating read-arm earning with spec
adoption. The depth-4 × T=1.00 combination has never been trained; this
experiment tests it prospectively rather than assuming additivity.

## Part A — leg decomposition diagnostic (zero GPU; gates Part C)

Question on record: a claim from outside the program asserts most of the
30-minute return's predictable movement is realized early in the window.
Verify in our own data before spending GPU on a 15m horizon.

1. Data: the retained store-v2 fold comparator predictions (Experiment-41
   Stage-C, cross-fitted Raw Patience-3, folds C/A/B, fold selection windows
   only) paired with leg returns computed from the raw 1-minute archive.
   No official-validation data is touched; no new model evaluation runs.
2. Construction: split each 30-minute label window into legs r(0→15) and
   r(15→30). Apply the exact target construction to each leg independently —
   per-name vol normalization with the leg's own √H scaling, cross-sectional
   median removal, per-(date, decision) midranks.
3. Measurement: per fold, the 30m head's prediction IC against each leg's
   midranks, with block-10 bootstrap intervals on the paired daily
   difference (leg-1 IC minus leg-2 IC). Report the early-realization share
   IC(leg1)/[IC(leg1)+IC(leg2)]. Informational secondary: the 60m head
   against r(0→15).
4. **Gate for Part C (frozen): proceed iff leg-1 IC ≥ leg-2 IC on at least
   2 of 3 folds.** If the gate fails, Part C is skipped entirely, the claim
   is recorded as not replicated in-house, and Parts A–B still complete.

## Part B — R1 + T=1.00 combination (9 trajectories ≈ 1.5 GPU-h)

1. Candidate: the R1 spec with soft-Spearman temperature 1.00 — depth 4,
   dilations 1/4/16/32, store-v2 34 fields, all other contract values
   incumbent (lr 3e-4, wd 0.01, incumbent dropout, ρ 0.125, patch 5).
2. Seeds 11/29/47 × folds C/A/B, 20-epoch trajectories, fold monitors,
   standard readout (cross-fitted Raw Patience-3 primary, final EMA-0.995
   secondary, both states archived).
3. Comparison: paired daily deltas versus the archived Experiment-47 R1
   Stage-2 trajectories (reused unchanged — R1 is not retrained), block-5/10
   bootstraps, horizon/TOD guardrails.
4. **Decision (frozen):** the combination becomes the next-generation
   training-recipe spec iff non-inferior to R1 — three-fold mean ≥ 0 AND no
   fold < −0.0005. T=1.00's independently confirmed positive effect is the
   recorded justification for preferring the combination at parity. If it
   fails, the spec remains R1 at T=0.50 and the temperature effect is
   recorded as base-dependent. Whichever wins is the **Part-B winner** and
   the base for Part C.

## Part C — the 15-minute head (target sidecar + 9 trajectories ≈ 2 GPU-h)

Runs only if Part A's gate passed.

1. **Target sidecar.** 15-minute-horizon targets under the identical
   construction as the three incumbent horizons: vol-normalized (σ√H,
   H=15 min), cross-sectionally median-removed, per-(date, decision)
   midranks. Development-only window ending 2025-06-30, exactly as prior
   target sidecars; the held-out test period is never touched. Immutable,
   hashed, availability/mutation audited before any training run.
2. **Candidate.** The Part-B winner spec plus a fourth output head at 15m,
   trained with the equal-weight joint soft-Spearman loss over four horizons
   (per-horizon weight moves from 1/3 to 1/4; this re-weighting of the
   incumbent horizons is an accepted, recorded property of the design).
   Standard head initialization; no other change. Seeds 11/29/47 × folds
   C/A/B, standard protocol and readouts.
3. **Primary gate (frozen; the official metric's meaning is untouched):**
   the candidate's *3-horizon official-metric* fold IC, paired against the
   Part-B winner, must be non-inferior — three-fold mean ≥ 0 AND no fold
   < −0.0005. Per-horizon guardrails reported so any drain on 120m is
   visible.
4. **The new measurement (always reported when Part C runs):** the 15m
   head's own fold IC per fold with intervals, cross-fitted, alongside the
   30m head's IC for scale, plus its TOD profile. This is the program's
   first 15-minute measurement; expectation from the horizon gradient
   (0.046/0.041/0.039 at 30/60/120m) is at-or-above the 30m level, stated
   in advance as a calibration, not a gate.
5. **Adoption semantics (frozen):** if the primary gate passes, the
   four-head model becomes the next-generation spec (store-v2 fields,
   Part-B base, four horizons). If additionally the 3-horizon delta shows
   superiority (mean ≥ +0.001, every fold ≥ 0, supported intervals), that
   is recorded as read-worthy evidence for a future validation read. If the
   gate fails, the spec stays three-head, the 15m standalone IC is still
   recorded, and a dedicated single-purpose 15m model is noted as a future
   option — not built here.
6. Deployment context, recorded not gated: 15m trading roughly doubles
   turnover, so deployable value depends on alpha-per-trade clearing
   half-spreads plus fees at small capital; that pricing belongs to the
   cost/capacity analysis before any live use of the 15m output.

## Sequencing note (registered intent, not executed here)

If Experiment 48 yields a next-generation candidate with read-worthy
evidence, validation read event 5 (candidate vs deployed ten-seed) is the
next registration; the single held-out test read comes last, spent on the
recipe that will actually be traded. Neither is authorized by this document.

## Reporting and hygiene

One immutable program root: this registration; Part-A decomposition tables
and gate verdict; the 15m target sidecar with audits (if built); all 9+9
trajectory manifests, histories, and both-state prediction archives; all
paired analyses with intervals and guardrails; decisions quoting rules
verbatim; the final next-generation spec declaration; SHA-256 for every
artifact; `official_validation_accessed=false`, `test_accessed=false`.
Cleanup per the Experiment-46/47 retention pattern (all predictions,
manifests, analyses retained; epoch-20 plus honest-selection checkpoints
retained; other epoch checkpoints removed under reviewed inventory). The
paid instance is terminated and verified absent twice at completion.

## Explicit non-goals

No pool scoring, no additional horizons beyond 15m, no specialist or
single-horizon variants, no loss-weight tuning beyond the fixed equal
weights, no official-validation or test access, no deployment change, no
read registration, no post-score edits. Surprises go to future
registrations.

## Honest priors, recorded in advance

Part A: the leg-decomposition gate is a genuine unknown — either outcome is
informative (a fail kills the 15m thesis cheaply; a pass quantifies early
realization for the first time). Part B: the combination most likely
inherits both effects (~R1 + ~+0.0005) but interaction risk is real; the
fallback costs nothing. Part C: multi-task at a fourth horizon is aligned
with the program's one working multi-horizon result (shared head beat
specialists), so non-inferiority is the likely outcome, with the 15m IC
level itself as the true prize — it prices the first new tradeable output
the program has produced, and at ~2× more independent observations per day
it will be the program's most precisely measured horizon.
