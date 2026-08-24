# Experiment 44 — ensemble-science program, stage 1 (E1 + E2)

Preregistration-style instructions for a coding model in the Brazil-RV
repository. Freeze this section in `EXPERIMENT_LOG.md` (full text under
`research/preregistrations/`) before any combination score or trajectory
exists. One paid-instance session. Official validation and the held-out test
are NOT accessed anywhere in this program — all training monitors and all
readouts use discovery-fold windows only. A gate pass here earns an arm in the
next official-read registration, never deployment.

## Standing context (post-Amendment A1)

- **Comparator everywhere: the store-v2 3-seed fold ensemble** — the
  Experiment-41 Stage-C prune-R2 trajectories (34 fields, 24 removals zeroed
  in loader), folds C/A/B, seeds 11/29/47, honestly-selected cross-fitted Raw
  Patience-3 checkpoints, uniform tie-aware rank average. The 58-field parent
  is a *member candidate* in E1, never the comparator.
- Amendment A1's deployment reproduction (3 full-window store-v2 trajectories
  + exact-match verification) is a separate task; if not yet executed, run it
  first in the same session (~45 GPU-min). Nothing below depends on it.
- Two-track gate protocol applies. Every candidate in this program is
  **complexity-adding** (more members), so the gate is superiority:
  held-out mean delta ≥ +0.001 AND every held-out fold ≥ 0, supported by
  paired block-5/10 bootstrap intervals on daily deltas.
- Canonical RNG/optimizer/sampler contract, immutable artifacts, SHA-256
  manifests, `official_validation_accessed=false`, `test_accessed=false`,
  deletion-first cleanup, exactly as in Experiments 39–43.

## Stage 0 — inventory and freeze (no scores computed)

1. Inventory every retained fold prediction archive: 58-field parent members
   (Patience and EMA states), residual-aux and aux-bundle members, the ten
   Exp-28–37 dataset-adapter member sets (both states), Exp-40/42 options
   members (OI, opt-full, opt-iv), Exp-41 prune-R1/R2 members, Exp-19–23
   Phase-C candidates where archived, and the Exp-38 Kronos K0 score arrays.
2. Eligibility: an archive enters the roster only if it passes manifest/hash
   verification and covers a fold's full selection window under strict
   observation alignment. **Primary roster** = members with all three folds
   (C/A/B); **secondary roster** = members with A/B only (used only in the
   supplementary A↔B rotation).
3. Freeze by hash before any score: the member roster (with state labels —
   a member is a {family, fold set, seed, state} tuple), the E1 rule grid
   (§E1), the E2 bag seeds and subspace masks (§E2, deterministic derivation
   recorded), and this document.

## E1 — combination-rule re-analysis (CPU only; runs before/while E2 trains)

All combining is done in rank space: within each (date, decision, horizon)
group, average the members' tie-aware ranks (weighted where specified).
Paired daily deltas vs the comparator; block-5/10 bootstrap, 10k reps.

1. **Diversity measurement (deliverable regardless of outcomes).** Per fold
   and state: pairwise member–member and member–comparator prediction
   Spearman within groups, averaged — the program's first measured c matrix
   across families. Report alongside each family's standalone fold IC.
2. **Greedy forward selection with cross-fold honesty (primary).**
   Initialize with the three comparator members. Candidates: the primary
   roster, selection *with replacement* (repeats act as integer weights).
   Each step adds the member maximizing the selection-window IC of the
   rank-average; stop when marginal improvement ≤ 0 or at 12 additions.
   Rotation: select on {A,B} → evaluate on C; {A,C} → B; {B,C} → A.
   Selection and evaluation never share a fold. The gate is applied to the
   three held-out deltas. Supplementary: A↔B rotation including the
   secondary roster (reported, but cannot alone open a gate).
3. **Fixed-weight grids (secondary).** Comparator-ensemble weight
   w ∈ {0.5, 0.6, 0.7, 0.8, 0.9}, remainder uniform over predeclared sets:
   (a) residual-EMA-3; (b) residual-EMA-3 + options-OI members;
   (c) the five adapter-EMA families with the best original-screen EMA
   deltas; (d) the 58-field parent-3 (generation mixing); (e) full primary
   roster. All 25 cells reported on all folds; a cell advances only if it
   passes the gate on the three folds jointly; tie-break = highest mean;
   25-cell multiplicity recorded.
4. **Hygiene rules (parameter-light, reported on all folds):** shrunken-skill
   weights w_i ∝ max(0, c̄ + 0.5·(IC_i − c̄)) over the primary roster;
   per-group median-of-member-ranks; 20% trimmed mean of ranks.
5. **Forbidden:** covariance-/regression-optimal weights estimated from fold
   dates, any learned meta-combiner, any rule not in the frozen grid, and any
   post-score addition of rules or member sets.
6. **Advancement rule:** at most ONE E1 composition (the gate-passing
   composition with the highest held-out mean; greedy outranks grid at equal
   mean) is named a read-arm candidate. If none passes, E1 concludes
   "uniform weighting was not the binding constraint at the current pool"
   — itself decisive.

## E2 — member manufacturing (GPU; 45 trajectories ≈ 8.5 h)

All members: store-v2 34-field spec, folds C/A/B, seeds 11/29/47, one
20-epoch SAM trajectory each, canonical contract. Training monitors use the
fold selection window only. Two states archived per member: cross-fitted Raw
Patience-3 (primary, protocol-matched to the comparator) and final EMA-0.995
(secondary). All fold predictions archived for both states.

- **E2a — date-block bagged members (9 trajectories).** Per member, resample
  the fold's *fit-window* dates: consecutive blocks of 20 trading dates drawn
  with replacement until reaching the original fit length, truncated exactly;
  the date-stratified sampler runs over the resampled multiset (repeated
  dates appear with their multiplicity). Selection windows untouched. Bag
  seed derived deterministically from (family, fold, seed) and recorded.
- **E2b — feature-subspace members (9 trajectories).** Per member, zero 8
  additional uniformly-drawn surviving fields in the loader (on top of the 24
  store-v2 removals; ~76% of the 34 kept), mask drawn deterministically from
  (family, fold, seed) and recorded in the manifest. Same mechanism as
  Exp-39 F3 / Exp-41 — no store rebuild.
- **E2c — per-horizon specialists (27 trajectories).** Single-horizon head
  and loss (30m-only / 60m-only / 120m-only), all else unchanged. A
  specialist contributes ranks only to its own horizon's groups; rank
  averaging per group uses whichever members cover that group (tie-aware, as
  the masking machinery already does).

Readout philosophy, fixed in advance: E2 members are NOT expected to beat the
comparator standalone — bagged and subspace members are *expected* to be
0–0.002 weaker individually. The decision readout is the E1 machinery applied
to the enlarged pool:

1. Rerun E1.2 greedy (same rotations, same stopping rule) over
   {comparator members + all E2 members, both states}; then over
   {comparator + E2 + primary archive roster} (the full pool).
2. Rerun E1.3-style fixed grids with predeclared sets: comparator +
   {E2a-9}, + {E2b-9}, + {E2c-27}, + {all E2}, at the same five weights.
3. Informational, always reported: standalone member ICs; uniform
   comparator+family stacks; within-family and family-to-comparator c
   (the direct test of memo v3's premise that manufactured diversity reaches
   c ≈ 0.75–0.85 against seed-only c ≈ 0.92).
4. **Advancement rule:** same as E1.6 — at most one gate-passing composition
   from the enlarged pool is named a read-arm candidate; if both E1 and E2
   produce one, the higher held-out mean advances and the other is recorded.

## Outputs

One immutable program root containing: the frozen roster/grids/seeds/masks;
the c matrices (archive pool, E2 families, combined); every rule's full
per-fold results with intervals; greedy selection paths (order, repeats,
marginal gains); all gate verdicts quoting the rules verbatim; the named
read-arm candidate (composition, members, states, weights) or the recorded
null; E2 member manifests and both-state prediction archives;
`official_validation_accessed=false`, `test_accessed=false`; the
validation-access ledger untouched. Cleanup after decisions are recorded:
retain all prediction archives, manifests, and analyses; delete E2 member
checkpoints after prediction-archive verification (a future read arm retrains
its members on the full window from the recorded specifications — fold
checkpoints are not needed downstream).

## Explicit non-goals

No E3 items (distillation, GBDT, histogram members), no HPO sweep, no new
data or features, no changes to member states beyond the two predeclared, no
covariance weights, no deployment change, no official-validation or test
access, no second composition advanced, and no post-score edits of any
frozen list. If a surprise here suggests a new hypothesis, it goes into a
future registration.

## Budget and expectations, recorded in advance

E1: CPU/inference only. E2: 45 trajectories ≈ 8.5 GPU-h; total session
target ≤ 11 GPU-h including margin and (if pending) the Amendment-A1
reproduction. Honest priors: E1 alone +0.000–0.002 held-out (and even a null
closes the uniform-weights question); E2's value arrives through measured
decorrelation — the target is family-to-comparator c ≤ 0.85 at member skill
within ~0.002 of the comparator's members, which the ensemble-gain arithmetic
converts to +0.001–0.003 for the combined recipe. The decisive products are
the c matrices, the gate verdicts, and at most one read-arm candidate for the
next official-read registration (where it would sit alongside the 10-seed
store-v2 expansion arm).

