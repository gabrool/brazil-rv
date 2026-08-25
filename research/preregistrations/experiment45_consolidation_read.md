# Experiment 45 — consolidation read (official-validation access event 4)

Preregistration instructions for a coding model in the Brazil-RV repository.
Freeze this section in `EXPERIMENT_LOG.md` (full text under
`research/preregistrations/`) before any arm member is trained or any official
prediction is opened. One read event; the held-out test is NOT accessed and is
NOT authorized. This registration completes Amendment A1's intent (store-v2
deployment) through fresh measured members rather than a reproduction-tolerance
patch, and it consumes the two earned candidates: the Experiment-43 10-seed
transfer and the Experiment-44 `e2_plus_archive` arm.

## Root-cause fix, stated first

Experiment 43 deleted the measured arm's checkpoints, which forced Amendment
A1's reproduction attempt, which failed on GPU nondeterminism. Standing rule
from this read onward: **the deployed recipe's member checkpoints (selected
and final states) are retained until superseded by a future deployment**, and
every arm evaluated in a read retains its member checkpoints until the read's
deployment declaration is complete. The deployed artifact must always be the
measured artifact.

## Baselines (no new training)

- **Comparator for all paired analyses: the retained Experiment-43 store-v2
  3-seed prediction archives** (official IC 0.043235373). Manifest-verify
  before use.
- Canonical 58-field parent-3 archived predictions reported as a reference
  column only; no decision attaches to it. Its non-inferiority relation to
  store-v2 is already on the official record (Exp 43 + Amendment A1's
  two-track protocol) and is not re-litigated.

## Arm 1 — store-v2 at 10 seeds (deployment arm)

1. Train ten full-716-date store-v2 trajectories: seeds 11/29/47 and
   61/79/97/113/131/149/167 (the Exp-43 frozen expansion seeds), 34-field
   prune-R2 loader spec, official-monitor Raw Patience-3 matched exactly to
   the Exp-43 Arm-2 protocol. Each training run is logged in the
   validation-access ledger.
2. Evaluate the fresh 3-seed (11/29/47) ensemble and the 10-seed ensemble on
   official validation with the standard analyzer (paired daily deltas vs the
   comparator, block-5/10 bootstrap 10k reps, horizon/TOD guardrails, member
   ICs, seed-correlation matrix, the 3→10 ensemble-gain curve).
3. **Reproduction sanity band (informational guard, not a gate):** the fresh
   3-seed official IC is expected within ±0.0015 of the archived 0.043235. If
   it deviates by more, complete all measurements, HALT every deployment
   declaration, and report — that deviation would exceed nondeterminism and
   requires review.
4. **Deployment rule (frozen, from the Exp-43 registration):** the 10-seed
   form deploys if its official IC ≥ (fresh 3-seed IC − 0.0005); otherwise the
   fresh 3-seed form deploys. Either way the deployed members' checkpoints are
   retained (see root-cause fix) and the deployed-recipe declaration is
   issued. This implements the store-v2 deployment already authorized on
   non-inferiority grounds; it is not a new superiority claim.

## Arm 2 — `e2_plus_archive` consensus composition (superiority arm)

1. **Consensus rule, defined blind.** This rule was written without reading
   the three frozen fold-specific compositions, and must be applied exactly as
   stated: for every member (family, seed, state) appearing in any of the
   three compositions, compute its total repeat count across the three;
   include every member with total ≥ 2; its weight = its total repeat count;
   the three comparator (store-v2 seed) members are always included, each with
   weight = max(its total, 1). Cap the non-comparator members at the 16
   highest totals (ties broken by higher mean recorded held-out marginal gain
   in the greedy paths, then lexicographic family-seed-state order).
   Renormalize weights; combination is weighted tie-aware rank averaging per
   (date, decision, horizon) group, per-horizon membership respected for
   specialists. If no non-comparator member reaches total ≥ 2, the arm is
   withdrawn and recorded.
2. **Member realization at full window.** Store-v2 members: reuse Arm 1's
   fresh trajectories (matching seeds). E2 members: retrain on the full
   716-date window per their recorded Experiment-44 specifications — bagged
   members resample the full window with the same block-20 rule and the
   recorded deterministic seed derivation; subspace members use their recorded
   masks; specialists their recorded single-horizon configuration. Members
   whose composition state is Raw Patience-3 use the official monitor
   (matched protocol, ledger-logged); members whose state is final EMA-0.995
   train fixed-20-epoch with no monitor. Archive-family members: reuse
   retained full-window artifacts where manifests verify (e.g., the Exp-43
   residual final-EMA members, the canonical parent members); otherwise
   retrain per their recorded specification. Hard cap: if realizing the
   composition requires more than 20 new trajectories beyond Arm 1, trim the
   composition by ascending total repeat count until it fits, and record the
   trim.
3. **Gate (frozen; complexity-adding → superiority):** supported iff the
   paired block-10 95% interval versus the comparator excludes zero from
   above. Point estimates never override intervals.
4. **Promotion rule (frozen):** if Arm 2 is supported, the deployed recipe is
   the Arm-2 composition as registered here (with its fresh member
   realizations, checkpoints retained), and Arm 1's result is recorded and
   its deployment rule is superseded for this event. If Arm 2 is not
   supported, Arm 1's deployment rule stands and Arm 2's full results are
   recorded. No hybrid (e.g., 10-seed parents inside the composition) may be
   evaluated — compositions without fold evidence are out of scope.

## Read protocol and accounting

Standard analyzer for every evaluation; strict observation alignment;
interpretation against the Experiment-15 staleness profile. This is official-
validation access event 4: ledger entries for the event, both arms, and every
official-monitor training run. `test_accessed=false` everywhere; a test read
remains a separate future decision with its own preregistration. No second
read, no post-score edits, no additional arm, weighting, or composition after
any result is seen; a surprise that suggests a new hypothesis goes to fold
screening and a future registration.

## Outputs

Immutable read artifact: this registration; all member manifests, commits,
and training logs; every prediction archive; both arms' full paired analyses;
the sanity-band check; the promotion and deployment decisions quoting their
rules verbatim; the final deployed-recipe declaration (members, states, seeds,
weights, selection rules, ensemble method) with retained-checkpoint inventory;
the seed-correlation matrix and 3→10 gain curve for the program record;
SHA-256 for every output. Deletion-first cleanup afterward: retain all
prediction archives, all analyses, and — per the root-cause fix — the
checkpoints of every deployed member and of both arms until the deployment
declaration is pushed; then non-deployed arm checkpoints may be removed under
the reviewed inventory procedure.

## Budget

Arm 1: 10 trajectories ≈ 2 GPU-h. Arm 2: ≤ 20 new trajectories ≈ 4 GPU-h.
Analyses CPU. Session target ≤ 8 GPU-h including margin. Terminate and verify
absent the paid instance afterward.

## Expectations, recorded in advance

Arm 1: seed correlation ≈ 0.92 predicts the 10-seed form adds ≈ +0.0005 over
3 seeds; the deployment rule tolerates a null. Arm 2: held-out fold mean was
+0.00108; the program's one fold→official calibration point (the challenger's
+0.0015 → +0.00045) says shrinkage is likely; a supported interval is a
genuine coin-flip, and an unsupported-but-positive result would still leave
store-v2 deployed with the composition's evidence recorded for a future
bundled attempt. Either way this read ends with the leaner store-v2 recipe
deployed with retained checkpoints — the program's first deployment change
since Experiment 1 — and the read budget spent on decisions, not curiosity.
