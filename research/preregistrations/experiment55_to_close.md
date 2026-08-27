# Experiment 55 — horizon-conditioned to-close head

Frozen after the OOF manufacture completes and before the first Experiment-55
trajectory or fold score. This is a discovery-only C/A/B screen. It cannot alter
the deployed Experiment-45 recipe, access official validation, or access the
permanently spent held-out test.

## Target and model contract

The immutable target sidecar is built only through `TRAIN_END`. For each standard
decision entry it uses the exact entry open and final session close. The return is
cross-sectionally median removed, divided by the causal per-name
`sigma*sqrt(H)`, where `H=405-entry` ranges from 390 to 120 minutes, then centered
midranked per date/decision. Exact endpoint observations, point-in-time membership,
data readiness, source identity, hashes, invalid-zeroing, and availability are
audited.

The candidate leaves the full-width incumbent trunk and three incumbent readouts
unchanged. Its fourth score uses three parallel full-width linear readouts and
the basis `[1, H/405, sqrt(H/405)]`. Those readouts are zero initialized without
advancing the parent's RNG, so the first three outputs and subsequent stochastic
state have exact epoch-zero parity. The loss is equal-weight four-head soft
Spearman at temperature `0.50`.

## Trajectories and readouts

The deployed store-v2 specification plus the new head runs on Folds C/A/B with
seeds `11/29/47`, 20 epochs, and the canonical standard readouts. The primary
candidate uses the ordinary bidirectional odd/even cross-fitted Patience replay
selected on the four-head objective. Final EMA-0.995 is secondary. At most two
training processes run.

## Frozen measurements and decision

The three incumbent heads are compared with the exact store-v2 comparator.
Guardrail passes only when the mean fold delta is nonnegative and no fold is below
`-0.0005`. The fourth head's IC is reported overall and separately for decisions
`0..17`, `18..36`, and `37..54`. Positive late-third IC with nonpositive morning
IC records the predeclared future fallback of two half-day bucketed heads; it does
not authorize a retry here.

Economics reuses Experiment 54's exact Fold-C-frozen state buckets, causal market
inputs, measured taker cost, threshold 7 bps, capacity, name cap, and gross
allocator. Four-head ranks form the event state. To-close next-open actions exit
at the exact final close. The conditional-edge and daily/fold frontier tables are
retained. Economics passes only if the to-close frontier exceeds the best existing
three-head Experiment-54 frontier by at least `+2` NAV bps/day on at least two
folds. The head is adopted as an execution-layer feature only when guardrail and
economics both pass. Deployment and the official metric remain unchanged.

If and only if adoption fires, the already frozen OOF protocol is extended with
50 four-head refits under identical five-fold, ten-seed, fixed-20, no-monitor,
final-EMA rules. Their archive must pass the same per-sample fit-exclusion loader.
Otherwise that branch is skipped. All predictions, analyses, manifests, targets,
gates, and audits are retained; checkpoints are hash-inventoried and deleted after
use. `official_validation_accessed=false` and `test_accessed=false` throughout.

