# Experiment 50 — official-validation event 5

Frozen before any Experiment-49 number exists, but realized only after the
immutable Experiment-49 KEEP/DROP verdict is written. The held-out test remains
sealed. This document authorizes one official-validation access event and no
second read.

## Conditional realization

Experiment 49 fixes the head count before event 5 begins: four heads
(15/30/60/120) after KEEP, otherwise three heads (30/60/120). This choice is not
revisited using official results. With four heads, the existing hash-verified
Experiment-48 15-minute target sidecar is used; the fourth head is retained in
all archives but excluded by the existing three-horizon primary monitor.

## Sole candidate arm

Train exactly ten members with seeds
`11,29,47,61,79,97,113,131,149,167`: store-v2 34 retained fields, R1 depth four
with dilations `(1,4,16,32)`, soft-rank temperature `1.00`, official Raw
Patience-3 monitoring matched to Experiment 45, and uniform tie-aware rank
averaging. At most two training processes may run concurrently. No other arm,
seed, weighting, composition, head weighting, or post-score edit is permitted.

The sole comparator is the deployed Experiment-45 ten-seed store-v2 Raw
Patience-3 prediction archive, with every source manifest and required
prediction/checkpoint hash verified before access.

## Frozen decision

Compare candidate and comparator on the official 30/60/120 metric. DEPLOY the
candidate iff the paired moving-block bootstrap with block length 10 has a 95%
lower bound on candidate-minus-comparator daily IC of at least `-0.0005`,
inclusive. Record superiority when the same interval's lower bound is strictly
positive, but do not require it. If the gate fails, Experiment 45 remains
deployed. A fourth head's official 15-minute IC is reported as decision-neutral.

## Access, retention, and shutdown

Write the event-5 ledger entry before the first official training evaluation.
Every run and analysis records `official_validation_accessed=true` and
`test_accessed=false`. Retain all member manifests, histories, prediction
archives, analyses, and selected/final checkpoints through deployment. After the
declaration, retain deployed measured members' selected/final checkpoints until
superseded. Cleanup is deletion-first and limited to a reviewed exact inventory.

After final audit, experiment log update, commit, and push, terminate the exact
paid instance used for the run and verify that exact ID absent in provider
inventory twice. No held-out-test read is authorized by this registration.

