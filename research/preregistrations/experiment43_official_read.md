# Experiment 43 — official-validation read and conditional 10-seed expansion

Frozen before any new arm training and before opening any saved or new
official-validation prediction. The implementation under test is exact commit
`a441307f23cbc058f17fcbce5f102cb7a84d4c05`; this preregistration is the only
document-only successor allowed before launch. This is official-validation
access event 3, following the Experiment-1/14 event and Experiment 24. The
held-out test is not authorized and must remain sealed.

## Fixed inputs and hypotheses

- Canonical comparator: stored Experiment-1 parent reproduction at
  `parent_reproduction_4067962_e22dd67_20260819T131142Z`, seeds 11/29/47,
  official-monitor Raw Patience-3 (patience 3, minimum IC improvement `1e-4`,
  best raw checkpoint) and uniform tie-aware rank average. Reuse predictions;
  do not retrain.
- Challenger: the same parent-3 members plus the stored Experiment-24 residual
  auxiliary seeds 11/29/47 at final epoch-20 EMA-0.995 from
  `next_stage_official_921dd3a_20260821T085500Z`. The residual target sidecar is
  `auxiliary_targets/next_stage_3b60ac9_20260820T233000Z`, auxiliary loss weight
  is 0.5, and the ensemble is the uniform six-member tie-aware rank average.
  Reuse is permitted only if the manifest, commit, sidecar, 20-epoch, final-EMA,
  244-date alignment, and `test_accessed=false` contracts all pass.
- Store-v2 parent: the selected Experiment-41 prune-R2 mask from
  `feature_removal_stage_c_repair_d5b5e1f_20260823T232938Z`. Zero dynamic
  indices `9,11,14,22,24,25` and slow indices
  `1,2,3,12,13,14,15,16,18,20,22,23,24,25,26,27,28,29` in the loader from
  epoch zero. Train seeds 11/29/47 on all 716 training dates for 20 epochs,
  apply the matched official-monitor Raw Patience-3 rule, and uniformly
  rank-average the selected members.

The fold-evidence register is fixed at challenger `+0.0015` mean on A/B and
store-v2 `+0.0012` mean on C/A/B. These are two acknowledged hypothesis arms;
they are not recomputed or used to alter the rules here. Options, R3, hybrids,
new data, new folds, and additional arms are excluded.

## Official read and promotion

Evaluate each fixed arm against the identical canonical predictions on exactly
244 official-validation dates with strict observation alignment, primary IC,
paired per-date candidate-minus-canonical deltas, 10,000-replication moving
block bootstraps at lengths 5 and 10, horizon and time-of-day guardrails,
member ICs, pairwise prediction diversity, and ensemble gain versus mean and
best members. Interpret levels against Experiment 15 with no drift adjustment.

An arm is supported if and only if its paired block-10 95% interval versus
canonical has lower bound strictly greater than zero. If both arms are
supported, promote the arm with higher official ensemble IC. If exactly one is
supported, promote it. If neither is supported, canonical remains deployed
regardless of point estimates. There is no multiplicity adjustment and no
post-read change to arms, weights, composition, or thresholds.

## Conditional supplementary measurement

Skip this section entirely if neither arm is promoted. Otherwise append seeds
`61,79,97,113,131,149,167` in that fixed order to seeds 11/29/47.

- Challenger promotion: train seven parent trajectories with the matched
  official-monitor Patience rule and seven residual-auxiliary trajectories with
  the historical `921dd3a` model/optimizer contract, immutable sidecar, fixed 20
  epochs, EMA-0.995 updated each optimizer step, no official evaluation during
  training, and one final-EMA official prediction generation after fitting.
  Evaluate the uniform 20-member rank ensemble against the promoted six-member
  ensemble.
- Store-v2 promotion: train seven additional masked parents with the matched
  official-monitor Patience rule and evaluate the uniform 10-member rank
  ensemble against the promoted three-member ensemble.

Deploy the 10-seed form if and only if its official IC is at least the promoted
three-seed IC minus `0.0005`; otherwise deploy the three-seed form. Record the
10-by-10 seed-level prediction-correlation matrix and the fixed prefix ensemble
curve for seed counts 3 through 10. This is measurement of a frozen expansion,
not another hypothesis selection.

## Accounting, compute, retention, and termination

Use at most two simultaneous processes. Pre-read compute is exactly three new
store-v2 trajectories because the residual-3 final EMA archives are expected to
be reusable. Conditional compute is seven store-v2 trajectories or seven parent
plus seven fixed residual trajectories. No trajectory or prediction may touch
the held-out test.

The immutable result root must retain this preregistration copy, reused-source
hash inventory, every prediction archive, member manifests, both full arm
analyses, promotion decision, conditional supplementary analysis and seed
diagnostics when applicable, deployed-recipe declaration, validation-access
ledger, and `test_accessed=false` flags. After the declaration, only a reviewed
inventory-bound cleanup may remove redundant checkpoints; it must retain the
selected/final checkpoints of every deployed member. Record and push the result
before terminating the exact paid instance, then verify that exact instance is
absent twice.
