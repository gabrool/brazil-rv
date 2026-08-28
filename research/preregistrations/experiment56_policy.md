# Experiment 56 — four-head OOF and learned execution policy

Status: frozen before any Experiment 56 outcome is computed.

## Access and deployment boundary

This program is discovery-only. Official validation and the permanently spent held-out test are inaccessible to every loader and stage. It does not change the deployed Experiment-45 prediction recipe or authorize deployment. All policy training uses provenance-proven out-of-fold TRAIN predictions.

## Section A — standing decision and abort gate

The Experiment-55 to-close head is adopted as an execution-layer input on its own registered IC and incremental economic frontier. The historical Experiment-55 result remains unchanged; its prior-prediction guardrail is void only as a decision rule for adding an execution input.

Using the retained Experiment-55 cross-fit archives, form the exact three-member rank-average candidate independently on Folds C/A/B. Rebuild the Experiment-54 state buckets, measured taker costs, and capacity allocator. Measure next-open 30m, 60m, and 120m outcomes plus the registered to-close outcome. State-conditional expectations are estimated only within the same discovery fold.

The total frontier permits one action per name/refresh. For each event choose the horizon with the largest state-conditional expected net edge among 30m, 60m, 120m, and to-close; deterministic ties follow that order. A chosen action is eligible only when its expected gross edge is strictly greater than 7 bps and its expected net edge is positive. The Experiment-54 allocator then ranks eligible events by expected net edge and applies the frozen name, participation, gross, fee, and measured-spread assumptions. The three-head comparator is the best registered Experiment-54 threshold-7 frontier in that fold. Abort Sections B/C only if the four-head total is below the comparator on at least two folds. Report individual and total frontiers, conditional tables, and the retained Experiment-55 to-close IC by session third.

## Section B — four-head OOF manufacture

Unless Section A aborts, train exactly 50 trajectories: the canonical five purged TRAIN folds crossed with seeds 11, 29, 47, 61, 79, 97, 113, 131, 149, and 167. Use store-v2, the Experiment-55 to-close target, exactly 20 epochs, no monitor, final EMA 0.995, and an archived raw epoch-20 prediction. Run at most two training processes. Materialize and loader-verify one provenance-proven 716-date OOF rank archive. Report, without selection, per C/A/B slice and head: four-head first-three versus base-three rank correlation and IC, plus to-close IC. Delete checkpoints only after every archive and manifest hash passes.

## Section C — rotated policy training

Use the verified four-head OOF archive, Experiment-52 market/cost/CDI construction, PolicyState, NeuralPolicy, PolicyTrainer, and bounded projection. The execution configuration is canonical except `name_cap_fraction_of_gross=0.05`, `gross_target=2.0`, and four ordered horizons `(30m,60m,120m,to_close)`.

For each C/A/B evaluation window, eligible pre-window dates end before the five immediately preceding sessions. The last `floor(n/5)` eligible dates are the selection slice and all earlier eligible dates are fit. Fit and selection are chronological and disjoint; no evaluation-window value reaches either. Each run uses AdamW with learning rate 0.001, weight decay 0.01, gradient clipping 1.0, no SAM, and at most 100 epochs. After every fit epoch, evaluate the selection objective; patience is 10 strictly non-improving epochs and the best checkpoint is restored.

The exact objective, in bps of NAV, is:

`mean(daily net PnL - all-cash CDI PnL) - lambda * population_std(daily net PnL)`.

The fixed grid is lambdas 0.02 and 0.10 crossed with seeds 11, 29, and 47 and windows C/A/B: 18 runs. For each window designate the lambda with the higher mean best-selection objective across all three seeds; an exact tie chooses 0.02. Seeds are replicas, not a selection surface. Retain the three designated-lambda checkpoints per window and remove the other verified checkpoints.

For reporting and the graduation decision, average designated-policy daily results across the three seeds by trade date, then pool the three non-overlapping evaluation windows. Graduate as the standing execution candidate if and only if pooled mean daily net excess over all-cash CDI is strictly positive. No other readout vetoes it.

Report per-window and pooled daily net excess and net Sharpe; paired deltas versus C0, C1, and all-cash under identical dates/costs; oracle capture versus Section A; turnover and deployment; liquidity-tercile trade share and net PnL; per-trade edge versus cost; target-change versus spread at refresh; one-at-a-time inference ablation of each horizon rank and rank-change feature pair; time-of-day fills; objective curves, gradient norms, and gross paths from the all-cash initialization. Mean and paired readouts use a 10-session moving-block bootstrap with 10,000 replications and fixed seed 20260856. Sharpe intervals use the identical resampled blocks.

## Retention and termination

Each section has a separate immutable commit-bound root with input hashes, configs, manifests, outputs, and access flags. Preserve all readouts, diagnostics, decisions, histories, and designated checkpoints. After all three sections and operational logs are hash-audited and pushed, terminate only the exact paid instance and verify its ID absent twice.
