# Experiment 43 — Amendment A1: two-track gate and store-v2 reproduction

Frozen after the Experiment-43 official-validation result and before any
Experiment-44 score. This is an explicit user-authorized governance amendment;
it does not rewrite the historical Experiment-43 verdict under its original
gate.

## Historical result

Under the original Experiment-43 support rule, neither the residual challenger
nor the 34-field store-v2 arm was promoted because each block-10 interval
included zero. The canonical 58-field parent therefore remained the recorded
deployment decision at that time.

## Refined two-track gate

Future frozen decisions distinguish complexity-reducing and complexity-adding
candidates.

- A complexity-reducing candidate may be accepted as non-inferior when its
  point estimate is non-negative, every preregistered horizon is non-negative,
  and the lower endpoints of both paired block-5 and block-10 95% intervals are
  at least `-0.0005`.
- A complexity-adding candidate requires superiority: held-out mean delta at
  least `+0.001`, every held-out fold non-negative, and paired block-5 and
  block-10 intervals reported in support of the decision.

Store-v2 is complexity-reducing. Its already-observed Experiment-43 point
estimate was `+0.001595530`, so it becomes the standing comparator only after
one exact three-seed full-window reproduction verifies the frozen
Experiment-41 prune-R2 specification and reproduces the prior trajectories
without retuning. The reproduction may read only the already-consumed official
monitor window needed for exact verification; it creates no new candidate,
selection rule, official decision, or additional read. The held-out test stays
sealed.

All Experiment-44 E1/E2 candidates add ensemble members and therefore use only
the complexity-adding superiority gate. Experiment 44 cannot change deployment;
at most one passing composition may be registered for a future official read.

