# Experiment 57 — frontier replay, imitation initialization, fine-tuned policy

Status: frozen before any Experiment-57 conditional mean, replay, clone, policy,
selection, or evaluation number exists.

## Access and deployment boundary

This program is discovery-only. Official validation is inaccessible and the
permanently spent held-out test must not be read. No stage changes the deployed
prediction or execution recipe. All policy inputs are provenance-proven
four-head OOF TRAIN predictions, and no policy is trained on in-sample model
predictions.

## Stage 0 — cross-fold frontier replay

For each evaluation rotation `W` in `fold_c`, `fold_a`, and `fold_b`, estimate
the four-head conditional gross-edge mean for every Experiment-54 frozen state
cell and ordered horizon (`30m`, `60m`, `120m`, `to_close`) by pooling raw events
from the other two evaluation folds. An evaluation or policy-training event
whose state/horizon cell is absent from those two estimation folds has no
estimate and cannot trade. The expected net edge is the pooled cell mean less
that event's own measured all-in taker cost.

At an eligible refresh, choose the single horizon with greatest expected net
edge; exact ties follow the frozen horizon order above. A variant qualifies the
event only when its expected net edge is strictly greater than its threshold.
The variants are exactly `0.0`, `4.5`, and `7.0` bps. A name already assigned an
active horizon ignores new events until that horizon elapses; at exact expiry it
may accept the current refresh. The action formed at minute `t` enters on the
next observed open, and the assigned lock lasts through action minute `t+h` for
a fixed horizon. A to-close assignment remains active through the simulator's
registered close taper and terminal flattening.

The event's score sign supplies direction. Its raw magnitude is its current
per-name cap. The primary book passes those signed cap-sized scores through the
existing bounded dollar-neutral projection, which never scales gross up. The
diagnostic neutrality-free book clips the same scores to the same name caps and
scales them down pro rata only when their gross exceeds 2.0. Both books use the
exact Experiment-52/53 simulator machinery: next-open fills, 10% minute
participation and carry, lagged measured spreads, fees, CDI on margin-adjusted
cash, close taper, and flat-by-close.

For a rotation, threshold designation uses only the two estimation folds. Each
variant is replayed on those same estimation-fold dates with the pooled
estimation table; the greatest mean daily primary-book net excess over all-cash
CDI wins, and an exact tie chooses the lower threshold. This in-estimation
threshold choice is permitted because the held-out rotation alone supplies the
headline cross-fold result. All three variants and both book constructions are
still reported on the held-out rotation.

Readouts are daily net excess over all-cash CDI per rotation and pooled,
moving-block-10 intervals, oracle capture against the matched Experiment-56
Section-A frontier, turnover/day, mean deployed gross, deployment fraction,
per-liquidity-tercile PnL and trade share, per-session-third turnover, per-trade
realized gross edge and cost, horizon usage, target-change versus spread, and
paired deltas versus the exact C0, C1, and all-cash daily ledgers.

The primary selected-threshold rule is the teacher for the next stages. It
passes the repaired economic graduation bar only if pooled mean daily net excess
over all-cash CDI is strictly positive and pooled mean deployed gross is at least
`0.05 * 2.0 = 0.10` of NAV.

## Stage 1 — imitation initialization

For every rotation, materialize the primary teacher's exact target-weight path
over all pre-window dates ending before the five-session embargo. The same
two-fold teacher table and Stage-0-selected threshold are used. The partial
overlap between those table folds and policy-training dates is an accepted
initialization approximation; fine-tuning and evaluation remain chronological
and held out.

The pre-window dates use the Experiment-56 split unchanged: the last `floor(20%)`
are selection-only and all earlier dates are clone-fit dates. For seeds
`11`, `29`, and `47`, initialize the unchanged four-head `NeuralPolicy` and
behavior-clone for exactly 20 epochs. Each epoch minimizes the mean squared
error between the student's post-projection weights and the teacher weights over
all tradeable day/name/action observations in the fit scan. The replay state
follows the teacher path, so supervised features are the states actually visited
by the teacher. Optimization is AdamW with learning rate `0.001`, weight decay
`0.01`, gradient clipping `1.0`, no SAM, and no early-stopping gate.

Report flat weight correlation on all selection observations and separately
where the teacher target is nonzero, selection replay net PnL/excess for clone
and teacher, clone-minus-teacher paired differences, MSE history, gradient norm,
and clone/teacher deployment. Clone quality is diagnostic only; all nine clones
proceed to Stage 2.

## Stage 2 — full-cost fine-tuning

For the same nine rotation/seed cells, load the exact Stage-1 clone and optimize
the unchanged registered bps objective

`mean(daily net excess over all-cash CDI) - 0.02 * population_std(daily net PnL)`.

Use the chronological fit slice, full costs from epoch zero, AdamW learning rate
`0.001`, weight decay `0.01`, gradient clipping `1.0`, no SAM, patience 10 on the
selection slice, and a 100-epoch ceiling. The best fine-tuned epoch is compared
with its own untouched clone on the selection objective. Designate the
fine-tuned checkpoint unless its objective is strictly below the clone's; an
exact tie designates the fine-tuned checkpoint. Seeds are replicas and are
averaged by date for held-out evaluation.

Evaluate teacher, clone, best fine-tuned, and designated policies on each held-
out rotation under identical full costs. Report the Stage-0 ladder readouts plus
training objective, gradient, and deployed-gross trajectories from the cloned
start. The designated neural policy graduates only if its pooled mean daily net
excess over all-cash CDI is strictly positive and its pooled mean deployed gross
is at least 0.10 of NAV. If the teacher graduates and the neural policy does not,
or if both graduate but the neural pooled excess does not strictly exceed the
teacher's, the teacher is the standing execution candidate. Otherwise the
graduating neural policy stands. No graduation changes deployment.

## Budget, retention, and termination

Stage 0 is CPU. Stage 1 has exactly nine clone runs and Stage 2 exactly nine
fine-tuning runs. At most two training processes may run concurrently on one
paid GH200 instance. Each stage freezes into a fresh immutable commit-bound root
and retains manifests, input/config hashes, conditional tables, teacher plans and
paths, all histories, all daily results, all readouts and diagnostics, decisions,
and every designated prospective checkpoint. Operational failures may be
repaired only before an affected score exists and only without changing this
research contract; failed roots and logs remain recorded.

Every result and audit must state `official_validation_accessed=false` and
`test_accessed=false`. After all artifacts and operational logs are hash-audited,
the durable outcome is recorded, code and records are committed and pushed, and
local main matches GitHub. Only then may the exact paid instance used by this
program be terminated; its exact ID must be absent on two provider inventory
reads. No adjacent instance may be terminated.

