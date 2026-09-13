# Round 7 pathway extension

Authorized by the user on 13 September 2026 after review of the MASTER/HIGSTM
assessment. Registered before any extension fit or selection of an architecture
from Round 7 outcomes. The original Round 7 roster, execution and decision remain
frozen in their original checkout and run root. Report original and extension
decisions separately in the combined review. No 2025/2026 consumer access,
forward capture or deployment. No new data collection or store transformation.

## Question and fixed factorial

Use the accepted repaired Round 7 store and B4 all-family input contract. Four
fresh graphs, each with its own Stage-P parent for every seed:

| Cell | Temporal encoder | Peer attention |
|---|---|---|
| GL | one-layer GRU, width 64 | after temporal pooling |
| GE | same GRU | at each retained session before pooling |
| TL | one temporal self-attention block, width 64 | after temporal pooling |
| TE | same temporal self-attention block | at each retained session before pooling |

All use 60 calendar sessions and all decision-eligible names. Same four-channel
core encoding (value, validity, age, age-known) and same width-64 input projection
and normalization. GRU states are restored to calendar positions before any peer
operation. Preserve missing sessions inside the real suffix. No recurrent updates
on artificial left padding before the first real state.

Temporal self-attention uses four heads, sinusoidal calendar positions, pre-norm
residual attention and a GELU feed-forward block with hidden width 128, dropout
0.1. Full attention inside the available historical window, masked for padding;
no triangular causal mask is needed for the final decision. Do not interpret its
intermediate states as predictions available at earlier sessions. Transformer
and GRU counts need not be identical; report counts and measured cost. Do not
reduce lookback or capacity to obtain an artificial runtime match.

Peer block is shared across all times: pre-norm exact SDPA, width 64, four heads,
output projection, residual and normalization, dropout 0.1. Early peer attention
batches date x session; late attention batches date. Keys require decision-active
names and a real history suffix at that session; invalid queries are zeroed.
An interior missing observation remains an encoded missing observation, not
left padding. The current eligible cohort can use its known historical data.

Every cell uses identical learned last-step-query temporal pooling with a
bias-free 64x64 map and valid-session softmax. Late cells pool then apply the peer
block; early cells apply the same peer block then pool. All downstream B4 core
MLP, family encoders, joint projection, FiLM, pooled context, residual trunk and
D3/D5/D10 heads are unchanged. This compares placement within the core history
pathway, not replacement of the downstream characteristic-context mechanism.

## Training, reuse and engineering

Use Round 7 R recipe, SAM rho .05, existing learning rate/decay/dropout/precision,
unique-date batching and uniform tail averaging. Fresh Stage P: 60 epochs per
cell/seed. Fine-tuning inherits the B selected by the original B4 calibration;
extension evaluation cannot alter B or select epochs. Record clean-fit and
selection curves; an inadequate budget qualifies a negative result rather than
authorizing outcome-driven retuning. Seeds 11/29/47. Screen F2/F6/F10/F14:
12 fresh P fits and 48 F fits. Reuse exact compatible completed fits only.

Before fitting: calendar restoration versus current GRU summary; stock
permutation and padding invariance; all-missing finite zeros; real store
future-mutation isolation; meaningful synthetic peer-timing learning on unseen
examples with the peer identity defined by supplied information; BF16/FP32 loss
and gradient bridge; dynamic full-graph training/evaluation; exact interrupted
fit recovery. Synthetic tests do not assert that B4 is unable to learn a target.
Measure complete-step memory/throughput on the same sixteen full F14 dates and
then complete fit throughput. Use fused SDPA, shared encoders and bounded
concurrency. No real fit until acceptance. Changes to numerical execution need
documented engineering evidence and a fresh code freeze.

## Contrasts and interpretation

Prespecified pathway contrasts GE-GL and TE-TL; encoder contrasts TL-GL and
TE-GE; interaction (TE-TL)-(GE-GL). Also each versus B4, B9 and the original
Round 7 designated model, using existing exact scores where available. Average
seeds using the same registered forecast/rank ensemble convention before daily
primary IC. Report seed and fold sensitivity, paired daily IC and economic
differences with NW lag-10 SE and date-block bootstrap intervals. The two pathway
contrasts are the primary family; use Holm correction for their nominal tests.
Report the interaction and other comparisons as secondary, with multiplicity
and selection disclosed. Point thresholds are practical screening rules, not
statistical resolution or proof of absence.

To control cost, advance an encoder's complete early/late pair if either member
exceeds B4 by .002 mean screen primary IC with positive deltas in at least three
of four folds. Also advance the other encoder pair if its best member exceeds
B4 and is within .002 of the highest cell. Only apply that second rule when a
first pair qualifies. Thus 0, 2 or 4 cells receive ten additional folds (0/60/120
F fits); reuse the screen folds. Complete B4's remaining folds if original Round
7 has not already done so. No zero-signal CPU veto. A stopped screen is reported
as lack of demonstrated benefit under the tested budget, never falsification of
all peer mechanisms. A confidence interval whose upper bound is below .002 may
exclude that material gain only for the actual tested contrast and evaluation.

After confirmation freeze the highest-IC extension cell, but recommend replacing
the original designation only if paired full-panel IC gain is positive, the
nominal paired 95% interval excludes zero, every leave-one-seed-out gain is
positive and existing economic acceptance passes without a negative net-excess
point estimate relative to that designation. Otherwise retain original and report
uncertainty. A qualifying replacement receives three fresh seeds 61/79/97 for
itself and the original designation (reuse any exact existing fits). If it is an
early cell, additionally retain its matched late control for the six-seed pathway
audit. Freeze the candidate before these seeds; no runner-up shopping. The
six-seed comparison must retain the positive primary and economic point signs
and positive leave-one-seed-out gains, otherwise retain the original designation.
These are development-research decisions, not independent held-out validation.

## Supporting CPU diagnostic

Zero promotion weight. Use admitted prior-only residual-return peer means at
lags 1–5: leave-one-out correlation-cluster and sector groups. Market-wide leader
and ADR aggregates alone cannot rank stocks; multiply them by prespecified
stock-specific prior-only exposures. Fit-only scaling and ridge regularization
chosen on internal Stage P chronological splits, with label maturity/purges.
Compare a chronological S0-score control with score plus peer features, and
report standalone and conventional partial correlations (residualize both sides),
with dependence-aware uncertainty. OOF S0 training scores must be genuinely
out-of-fold and mature; if unavailable for early dates, report unavailable rather
than fitting the nuisance control using evaluation labels or in-sample scores.

Use repeated blocked/randomized controls with an explicit null construction;
never require a future-shifted feature to have zero IC. No negative linear result
can rule out nonlinear relationships, alternative peer definitions or trajectories.
No fundamental-family trajectory or macro-token architecture change is included.

## Operational integration

Develop in a separate git worktree and use a separate remote checkout/run root;
share the immutable store and compiler cache when safe. Never update the checkout
running original Round 7. Queue the extension on the same paid host after original
work, or overlap only after measured capacity permits without disrupting it.
Bind original run, calibration, accepted store, code and registration hashes in
the extension manifest. Recover/hash both programs and write one combined LLM
review before terminating the exact instance; preserve the original designation
and extension evidence as separately identifiable results.
