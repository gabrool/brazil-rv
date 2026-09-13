# Round 7 temporal pathway: implementation and engineering evidence

This is an extension to the running Round 7, not a replacement or a financial
result. The original checkout and experiment remain frozen at `8e00a2a`.
The extension runs from its own checkout, root and compatible Stage P parents.
[The registration](../research/preregistrations/v2_round7_pathway.md) defines
the comparison and records the engineering amendment described below.

## Implemented comparison

| Cell | Full 60-session encoder | Peer attention placement | Parameters |
|---|---|---|---:|
| GL | GRU, width 64 | After learned temporal pooling | 1,492,643 |
| GE | Same GRU | Before the same temporal pooling | 1,492,643 |
| TL | Historical-window attention, width 64 | After pooling | 1,501,283 |
| TE | Same historical-window attention | Before pooling | 1,501,283 |

Historical attention is bidirectional within the information available at the
decision, with position encodings and a 128-wide feed-forward block. Peer
attention has four 16-wide heads, exact masked SDPA, residual/normalization and
shared weights across historical sessions. GRU states are restored to actual
calendar positions before peer interaction. Missing internal sessions are not
compressed. Current decision eligibility and the real history suffix mask
control keys and outputs. No stock identities are embedded as arbitrary slots.

Downstream B4 family networks, common-state FiLM, characteristic context and
three traded horizon heads remain unchanged. No shortened history, reduced
eligible population or extra observation delays. Early/late pairs have identical
parameter sets and initialization. This is a controlled topology experiment,
not an implementation claim of exact MASTER or HIGSTM replication.

## Synthetic findings and explicit acceptance amendment

The teacher supplies an observable two-group identity and an independent shock
at historical lag three. Each target is the mean shock of the other seven names
in its group. It consumes no financial target. Conditional cross-sectional rank
structure allows a negative-own-shock shortcut to obtain nonzero IC, so mere
positive IC does not demonstrate successful peer aggregation.

Initial full-model and isolated-component probes with inherited small `.02`
attention initialization failed the reference IC threshold. Xavier QKV/output
initialization allowed all four temporal-and-peer components, followed by a
linear diagnostic head, to learn the teacher on unseen dates in one controlled
run: GL .9722, GE .9944, TL .9932, TE .9921. The other B4 initializers stay intact.
A subsequent canonical initialization produced .9645/.9637/.4599/.8555. These
results show initialization sensitivity; they do not promise reliable learning
from every initialization.

Full-model SAM `.05` remained near .45 IC with soft Spearman on this task at
960 updates. A GL run using 4,800 updates and the registered warmup/cosine
schedule also remained at .4488. Matched short full-model AdamW diagnostics
learned it: GL .9044 and GE .9819. MSE/SAM probes also failed. This establishes
optimizer sensitivity for this artificial task; it does not establish that SAM
is harmful on Brazilian equity labels or identify its unique failure mechanism.
All completed probe results and source hashes are in
[the evidence JSON](v2_round7_pathway_engineering.json). Original scripts and
complete operational artifacts remain in the external audit root for recovery.

After observing these results, before any extension financial fit, the initial
IC-threshold acceptance interpretation was amended explicitly: the unseen-date
teacher is a mandatory diagnostic with zero advancement weight, not a binary
veto. No failed probe is called passed. Structural invariants, causality, finite
gradients, exact recovery and the existing full-model CUDA precision, compilation
and synthetic-fit requirements remain mandatory. The financial factorial keeps
R/SAM `.05`. A negative market result must be described as conditional on that
recipe and budget, not a capacity theorem or evidence that peer information is
absent. This change avoids substituting an artificial task for the registered
market-data comparison; the amendment itself must be visible to reviewers.

## Verification and experiment execution

Targeted checks cover exact GRU calendar restoration, stock permutation,
padding/nonfinite-mask invariance, date-batch isolation, finite gradients,
one dynamic full training graph and exact interrupted GE/TE resume including
averaged weights and scores. CPU diagnostic tests protect future-mutation
isolation, exclusion of the subject from peer means, fixed training-only scaling
and explicit missing values. Decision tests protect complete early/late pair
advancement, the original designation fallback, six-seed agreement and Holm's
two-test primary family.

At the implementation freeze, 36 targeted tests pass and Ruff passes. CUDA
acceptance is a separate required phase and is not implied by this CPU result.

The extension first requires four CUDA engineering cases on sixteen real F14
dates with the full 60-session window. It then runs 12 fresh P fits and a matched
48-fit screen, inheriting B from the original B4 calibration. Complete fit
throughput and peak memory, not only attention tensor estimates, control
concurrency. Confirmation and fresh seeds follow the fixed advancement rules.
The original result and extension outcome remain distinguishable in the combined
review. No financial extension result is available at this checkpoint.

Supporting peer ridge diagnostics use only admitted decision-causal channels,
PIT sectors, prior-return correlation clusters and stock-specific interactions
with leader/ADR-gap aggregates. The exact narrower support and chronology are
registered. OOF nuisance controls begin only where genuinely historical OOF S0
forecasts exist; unavailable early controls are not replaced by in-sample scores.
These diagnostics cannot promote or veto a neural cell.
