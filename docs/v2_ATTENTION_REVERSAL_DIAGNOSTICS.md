# Wider-attention reversal: investigation in progress

The current evidence does not establish that an accounting correction erased the
model's edge, or that overfitting explains the reversal. The September 21 user
request makes this investigation the priority; the unstarted LSTM wave is deferred.
Resolve the `scaling_*` entries in `v2_economic_data_scaling_run.json` for exact
plans, executed recipes, saved controls and output hashes.

## Why the original seven-bps result is a different comparison

The earlier C6 flexible-net account earned 6.70751 bps/day **above CDI**, with
zero-rate Sharpe 1.88147 over 1,738 continuous 2018–2024 sessions. It allowed a
45% net cap; the current screen uses 5% and four separate half-year accounts.
On those same four periods C6 was already much weaker before corrections.
[Performance comparability](v2_PERFORMANCE_COMPARABILITY.md) retains both
benchmarks and account definitions. There is no corrected continuous flexible
result proving that the 6.71 either survives or disappears.

Four folds describe evaluation, not training. Original and corrected attention
fits use the same expanding histories: 1,377 P training sessions, then 419, 912,
1,409 and 1,907 F training sessions for F2/F6/F10/F14, with separate 55-session
checkpoint selection and ten-session purges. The new diagnostic design fixes
F2/F3/F6/F7/F10/F11/F13/F14, reserving six other development folds from new
corrected comparisons. Earlier research already used development years, so these
reserves are not pristine historical holdouts. No 2025/2026 consumer is opened.

## What changed economically

Thirty original/corrected attention fit pairs preserve graph, optimizer recipe,
training dates and selector. The expected changes are store, conditioning,
compatible parent weights and code provenance. The actual consumer changes route
dated security history; they do not shorten the 60-session history or reduce the
population. Previously qualified source and tensor proofs are reused.

A separate saved-contract check confirms that the two widths consume identical
stores, conditioning, target windows, access windows and name padding in all
30 original/corrected matched slots. Only hidden width64→96 differs in their
model configuration. Forty parent conditioning fields change across the data
correction, in the financial/event/sector/magnitude/cross-market families; the
largest scale-ratio change is about9.77%, not an orders-of-magnitude unit jump.
This is descriptive evidence, not a fresh source audit or proof that every
assumption is correct. Code inspection retains width-derived four-head scaled
dot-product attention, explicit key/output masking and fit-only conditioning.

A frozen 24-book decomposition holds the corrected account and new portfolio
inputs fixed while separately exchanging forecasts and allocation risk. For
wider attention, the mean four-fold forecast effect is -1.55229 bps/day; the
other-input effect is +0.01514, the risk effect +0.01244, and their interaction
+0.02159. The largest fold is F10: old forecasts with fixed new inputs/old risk
earn +1.43619 above CDI; new forecasts under those same inputs/risk earn -4.83950.
These are matched diagnostic contrasts, not independent additive observations.

The F10 difference mainly comes from stock positions. Gross stock contribution
changes -6.92253 bps/day, partly offset by +0.56618 from the hedge. Turnover falls
from .15883 to .14116 daily NAV and cost differences are small. Mean gross
exposure is almost unchanged, while mean absolute stock-weight distance is
0.76473 of NAV. A .96247 rank correlation can coexist with materially different
constrained portfolios. The largest ordinary mark/fill losses include CURY,
ANIM, GOAU4, ARML and BPAN; corporate transfer legs are not treated as standalone
stock returns. Cash/claim residuals are reported separately. This attribution
does not identify a single source repair as the cause of changed forecasts.

## Numerical checks and their limits

All 24 corrected attention F checkpoints reproduce their own saved forecasts
within the unchanged eager/compiled tolerance across all evaluation dates.
Across original and corrected checkpoints, 3,219,606 valid coordinates were
compared. Two original TE_wide seed29 fits retain four tolerance crossings:
UNIP on September20 2018, BPAC on October3 2018, and TASA on November26 and
December15 2020. All scores are finite; the full verification result remains
false because the tolerance was not relaxed. Two additional ensemble books,
using all three original eager exports instead of compiled exports, change
F2/F6 returns by only +.00683/+.00216 bps/day. Those bounded precision effects
do not explain the multi-bps refit reversal.

Both corrected F10/seed29 attention widths also pass an actual compiled two-pass
ASAM update with normal/repeated/NaN/zero/one temporary allocations. Identical
weights, optimizer state and RNG give exact resulting model states and identical
loss/gradient-norm readouts. This covers one actual training batch shape per
width, not every historical training kernel. No diagnostic update is adopted.

## Specific stopping mechanism under test

Original-data wider-attention P seed29 ran 25 epochs and selected epoch20.
Corrected-data P seed29 stopped after nine and selected epoch4. At epoch7 the
original trajectory improved validation enough to reset patience; the corrected
one narrowly did not. The original later improved substantially. Corrected F10
seed29 subsequently earns -9.13486 above CDI versus +8.00419 for its original-data
forecast under corrected accounts. The other two seeds also decline less.

The new one-factor diagnostic changes only that corrected parent's patience
from five to twenty, preserving the maximum60/schedule60, original selector and
all data/model/optimizer settings. Its F10 child keeps the original F recipe.
An exact epoch9 resume was impossible because sealed intermediate epochs omit
optimizer/RNG state; a fresh trajectory was necessary. All nine prefix epochs,
14,170,203 model-state cells and all history fields except timing match exactly.
The runtime commit differs only through an unused C6 head-expression change and
account code outside this training path; the source diff is preserved. The child
and a three-seed ensemble replacing only seed29 are evaluated against saved
controls. This is an outcome-informed mechanism diagnostic, not an unbiased
confirmation or automatic model adoption.

The parent completed51epochs and selected31 with validation IC .0345219,
compared with the corrected control's epoch4/.0289792. Its child completed11
epochs and selected6/.0776904 versus the control's epoch3/.0682388. At R$10m in
F10, with the same corrected data and account:

| Forecast | Original corrected fit, above-CDI bps/day | Extra parent patience | Difference |
|---|---:|---:|---:|
| Seed29 | -9.13486 | -2.08706 | +7.04780 |
| Ensemble, only seed29 replaced | -4.78314 | -2.41903 | +2.36412 |

Both new books pass independent saved-component NAV arithmetic within
R$3.73e-9 and have no overdue principal. This demonstrates material sensitivity
to parent stopping in the selected worst fold; it does not restore the original
seed's +8.00419 or ensemble's +1.43619, nor establish a generally better stopping
rule. The original model results remain unchanged. Parent/child fit wall times
are recorded separately from account evaluation and are not future fit estimates.

The eight-period comparison is now running with the original corrected parents
and recipes for both attention widths. It reuses6parents/24existing children and
48portfolio books, adds only24children and48primary books for the four new
periods, and keeps the patience diagnostic separate. More periods measure the
stability of the existing comparison; they do not certify a retrospectively
chosen winner. Further changes require their own explicit controlled contrast.

## Attempts, recovery and storage

All numerical controls and original results remain. The decomposition's first
case saved arrays before a missing scenario label stopped report generation;
that one case was repeated for its absent detailed readouts, with initial arrays
retained. The forecast verifier's two initial failures concerned JSON tuple/list
comparison and junction-path aliases, both before a new model forward. Existing
successful exports were reused. The first prefix qualifier rejected the changed
git identity before comparing weights; a bounded source diff resolves the unused
C6/account changes, and the nine weight comparisons then pass. No fit reran due
to that qualifier. The training probe's post-execution lint correction changes
only closure cleanup; its exact executed recipe is retained.

The user's cleanup permission removed disposable compiler caches and358
redundant intermediate depth checkpoint copies, then182 native GRU96 width
checkpoint copies. Each checkpoint matched its decompressed member in the
corresponding verified archive before deletion. The width operation skips the
TE128 directory junction. Selected
weights, histories, forecasts, books, original fits and immutable sources remain.
Restore intermediate depth files from the recorded archive member/hash map when
needed. C free space rose from about33MB before cleanup to about2.53GB, before
the new probe rebuilt its cache. Earlier physical relocation and lossless runtime
compression remain recorded separately; one later native-library compression
attempt failed for lack of space and is not described as completed.
