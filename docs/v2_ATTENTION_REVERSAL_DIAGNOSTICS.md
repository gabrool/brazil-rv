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

The actual F3 source-book identical-intention check found an additional concrete
accounting defect, before the newly introduced cash revision was reached. On
January8 2019 a roundoff hedge trade of aboutR4.66e-10 created another R10 minimum
loan fee in the independent account. The first four NAV comparisons differed
only by normal floating-point error. A bounded five-day charge capture attributes
the discrepancy to the hedge root, rather than the corporate event. Initial
parity failure, first-divergence diagnostic and charge capture remain exact saved
recipes and outputs. The later opposite-direction spot guard no longer fires
when both accounts follow the corrected intentions; the guard was not relaxed.

The correction plan was frozen before corrected outcomes. Both accounts suppress
only hedge changes within eight Float64 epsilons of their target/current scale;
there is no absolute currency/quantity floor. Six targeted no-trade, genuine tiny
opening/partial/reversal/terminal and gradient cases pass, as do37 affected account
and loan-notice cases. Twelve separate F3 corrected books pass saved NAV checks;
their actual identical-intention comparison agrees withinR1.863e-8 across all1464
account days. AtR10m the full/wide ensemble final-path changes are+.15514/+.18274bp
over122sessions, not daily returns. All capital/seed contrasts remain saved.
An initial qualifier failed on a missing plan-wrapper key after all twelve books
were saved; only the qualification wrapper changed, and no book was repeated.

A read-only inventory of the16 original four-fold R10m source/refit ensembles
found no tiny hedge fills inF2/F6 and only sub-R1e-11 total tiny-root fees per book
inF10/F14, after the old minimum-fee regime. This is direct-fee attribution, not
an all-path adaptive bound; it supplies no reason to repeat their broad economic
matrix and does not explain the multi-bps reversal. The unfinished eight-fold baseline
uses a preserved pre-correction source snapshot, and GPU fits keep their isolated
training runtime. This defect is not yet a demonstrated explanation for the wider
model's reversal.

The added2019 one-factor funding/delivery bounds completed36 books with six
unexposed bonus variants skipped. All saved-account checks pass. AtR10m,
debit+100annualbp changes full/wide daily returns by-.03994/-.03982bps;
the tested delivery/disposal changes range from-.00845 to+.03266bps/day.
These source sensitivities retain their original pre-hedge-correction books,
explicit payment/valuation assumptions and separate numerical uncertainty.
They are not joint or all-interior bounds. Newly exposed2021 source leads remain
separate; candidate original notices do not yet make the F7 account accepted.

The expanded 2019 period exposed held Fibria, Guararapes PN and QGEP transitions.
Ten selected issuer originals and fifteen readable pages supply the dated terms;
this is bounded source recovery, not a repeated source census. Fibria cash was
announced as50.12 and revised to50.20 before payment. Both accounts now retain
original entitlement units when repricing pending cash at the later knowledge
date, even after delivery/disposal of the shares. Three new targeted cases and
29 affected cases pass; earlier empty-revision terms retain their original branch.

Twelve separate F3 source-only books reuse identical forecasts, static policy
coordinates and allocation risks. At R10m the ensemble earns4.43000/full and
3.13913/wide bps/day aboveCDI, changes of approximately+.195/+.171. These small
changes do not explain the earlier multi-bps attention reversal. Independent
saved-account arithmetic and Decimal entitlements/pending-cash reconstruction
pass; the latter checks36 locked quantities,24 arrivals,12 split quantities and
1464 daily receivables/payables each. The maximum cash discrepancy isR5.83e-11.
Fibria Jan8 closing custody enters the Jan9 decision, and Guararapes May6 closing
bonus enters May7. Guararapes PN delivery remains an explicit hypothesis, and
the Fibria pending-cash mark uses the last announced amount before revision.
These books are not observed client execution or a complete sensitivity bound.

The originals also disprove two accepted model-data action scalars: Guararapes
May2 2019 is an8-for-1 split, and QGEP April22 is q1 plus dividend1.90684828735,
paidMay7. The separate account overlay corrects them without changing the
accepted model store or current fits. Their remaining feature/target implications
must be attributed separately; the eight-period comparison cannot be called
globally corrected data. Current F3 comparisons use the same frozen imperfect
model-data contract for both widths. No new architecture or training rule follows
from these source findings.

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
