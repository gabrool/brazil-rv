# Wider-attention reversal: evidence and remaining causal questions

The eight-period comparison is complete. Wider attention remains weaker under
the current corrected-data recipe, with substantial uncertainty. The strongest
demonstrated partial mechanism is much earlier parent stopping: changing only
patience recovers part of the worst-fold loss. A real hedge roundoff defect is
fixed, but its measured effect is far too small to explain that reversal. There
is no evidence here establishing a general inability to scale attention, nor a
complete single-cause explanation. The unstarted LSTM wave remains deferred.
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

The eight-period comparison completed with the original corrected parents and
recipes for both widths. It reuses6parents/24existing children and48portfolio
books, adds24children and48primary books, and keeps the patience diagnostic
separate. All96 saved primary books qualify. The24 new fit times sum to6676.52s;
the48 new account times sum to92.31s. These are measured runtimes, not estimates.

## Expanded chronological comparison

The fixed eight periods contain993 distinct evaluation sessions, up from501.
F1/F4/F5/F8/F9/F12 remain reserved from new corrected comparisons; historical
development reuse and the untouched2025/2026 consumer boundary remain explicit.
Each account starts independently. These are not a continuous seven-year account.

R$10m, three-seed ensembles, equal period weights, bps/day **above CDI**:

| Evaluation contract | Full attention64 | Wider attention96 | Wide minus full |
|---|---:|---:|---:|
| Original four-period corrected-data recipe | .99921 | -.14375 | -1.14295 |
| Same recipe over all eight periods | 2.97320 | 1.93671 | -1.03648 |
| Eight periods with separate added-source account overlays | 3.01618 | 1.92001 | -1.09617 |

The last row changes only the source/account treatment in F3/F7/F11; forecasts,
model inputs, risk, training and calibration remain fixed. The hedge roundoff
diagnostic and the extra-parent-patience result are not silently folded into it.
The R$1m/R$5m source-overlay means are2.89433/3.00519 for full and1.81806/1.91094
for wide. Small capital sensitivities do not reverse this comparison.

| Period | Full, with source overlay | Wide, with source overlay |
|---|---:|---:|
| 2018 Jul–Dec | -.68461 | -.50311 |
| 2019 Jan–Jun | 4.43000 | 3.13913 |
| 2020 Jul–Dec | -7.19915 | -5.93836 |
| 2021 Jan–Jun | 3.18393 | 5.45372 |
| 2022 Jul–Dec | .97481 | -4.78314 |
| 2023 Jan–Jun | 8.26325 | 5.85459 |
| 2024 Jan–Jun | 4.25544 | 1.48765 |
| 2024 Jul–Dec | 10.90578 | 10.64963 |

Wide wins3/8 periods. The source-overlay paired40-session block interval for
the pooled-day difference is[-2.69949,+.52282]bps/day at95%, with estimate-1.10723.
The20/60-session intervals also cross zero. The baseline's three seed-level
equal-period differences are-.47445/-.84944/-.68445, while the ensemble differs
by-1.03648; nonlinear portfolio construction means these need not average.
Intervals preserve period boundaries and remain nominal development uncertainty.
There is no adoption or claim that the narrower architecture is inherently superior.

The baseline report includes all three currency-consistent Sharpes, drawdowns,
turnover, winning/losing days, holding spells, individual seeds and funded exposure.
Mean period Sharpes are not the Sharpe of a continuous account. F13 has no
unpriced inventory, overdue principal or loan-cash redemption; both ensembles
have121 funded sessions. Existing debit-spread mechanics are qualified; a separate
F13 adaptive spread contrast has not been run, and earlier-period endpoint results
are not claimed as its numerical bound.

Validation rank correlation is the checkpoint-selection criterion, not the
economic portfolio objective or an independent final test. In the original
four-period comparison, mean selected IC rises .04206→.04338 for full and
.04522→.04629 for wide despite the wider model's return decline. A higher selected
validation statistic therefore does not imply better portfolio returns, and does
not by itself rule out selection noise or overfitting. Neither explanation is
assumed to be the sole cause. The exact stopping-prefix experiment supplies more
specific evidence than that general possibility.

Remaining source limits matter: the accepted model store has not yet incorporated
the newly evidenced GUAR/QGEP/GPC/Wiz/RLOG/Smiles data dependencies. The source
overlay retains the unvalued Linx BDR/final-cash gap in F7 and admitted unknown
Smiles/ENAT fractional claims in F7/F14. This comparison must not be called fully
source-corrected. The next justified training contrast is a matched test of parent
stopping across both widths and the fixed seeds/periods, with no retrospective
adoption of the single favorable seed29 diagnostic. That contrast is not yet run.

The first summary saved all96 book records, then failed JSON serialization of a
NumPy integer fold count. The qualified summary casts only that report count,
reuses the exact saved records and recomputes short summaries/intervals; no fit,
forecast or account book was repeated. Initial code/stdout/records remain saved.

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
They are not joint or all-interior bounds. The separately qualified2021/2023
overlay below resolves specific held transitions; F7 retains explicit gaps.

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

The added2021/2023 source overlay uses eight originals and thirteen visually
qualified pages, including GPC/Wiz meeting approvals after their conditional
notices. RLOG is one CSAN per3.943112 RLOG, with expressly permitted March8 owned
disposal and March10 physical credit. Printed93.72 fraction cash is knownMarch26
and paidMarch29. Smiles' non-election default is .6601GOLL plus5.11719919cash;
its elective alternative cannot be selected with hindsight. June9 custody and
June23 cash remain distinct; unknown fraction auction stays null. GPC→DEXP and
WIZS→WIZC use existing same-company held-position transfers without a loan-source
alias or model-history change. Two additional Linx leads remain unadmitted:
the StoneCo BDR leg is outside933axes and final adjusted cash is unknown at the
June30 end. Its three held dates are not assigned zero value or future prices.

All24 new source books pass saved NAV checks, independent dated Decimal
entitlements and identical-intention two-account comparisons (maxR2.794e-8).
AtR10m, full/wide F7 returns are3.18393/5.45372bps/day aboveCDI; F11 returns
are8.26325/5.85459. Their source-only changes are respectively+.19423/-.12626
and-.04601/-.17868bps/day. These are same-forecast account effects, not data
refits. F7 remains unresolved; F11 has no remaining unpriced holdings. RLOG/Smiles
labels and GPC/Wiz model histories remain separately open along with GUAR/QGEP.

Thirty-six exposed one-factor bounds qualify;24 out-of-period/unexposed branches
are skipped. AtR10m, debit+100annualbp changes daily returns by-.0470 to-.0531bp.
RLOG custody-first changes wideF7 by-.02335bp/day; printed-versus-aggregate
fraction precision is about4.31e-8bp/day and the Smiles custody shift is below
1e-9bp/day on these paths. These are actual path endpoints, not joint or universal
bounds. Initial execution saved two funding books before a harness used
disposal_session equal to custody, rejected by the existing contract. Custody-first
correctly means no early-disposal override; only that harness encoding changed,
and the two completed books were reused. The production account was unchanged.
The later parity receipt originally retained2019-specific descriptive prose;
its qualified receipt changes only that description and binds the original.

The preserved baseline worktree matches all235 snapshot source files after newline
normalization:120 exact bytes and115 CRLF/LF-only differences. An initial stricter
byte assertion failed before the shell continued into the twelve F11 evaluations;
the subsequent source comparison verifies no semantic differences and those
completed books are reused. No GPU fit or numerical result was rerun for this.

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
