# Direct portfolio-objective experiment: implementation and evidence

Status: implementation, engineering admission and the 72-fit screen are complete;
the sole admitted candidate is undergoing confirmation. This is an interim review,
not a promotion decision.
Resolve [the canonical run pointer](v2_portfolio_objective_run.json). The complete
contract is [the registration](../research/preregistrations/v2_portfolio_objective.md);
[progress and recovery instructions](v2_portfolio_objective_progress.md) identify
what remains.

## 1. The question being tested

Does training the predictor through an economically coherent portfolio account
improve subsequent returns relative to continuing the ranking objective? Does
selecting checkpoints by economic utility help independently of changing the loss?
The previous auxiliary Huber experiment did not answer the first question: its
cardinal head was not the preference consumed by the allocator, and selection
remained based on IC. We now give the shared encoder direct economic gradients.

This is a continuation experiment on already learned representations. It does not
determine the best architecture or economic training recipe from scratch, and a
negative outcome would not establish that economic learning is impossible.

## 2. A material numerical defect discovered during admission

The existing allocator's forward solutions were accurate, but its native OSQP
backward was inaccurate for a realistic small-variance hedge problem. On an
eight-session synthetic account, the analytic derivative was +.285666, whereas
centered perturbations gave -1.336235. Differentiating the account without the
allocator agreed with perturbations. Mathematically equivalent objective scaling
changed the OSQP derivative substantially, so an arbitrary rescale was rejected.

The repair retains the Clarabel forward solution and differentiates its active
face directly. Strict zero, no-trade and bound coordinates are fixed locally; the
remaining positive diagonal-plus-market covariance Hessian and active exposure
constraints determine the sensitivity. This avoids a second QP solve and the
unstable native adjoint. Gradients are local and piecewise smooth: at changes in
the binding constraints, finite differences can cross into another active face.

Thirty targeted tests passed, including multi-day numerical gradients, hedge
variance, inventory/uncertainty derivatives, cloned SAM state, and mutations of
future labels that must not alter model preferences. Three historical twelve-day
fitting paths also agree with numerical perturbations. Maximum independent-account
NAV error on those paths is below 8e-13. The four actual neural GPU cases have NAV
error below 3e-11. [Source-bound evidence](v2_portfolio_objective_engineering.json)
records the individual measurements.

This finding qualifies earlier learned-controller failures: the financial gradient
supplied to those controllers could be wrong. It does not invalidate their sealed
forward books, nor the ranking/Huber neural fits, and it does not guarantee a
successful controller after repair. The new experiment tests that prospect.

## 3. Bounded portfolio and financing closeout

Before changing neural objectives, the program completed 22 additional CPU books
and reused four sealed full-map books. All figures below are mean daily basis
points above CDI. These are separate complete accounts, not additive attribution.

| Mapping / assumption | C6 neutral | C6 flexible | Attention neutral | Attention flexible |
|---|---:|---:|---:|---:|
| Full calibrated forecast | 4.000 | 6.707 | 4.497 | 5.937 |
| Remove calibration intercept | 4.466 | 5.823 | 4.506 | 5.225 |
| Intercept only | .503 | 2.154 | .503 | 2.154 |
| Full forecast, 8 bps per side | 3.513 | 6.172 | 3.958 | 5.370 |
| Full forecast, no short-proceeds interest | 1.037 | 4.074 | 1.419 | 3.221 |
| Full forecast, 3% debit spread | 3.893 | 6.377 | 4.394 | 5.649 |

Neutral and flexible absolute net caps are .05 and .45 respectively; both retain
the .05 beta constraint. Part of flexible exposure's gain is an unconditional
common residual tilt; it is not evidence of successful conditional market timing.
The rank component also benefits from flexibility. Estimated book beta and realized
return beta differ, so dollar neutrality alone does not explain exposure.

Cash and short-proceeds remuneration assumptions matter materially. These scenarios
are not verified brokerage terms. Large stale-mark settlements include TRPL, CIEL,
BRML, ALSO and CPLE units. Source inspection binds the available quotes and action
rows, but the accepted store has zero established ISIN succession links and retains
inferred corporate-action terms. Contractual dispositions remain unverified.
[The closeout export](v2_portfolio_objective_closeout.json) preserves these limits.
No affected name was removed, and no hypothetical settlement was relabelled as an
observed executable fill. The objective comparison therefore retains the established
neutral allocator, with flexibility and accounting stresses as secondary readouts.

## 4. Matched training design

The two arms are C6 (slow GRU, fundamentals and magnitudes, SAM .125) and TE_all
(all registered families, early temporal/peer attention, ASAM .2). Both retain the
accepted repaired store, checkpoint-bound preprocessing, 60-session history and
every eligible security. Each fold/seed starts from its verified Phase-3 neutral
selected F model. No pretraining is repeated.

The economic cache covers 2016-07-18 through 2024-12-30, with all 363,314 accepted
active stock-days and identical permanent-identity/date axes in both arms. Its
cached score masks exclude zero accepted active observations. Earlier feature
history and the original pretraining remain bound to the accepted full store.

Every matched parent produces three continuations: ranking only; ranking plus
portfolio utility; and utility only. All receive a zero-initialized linear cardinal
head on the shared representation. The allocator's preference is the causal fixed
rank-to-return mapping, made differentiable using smooth cross-sectional ranks,
plus that unranked daily-return correction. One head unit represents one daily
basis point. No tanh, clipping or reranking erases its magnitude.

The smooth-rank temperature is .1 in standardized-score units. On the first 64 F2
fitting dates, mean absolute preference changes from hard ranks are .00560 bps for
C6 and .01686 bps for attention; maxima are .0282 and .0887 bps. The smooth interface
is shared by all controls. These fitting-only bridge measurements did not select
a temperature on evaluation outcomes.

Utility is actual daily net excess over CDI minus 2.5 times causal portfolio
variance. Gradients pass through allocation, execution costs, borrowing, payments,
shares and cash. Overlapping multi-day labels are never added up as portfolio P&L.
The allocator permits cash and jointly constrains stocks/BOVA: gross 2.25, net .05,
beta .05, individual stocks .05 and hedge .60, with risk aversion 5 and five-session
planning. The cardinal preference is not a separately learned market forecast.

Training visits dates chronologically in 32-session gradient blocks. Account state
carries across blocks, while gradients are truncated at those boundaries. There
is no internal liquidation or cash reset. SAM's two passes and overflow retries
restart from the same account state and RNG. The clean-pass account advances the
path. This is a practical approximation to full-history portfolio differentiation;
it does not reduce the encoder's 60-session lookback.

The relative utility weight equals the median ratio of rank/utility gradient norms
on the shared encoder at three pre-specified initial fitting blocks. It is reused
across all three objectives. This normalizes gradient units using fitting data;
it is not an optimization of the weight on selection or evaluation returns.

Continuation lasts at most twelve epochs, at least six; thereafter it stops if
neither selector has improved for five epochs. Peak learning rates are 3e-5 for
inherited parameters and 1e-4 for the new head, with the inherited SAM/ASAM and
AdamW decay routing. Both selectors may retain epoch zero. Small improvements
below .0001 IC or .01 daily bps utility retain the earlier checkpoint.

## 5. Evaluation and causal boundaries

The screen covers folds F2/F6/F10/F14 and seeds 11/29/47: 72 fits. IC and economic
selectors share each trajectory and all saved epochs; they do not receive separate
training budgets. Selection starts from cash on the original first selection date,
after the inherited F model's fitting labels have matured. Selection targets stay
inside their permitted window. Evaluation uses only the post-selection embargo
as unreported inventory burn-in. Fitting inventory never enters selection.

The economic selector is the sole advancement route. Admission requires positive
paired ensemble net and utility, positive utility differences in at least three
of four folds, and at least two of three seeds. Only admitted arm/objective pairs
and their matched rank controls receive the remaining ten confirmation folds.
Confirmation requires positive lower 95% paired 40-session block intervals for
both net and utility, and positive utility for at least two seeds. The 20/60-session
intervals are sensitivities. IC selection and comparisons with the unmodified
warm start remain visible for every cell.

Separate fold accounts must not be presented as uninterrupted compounded history.
If confirmation is triggered, all fourteen folds additionally feed continuous
accounts carrying real inventory across fold boundaries. Results remain reused
development evidence through 2024, not a fresh untouched holdout. Nominal intervals
do not correct the entire historical research search, and unresolved settlement
and financing assumptions qualify any economic conclusion.

## 6. Efficiency and recovery

The implementation reuses exact compact caches across the three objectives,
compiles neural and loss kernels, uses FP16 AMP with FP32 master weights and FP64
finance, and uses the existing fused optimizer. Microbatches contain at most sixteen
dates. No data family, eligible security or history session was cut for speed.

Four representative GPU engineering cases cover both arms at F2 and F14. Warm
32-session SAM steps take .765-.938 seconds, with peak allocated GPU memory of
1.32-3.14 GiB. Compilation and cache construction add startup time. An initial
engineering attempt hit Inductor's donated-buffer restriction when retaining a
graph for two gradient-norm calculations; separate forwards repaired that before
financial fitting, preserving the optimized execution path.

The financial worker runs from isolated clean commit 33086be, with training math
identical to accepted engineering source 295f416. Every trajectory saves epoch
checkpoints, optimizer/scaler/RNG resume state, both selected checkpoints and
hash-bound manifests. The audit verifies matched initial weights and reproduces
both selectors from saved history. A hidden supervisor prevents sleep during the
campaign. The existing heartbeat monitors progress and handles final verification,
review, recovery archiving and GitHub publication.

## 7. Screen findings and pending confirmation

All 72 fits passed the trajectory audit, with zero zero-gradient blocks and zero
AMP retries. The readout audit verified 168 forecasts and 392 books across eight
arm/fold panels, including exact eligible populations and permanent-identity axes.
[The compact screen export](v2_portfolio_objective_screen.json) binds the full output.

Primary economic-selector comparison, in mean daily bps above CDI:

| Arm | Objective | Net | Difference from matched rank | Utility difference | Pass screen? |
|---|---|---:|---:|---:|---|
| C6 | Rank | 1.623 | — | — | Control |
| C6 | Hybrid | 1.255 | -.368 | -.356 | No |
| C6 | Utility | .952 | -.671 | -.674 | No |
| Attention | Rank | 1.305 | — | — | Control |
| Attention | Hybrid | 1.311 | +.006 | -.002 | No |
| Attention | Utility | 2.149 | +.843 | +.828 | Yes |

Attention utility-only has positive paired utility in F2/F10/F14, negative in F6,
and positive mean differences for all three seeds. Its 40-session paired net
interval is [-.745, 2.418] bps/day; utility is [-.760, 2.402]. Passing this screen
means it merits confirmation, not that superiority has been established. Under
the diagnostic IC selector its net advantage is +.552 [-.730, 1.874] bps/day.
The hybrid attention result is effectively flat, and the C6 economic-selector
comparisons do not support continuation with either economic loss.

These means cover 501 evaluation sessions in four separate fold accounts. They
must not be compared directly with the earlier 4.0/4.5-bps continuous full-history
figures; dates, account boundaries and the matched continuation/interface differ.
The registered confirmation started automatically: sixty TE_all fits, comprising
rank and utility across the remaining ten folds and three seeds. No C6 or hybrid
confirmation is run. No training contract was changed after observing the screen.

The final review will add confirmation results, selector comparisons, economic
and IC metrics, BRL/CDI, USD/EFFR and zero-rate Sharpes, winning/losing days, drawdown,
exposure, costs and halves. It will state whether any candidate passes the registered
gate and which assumptions limit the conclusion. No new objective or model is
promoted at this interim checkpoint.
