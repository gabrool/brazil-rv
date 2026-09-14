# Decision research — Phase 1 results and acceptance

Phase 1 is complete: 315 books covering three frozen forecasters, seven policy
cells, fourteen development folds and one continuous model-switch book per cell.
No forecaster was retrained. All nine recovered cache files match the sealed
source inventory. All ninety original-calibration/cash controls reproduce their
sealed economic series within the documented numerical acceptance bounds.

The most useful finding is **instability in the three-horizon calibration**, not
a universal failure of attention or a universal benefit from holding less risk.
Fitting one coefficient to the equal-average horizon rank substantially improves
S0 and early attention in these development results, with markedly lower trading
costs. It makes little economic difference to C6. Correcting the stock/hedge
return definition removes much of the systematic hedge saturation but does not
by itself improve every forecaster. The fixed uncertainty penalty is too costly
for C6 in this comparison.

Sources: [registration](../research/preregistrations/v2_decision_research.md),
[complete numerical readouts](v2_decision_phase1_results.json),
[canonical run pointer](v2_decision_run.json). Financial books were produced by
commit `eed4a4e`; earlier stopped numerical attempts are retained in the same run.
These are development comparisons on previously studied periods, not a new
untouched holdout or a live-performance claim.

## What was held fixed

The accepted repaired data store, point-in-time identities/eligibility, full
forecast population, three forecast seeds, existing out-of-fit prelude, fourteen
folds, execution prices, shareholder-action accounting, CDI, borrow assumptions,
four-bps-per-side transaction costs and declared risk/position constraints are
unchanged. The continuous book carries inventory across forecaster switches and
liquidates only at the final boundary. Fold books retain their registered burn-in
and individual boundary liquidation.

The six active policies are original raw calibration; its zero-intercept
mechanism diagnostic; benchmark-residual calibration; that calibration with one
parameter-estimation standard-error penalty; one-regressor equal-rank benchmark
calibration; and a stock-only benchmark diagnostic. Cash is the seventh cell.
All active policies use the same cost-aware QP, which can retain existing
positions or choose cash. Stock-only is the sole declared hedge-availability
change. It retains the other stock, gross, net and beta constraints.

The coherent label is five-session shareholder excess over compounded CDI,
less decision-time beta times the matching BOVA excess return, divided by five.
The benchmark retains zero residual alpha. Fitting labels mature entirely inside
the permitted fit interval. Future BOVA returns are labels only. **There were
zero additional masked calibration observations from missing benchmark labels,
and no prediction input or eligible stock-day was removed.**

## Continuous economic results

Mean daily net return above the same all-cash CDI benchmark, in bps:

| Policy | S0 | Early rich attention, TE_all .2 | C6 |
| --- | ---: | ---: | ---: |
| Original raw calibration | -0.913 | 2.724 | 4.693 |
| Raw slopes, intercept removed | -1.137 | 1.903 | 5.644 |
| Coherent benchmark residual | -0.363 | 2.535 | 4.972 |
| Benchmark residual plus uncertainty penalty | 0.886 | 1.716 | 2.753 |
| Equal-rank benchmark residual | 4.112 | 4.555 | 4.897 |
| Benchmark residual, stock-only | -1.026 | 1.526 | 5.316 |
| Cash | 0.000 | 0.000 | 0.000 |

The registered utility subtracts 2.5 times causal daily portfolio variance from
net excess return. Benchmark/equal-rank utilities respectively are -1.016/3.673
bps for S0, 2.033/4.087 for TE, and 4.506/4.470 for C6. Lower gross alone does not
explain away the benefit to S0/TE; it also reduces the stated risk penalty.

The maximum point estimate in this table is not an instruction to select that
policy retrospectively. Removing the intercept or removing the hedge are
diagnostics. There is no robust common superiority of either intervention.

## Dependence-aware comparisons

Equal-rank minus the three-regressor benchmark calibration, continuous net bps/day:

| Forecaster | Paired difference | 40-session circular 95% interval | Fold-book paired difference and interval |
| --- | ---: | --- | --- |
| S0 | 4.475 | [1.733, 7.487] | 4.595 [2.709, 6.556] |
| TE_all | 2.020 | [0.183, 3.946] | 2.082 [0.652, 3.593] |
| C6 | -0.076 | [-2.114, 1.760] | -0.030 [-1.880, 1.779] |

The S0 and TE continuous intervals also remain positive with 20- and 60-session
blocks. Their respective lower endpoints are 1.764/1.672 and 0.240/0.144 bps.
C6's intervals include zero throughout. The uncertainty penalty's C6 continuous
difference is -2.220 [-4.097, -0.446] bps at the primary block length. Its TE and
S0 differences remain inconclusive.

All new intervals use 10,000 circular resamples and the registered seed. Every
date has equal marginal sampling probability; the prior finite-block boundary
underweighting is removed. Fold resampling preserves fold boundaries. Both net
and utility results, all cells, all three block lengths, and resample means are
saved. Resample means are close to their paired estimates; no material centering
failure was found. These nominal intervals do not adjust for the broader history
of development research or establish seed-robust superiority: this phase reuses
the fixed three-seed forecast ensemble, rather than fitting independent new
forecasters or optimizing a policy per seed.

## Calibration and mechanism diagnostics

The original fitted intercept ranges from 3.340 to 8.177 daily bps across folds.
Benchmark adjustment reduces this to 1.637–4.614 bps. These intercepts coincide
across arms because their fitting populations/outcomes coincide. Residual alpha
need not have zero mean: a universe-wide residual premium is still an estimated
quantity, whose future reliability is not guaranteed.

Horizon-rank correlations are very high: 0.904–0.994 for S0, 0.919–0.987 for TE,
and 0.912–0.987 for C6. Regularized Gram-matrix condition numbers range from
598–1,409, 135–239 and 625–866 respectively. Equal-rank fitting has condition
number one. In S0, the benchmark-calibrated D5 coefficient is negative in every
fold, ranging from -19.986 to -2.056 daily bps per fitted standard deviation,
while D3 and D10 coefficients are positive. This creates a horizon-difference
trade with large offsetting coefficients, rather than simply combining three
similar positive signals.

Negative coefficients are not intrinsically an error; differences between
horizons can contain information. The evidence against this particular S0/TE
mapping is the matched equal-rank economic comparison. C6 demonstrates why an
automatic nonnegative-coefficient restriction would be unwarranted.

Out-of-fit cardinal slopes also vary substantially: -1.507 to 5.699 for S0,
-3.034 to 7.840 for TE, and -5.081 to 5.771 for C6. Three, three and one of the
fourteen respective folds have negative slopes. The saved readouts include
equal-date predicted/realized means, RMSE and fixed upper/lower rank-tail
outcomes. For example, F1 predicts a universe residual mean near +3.85 bps/day
while the realized value is -2.36; its TE upper/lower rank tails nevertheless
realize +4.60/-8.82 bps/day. A common premium, cross-sectional ordering and
cardinal magnitude are different questions. A pooled slope is a descriptive
diagnostic, not proof that all date-conditional relationships reverse.

Continuous exposure and costs provide a clearer mechanical explanation:

| Forecaster | Benchmark / equal-rank mean gross | Benchmark / equal-rank daily turnover | Benchmark / equal-rank trading cost, bps/day |
| --- | --- | --- | --- |
| S0 | 2.233 / 1.982 | 0.538 / 0.116 | 2.154 / 0.463 |
| TE_all | 2.112 / 2.023 | 0.296 / 0.130 | 1.184 / 0.519 |
| C6 | 2.005 / 1.924 | 0.323 / 0.092 | 1.290 / 0.369 |

As an exposure-normalized descriptive ratio, net bps divided by mean gross
changes from -0.163 to 2.075 for S0, 1.200 to 2.252 for TE, and 2.480 to 2.544
for C6. This ratio is not a matched-risk counterfactual or a Sharpe ratio.

Original short-hedge-cap occupancy is 94.3%, 99.7% and 96.7% of continuous
decision dates. Coherent calibration lowers it to 64.5%, 65.2% and 64.9%.
Gross-cap occupancy also falls, but remains high: 92.9%, 79.2% and 65.4% under
the benchmark cell. Removing the calibration mismatch therefore fixes a real
pressure toward a stock/ETF basis position; it does not make the allocator a
reliable regime detector. The fixed uncertainty penalty reduces mean gross to
1.341/1.411/1.303 and gross-cap occupancy to 19.1%/7.9%/21.3%, but sacrifices
profitable C6 exposure. A learned reliability mechanism must earn its value
against these explicit deterministic controls.

## Numerical acceptance and operating record

Ninety raw/cash controls passed. The largest daily bps-field deviation from the
sealed host is **0.000741 bps**; the largest mean deviation is **0.00000180 bps**;
the largest fraction-field deviation is **0.00000164 NAV**. These are materially
smaller than the final declared daily/mean tolerances. No meaningful original
financial result changed. Initial overly tight, unit-insensitive checks stopped
two partial attempts; those attempts and the numerical amendments are retained.
No financial calculation was changed to clear them.

Local memory limits required one cache at a time and a fresh process for the
remaining C6 books. Windows filesystem compression reduced the three cache
files' physical storage from approximately 2.10 GB to 0.57 GB, with unchanged
logical bytes and source SHA-256s. A typical half-year active replay took about
one second; continuous replays took roughly 15–21 seconds. The readout is cheap;
source recovery and numerical acceptance dominated this phase's elapsed time.
No paid instance was launched.

Targeted calibration, covariance, accounting, allocation, causality and bootstrap
tests passed before financial execution. The source-bound books include account
arrays, order/fill tables, financial series and per-book diagnostics. The
complete forecast input population remains intact. The fixed funding,
missing-price settlement and capacity assumptions retain their earlier limits.

## Phase acceptance and the next step

After completion, section 13 of the postmortem was reread. Every Phase 1 item is
covered: frozen controls, corrected return definition, uncertainty variant,
deterministic retention/sizing, cash/hedge attribution, numerical reproduction,
calibration/tail/exposure/cost diagnostics, fold and continuous comparisons, and
corrected dependence-aware inference.

Proceed to Phase 2's independent-date behavioral acceptance, then its registered
small real-data controller screen. The coherent three-regressor benchmark
remains the predeclared reference; equal-rank is an important additional
deterministic comparison, not an evaluation-selected rewrite of the reference.
Do not describe a controller as improved merely because it beats a weak mapping
while losing to equal-rank. C6 remains the strong economic comparator and TE_all
.2 remains the working attention candidate. Phase 3's objective experiment stays
authorized regardless of whether Phase 2's controllers qualify.
