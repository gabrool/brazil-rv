# Reading the old and corrected performance figures

The closest identified match to the user's paired recollection of about seven
bps/day and zero-rate Sharpe near two is **C6/flexible_net: 6.70751 bps/day ABOVE
CDI and zero-rate Sharpe 1.88147**. Its absolute BRL return was 9.74623 bps/day.
The return in that older headline already subtracts CDI. Benchmark subtraction
therefore cannot explain its difference from the current excess-CDI headline.

That result is in `v2_opportunity_results.json`, `arms.C6.books.flexible_net`.
It covers 1,738 continuous 2018–2024 evaluation sessions, allowing 45% net
exposure while retaining a 5% beta cap. The current primary screen instead uses
a 5% net cap and four separate half-year accounts. The old report already shows
a substantial policy difference before any current correction:

| C6 comparison | Above CDI bps/day | Absolute BRL bps/day | Zero-rate Sharpe |
|---|---:|---:|---:|
| Original continuous history, flexible 45% net cap | 6.70751 | 9.74623 | 1.88147 |
| Original continuous history, neutral 5% net cap | 4.00048 | 7.03920 | 1.71376 |
| Original four-period neutral screen | 0.66502 | 3.74677 | 1.34691 |
| Corrected/refitted four-period neutral screen | 0.45646 | 3.56240 | 1.20549 |

The first two Sharpes use the continuous return series; the last two are equal
means of fold Sharpes. The last two rows are the matched correction comparison.
The earlier continuous rows are context, not a pure attribution bridge: policy,
account boundaries and evaluation periods differ. No corrected continuous
flexible-net replay yet establishes the survival or disappearance of its 6.71.

A separate earlier result near seven **absolute** bps/day is the attention rank/IC
three-seed ensemble in `v2_portfolio_objective_results.json`,
`continuous.TE_all:rank:ic:ensemble`. Across 1,738 continuous evaluation sessions
it earned 7.63587 bps/day after costs before subtracting CDI, 4.59715 above CDI,
and Sharpe 1.74387 against zero or 1.05029 against CDI. Its average CDI was
3.03872 bps/day. The initial investigation found this attention figure first;
the C6 flexible-net pair above is a closer match to both numbers in the question.
The original financing and settlement limitations of both remain recorded.

Stage C instead evaluates July–December of 2018, 2020, 2022 and 2024 (F2, F6,
F10 and F14), with separate accounts initialized in cash and equal fold weights.
The original models were already much weaker on these four periods. Earlier
full-history figures also include other forecast versions, and some older
six-bps **excess-CDI** results used a flexible 45% net-exposure cap rather than
the primary 5% cap. These are separate comparisons.

## Actual matched before/after comparison

All entries below are R$10m three-seed ensemble bps/session **after account costs
and before subtracting CDI**, equally averaged over the same four screen folds.
Original figures are read from the sixteen immutable original ensemble books
bound by `stage_c_account_results.primary`; current figures use
`stage_c_refit_results.model_summary`. Corrected-account/source figures add the
corrected benchmark to the saved matched excess-return means. No fit, forecast
or portfolio replay was run to produce this comparison.

| Model | Original account/data | Corrected account/source, old forecasts | Corrected data and refit | Final minus original |
|---|---:|---:|---:|---:|
| Full attention | 3.44855 | 3.63540 | 4.10514 | +0.65660 |
| Wider attention | 4.78689 | 4.48690 | 2.96219 | -1.82470 |
| GRU early peers | 4.13113 | 4.21346 | 3.78663 | -0.34450 |
| C6 | 3.74677 | 3.87412 | 3.56240 | -0.18438 |

Corrected CDI averages 3.10594 bps/day on these periods, versus the original
3.08175. Thus the current full-attention headline of 0.99921 above CDI is
4.10514 absolute BRL minus 3.10594 CDI. Primary remuneration of eligible settled
short proceeds remains 100% CDI; subtracting CDI as a reporting benchmark does
not remove that income from the account.

| Model | Current absolute BRL bps/day | Current above CDI bps/day | Original Sharpe versus zero | Current Sharpe versus zero | Current Sharpe versus CDI |
|---|---:|---:|---:|---:|---:|
| Full attention | 4.10514 | 0.99921 | 0.69171 | 1.03756 | 0.21312 |
| Wider attention | 2.96219 | -0.14375 | 1.02689 | 1.00386 | 0.13692 |
| GRU early peers | 3.78663 | 0.68069 | 1.05728 | 1.00646 | 0.21667 |
| C6 | 3.56240 | 0.45646 | 1.34691 | 1.20549 | 0.40230 |

All Sharpes in this table are means of four separate fold Sharpes, not the Sharpe of
an uninterrupted seven-year account. Full attention earns -7.19915 excess
bps/day in F6 and +10.90578 in F14, showing why period selection matters.
The original zero-rate Sharpes are reconstructed directly from each bound
original book's saved `daily.absolute_bps`: sqrt(252) times its mean divided by
sample daily standard deviation, then equal-weighted across the four folds.
All sixteen books have zero burn-in and matching saved return/date lengths.
This reads saved returns only; it is not a portfolio or model rerun.

## What can be concluded

Accounting/source corrections alone do not erase several bps/day across these
models. Wider attention does deteriorate materially after corrected data and
necessary compatible retraining: that step costs 1.52471 bps/day. Full attention
improves by 0.46974 over its corrected-account old-forecast control; GRU and C6
decline by 0.42683 and 0.31172. These refit differences combine changed inputs,
targets, fit-only conditioning and the resulting training trajectories. The
saved comparisons do not isolate one repair as their cause, and the finite
seed/fold uncertainty remains in the matched-results report.

No corrected continuous-history result yet establishes either preservation or
loss of the earlier C6 flexible-net 6.71 excess-CDI bps/day. The September 21 user
amendment now authorizes a broader diagnostic comparison despite the failed
original four-fold gates. Eight development evaluation folds are fixed in
[the investigation contract](v2_SCALING_INVESTIGATION.md), with six reserved from
new corrected comparisons. Training already uses the expanding historical past;
the four-fold restriction applied to evaluation. These reserves are not
retroactively called untouched tests. Reporting must keep period, account
continuity, exposure policy, forecast version, return benchmark and Sharpe
aggregation visible. The attention reversal is under investigation; the corrected
four-period result is not a demonstrated replacement for the older continuous
flexible-net result.

Evidence: [earlier C6 flexible and neutral accounts](v2_opportunity_results.json),
[earlier attention continuous account](v2_portfolio_objective_results.json),
[earlier policy and period distinctions](v2_FOUNDATION.md),
[matched accounting/source attribution](v2_MATCHED_ECONOMIC_REPLAYS.md),
[matched corrected refits](v2_MATCHED_RESULTS.md), and the canonical bindings in
[the economic run pointer](v2_economic_data_scaling_run.json).
