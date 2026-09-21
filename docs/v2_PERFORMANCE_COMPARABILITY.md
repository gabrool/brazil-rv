# Reading the old and corrected performance figures

The latest four-period screen is not a corrected replacement for the earlier
continuous 2018–2024 account. Comparing its excess-CDI return directly with the
older absolute BRL return materially overstates the apparent deterioration.

An earlier result close to the user's recollection is the attention rank/IC
three-seed ensemble in `v2_portfolio_objective_results.json`,
`continuous.TE_all:rank:ic:ensemble`. Across 1,738 continuous evaluation sessions
it earned 7.63587 bps/day after costs before subtracting CDI, 4.59715 above CDI,
and Sharpe 1.74387 against zero or 1.05029 against CDI. Its average CDI was
3.03872 bps/day. This identifies a comparable scale of earlier reporting; it does
not establish that this is the exact model the user remembered. Its original
financing and settlement limitations remain recorded.

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

| Current corrected model | Absolute BRL bps/day | Above CDI bps/day | Mean fold Sharpe versus zero | Mean fold Sharpe versus CDI |
|---|---:|---:|---:|---:|
| Full attention | 4.10514 | 0.99921 | 1.03756 | 0.21312 |
| Wider attention | 2.96219 | -0.14375 | 1.00386 | 0.13692 |
| GRU early peers | 3.78663 | 0.68069 | 1.00646 | 0.21667 |
| C6 | 3.56240 | 0.45646 | 1.20549 | 0.40230 |

The current Sharpes are means of four separate fold Sharpes, not the Sharpe of
an uninterrupted seven-year account. Full attention earns -7.19915 excess
bps/day in F6 and +10.90578 in F14, showing why period selection matters.

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
loss of the earlier roughly seven absolute bps/day. The registered additional
ten-fold tests are conditional; both nominated Stage C leads failed their
positive-advantage condition. This explanation does not authorize an unregistered
full-history refit or change those gates. Capacity experiments retain their own
predeclared conditions. Reporting must keep period, account continuity, exposure
policy, forecast version, return benchmark and Sharpe aggregation visible.

Evidence: [earlier continuous account](v2_portfolio_objective_results.json),
[earlier policy and period distinctions](v2_FOUNDATION.md),
[matched accounting/source attribution](v2_MATCHED_ECONOMIC_REPLAYS.md),
[matched corrected refits](v2_MATCHED_RESULTS.md), and the canonical bindings in
[the economic run pointer](v2_economic_data_scaling_run.json).
