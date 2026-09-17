# Opportunity identification and portfolio arrangements

Authorized 2026-09-17 after completion of decision Phases 1–3. This is a new
development experiment: earlier results and their admission decisions stay sealed.
The user requests optional direction, sector neutrality, currency-consistent
performance reporting, and investigation of the remaining economic weakness.

## Inputs and boundary

Resolve `docs/v2_decision_run.json` and `docs/v2_data_inputs.json` at preparation.
Reuse the verified Phase-3 **neutral** three-seed ensembles for TE_all and C6,
2018–2024, with all 14 completed development folds and the original dated ISIN
population. No neural retraining, Lambda, forward capture or 2025/2026 reads.
Keep the old source root immutable. Write a new source-bound run and pointer.
The original per-fold equal-rank economic mappings remain the baseline, including
their explicitly recorded older OOS calibration population. New conditional fits
use only earlier new-ensemble OOS dates; they do not manufacture a 2016 prelude.

## A. Attribution and reporting

Report reconciled equity, hedge, financing, costs, borrow and settlement terms by
half-year and year. Attribute stock P&L to dated sectors/issuers where possible;
missing classifications remain eligible and are separately reported. Decompose
equity P&L into a lagged-beta market component and residual; this is an ex-post
linear attribution, not proof that a loss was predictable. Compare predicted
opportunity with subsequently matured shareholder outcomes. Preserve unresolved
valuation and terminal-haircut sensitivities.

For each book report annualized daily Sharpe (sample standard deviation, 252
sessions): BRL absolute returns minus realized CDI; USD-converted returns minus
US overnight cash; and BRL returns minus zero. USD conversion uses the ratio of
successive BRL-per-USD PTAX valuation fixes. US cash is EFFR (an explicitly labelled
overnight cash proxy, not a Treasury bond), ACT/360 accrued over calendar days
between account endpoints. Use dated fix/rate observations and disclose carried
holiday fixes and coverage; never use future fixes to fill a past endpoint. These
are reporting benchmarks, never model features. Include absolute and CDI-excess
win/loss/flat fractions, compounded return, annualized volatility and maximum
drawdown including initial wealth. Undefined Sharpe is null, not zero.

## B. Fixed portfolio arrangements

Same forecasts, mappings, costs, cash remuneration, execution and boundaries:

1. Neutral control: existing gross 2.25, absolute net/beta .05.
2. Flexible net: absolute net .45, beta .05. Isolates dollar neutrality from beta.
3. Direction permitted: absolute net and beta .45, zero additional market alpha.
4. Sector band: neutral control plus absolute net .05 per known CVM sector.
5. Sector neutral: neutral control plus zero net per known CVM sector.

The .45 NAV net band corresponds to 60/40 at gross 2.25; it is **not** a fixed
60/40 ratio at lower gross. Cash and either direction remain optional. Report
realized drift and planned exposures separately. Sector neutrality refers to
stock-sector dollar exposure, not sector-beta neutrality; BOVA is not assigned
an invented sector. Unknown-sector stocks are retained unconstrained by sector,
with their exposure disclosed. Use the existing receipt-dated CVM FCA bridge;
current B3 taxonomy cannot be retrospectively assigned without historical proof.
Strict neutrality can suppress small/singleton sectors: measure this effect.

## C. Identify conditional opportunity before increasing controller complexity

Use a small supervised daily regression, with one observation per date, to predict
the five-session equal-rank shadow payoff in daily return units and separately
BOVA excess return. Outcomes include causal decision-date beta residualization.
An origin becomes usable only after its five-session outcome closed on a previous
session (lag six). Missing labels are not replaced with zero. Shadow updates remain
observable when the traded account is in cash. Inputs are the previously verified
common market context, newly recomputed ensemble disagreement, and matured shadow
state. Fit all transforms only on preceding observations.

Earliest two years of new OOS forecasts provide calibration. For each subsequent
half-year, select ridge strength from {1, 10, 100} using the preceding 126-session
selection window and purged earlier fit data; then refit on all matured prior
dates. Compare against an expanding unconditional mean on identical dates.
Report paired squared-error changes with 40-session block-bootstrap intervals,
20/60-session sensitivity, calibration and signs. 2020–2021 is the development
screen; 2022–2024 is a temporally later check, **not untouched held-out data**.
Do not claim failure of a small regression proves all timing is impossible.

Diagnostic replay compares unconditional economic calibration with supervised
conditional alpha and a coherent market-forecast directional variant (stocks get
beta times the same market forecast given to BOVA). No rank spread is relabelled
as directional confidence. A matched conditional-alpha book with the same relaxed
net/beta bounds and zero market forecast isolates the added market forecast;
a fresh 2020-start neutral book isolates changed calibration from inherited
2018 inventory. Thus there are ten continuous books per architecture, rather
than hundreds of repeated fold/seed replays. Promotion requires improved opportunity prediction and
positive paired economic evidence in both periods, with no concealed settlement
or cash-accounting advantage. A rule-only improvement does not validate ML timing.

## D. Gates and scope

Use these results to decide whether multi-period retention optimization or a
separate economic checkpoint-selection experiment is justified. Neither follows
automatically from negative quarters: first identify forecast decay/turnover or
selection mismatch in saved evidence. Document every gate, including deferrals.
No broad hyperparameter sweep or additional neural campaign in this pass.

## Execution and evidence

CPU sparse allocation, one BLAS/PyTorch thread per process, at most two independent
arms concurrently. Cache scores and common context once. Preserve full history,
all eligible names, exact ledger and FP64 accounting. Targeted tests protect
currency/calendar alignment, maturity/fit isolation, group constraints and
unchanged baseline. Commit source before financial replay, hash its inputs,
record every variant, retain paired daily series and report failed experiments.

## Recorded follow-up after the original twenty books

The original conditional models failed the predictive/economic comparisons, and
all twenty prior-only fits selected the strongest offered ridge penalty (100).
Before dismissing conditional learning, run a **post-hoc development diagnostic**
using the same frozen inputs and chronological windows. This is not a fresh
confirmation and does not replace or erase the failed books.

Compare the existing common context against a compact context: recent market
return, median daily volatility, cross-sectional dispersion, matured shadow
payoff/volatility, and seed/horizon disagreement, plus explicit validity flags.
Choose penalties {1,10,100,1000,10000,constant mean} independently for shadow
payoff and market return using the same purged prior 126-session selection.
Refit only on matured earlier OOS labels. The constant is a selectable causal
fallback, never chosen using the next evaluation period. Do not select a feature
roster or penalty from full-period financial P&L.

This follow-up initially predicts only. It triggers additional financial replay
only if conditional squared error improves with a positive lower 40-session
paired interval in both 2020–2021 and 2022–2024 for the relevant head. Report
20/60-session sensitivity and all attempted feature sets. Otherwise close this
timing branch without interpreting it as proof that all timing is impossible.
