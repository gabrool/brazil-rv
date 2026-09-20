# Historical cost block closeout

2026-09-20. Dated spot, loan and custody costs now have explicit implementations
and qualified precision/payment hypotheses. This closes the historical-cost
delivery block for its stated research contract. It does **not** accept Stage A:
calendar/event/data composition and integrated admission remain, then C and D.
Reuse all earlier engineering evidence and both immutable accepted stores.

## Invoice contract

Both actual accounts retain unrounded fractional research costs as primary and
support `security_day_6dp_cent` as one invoice hypothesis. The existing original
[017/2023-VPC](https://www.b3.com.br/data/files/BB/35/6C/6C/E810B810E9C1AAA8AC094EA8/OC%20017-2023-VPC%20Consolida%C3%A7%C3%A3o%20de%20Regras%20da%20Pol%C3%ADtica%20Tarifa%C3%A7%C3%A3o%20Produtos%20Mercado%20Renda%20Vari%C3%A1vel%20PT.pdf)
is bound by `historical_tariff_sources`, including its already visually qualified
pages 17–19. No repeat download or page qualification was needed.

The selected scope is one ordinary own-account CNPJ, clearing member, participant,
normal cash-market category and configured execution phase. Actual same-direction
notional consolidates by security/day, including the separately identified BOVA
hedge. The hypothesis rounds group notional half-up to six decimals, rounds each
trading/clearing fee to six decimals, then sums integer micro-BRL fees by category
and truncates each account/category total to cents. Integer micro-BRL avoids a
second binary summation crossing an exact cent boundary. Brokerage and shortfall
remain unchanged. Fractional research units, this grouping of stock/ETF cash
business, and application before the recovered 017/2023 source are explicit
hypotheses, not observed client invoices or independently recovered old rounding
permission. The source's integer quantity convention is not falsely attributed
to the fractional simulator.

`spot_invoice_adjustment` reports invoiced minus unrounded trading/clearing costs
in BRL. `execution_charges` and total cost already include this amount: never add
it twice. Individual `Fill.cost` and the old hedge cost diagnostic remain unrounded;
their expense reconciliation requires the adjustment. Recognition follows actual
fills and current intentions; cash follows those fills' original dated T+3/T+2
settlement. Prior-close funding is unchanged. Existing settlement state carries
unpaid amounts through terminal boundaries and independent SAM/TBPTT copies.
Exact floor derivatives are zero locally; no straight-through estimator is used.

No opposite same-security/day fills occur in the selected evidence. The existing
guard therefore remains; an exposed opposite-direction model path must receive
separate daytrade admission. Same-direction multi-fill consolidation is verified
by a deterministic fixture; all 4,287 historical groups here contain one fill.
Neither corporate conversion nor a loan return is treated as a spot fill.

## Six new variants, six saved controls reused

The plan was frozen in `spot_invoice/plan.json` before implementation/outcomes.
Only six invoice variants ran: October24 2016–February7 2017 (73 sessions) and
the existing long-focus SOMA/ENAT July24–September30 2024 window (49 sessions),
each at R$10m/R$1m/R$5m. Their six unrounded parents were read, never replayed.
All 933 names, original calibration and old PolicyData coordinates remain;
accounting/source replacements use a shallow copy. Frozen synthetic preferences
and fixed engineering risks match each parent. Corrected CDI, qualified lending,
strict prior references, dated tariffs and corporate custody remain identical.
There was no neural forward pass, scoring, fit or model-profit measurement.

All six first invocations passed. Independent saved checks cover 366 daily NAVs,
1,722,762 account cells and 4,287 fills/security-day groups. Decimal arithmetic
independently reconstructs every grouped invoice and adjustment. Separate signed
fill arithmetic rebuilds the T+3/T+2 net unsettled cash queue, including charges.
NAV, prior funding, income and both accounts' cost components reconcile.

| Check | Maximum absolute error, BRL |
| --- | ---: |
| Saved NAV identity | 3.725290298461914e-9 |
| Decimal grouped B3 invoice | 9.094947017729282e-13 |
| Decimal invoice adjustment | 6.768396607070848e-13 |
| Fill costs plus adjustment | 1.3642420526593924e-12 |
| Cost component sum | 1.8189894035458565e-12 |
| Independent unsettled spot cash | 5.471520125865936e-9 |
| Prior funding balances | 5.122274160385132e-9 |
| Income | 5.4569682106375694e-12 |
| Identical-intention account NAV | 9.313225746154785e-9 |

The first invoice difference is on day zero in every case. The first current
intention is exactly unchanged; the historical pre-difference NAV prefix is empty,
not an additional multi-day causal proof. Deterministic fixtures independently
verify settlement lag and prior-funding ordering.

Independent adaptive maximum path differences at R$10m/R$1m/R$5m are
0.000000469297171/0.000001118424116/0.000000368075445bp, with maximum target
distance 3.962279622893999e-9. **All larger previously measured adaptive
fixed-fee uncertainties remain applicable.** These smaller new discrepancies do
not erase the earlier R$1m 0.29492822bp or other case-specific limits.

| Window | Capital | Final/max path contrast, bp | Direct trading/clearing adjustment, BRL |
| --- | ---: | ---: | ---: |
| 2016–2017 | R$10m | +0.000731838759 | -0.376787288 / -0.363535325 |
| 2016–2017 | R$1m | +0.006978274280 | -0.330929972 / -0.374224109 |
| 2016–2017 | R$5m | +0.001494635906 | -0.349721502 / -0.406346979 |
| 2024 | R$10m | +0.000526152058 | -0.269040560 / -0.245202801 |
| 2024 | R$1m | +0.005563991892 | -0.270227572 / -0.271137859 |
| 2024 | R$5m | +0.001017787842 | -0.259403735 / -0.237018677 |

Contrasts are total synthetic adaptive paths against their saved unrounded
parents, not daily alpha, pure direct charges or all-interior adaptive bounds.
These signs do not establish resolved economic improvement relative to the
larger prior uncertainty. None of the six books has a Cielo loan cash redemption;
its already frozen cent variants remain conditional on future actual exposure.

## Attempts and verification

Five distinct new tests and 53 affected existing tests qualify. The initial new
batch had four passes and one fixture failure: chosen 2024 notionals happened to
produce exact cent fees, contradicting the fixture's expected nonzero difference.
Only the test target was corrected. The combined batch then had 57 passes and
one exact-equality fixture failure from 2.22e-16 funding roundoff. A narrow 5e-15
absolute tolerance in that comparison fixed it; the two affected parameter cases
passed. Overlapping batches are not additive, and `qualified_tests` contains
the second failed attempt rather than an all-passed run. Exact failed/qualified
test bytes and stdout are retained. Neither correction changed production or
reran any historical book. Gradient, independent-copy and detach checks pass.
Executed production bytes equal the current implementation; Ruff passes.

Audit time was 10.6133398 seconds (sum of cases 9.5467013); independent saved
qualification took 0.1208804 seconds on CPU. These are engineering runtimes,
not GPU-fit estimates. Preliminary read-only path/PowerShell errors and a plan
iteration error occurred before any resolved plan/book; prose descriptions are
labelled reconstructed, not falsely presented as original failed command bytes.

## Bounded calendar source result

The www-host dated B3 index recovered original 131/2015-DP, December8 2015,
and all five scanned pages were visually read. It lists December30 2016 as a
non-trading date and gives separate FX, OTC and Treasury exceptions. It does
not explicitly establish the former equity clearing house's delivery calendar.
Those other-market rules cannot be imported into equity settlement or loan rent.

Two manual leads were also retained. A BM&FBOVESPA-authored manual hosted by
SITA is dated March2011; no 2016/2017 revision continuity was established. The
B3-hosted February24 2017 manual explicitly says it is **not yet approved** by
the BCB/CVM. Its page134 trading-day settlement definition therefore cannot
establish the older dates. Their cover pages and the draft's page134 were visually
qualified (three pages), without claiming a complete manual audit. The selected
Modal mirror returned403; a targeted official-gazette URL for an Elektro erratum
returned404. A secondary search reproduction of that erratum is only a lead,
not admitted dated primary evidence. Poppler font warnings did not prevent
readable qualification. Successful source recipes, originals, receipts, extracted
text and selected pages are preserved; no repeated source census.

The previously accepted November20 2017 two-source inference remains intact.
December30 2016 and January25 2017 remain explicitly unresolved, with **no
calendar/model array change**. The next composition block must give these dates
specific sourced dispositions or separate bounded hypotheses; it must not add
all monetary dates, treat non-trading-day cash settlement as a fictitious trade,
or silently make delivery and rent-accrual calendars identical. This search is
complete as a bounded attempt, not a source-completeness claim or a calendar bound.

The remaining delivery blocks are corporate/calendar/source composition,
including actual succession-data implications, followed by integrated Stage A
admission. Exact broker invoices, obtained quotes and unknowable archive revision
completeness are not new prerequisites. C/D remain unstarted.
