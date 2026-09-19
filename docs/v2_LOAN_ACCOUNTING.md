# Loan accounting: verified terms and implementation boundary

2026-09-19. Stage A of the economic/data/scaling program is still open.
This document distinguishes the implemented tariff and contract mechanics from
remaining settlement terms and historical source admission. It is not a new performance readout.

## Implemented dated exchange-rate schedule

The old code applied one modern normal-electronic fee formula to every date,
combined its components before compounding, and did not distinguish modality.
Both replay and training now use the same dated component schedule; the independent
account calculates the equity and hedge charges separately. Readouts split total
loan rent from B3 loan fees, including the hedge. The old configurable aggregate
fraction/floor/cap implementation is removed.

| Accrual period | Normal electronic | Direct electronic | OTC |
| --- | --- | --- | --- |
| Before 2020-10-26 | Ordinary voluntary loan: fixed 25 annual bps | Same pre-platform convention | Same pre-platform convention |
| 2020-10-26 through 2022-11-11 | Trading 2% of rent, 0.25–10 annual bp; post-trading 18%, 2.25–90 bp | Trading 2.5%, 0.6–15 bp; post-trading 18%, 4.4–110 bp | 30%, 5–150 bp |
| From 2022-11-14 | Trading cap 7 bp; post-trading cap 63 bp | Trading cap 10 bp; post-trading cap 85 bp | Cap 120 bp |

The other percentages/floors stay unchanged at the 2022 transition. Compulsory
borrowing has its own schedule and is not assumed for ordinary positions.
Every component clips, rounds and compounds separately. For example, a 4.2%
direct loan rate produces 10 + 75.6 = 85.6 annual fee bps under the later table;
clipping an aggregate 20.5% formula would incorrectly charge 86.1 bps.

The initial 001/2020 announcement explicitly deferred implementation; its January
publication is not a fee-effective date. The November 2022 reduction applies to
the applicable accrual interval of existing contracts, not their entire past life.
The new code has tests for these boundaries and separate component binding.

Normal electronic sourcing remains an explicit research assumption after platform
launch. It is not proof that a published blended rate was executable in that
modality. Actual modality/rate-source admission remains open. A fee multiplier
scales the calculated monetary charge for controlled sensitivities; it does not
rewrite the exchange's published annual rates.

## Source receipts

[Dated source manifest](v2_economic_data_loan_sources.json) binds the archived
documents and their hashes on the resolved program root.

- The [historical exchange cost page](https://bvmf.bmfbovespa.com.br/pt-br/servicos/custos-e-tributos/custos-operacionais/acoes.aspx?idioma=pt-br)
  gives voluntary lending at 0.25% annually, minimum R$10, and automatic lending
  at 0.50%. The archived BDR loan guide independently states the same market fee.
  Its filename's 2018 date is a retrieval/search classification, not an inferred
  effective date. No unique start date for the old lending tariff is established;
  its extension across the earlier development dates remains a stated assumption.
- B3 125/2020-PRE establishes the platform launch and its fee table. B3
  049/2020-VPC corrects a misleading formula-unit presentation.
- B3 081/2022-PRE supplies both historical tables and the November transition.
  Its table was checked visually as well as by text extraction.
- The B3 loan contract specifies a prior-session published average price (last
  available if necessary), contracted annual rent, business-session accrual and
  payment upon delivery/return or renewal.

## Implemented contract mechanics

Both accounts now keep sparse fixed-reference, fixed-rate loan cohorts. An increase
registers only its new borrowed quantity using the supplied causal published-average
reference and annual rate. Later market prices or lending observations do not
reprice an existing contract. Same-security splits change deliverable quantity but
preserve principal. Partial returns retain the original terms and divide accrued
liabilities pro rata. Separate tariff intervals preserve earlier accrued fees.

Accrual excludes registration and includes return. Rent and B3 charges are expenses
and unpaid liabilities; cash pays them on the modeled return date. NAV subtracts
that liability, preventing either premature cash deductions or a second loss on
payment. Buying to cover removes market exposure but the loan persists until its
return settles. A terminal cover does not extinguish an outstanding loan or its
accrued liability. No future holding cost is silently charged on the final date.

For pre-platform voluntary contracts, the R$10 minimum is provisioned once on first
accrual, rather than once per partial return. Actual partial fees are paid on their
return dates; any residual minimum is paid on final return. This last payment-timing
choice is an explicit research assumption; the historical source establishes the
contract minimum but not that partial-payment allocation. Returns are allocated pro
rata across open loans in the same security, without selecting expensive contracts
using hindsight. One daily borrowed increase per security is treated as one contract;
a published aggregate rate does not establish the real broker's contract grouping.

The primary return hypothesis is a prearranged return when the covering spot trade
settles: T+3 before 2019-05-27 and T+2 thereafter. B3's dated [T+2 announcement](https://www.b3.com.br/pt_br/noticias/liquidacao.htm)
is archived in the source manifest. Same-settlement return assumes the requisite
request/custody cutoffs are met; it is not a universal broker service guarantee.
The B3 contract supports payment at partial/full return or renewal. Delivered
corporate offsets can return shares already in custody that session, subject to
source-specific admission. Generic market settlement remains separate work below.

Costs retain their original entry identifier through partial returns and successor
transfers. Liquidity attribution now uses actual cohort charges instead of
recalculating current-rate costs. Borrow-quality shares use outstanding fixed equity
loan principal, including pending returns, and retain opening-observation provenance.
Evaluation V24 hashes loan references and binds them and annual rates in paired
comparisons. Readouts expose loan liabilities, payments, outstanding principal and
sparse original-entry charges. They are monetary values in the declared capital units.

The cash/share accounts remain separate implementations, but intentionally share
one vectorized loan subledger. Agreement between accounts therefore does not supply
an independent oracle for loan arithmetic: closed-form contract-formula tests do.
SAM/TBPTT starts copy all loan tensors and metadata independently; mutable corporate
claim containers are copied as well. Loan state survives truncation without retaining
an earlier gradient graph. Replay batches actual same-session fills by name; it does
not change the accepted fills, dates, eligible population or training budget.

## Still required before full accounting acceptance

1. Admit causal published-average reference panels, including BOVA11, and the
   recovered donor/taker observations and source dates. No missing opening reference
   is replaced with a convenient current price. Old accepted stores/caches do not
   contain this new panel and must be explicitly enriched in a new bound replay
   artifact; they are not silently rescored. Preserve published averages even where
   turnover/quantity differs. Confirm source identity and exact units.
2. Admit actual custody and clearing-calendar boundaries alongside the implemented
   dated spot cash/proceeds settlement below. Freeze a renewal/maturity/recall convention and sensitivity using
   source terms; the current subledger does not invent daily renewals, an unlimited
   legal loan maturity, or a broker renewal toll. The B3 contract permits at most
   two years, so unresolved long-lived inventory needs explicit treatment before
   historical admission.
3. Admit each corporate case's loan terms. Basket transfers now require explicit
   loan principal fractions that sum to one, independently of market-value allocation
   of inventory cost basis and proceeds. Original principal/rate survive staged
   delivery and returns. Copel's unquoted issuer allocation uses the explicitly
   labelled 20% ON / 80% PN research case and the full 0%-100% ON sensitivity;
   it is not a recovered contractual K. Existing-listed successor mechanics are covered; nested actions,
   unavailable successor marks, loan obligations during a delivery interval,
   fractional entitlements and event-specific cash/loan settlements need evidence.
   Cielo and BR Malls have source amendments; Copel's shares/custody are sourced
   with a separately bounded loan-allocation assumption. These do not finish A.
4. Check minimum billing at partial returns and old/new tariff transitions against
   invoices/manual terms. Current calculations use continuous float64 currency for
   differentiability, rather than invoice cent truncation. Measure/bound these small
   rounding differences at the registered capital sizes before final acceptance.
   Modality, locate capacity and broker intermediation remain explicit assumptions.
5. Use R$10m for the registered primary replay and R$1m/R$5m sensitivities. The generic
   engine's R$1 normalized test/default account cannot stand in for actual capital
   when an absolute R$10 minimum applies. Capital is now recorded in the evaluation
   contract. No ordinary-CNPJ fund discount or unsupported retail surcharge is added.

## Input compatibility and verification

Forecast tensors, accepted stores, sealed fit roots and serialized policy caches
are unchanged. Newly constructed policy data contain a dated borrowing-cost feature;
their implementation and full pickle hash are already recorded and checked against
trained-policy provenance. Do not regenerate that coordinate and feed it silently
to a policy trained on an old cache. Accounting-only replay must retain the frozen
policy feature coordinates or explicitly refit and report a separate contrast.

Validation of this milestone: 247 targeted tests passed, followed by 156 affected
checks after batching loan fills (overlapping, not additive), and 65 final reporting/attribution/minimum checks passed. These cover source
formulas, payment conservation, partial returns, terminal liabilities, fixed-rate
future mutations, gradients, independent SAM restarts and downstream attribution.
Two unaffected synthetic paths are bit-identical to `4e93bde` on 13 account arrays
and fills/intentions. The acceptance receipt is `v2_loan_contract_acceptance.json`.

A synthetic full-population CPU measurement (243 names, 252 sessions, 15,528 fills)
took 1.44 seconds for contractual replay versus 0.22 seconds for the former proxy;
64 sessions of differentiable account forward/backward took 0.26 seconds. This is
an account benchmark, not a neural training forecast or a statistical speed claim.
It preserves every supplied name. Historical data, fits and profitability were not
used to select the implementation. Source admission and all remaining stages stay open.

## Spot cash value dates and settled-proceeds income

The accounts now distinguish trade-date beneficial inventory from spendable settled
cash. A fill records a dated free/restricted-cash obligation, using T+3 before
2019-05-27 and T+2 thereafter. Actual fills and their costs enter economic NAV on
trade date; the money is available on the value date. Same-value-date receipts and
payments net. A final-window fill leaves its future obligation in NAV, rather than
manufacturing a final cash transfer. This does not impose an additional delay on
trading beneficially owned inventory or discard any names.

Free-cash income, short-proceeds income and debit financing are reported separately.
For the close-to-close interval ending on session t, funding uses settled balances
carried from the previous close. A payment or receipt during t affects funding from
the next interval. The initial account is treated as funded before the first session,
matching the all-cash CDI benchmark. This is an explicit daily investment-cutoff
convention; it neither grants income before receipt nor charges a prospective
purchase payable as an already drawn debit. A restricted deposit earns the configured
CDI fraction but cannot simultaneously finance long holdings. The primary remains
100% CDI on eligible settled proceeds and zero execution brokerage.

End-of-session conservation is:

`NAV = settled free cash + settled restricted proceeds + unsettled net cash
       + marked beneficial holdings + signed corporate cash claims - loan liability`.

Original-sale restricted cash remains until the covering purchase settles, including
a cover executed before the sale's own settlement. Releases cannot withdraw a
receipt that has not arrived. Successor mechanics transfer pending proceeds along
with their associated basis. A cash-only cancellation retains the short proceeds
until the stated payment date, or indefinitely while payment remains unknown.
Known due dates can affect the decision state; future marks, fills and payment
realizations cannot rewrite earlier intentions. SAM copies and TBPTT truncations
retain independent dated cash obligations and the interval's funding snapshot.

The [2021 B3 clearing manual](https://www.b3.com.br/data/files/17/92/CA/18/2DE377108F39C077AC094EA8/Manual%20de%20Procedimentos%20Operacionais%20da%20Camara%20B3_20210126.pdf),
archived and hashed in the source manifest, supports client-level multilateral money
netting (p.160). It also distinguishes D+0 and D+1 loan accrual conventions (p.112),
renewal approvals/reference repricing (pp.109–111), and actual custody authorization
(pp.166–168). Those distinctions must be resolved for the admitted loan modality;
the earlier fixed-loan mechanics receipt does not establish a complete broker contract.

Remaining boundaries: the helper counts the supplied trading-session axis, so unusual
clearing-calendar closures still need reconciliation. Ordinary fills assume timely
delivery; the new queue tracks money, not a complete physical custody inventory.
Corporate offsets against a recently purchased but not-yet-delivered holding must
not be admitted as an immediate loan return without evidence or a custody extension.
Event-specific principal allocation, pending share claims, loan maturity/renewal/recall
terms and the source panels remain pending. A sourced settlement date does not itself
establish a broker's investment sweep cutoff or an actual negotiated package.

Forecasts, accepted stores, caches and fit roots are unchanged. Serialized static
policy features are preserved. The current account's dynamic cash state changes
as part of the accounting repair and must be labelled as such in an accounting replay;
the evaluation schema is V24. The spot receipt binds synthetic correctness checks
and timings, not historical profitability or completion of Stage A.

## Recovered source inputs

The economic/data/scaling run pointer binds the newly admitted reconciled lending
archive separately from the sealed Round-6 archive. It expands positive-flow rate
coverage and earlier balance history, with the same causal 60-session own-rate and
cross-sectional fallback rules. The 2% pre-July-2023 placeholder is unchanged.
Taker and donor averages are retained; their difference is not asserted to be the
account's executable intermediation charge. Borrowing availability can change and
needs its own replay attribution rather than being labelled a pure cost adjustment.

The stock/hedge source panel implements section 3 of the archived B3 loan contract:
the previous session's published average, or the last available one. It retains
the same-ISIN publication date and never substitutes a current mark. These prices
are contract references, not observed model inputs, executable marks or labels.
The contract does not establish a generic adjustment based on successor prices;
event-specific adjustments still require source terms before the affected accounts
are admitted. The first model-store date lacks 125 stock references; all subsequent
eligible stock-days and every hedge session have prior references. No new loan may
silently replace a missing reference. BOVA11 observed rates are aligned separately,
with NaN explicitly delegating to the configured fallback, and do not contaminate
the equity cross-sectional imputation universe. See `v2_lending_source_admission.json`.

## Compulsory loan cash settlement

Evaluation V25 admits a distinct source-bound loan cash event. It cash-settles all
outstanding contract quantities on the declared date, including requested returns
due that day or later, and retains the original principal/rate for rent and B3 fees
through payment. This principal cash is not a borrowing expense or a shareholder
redemption. Both accounts retain a covering asset when its planned physical return
is superseded and preserve external spot cash value dates while releasing the
extinguished loan's restricted proceeds. Prior intentions do not read the final
settlement price; no new shorts are allowed from the declared effective date.

The Cielo B3 circular supplies the concrete reason for this distinction. The source
receipts and issuer timetable are described in `v2_CORPORATE_EVENTS.md`. Its offer
document calls for SELIC correction after the auction, distinct from the pre-auction
CDI adjustment. The recovered rates support a continuous calculation, with invoice
rounding still bounded separately. Neither a later shareholder cash price nor an
unverified current mark may substitute for the contractual loan closeout amount.

Synthetic parity, source-clock, cash conservation, gradient, policy constraint and
input-hash checks are bound by `v2_loan_cash_acceptance.json`; ordinary books preserve
bit-identical paths against commit 601b75a. General physical custody, failed-delivery
and restored holding-basis metadata are not established by the superseded-return
cash/NAV fixture. Historical event-axis admission remains separate from these
mechanics, and model-data changes require their own contract and matched refits.

## Purchase custody and shareholder fractions (Evaluation V26)

Scheduled owned purchases carry actual spot value dates through succession.
Netting beneficial holdings does not itself return a loan. Corporate offsets use
settled owned shares first, then earliest undelivered receipts; each portion retains
loan accrual and restricted proceeds until custody is available. External trade cash
dates remain unchanged. Cover purchases are already committed to their scheduled
loan return and cannot be reused. Return cohorts convert even when beneficial source
inventory is flat. SAM copies and TBPTT retain independent custody receipts.

This addresses regular-way custody without a failed-delivery simulator. Special
return overrides may postpone covering-asset availability; broker cutoffs and failed
deliveries still need explicit conventions. The single-leg BRML fraction auction
requires source purchases to settle before corporate credit, as its Jan6 last trade
and Jan11 credit permit. Multi-leg/nested fractional auctions remain unadmitted until
their specific allocation and registration terms exist. Long fractions remain in
the existing share-claim machinery, including constituent risk and causal marking,
and become cash receivables only when the later auction result is available.
BRML loan fractions remain contractual quantities under its explicit circular;
Dommo's separate provisioned-fraction treatment is described below.

The source/calendar oracles are bound in `v2_corporate_replay_acceptance.json`.
Runtime verification preserves bit-identical ordinary book cash, NAV, loans, fills
and intentions versus the previous source commit. Loan attribution now omits only
identically zero rent/fee rows for old paid roots; every nonzero original-entry
charge and all money arithmetic remain unchanged. This avoids unnecessary output
work in long replays with frequent trading.


### Deferred loan redemption and cash-calendar amendment

Evaluation V29 separates `loan_redemption_liability` from the total loan liability.
Cash election removes physical quantity at its event, retains original accrued
rent/fees until their own payment date, and carries the marked redemption liability
until actual payment. No rent accrues after the source-bound extinction date.
Dated value updates run after the current intention; SAM/TBPTT copies own independent
claim tensors. An active-only lender election excludes entire original roots with
pending returns and new same-day D+1 loans. This differs from an all-contract
compulsory closeout, which supersedes late physical returns and preserves the
covering assets. Dommo's PNB endpoint is admitted separately; its default PNA loan
fractions are not converted into tradable fractional loans by this change.

The deep cash-source audit traces `_fetch_cdi` through `load_daily_cdi_rates`, the
V2 loader and actual accounting inputs. Raw SGS12 units/dates were preserved, but
same-date equity alignment omitted 23 monetary dates and used the ending date's
rate for a prior-close interval. The explicit new `cash_calendar` panel compounds
source dates in [previous equity close, current equity close). It changes 67 of
2,098 supported sessions; cash-benchmark mean rises 0.030318289 bp per equity
session. This is not strategy alpha. Old fits/features/results remain frozen;
corrected replay must explicitly apply the panel to all income/debit/benchmark
paths. Early unsupported intervals remain NaN and fail requested replay; no
fabricated rates. Loan day-count and actual clearing closures still need the
separate lifecycle audit. All ten auxiliary producer-to-store paths have now been
independently reconciled, but their upstream publication/revision clocks and final
fit conditioning still require the remaining deep audit.

### Provisioned loan fractions

Default Dommo conversion now truncates each original loan independently and keeps
its original reference principal/rate on the whole successor quantity. The signed
fraction is a marked share claim, followed by a payable when auction terms become
available, not a tradable loan share. Its proportional proceeds remain restricted
until cash payment. The separate tiny-zero-quantity rent endpoints and the actual
last-trade/settlement boundary are recorded in `v2_CORPORATE_EVENTS.md` and the
source-bound manifest. SAM/TBPTT retains differentiable fractional claims; a
finite-difference check protects the quantity-dependent proceeds allocation.
Previously known corporate cash claims pay before the decision together with
their known proceeds release; newly announced same-day cash remains a realization
after the intention. Prior-close balances still determine interval income.
Old ordinary books remain bit-identical against d7c6a2f. See
`v2_dommo_conversion_acceptance.json` for actual-calendar arithmetic evidence.
