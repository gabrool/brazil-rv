# Loan accounting: verified terms and implementation boundary

2026-09-19. Stage A of the economic/data/scaling program is still open.
This document distinguishes the implemented tariff repair from the remaining
contract ledger and source admission. It is not a new performance readout.

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

## Still required before accounting acceptance

1. Replace daily marked-notional/current-rate rent with contract cohorts that retain
   their registration reference and rate. Increases need new terms; partial returns
   preserve the remaining contract. Track accrual as a liability until actual
   payment, rather than immediately removing all daily accrual from free cash.
2. Model cash and security settlement, repayment delays and renewals coherently.
   Separate beneficial short inventory from loans still outstanding during return.
   Do not invent an automatic daily renewal or an unsupported broker notional toll.
3. Integrate the pre-platform R$10 voluntary-contract minimum once, at its evidenced
   contract/settlement boundary. The implemented annual-rate function does **not**
   include this monetary minimum. Current account results are therefore not accepted
   as full historical contractual economics. Check its incidence at R$1m/5m/10m.
4. Use published COTAHIST average references, including BOVA11, without replacing
   the source averages with turnover divided by quantity. Fixed-loan transformations
   through cash distributions, splits and successor baskets require the applicable
   B3 event terms. Generic shareholder-wealth conservation is not sufficient proof
   of the transformed loan principal or payment terms.
5. Admit the recovered donor/taker rates, modality and publication timing, preserving
   unknown locates/intermediation as explicit assumptions. No new historical rates
   or corporate events have yet been attached to the accepted data or old fits.

Evaluation V22 labels the remaining daily marked-notional proxy and missing
contract minimum. Source auditing, repayment mechanics and financing terms remain
Stage A work; no economic candidate is accepted from this intermediate revision.

## Input compatibility and verification

Forecast tensors, accepted stores, sealed fit roots and serialized policy caches
are unchanged. Newly constructed policy data contain a dated borrowing-cost feature;
their implementation and full pickle hash are already recorded and checked against
trained-policy provenance. Do not regenerate that coordinate and feed it silently
to a policy trained on an old cache. Accounting-only replay must retain the frozen
policy feature coordinates or explicitly refit and report a separate contrast.

Targeted tests cover the dated component tables, stock/hedge agreement between
NumPy replay and Torch accounting, uniform and sourced rate scenarios, conservation,
future-rate mutation, gradients and downstream reporting. Synthetic unaffected
long-only and fee-disabled long/short paths are bit-identical to `b1eed6c` on 13
accounting arrays, fills and intentions. These checks establish implementation
correctness for this change; they do not establish historical profitability.
