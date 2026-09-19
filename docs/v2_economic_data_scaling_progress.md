# Economic accounting, data audit and scaling progress

2026-09-19: user authorized accounting repairs, a deep source-to-model audit,
additional historical B3 retrieval and economic-primary attention/GRU capacity
follow-ups. Read `research/preregistrations/v2_economic_data_scaling.md` after each
stage. Foundation outcomes remain sealed at the canonical foundation pointer.

Initial inspection: the account supports only one scalar successor and rejects a
successor already held; multi-leg Copel delivery and mixed contractual consideration
need explicit support. Borrow charges currently use each day's marked exposure and
latest causal archive rate, rather than carrying a fixed contract reference/rate.
These are model assumptions to reconcile with the dated B3 contract, not yet fixed.
Published outstanding lending balances are used as shortability evidence and are
not proof of free locate capacity. The user specified a likely retail-adjacent
account and zero brokerage; exchange and borrowing costs remain separate.

No new GPU fit has launched. Next: source-backed accounting/borrow specification,
source-to-store audit map and prioritized reproducible defect evidence, then fixes
with matched replay. Preserve held-out consumer restrictions and immutable stores.

First source-backed repair: BDI registered-loan extraction now reconstructs printed
wrapped dates, tickers and ISIN check digits, and unambiguously separated thousands
groups where quantity/BRL columns touch. Inspection of 2024-11-27 page 67 confirmed
the physical table; the complete bulletin reconciles 1,632 printed numerical rows.
The arbitrary 100-row minimum is replaced by printed-row reconciliation so small
complete tables survive and partial modality extraction cannot silently bias rates.
Four regression cases plus existing lending tests pass (14 targeted tests).

The active Round-6 economics pointer still references the original lending archive:
no new Round-5 rate vintage was admitted. That archive starts actual rates on
2023-07-10, with a 2% pre-history placeholder in the consumer despite contradictory
older manifest prose. Existing books retain that source. `ops/audit_lending_sources.py`
reparses the admitted-rate era and all cached later-2024 PDFs, plus quarterly earlier
source checks, into a separate audit directory with per-PDF hashes and failures.
Recovered rows are not yet admitted into features or accounting. Zero-flow printed
carried rates do not become fresh trades. Brokerage is assumed zero; published
taker/donor rates remain separately observable proxies.

Source reconciliation follow-up: the 390-PDF census exposed independently wrapped
identifiers in older bulletins and an AMER3 currency/300% rate column collision.
The parser repairs only printed text; it does not clip this high borrow rate or
guess identifiers. All 16 targeted tests pass. Initial extraction-attempt evidence
is retained separately; the complete census is rerun against each amended parser
hash. Final source admission and causal feature rebuilding still remain.

The actual C6 economic cache was hash-verified and inspected: it has 1,730 date rows
of the 2% placeholder, with the first non-placeholder date 2023-07-11. The binding
and source hashes are in the new run's `active_borrow_binding.json`.

`ops/audit_daily_sources.py` reconciles all 1,206,107 observed stock-days to the
accepted COTAHIST sources, including 32,530 previously repaired BDI continuation
quotes: zero missing-source observations and zero mismatches across OHLC, quantity,
volume and trade count at the stored dtype. There are 9,561 published-average versus
volume/quantity exceptions, confined to 2010-2017. These are audit candidates,
not established errors or permission to overwrite published prices. A raw 2011
BRIN3 record reproduces the discrepancy, so it is not introduced by our parser.
The unusually high nominal price alone is not evidence of an error.

Dated B3 sources are archived and hash-bound in `docs/v2_economic_data_loan_sources.json`.
Circular 125/2020 establishes electronic lending from 2020-10-26; 081/2022 changes
caps on 2022-11-14 and specifies transition treatment. Each fee component has its
own alpha/floor/cap; current normal electronic fees are not a historical universal
tariff. The B3 contract fixes the loan reference to the previous average quote,
with effective annual rent, accrual and payment at return/renewal. Loan accounting,
dated fee integration, pre-2020 rules and broker remuneration remain to implement
or bound explicitly. No account-repair or architecture outcome is claimed yet.

Resume in this order: inspect the current loan audit worker's actual command and
result; resolve any remaining printed-row failures, quantify eligible-name changes
and recover cached pre-2022 balance history using contemporaneous COTAHIST identity.
Then implement contractual multi-leg/delivery/claim accounting coherently in both
accounts and replace fabricated missing-quote liquidation. Complete the source-to-
tensor family audit and evidenced derived-store changes before the matched economic
replications and capacity contrasts in registration C-D. Both stages A and B remain
in progress; do not mistake these two source checks for the full data audit.
