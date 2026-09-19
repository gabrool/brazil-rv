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

### Source audit follow-up, 2026-09-19 13:30 UTC

The registered-loan census is complete: 390 PDFs, 320,132 printed rows,
zero extraction failures, and every 41,353 accepted rate observation recovered.
The repaired parser hash is
`310bd650446d2b3fbd11e8aa628b24cbd48d744815159a626935d5f8dcecb592`.
Across model security axes there are 58,881 additional rate observations; restricting
to stocks eligible at their causal next-session availability leaves 25,728 additional
observations and 2,916 changed existing rates. Of the changed eligible observations,
2,142 decrease and 774 increase. The average absolute change over all 38,819 eligible
overlaps is .00009837 in annual-decimal units, so count alone must not be mistaken
for a large economic effect. `eligible_attribution.json` binds this calculation;
`ops/audit_lending_sources.py` reproduces it without altering any consumer store.

The same source contains 344 positive-flow BOVA11 days, including the last bulletin
whose next-session availability lies outside development. Its annual rate median
is .8215%, range .1417%-2.7992%. The current independent ledger floors a supplied
hedge rate at 2%, while the training account always uses a fixed 2%; the main
EvaluationInputs contract currently carries no hedge-rate series. Both also need
consistent B3 fee treatment. These are additional concrete account-path defects
to fix during A, not a measured P&L improvement or an executable borrow quote.

The earlier-balance audit resolves its 587 original PDFs through the accepted source
manifest, not a recursive data scan. It uses the actual balance date's COTAHIST
ticker/ISIN and report-D+1 publication timing. Repairs now preserve touching large
quantity/BRL columns, a printed `02` market code touching the ticker, wrapped BRL
amounts, and boundaries before account-margin/custody tables. Named rows reconcile
and published BRL totals are checked where printed. Source inspection recovered
ITSA4's previously omitted 2019-11-01 row and removed false `02ITSA3` identities.
Twenty-five targeted lending/parser/archive tests pass.

The completed `c307d68` balance census has 229,416 parsed rows, of which 149,130 map
to model identities, spanning balance dates 2019-10-31 through 2022-03-17. It has
zero conflicting ISIN assignments, 94 quarantined PDF attempts, and three no-table
receipts. Sixty-two failures concern duplicate printed tickers; a rendered original
2020-10-27 page also shows an unidentified balance row. Other failures include
touching ungrouped numbers, wrapped company/value lines and published-total
differences. These are preserved audit exceptions, not blanket final exclusions.
Review remaining recoverable fields and per-security admission rather than dropping
good securities because another row on the same page is ambiguous. Unmapped legacy
tickers receive no guessed identity. None of these new balance rows has yet entered
the model, so the accepted information set has not been reduced.

Attempt receipts are saved separately (`extraction_attempt_f8fae21.json` and
`extraction_attempt_44c3376.json`); `lending_balance_audit_worker.json` binds the
latest worker and log names. It has finished. A bounded PyMuPDF extraction trial
recovered some spacing but left genuine source ambiguities; no dependency or parser
switch was adopted. Avoid rerunning complete successful censuses without a source
or parser change. Inspect representative failed dates directly next.

Archived B3 001/2020-VPC describes the change from a fixed annual fee with a BRL
minimum per contract to rate-dependent fees, but explicitly leaves implementation
timing for a later announcement. It does not establish January 2020 as the effective
date. The 049/2020 formula erratum is also archived. Pre-2020 tariff levels, the
actual implementation date, modality-specific aggregation and retail financing
remain to establish or bound. Execution brokerage stays zero by user assumption;
do not double-count borrower intermediation already embedded in published taker
rates, or assume short-sale proceeds earn CDI merely because brokerage is free.

**Current boundary:** A/B are still in progress. No accounting transition has been
changed, no recovered source is admitted, and no GPU follow-up has started. Next
implementation must handle multi-leg/delivery/netted succession and remove synthetic
last-mark fills in both accounts, with meaningful conservation/causality tests and
explicit source provenance. Finish the remaining family/label/tensor audit and
causal source admission before C-D. The active heartbeat continues this program.
