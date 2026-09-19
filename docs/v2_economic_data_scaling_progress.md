# Economic accounting, data audit and scaling progress

2026-09-19: user authorized accounting repairs, a deep source-to-model audit,
additional historical B3 retrieval and economic-primary attention/GRU capacity
follow-ups. Read `research/preregistrations/v2_economic_data_scaling.md` after each
stage. Foundation outcomes remain sealed at the canonical foundation pointer.

**Current status:** A/B remain in progress; C/D have not started. Corporate-account
research and the source censuses below are complete. BOVA11 borrowing-path
consistency, removal of fabricated missing-quote liquidations, inventory netting and
multi-leg/delayed-delivery, fixed-contract loans and dated money settlement are
implemented and tested. Reconciled lending observations are now admitted to a
separate economic archive; prior stock/hedge references are bound. Event-specific
terms and full historical account acceptance remain open. Cielo's source-based
accounting artifact and ALLOS/ISA rename admissions are now verified as described
at the end of this log; a corrected full model store remains pending. No training or
source-census worker is active.

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

**Boundary at the source-census checkpoint:** A/B were still in progress. No accounting transition had been
changed, no recovered source is admitted, and no GPU follow-up has started. Next
implementation must handle multi-leg/delivery/netted succession and remove synthetic
last-mark fills in both accounts, with meaningful conservation/causality tests and
explicit source provenance. Finish the remaining family/label/tensor audit and
causal source admission before C-D. The active heartbeat continues this program.

### Corporate account amendment, 2026-09-19

The user replaces the initial retail-adjacent account with a well-capitalized CNPJ
low-touch setup and explicitly requires interest on short-sale proceeds. The
registration and PROJECT_CONTEXT now bind `docs/v2_CORPORATE_ACCOUNT.md`: primary
100% historical CDI on eligible settled proceeds, zero execution brokerage, no
unsupported retail loan-turnover surcharge, and individually attributed B3,
borrowing, financing and execution costs. R$10m is an explicit planning assumption
with R$1m/R$5m sensitivities, not confirmed capital or admission minimums. Public
provider capabilities and tariff evidence are separated from unquoted negotiation
targets. No-interest proceeds is not a required new admission case.

The existing accounts already remunerate proceeds at 100% CDI; the account choice
does not automatically improve the old results. Their old 4 bp execution cost is
bundled: do not add full B3 charges to it again. Current B3 cash-market discounts
are progressive and differ from historical ordinary-CNPJ versus local-fund fees;
current loan charges also depend on modality. Public standard XP borrowing costs
include a 0.25% notional liquidation/renewal commission, not a 0.25% annual rate;
that retail-style schedule is not imported as the negotiated low-touch default.

Thirteen official-source retrieval receipts are recorded in
`docs/v2_corporate_account_sources.json`; ten original snapshots were archived on D.
Three direct downloads returned HTTP 403 although web-indexed text was available.
No quote was obtained or outreach sent. Source/data consumers, sealed books and
training defaults have not changed in this amendment. Complete the actual Stage A
accounting and Stage B audit before interpreting corrected economics or starting
the registered follow-up fits.

### BOVA11 borrowing-path consistency repair, 2026-09-19

The independent ledger no longer floors a supplied hedge loan rate at its 2%
fallback: finite observed rates, including zero, are used as supplied; only missing
observations use the configured fallback. EvaluationInputs now carries the causal
hedge-rate series into the ledger, sliced replay windows and input hashes. Evaluation
schema V18 distinguishes this economic contract from previously sealed reports.

The policy allocator and differentiable account consume that same series. Borrowing
costs are calculated from input rates and the actual account configuration rather
than a cached array constructed with default charges. Stress replays retain the
primary decision estimate while changing realized costs. The configured B3 loan fee
is now applied to hedge borrowing in both accounts, as it already was to individual
equities; long hedge positions pay no borrowing charge. This removes an omitted
cost as well as the overly high observed-rate floor, so it is not an unconditional
performance improvement. Historical accounting results remain sealed.

Validation: 149 targeted independent-ledger, differentiable-account, policy and
evaluation tests pass. New cases cover rates below/above fallback, observed zero,
missing observations, long versus short hedges, alternative fee scenarios, a replay
starting partway through the source series, future-rate mutation and provenance.
Both accounts agree on daily NAV to 1e-12 in the targeted hedge cases. Ruff passes.

This is a bounded path repair, not final loan-source or contractual-accounting
acceptance. No recovered historical lending series has been admitted or backcast,
no old policy cache has been rewritten, and no financial comparison has been rerun.
The daily marked-notional/latest-rate accrual and static fee schedule still require
replacement by the registered fixed-contract reference/rate and dated fee rules.
Multi-leg corporate actions, delivery/payment claims, existing-destination netting
and removal of synthetic missing-quote liquidation remain the next accounting work,
alongside causal source admission and the remaining source-to-model audit. No GPU
follow-up is launched before those contracts are ready.

### Missing-quote inventory repair, 2026-09-19

Removed synthetic last-mark liquidation after ten missing quotes and at an
evaluation boundary in both accounts. Unavailable equities and BOVA11 now remain
signed inventory: no fabricated cash release, trading cost, turnover or short
cover. Restricted short proceeds remain restricted, financing/borrow continue and
pending exits execute only when a real quote returns or a contractual event resolves
the claim. The ten-session rule requests an exit; it no longer fabricates its fill.
Removed permanent post-settlement exclusion, redundant settlement intentions,
their cancellation branches, dead exclusion counters and the terminal-settlement
configuration switch throughout active callers. A returning quote executes at its
observed price and the security may subsequently re-enter normally.

Evaluation V19 and book readouts expose daily unpriced inventory/count/NAV fraction,
remaining terminal inventory and the unresolved-economics flag. Summary inventory
fields describe the terminal state; the material-exposure flag uses the maximum
daily fraction. The configured 30% haircut is only a transparent valuation
sensitivity on currently unpriced inventory. It does not affect cash, orders,
financing or headline NAV and disappears when a quote resolves the uncertainty.
It is neither a collateral haircut nor an asserted recovery value. Comparisons keep
the common calendar and disclose unresolved exposure instead of silently dropping
difficult folds. Previously sealed reports/results remain unchanged; historical
source-report inspectors still read their recorded former fields.

Validation: 209 targeted ledger, policy, account, causality, evaluation, checkpoint,
readout and pipeline checks pass (208 in the broad run, the remaining numerical
check after its bounded test correction). New/revised cases verify long and short
outages beyond ten sessions, retained terminal hedge positions, restricted cash,
continuing rent/CDI, resumption at a changed quote, subsequent re-entry, absence of
synthetic fees and unchanged primary NAV under valuation-haircut changes. Both
accounts reconcile. Ruff and diff checks pass. A pre-existing loss test demanded
bitwise equality from batched versus per-head float32 reductions: its maximum
gradient difference was 1.862645149230957e-9. The test now uses strict numerical
tolerances; no loss/training implementation changed. Obsolete golden hashes of
pre-repair full reports were removed rather than replaced with meaningless new
goldens; score-only and independent-account agreement remain directly tested.

The resolved run root contains `missing_quote_acceptance.json` and its SHA-256:
against source commit `62e546c`, an all-printed 25-session/60-name synthetic fixture
has bit-identical NAV, cash, positions, borrowing/trading costs, turnover, claims,
cash benchmark and reconciliation, plus identical actual fills. This establishes
unaffected-path agreement, not a historical performance result. The acceptance
receipt binds the four changed accounting/consumer source files.

Stage A is still open: integrate multi-leg/delivery/netted succession, dated signed
cash/share claims, fixed-contract loan reference/rate and dated exchange fees, then
admit recovered causal lending sources. Stage B's remaining family/identity/label/
tensor audit also remains open. Stage C corrected historical replay and Stage D
scaling have not started. The registration was revisited: these are repairs within
A, not a reason to skip A/B or to claim either candidate improved economically.

### Existing-successor netting, 2026-09-19

Both the independent ledger and differentiable account now support delivery of a
single successor into an already held security. Previously they raised an error;
an unheld predecessor could also overwrite the successor's state. Same-direction
positions combine; opposite positions offset without a market fill, fee or turnover.
Only the extinguished short portion releases restricted proceeds. The remaining
position retains its proportional cost basis and oldest surviving entry age.
An existing recipient keeps its own causal mark; no current or future quote is
used to construct the decision. An unheld source does not erase recipient inventory.

Pending exits combine in signed share units after applying the conversion ratio,
then become a fraction of the remaining position. They can reduce that position
but cannot turn into an unintended opening on the other side. The original public
decision intentions remain immutable; outstanding instructions follow succession.
Consumed instructions and entries into the cancelled predecessor carry an explicit
corporate-action cancellation reason. Signed cash entitlements remain separate
receivables/payables through their payment date, including after full share netting.
Evaluation V20 declares this accounting convention; old sealed reports are unchanged.

Targeted account, ledger, policy, action-causality, evaluation and readout checks
pass. The new cases cover both signs, partial/full offsets, both possible surviving
positions, partial exits/fills, missing successor prints, an unheld predecessor,
delayed signed cash payment and event-day decision invariance. The differentiable
netting path's gradient matches an independent finite-difference check away from
the net-zero kink. Ruff and diff checks pass.

`docs/v2_successor_netting_acceptance.json` and the identical receipt in the resolved
run root bind the changed code to reference commit `bd3a88d`. Three synthetic
20-session fixtures (no action with a long quote outage; splits/cash claims; an
empty successor) have bit-identical cash, NAV, positions/marks, costs, borrowing,
turnover, gross exposure, claims, benchmark and reconciliation, plus identical
actual fills and original intentions. Receipt SHA-256:
`5e4bef4efa6e9fcf927569e09fa979656b444386fabbf44ef8ae32f03f6327a3`.
This is an unaffected-path engineering check, not a historical financial result.

The registration was revisited. This completes the existing-destination part of A,
not the whole contractual-event stage. Multi-leg claims and the interval between
economic succession and custody delivery still require integration; do not admit
CPLE/BRML/DMMO events by treating delivery as immediate. Fixed-contract lending,
dated fees, causal recovered-source admission, the rest of the deep audit and all
corrected historical comparisons/scaling remain outstanding. No new input store,
historical replay, GPU fit or promotion is implied by this repair.

### Multi-leg claims and delayed delivery, 2026-09-19

Implemented a sparse source-bound distribution contract in both accounting paths.
Each event carries its information-availability/effective sessions, successor share
ratios, separate custody-delivery sessions, cash entitlement/payment session and
evidence reference. Later-known terms cannot be backdated. This is accounting input,
not a new feature coordinate or a mutation of accepted stores.

At the effective session, the predecessor becomes a non-tradable signed basket;
cash becomes a separate receivable/payable. The basket retains NAV, exposure,
restricted proceeds and financing until custody delivery. Its constituents are
marked using causal successor references and subsequent observed quotes. Delivery
can occur leg by leg, including into existing opposite positions, through the
completed netting transition. No conversion trade, fee or turnover is fabricated.
Undelivered terminal claims remain outstanding even when their constituents are
fully priced. Priced-but-unavailable inventory is reported separately from unpriced
inventory. A genuinely cancelled predecessor cannot reopen, including in a later
flat-start replay; that replay does not invent historical positions or cash claims.

`delivery_session` means the first decision session with custody available, not an
automatic mapping of every calendar credit date to the start of that day. For an
already recognized basket, that known delivery is processed before the decision
using prior marks, allowing trading on the available session without an arbitrary
extra-day lag. Immediate delivery of a newly recognized event is a realization
after the decision, like the existing scalar-action contract. The historical source
adapter must establish this timing from the issuer/B3 evidence before admission.

The optimizer freezes undelivered basket exposure, includes it in remaining capacity,
uses constituent betas/sectors, and includes the fixed basket's idiosyncratic
covariance with an already held underlying. The legacy book cannot budget for a
known impossible basket exit. Training utility and replay/readout risk expand
claims to their actual constituent exposures. Compact daily claim records permit
reconstruction and custody auditing; they avoid a dense extra date/name tensor.
Evaluation V21 hashes the distribution contract, exposes its terms, and saves
undelivered notional and claim positions with new books. Flat-window rebasing retains
prior cancellation identities and future payment/delivery dates.

Validation: **199 targeted tests pass**, including account, independent ledger,
policy, action-causality, evaluation/readout, portfolio objective and controller
learning checks. New fixtures cover both signs, mixed cash/shares, separate and
partial delivery, existing positions, terminal/unknown delivery, ongoing financing,
marking, gradients and truncation, and prior-identity handling. Both market-neutral
and sector-constrained controller paths agree with the independent ledger. Mutating
a delivery-day price leaves that day's state and original intentions unchanged.
Ruff and diff checks pass.

`docs/v2_share_distribution_acceptance.json` and the identical, verified receipt on
the resolved D run root bind this implementation to `91ca1c4`. Four synthetic
20-session paths without new distribution events (including outages, scalar
splits/cash, empty successors and existing-successor netting) are bit-identical on
13 cash/NAV/inventory/cost/exposure/claim arrays and on fills/intentions. Receipt
SHA-256: `bfb869ccaef4ff5e447b61f47504af1f6f4ff667c8e48a74456d5a378e84a16d`.

This completes the multi-leg/delivery **mechanics**, not source admission or all of
Stage A. The current sparse contract is for preannounced distributions into listed
successors with causal reference marks. An unpriced/new-listing leg or a nested
action on an outstanding basket requires explicit evidenced valuation/claim terms;
the code stops instead of allocating an invented price or discarding the security.
Verify these boundaries, fractional entitlements and actual custody availability
when admitting CPLE/BRML/DMMO and the remaining exposure-ranked events. No historical
case has been silently attached to an old fit or accepted store by this change.

The registration was revisited. Next are fixed-contract loan reference/rate/accrual,
dated fees, actual corporate/lending source admission and remaining source-to-model
audit work. Borrow accrual on these baskets still uses the existing marked-notional
proxy pending that loan repair; it is not yet a claim of exact B3 contract accrual.
Stages C/D remain unstarted, with no new performance result or GPU fit to report.

## 2026-09-19: dated B3 loan tariff implementation

Implemented `execution/loan_fees.py` and replaced the undated aggregate fee formula
in the independent ledger, Torch account's cost provider, allocation cost estimates
and original-trade attribution. The primary scenario explicitly assumes normal
electronic sourcing after platform launch; direct, OTC and compulsory schedules
are represented separately. The schedule uses the prior fixed annual tariff before
2020-10-26 and separate component floors/caps thereafter, with the 2022-11-14 cap
transition. Rate rounding and component compounding follow the archived circular.
Removed the three obsolete aggregate fee configuration fields and their callers.
The uniform-borrow Torch path now also honors its configured equity rate, matching
the independent comparator. Readouts separate all stock/hedge loan rent from B3 fees.

New archived evidence: B3's BDR loan guide (`872b9e39...`) and the historical
BM&FBOVESPA cost page confirm the former voluntary 25 annual bp/R$10-minimum
tariff. These add to the existing 125/2020, 049/2020 and 081/2022 circulars.
The seven-file source manifest hashes to
`ebccf40949c6ba719cd6b8244aa2ac0c98733ab30bf9c5620aaba6b123e04db7`;
all archived source hashes were verified. No new historical loan observations or
corporate cases were admitted. No accepted datasets or completed fits were changed.

Validation: 191 targeted tests passed for the dated-fee integration, followed by
101 targeted tests after adding the stock/hedge component reporting. These runs
overlap and must not be added as a unique test count. Tests include date boundaries,
component caps, equity/hedge NumPy–Torch agreement, uniform/sourced scenarios,
future-rate mutation, gradients and downstream consumers. Unaffected synthetic
long-only and fee-disabled long/short books are bit-identical to `b1eed6c` on 13
arrays, fills and intentions. The acceptance receipt is
`docs/v2_loan_fee_acceptance.json`, identical to the verified D-root copy, SHA-256
`0474a68fcde867916a63730da2439d8e3620143ad8776a3bd8e3f95a14de32bd`.

**This is an intermediate Stage A repair.** Evaluation V22 explicitly reports
that fixed loan principal/rate, accrued liabilities versus cash payment, return
settlement and the historical R$10 minimum are still pending. It must not be read
as accepted full contractual economics. Newly built policy cost features differ;
preserve the serialized policy-data contract for any accounting-only replay rather
than silently feeding new coordinates to an old learned policy. Details and next
steps are in `docs/v2_LOAN_ACCOUNTING.md`.

The registration was revisited. Next: integrate the contract lifecycle and its
published-average references, then actual source admission and remaining Stage B
audits. Preserve the completed corporate-claim mechanics and resolve event-specific
loan transformations from source terms. Stages C/D remain unstarted; no GPU worker
or historical performance comparison launched in this step.

## 2026-09-19: fixed-contract loan mechanics and account integration

Both accounts now carry original published-average reference principal and agreed
rate, separate accrued rent/B3 liabilities from cash payment, retain loans through
cover settlement, split partial returns without repricing, and provision the
historical R$10 minimum once per original contract. Same-security splits retain
principal; successor transfers preserve original-entry attribution. Source-specific
loan transformations and delayed claims still need admission. The old daily marked
notional/current-rate accrual is removed from the active accounts and original-trade
cost attribution. Explicit daily marginal cost estimates for allocation remain
estimates of a prospective new loan, not the accounting charge on existing contracts.

Evaluation V23 binds reference panels and paired opening-rate inputs, records capital
and reports pending loan liabilities/payments/principal. Borrow-quality readouts use
original-observation provenance and fixed outstanding principal. Training SAM/TBPTT
restarts independently copy loan state and mutable corporate claim containers.
Ledger replay batches actual same-session loan openings/returns without modifying
fills; the gradient account retains its vectorized operations. Both accounts share
the loan subledger; independent closed-form tests, not their mutual agreement alone,
verify contract arithmetic.

Validation: 247 targeted tests passed; after the batching optimization, 156 affected
checks also passed, followed by 65 reporting/attribution/minimum checks (overlapping counts). A 243-name/252-session synthetic replay with
15,528 fills takes 1.44 seconds versus .22 for the former proxy; a 64-session account
forward/backward takes .26 seconds. These are single-run local CPU observations,
not neural-fit estimates. Expense equals payment plus terminal liability to less
than R$1e-7 at R$10m. Two unaffected synthetic paths are bit-identical to `4e93bde`
on 13 accounting arrays and fills/intentions. The receipt and reproducer are archived
on the resolved D root; `docs/v2_loan_contract_acceptance.json` binds the evidence.

No historical opening references, recovered rates or corporate cases are admitted
yet, and no old policy features were regenerated. Missing opening references cause
an explicit stop at actual new borrowing, never a fabricated source price. The
minimum's partial-payment allocation, same-settlement return cutoff, continuous
currency arithmetic and corporate principal allocation remain documented research
assumptions in `docs/v2_LOAN_ACCOUNTING.md`. General spot cash/proceeds settlement,
renewals/maturities/recalls and issuer-specific loan terms still need completion.
The ordinary engine's normalized test capital is not the registered R$10m replay.

The registration was revisited. Continue Stage A with these source/settlement gaps
and actual source admission, then finish the Stage B source-to-tensor audit. Preserve
both completed censuses; inspect only their unresolved cases. Stage C corrected
comparisons and Stage D capacity waves remain unstarted, with no GPU worker active.
This milestone does not constitute historical accounting or program acceptance.

Loan-contract acceptance receipt SHA-256: `58db32e966dfe4108a1c6f5101eefe2158a5ba37cd9402ee5f74c4b29fc8a2cf`. The D-root recovery copy is byte-identical.

## 2026-09-19: dated spot cash and settled-proceeds financing

Both accounts now preserve trade-date beneficial holdings and NAV while separating
actual settled cash from T+3/T+2 money obligations. Unsettled purchases do not create
an actual debit yet; unreceived short-sale proceeds do not earn CDI yet. Short cash
remains segregated through cover settlement. Same-value-date money nets, and ending
the evaluation leaves outstanding claims rather than inventing a final cash transfer.
Cash-only event proceeds remain restricted until the specified payment date. Known
settlement dates inform decisions without future prices; SAM/TBPTT retain independent
pending-settlement state. Evaluation V24 reports unsettled net cash and separates
free-cash income, short-proceeds income and actual debit financing. Static forecast
and policy feature coordinates, original stores and fit roots are unchanged.

Validation batches passed 148 account/policy/claim checks, 75 evaluation/readout/loan
checks, 43 expanded funding/account checks, then 151 final affected checks. Counts
overlap. Eleven dedicated value-date tests include closed-form money/income oracles,
an early cover before the short sale settles, debit financing, same-date netting,
cash-only payments, transition-date netting, gradient finite differences and independent
SAM copies. Two zero-CDI synthetic paths retain bit-identical NAV, fills, intentions,
prices, positions, costs and exposure versus `0dc7c7b`; settled plus unsettled cash
matches old total cash to less than R$2e-9 at R$10m. A 243-name/252-session replay
takes 1.99 seconds versus 2.15 for the fixed-loan trade-cash baseline in the same
measurement; 64-session account forward/backward takes .55 seconds. These are
single local CPU observations, not a speedup claim or neural-fit ETA.

The archived 2021 B3 clearing manual adds specific custody, netting, loan modality
and renewal evidence. No recovered loan observation or historical corporate case
has yet been admitted. The current queue tracks money, not a full physical custody
inventory: corporate offsets against a not-yet-delivered purchase must be resolved
before admission. Historical clearing-calendar boundaries, D+0/D+1 loan conventions,
maturity/renewal/recall, issuer terms and actual source panels remain open. The
supplied trading-session axis and prior-close funding are explicit assumptions,
described in `v2_LOAN_ACCOUNTING.md`; no unsupported broker tariff was added.

Receipt `docs/v2_spot_settlement_acceptance.json`, SHA-256
`be8a7a157909c5fc8b0ced719efc0e5432bad7e466ad4bb65417bee02c3ec064`,
is byte-identical to its D-root copy. The expanded nine-source manifest hash is
`1c6a46252b66c726c0c1a503ff8fb729897673990f710bf2cb86700a17770397`.
The registration was reread: this is still intermediate Stage A, not acceptance of
historical economics. Continue loan/custody/event source admission, then the remaining
deep Stage B audit and corrected Stage C/D experiments. No GPU/census worker started.

Incidental diagnostic found for follow-up: the legacy `StatefulLedgerResult.summary()`
gross-shortfall decomposition does not reconcile on an arbitrary target-policy book.
Current learned-policy books use their dedicated `book_summary` instead; do not feed
such books through the legacy fixed-slot decomposition or remove its invariant check.

## 2026-09-19: hedge published-average reference recovery

`ops/recover_bova_loan_references.py` recovers all 3,963 BOVA11 published-average
quotes from the already bound 2009–2024 COTAHIST archives. Exact ISIN, cash-market
type, instrument specification and dated BDI categories are retained. Every date
and close exactly reproduces the accepted hedge series; published quote factors
are used rather than replacing averages with turnover/quantity or current marks.
No held-out archive was opened and no raw/accepted source was changed. The source
manifest and new parquet are bound by the run pointer; data SHA-256 is
`98a6ae965e93eebe52e142ff4ea91cd047d4f7f172773c182ec56096e2c72c19`.
This fills the missing hedge quote recovery, not causal panel admission: consumers
must receive the appropriate prior publication and action-adjusted contract terms.

Additional manual evidence for source admission: 2021 B3 manual pp.133–136 separates
cash events from changes in the borrowed asset. Multi-asset contract principal uses
the issuer's allocation factor; it is not generically justified by relative current
constituent prices. Redemption can also settle loan remuneration separately from
issuer cash payment. Preserve the original mechanics evidence, but replace these
declared intermediate assumptions with source-bound event terms before accepting
the affected historical cases. Do not retrospectively change earlier decisions.

## 2026-09-19: recovered lending admission and causal loan inputs

The run pointer now binds `recovered_lending`, a new economic archive built from the
completed source censuses without parsing the PDFs again. Verified printed rows
yield 202,698 rate observations versus 41,353 previously, and 230,454 balances versus
81,324. Positive-flow quantity-weighted taker and donor rates remain separate; a
zero-flow carried rate never refreshes an observation. Historical balances use the
printed ISIN or same-date COTAHIST identity and become available after the report,
not after a potentially older position date. Unknown identities remain explicit.
All previously admitted balances are preserved. Reconciled rates repair their own
dated keys; source-specific rates are not clipped or replaced by a universal floor.

The real loader was exercised on all 3,717 dates and 933 security axes, including
568,815 eligible stock-days. Eligible imputed rates fall from **21,144 to 2**;
29,994 eligible aligned rates change (14,637 lower, 15,357 higher). Balance-policy
borrowing availability gains **21,133** eligible stock-days and loses none. These
aligned/aged-panel counts differ from the previously reported counts of new printed
observations. The **498,823 eligible placeholder cells remain unchanged**: these
source balances do not manufacture historical borrowing prices before July 2023.
Both cost-only changes and changed borrowing availability must be attributed
separately in the corrected replay. No P&L improvement is inferred from coverage.

`loan_source_panels` binds the last same-ISIN published average strictly before each
session, retaining the source date. This follows the B3 contract's previous/last
available quote rule; it is not a forward-filled model price or label endpoint.
No current quote, successor mark or guessed adjustment enters that reference.
All eligible stock-days after the first store date have a reference; the only 125
missing cells are on 2010-01-04, before the model's history warm-up. 148 eligible
references are over seven calendar days old and retain that fact; the contract
permits the last available average. Every hedge session has its prior reference.
The separate BOVA11 rate series has 369 decision-available observed sessions,
median annual rate .7821%, and 3,348 explicitly fallback sessions. Computing this
series separately does not alter the stock cross-sectional imputation universe.

The 94 failed legacy balance attempts remain individually auditable. Direct
inspection of 2020-10-27 finds multiple printed AGRO3 rows with different quantities
in the same open-balance section, plus an unidentified row. It is not safe to call
these duplicate extraction artifacts, pick one or sum them without interpretation.
Other failures include spacing defects and residual printed totals; their old
receipts and originals are preserved for targeted resolution. None is a blanket
permanent security exclusion. These pre-2023 balance uncertainties cannot establish
missing rates; the unchanged placeholder convention already preserves eligibility.

Validation: ten source-admission/loader tests passed; six recovery tests then passed
after adding prior-reference alignment (overlapping counts). The new cases verify
zero versus missing rates, report-date timing, future mutations, reference dates,
security-axis isolation, conflicts and preservation of unaffected rows. Ruff passed.
Full-population assembly took approximately three seconds, including interpreter
startup and source verification. Neural inputs, accepted stores and original fits
were not changed, and no historical outcome or GPU fit was read/run.

Source admission is not full accounting acceptance. Event-specific reference and
principal allocation, fractional entitlements, custody offsets, maturity/renewal,
loan modality and remaining dated costs still need completion. The source panel
deliberately does not infer an ex-date adjustment absent an explicit source term.
The registration was revisited: continue A and the remaining deep source-to-tensor
B audit, then corrected comparisons C and bounded capacity D. The receipt is
`docs/v2_lending_source_admission.json`; its artifact bindings and recovery archive
make this work resumable without repeating either census.

Implementation commit: `38bc5ad`. The verified 24-file recovery archive is bound by
`lending_source_recovery.json` through the run pointer; ZIP SHA-256 is
`4a26d6ff5472cceda6d4edd942e1371eccf2cbb404603abae9131aca660b0917`.
The admission receipt SHA-256 is
`39031ab45e5984d989f13bd3e392d7c33fbab195934bdde6fa11542138f13640`.
All archive members were read back and hash-verified; reference-panel axes and the
development date boundary were checked from the archive itself. No immutable raw
or accepted dataset was duplicated.

## 2026-09-19: compulsory loan cash events and issuer-source recovery

Cielo's B3 circular exposes a separate contractual path: remaining loans cash-settle
on D+4 after last trading, whereas remaining shareholders receive the later approved
redemption. Both accounts now implement this distinction, preserving original loan
rent/fee references and liabilities through the loan payment date. Later physical
returns are superseded without discarding their covering assets or changing external
spot cash value dates. Principal cash is attributed separately from borrowing cost.
The final payment price never rewrites prior intentions; a new-short prohibition
survives flat-start slicing. Evaluation V25 binds the event terms. Static policy
coordinates, all source stores, model labels/features and checkpoints are unchanged.

Seven new primary-source receipts are archived: the Cielo and BR Malls B3 loan
circulars; Cielo's auction result, shareholder procedure, notice and embedded offer
document; and a bounded 30-observation BCB SELIC retrieval for August–September 2024.
Four existing issuer/B3 receipts are hash-verified and referenced without copying.
The Cielo offer explicitly distinguishes pre-auction CDI from subsequent SELIC.
Source-derived continuous loan consideration on August 30 is R$5.842895570784521;
this is not an obtained invoice quote. The September shareholder cash term is R$5.89,
payable September 26, and must not be backdated into August. BR Malls' newly recovered
terms distinguish January 9 economic succession, January 10 end-of-day loan
conversion, January 11 custody and January 20 cash, preserving loan principal/rates.
Details and remaining event-specific admission are in `docs/v2_CORPORATE_EVENTS.md`.

Validation: 227 affected account/policy/evaluation/objective checks passed, then
seven focused loan-cash checks passed after adding the real allocator/input-hash
test (six overlap). The new oracles cover signed shareholder/loan asymmetry, cash
conservation for superseded covers, future-price mutation, gradients, source clocks,
borrow constraints and slicing without static-feature drift. Ruff passed. Two
synthetic long-only and long/short books are bit-identical to commit 601b75a across
NAV, holdings, marks, fills, intentions, cash/financing, turnover, liabilities,
payments and original-entry charges. `docs/v2_loan_cash_acceptance.json` binds the
receipt and reproducer. No redundant census, GPU fit or historical outcome was run.

This is still intermediate A/B. Actual historical event-axis admission is not yet
complete: the accepted CIEL store retains q=1/d=0 and unresolved coverage after
August 26. Source-derived overrides must be applied explicitly in a new accounting
contract, with any model-data effects separately handled in B. Remaining custody,
multi-leg principal/fractions, maturity/renewal, calendar and source-family audits
remain required before C/D. The superseded-return fixture proves cash/NAV arithmetic,
not full failed-delivery/holding-basis metadata. Invoice cents remain a bounded
research convention. The registration was reviewed; no stage or candidate is promoted.

Implementation commit: `a401665`. The 23-file D recovery archive is verified and
bound through `loan_cash_recovery.json`; ZIP SHA-256 is
`d3e73b3e4d39dce3e270f167c05bc5dbd2437f02d08eaf0feb069b3290beffc3`.
Acceptance SHA-256 is
`8b4dfe34f095db94a37431b30edf38296efcea914ecd96e85448a55c9bc8dad9`.
All members were read back, and all eleven original source receipts were verified.
The run pointer and its hash sidecar were refreshed together; no immutable dataset
was copied into the archive. Main implementation was pushed to GitHub.

## 2026-09-19: Cielo historical admission and rename-loader repair

The new explicit `corporate_replay` pointer binds Cielo to the original full
3717-session/933-name axes. Loan cash closes August 30 at the independently sourced
SELIC-adjusted R$5.842895570784521. Remaining shareholders retain inventory until
September 24 recognition of the September 23 announcement, with R$5.89 paid
September 26. Coverage is resolved from the known August 27 closed-register state.
Changes: 86 coverage cells, including six eligible days, and one q/cash/action/
payment cell each. No quote, eligibility, score, feature or target changes. A
full-axis hash check confirms the eight input source arrays are unchanged.

Predetermined 4% long and short positions opened at the actual August 26 quote,
with the other 932 names flat and zero costs/CDI, exactly reconcile both accounts
against closed-form event cash flows. Source/reference/calendar hashes are bound
in `v2_corporate_replay_acceptance.json`. This is a mechanical historical oracle,
not a candidate outcome. The one-cent loan invoice convention remains bounded
per actual share count. Flat-start slicing retains prior borrowing prohibition and
future payment dates; static serialized policy features retain their original
objects. No old policy was reconstructed with changed static cost features.

The identity admission review found a distinct loader defect: verified ticker-plus-
ISIN renames could not pass the same-ticker heuristic candidate gate. Explicit
evidenced pairs now bind directly to exact original quote boundaries, with
non-overlap, successor ticker and one-to-one checks. Same-ticker proposals still
do not auto-admit anything. The newly recovered issuer November 7 ISA notice is
archived, hash-bound and visually checked; its source typo TRLP4 is explicitly
resolved by the November 18 issuer notice and COTAHIST TRPL4 observations.

Three unit-ratio/no-cash rename links are now bound for the next derived store:
ALSO3→ALOS3, TRPL4→ISAE4 and TRPL3→ISAE3. In a bounded six-name replay of the
existing causal history/universe functions, ALOS3 recovers 60 eligible days and
ISAE4 recovers 28. ISAE3 gains no eligible days under unchanged liquidity rules.
Twelve stale predecessor-active cells retire, no existing eligible successor days
are lost, and successor eligibility never predates the identity boundary. This
does not yet update actual model tensors, shareholder wealth, labels or sidecars.
`v2_identity_source_admission.json` records the exact source rows and hashes.
Original accepted inputs and foundation fits are untouched.

Validation: 23 targeted corporate-admission, data-foundation and causal-continuation
checks passed, plus the full-store conversion test (one passed, 49 deselected).
Ruff passed for all changed Python files. Earlier mechanics/censuses were not rerun.
The registration remains unchanged and was revisited: A/B are incomplete; C/D have
not started. Next resolve corporate custody/loan lifecycle and BRML/DMMO/CPLE,
then propagate these identity links and finish the deep source-to-tensor audit.
The BDI's predecessor loan aliases can outlast a spot rename; do not silently map
those onto new neural features or infer a loan renewal date from the quote change.

Implementation commit: `3d9eff4`. The verified D recovery archive has 25 members;
all members, restored accounting/identity manifests and the allowlist bytes were
checked, along with eleven original source receipts. No immutable dataset was
duplicated. `source_admission_recovery.json` is bound through the run pointer;
archive SHA-256 is
`1256e3f8210994131984a805c0ef072f2f2783fff9a01fea0a76c370fc0145ed`.
The implementation is pushed to GitHub. This closes the Cielo admission and
rename-loader milestone only; the broader A-D program remains active.

## 2026-09-19: custody timing, BR Malls admission and faster loan attribution

Both accounts now track scheduled owned purchases separately from beneficial
holdings and cash value dates. An offset against an undelivered long no longer
returns stock or releases proceeds early. Settled shares are used first, then the
earliest receipt. Original principal/rate accrue through the actual return date;
external cash obligations keep their original value dates. SAM/TBPTT clone or retain
the independent custody queue. Outstanding return cohorts also follow contractual
identity even when the beneficial source holding is flat.

The explicit corporate manifest now admits BRML on source axis 169/BRBRMLACNOR9 and
successor axis 34/BRALSOACNOR5. It recognizes Jan9 economic succession, Jan10-close
loan conversion represented at Jan11 opening, Jan11 available custody and Jan20
cash of R$1.62899410177968 per prior share. The share ratio is .398551577675763.
All original loan principal/rate stays with the single stock successor. The issuer
already specified final cash including its projected CDI adjustment; no extra
correction is fabricated. Source-array effects are 494 additional BRML coverage
cells, six eligible; combined with Cielo this is 580 coverage cells and 12 eligible.
Old quotes, membership, predictions, targets and static policy features are unchanged.

The January25 issuer fraction-auction result was retrieved, archived, hash-bound
and visually verified. Long shareholders receive whole successor shares, retaining
their fraction as a non-tradable marked entitlement. The announced R$17.694416 per
fractional share is recognized Jan26 and paid at the issuer's Feb2 deadline.
Fractional loan quantities remain intact, per B3. Future auction amounts do not
enter earlier NAV or intentions. The fraction participates in constituent risk
until it becomes a known cash receivable. This uses the existing claim machinery.

Actual-calendar long/short 4% position oracles at R$1m/R$5m/R$10m agree exactly
between accounts and independent cash-flow formulas. A separate R$10m case buys
R$200k of ALSO Jan10 against the BRML short: the netted loan part returns Jan12,
the remaining Jan11 cover returns Jan13, and original-reference rent is respectively
R$140.6798916651383 and R$123.84628958666255. This is a prescribed mechanics oracle,
not a new model return. `v2_corporate_replay_acceptance.json` binds the source axes,
unchanged array hashes and all results. Its SHA is
`93b5c1f0d2e8764e33c8337d8a04c3dd294254529b39cc939785e99d492edf9a`.
The new corporate manifest SHA is
`f9e42a83a091b06ef54aa6ca99f2dc55d5583e466c0efd793a26bd7c29fdf2a6`.

Validation: 241 affected tests passed, followed by 25 overlapping loan/custody/
readout tests after the reporting optimization. Ruff and diff checks passed. The
runtime reproducer preserves exact ordinary long-only and long/short NAV, holdings,
cash, loans, costs, fills, intentions and every nonzero original-entry charge against
56adc09. Extracting attribution arrays once instead of indexing Torch scalars for
each row reduced a 243-name/252-session stress replay from 16.90 to 6.95 seconds.
All 2,640,861 nonzero charge rows and 61,230 fills remain in that stress case;
identically zero rows alone may be omitted. The 64-session account forward/backward
took .23 seconds versus .30 in a single observation and preserved NAV/gradient
exactly. These are CPU account measurements, not neural-fit estimates.
`v2_share_custody_acceptance.json` binds tests, code hashes and runtime receipts.

The registration was revisited. A/B remain incomplete and C/D have not started.
BRML's reference waits for custody before disposal. Its issuer also allows trading
of the new issue from Jan9; resolve or bound a prearranged Jan9-Jan10 sale settling
after credit before final accounting/candidate acceptance. Earlier fraction cash
receipt Jan26-Feb2, integer order sizing and unusual gross custody registers remain
explicit bounded execution assumptions. Next continue DMMO/CPLE and other exposed
events, finite maturity/renewal/recall and clearing-calendar terms, then complete
the deep source-through-tensor audit and propagate accepted identity fixes. Do not
repeat completed censuses or this passed mechanics work. No GPU fit or model
profitability result has been produced in this milestone.

Implementation: `fb1e0e5`. The 32-member D recovery archive was restored and every
member checked, including the restored event manifest and acceptance receipt.
Ten bound original source receipts were verified; immutable datasets were not
duplicated. `brmalls_custody_recovery.json` is bound by the run pointer. ZIP SHA-256:
`b6393f8f6d4747a84cfb7eb20e7553a8e41fc8b66899fb211530af2142ce4e3c`.
Acceptance SHA-256:
`e8ef32f90f97ed6f55d716c7bc706bed922d330106801d03c284c0315de498ff`.


## Copel accounting admission and recovered Dommo fraction notices (September 19)

The run pointer now resolves `corporate_replay/copel_manifest.json` (schema V3,
SHA `6af736a991fbf14a9a986de86c5cd5996b785fcd8332ee1874da563f98c6adc3`).
Cielo/BRML terms remain, with one canonical leg representation shared by single
and multi-leg distributions. Copel delivers one ON plus four PN per unit on
December 28 after December 26 economic succession. The coverage amendment adds
254 cells/six eligible cells; total amendments now cover 834 cells/18 eligible.
Accepted quotes, eligibility, model features/targets and serialized static policy
coordinates are unchanged. No new model store or neural fit has been created.

Evaluation V27 binds explicit loan principal fractions, which must sum to one.
Both accounts allocate original loan principal/rent independently of market-value
inventory basis/proceeds. The unquoted Copel issuer K is openly bounded using a
20%/80% reference and 0%/100% ON endpoint sensitivities; it is not represented as
an obtained contractual term. Original total principal/rate and partial-return
liabilities remain intact, including zero-principal endpoint legs. This replaces
the earlier marked-value loan allocation, without a legacy parallel path.

`v2_copel_replay_acceptance.json` binds 36 prescribed full-axis historical oracles
at R$1m/R$5m/R$10m. Both accounts agree exactly and match closed-form cash/rent.
When both legs return January 3, K has no effect. With ON returned January 4,
its entire allocation-range rent span is R$61.55 at R$10m (0.06155 bp NAV over
the prescribed path). These are mechanics/sensitivity figures, not daily alpha.
The oracle took 1.62 seconds. Four new allocation tests cover endpoint conservation,
staggered delivery/returns, rejection of unstated multi-leg allocations and analytic
gradients. The affected account/claim/policy/objective/evaluation batch passed 145
tests in 42.92 seconds. Initial new fixtures needed a missing annual-rate argument
and float64 scalar conversion; no production financial mismatch was concealed.
Ruff and diff checks passed. No old stress benchmark or completed census was repeated.

Recovered Dommo's March 17/30 original fraction notices through the public CVM
2023 IPE index; both were visually checked and preserved on D. The March 30 result
changes the tentative payment deadline to April 6, with March 31 recognition of
the later-known price. `v2_dommo_source_terms.json` binds these and the earlier
loan/payment sources. It also records why PNB requires December quantity extinction,
December28 rent payment and a separately valued January13 liability, rather than
reusing Cielo immediate cash closeout. Default PNA loan fractions need their own
provisioned-cash treatment, unlike BRML's special fractional loans. No Dommo event
was silently applied. Its CVM index now displays PRIO FORTE: use CNPJ/CVM plus dated
original issuer text, never that current display name as historical identity.

The registration was reread. A/B remain incomplete and C/D unstarted. Next prioritize
Dommo's distinct loan liability/fraction terms, finite loan lifecycle and
presettlement execution bounds; use independent CPU work for the full deep audit
and actual propagation of the accepted identity links. Nine of ten current auxiliary
family parquets carry a decision date and field ages but not original timestamps;
trace their bound upstream sources rather than treating these final parquet dates
as proof of public availability. This schema inspection is not completion of B.
Do not repeat the completed quote/lending censuses or passed mechanics checks.

Implementation `293edbd` is preserved in a 30-member verified D recovery ZIP.
All members were restored and checked, including the corporate manifest and source
code hashes; 17 original source receipts were verified without duplicating
immutable inputs. `copel_allocation_recovery.json` is bound by the run pointer.
ZIP SHA-256: `d3d0a8d1a61ccb3c1ca32523b408e00232603ef636452e674a99acce49b9639d`.
Loan-allocation acceptance SHA-256: `3245e35dafc8ed0e3d24b7f3367690ef47cc129d7c5d1e9b2de7b0755b72491d`.


### 2026-09-19: Dommo lender-cash endpoint and cash-calendar source defect

Implemented deferred loan redemption in both accounts: physical quantity ends at
the source event, rent/fees stop then and pay on their own date, cash liability
keeps dated causal marks until redemption, and proceeds stay restricted until
payment. Existing all-contract cash closeouts preserve their behavior. Active-only
lender election is after actual fills, excludes whole original contracts with
pending returns and new D+1 registrations, and does not prohibit non-elected later
loans. Payment releases proceeds and debits the known liability together before
decisions. Independent SAM/TBPTT copies include claim quantities/marks. V28 hashes
all terms and reports cash principal liabilities separately from borrowing expense.

Actually admitted the source-bound Dommo PNB endpoint to a separate scenario
manifest; the primary Cielo/BRML/Copel pointer is unchanged. Recovered Oct24 issuer
approval and 58 daily BCB CDI observations; the compounded final value matches
R$1.90432468607 to 6e-13. Dec26's R$1.891796083009226 uses elapsed CDI only; Jan9
first recognizes Jan6's fixed final amount. Six actual-calendar, full-933-name
oracles at R$1m/R$5m/R$10m with zero or sourced 100% CDI match both accounts and
independent closed forms exactly (0.56 seconds). R$10m rent is R$61.5941 paid Dec28,
redemption R$407342.1778 paid Jan13. No predictions/labels were read for these
prescribed positions. This does not admit default PNA or claim model performance.

Independently advanced Stage B by tracing cash sources through the actual consumer.
The bound source preserves SGS12 dates/units, but same-date selection drops money-only
dates and shifts changes in rates relative to close-to-close accrual. The new
explicit cash-calendar panel changes 67 of 2,098 supported sessions and recovers
23 omitted monetary dates; benchmark mean changes +0.030318289 bp/equity-session.
All 58 independently retrieved overlapping observations match the bound vintage.
No old input/store/cache was overwritten; `apply_cash_calendar` shallowly replaces
only accounting CDI/provenance after frozen policy loading. Loan/spot clearing-day
clocks and all remaining auxiliary families still require audit. No model gain is
inferred from the higher benchmark or funding income.

Validation: 183 affected tests passed in 8.37s before the final payment-order
refinement; afterward 98 tests passed in 42.48s with one new cash-fixture failure
(the fixture used global indices 100+ against a sliced 25-row calendar). Correcting
that fixture passed both cash tests (1.81s). Ten loan-cash checks also passed after
the ordering refinement. These batches overlap. Ruff passed. New tests protect
whole-contract eligibility, rent/payment clocks, retained proceeds, future-value
and payment-date causality, analytic/finite-difference gradients, independent claim
state and static-policy reuse. The small unaffected comparison initially exposed
only a 1e-10-BRL summation-order difference in outstanding-principal metadata;
restoring the original summation order yields bit-identical ordinary long and mixed
books in both accounts, all original arrays, fills, intentions and loan charges.
No completed census or old stress benchmark was rerun.

Registration reread. A/B remain incomplete; C/D have not started. Next: default
Dommo PNA provisioned loan fractions, finite maturity/renewal/recall and the full
loan/spot clearing-calendar clocks, presettlement disposal bounds, and every-family
source-to-tensor audit plus full identity propagation. Then apply accounting, new
cash calendar and source availability separately in corrected model replays. Keep
old static coordinates; refit only under an explicit changed data contract.

Implementation `4640921` is archived in `loan_election_cash_calendar_4640921.zip` (39 restored/hash-verified members; 20 original source checks), with no immutable dataset duplication. Restored source acceptance hashes and cash-panel arrays match. Archive SHA-256: `1f26d000daf75526d684d1b652d2ede10f0849768d7a08350ba0bf0254cb14f6`. Recovery is bound by `loan_election_cash_calendar_recovery`; prior archives remain valid for their source commits.


### Default Dommo conversion and all-family tensor boundary (2026-09-19)

Default PNA is now admitted on original 3717-session/933-name axes: .0375 PRIO
and R$.4625 per DMMO, Jan9 effect, Jan11 credit, Jan17 cash. The primary manifest
is `dommo_manifest.json`; the lender PNB endpoint now also binds the default for
remaining non-elected contracts. Source receipts are unchanged. Per-original-loan
truncation preserves original principal/rate and provisions the signed fraction;
whole loan shares alone can be traded/returned. Mark the fraction until Mar31
recognition of the Mar30 result, then pay the printed approximate R$31.94031 by
Apr6. Proportional proceeds remain remunerated until payment. Sub-one-share loans
have an explicit stopped-rent reference and continued-rent-to-payment endpoint,
not a recovered invoice. Last Jan6 source trades settle Jan10 under registered
T+2 before Jan11 credit; gross custody/return variants require their own terms.

Six prescribed actual-calendar long/short oracles at R$1m/R$5m/R$10m reconcile
both accounts and closed forms exactly. R$10m original reference is R$1.78,
principal R$404545.4567375262, whole PRIO quantity 8522, provisioned fraction
.7273189085572085 and 4% annual rent R$314.93524816795866 through Jan13 return.
Zero execution/B3/CDI isolates these mechanics; no model alpha is inferred. Dommo
adds 494 resolved coverage cells/six eligible; all four cases total 1328/24.
No quote, eligibility, label, feature or static learned-policy coordinate changes.

Independently implemented row/transform oracles reconcile all ten auxiliary
families, 113 fields and 41942683 valid active observations, with zero value/mask/
age mismatches. Nine decision-dated producers took 16.51 seconds. The separate
odd-lot oracle took .61 seconds, verifies exact D+1 for all 1253007 archived rows,
and retains 2009 lag-five warmup (125 valid names per field at first store date).
This is the producer-to-sealed-tensor boundary; upstream original publication,
revision, denominator, conditioning and final neural-tensor audits still remain.
Completed quote/lending censuses and old stress benchmarks were not repeated.
See `v2_AUXILIARY_TENSOR_AUDIT.md` and pointer-bound receipts.

Validation: 132 affected checks passed in 59.99s, then 73 overlapping checks in
3.51s after a payment-order refinement, including seven new fractional-loan tests.
Ruff passes. These new tests found and repaired separate payable classification,
a differentiable tensor-view issue and known-claim payment/proceeds ordering.
Previously known corporate cash settles before the decision; new same-day terms
still realize after the intention and funding uses prior-close balances. Two
ordinary long/mixed books remain bit-identical against d7c6a2f in every original
array, fills, intentions and nonzero loan charge. Acceptance and reproduction
identities are in `v2_loan_fraction_acceptance.json`.

A/B remain incomplete and C/D unstarted. Next priorities: finite loan maturity,
renewal/recall and actual clearing/accrual calendars; presettlement disposal and
remaining exposed events; upstream source-clock/unit audit and complete rename
propagation through wealth/features/labels/actual tensors. Then corrected matched
replay and the registered conditional scaling waves. No GPU job launched.
