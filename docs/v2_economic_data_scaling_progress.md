# Economic accounting, data audit and scaling progress

2026-09-19: user authorized accounting repairs, a deep source-to-model audit,
additional historical B3 retrieval and economic-primary attention/GRU capacity
follow-ups. Read `research/preregistrations/v2_economic_data_scaling.md` after each
stage. Foundation outcomes remain sealed at the canonical foundation pointer.

**Current status:** The complete Stage B derived store is accepted. Stage A and
final economic admission remain incomplete; C/D have not started. The latest
denied-renewal/recall engineering checkpoint is documented at the end of this file
and in `v2_LOAN_RETURN_NOTICES.md`. Corporate-account
research and the source censuses below are complete. BOVA11 borrowing-path
consistency, removal of fabricated missing-quote liquidations, inventory netting and
multi-leg/delayed-delivery, fixed-contract loans and dated money settlement are
implemented and tested. Reconciled lending observations are now admitted to a
separate economic archive; prior stock/hedge references are bound. Event-specific
terms and full historical account acceptance remain open. Cielo's source-based
accounting artifact and ALLOS/ISA rename admissions are now verified as described
at the end of this log; a corrected full model store remains pending. No training or
source-census worker is active. The latest FCA candidate now includes source-bound
MGLU/ABC corrections and five dependent families; see the final entry and
`v2_FCA_IDENTITY_PROPAGATION.md`. It is not yet a replacement model store.

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

Recovery: implementation `6576955`; `dommo_fraction_auxiliary_6576955.zip` SHA-256
`b6155b590e7cccdea587530f4d43bbbbc8397cf5ff78f627ba0cf6f220ce45b5`. All 278 members restored and hash-checked,
including the complete current research source package and acceptance code hashes;
18 primary-source references verified. Immutable stores, family parquets,
raw datasets and old fits were not copied. The run pointer binds the recovery receipt.


### Finite loans and saved conditioning (2026-09-19)

The registered 63-session term now renews four B3 sessions before maturity under
explicit assumed approval. Only remaining unreturned quantity renews. Pay old
accrued charges once, reset reference/rate from the current causal source panel,
retain original investment entry attribution and keep pending returns on old terms.
No market fill or proceeds release is invented. Renewed contracts receive their
own historical minimum. The 30/126-session sensitivities, no ordinary unscheduled
recall and prearranged approvals are negotiated-account hypotheses, not observed
contract history. V30 reports renewal records and hashes the lifecycle convention.
Dated D0/D1 rent/fee endpoints and denied-renewal/recall execution remain open.

Ten new archived B3 calendar receipts explain 20 of 23 money-only dates as equity
clearing/custody closures; nine dated holiday controls match the store. Three dates
(2016-12-30, 2017-01-25, 2017-11-20) still need older equity-specific evidence. Do
not use the older circular's different clearing-house name as proof of the equity
house's treatment. No settlement or model arrays were changed. Full-store 126-session
terms span at most 194 calendar days, below the two-year contract maximum. This
calendar evidence does not establish every rent-accrual day-count rule.

All 120 foundation manifests and selected-checkpoint preprocessing payloads match.
Independent reconstruction of 15 coordinate systems from 34,852,386 unique scalar
fit observations finds zero median/scale/support/inheritance mismatches. The stricter
actual CPU input check verifies 1,856,628 packed cells over 45 full-population,
60-session samples, including field routing, masks, ages and permanent name indices.
No neural forward or forecast scoring occurred. Source reconstruction took 25.38s;
checkpoint/tensor verification 24.85s. See v2_FIT_CONDITIONING_AUDIT.md. This closes
conditioning and the tested input-routing boundary; upstream original publication,
revisions, financial denominators, complete identity propagation and wealth/labels
remain. No completed source census was repeated.

Validation: six new renewal checks, followed by 111 affected checks (overlapping),
all pass; Ruff passes. Two 30-session ordinary long/mixed books remain bit-identical
to 464b8c4 in original arrays, account records, fills, intentions and charges. Final
NAVs are R$10,116,715.718244806 and R$10,131,576.35573736 at R$10m start, respectively;
these are synthetic unaffected-path oracles, not model alpha. No stress benchmark
was repeated. Acceptance is bound by v2_loan_renewal_acceptance.json.

Registration reread. A/B remain incomplete, C/D unstarted and no GPU job launched.
Next: dated modality accrual/clearing and presettlement-disposal/recall bounds,
remaining exposed cases, upstream units/vintages and full identity/wealth/label
propagation. Corrected replay must keep frozen static coordinates and separate
accounting, source rates/availability and changed-store/refit effects.

Recovery: implementation `92c338a`; `loan_renewal_conditioning_92c338a.zip` SHA-256
`fb2fdd7e5fccef9bac56adb9ac47bd8afb1278ee6435f88f3f123b10ac52e7d2`. All 264 members restored and hash-checked,
including complete current research source, acceptance code hashes and audit
artifacts. 19 original-source receipts verified. Immutable inputs and old fits
were not duplicated. The run pointer binds the recovery receipt.

## Dated loan accrual and upstream US source timing — 2026-09-19

V31 implements the archived 2021 B3 manual's p112 distinction in both accounts.
Registered/D0 rent includes registration and excludes physical return; B3 fees
include both. Electronic D1 starts after registration and includes return. A
same-registration-day request settling next session incurs one day of rates.
Explicit corporate stop dates retain their separate contractual treatment. Charges
now accrue after actual fills/events/renewals and before payment; earlier intentions
remain unchanged. Partial cohorts, renewal minima, liabilities and original-entry
attribution follow the same interval rules. The pre-2020-10-26 registered-D0 primary
and subsequent electronic-D1 primary are explicit modality hypotheses; blended BDI
rates cannot identify actual contract type. Electronic D0 remains a sensitivity.

Eight new closed-form/day-count/partial/renewal/tariff/corporate-stop/gradient cases
pass, included in a 119-check affected batch (50.76s). Two subsequent evaluation
checks, 22 distribution/custody checks and three corrected historical-fixture checks
also pass; counts overlap. Ruff and diff checks pass. The two ordinary D1 books
preserve exact NAV, cash, positions, fills and intentions versus 197d774. Moving
accrual after cohort splitting produces only measured floating-point loan rounding:
at most R$2.28e-13 in account loan amounts and R$1.78e-15 per original-entry charge.
Do not claim every loan metadata array is bit-identical. See the bound
v2_loan_accrual_acceptance.json and loan_accrual_unchanged evidence. No stress
benchmark or previously passed census was repeated.

The independent US-source audit verifies 19 bounded original files, 71,135 bars
and 69,029 returns with zero mismatches (1.43s). A separate first-available-decision
reconstruction checks 1,136,631 active EWZ input cells (5.85s), with exact values,
masks and ages. Fourteen early-close dates correctly enter the same B3 decision;
ordinary US closes enter later. Untimed pre-NYSE SUZ history remains preserved,
without invented availability. Only historical 2010-2024 payloads are decoded;
present-day quote metadata is not consumed. Single-vintage historical revisions
remain unknown. See v2_US_SOURCE_AUDIT.md and the two bound audit receipts.

Registration reread; A/B remain incomplete and C/D unstarted. No GPU fit or model
profitability result was produced. Remaining priorities are denied-renewal/recall
and pre-custody-disposal bounds, three older clearing ambiguities, upstream clocks/
units/financial denominators across the remaining families, and full identity,
universe, warmup, wealth and label propagation. Completed conditioning and auxiliary
tensor checks need not be repeated. Accounting-only replay must preserve frozen
static policy coordinates; data repairs require a separately bound store/refit.

Recovery: implementation `7982bcd`; `loan_accrual_us_audit_7982bcd.zip` SHA-256
`2eb0769e3865432528b9d8cdebcf7d2b1fba074ae53b43242acb342da0af71aa`.
All 264 members were restored and hash-checked, including complete current research
source, the patch, audit reproducers/results and acceptance source-code hashes.
All 28 bound original source receipts were verified. Immutable inputs and old fits
were not duplicated. The canonical run pointer binds loan_accrual_us_recovery.

## CVM capital/receipt audit and recovered issuer coverage — 2026-09-20

The independent CVM audit reconciles 21,783 own-version documents, including 21,767
exact-minute receipts and 16 date-only bounds. Those clocks reproduce 880,858
statement-age rows. All 21,749 HTML capital tables reconcile in their printed share
units; 184 prior own-note dispositions retain exact source identities/arithmetic.
109,777 source-file hashes match. Thirteen remaining ZIP capital records also
match. Timings: 96.30s and .83s. These are new upstream checks, not repeats of the
completed quote/lending/auxiliary/conditioning censuses. Original financial account
numerators, TTM and every valuation denominator still need their own audit.

Two missing C&A capital tables were recovered with exact filing dates. The 2019
reference/version2 record is first usable August21 2020, not backdated; the 2022
record enters May6. Their incremental feature effect is zero under both original
and repaired issuer joins. A real upstream defect emerged instead: subsequent FCA
fallbacks discarded literal CEAB3 tickers, and a generic-shares ticker was forced
through an inexact legal-name join. C&A retained model eligibility but lost its CVM
link after May26 2020.

The parser now reads exact HTML tickers and flat FCA XML with validated envelope,
issuer/year/version and strictly recorded UTF-8/Windows-1252 decoding. Generic
shares take ON/PN/suffix only from prior dated B3 observations; no modern enum label
or fuzzy name assignment. Multiple securities within a tab retain their own ticker
and listing dates. The immutable loader reparses verified source bytes and leaves
source manifests unchanged.

The admitted C&A source/financial overlay restores 1,146 already eligible dates,
May27 2020 through December30 2024, growing issuer rows from 140 to 1,286. Existing
identity and financial rows remain exact; future-document removal preserves earlier
rows. Eight financial fields gain 8,578 usable values; statement-age and dependency
metadata are reported separately. Paired generation took 15.50s. No eligibility,
quote, label, neural tensor or old policy coordinate has changed. Full derived-store
propagation and compatible refits remain required before using these observations.

A bounded reread of the 2,820 old FCA fallback records now completes without errors,
finding changed metadata in 1,068 documents across 327 issuers; 283 existing ZIP
packages yield original XML. The first pass exposed 71 multi-security row groups;
the new row association and final affected reread resolve them (9.97s). The prior
attempt and code snapshots are retained. These wider candidates are not blanket
admitted: global issuer conflicts, dated identity and all dependent families need
quantification. C&A's ten refreshed metadata records are identical under the final
row parser, without repeating its financial replay.

Twenty-four affected parser/identity tests pass, including new encoding/source
binding, literal ticker, per-security row grouping and future-identity protections;
Ruff passes. The ZIP filename and admission-report junction failures were audit
harness defects, not financial-data mismatches. See v2_CVM_SOURCE_AUDIT.md and the
run pointer's CVM/CEA receipts. Historical converter/revision fidelity remains
unknown for a single archived vintage.

Registration reread. A/B remain incomplete, C/D unstarted; no GPU job or corrected
model-profitability result. Next: admit the wider FCA improvements only after a
global causal identity check, propagate C&A/ALLOS/ISA and other supported repairs
through dependent families/wealth/labels into a separately accepted store; complete
remaining upstream account/denominator clocks and Stage A execution bounds. Preserve
all names/history and separate accounting-only replay from new coordinates/refits.

Recovery: implementation `44f797d`; `cvm_sources_identity_44f797d.zip` SHA-256
`cf7466cb78b307f79354e2b9944e7fc4f52c93d53dc22d484d5deb1814408efc`.
All 335 members were restored and hash-checked, including complete current research
source, exact captured reproducer bytes, the prior parser snapshot, source indices,
attempt evidence and C&A parquet overlays. Fifty-two newly recovered original
source files were verified; the earlier 109,777-file audit was not rerun. Immutable
inputs and old fits were not duplicated. Resolve cvm_source_identity_recovery.

## Full FCA identity and financial/event propagation — 2026-09-20

The full causal reconstruction first reproduced all 880,858 sealed identity rows
exactly. Its broad fallback/C&A candidates exposed 708 lost eligible mappings;
source review recovered wrong CSN/Eletromidia ticker literals and duplicate
Recrusul class tickers. Thirteen explicit amendments bind original offerings,
prior own-ID FCA rows and dated quotes. The new per-security correction clock
keeps later offering evidence out of earlier filing decisions, without delaying
unrelated metadata. Raw sources and all 933 security axes remain intact.

The verified overlay has 943,419 rows: 64,333 added, including 26,271 eligible,
and 1,772 obsolete inactive identity rows removed. All 708 originally endangered
eligible mappings are retained. Shared issuer, CVM, class and sector values are
unchanged; method/sector-mapping provenance changes are separate. No cross-issuer
conflict. Future-document/quote deletion preserves the full pre-2021 prefix.
The baseline proof was reused, not repeated. Revised reconstruction took 58.37s.

The next stage actually propagated the overlay into financial and event parquets.
An incremental extraction found 427 additional own-version financial filings in
the existing archives; the old 21,783-document cache remains immutable. Eight
financial fields gain 154,656 usable eligible observations, statement age gains
26,271 separately, and seven event fields gain 181,629. These are totals versus
the sealed store, including the earlier C&A overlay. No prior valid eligible
value is lost. All shared valid financial numbers/event values remain exact;
213 eligible valuation-availability flags improve after class-identity resolution.
Every financial/event value and age for 203 unaffected names remains exact.
Extraction and propagation took 131.36s, not a model-training ETA.

Read `v2_FCA_IDENTITY_PROPAGATION.md` and `v2_fca_identity_acceptance.json`; resolve
fca_identity_admission/fca_financial_propagation through the run pointer. Ten
affected identity tests and Ruff pass. The source-clock guard and JSON-date
diagnostic failure are preserved; no source/model outcomes were changed by failed
attempts. No prior quote/lending/auxiliary/conditioning/CVM census was repeated.

This is an intermediate derived contract, not full B acceptance. Remaining FCA
anomalies include MGLU3 labelled PNA in annual data and ABCB4 preferred suffix
inconsistencies, plus missing early literal tickers. Magazine Luiza's original
April29 2011 offering is archived and visually verified but not yet applied to
those additional records. ABC's indexed 2007 underwriter PDF returned non-PDF
content; current FAQ text is not used for historical admission. Resolve exact
source-specific class/ticker evidence rather than weaken global identity checks.
Other identity-dependent families, ALLOS/ISA full history propagation, financial
numerator/TTM/denominator/source clocks, wealth/labels and actual new tensors remain.
Reuse the new 427-document cache on subsequent identity amendments instead of
re-extracting it. Accepted stores/static learned-policy coordinates are unchanged.

Registration reread. A/B remain incomplete and C/D unstarted. No GPU job or
corrected profitability was produced. Stage A still needs execution/recall and
presettlement-disposal bounds; the source audit must not replace those requirements.

Recovery: implementation `c914056`; `fca_identity_financial_c914056.zip` SHA-256
`7bfd9aedb45c82b21f8afa74b9de70f7e8294cc647d1d5d3c788ab301cd40726`.
All 306 members were restored and hash-checked, including complete current
research source, exact bytes for the initial/final audit implementations, source
patch, reproducers, identity deltas, the new 427-document cache and financial/event
parquets. All 539 bound original-source receipts were verified. Immutable inputs
and old fits were not duplicated; prior passed censuses were not repeated. Resolve
fca_identity_financial_recovery. Acceptance SHA-256:
`cc907ccedb4c73e598f5be31749a1af4a067cee04199d0e289953da44c5d32c2`.


2026-09-20 — MGLU/ABC source exceptions and dependent peer propagation:

- The issuer's original MGLU Apr29 2011 announcement binds MGLU3/BRMGLUACNOR2,
  ordinary shares and next-business-day trading. Twenty-six exact FCA amendments
  resolve omitted tickers/listings or conflicting ON/PN fields. ABC's original
  Jan30 2008 release was recovered from its 2007-only issuer archive and visually
  verified; it explicitly names ABCB4. Thirty-two exact amendments join original
  CNPJ/CVM filings and prior COTAHIST BRABCBACNPR4/PN. PNA wording remains in the
  source evidence; this is a security-specific exchange join, not an asserted
  legal conversion. No failed 2007 prospectus bytes or current FAQ were admitted.
  The issuer archive's quarter-end metadata is not used as publication time.
- New exact-ISIN constraints accompany external correction clocks; no corrected
  ticker can bind another security. All 71 explicit source amendments retain
  before-fields, evidence hashes and receipt/effective dates. No generic class
  relaxation or original-source mutation.
- Full reconstruction: 949,539 rows; 70,453 additions/31,858 already eligible days,
  1,772 inactive removals/no eligible loss. The earlier 943,419-row overlay remains
  exact. MGLU/ABC add 6,120 full-calendar rows, 5,587 already eligible, beyond that
  overlay. Original baseline proof was reused; new global/prefix work took 70.09s.
- Reused all 427 incremental filings, with no source re-extraction. Total recovery
  versus sealed inputs: 179,685 values across eight financial fields, 31,730
  statement-age values separately and 219,291 across seven event fields. Existing
  valid values and the earlier overlay's shared values remain exact. The existing
  213 eligible valuation-flag corrections remain. Runtime 119.11s.
- Actual sector, cross-market and lending-utilization parquets now propagate the
  full identity repair while holding each family's original market inputs fixed.
  Sector fields add 91,468 usable values and change 283,551 existing values;
  exposure/shock-interaction fields change 2,060,611; utilization adds 2,652 with
  every shared valid value exact. No eligible valid values are lost. Other names
  can change through their corrected issuer-equal peer groups. Every coordinate
  outside affected sectors is exact, as are ALL non-exposure cross-market fields
  and the other six lending fields/ages. Raw lending additions remain separate.
- Peer computation took 124.36s. Its first utilization output accidentally kept
  Float64 against the old Float32 storage; max1.39e-8 differences vanish after
  restoring the existing type. Initial output/reproducer is retained. Only the
  typed utilization result/comparison was corrected, without repeating passed
  sector/exposure work. A UTF8 diagnostic console issue had no data effect.
- Eleven affected identity tests and Ruff passed; full causal prefix and prior
  overlay preservation passed. Readout/source bytes, receipts and per-field deltas
  are bound by v2_fca_identity_acceptance.json and the run pointer. The registration
  was reread. These are CPU source/feature results, not fit timings or profitability.
- A/B remain incomplete and C/D unstarted. Next B priorities: independently trace
  financial/float numerators, TTM and denominators (including the new427 filings),
  remaining original clocks/revisions and ALLOS/ISA full history/universe/warmup/
  wealth/labels, then actual tensors in an explicit new store contract. Current
  peer artifacts hold original wealth fixed; final wealth changes must propagate.
  A still needs denied-renewal/recall, presettlement-disposal and adaptive-book
  source sensitivities. No old checkpoint was scored on changed coordinates;
  no new GPU fit, held-out consumer or deployment was launched.

Recovery closeout for implementation `560c1a3`: `fca_issuer_peer_recovery.json`
verifies all316 restored/hash-checked members, 67 source/prior-artifact receipts,
all executed code bytes and all new family outputs. The 75,968,940-byte archive
`fca_issuer_peers_560c1a3.zip` has SHA-256
`73f08d9457156c61afab3fd8ff8bf71a5cec5f6eb6fca38206b8b1cb5086e54c`.
It reuses the verified earlier recovery for the427-file extraction cache, without
duplicating immutable sources or old fits. Acceptance SHA-256 is
`faca30ee5915a164755bb13c4fda8277fdce24c7dcae13f1b8c09a754be404df`.
The canonical pointer binds this recovery; earlier receipts remain valid for their
recorded source commits. All CPU work is complete at this checkpoint; no worker
or GPU job is active. The heartbeat remains active for the remaining A-D work.

## Financial numerator, formula and denominator audit — 2026-09-20

- Independent original-row extraction checks all22,210 own-version filings,
  including the427 incremental filings:273,625 selected account values, periods,
  basis, descriptions and BRL scales reconcile. Channels are240,434 annual-CSV,
  33,086 own-version HTML and105 ZIP/XML account rows.21,431 consumed file hashes
  match;81.05s. This is the previously untested account-value boundary, not a repeat
  of the completed capital/header census.90 reviewed unit after-states also
  reconcile; a narrow mutation check rejects a missed correction.
- Independent as-of formulas check949,539 financial-overlay rows,7,596,312
  value/missingness comparisons and their dependency/update ages:19,451,154
  initial comparisons,33,084 state recalculations,482.60s. The initial1896
  differences all concern Camil's February fiscal year. The audit's date function
  missed leap-year month-end preservation; production was correct. Only Camil's
  full1406-row history was rerun after correcting the audit (15.22s), with zero
  remaining mismatches; max independent SUE arithmetic difference8.89e-16. The
  passed remainder was not repeated. Original failed output/source are preserved.
- All427 additional capital tables and8991 FRE float rows reconcile in1.89s.
  The float includes4104 exact-minute receipts; seven source rows have a future
  measurement date, retained as printed. An independent actual-consumer audit
  checks242,232 balance candidates and reproduces all29,943 valid utilization
  ratios/ages exactly in4.98s. Four relevant future-measurement candidates are
  excluded; no selected denominator is below10,000 shares. Capital events, class
  identity and observed unit boundaries remain effective guards.
- No production financial/lending implementation, raw source, stored value,
  eligibility, model coordinate or forecast changed. These results increase
  confidence in the earlier issuer-recovery propagation; they are not alpha.
  Selected-account agreement does not certify completeness of all potentially
  useful disclosures or single-vintage historical revision fidelity. Peer/financial
  contracts still hold original wealth and lending inputs fixed.
- `v2_FINANCIAL_FORMULA_AUDIT.md` and `v2_financial_formula_acceptance.json`
  bind the full and narrowly qualified evidence. Acceptance SHA-256:
  `2bae10a2fd7b6d9a49c3578a7ca857194ce2cd89dc075d6cf099062396ba2e76`.
  Four new audit reproducers pass Ruff. No passed model test suite was repeated.
  The registration was reread; all CPU workers have finished and no GPU fit ran.
- A/B remain incomplete, C/D unstarted. Continue remaining upstream-family
  clocks/revisions/units and ALLOS/ISA full history/universe/warm-up/wealth/labels,
  separately propagate recovered lending, then accept the derived store and
  actual tensors. A still needs denied-renewal/recall, presettlement disposal and
  adaptive-book bounds. Do not repeat these completed financial audits.

Recovery closeout for `01702ca`: `financial_audit_recovery.json` binds293
restored/hash-checked members, all new audits and executed code, including the
initial Camil failure and its narrow qualification. The5,408,623-byte archive
`financial_audits_01702ca.zip` has SHA-256
`37aae93be7ba035c911ba3fd595588e9fd3afd0bcafd352b3e522154a1ae7e2a`.
It binds22,317 original-source receipts already verified by these new audits;
recovery does not reparse or rehash that source census. The earlier issuer/peer
recovery archive is verified and retained. Immutable inputs and old fits were
not duplicated. Canonical pointers are updated; the heartbeat remains active.

### ALLOS/ISA daily and peer-history propagation (2026-09-20)

- The three sourced rename links now propagate on all3,717 sessions/all933
  permanent axes with2009 warmup: +60ALOS/+28ISAE4 eligible days,12 stale
  predecessor entries retired, no other eligibility change. Raw quotes remain
  unchanged. This is an intermediate daily-array contract, not a replacement
  accepted store or permission to score old fits on new coordinates.
- The actual consumer had still read the successor's empty earlier column.
  Dated unit/no-cash mappings now route predecessor values/masks/ages into later
  60-session windows, including compact names, while preserving permanent output
  indices. Effect and announcement clocks both gate routing. Public pre-birth
  successor wealth/risk/history coordinates stay empty; the producer's internal
  history basis is separate.
- The first full pass rebuilt32 slow fields in195.63s. Its loss census exposed
  a second gap: monthly clustering ignored predecessor history. The narrow
  five-field repair carries the current cluster and uses the predecessor's
  prior126 sessions at monthly fitting, with101 observations still required.
  Its1/5/21-session return inputs reproduce exactly, the other27 fields and
  pre-event prefix stay exact, and it runs in14.99s. Combined slow masks gain
  5,453 values/lose214:124 belong to retired predecessor entries,90 still-active
  cases have only two other valid peers under the unchanged three-peer rule.
  Every loss is enumerated; no difficult security is excluded. Shared valid
  slow values change642,975, including legitimate cross-sectional effects.
- Labels gain461 valid name/horizon outcomes, none lost. All arrays are exact
  before the ten-equity-session window that can cross the first rename.
  The first label attempt used stored float32 prices and disturbed ties; the
  archived Round-7 decimal-cent precision is now reproduced. A calendar-day
  rather than session-count qualification bound was fixed without repeating
  the passed slow build. Exact initial code/output attempts remain onD.
- The final actual input audit covers12 dates,933 names/full60 sessions and
  65,171,916 array/packed-CPU-tensor cells withzero mismatches. The pre-rename
  tensor is exact against the sealed store. Final audit19.67s; no neural
  forward/scoring/GPU fit and no alpha claim. Three new routing tests passed;
  a65-pass affected batch caught one read-only scratch-write error, fixed by
  changing only the new published copies. Subsequent six- and eight-test
  affected/causality batches pass (overlapping), including cluster history.
  Ruff passes. The registration was reread.
- Source-specific remaining boundaries are recorded. GOLL2011-02-16 has invalid
  OHLC bounds but useful printed activity. Rename-only attribution retains the
  sealed mask so it does not silently remove20 unrelated eligible sessions;
  separate activity/price-mask admission remains. The current FCA overlay has
  noISAE4 issuer rows and misses the first twoALOS days; source-bound identity
  inheritance must propagate these, rather than fuzzy matching or loan aliases.
- Read `v2_RENAME_HISTORY_PROPAGATION.md` and
  `v2_rename_history_acceptance.json` (SHA
  `0f0c277c4297c8a309983beb74b258506ad284d803297fcea29381bf71b2a6cd`).
  Resolve `rename_history_propagation`, `rename_peer_propagation` and
  `rename_input_audit`. A/B remain incomplete andC/D unstarted. Next propagate
  dependent issuer/auxiliary/M1/lending families, remaining contractual
  wealth/labels and accept the complete new store. A's execution/lifecycle
  sensitivities remain required. Do not repeat the completed32-field build,
  five-field qualification, actual tensor checks or prior source censuses.

Recovery closeout for `6b7f83a`: `rename_history_recovery.json` binds345
restored/hash-checked members and66 arrays actually reconstructed from sparse
deltas against the sealed parent, including initial attempts and history bases.
The46,562,448-byte `rename_history_6b7f83a.zip` has SHA-256
`70b0021f505f04af4b5e24a903d3da65b99cc443a9c5d4595f59e7f5b599a0cb`.
Exact runtime code, source patch, audits and historical-code resolutions verify.
The two input-audit stores are rebuildable from the same arrays/axes and their
archived manifest/table bytes; duplicate data copies are omitted. No immutable
inputs or old fits were archived again. Canonical recovery pointer updated.

### Rename issuer/dependent families and GOLL activity (2026-09-20)

- Source-bound unit links now preserve exact issuer/class identity through
  ALLOS/ISA. The949,568-row overlay adds31 mappings/30 eligible days and retires
  two stale inactive rows; shared mappings remain exact. Sector metadata keeps
  its separate known date. Two future-row deletion prefixes and conflict/late
  knowledge fixtures pass. Prior-price valuation follows the same admitted
  links without erasing predecessor capital barriers or inferring loan aliases.
- Recomputed only the two affected financial issuers using the existing audited
  extraction: +240 eight-field financial values, +30 statement ages separately,
  +210 event values; no valid loss or changed shared value. Runtime21.00s.
- Independently sound printed GOLL2011-02-16 volume/quantity/trades now survive
  its invalid OHLC row. Prices remain unobserved, eligibility unchanged. The
  four-field old control matches74,640 cells. The activity amendment changes471
  ranked values across124 names in the next20 decisions, no mask/age loss, and
  changes20 physical log-volume observations. Invalid returns remain invalid.
- Rename propagation now updates sector, magnitude and cross-market families
  on the changed tail with all933 names and original lookbacks/support. Relative
  to prior FCA families: sector+457 valid/change3,648; magnitudes+352/change662
  includingGOLL; cross-market+6,508/change19,111. No valid values lost. Original
  oil/non-oil feature return contracts and cash-source coordinates stay distinct.
  Source shocks/flows are routed to restored eligibility without inventing ADR
  pairs or publication clocks. This is not a completed new store or model replay.
- The91.31s dependency run initially promoted volume toFloat64. A narrow
  log-volume/flow qualification restoresFloat32. Its first one-column sum used
  NumPy's different pairwise reduction; a strided reduction now matches the
  original all-name producer. Initial outputs/executed code remain preserved;
  passed sector/volatility/regression work was not repeated. An earlier GOLL
  audit-axis error used a replicated matrix where the universe needs a vector;
  corrected without changing source values. Final activity qualification1.86s.
- Independent scope checks preserve all values outside affected issuer/sector/
  activity paths. The actual dated-family loader matches3,090,096 value/mask/age
  cells across92 fields,12 dates and933 names,3.18s. This verifies intermediate
  family alignment, not a neural forward or complete-store acceptance.36 affected
  tests and Ruff pass; no accounting/source census was repeated. Registration
  reread, all workers finished, no GPU fit or corrected profitability result.
- `v2_RENAME_DEPENDENCIES.md` and `v2_rename_dependency_acceptance.json` bind the
  milestone; acceptance SHA-256
  `b856e9e0ba8e0bd1afca877e33125077a59c8a6e2d48adf17a642aa01a0fb52a`.
  Resolve `rename_issuer_propagation`, `goll_activity_admission`,
  `rename_dependents`, `activity_magnitude_propagation`, `dependency_update_audit`.
  A/B incomplete, C/D unstarted. Next propagate recovered lending and remaining
  M1/auxiliary families, finish contractual wealth/labels and remaining source
  clocks/revisions, then accept full derived tensors/refits. Stage A execution
  bounds remain required. No changed coordinates may enter old checkpoints.

Recovery closeout for `b8dfd19`: `rename_dependency_recovery.json` binds289
restored/hash-checked members, including16 new derived parquet files, all current
research source, the source patch, initial attempts and four historical artifact
resolutions. The188,533,587-byte `rename_dependencies_b8dfd19.zip` has SHA-256
`8c39f0ad6d78612d99a83ea161ffc2b48c45e791498534d501e1bb5f651fbb2b`.
Recovery reused prior source/rename/FCA/financial receipts without another census;
no immutable source dataset or old fit was copied. Canonical pointer updated.
The heartbeat remains active until the complete A-D program is finished.

## Recovered lending features and remaining M1 history (2026-09-20)

- The new feature-source union retains all242,232 original balances exactly,
  including13,501 absent from the recovered economics archive, and adds1,722
  qualified rows:243,954 total. All47,836 old feature-rate keys occur in the
  reconciled202,698-row rate archive;29,661 rate/flow keys change. No source
  census or completed financial/FCA reconstruction was repeated.
- Earlier July15 2020 monetary-unit uncertainty remains on267 mapped amounts;
  quantities remain usable. The single October15 2020 GMAT source quantity/amount
  remains unsupported. The recovered printed rows remain untouched; no decimal
  rescaling or wholesale exclusion of securities/reports was invented.
- Six-field control reproduces184,222 original active-date rows. Seven lending
  fields gain76,840 usable values and change46,673 shared values versus priorFCA
  features on renamed eligibility. The three balance fields gain38 observations
  each from predecessor ADV history; other source effects are separately reported.
  No loan alias, rate, renewal date or flow record is inferred from a spot rename.
- New utilization checks retain predecessor capital-unit uncertainty with dated
  links.126ALOS values lose support, explicitly enumerated: FRE127027 has
  March21 2023 measurement/May19 knowledge and532,365,440 shares; predecessor
  DISMES102to103 onMay2 lies inside that denominator interval. This is uncertainty,
  not proof of a split or an erroneous printed count. Exact prior ratios remain
  recoverable. No other feature loses valid support.
- Qualified economic-source panels preserve all rates/imputation/placeholders,
  strict/open availability, and every eligible balance-availability cell versus
  the recovered economics view. The actual archive loader verifies all six arrays.
  An initial generated manifest lacked byte counts; only its inventory was fixed.
- Future-source deletion preserves the prefix throughDecember28 2023;195,930
  actual family-loader cells match. Qualification took3.26s. Initial control
  inactive-row/unknown-age comparisons, the too-strong no-loss assertion and a
  restricted-calendar fixture failure are retained with executed bytes. No full
  passed propagation was repeated for the final denominator qualification.
- Nine affected B3 tests pass (two earlier focused tests overlap); Ruff passes.
  This is a new intermediate feature contract, not old-policy scoring or finalB.
  `v2_LENDING_FEATURE_PROPAGATION.md` and `v2_lending_feature_acceptance.json`
  bind the results; acceptance SHA
  `4d3666f0e31105dba333443e739d1b5b5bcdde7a7dad3e9d8d3a3d5d6b48ffdb`.
- M1 diagnostic checks91,080 actual native-consumer cells on all88 restored
  eligible dates, with exact permanent indices/current-day streams. Both new
  ISINs lack scaled-return/range channels for their first20sessions; same-clock
  volume supports only2ALOS/4ISAE4 dates in that interval. Source mappings and
  exact producer paths are bound. This is a concrete remaining history repair,
  not completion of M1 or permission to fill missing channels synthetically.
- Resolve `lending_feature_propagation`, `qualified_lending`,
  `rename_m1_support_audit` and `lending_feature_acceptance`. Next repair bounded
  native M1 predecessor windows, remaining auxiliary histories, contractual
  wealth/labels and full derived-store/refit tensors; StageA execution/lifecycle
  bounds remain. A/B incomplete, C/D unstarted, no GPU fit or profitability result.

Recovery closeout for `eca5b6b`: `lending_feature_recovery.json` binds268 unique
archive members and277 restored logical members, including nine deduplicated
new-artifact aliases. The12,797,344-byte `lending_features_eca5b6b.zip` has SHA-256
`18cd2df0ff4c93f629ed9d93402cfe491d2895f72531dc61e9bac9869d100969`.
The restored actual archive loader reproduces all six loan arrays. Existing
recovered rate bytes remain an explicit external dependency of the earlier
lending-source recovery; no immutable dataset or old fit was duplicated. The
current research source, binary patch, initial attempts and final qualification
are recoverable. A/B remain incomplete; heartbeat stays active.


## Native M1 rename history admission — 2026-09-20

The bounded original-source M1 continuation now repairs ALOS/ISAE4 reset histories,
preserving effect/knowledge gates, original assignment bounds and missing-minute
semantics. No source, accepted store, eligibility or old checkpoint is changed.
Native validity gains7,419 cells/no losses; first20-session scaled-return/range
support is restored, with same-clock volume still requiring16/20. Current-day
stream/patch/age channels remain exact. Initial native numeric changes11,253;
an additional28,287 ALOS values propagate the already admitted small daily-scale
differences through2024 rather than reverting to old coordinates.

The sealed bounded controls match243,916 cells. Future-source deletion preserves
prior native/scalar outputs; the fixture checks delayed knowledge, source bounds
and entry-bar exclusion against an uninterrupted-source oracle.18 existing
affected tests passed; the new case plus one overlapping assignment check pass
after correcting displaced fixture-tail assertions. Ruff passes. The actual
dataset/collator checks314 full933-name/60-session samples and36,892,996 packed
native cells, zero mismatches; all88 restored eligible dates are included.
No model forward, GPU fit or profitability inference.

Raw scalar gains196 (71ALOS/125ISA) remain intermediate: final cross-sectional
transforms/ages and to-close targets are still required. The native input audit
does not claim these boundaries. Reconstruction9.35s and resumed consumer audit
18.57s are not fit estimates. An initial68-versus69 patch-padding comparison and
a collator-key readout failed in the audit, were corrected, and retain executed
bytes/outputs. The passed tail source computation was reused.

Resolve rename_m1_propagation, rename_m1_input_audit and rename_m1_acceptance;
see v2_M1_RENAME_HISTORY.md. Next finish scalar/auxiliary histories, remaining
upstream clocks and contractual wealth/labels, then accept the complete derived
store/refit contract. A execution/lifecycle bounds remain. A/B incomplete,
C/D unstarted; heartbeat remains active.

Recovery closeout for b59b869: rename_m1_recovery.json verifies283 unique archive
members/297 restored logical members, including14 deduplicated aliases. ZIP
rename_m1_b59b869.zip is1,966,223 bytes, SHA-256
`cd366eaedd4fc54f3bebd2ab2c15ced8e1504c2cde18af4c7c776a029edbb8af`.
Restored sparse amendments reconstruct82,102,962 actual audit-view native cells
exactly against the sealed dependency. Complete current research source, source
patch, exact initial/final reproducers and initial outputs are recoverable;
immutable inputs, original sources and old fits are not duplicated. Acceptance
SHA `ea75bfca068c741a01624532dfbcd2e902fcebf96965406cb4fb5706bc3c123b`.
Prior source/rename/lending recovery receipts remain valid.


### M1 scalar/age and original-convention target assembly (2026-09-20)

The bounded all-name scalar cross-sections now match 3,358,800 sealed value/mask
cells after narrow ALOS consistency qualifications. No passed native or source
census was repeated. Final transforms add1,116 valid cells/no losses:936 from
eligibility and180 from inherited history. Shared valid values change104,448;
1,900 ages and350 support cells change.78 valid to-close outcomes are added,
76from eligibility and2from the original-reference rename bridge. Source ages
carry only the pre-effect snapshot after effect/knowledge.

The actual dataset/collator verifies90full933-name/60-session samples and
10,412,280 scalar/target cells, zero mismatches. An independent arithmetic oracle
matches8,786 supported outcomes.14feature/age and5affected store tests pass;
Ruff passes. Initial raw-tail/age-harness attempts and successful reused ALOS
assembly remain recoverable. No model scoring or GPU fits.

The separate clock audit finds all89,799 existing valid auxiliary outcomes use
fixed405/345normalization despite dated sessions;65residuals are clipped. AERI's
Dec30 2024 sealed M1 retrospective boundary differs from its stored action arrays;
retain the old non-rename diagnostic until the corporate source/wealth audit.
These are still intermediate new data. See v2_M1_SCALAR_ASSEMBLY.md and resolve
m1_scalar_assembly/input_audit/attribution/target_oracle/acceptance. A/B incomplete,
C/D unstarted. Next correct the auxiliary target clock and finish remaining
auxiliary/corporate wealth/label and full derived-store acceptance, alongside A.

Recovery closeout for2cb6c36: m1_scalar_recovery verifies312 unique/318 logical
members with6 deduplicated aliases. m1_scalars_2cb6c36.zip is8,424,288bytes, SHA
`ae45f0b9ba7cb49287fd0b0c0a7a2c90a2a934071cce63a952762de0566fa5a7`.
Restored sparse amendments reproduce16,495,440 scalar/target audit-view cells
exactly. Current research source, patch, initial/qualified executed recipes and
raw scalar cross-sections are recoverable without immutable-input or old-fit
duplication. Acceptance SHA
`d02c4400927f6cdb1521a64acbc0a3935b1b5056417e48ffe194b765b03c4623`.
Attribution source formatting/helper equivalence is explicitly qualified; its
original hash and unchanged report remain recorded. Prior recovery dependencies
are preserved. No worker remains active.

### Dated auxiliary normalization and AERI lineage (2026-09-20)

The producer now supplies dated session/prefix clocks to the optional to-close
target. The current pre-decision RSS estimator and median/clipping/ranking remain
fixed. Reused 17,817 saved inputs plus 72,060 necessary original-source name/date
endpoints reproduce all 89,877 existing/rename-admitted outcomes exactly under
the old convention. Clock-only changes: 88,311 residuals, 12 ranks on four dates,
zero validity/raw-return changes; clipped residuals65→19. Independent arithmetic
matches every outcome; actual dataset/collator18samples/all933/full60 checks
67,176 target cells exactly. Recovery23.68s, total24.37s, consumer6.02s.

AERI's Dec30 inherited diagnostic is a pre-Round-7 U2 inference explicitly
reclassified as large_move_no_action in the already bound review. It was not a
missing issuer-confirmed bonus. Exact old-factor arithmetic identifies four
inherited wealth cells; the sparse q1/cash0 amendment restores the 31.29% raw
closing loss and one retrospective consistency bit. Existing five-horizon
outcomes remain exact; the auxiliary target is still unsupported for independent
prefix-return support. No decision through Dec30 consumes the changed final daily
wealth row. Original Dec9 debt/Dec10 controlling-shareholder notices were archived
and visually verified, with no invented equity conversion terms.

18 affected tests pass plus the separate raw-to-store causal fixture; Ruff passes.
No model forward/fit/scoring. Initial AERI date-schema/renderer-path attempts are
retained; successful source bytes reused. Resolve to_close_clock/input_audit and
aeri_boundary/wealth_qualification; see v2_TO_CLOSE_CLOCK_AERI.md. These are
verified amendments, not a final accepted store. Next finish remaining auxiliary
histories/joins, contractual wealth/labels and the complete store/tensor/refit
contract alongside A's remaining adaptive-book bounds. A/B incomplete, C/D unstarted.

Recovery for0fe77e9 verifies335 unique/356 logical restored members, including21
deduplicated aliases and both new original CVM notices. Archive
to_close_clock_aeri_0fe77e9.zip is4,345,604bytes, SHA
`75d18056132788e96cfb5d51ed536fc8ed8fb6416e85a864cd5855dc19b7ffe4`.
Restored sparse changes reproduce2,015,280 actual target audit-view cells and
all five AERI wealth/diagnostic changes exactly. Complete current research source,
source patch, initial/qualified executed recipes and new target inputs recover;
immutable inputs and old fits were not duplicated. Acceptance SHA
`569544f83eadc2feb2247068dfcb686be7fe04caafa760bac1bd5ea616f75f03`.
Prior M1/native/history/dependency/lending recoveries remain explicit dependencies.


## Four remaining auxiliary histories — 2026-09-20

Activity/options retain exact 20/5-session support and publication clocks across
ALLOS/ISA. Odd-lot retains exact five-session vintage joins. Index pressure now
routes both preview and prior composition only after the share rename is known
and effective; prior ADV follows its admitted share history. All 933 names remain.
Four families add 778 valid cells and retire 16 predecessor cells; no valid
still-eligible loss and no shared valid numerical change. There are 38 restored
index rows: 27 already have standalone ADV but need the composition correction,
11 need inherited ADV too. Read the source-audit attribution qualification,
which supersedes only the initial index attribution counts, not its passed arrays.

The bounded producer control matches 3564 activity and1615956 index grid cells.
Independent formulas check621 activity/702 odd-lot fields. All44 original index
portfolios/12712 rows reconcile. Eight original IN/PR headers reproduce selected
versions around both renames. Older IBRX naming, alphanumeric B3SA3 and Mac ZIP
metadata caused audit-only failures; executed versions and completed-workbook
reuse are retained. GOLL activity already existed in this family's cash path.

Thirteen targeted tests and Ruff pass. Actual dataset/collator90samples/all933/
full60sessions checks6549660 packed field cells, zero differences; full array
qualification135250479cells. Producer4.48s/consumer26.45s; source final resumption
.58s excludes already completed work and is not a fit ETA. No new forward,
scoring or GPU fit. Resolve remaining_auxiliaries, remaining_auxiliary_input_audit,
remaining_auxiliary_source_audit and v2_REMAINING_AUXILIARY_HISTORY.md.

Final combined store/refit contract, contractual wealth/labels, remaining original
option/cash numerical source boundary and other upstream gaps still require work.
Stage A's adaptive-book execution/lifecycle bounds remain. A/B incomplete;
C/D unstarted. No passed source census or earlier model tests repeated.

Recovery for dea8c53 verifies572 unique/575 logical restored members and237
previously verified source receipts. Archive auxiliary_histories_dea8c53.zip is
17,291,760bytes, SHA a6dd5fcb4a1ccf684bee3427ce00635fb39b790869681efa62e0768bd6ad2989.
Restored sparse deltas reconstruct196,489,800 actual auxiliary audit-view cells
exactly. Current research source, source patch, qualified families, original
executed recipes and failed audit attempts recover without immutable input/old-fit
duplication. Acceptance SHA a2524456b3659adb4648eb9c5c846f51fe297d3db01eae6c700c32f6cc78f545.
The earlier history/M1/lending/target recovery chain remains required and valid.

### Original option/cash numbers and corporate basket targets, 2026-09-20

All1283 original IN/PR publication jobs pass6334727 field comparisons:
210071 OI aggregates,555511 cash rows,41371764 listed series and18676933
observed OI series. Exact creation-version selection, underlying IDs, dated
listing limits, BRL monetary units and missing/complete OI reconcile without
production parser reuse. Retained malformed/moved report exceptions remain
explicit. Four manifest-relative path errors were qualified without changing
hashes or repeating passed jobs. Eight workers completed the final invocation
in1273.50s, reusing six earlier passes; aggregate worker time10138.18s.

Independent fixed-width option slices reconcile9052797 printed records from
15 original2010-2024 ZIPs into261904 rows/785712 comparisons. The separate
auxiliary cash axis reproduces1173583 rows/5867915 fields from the already
audited equity normalization. The original quote census was not repeated.
All15 option scans were reused while correcting the cash harness's _brl field
names and redundant quote-factor division. Final cash resumption6.19s; initial
raw option wall time was not separately recorded. Single-vintage revisions
and exact first internet-publication times remain unknown. No production data
changed in these source audits.

Sourced BRML/DMMO/CPLE target baskets preserve all legs and cash receivables.
The 33-date/all933 old control matches1693560 cells after replacing the
audit's stored float32 terms with original float64 verified terms; the initial
79 disturbed rank cells and executed recipe remain. Closed forms verify21
new horizon outcomes per conversion,63 total/no losses; all precede fraction
auction recognition. Custody is distinct from economic effect, missing exact
leg endpoints fail, and no basket OHLC is invented. The physical rank changes
8243 cells; the actual virtual neutral target gains63 and changes8186 shared
valid values (8248 total numeric changes), holding repaired risk inputs fixed.

The first consumer view omitted sigma and therefore did not expose the virtual
target; corrected views pass33 full933/full60 samples/4310460 packed cells
and33 revoked endpoint windows. Independent SVD projection verifies the actual
neutral-target arithmetic.16 targeted tests and Ruff pass. Producer.71s and
consumer12.76s are not fit ETAs. No forward/scoring/GPU fit.

Six post-effect liquidity-eligibility cells remain on each converted source,
18 total; their quote masks are missing. Preserve these cells pending explicit
claim-close and feature/eligibility treatment. An erroneous zero-eligibility
sentence was corrected from already accurate case counts without array reruns.
Resolve bvbg_source_number_audit/qualification, option_quantity_source_audit,
corporate_target_propagation/input_audit/neutral_attribution and
docs/v2_BVBG_CORPORATE_TARGET_AUDIT.md. Complete combined store/final tensors/
refit contract, remaining corporate wealth and Stage A adaptive-book execution
bounds remain. A/B incomplete, C/D unstarted.

Recovery for40c9660 verifies2988 unique/2990 logical restored members,2 aliases,
2564 original XML and15 original option ZIP source receipts. Archive
bvbg_targets_40c9660.zip is5,086,724bytes,
SHA1467ae048b3c8dba3108f5ac887952035c6116c6e9e05a07f6ec092209ac3d09.
Restored sparse amendments reconstruct120242760 actual target-view/row cells
exactly; code, patches, source indices and all executed failed/qualified recipes
recover without immutable inputs or old fits. Source bytes were not rehashed
during recovery. Acceptance SHA88558de8fde377eac4e00e943406a1b8cbc95214477c3cfd657138cf3360c4ee.
Prior source/history/feature recovery dependencies remain required and valid.


## Complete derived-store assembly and corporate-source disposition, 2026-09-20

The explicit new store resolves the 18 converted-source liquidity cells and six
closed-register Cielo cells without deleting membership or inventing quotes.
Eighteen closing entitlement rows reconcile to exact terms/Decimal arithmetic;
all source wealth/feature masks retain their original meaning. 1,328 full-axis
entry permissions close causally, including 24 still-eligible cells, all already
unobserved. Funded/quantity-specific claims stay in the account ledger.

All verified history, issuer/financial/dependent/lending, GOLL, native/scalar M1,
remaining auxiliary, target-clock/AERI and basket-target amendments compose in
63.30s. No old source reducer/census or fit was repeated. All3717sessions/933names,
2009warmup, full60history and original feature schema remain. New P/F conditioning
and weights are required; explicit old-parent transfer is rejected.

Independent array composition1798674885cells and six-family direct source-row
arithmetic1040388300cells pass;33 unchanged arrays are hash-exact to parent.
Actual156 full-population/full60 CPU samples,14pretrain, check1856671832 packed
cells, including all88 restored eligible dates and corporate/final-date scopes.
Virtual neutral targets use final sigma/risk coordinates and independent SVD
arithmetic. All comparisons zero mismatch;74.64s. Five private-runtime future-link
deletions/future-feature-read guards and revoked-target checks pass in9.05s.
Ruff passes for four new recipes; no executed reconstruction failure this turn.

Final losses on still-eligible names vs sealed are exactly158:68ALOSutilization
within the prior126barrier dates and90Embraerpeer fields, all enumerated. Other
losses belong to retired predecessors. Different overlay baselines explain the
126-versus68 counts; do not add incremental feature totals together.

Resolve docs/v2_economic_data_inputs.json, docs/v2_derived_store_contract.json,
corporate_claim_rows, derived_store_assembly, derived_store_input_audit and
 derived_store_boundaries. Full report docs/v2_DERIVED_STORE_ACCEPTANCE.md.
Store manifest2e16c8d6dfbba8c8eb65330635507b1eac473e9790c573d3329d889692e9b3f2.
No old accepted pointer/input or checkpoint changed; no model forward/scoring/GPU.
A's adaptive-book lifecycle/disposal/rounding/calendar/exposed-event bounds remain
before C matched economics and D conditional capacity. Program stays active.

Recovery for implementation 523992938f362416caa25f59cd3dd5edc17907a8 verifies 620
hashed members and reconstructs all 106 arrays /
2839063185 cells plus 46 tables
and both indices to exact sealed byte hashes. ZIP economic_store_5239929.zip
is 18039750 bytes, SHA 273d92e7bbe9b17bff1bca4f58ef8c33c81dc73226e829c210ffde0146579a60. Runtime
46.71s. Sparse amendments reuse recorded parent/history dependencies;
no immutable raw archives or old fits are copied. Exact source, executed recipes,
contract, acceptance, new claim rows and complete store recovery instructions are
recoverable. Resolve economic_store_recovery; earlier recovery receipts remain
valid for their source commits. No GPU/source worker active at this checkpoint.

### 2026-09-20 — denied renewal/recall and numerical accounting qualification

V32 implements the frozen denied-all-renewals and two/four-session ordinary-recall
stresses in both actual adaptive accounts, stock and hedge constraints, notices,
overdue-principal readouts and independent copied state. Old references/rates and
physical-return obligations remain; no unquoted fill, buy-in charge or replacement
locate is invented. Notice rights/deadlines are hypotheses. Known morning delivery
precedes recall identity; three final narrow cases pass without repeating books
whose recall dates do not intersect any sourced delivery.

Historical engineering exposed full-cover roundoff roots being renewed and billed
old minimums. Actual fills now carry full-return intent; relative summation bounds
retain genuine tiny partial loans. Mandatory closure bypasses the ordinary order
deadband. Strict optimizer faces use the existing adjoint tolerance and dual signs;
free optima remain. Initial failed root/zero-exit/adaptive qualifications are retained.

Twenty-four 128-session/all933-name engineering scenarios passed identical-intention
money comparison, max R$2.0489096641540527e-8. Actual adaptive synthetic-preference
books use R$10m/R$1m/R$5m over two source-bound 2019/2024 windows, with the unchanged
bundled4bp engineering bridge (no B3 spot addition). They are not model forecasts
or final corporate pricing. Independently adaptive paths have maximum total NAV
differences .01943544645/.29492822018/.13988375371bp respectively, reflecting discrete
old minimums on small different trades. Max target distance3.1801671271300402e-6 is
a two-path comparison, not individual QP infeasibility; the original misleading
field label is qualified without book reruns. Carry this measured uncertainty into
actual-model attribution and remaining rounding/grouping bounds.

At R$10m, 2019 denied/recall2/recall4 retain65/76/74 overdue sessions; 2024 denial
retains5. Missing-quote held exposure ranks NATU BRNATUACNOR6 R$157068.2990,
ALSC BRALSCACNOR0 R$141608.5343 and SOMA BRSOMAACNOR3 R$28065.4469 for next original
source review. These are inventory priorities, not admitted event terms. Source
obligations remain unresolved. No accepted store was changed.

Successful 24-case audit95.0149s, not fit ETA. Ten distinct new lifecycle cases
passed across focused batches; eight allocation tests, six earlier finite-renewal
tests and five directly affected loan/gradient/copy checks passed during development
(overlapping batches not summed). Final changed-code Ruff passes. Resolve
loan_return_notice_audit, loan_return_notice_qualification,
loan_return_notice_runtime_qualification and loan_return_notice_acceptance. Full
report `docs/v2_LOAN_RETURN_NOTICES.md`. Stage A still needs exposure-ranked events,
pre-custody disposal, allocation/timing/sweep/fraction/invoice/grouping/minimum and
older clearing-calendar bounds before matched C economics/refits and conditional D.

Verified loan_return_notice_recovery for implementation ceec715bcfadd19173eb46d3b7dcd2582c188201:
769unique/823logical members,54aliases,24completebooks/11532288 recovered account
cells. ZIP loan_return_notices_ceec715.zip,25271773bytes,
SHA b4805ecc37485cebfd723b1e331574b9943d7bed2820eeec282a6c4b1d9496d9.
Recovery2.26s. Complete research, source patch, recorded initial/qualified executed
recipes and evidence recover; prior dependencies reused without raw/store/fit copies.
Acceptance SHA c265a296b86a31632c31f7412c24bb6f0e6ce95d0ad8accb889259e93bcbdd18.
All workers finished. This completes the lifecycle engineering checkpoint, not A
economic admission or the program.

### 2026-09-20 — held-event originals and evidenced Natura gross-data amendments

The user-named held NATU/ALSC/SOMA investigations now bind 17 original PDFs,
216 bounded RAD issuer/year records and 20 original normalized quote rows. Exact
shareholder conversion ratios are 1 NTCO per NATU, .787808369 successor ordinary
per ALSC and .121695988348 AZZA per SOMA. NATU effect December18/creditDecember20
2019 and SOMA effectAugust1/creditAugust5 2024 are separately sourced. ALSC effect
August6 2019 is confirmed; physical credit and fraction-auction receipts remain
unestablished. Original SOMA auction terms firstknownAugust23/paybyAugust26 retain
printed approximate-price/invoice uncertainty. B3 index circulars do not prove loan
conversion, fraction or rent terms. All three first successor prior references
remain missing; a causal valuation bridge must precede account admission.

Original Natura notices prove two defects in both accepted stores: September18
2019 q1.977869987487793 should be q2 (September20 custody); November7
q.9102639555931091/cash0 should be q1/grossJCP.12784527353, payableFebruary26 2020.
Use September17 17:41UTC and November4 00:55UTC minute-end receipt clocks; the
later document date/accounting credit never backdates knowledge/payment. Issuer
15% withholding, any supported tax credit and lender compensation stay separate.
Both accepted stores/old fits remain sealed; a new full dependency acceptance is
now required before economic-data refits. This is new evidence, not a reason to
repeat the completed StageB baseline assembly or other source censuses.

Canonical term-loader/clock qualification checks50 actual control cells, two
future-deletion prefixes, two delayed clocks, four minute-boundaries and signed
Decimal entitlements. The initial minute-start metadata is preserved with its
explicit minute-end resolution; actual event-date arrays remain exact.

The bounded target producer matches1129040 control cells on22dates/all933 names.
Forty-two existing gross horizon outcomes change; independent Decimal arithmetic
and1159832 future-endpoint prefix cells pass, no support gains/losses. Accepted
sigma remains fixed for attribution: physical ranks2326, residuals3402,
shareholder ranks2289 and price ranks2251 change. Final virtual primary targets
still require the corrected actual risk coordinates; physical arrays are not
silently substituted. Two target harness failures (index inventory, 2D/3D selector)
are retained; no raw recovery was repeated.

Only64 original Natura quotes September17–December17 are read for the necessary
wealth tail (overlapping initial quote rows). The shared pre-effect Float32 seed
and unchanged recurrence reproduce256 controls exactly. Four wealth values change
on63days,252cells; masks/raw quotes unchanged. Event Decimal arithmetic and two
future-term prefixes pass. No successor mark/history or executable fill invented.
These action/wealth/target outputs are intermediates; daily/native/auxiliary/risk
propagation, same-ISIN bonus custody and the three conversion accounts remain.

Successful source/clock/target/wealth qualification times .2501/.07465/.76984/
.10418s exclude retrieval/rendering and failed attempts; not fit ETAs. Ruff passes.
Resolve held_event_source_audit, held_event_source_qualification,
natura_gross_action_amendments, natura_gross_target_attribution,
natura_wealth_amendment and held_event_source_acceptance. Full report
`docs/v2_HELD_EVENT_SOURCE_AUDIT.md`; acceptance SHA
742f1f8b9e6f77ef6f1077fdf686d15bf6fe871c5718c76b49e6b9540d0d6e46.

ENAT's R$35513.8717 held missing-quote exposure is next after the user-named three.
Preserve all V31/V32 mechanics and measured independent-adaptive minimum-fee
uncertainty. Adaptive disposal/timing/fraction/rounding/grouping/old-minimum/clearing
bounds, final corporate pricing, C matched replays/refits and conditional D remain.
No new neural forward or profitability is claimed.

Verified held_event_source_recovery for implementation
fbdf4d9dc43d037419640ddf4b2d5f46c6275adc restores684unique/697logicalmembers,
13aliases,17newPDFs and the actual two-term loader. Sparse reconstruction verifies
1129040target-attribution cells and256wealth-view cells. ZIP
held_event_sources_fbdf4d9.zip,17273050bytes,SHA
01dbc59678f7bd4b45c256da61c3a601a0bcacad9b588b1efaf227803f5a129d.
Recovery1.7444s; complete current research/patch and initial/qualified recipes
retained without immutable input, accepted store or old-fit duplication. Earlier
dependencies remain required. This is source/intermediate acceptance, not A/C
economic admission or final Natura dependency acceptance.


## 2026-09-20: complete Natura data dependency acceptance

The two source-proven2019 scalar corrections now reach a separate complete106-array
store through daily/peer/risk/physical/cross-market/actual neutral-target paths.
Resolve docs/v2_natura_data_inputs.json and economic_refit_inputs; prior economic
baseline and foundation stores/weights remain sealed. See
docs/v2_NATURA_DATA_PROPAGATION.md for every effect, attempt, qualification and limit.
All933names/full60 retained; no mask/age/eligibility loss. Natura has no admitted2019
M1 stream, so no raw M1 recovery or scalar/toclose change was performed.

Daily206samples/2241413076packedcells and final33combinedsamples/383630940cells
pass;506333457changed-array/scope cells and85untouched array hashes agree.
52968slowvalues,5076clusters,60sigma,191magnitude and1114cross-market values
change; actual neutral targets14039numeric/no support change. The earlier42
gross endpoint changes remain separately sourced, not doubled. Old-parent transfer
is rejected before weights. InitialNaNreport serialization,18unchanged inactive
scratch-sigma differences andFloat64-vs-Float32 product comparison are retained and
explicitly qualified. No censuses, original retrieval or prior baseline assembly
were repeated. New recovery is verified below; A final economics and
C/D remain incomplete. Next: causal opening claims and distinct custody/loan/tax
contracts for NATU/ALSC/SOMA, ENAT exposure-ranked original sources, then remaining
adaptive disposal/rounding/grouping/cost and clearing sensitivities.

Verified `natura_store_recovery` for implementation
601d356d8e35cd9ed17b7a446d7454c9867be75d restores676unique/724logicalmembers,
48deduplicated aliases. All21changed arrays reconstruct506333457cells to exact
complete-store byte hashes;85unchanged arrays,46tables and2indices retain explicit
previously verified parent dependencies. ZIP `natura_store_601d356.zip`,53523402bytes,
SHA f8d188caacb50177321c4ec3229325d404d64912f35c2a8967be95415f3bd92f.
Recovery7.5620s; complete current research, implementation patch, failed/qualified
executed recipes, bounded raw reducer outputs, sparse amendments, contract and
acceptance recover without immutable raw, accepted-array or old-fit duplication.
The economic-store and held-event-source recovery receipts remain required.
Natura acceptance SHA216f14221cac61d3441a1a8fed358e549f54131c5dd10f94d81a0c354d821220;
complete store manifest93f65b11f31979c6eafee0ca7e68afe5bb591a303eb8b5e2148072828e8d04bf.
No scoring, GPU fit, held-out observation read or model profitability is claimed.

## 2026-09-20: explicit opening claims and30bounded adaptive books

V33 adds an opt-in single-leg predecessor-value continuity hypothesis in both
accounts; first observed successor quotes replace only realized values, never
earlier intentions. Loan references stay unchanged. Unquoted deliveries retain
separate entry restrictions and independent SAM/TBPTT copies. Unknown ALSC custody
remains unknown/locked. No accepted data, source observations or original fit changes.

Resolve opening_claim_audit/identity/terms/acceptance and
docs/v2_OPENING_CLAIM_VALUATION.md. Thirty30-session/all933-name adaptive engineering
books cover focus-long/short NATU, ALSC and three SOMA fraction/precision cases at
R$10m/R$1m/R$5m. Identical-intention NAV maxdifferenceR$1.1175870895385742e-8;
independent900-day component identity maxR$3.725290298461914e-9. Independentadaptive
path differences remain measured, max .0001117427/.1008967008/.0000675182bp at
R$10m/R$1m/R$5m; oldV32uncertainty is also retained. These are synthetic preference/
fixed-risk actual allocator books with the oldbundled4bp bridge, not model alpha or
final corporate pricing. SOMA short continuous-loan versus fraction provision
final-.0066375240/-.0490403284/-.0141229283bp, original quantities/principal retained
under explicit alternative hypotheses. Auction precision path effects below.000001bp.

All source post-effect fills absent, custody respected, scenario prefixes exact.
Initial missing save_book scenario metadata repeated only the first new30-session
book to recover absent detailed readouts; its original arrays remain. All30qualified
books then completed before a Float32 oracle JSON failure. Saved-book resumption
reused them all and computed three scalar divisions only. Executed opening_claim_value
is a mislabeled effect-close value; canonical label corrected, opening oracle separate.
17affected distribution tests plus two additional affected loader/causality cases
pass; five overlapping focused reruns are not added. Ruff passes. Aggregate case
29.6064s/qualification.2136s/independentidentity.0542s; total initial wall unknown.

Canonical corporate_replay stays unchanged; opening_claim_terms is explicit
engineering scope. Next: same-ISIN Natura bonus custody/JCP withholding/payment,
remaining loan/fraction/custody admission and ENAT originals, then remaining
adaptive disposal/cost/clearing bounds. StageA/C/D remain incomplete; both accepted
StageB stores stay sealed.

Verified opening_claim_recovery for implementation
e10fc66189cb42c07daceb4200be1fb80f551b1a:757unique/854logicalmembers,
97deduplicated aliases and30completebooks/3378600accountarraycells restore/hashcheck.
ZIP opening_claims_e10fc66.zip,7053749bytes,SHA
fdfb7183bfbbe6701b88855f29b2097e599a22a91a79c1509c77c75d8511e608.
Recovery1.0097s; complete research/patch, failed/qualified executed recipes,
source-bound hypotheses and acceptance recover without immutable inputs/stores or
old-fit copies. Prior source/lifecycle/economic-store recoveries remain dependencies.
Acceptance SHAe0bc46d7acb25723b7e1e0d6365c6811492cbe3238f212123ee46c6149b908ba.


## 2026-09-20: Natura bonus custody and JCP account engineering

V34 supplements gross scalar actions with explicit settlement terms in both actual
accounts. September18 q2 reserves only the incremental signed bonus to September20;
original inventory remains reducible. Owned purchases retain separate base/bonus
custody clocks. Original loan principal/rate/minimum stay; whole-return deferral to
credit is explicitly hypothetical. Pending/new covered-loan proceeds stay restricted
until return while spot purchase money retains its original settlement date.
November7 gross JCP remains .12784527353; long15%withholding is separate from gross
price/labels, February26 payment and gross-versus85%net lender compensation. No
ordinary-CNPJ exemption, tax-credit asset, observed quote or loan alias is invented.

Resolve natura_settlement_audit/qualification/runtime_qualification/terms/acceptance,
natura_bonus_disposal_audit and docs/v2_NATURA_ACCOUNT_SETTLEMENT.md. Thirty116-session
all933-name books plus twelve14-session reversal probes use actual adaptive allocation,
frozen synthetic preferences/fixed risks, shallow-copied old PolicyData, correctedCDI
and qualifiedloans. Oldbundled4bp bridge has no added B3spot; final corporate pricing
and model profitability remain unestablished. Identical-intention maxNAVerror
R3.91155481338501e-8; independent3648daily component identities maxR3.725290298461914e-9.
Savedaccountcells13701888. Independentadaptive maxpath.019685697574168444/.19440065797418357/
.0392659758310765bp atR10m/R1m/R5m; targetdistance7.048151527220409e-7. Preserve earlier
V32/V33uncertainty. Decimal entitlements and actual bonus-plus-fill conservation pass;
actual reversal probes retain the signed bonus until credit. Separate source/withholding/
custody/lender contrasts and allcase scopes are bound; no daily-alpha interpretation.

14distinct new settlement cases and3affected existing cases passed in focused batches;
overlap not summed. A bitwise prefix fixture differed by2.22e-16 classification roundoff
and uses narrow1e-15arithmetic tolerance. A new called-bonus fixture corrected the
independent zero-target admission guard; runtime qualification proves all42historical
books unaffected and reuses them. No historical book failure/rerun. Ruff passes.
Audits104.6855s and5.8583s, savedqualification.4278s/runtime.0525s; not GPUfit ETAs.
Both accepted stores/oldfits remain immutable, no neural scoring/fit or heldoutconsumer.

Next: still-missing ALSC credit/auction and event-specific loan/fraction evidence,
ENAT original receipts by held exposure, BRML/DMMO/Copel adaptive pre-custody disposal,
remaining allocation/rounding/grouping/oldminimum/clearing and separatedcorporatecosts.
Do not repeat the Natura source/data/opening/settlement books. FinalA/C/D remain pending.

Verified Natura settlement recovery for implementation8262c0b006782076a2a09adee845c81d4fd84fcc:
851unique/1227logicalmembers,376aliases restored/hashchecked;
42completebooks/13701888accountcells. Archive
natura_settlement_8262c0b.zip,23675270bytes,SHA
564e41252e0220178be927b6e0f52de71c92fcc9868e28b8424c375306965db1. Recovery1.9166s.
Acceptance SHAeb6a96325d950ff449ebd72f8ce68b9629c9162ae74e734b366af7fdc9e39afc. Complete currentresearch/patch,
executed historical recipes, source hypotheses, tests/qualification and newbooks
recover without immutable sources/stores/oldfit duplication. Prior opening/source/
loan/economic/Natura-store recoveries remain required. All workers finished.


## 2026-09-20 — ENAT undated fraction engineering / ALSC auction source

V35's unknown auction uses null date/price/payment: whole shares deliver, residuals stay signed/locked, no cash invented. Original ENAT finalratio.805012676, July31legalclose/August1effect/August5credit; continuousloan primary and per-contract provision separate hypotheses.18x30/all933 actual adaptive syntheticbooks complete, identical-intention maxR9.313225746154785e-9; independent540dailyNAVs maxR3.725290298461914e-9,2028240accountcells. Decimalcohort/entitlement, no-source-fill and exactprefix checks pass. Independentadaptive maxpath~.000003005bp/target6.275163269217621e-9; preserve larger priorV32–V34 uncertainties.7newcases,7affectedloan-fraction and16targetcases pass, focusedrepeats overlap; Ruffpasses. Engineering16.8134818s/qualification.20472s, not fitETAs. No earlierbooks repeated.

New5originalPDFs/8visualpages,153ENAT2024+41issuer223572020 savedRADrows. ALSC original732740 reports Jan15 2020 auction/2076shares/R54.26688776859proceeds, distributednetunspecifiedfees within7businessdays ofJan30notice. KnownJan31/deriveddeadlineFeb10, noobservedpayment/physicalcredit. Earlier2019receipt search could not reveal this laternotice; oldwholeclaim stayslocked. One new2020 indexselection localvariable shadow failed beforeoutputs, preserved/fixed; allretrievals/books saved. Both acceptedstores/oldfits immutable; noforward/scoring/GPU. StageA finaladmission andC/D remainincomplete. See v2_ENAT_FRACTION_SETTLEMENT.md and enat_settlement/remaining_held_source pointers.

Verified ENAT recovery implementation1a68f36c6c9eff84b3209ba48dd2cfaeb43c8b31: 706unique/899logicalmembers,193aliases restored/hashchecked;18books/2028240accountcells and5newPDFs match originalreceipt hashes. Archiveenat_settlement_1a68f36.zip,7553887bytes,SHA1edd704ea68353e5a8d6d48257b21ddeceb4a488626ec8a87abfc2a5578c8a28; recovery1.029293s. Current research/patch/failed-qualifiedrecipes/newsource andbooks recovered without immutableinput/store/oldfitcopies. AcceptanceSHAf2bd9d273867d094de7d38b3c12334e3e893d8f49db93e47dff64d80064ad913; priorrecoveriesremainrequired. Allworkersfinished; StageA/C/Dincomplete.


## 2026-09-20 — Prearranged owned disposal before custody

V36 adds explicit economic sale permission while preserving incoming custody, physical loan return and proceeds release.54x14/all933 adaptive synthetic books, custody/effect/next for BRML/DMMO/Copel and both sides/R10m/R1m/R5m, completed without book retry. Independent756NAVs maxR3.725290298461914e-09; 2839536cells; Decimalshareflow max5e-12; identical-intention maxR1.1175870895385742e-08. Short permission variants exact; adaptive implementations retain measured uncertainty. R10m long effect-vs-custody finalBRML-.000144115bp/DMMO-5.846263298bp/Copel-.330398353bp, total synthetic path not alpha.7new+17existing+3bonus tests pass/Ruff. Initial wrong Dommo ISIN suffix stopped beforebooks, retained/resolved from canonical terms. CPUaudit27.262729s/qualification0.448965s. No source retrieval/store alteration/neural scoring/fit; canonicalcorporate_replay remains unchanged. See v2_PRECREDIT_DISPOSAL.md and acceptance/qualification pointers; remaining costs/eventbounds/A/C/D incomplete.

Verified V36 recovery: implementation b3c765dc40101df440cae7f739ee7102da8366cc,898unique/1428logicalmembers,530aliases restored/hashchecked;54books/2839536accountcells. Archiveprecredit_disposal_b3c765d.zip,6616536bytes,SHA37ebe48fbb5877487f678e9d4705fffde317563f2c563a3666d8b742ffe4fc2b;1.591935s. AcceptanceSHAbf4d4ecf980a16c5dfbf3521b77bc2c70f77a2795c4ff9fda971fac8203917a6. Complete currentresearch/patch/failed-qualifiedrecipes/terms/newbooks recover; no immutableinputs/stores/oldfits copied. All workers finished; finalA/C/Dremainopen.


## 2026-09-20 — Copel original principal and net-loan timing

V37 separates net-borrowed conversion from sourced positive custody, preserving original loan rates/principal/fees and actual owned receipts.30x30/all933 adaptive synthetic books, K0/.2/1 and December27/28/29 one-factor dates, both signs/R10m/R1m/R5m, completed once.900NAV maxR3.725290298461914e-09; 3380400cells;660Decimal source/day charges includingzero controls. R10m shortK0/K1 final-.000435061/+.001740257bp; early/late final-3.999501232/-3.949972862bp(max7.449071841/4.839939996), total synthetic path not alpha. Long arrays exact acrossvariants; independentadaptive max.000006878bp retains earlier uncertainty.11new+7affected tests/Ruff pass. Extra fixture sizing failure fixed only test after saving exact failedbytes; allbooks reused. Audit27.535423s/qualification0.495864s. No retrieval/store/oldfit changes. Flat/positive source pendingloan timing, othercost/payment/rounding/clearing/source/data admission and A/C/D remain open.


Verified recovery: implementatione423e462a185d240a026c38bfdecb07b46f30c63; archive copel_loan_bounds_e423e46.zip, SHA24739f9b1089618a017104cbe41f26a8badc489555cc2cc61977bc9a3416fd2a, 5598321bytes. 724unique/1024logical restored/hashchecked members, 300aliases;30completebooks/3380400accountcells. Recovery1.142679s. Current research/patch/frozen plan/five term variants/executed recipes/failed-qualified tests/newbooks recover without immutable inputs/stores/oldfit duplication. Prior precredit/ENAT/Natura/lifecycle/economic-store dependencies remain required. AcceptanceSHA049f9b06a502b38de1e9a1ebe68d9c16f6db827fc27ce9c090d3964fdc8fb263; final A/C/D remain open.


## 2026-09-20 — Payment sweep and precision qualification

The [payment checkpoint](docs/v2_PAYMENT_BOUNDS.md) qualifies V38 same-day newly-known fraction proceeds settlement in both accounts and54 frozen full933 adaptive books. BRML/Dommo sweep/precision/tiny-loan variants have held exposure; Cielo cent variants have no held loans and remain unqualified for that exposure. Independent2568NAVs, prior-close funding and Decimal cash/cohort checks pass. All earlier books/terms and accepted stores remain sealed; no model profitability. Invoice/security-day/old-minimum costs, actual-exposure Cielo and final A/C/D remain open.

54books/2568dailyNAVs/9645408accountcells; independentNAVmaxR3.725290298461914e-09, fundingR7e-09, sixnewtests/Ruff. Initial same-day release queue defect fixed before books;46saved books reused after Cielo exposure assertion, only remaining8 ran. Runtime target assertion narrowed to no negative targets; later positive unfilled intention retained. No oldbook/source/store rerun. Adaptive discrepancy~.00002bp exceeds the precision contrasts; earlier uncertainty retained. Sumcase89.283086s/resumption8.831308s/qualification.720953s, initialwallunknown. Canonicalcorporate replay unchanged.


Verified recovery: implementatione0a5af96502852e0d4f4169ccbbb04e6f37483bd; payment_bounds_e0a5af9.zip, SHAe7e52af7fe999f6f583aa49f2089453690d3d86123a8d4278e5fd5d7a3d924b5, 26482819bytes. 883unique/1248logical restored/hashchecked members, 365aliases;54completebooks/9645408accountcells. Recovery2.131333s. Current research/patch/frozen plan/nine term variants/initial-failed-qualified-resumed recipes/tests/newbooks recover without immutable sources/stores/old-fit copies. Prior Copel/precredit/ENAT/Natura/lifecycle/lending/economic-store dependencies remain required. AcceptanceSHAaea3d8b96cea1cc0d54d4c9ccc8110dec20bcf14f41bb10be0877fb351547e33; actual-exposure Cielo and final A/C/D remain open.


## 2026-09-20 — Loan invoice and minimum allocation checkpoint

The [loan invoice checkpoint](docs/v2_LOAN_INVOICES.md) qualifies V39 cent/payment grouping and partial old-minimum credit hypotheses in BOTH accounts,36full933 adaptive books/2304NAVs. Independent9503invoice groups and minimum-credit arithmetic pass;14newtests, prior books/stores/oldfits preserved.72 Decimal operand-recomposition cent boundaries are enumerated; actual Float64 invoice coordinates round exactly. Larger fixed-fee adaptive uncertainty remains; no model profit. Actual-held-Cielo, final historical corporate costs/source-data admission and C/D remain open.

Independentadaptive maxpath .010282502409070731/.00000816549058072269/.000022326227277517318bp atR10m/R1m/R5m. Preserve larger V32-V38 fixed-fee uncertainty. Many2019R10m path signs are below this bound. Total synthetic paths, not dailyalpha/modelprofit or an all-interior adaptive extremum proof.

14distinctnewtests,9existingloancontracttests,6existingfractionpaymenttests pass; overlapping batches not summed. Ruff passes. Engineering74.986028s/qualification1.087650s; all36 first-pass, no oldbook/source/store rerun.


Recovery verified: `loan_invoice_5e1c5b0.zip`, SHA `07dcd80eb46025c661fa790a342b885543aa7a87f3b6c09991dc01b201a51d1c`, 15741455 bytes. 820 unique/1006 logical restored/hash-checked members, 186 aliases; 36 complete books/8660736 account cells, 1.655622s. Complete current research, implementation patch, frozen plan, failed/qualified executed recipes and new books recover without immutable-input/store/old-fit duplication. Prior payment/Copel/precredit/ENAT/Natura/lifecycle/lending/economic/Natura-store recoveries remain required. All workers finished; final A/C/D remain open.


## 2026-09-20 — Dated spot component checkpoint

The [dated spot checkpoint](docs/v2_HISTORICAL_SPOT_COSTS.md) qualifies V40 ordinary-CNPJ 2021-2024 spot component separation in BOTH actual accounts and30full933 adaptive books/960NAVs. Primary3bp B3+1bp provisional shortfall+zero brokerage exactly reproduces the old4bp bridge; bounded auction/brokerage/shortfall/proceeds contrasts are qualified. No adaptive debit exposure; funded debit is separately fixture-verified. Three dated B3 originals distinguish activated monthly custody from proposed daily custody. Final custody/earlier tariffs/spot invoices/daytrade/source admission/Cielo held-loan bounds and C/D remain open; preserve all earlier checkpoints and adaptive uncertainty.

All30books first-pass, seven new and32existing tests pass. 960NAVs/3613440cells, independent9888fill Decimal charges, funding and prefix checks pass. Engineering30.9497025s/qualification.5765957s; no source/store/oldbook repeat or GPU. Recovery follows below.


Recovery verified: `historical_spot_2c79cec.zip`, SHA `9d222b264c8d498d1070d7ac8a8131bf52619c79552b766184f07d1d50a6cee5`, 9444339 bytes; 786 unique/1009 logical restored/hash-checked members, 223 aliases,30books/3613440cells and3original PDFs; 1.3285189s. Complete current research, implementation patch, frozen plan, executed recipes/tests and sources recover without immutable input/store/old-fit duplication. Prior recovery dependencies remain required. Final A/C/D still open.


2026-09-20 V41 ordinary monthly custody bounded acceptance. The [ordinary custody checkpoint](docs/v2_CUSTODY_FEES.md) qualifies explicit monthly physical-stock fees in both actual accounts for2023/2024, with sourced progressive brackets/exemptions and separate client-payment/base hypotheses. Thirty48-session/all933 synthetic adaptive books and independent1440NAV/21870fill-flow/48monthly-assessment checks pass; no model score/fit. Pending owned sales/new loans/covers and unpaid terminal invoices remain explicit. Corporate physical transitions still require separate custody admission; earlier tariffs and finalA/C/D remain open. Both accepted stores and old fits stay sealed. Do not repeat these or earlier completed engineering/source/store audits. The recovery receipt follows.


Recovery verified: `custody_fees_0194723.zip`, SHA `de066862feefe220722ee63359eaff46a1fd365d1658729e1329a189c5ec593d`, 19741243 bytes; 851 unique/1028 logical restored/hash-checked members, 177 aliases, 30 books/6770880 account cells and 4 original PDFs; 3.1209187s. Complete current research, implementation patch, frozen plan, failed/qualified executed recipes/tests and new sources recover without immutable input/store/old-fit duplication. Prior dated-spot/loan/account/source/store recoveries remain required. Final A/C/D remain open.


2026-09-20 closeout trajectory: [Stage A closeout](v2_STAGE_A_CLOSEOUT.md) consolidates historical costs, corporate/calendar/source composition and integrated admission. Reuse all V31-V41 proofs; new runs require an untested interaction or actual exposure. A read-only inspection of all28 Aug30-covering books from the existing140-book foundation index checks30000 Cielo focus cells/761fills: no negative economic inventory August26-30, but no saved contractual loan cohorts, so no held-loan cent bound. Producer first-pass .9129214s; no replay/scoring/source/store/fit mutation. C/D remain unstarted.


Closeout/exposure recovery verified: `cielo_closeout_b2a973f.zip`, SHA `e905860d185c97cb1a02b90f2143810d3c0d17a5f336ba9a55ff276fc3d5815a`, 244886 bytes, 18 restored/hash-checked members, 30000 focus cells from 28 saved books; 0.0997858s. Current closeout plan, producer, source bindings, selected focus evidence and implementation patch recover with V41 and original-book dependencies; no immutable source/store/old-fit duplication.

## 2026-09-20 — Historical cost coverage and corporate custody integration

The [combined cost report](v2_HISTORICAL_COST_COVERAGE.md) and `historical_cost_acceptance` close the earlier tariff and exposed corporate custody implementation gaps. All 2,099 development dates have sourced dated regimes: 50 original monthly archives supply 1,014 dates, 976 use the fixed regime, and 109 retain source-supported missing/not-yet-known bounds. Nine tariff/technical PDFs and 33 visual pages are qualified. An additional dated clearing-migration original/page resolves November20 2017 by source composition; two older dates remain, with no calendar array changes.

Both accounts retain pending spot/new-loan flows, delayed bonuses, successor deliveries, cash closeout and locked unknown claims. Positive rights, physical base and maintenance have separate readouts; fee inclusion/valuation/client payment remain explicit hypotheses. Eighteen historical-cost and 34 corporate books complete first-pass; 14 unexposed rights variants skip. Independent 2,635 NAVs, 12,396,181 saved account cells, 50,253 fills and 141 assessments qualify, with separately documented oracle scopes. Seventeen new corporate and four new historical tests plus 72 affected regression cases pass; overlapping batches are not additive. Earlier 18 books remain valid under 498 saved-state current-code checks, without replay.

Corporate adaptive differences reach 0.0387828161/0.2010558935/0.0209277910bp at R$10m/R$1m/R$5m; preserve larger earlier uncertainty. Several fee contrasts are below this uncertainty. These are synthetic path results, not model profit. All 34 new paths have no Cielo loan cash redemption, so the held-loan cent bound remains conditional. All failed fixtures/retrievals/qualifier attempts and corrected report prose are retained with accurate provenance. Historical/corporate numerical batches took 20.9745/72.9160 seconds; no GPU estimate follows.

The three-block closeout remains active: spot invoice/daytrade disposition, two calendar gaps, event/source and separately attributed data composition, then integrated Stage A admission. Stage A is not accepted; C/D unstarted. No earlier completed source/store/book matrix was repeated. Recovery receipt follows after archive verification.

Recovery verified for implementation `3c8b5b3a8c23c73842f7d758316dd603ce453bc7`: `historical_cost_coverage_3c8b5b3.zip`, SHA `6bdc0e55c4b311d453edcf3bc9fc2f3e62d132e5b0d9c5314d9488a32941f7c2`, 44,185,397 bytes. All 1,447 unique / 1,916 logical members (469 aliases), 52 books / 12,396,181 account cells, 10 original PDFs and 50 monthly archives restore and hash-check. Recovery 5.474825 seconds. Current research, implementation patch, frozen plans, initial/qualified recipes, tests, sources and new books recover without immutable input/store/old-fit duplication; bound prior dependencies remain required. Acceptance SHA `a8f21494aad6a070106f9b39e876349f3a334c5f6247d0abca0bef8ffd6f3c5a`. All workers finished. Stage A final admission and C/D remain open.

## 2026-09-20 — Historical-cost delivery block closed

[Spot invoice closeout](v2_SPOT_INVOICE_CLOSEOUT.md) completes applicable precision treatment in both accounts while retaining unrounded fractional costs as primary. Six new invoice variants qualify against six saved controls without replay:366NAVs/1,722,762cells/4,287fills, independently reconstructed Decimal invoices and T+3/T+2 cash queues. Five distinct new tests plus53 affected existing cases qualify; fixture-only failures and exact attempts are retained, with no production fix/book rerun. Audit10.61334s/summedcases9.54670s/savedqualification.12088s are CPU runtimes. All prior larger adaptive uncertainties remain; no model-profit claim.

The existing50monthly-source/52cost-corporate books and V31–V41 remain dependencies. No opposing security/day fills or Cielo cash-loan redemptions occur in these six paths; related admission stays exposure-conditional. New B3 original131/2015 (five visually checked scanned pages), a March2011 mirror (cover) and an explicitly unapproved February2017 draft (cover/page134) do not resolve the two old equity-clearing dates. Selected403/404 failures and source receipts are preserved; no calendar/store changes. Historical costs are one completed delivery block, not acceptance of StageA or a completion percentage. Calendar/event/source-data composition and integrated admission remain; C/D unstarted. Recovery follows.

Recovery verified for implementation `505a7e83fc915c730e453b2973a44b577674d43e`: `cost_closeout_505a7e8.zip`, SHA `b67b7d15437f68d97b8e79fb5802fa2d20cd219b12b05a9dd4f3c55afab1d085`, 12,728,424 bytes. All658 unique/841 logical members (183 aliases), six books/1,722,762 account cells and three calendar PDF leads restore and hash-check in1.5254132s. The PDFs are one dated original, one2011 mirror and one explicitly unapproved2017 draft; no calendar bound is claimed. Current research, patch, frozen/executed recipes, failed/qualified tests and new evidence recover with bound prior dependencies and no immutable source/store/old-fit duplication. Acceptance SHA `02deb94c7fc70d69d69ba26ff4d2b455e97a0865e40ce8715756c320015c8868`. Historical-cost block complete for its explicit contract; finalA/C/D open.

## 2026-09-20 — Event/source composition within the existing closeout block

[Composition evidence](v2_EVENT_SOURCE_COMPOSITION.md) binds the seven-event/Natura/Cielo primary with identical numeric loader terms. The new sparse data attribution checks1,693,560 accepted control cells,84Decimal endpoints and33full933/full60 consumers/4,310,460target cells:72valid gains/no losses,10,205actual neutral-target changes. All24post-effect eligibility cells stay;2,806already-unquoted source entries close. First-pass2.27907s/10.36491s; no book, neural inference, fit or accepted-store overwrite.

Following the specific saved Elektro erratum lead in existing RAD receipts recovered original528672, one visually qualified page. It explicitly moves the dividend date because December30 2016 has no BM&FBOVESPA trading/settlement. Combined with the dated calendar this supports22of23monetary-only date dispositions, no calendar/rent-array changes. January25 2017 remains. Initial wrong-viewer query/base64 assumption failed before PDF output; exact code/viewer retained, qualified public route succeeded. Source retrieval.55766s; no repeated calendar census.

The separate700-row identity scope confirms no SSBR->ALSO or ARZZ->AZZA link in the accepted store and zero successor eligibility in each first60sessions. This is an inventory, not a claim that all60 pass unchanged liquidity thresholds. Original surviving-issuer rename evidence and source assignments are bound for necessary bounded propagation; acquired-company index-history merging and new loan aliases remain inadmissible. Both accepted stores remain sealed. This delivery block and integratedA are incomplete; C/D unstarted. Recovery follows.
