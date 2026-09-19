# Economic accounting, data audit and scaling progress

2026-09-19: user authorized accounting repairs, a deep source-to-model audit,
additional historical B3 retrieval and economic-primary attention/GRU capacity
follow-ups. Read `research/preregistrations/v2_economic_data_scaling.md` after each
stage. Foundation outcomes remain sealed at the canonical foundation pointer.

**Current status:** A/B remain in progress; C/D have not started. Corporate-account
research and the source censuses below are complete. BOVA11 borrowing-path
consistency, removal of fabricated missing-quote liquidations, inventory netting and
multi-leg/delayed-delivery mechanics are implemented and tested (see the latest
entries); recovered historical sources are not yet
admitted. No training or source-census worker is active.

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
