# Stage A closeout contract

2026-09-20, after ordinary custody acceptance V41. This consolidates the remaining
work following the user's request for a clearer trajectory. It does not change the
registered economic endpoints, authorize held-out reads, accept Stage A, or start C/D.

Most account mechanics are implemented. Both complete derived stores are accepted
and immutable. The remaining deliverable is one historically supported, internally
consistent account configuration with explicit uncertainty. Completed engineering
checks are evidence to reuse, not a queue to replay.

| Block | Work remaining | Completion criterion |
| --- | --- | --- |
| Historical spot costs | Earlier dated regimes and the previous-month **global market** ADTV input; applicable spot invoice/daytrade treatment | Every requested replay date has a sourced regime and explicit rate determination. If a monthly input remains unrecovered after targeted source work, use only a source-supported tariff range as a labelled uncertainty contract; never account capital as market ADTV or an arbitrary flat rate. |
| Historical custody costs | Earlier dated thresholds/maintenance and corporate physical custody treatment | Both accounts can process every held name and event under the complete dated fee contract. Preserve V41 ordinary physical-flow proofs. Resolve exposed corporate transitions using existing custody/loan claims and explicit source facts or bounded hypotheses; no name deletion or silent zero fee to bypass the current guard. |
| Calendar | Three older equity-clearing ambiguities: 2016-12-30, 2017-01-25, 2017-11-20 | A specific sourced disposition, or separately attributed date hypotheses where the evidence remains genuinely insufficient. Do not add all monetary-only dates to equity settlement/accrual. |
| Event/source composition | Compose the qualified primary event terms; retain ALSC unknown physical credit and net cash, ENAT unknown fraction auction, and explicit loan conventions | One bound primary term table and separately named one-factor variants, with no contradictory event clocks, invented receipts, new loan/issuer aliases or overwritten stores. Unknown claims stay locked where required. |
| Cielo held-loan precision | Identify actual loan exposure at the August30 closeout, then apply the already frozen per-share cent variants if that branch is exposed | A held-loan bound requires actual loan cohorts under adaptive decisions. No preference/rate/availability changes to force a holding. An unexposed path may establish zero impact **for that path**, never a general held-loan precision bound. Carry that distinction into the later registered model books. |
| Remaining succession data implications | Determine whether newly sourced succession changes the accepted model-data contract | Record each actual eligibility/history/wealth/target implication separately. Produce an amendment only for evidenced changes and retain unchanged stores. Account settlement hypotheses do not authorize unrestricted issuer histories or loan aliases. |
| Integrated admission | Combine the above with existing cash, loan, execution and event mechanics | Verify the remaining interactions and full date/exposure coverage, reconcile components in both accounts, and retain all measured adaptive uncertainty. Publish one acceptance with explicit residual limitations. No model-profit claim from synthetic preferences. |

The rows form three delivery blocks: historical costs; corporate/calendar/source
composition; integrated admission. They are not seven new experiment campaigns.

## Work discipline through admission

- Reuse V31–V41 books, source receipts, Decimal oracles, accepted stores and recovery
  dependencies. Do not repeat a completed test matrix to create a newer checkpoint.
- Add an engineering run only for a specific untested interaction, a changed
  implementation branch, or actual newly identified exposure. Freeze its contrast
  before outcomes and preserve the full population and original calibration.
- Reuse existing source indices before retrieval. Do not restart source censuses
  to resolve unknown historical publication/revision completeness.
- Negotiated client payment, proceeds remuneration, debit spread, brokerage and
  unresolved lender instructions remain explicit hypotheses. Exact client invoices
  and obtained broker quotes are not prerequisites invented by this closeout plan.
  Historical exchange facts still require their dated sources or honest bounds.
- Preserve the existing primary and one-factor sensitivities. Do not introduce
  additional capital sizes, parameter grids, seeds or learning-budget changes.
- C starts only after the integrated Stage A acceptance. It separates old-forecast
  accounting/source replay from the required new-data P/F fits. D remains conditional
  on the registered C results; neither stage has started.

## Latest exposure evidence

The existing foundation settlement index contains 140 books. Selecting every book
whose saved state calendar contains August30 2024 yields 28 ensemble/reference
books. Reading their saved Cielo column and fills checks 30,000 focus array cells
and 761 Cielo fill records, without replay or neural inference. None has negative
economic Cielo inventory on August26–30. Their earlier synthetic terminal records
also all have sell-side Cielo dispositions.

These old accounts did not save contractual loan cohorts or pending loan returns.
Their nonnegative inventory therefore supplies no held-redemption loan candidate
and does not prove the absence of pending old loans. This bounded index is not an
exhaustive search of all seed books or future corrected paths. The separate V38
unexposed controls and V40 V32-book inspection remain unchanged. No Cielo cent
bound has been qualified by this read-only inspection.

The producer `ops/inspect_foundation_cielo.py` binds the old book/account/fill hashes,
retains the selected Cielo arrays and actual fills, and records the limitations in
`cielo_foundation_exposure`. It completed its first invocation in 0.9129214 seconds
on CPU. No source, store or old-fit bytes changed.

## Time and progress reporting

There is no defensible completion percentage or GPU completion estimate yet.
Recent engineering batches generally took under two minutes; historical source
interpretation, integration and verification have dominated elapsed work. The next
meaningful progress report should identify which of the three remaining delivery
blocks closed, which evidence changed an admission decision, and what still blocks
the registered model comparisons. Small fee sensitivities are not separate stages.
