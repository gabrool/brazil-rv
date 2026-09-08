# Brazil-RV v2 refactor review handoff

This note is for the models that proposed the multi-day refactor and its later
fix passes. It summarizes the design choices actually implemented, the places
where competing suggestions were resolved, and the work intentionally left
undone. Exact results, roots, hashes, and the full chronology remain in
`V2_EXPERIMENT_LOG.md`; this note is an engineering explanation, not a new
research registration.

## Starting point and governing choice

V2 is an incompatible multi-day research model, not an extension of the v1
intraday deployment. The v1 store, checkpoint, official-validation record,
and spent test result remain immutable historical evidence. V2 received its
own experiment log, schemas, targets, evaluator, ledger, and registrations.
No v2 result changed the deployed v1 model.

Where the suggested plans disagreed, the implementation preferred the design
that made the historical information set and accounting state explicit. A
missing contractual source was labelled unsupported instead of being inferred;
a score-bearing failure was frozen instead of retried; and an operational
failure could be repaired only before a result, at a new clean commit and in a
fresh root. This choice is why several apparently short routes were not taken.

## Core refactor that was implemented

- The data clock has one explicit 15:45 decision row. Input observations,
  masks, features, labels, orders, fills, and evaluation dates have distinct
  contracts. Entry-bar and future-label leakage remain forbidden.
- Security identity is permanent and dated. Same-ticker consecutive ISIN
  successions are audited and linked for history/survival purposes rather than
  treated as ticker identity or automatic delistings.
- The daily store is a typed, revision-bound artifact. Feature and target
  families are streamed into float32 memmaps; family-local arithmetic may use
  float64. Rank Gaussian normalization is row-wise. The accepted real store
  has 4,102 sessions and 933 linked identities. The final rebuild peaked at
  7.139 GiB and retained the 8-GiB measured-build invariant. The user's waiver
  bypassed only the separate 10-GiB admission check.
- Internal feature families retain the unconditional survivorship gap gate.
  External sidecars use hash-bound D+1 contemporaneity plus the registered
  name-clustered, ADV20-stratified composition audit. Sparse strata are
  reported rather than silently treated as evidence. Targets keep their
  separate gate.
- Foreign-OS artifact paths resolve through an environment-selected override,
  but identity remains the recorded SHA-256. The sealed manifest is not
  rewritten when an artifact moves.
- Dataset boundaries zero invalid cells under their masks and assert that every
  tensor supplied to the model is finite. Masked reductions use structural
  selection rather than `NaN * 0`. Empty lookbacks and empty fast patches have
  explicit finite behavior, and training asserts finite per-step loss.
- The old stateless economic approximation was replaced by a stateful
  multi-session ledger with intended orders before fills, pending entries and
  exits, signed shares, cash, claims, transaction costs, risk trims, action
  handling, marked inventory, and labelled terminal settlement. Training and
  economics share registered common populations without sharing future label
  increments across selection/evaluation blocks.
- Slow wealth intervals break only on missing endpoints, raw-series restarts,
  or unresolved actions. Resolved dividends/splits/conversions already carried
  by the wealth index are not masked a second time. Raw-price cross-session
  features retain the stricter action/unit guard; scale-free same-session
  features do not receive an unnecessary action mask.
- Temporary eligibility loss is stateful: held positions receive the
  registered five-session grace, while entries remain eligibility-bound and
  terminal/no-print settlement keeps precedence. This removed the D5 defect
  without pretending that missing eligibility means an instantaneous fill.
- The evaluation target was revised to a characteristic-neutral residual using
  causal volatility-decile, beta-quintile, and rank-Gaussian log-ADV controls.
  The original target remains a legacy diagnostic only. Neutral and legacy
  readouts use explicit, hash-audited common populations.

## Constructed-book and borrow decisions

The rev4 construction ranks long and short candidates inside causal
Yang-Zhang-volatility quintiles over the full eligible set, retains positions
against their current quintile, scales quotas for small strata, and spills
only into unused names on the same side. The competing global top/bottom-half
interpretation was rejected after it made balanced occupancy mathematically
impossible for signals correlated with volatility.

BOVA11 is the beta hedge. Fixed-width COTAHIST proves its registered identity
is `TIPREG=01`, `CODBDI=14`; the suggested BDI02 restriction was not
implemented because it would leave only 92 observations in 2019. The hedge is
not an equity name. In rev4c, equity gross, equity absolute net, name caps,
target bands, hard bounds, and D4 all apply to the equity book only. BOVA11 has
an independent absolute 0.60-NAV cap; capped targets, residual beta,
whole-book gross, and whole-book dollar net are reported.

The lending archive is a separate immutable D+1 source. `borrow_balance` is
headline; `borrow_strict` requires a recent observed lending trade;
`borrow_open` is an unlimited-supply bracket. Rates use the last observed
taker rate plus 25 bps/year, with causal same-day cross-sectional imputation
where registered. Before the first causal published rate on 2023-07-11,
rev4c uses a separately flagged 2% placeholder in all three cells rather than
backfilling a future observation or refusing every short. The archive itself
was not rewritten.

## Suggestions not fully implemented

- No heuristic corporate-action price-ratio adjustment was accepted. Both the
  legacy and proposed strict fallback failed the preregistered coverage and
  precision audit, so uncovered action intervals remain explicitly unresolved.
- No unverified historical exchange calendar, auction mark/capacity, action
  conversion term, or terminal consideration was invented. Current economic
  results remain labelled development-grade close-proxy/inferred-action
  evidence, not execution-grade claims.
- Options, fundamentals, rebalance, and other absent contractual sidecar fields
  were not synthesized. Consumers use only materialized families and verify
  that omitted families are recorded source-missing.
- The 127 official B3 Chapter 05 PDFs downloaded for 2024 H2 were preserved but
  not parsed into a new source merely to improve a scored result. The later
  direct lending archive solved the registered F3 coverage problem without
  mutating the sealed store.
- Earlier Round-1/Round-2 results were not resumed across incompatible target,
  clock, action, or ledger schemas. They remain historical engineering
  evidence. Rev3 Round 2 was frozen but stopped at its no-score compiler smoke;
  that failed root and its evidence were sealed, but no registered Rev3
  Round-2 score completed. Rev4/rev4b/rev4c/rev4d are new registrations rather
  than post-result edits to it.
- No official validation, permanently held-out test, deployment change, or
  production trading infrastructure was authorized or performed.
- Rev4c CPU Round 1 and all rev4c Round 2 GPU work were not run. The prerequisite
  16-book acceptance produced a binding scored stop, so continuing would have
  violated the supplied registration.

## Current disposition

Rev4d code and registration are at
`733ac1e729f98af964e022869a9dfaf617f93d3b`. All 899 tests passed before the
fresh acceptance score. Exact roots, hashes, and full tables are recorded in
`V2_EXPERIMENT_LOG.md`.

The rev4c failure was resolved with one parameter rather than a new sizing or
selection system. The prior four-times threshold scaled a quintile unless it
had 48 names, even though disjoint 12-name long/short bands require only 24.
Rev4d therefore uses `2 * (k_q + b_q)`. This choice preserves the proportional
small-stratum rule, matches the original global half-population contract, and
leaves all clocks, identity, target, borrow, hedge, cap, and accounting rules
unchanged. It was not chosen by tuning the gross floor: it was preregistered,
unit-tested at 38 and 20 names, and frozen before the new score.

The new acceptance passes every registered hard gate. All 16 books have
D1--D5 zero and equity mean gross between 1.897 and 2.005; quota is six in
every quintile and the `small_universe` shortfall is exactly zero. Hedge,
occupancy, null, post-hedge beta, chronology, hashes, and protected-access
checks pass. The result remains honestly labelled
`development_grade_inferred_actions` because the refactor did not invent
verified actions, auctions, or historical borrow execution.

The authorized CPU Round 1 then completed only the five controls and the
`b_intraday` GBDT on F1--F3. `b_intraday` is the sole eligible GBDT rung and
designated parent. Its pooled neutral IC is .019991
[.011608,.029269], while headline balance net excess is -5.604 bps/day
[-9.727,1.852]; this is not profitable-deployment evidence. The strongest
control neutral IC is momentum 12-1 at .021473, but controls are deliberately
ineligible for GBDT parent designation.

No suggestion was silently broadened in rev4d. In particular, the gross
bounds were not relaxed, the scored rev4c root was not overwritten or retried,
the unparsed B3 PDFs did not mutate the sealed lending source, and no protected
period or deployment was accessed. The user explicitly withheld Round 2, so
Stage P, A/B arms, R2.2, and GPU work were not started. There was no paid
instance to terminate; two provider reads returned zero nonterminal instances.
