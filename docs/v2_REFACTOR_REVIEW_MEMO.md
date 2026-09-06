# Brazil-RV multi-day refactor: implementation and design-disposition memo

This memo is intended for the authors/reviewers of
`v2_fix_pass_3_addendum.md`, `v2_fix_pass_3c.md`, and
`Brazil_RV_Multiday_Refactor_Plan.md`. It explains how their recommendations
were reconciled, what was implemented, what was intentionally changed or
deferred, and why the result stops before new research experiments.

The short version is:

- The full multi-day refactor plan was treated as the semantic authority when
  it conflicted with the narrower fix-pass notes. It is the only proposal that
  repairs the complete chain from observations through accounting rather than
  preserving known legacy assumptions.
- Compatible tactical requirements from the addendum and 3c were retained,
  especially the real-scale memory work, raw validation, as-of revisions,
  decision-prefix causality, mask-aware consumers, intended orders before
  fills, and common-population evaluation.
- The implementation path through contracts, source adapters, features,
  consumers, accounting, evaluation, and artifact migration was completed on
  a new incompatible v2 schema. Old stores/runs remain immutable and cannot be
  resumed under the new semantics.
- Real-data engineering acceptance was **not** declared. The available
  corporate-action evidence failed the preregistered audit thresholds, and the
  repository lacks an authoritative dated B3 session schedule and several
  other sources needed to validate headline economics. Those capabilities are
  reported as `unsupported`, not silently imputed.
- No revised development study, Section-D experiment, official-validation or
  held-out outcome read for a research decision, or deployment change was
  performed. The paid GH200 used by the superseded Round-2 run was terminated
  before this refactor; no replacement paid instance was launched and none is
  currently running.

## 1. How conflicts between the documents were resolved

The documents were applied in this order of authority:

1. Leakage, identity, unit, and economic-accounting correctness.
2. The complete end-to-end contract in `Brazil_RV_Multiday_Refactor_Plan.md`.
3. Compatible bounded fixes in the addendum and 3c.
4. Legacy behavior only where it remained semantically valid.

This ordering matters because 3c explicitly declined the larger plan's clock,
FeatureSpec, native M1, contractual-action, and order/fill changes. Those
exclusions would have left several of the reviewed root defects in place. The
user subsequently asked for both proposals to be reconciled and authorized us
to choose the sounder argument, so those exclusions were not treated as a
scope limit.

The implementation remains lean. The additional contracts are small and tied
to concrete corruption risks: a decision clock, an ordered FeatureSpec, a
canonical sample view, verified action terms, and a persistent signed-share
ledger. It does not add a general registry framework, generic OMS, live
trading system, broad optimizer, or a second parallel pipeline.

## 2. Principal design choices implemented

### 2.1 One canonical decision row and clock

The canonical model row `t` now means the information available at exactly
15:45:00 `America/Sao_Paulo` on session `t`:

- internally derived daily market fields end at `t-1`;
- completed intraday bars end by the decision, excluding the bar that starts
  at 15:45 and every later field;
- a publication may enter row `t` only if its timestamp, precision, declared
  latency, and revision state prove availability by the decision; and
- source lags are applied once while building the canonical snapshot.

Every neural stage, GBDT, baseline, score path, and evaluation path consumes
that same stored row. We deliberately did **not** implement 3c's later
`slow_row_index = sample_date_index - 1` shift. Under the new contract this
would shift a snapshot whose daily market information already ends at `t-1`
to row `t-1`, double-lag publications and market state, and give different
semantics to pretraining and fine-tuning. Regression tests exercise both the
missing-lag and double-lag cases.

The new schedule interface requires a versioned, dated B3 equity-session
schedule with exact 15:45 decisions and programmatically validates chronology,
uniqueness, timezone, continuous-session boundaries, and source identity.
The old “at least 50 observed names means this was a session” rule is no longer
calendar authority. It remains only a coverage diagnostic.

### 2.2 Raw observations, activity, identity, and masks

COTAHIST/M1 promotion now separates facts that the old pipeline conflated:

- source session complete;
- valid OHLC observation;
- an actual trade/security row;
- valid activity, including a verified zero on a complete source session; and
- missing or invalid source data.

Impossible/nonpositive OHLC, inconsistent bounds, negative activity,
unsupported units, malformed structural audits, and conflicting duplicates
cannot enter accepted arrays. Exact duplicate economic records collapse once
without summing their activity twice. Invalid records retain reason codes for
auditing. The universe and activity features consume the same explicit
20-session activity contract, so a missing source day cannot masquerade as a
no-trade day.

ISIN remains permanent security identity and ticker remains a dated
attribute. Same-ticker adjacent ISINs are only proposed as link candidates.
They affect history or economics only when a verified allowlist supplies the
conversion ratio, cash terms, effective date, first-known timestamp, and
evidence. The current allowlist is intentionally empty. Accepted conversions
rebase dimensional state and prevent predecessor/successor double counting;
there is no generic history copy across an unverified ticker succession.

### 2.3 Corporate actions and return families

The accepted action representation is contractual rather than inferred from
realized returns. It records, with evidence and coverage status:

- shares received per prior share;
- cash entitlement per prior share;
- ex/effective and payment sessions;
- predecessor/successor claim mapping; and
- the historical availability timestamp.

The same action primitive feeds shareholder-wealth features, targets, and the
ledger. Splits change units; distributions create receivables/payables; a
payment clears the claim once and creates no second return. Shorts receive the
opposite cash obligation. Unsupported complex claims and uncovered intervals
stay unresolved.

Three return meanings are no longer conflated:

1. price return for the same contractual claim, excluding distributions;
2. gross shareholder holding return, including contractual share/cash
   entitlements over `(t, t+H]`; and
3. the median-adjusted, volatility-scaled primary model target.

Known zero terminal wealth remains a valid -100% economic outcome and is
bottom-ranked. Unknown terminal wealth is not converted into a finite loss or
silently dropped. Synthetic market-median cash flows are absent from the
headline target and P&L path.

Price/quantity jumps, `DISMES`, and open gaps are retained only as audit
diagnostics. In particular, the 3c open-gap formula is not an accepted split
term and cannot rescale prices, shares, or cash. Same-session safe features
survive an unresolved cross-boundary action; only the wealth/unit-dependent
history is invalidated. This is more conservative than treating a realized
price ratio as a contract while avoiding the earlier bug where one unresolved
action erased unrelated OHLC, activity, and observed-history features.

The 20-session Yang-Zhang estimator was retained, but it now consumes coherent
shareholder-wealth OHLC and the scale available by the decision. The target
uses that same-row decision-time sigma without lagging it a second time.
Median subtraction occurs before name-specific scaling.

### 2.4 Typed feature semantics and revision-safe sidecars

The full plan's typed `FeatureSpec` was adopted instead of 3c's instruction to
retain blanket rank-Gauss. Every enabled model field binds its semantic name,
family, unit, availability/lag rule, formula/window, support, validity,
age/staleness policy, transform, version, and position. Store open now requires
the complete ordered FeatureSpec and its hash; a names-only schema can no
longer be sealed as current.

Transforms preserve type:

- continuous relative states use tie-aware per-date rank-Gauss where declared;
- binary flags remain 0/1;
- bounded fractions and signed descriptors preserve direction and boundaries;
- event ages retain raw session age/censor state plus a bounded model form;
- rebalance ramps retain signed size and timing; and
- absolute annual lending rates use explicit annual-decimal units and the
  declared bounded transform.

`log_adjusted_close` and irreversibly transformed legacy pseudo-raw fields are
not current model features. A minimal market-return/volatility/dispersion
panel is stored as an audit-only common-state diagnostic; it was not slipped
into the model because the plan reserves a common-context arm for a later
registered representation comparison.

Lending and odd-lot changes are publication-ordered as-of calculations: a
future revision can change a later decision snapshot but cannot rewrite an
earlier one. Event, odd-lot, and other daily-panel adapters no longer create
archive-sized Python lists of dictionaries. Timestamped events reset exact
session age, including overlapping filings. Options use supplied put/call
counts and `log((put_OI+1)/(call_OI+1))`; unsupported ATM-IV and clipped
pseudo-raw fields remain disabled. Fundamental ratios use supplied compatible
raw liabilities/assets rather than inversion of clipped legacy values. These
are fixture-level transforms given a conforming upstream source: the adapters
do not independently attest options OI unit/multiplier/expiry/adjustment
metadata, complete filing reference/version/basis/issuer applicability, or
rebalance announcement/current/preview/effective semantics. Those real
capabilities stay unsupported. Unknown timestamps, mixed revisions, and
equal-priority conflicting records fail explicitly.

### 2.5 Bounded store build and memory contracts

The addendum's memory diagnosis was accepted. The builder now:

- computes one family/sidecar group at a time;
- performs per-date rank transforms row-by-row;
- writes directly into preallocated disk-backed arrays;
- stores feature and target arrays as float32 while permitting bounded
  float64 working calculations;
- streams each horizon's target construction;
- closes/releases mapped workspaces before the next family; and
- records peak resident memory in the build manifest.

The production-axis regression uses 4,348 sessions by 933 securities with all
families enabled, requires nonzero target support, and incorporates committed
real archive heights from `research/configs/v2/archive_scale.json`. Native
training collation/forward has a separate memory check. The earlier stale
positional target call was replaced with named arguments.

For chronology: the earlier fix-pass implementation did attempt a rebuild
under the superseded semantics. It exhausted the 16-GiB Windows host while the
events adapter materialized an archive-sized Python row list, and it stopped
before atomic promotion or any score. That failure is what motivated the
addendum and the bounded rewrite above. After the broader semantic refactor,
no new current-schema real rebuild was attempted because the enhanced action
and calendar gates stop it earlier.

### 2.6 Canonical consumers and native model path

One decision-sample contract supplies ordered values, per-feature validity,
source ages, real calendar/padding state, entry eligibility, compact native
fast inputs, and separately held outcomes. Neural, GBDT, baseline, scoring,
and evaluation adapters bind the same date/security/feature identity. Their
missing-value representations differ intentionally, not their underlying
sample.

Neural inputs zero invalid payloads with `where`, concatenate value, validity,
bounded age, and age-known state, project to the latent width, and normalize
after that learned projection. Left padding cannot advance the GRU; a genuine
missing exchange session remains a timestep. Active names with no valid slow
history get the deterministic zero initial state. GBDT receives NaN only for
masked-invalid numeric cells and rejects infinities or nonfinite valid cells.

The default fast path is native to the current model:

- seven separately masked five-minute channels;
- real dated-session prefixes (normally 69 patches), not 12 compatibility
  patches;
- completed-bar endpoint and no-gap-bridging rules;
- compact collation for present names only; and
- fresh seeded weights by default.

The old v1 fast encoder survives only as an isolated
`legacy_v1_contaminated` diagnostic adapter. It requires explicit opt-in,
hash identity, and ancestor chronology; it is refused for clean/official
claims. Hash equality alone is not temporal admissibility.

All stages now use the same row semantics. The five D1/D2/D3/D5/D10 heads
remain primary and are averaged head-by-head over their supported dates. The
to-close head remains available but defaults to weight 0; the only declared
enabled weight is 0.2. Microbatching preserves full date cross-sections (or
complete adjacent-date pairs for persistence), and accumulated SAM perturbs
and updates once per effective batch.

### 2.7 Intended orders, fills, and signed-share accounting

The larger order/fill repair was adopted despite 3c's initial request not to
add an order framework. This is a small research execution boundary, not a
generic OMS:

- deterministic policy emits immutable intended orders from the decision
  snapshot, prior marks, positions, pending orders, and known assumptions;
- later observations determine fills, partial fills, expiry, or no fill;
- a missing top-name print reserves its slot and cannot trigger a hindsight
  replacement;
- loss of eligibility requests an exit but does not manufacture a sale; and
- unresolved held assets remain positions/claims with explicit scenario
  values rather than assumed proceeds.

The ledger carries signed shares, free/restricted cash, receivables/payables,
pending inventory, costs, financing, borrow, and valuation status. It applies
contractual actions once and reconciles daily equity. It reports unresolved
inventory separately, compounded excess versus all-cash equity, actual holding
ages, exposures, fills, costs, and risk breaches. Nonpositive NAV stops future
orders. Reference planned controls are gross 2.0, planned gross cap 2.25,
absolute net cap 0.10, and per-name cap 0.05.

The addendum's empty-book concern was generalized into an invariant: a fold
whose mean deployed gross is below 50% of the gross target is
`economics_unresolved`, and its paired economic delta is null/not admissible
for a not-worse conclusion. The old Round-1 artifacts remain voided
engineering evidence and were not reinterpreted as performance.

### 2.8 Evaluation, artifacts, and stale-run rejection

Primary IC uses one population for `P={D1,D2,D3,D5}` on each date: the
intersection of eligibility, finite positive risk scale, all four outcome
masks, and all compared score masks, with at least 20 names. Every head's
Spearman is computed on that same population, and the daily primary is defined
only if all four heads are defined. D10 has a separate all-five diagnostic and
never controls primary checkpoint selection.

Reports keep the primary normalized IC, shareholder-return IC, price-return
IC, raw spreads, incremental holding-wealth intervals, support reasons,
coverage, persistence, holding ages, accounting attribution, and unresolved
exposure. Candidate comparisons use exactly common names/dates and preserve
fold boundaries and missing calendar dates in moving-block resampling.
Undefined comparisons fail the not-worse guard.

The store, model input, checkpoints, scores, evaluation reports, runner plans,
and protocols use bumped schemas and hash-bound identities. Current stores
must carry the complete FeatureSpec, clock/calendar, action contract, axes,
masks, and source provenance. Superseded stores, checkpoints, scores, partial
roots, and old Round-1/Round-2 entry points fail before payload use. There is
no in-place migration and no compatibility flag in the canonical path.

## 3. Disposition of the narrower proposals

### 3.1 `v2_fix_pass_3_addendum.md`

| Recommendation | Disposition | Reason/result |
|---|---|---|
| Vectorize event/odd-lot daily adapters and test against a row oracle | Adopted | Removes the observed Python-object memory blow-up while preserving exact as-of semantics. |
| Scale memory test to real archive heights and retain the 8-GiB ceiling | Adopted | The committed archive-scale fixture and populated production-axis test now exercise the previously missed path. |
| Log/refuse a real rebuild below 10 GiB free | Adopted as preflight policy | A real rebuild was not attempted because an earlier source gate stopped acceptance first. |
| Produce the detailed action timing/bucket breakdown before rebuilding | Adopted | The immutable-source audit and its hashes are described below. |
| Enable strict fallback only under the frozen thresholds | Thresholds applied; fallback rejected | It did not meet the recall/precision gates and neither heuristic is accepted as contractual economics. |
| Treat underdeployed books as unresolved | Adopted and generalized | Implemented in the canonical ledger/evaluation contract rather than as a special case for two old controls. |
| Continue directly to rebuilt store and revised Round 1 | Not performed | The audit triggered explicit stop conditions, and the stronger full-plan action/calendar prerequisites are unavailable. |

### 3.2 `v2_fix_pass_3c.md`

| Recommendation | Disposition | Reason/result |
|---|---|---|
| Decision-prefix invariance and bar-345 exclusion | Adopted | Current decision inputs cannot depend on the entry bar or post-decision close/quantity. |
| Treat a >30% open gap as an action boundary | Diagnostic only | It is causal as an observation but is not evidence of contractual units. It cannot adjust accepted economics. |
| Raw OHLC/activity/duplicate gate | Adopted and extended | Source completeness, trade observation, and activity validity are now independent masks. |
| Calendar diagnostic on an authoritative schedule | Interface and hard gate adopted | No verified historical schedule artifact was available to satisfy it on real data. |
| Same-ticker link allowlist | Adopted and strengthened | Accepted links also require explicit conversion terms/evidence; the allowlist remains empty. |
| Sao-Paulo/absolute-time assertion | Adopted | Timestamp precision and latency are included, not timezone label alone. |
| Lending/odd-lot as-of revisions | Adopted | Historical changes use only the vintage known at each decision. |
| Repair target memory fixture | Adopted | Named target arguments and nonzero validity are asserted. |
| Orders-before-fills and unresolved inventory | Adopted and extended | Full signed-share/cash/claim reconciliation was necessary to repair F03/F04/F06/F07. |
| Neural validity masks and GBDT NaN contract | Adopted and extended with feature ages | Missing and valid zero are distinguishable in every consumer. |
| Fresh fast initialization; contaminated legacy arm isolated | Adopted | The canonical path no longer depends on v1 assets or unverified v1 chronology. |
| Shift every stage to `sample_date_index-1` | Rejected | Canonical row `t` is already a decision snapshot whose daily fields end at `t-1`; shifting again double-lags. |
| Set to-close default to zero | Adopted | Daily five-head objective is primary; auxiliary is explicit opt-in only. |
| Common four-head evaluator population | Adopted | D10 is kept as a separate diagnostic. |
| Do not add typed transforms, clock, native M1, action terms, or broader order/fill boundary | Rejected as a scope restriction | Those components are required to solve confirmed root defects in the later complete plan. |
| Register revised research immediately after code changes | Not performed | Real-data engineering acceptance is unsupported, so preregistration/research must wait. |

### 3.3 `Brazil_RV_Multiday_Refactor_Plan.md`

Its Phases A-F were used as the implementation structure: contracts, raw
foundation, features/as-of state, consumers, orders/accounting, unified
evaluation, and migration. Phase G is source-gated and is reported as
unsupported. Phase H, the registered development sequence, was not begun.

The plan's “small research sequence after engineering acceptance” is therefore
still future work. This includes corrected naive/GBDT reruns, native neural
development runs, the single representation comparison, and the D5-only
execution diagnostic. None was opportunistically run on the same development
folds while plumbing semantics were changing.

## 4. Corporate-action audit and why the real rebuild stopped

The enhanced source-quality audit was run against the immutable previously
accepted store, without generating a model score or consuming an
official-validation/test research decision. Its action/timing coverage spans
the source archive; that must not be confused with evaluating a model on a
sealed outcome window. The evidence root is:

`D:\quant-data\b3\processed\v2_refactor_action_audit_c316330_20260906T013921Z`

Key immutable identities:

- audit SHA-256:
  `d717899d7174cfb4645d06db63957c88ab3fea608ad3f33ebf49aac96c340d43`;
- detailed breakdown SHA-256:
  `98bd1b9133c30cc30ca1337c8219a7692c903f3a183a2e714b82e0576692e871`;
- source-store manifest SHA-256:
  `6a7e13195c6cde92fbdc756a585e4cb65d73998faa94e237595c7be7cdfb6919`;
  and
- audit implementation commit: `c316330`.

The preregistered decision inputs were:

- 478 provider-labelled split rows;
- among 58 rows with `|log(factor)| <= 0.08`, only 81.03% had a `DISMES`
  change within two sessions, below the required 90%;
- among 420 rows with `|log(factor)| > 0.08`, only 21.67% had a DISMES-only
  split detection within two sessions, below the required 90%;
- among 272 rows with `|log(factor)| > 0.30`, none showed the price jump one
  or two sessions before `DISMES`, so the rule that would enable the strict
  fallback did not fire; and
- within +/-2 sessions, DISMES-only recall/precision were approximately
  0.1925/0.4646, while adding the fallback produced approximately
  0.2134/0.3003. Precision remained far below the required 0.70.

The old pipeline also marked 114,604 cash events, 102,372 of which (89.33%)
had no `DISMES` change. This confirms that synthetic residual cash inference
was not a defensible economic source.

The correct decision was therefore not “pick the less bad classifier.” The
audit explicitly records:

- `canonical_price_ratio_adjustment_authorized=false`;
- legacy classifier gate failed;
- strict fallback not enabled; and
- accepted economics requires verified contractual share/cash terms, with
  uncovered intervals unresolved.

This supersedes the earlier fix-pass statement that DISMES-only could become
the canonical replacement merely because the strict fallback was worse. The
new breakdown applied the addendum's own wider timing tests and then the full
plan's stronger distinction between a detection heuristic and an action-term
source. Since those requirements were not met, the real store rebuild and all
downstream economic scoring correctly stopped.

## 5. Source-dependent capabilities deliberately left unsupported

These are not unfinished silent assumptions. Their interfaces, failure modes,
and fixtures exist, but real acceptance needs evidence not present in the
repository:

1. **Authoritative dated B3 schedule.** The loader/validator is implemented,
   but there is no verified historical schedule covering holidays,
   exceptional sessions, and dated continuous/auction boundaries. An inferred
   market-data calendar is not substituted.
2. **Complete contractual action master.** Available provider observations and
   DISMES/jump diagnostics do not prove complete historical share, cash,
   conversion, payment, availability, and terminal-claim terms, especially for
   inactive names and complex events.
3. **Verified auction marks/capacity.** COTAHIST `PREULT` is a daily last-trade
   close proxy, not proof of a closing-auction print or available auction
   volume. The code labels the proxy and does not claim auction execution.
4. **Historical terminal asset status and settlements.** Suspension,
   delisting, conversion, and terminal recovery require a verified source.
   Unknown holdings remain unresolved instead of being sold at a stale mark.
5. **Executable borrow availability/rates.** Published stock-lending market
   fields are features/proxies, not guaranteed strategy borrow. Any future
   unrestricted-short/uniform-rate result must remain an explicitly
   conditional scenario.
6. **Unsupported sidecar fields.** True earnings-announcement age/SUE without
   timestamped announcements and point-in-time expectations, ATM-IV fields
   absent from the archive, and market-cap/book-to-market/profitability fields
   lacking coherent share-class/filing inputs remain disabled rather than
   being reconstructed from clipped proxies. The current raw options,
   fundamentals, and rebalance adapters are bounded transform interfaces, not
   source attestations: the available inputs do not establish canonical OI
   units/multipliers/expiries/adjustments, complete filing
   reference/version/basis/applicability, or announcement/current/preview/
   effective rebalance semantics.
7. **Complex actions.** Rights, spin-offs, multi-claim mergers, and conversions
   without complete mappings remain unresolved; they are not squeezed into a
   scalar split/dividend formula.

Obtaining or purchasing new sources was outside this refactor request. The
machine-readable acceptance result therefore must distinguish `unsupported`
from both `pass` and `fail`.

The current hard Phase-G blockers are items 1, 2, and 4. Items 3 and 5 limit
the strength of execution claims but do not prohibit every acceptance path:
the full plan explicitly permits a labelled daily-last-trade `close_proxy` and
conditional uniform-borrow or long-only scenarios. Such results could support
screening, but not claims of verified auction capacity or executable
stock-specific borrow.

## 6. Work completed before and after the usage reset

### Before the reset

The branch already contained reviewable commits for:

- expanded corporate-action timing/bucket audit;
- hardened COTAHIST foundation, decision clock, and identity-link allowlist;
- post-close-invariant intraday inputs and scheduled daily summaries;
- the typed FeatureSpec and exact transformation semantics;
- revision-safe, bounded sidecar processing and real archive-scale fixtures;
- verified contractual actions, shareholder wealth, and streamed economic
  targets;
- native five-minute feature construction, disk-backed streaming, compact
  fast collation, and the fresh native model path;
- signed-share/cash/claim accounting and order-before-fill behavior;
- independent return-family evaluation and one common primary population;
- mask-first target capabilities and current schema versions; and
- hard voiding of stale research roots/presets under the superseded semantics.

Those commits are part of the final branch history; the reset did not restart
or discard them.

### After the reset

The final integration pass concentrated on gaps that could still produce a
false acceptance:

- made `source_session_complete`, `trade_observed`, and `activity_valid`
  explicit through panel, universe, feature, intraday, and store boundaries;
- ensured unresolved cross-session action chains do not erase safe same-day
  price/activity/history features;
- made the complete ordered FeatureSpec mandatory at store open and rejected
  names-only current schemas;
- gave audit-only common-state arrays a validated date/common-field axis;
- removed the canonical store's undocumented-split fallback option;
- validated in-memory schedules as strictly as file-loaded schedules;
- fixed the final leakage-audit finding in lending ADV20: a ratio requires all
  20 canonical activity observations, internal missing activity invalidates
  the window, and an observed zero-volume session remains valid support;
- made a completed classical-only validation run report engineering acceptance
  as `unsupported`, rather than implying that process completion was a pass;
- added an authentic T24 route from fixed-width raw COTAHIST and timestamped
  M1 through parsed/audited canonical store, tiny native fit/score, ledger,
  report, relocation, and stale-resume rejection; and
- closed disk-backed workspace mappings explicitly so the same end-to-end path
  is clean on Windows as well as Linux.

Documentation was rewritten around the current model instead of appending
exceptions to the old pipeline, and a migration note makes the semantic break
explicit.

## 7. Deliberately excluded or deferred work

The following suggestions were not implemented as part of the canonical
refactor because they are either research questions, need missing data, or
would create unjustified infrastructure:

- no broad architecture, feature, loss, optimizer, volatility-estimator,
  horizon, threshold, buffer, or cost sweep;
- no risk-scaled-input or common-context performance arm before registration;
- no D5-only economic run before engineering acceptance;
- no v1-initialized clean arm—the only v1 path is explicitly contaminated and
  diagnostic;
- no automatic same-ticker identity linking or price-ratio action repair;
- no global learned scaler fitted during store construction;
- no interpolation of OHLC/M1 gaps or conversion of source absence to
  no-trade;
- no invented auction fill, terminal sale, cash entitlement, borrow quote, or
  market capacity;
- no generic OMS, live routing, sophisticated impact model, general portfolio
  optimizer, or production deployment machinery;
- no in-place rewrite of old stores/checkpoints/results; and
- no revised preregistration or new development experiment while Phase G is
  unsupported.

“Phase D” in the full plan denotes consumer implementation, which was code and
test work. It should not be confused with the user's prohibited experimental
Section D. No new experiment was run.

## 8. Final verification

- Final implementation commit:
  `f0cf568303715e8783e539a683568535e2232c7f`.
- Documentation/evidence payload commit:
  `8c9121e3a0d909ea5336bdc491d1638d931cdab6`.
- Main/GitHub synchronization: verified after the final push; local `main` and
  `origin/main` resolve to the closeout commit containing this statement.
- Ruff/compile/full-suite result: `uv run ruff check src tests` passed;
  `uv run python -m compileall -q src tests` passed; the research suite passed
  **810 tests in 372.11 seconds** with PowerShell
  `$env:BRAZIL_RV_TEST_SCRATCH='D:\'; uv run pytest -q`; and
  `uv run --group dev pytest tests/test_parse_b3_cotahist.py -q` passed the
  **4-test** collector COTAHIST suite in **0.18 seconds**.
- T23: all **5** dedicated resource tests passed in **213.00 seconds**. The
  component fixture separately exercises populated 4,348 × 933 family/target
  arrays, bounded derivations at committed archive row counts, and native
  60-step collation/forward under an asserted peak-RSS ceiling of 8 GiB. It
  does not run one integrated large-source materialization/full builder, and
  only the `<8 GiB` bound—not exact peak bytes—is durable. The admission check
  requires at least 10 GiB of conservative physical/commit headroom.
- T24: the authentic deterministic raw-fixture path passed **1 test in 78.53
  seconds**, covering fixed-width COTAHIST and timestamped M1 parsing through
  store build, native fit/score, orders/fills/ledger/report, relocation, and
  stale-artifact/resume rejection.
- Machine-readable acceptance report:
  `research/acceptance/v2_refactor_20260906.json`, status `unsupported`,
  SHA-256
  `d41d64825092ee4870a8cd9ff4e7eb4d9fd1f1ea583f5a8a06956e2b4e0522c6`.
- Paid-compute hygiene: no instance was launched for this refactor. Two fresh
  provider inventories at 2026-09-06 10:36:04 and 10:38:27
  `America/Sao_Paulo` each returned zero instances; the earlier exact instance
  `46ee1e1c16f14d7c8fd737919ed66400` was absent in both.

The expected semantic conclusion is unchanged by test-count bookkeeping:
current-path engineering behavior may pass its local regression matrix, while
real-data economic acceptance remains `unsupported` until the missing calendar
and action/status evidence exists. That distinction is intentional and is the
main safeguard against turning repaired plumbing into an unsupported research
claim.
