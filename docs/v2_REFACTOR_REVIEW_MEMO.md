# Brazil-RV multi-day refactor: implementation and design-disposition memo

This memo is intended for the authors/reviewers of
`v2_fix_pass_3_addendum.md`, `v2_fix_pass_3c.md`, and
`Brazil_RV_Multiday_Refactor_Plan.md`. It explains how their recommendations
were reconciled, what was implemented, what was intentionally changed or
deferred, and why the result stops before new research experiments.

Section 8 records the later `v2_fix_pass_4.md` continuation. That pass does
not reverse the refactor's leakage, identity, decision-clock, or accounting
rules. It adds two explicitly labelled development-data tiers so the model can
eventually produce a screening number without misrepresenting inferred terms
or a reconstructed calendar as verified source truth.

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
- A later development-only route now exists for inferred COTAHIST/DISMES action
  terms and a reconstructed session schedule. It remains unaccepted on real
  data because the mandated local build could not start with only 6.45 GiB of
  free physical memory against the fixed 10-GiB admission gate.
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

## 8. Development-grade continuation (`v2_fix_pass_4.md`)

Pass 4 correctly identified a contract deadlock: the verified action path
cannot cover the historical inactive-name panel with the available sources,
and provider timestamps fetched in 2026 cannot be backdated into historical
decision inputs. Rather than weakening the verified contract, the
implementation adds a separate and permanently labelled tier:

- `action_terms_source=inferred_cotahist_dismes_v1` infers U1 unit changes,
  C1 bounded market-relative cash drops, and U2 undocumented unit changes
  uniformly from COTAHIST. Provider rows are excluded from every model array
  and retained only for audit. Inferred rows remain `coverage_status=inferred`
  and are never promoted to verified terms.
- Prior price and activity windows run in cumulative unit-adjusted space, so
  a U2 followed shortly by a DISMES change cannot create a second synthetic
  jump. U2 keeps the specified per-trade inverse-move test, falls back to total
  quantity only when trade counts are unavailable, and retains the exact 0.35
  tolerance.
- A row for session `e`, timestamped at end of session, is first available to
  the next decision `e+1`. The daily action alignment therefore uses the next
  session's 15:45 cutoff. The same-session intraday path remains isolated and
  uses only its causal open-gap boundary diagnostic.
- `schedule_source=reconstructed_v1` is generated from COTAHIST sessions with
  at least 50 records, cross-checked against a committed ANBIMA holiday table,
  and reconciled against an explicit zero-unexplained-exception inventory.
  M1 cross-name modal bounds support dates from 2021-08-16; earlier dates and
  anomalous M1 bounds use an attributed dated regime table. This is a
  reconstructed development schedule, not an authoritative exchange claim.
- Verified terminal status is no longer a universal build blocker. The signed
  ledger's unresolved inventory counts/notional and last-mark plus haircut
  scenarios are mandatory report fields. Verified auction marks and
  executable borrow remain unsupported, so economics retain the
  development-grade close-proxy label.

The pass-4 audit extension was run against the immutable old store before any
rebuild. Its new evidence root is
`D:\quant-data\b3\processed\v2_development_action_audit_d42df9a_20260906T152426Z`;
the audit SHA-256 is
`9db65755c654b54da26dbef62ef5704ebe8e4832b7c99090cd0a0045925762df`.
Of 272 provider rows with `|log factor| > 0.30`, 222 (81.62%) had a matching
large COTAHIST price move within two sessions. All 222 price-corroborated rows
had a DISMES change within two sessions. U2 matched 1.80% under the per-trade
test and 0.45% under the total-quantity test. These figures are diagnostic by
registration: they neither enable nor disable U2 and cannot tune its rule.

Provider invariance is structural and tested: changing the provider bundle in
the inferred tier leaves all model arrays byte-identical. The store and every
training, checkpoint, score, evaluation, ledger, acceptance, and research
artifact carry both source-tier labels. A separate independent native-fast
audit re-computes all seven channels from raw M1 for exactly 20 names by 20
sessions and is required by the development acceptance report.

The revised registration is
`research/preregistrations/v2_round1_round2_rev2.md`. The Round-1 and Round-2
candidate roster is unchanged. Round 2 now starts with one disposable
Arm-A/F1/seed-11, one-epoch, no-score smoke trajectory; it is never reused by a
registered run, and Stage P cannot be planned unless that smoke passes. The
legacy v1 fast checkpoint remains excluded from clean research; Round 2 uses
fresh native fast weights.

The completed source-tier implementation is commit
`d42df9a61c0a45ddcd00c184c5b9e97fb9f91781`. Ruff and all 814 research tests
passed, including the 4,348-by-933 all-family peak-RSS invariant under 8 GiB.
No real store, development acceptance number, Round-1 candidate, Round-2
trajectory, official-validation/test access, deployment change, or new paid
instance followed. The local host exposed only 6.45 GiB of free physical
memory, so the unchanged 10-GiB preflight stopped the real rebuild exactly as
specified. Consequently the branch is pushed for review but is not merged to
`main` and the rev2 registration remains frozen-but-unexecuted.

## 9. Pre-pass-4 verification record

- Final implementation commit:
  `f0cf568303715e8783e539a683568535e2232c7f`.
- Initial documentation/evidence payload commit:
  `8c9121e3a0d909ea5336bdc491d1638d931cdab6`.
- Final machine-evidence content commit:
  `b356a562691217313f326f3b6fbcb7bce27d631d`.
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
  `38393706d8736c08deafe9eda22074a2b4518ab3e06e6e2c63f61b1b7523250e`;
  finalized at 2026-09-06 10:47:28 `America/Sao_Paulo`, after both provider
  observations.
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

## 10. Pass-4 addendum: host admission and executable registration binding

The addendum exposed four prose contradictions in the unexecuted revision-2
registration. They are now corrected to the implementation that already
existed: chronological fit, 10 purge sessions, 55 selection sessions, 10 purge
sessions, and one continuous evaluation path; one selected model per fold and
seed; GBDT early stopping on the complete selection window; canonical decision
rows for both pretraining and fine data, with pretraining intraday fields absent
and `fast_present=0`; and tie-aware rank averaging of the three seed panels on
the evaluation window. The obsolete parity stitching, selection-parity,
consumer-side lag, and nonexistent age-field descriptions were removed. The
statistical headline is named once as the median-adjusted,
volatility-scaled primary IC over D1/D2/D3/D5.

The registration now ends with one machine-readable JSON protocol. Its purge
and selection lengths are sourced from the same constants used by
`development_folds`; its evaluation bounds come from the registered fold
windows; its primary population comes from the evaluator contract; its
headline signal and complete ledger configuration come from the actual default
ledger; and its source labels come from the development-tier constants.
`freeze_round1` parses and compares this block before reading Git identity or
hashing the registration, so prose/code drift stops before an immutable root is
created. The dedicated test also constructs the folds and checks their realized
evaluation endpoints, population, headline ledger, and tier labels.

The host preflight is now similarly executable. Before source loading it
records physical and commit memory, the resolved process temp and all known
store staging parents, the output-drive free bytes, and the exact prior-store
size. It rejects any Windows C:-drive staging path, retains the 10-GiB
conservative memory requirement, and requires output free space at least three
times the prior immutable store. `--previous-store` is mandatory, so the disk
test cannot silently use a guessed size. The report is printed before admission
is enforced, which preserves the reason even for a score-free refusal.

I deliberately corrected and committed the registration before rebuilding,
rather than following section 3's literal “rebuild, then commit registration”
ordering. The existing acceptance and Round-1 gates require the store,
acceptance report, and current clean Git identity to match exactly. A
registration commit after the build would therefore make the accepted store
unfreezable. The conservative coherent order is: finish and commit the
registration guard, rebuild and accept at that exact commit, fast-forward
`main` to the same commit, then freeze Round 1. This changes no research rule;
it preserves the addendum's exact-identity rule.

The code-bearing addendum commit is
`23ff904967af699b9d1a682bc597ffcb0fc22e61`, pushed on
`fix/v2-development-grade-data`. Ruff and compile passed; the complete research
suite passed **816 tests in 320.79 seconds**, followed by **31** focused
registration, split, and preflight tests. The prior store is 2,578,928,232
bytes (2.402 GiB), so the three-times output requirement is 7,736,784,696
bytes (7.205 GiB).

The actual clean-commit preflight ran with `TMP`, `TEMP`, and
`BRAZIL_RV_TEST_SCRATCH` on D:. It recorded 10,502,467,584 bytes (9.78 GiB)
of free physical/build memory, 12,054,888,448 bytes (11.23 GiB) of free commit,
a 21,059,461,120-byte (19.61-GiB) commit limit, and 278,246,088,704 bytes
(259.14 GiB) free on D:. Every workspace/staging path resolved off C:, and the
disk gate passed; only the unchanged 10-GiB memory gate failed. The log is:

    D:\quant-data\b3\processed\model_runs\v2_store_build_preflights\preflight_23ff904_20260906T130537.log

Its SHA-256 is
`eb165ea2fb590dcdffa292a23392e258569588079abf522589c7b1c2120c173f`.
The proposed store root was not created and no source was loaded. I initially
stopped at this gate. The user then explicitly authorized bypassing only the
10-GiB memory admission check. Commit
`ba2d92785443729a6beabde01e27a015363b4b58` added a narrowly named, explicit
acknowledgement flag: it leaves the D:-staging and three-times-store disk gates
binding, records both raw and effective admission decisions, and changes no
data, leakage, identity, target, simulator, accounting, or research gate.

Two subsequent score-free integration defects were fixed before a result. The
native-fast audit had incorrectly required the 10:03--15:45 raw prefix to be
divisible by five even though production deliberately floors the leading
partial bucket; commit `3a845d9` made the independent audit use that same
flooring and added an exact regression for the real prefix. The validator then
looked for obsolete v1 identity/calendar flags; commit
`12e6ae08eb67014c40160809d47212453a4f3f90` instead verifies both current axis
hashes and the sealed zero-row calendar-completeness audit. The exact
builder/consumer commit guard was not weakened: an earlier-store validation
attempt stopped before root creation, and the store was rebuilt at `12e6ae0`.
Ruff and all **817** tests passed at that commit.

The commit-matched store is
`D:\quant-data\b3\processed\v2_daily_store_12e6ae0_20260906T173300Z`, manifest
SHA-256
`2f537944ba857675265031da4f2f4327fdffd6d452b76d4de818d752f1272764`.
It peaked at 7.802536 GiB RSS, so it also satisfied the original 8-GiB measured
build invariant despite the admission override. Calendar completeness was
zero, both internal and target survivorship gaps passed, all external
contemporaneity/composition checks passed, and protected access remained
false/false. The matching 20-name-by-20-session native-fast audit is
`D:\quant-data\b3\processed\model_runs\v2_native_fast_audit_12e6ae0_20260906T183448Z`;
its audit SHA-256 is
`f1c23ef1884b5bcf3f0224ee4db22533e0bfc6e67d62cae0b0cf701f9ebd7174`,
with exact equality for every channel and mask.

The first complete development-grade classical acceptance then stopped the
program at its unchanged gross-utilization gate. Its root is
`D:\quant-data\b3\processed\model_runs\v2_development_acceptance_12e6ae0_20260906T183511Z`;
pipeline-manifest SHA-256 is
`34ac98f8646934cda82290826a1ad0fd12921393876d726a829cb53825669c38`,
and the log-inclusive artifact-inventory SHA-256 is
`8b011067471067c3ceafb2b78b2153b6331b18a66db4bd89ba6490da04207445`.
All 15 baseline evaluations and the 25-member F1 GBDT ensemble completed. The
naive pooled IC checks passed (`momentum=0.0564571`, `reversal21=-0.0193444`,
`reversal5=-0.00774240`, `blend=0.0446152`), the reversal-5 sign check passed,
the independent fast audit passed, and mean terminal unresolved inventory was
0.0179667 NAV, below the 0.02 ceiling. However, mean deployed gross ranged from
0 to 0.947743 NAV across the 16 evaluated books, versus the binding 1.8--2.2
range around target gross 2. Every book therefore failed the same gate; the F1
GBDT deployed only 0.477454 NAV gross. The report is correctly labelled
`unsupported`, not accepted research.

Per the registered stop rule, Round 1, Round 2 smoke, Round 2 arms, a merge to
`main`, and any paid instance were not started. Official validation and the
permanently spent test remained unread, and no deployment changed. Suggestions
to relax or reinterpret gross utilization were not implemented because that
would alter a binding preregistered acceptance rule after observing the first
honest result. The branch remains the review surface; a future continuation
must diagnose feasibility and be separately authorized or preregistered.

## 11. Pass-4b disposition: repair the ledger, preserve the research boundary

Pass 4b correctly identified four interacting evaluator defects: a blanket
entry stop while any exit was pending, paired-side refills, day-wide aborts on
one blocked candidate, and full-book liquidation on an ordinary risk breach.
The implementation replaced them with occupied-slot accounting, independent
alternating side queues, candidate-local skips, and partial per-name/gross/net
risk trims. The planned absolute net cap moved from 0.10 to 0.20 exactly as
specified and is bound in the executable registration. Gross target 2.0,
planned gross cap 2.25, name cap 0.05, fills, costs, claims, target construction,
score masks, folds, and all acceptance thresholds were unchanged.

One requested synthetic test was adjusted in mechanism but not purpose. A 15%
rally in the long side of a self-financing long/short book increases NAV faster
than gross exposure and therefore cannot create a gross-fraction breach. The
test uses an equally sized 15% market rally adverse to the short side, which
does create the intended gross breach, and verifies that both sides are trimmed
only in proportion to their excess. Encoding the literally impossible setup
would have produced a test that could never exercise the requested branch.

The first replay attempt at commit `2bfa986` completed one evaluation and then
stopped in its reporting comparator. Four `mask_coverage` fields—actual risk
breach dates, stale-mark name-days, unresolved-action name-days, and valuation
scenario count—are derived from the ledger path, so classifying them as
non-ledger identity was incorrect. The failed root and logs were retained. The
bounded commit `48d5562` classifies only those four fields as ledger-derived;
it changes no score or economics. The fresh replay then hash-verified and reused
all 16 prior score panels without fitting or scoring a model. All genuine
non-ledger fields, including IC, persistence, and coverage, were bit-identical.

The successful immutable replay root is
`D:\quant-data\b3\processed\model_runs\v2_development_acceptance_ledger_replay_48d5562_20260906T192609Z`.
Its manifest SHA-256 is
`b62d53d31986d304fabedd6fbfde6dff32167198e634f5b9fb2dad9aab49bdeb`;
the complete gate diagnostics and log-inclusive inventory SHA-256 values are
`1bdeab6d58da51fec251fec7643f974cb7a617ea676c68deaf806cd4868c8953`
and `dfa783d6500cc72c0cd124ea75cf435f935494a259f9f9a43075265ff7898bcd`.
The access audit passed false/false with no deployment change.

Utilization improved substantially, proving that the policy repair was real,
but the unchanged acceptance contract still failed. Four books entered the
1.8--2.2 mean-gross band; 12 did not, including F1 GBDT at 1.773067. Mean
terminal unresolved inventory also rose to 0.0361789 NAV, above 0.02, because
the repaired ledger now carries positions that the starving implementation had
never opened. The exact momentum/blend masks expose a deeper contract conflict:
they provide only 15--46 eligible names per day, and F1/F3 peak at 34/24. A 5%
name cap makes the 1.8 gross floor unattainable on those dates even if every
eligible name is held. This is a population/cap/gate incompatibility, not a
remaining queue-policy defect.

Accordingly, the suggestions to fast-forward `main` and start Round 1/2 were
not implemented: the source document made those actions conditional on every
acceptance gate passing. No bound, mask, cap, or population rule was relaxed
after the result; no scored candidate was retried; and no paid instance was
launched. The review branch is intentionally left as the complete engineering
surface. Any next research pass must preregister how sparse score populations
interact with portfolio gross, rather than retrofitting the observed result.
Final verification on the completed branch passed Ruff, Python compilation,
and all 824 research tests in 380.76 seconds.
Provider inventories at 2026-09-06T19:40:23Z and 19:40:25Z each returned zero
instances, confirming that this continuation neither launched nor left paid
compute running.

## 12. Pass-4c pre-rebuild diagnosis: the registered premise did not hold

Pass 4c correctly required a no-rebuild validity waterfall before changing
the shareholder-wealth recurrence, and explicitly said to stop if
missing-print restarts were not the dominant cause of sparse 12-1 momentum.
That stop condition fired.  On the sealed `12e6ae0` store, the slow-feature
`momentum_12_1` mask covers 98.97%, 98.65%, and 98.22% of active name-days in
F1/F2/F3; its minimum daily coverage is 98.06%, 97.87%, and 96.69%.
After requiring both wealth endpoints and excluding unresolved actions, the
specified restart scan removes exactly zero additional evaluation name-days
in every fold.  A missing-print restart therefore cannot explain the observed
15--46-name baseline panels.

The same audit isolated the actual mask collapse.  The naive baseline uses
`decision_action_boundary_mask`, which includes every resolved corporate
action as well as unresolved actions.  Over its 12-1 interval this removes
20,844, 19,855, and 20,113 otherwise-valid name-days in F1/F2/F3, leaving only
14.35%, 14.43%, and 11.15% coverage.  Applying the pass-4c-described
unresolved-action-only guard instead would leave 99.09%, 98.68%, and 98.37%.
This is a separate baseline interval-mask defect, not evidence for the proposed
wealth-index rebuild.

The immutable score-free diagnostic root is
`D:\quant-data\b3\processed\model_runs\v2_pass4c_prechange_waterfall_19a9dbe_20260906T195326Z`.
Its manifest and diagnostic-result SHA-256 values are
`8343c4c98619dae5b3cbbd53d75ec28834f9435ace70c6570d202c8d5f34d4de`
and `f83bf1c53abdcf862ca89ebb6d495761d966605a54840872beb3592c62d2b491`.
Protected access remained false/false and no score was written.  In accordance
with the document's first explicit gate, no store rebuild, ledger change,
acceptance rerun, merge, Round 1/2 run, or instance launch was performed.
Proceeding now requires a corrected pre-result instruction that authorizes the
resolved-action baseline-mask repair and independently decides whether the
wealth recurrence should still be changed as a latent data-quality fix.

## 13. Pass-4d disposition: consumer-specific interval guards and auction slots

Pass 4d supplied that corrected pre-result contract. The wealth-bridging idea
was withdrawn because the pass-4c audit found no evaluation loss from
raw-series restarts. We did not add a synthetic bridge or manufacture missing
wealth observations. Instead, the canonical store now measures active-name-day
restarts by year, and any return spanning one remains invalid by construction.

The central design choice was to remove the overloaded
`decision_action_boundary_mask`. A single mask cannot correctly serve
shareholder-wealth returns, raw-price cross-session fields, targets, and
execution. Wealth features and baselines now share one exact validity path and
reject only unresolved actions, missing endpoints, or a raw-series restart.
Raw-price cross-session intraday fields reject unresolved actions and unit
changes, including successor conversions, but accept resolved cash actions.
Same-session scale-free intraday fields need no corporate-action guard.
Targets and the ledger keep their existing aligned economic terms. This is a
semantic schema break, so the store/evaluation/acceptance schemas were advanced
instead of preserving a compatibility shim.

The current-store audit justified rebuilding for this change: corrected
intraday coverage improved by up to 24.471996 percentage points, with 24
feature/fold cells above the preregistered 10-point trigger. The immutable
audit root is
`D:\quant-data\b3\processed\model_runs\v2_pass4d_intraday_guard_audit_5e500a4_20260906T200144Z`;
manifest and result SHA-256 values are
`ffb537f0cf05515a355e42f9cb6833baae46f208aadbda5f005b89dec87564da`
and `0822c3ac5629d893edcb5dade4d6a37a9e948d0e71e69045b53a2c1ad42fab9b`.

For the ledger, a complete close-proxy exit and replacement entry are treated
as the same auction: the submitted exit releases its slot immediately, older
pending exits do not, and a failed exit plus filled replacement is handled by
the existing subsequent risk trim. Small universes use
`min(K, floor(N/2))` slots per side without changing configured slot notional.
We retained risk accounting that does not presume an exit filled, because
doing otherwise would hide the exact temporary over-allocation the repair must
surface.

Finally, the terminal-unresolved bound was changed only after a pre-result
reason audit. Missing terminal prints represented 88.8999% of unresolved
terminal notional (19 of 22 positions), so a point-in-time terminal liquidation
test was mostly a data-end artifact. The gate now measures every evaluation's
mean daily unresolved-or-stale inventory below 2% of NAV; terminal count,
notional, and reasons remain visible. No gross, name, net, target, fold, seed,
roster, cost, leakage, identity, or protected-access rule was changed.

## 14. Pass-4d final disposition: what was accepted, rejected, and deferred

The final implementation follows pass 4d's semantic split rather than either
earlier proposal wholesale. It keeps the multiday refactor's identity,
information-clock, economic-return, and stateful-ledger boundaries, but rejects
the proposed wealth bridge. The observed restart audit showed no F1/F2/F3
evaluation loss, so bridging would have introduced unobserved continuity for
no current research benefit. Exact endpoint availability, raw-chain restart,
and unresolved-action checks are instead shared by slow wealth features and
naive baselines. Resolved actions remain in the wealth index and do not cause
double masking.

The old all-purpose corporate-action mask was removed completely. Raw-price
cross-session intraday fields now receive the stricter unresolved-or-unit-change
guard, including successor conversions. Same-session scale-free fields receive
no unnecessary action guard. Targets and execution retain their own already
aligned economic contracts. Store, evaluator, and validation schemas were
advanced because this is a real semantic break; compatibility aliases and
parallel legacy paths were deliberately not kept.

The ledger adopts same-auction slot reuse only after a complete exit order is
submitted at that close. Older pending exits still consume slots, and a failed
exit plus filled replacement is exposed to the existing next-day risk trim.
This preserves honest state and costs. `K_eff = min(K, floor(N/2))` handles
small candidate sets without changing configured slot size, gross target, or
the 5% name cap. Per-book mean exits, same-close replacements, unresolved/stale
daily exposure, terminal inventory, and reason breakdown are all persisted.

The rebuilt V3 store is
`D:\quant-data\b3\processed\v2_daily_store_8021e42_20260906T202315Z`,
manifest SHA-256
`deb9ca8449c5b9a83bf25ac19218069c006e183361b6f6ba836717ade63b491b`.
It peaked at 7.138950 GiB RSS, passed every data and protected-access gate, and
restored 98.22%--99.94% naive baseline coverage. Its independent native-fast
audit SHA-256 is
`98d2e5346a0e2555b77fdb0cb034c63e9d8fa7860eb30f6e749680883ba29218`.

The full scratch acceptance root is
`D:\quant-data\b3\processed\model_runs\v2_development_acceptance_8021e42_20260906T184700Z`;
pipeline and log-inclusive inventory SHA-256 values are
`f7ce5a47a5be4bc8d8f4e11b5adde83a97c6e334b87038068f0f56a7ba7beaa6`
and `2caf60d4d3202fa3228b295aa3245d04bf6bf92dc197a00264dbdab9d3672bf2`.
The action-mask and slot repairs raised gross utilization into the registered
band for 13 of 16 books. Three still failed gross, and nine books failed the
pre-result requirement that mean daily unresolved/stale marked notional be
below 2% of NAV. F3 is the clearest remaining problem: all five naive books
fail the unresolved/stale gate, with means from 3.90% to 10.92%. The F1 GBDT
also fails at 2.45%.

Accordingly, the conditional suggestions to merge `main` and launch Round 1
and Round 2 were not implemented. Doing so would violate the source documents'
own acceptance stop. We also did not relax gross, the daily unresolved bound,
the name cap, score masks, or ledger accounting after seeing the result, and
did not retry any score. The complete engineering work is pushed on
`fix/v2-development-grade-data` for independent review; `main` intentionally
remains at the last accepted state. Official validation and test data were not
read, no deployment changed, and no paid Lambda instance was required.
Two direct provider inventories at 2026-09-06T21:55:41Z each returned zero
instances.

## 15. Pass-4e disposition: remove the latch, retain the failed gate

Pass 4e's diagnosis-first ordering was implemented exactly. Before touching the
ledger, the sealed `8021e42` store and acceptance were replayed only far enough
to classify every held name-day in the unresolved/stale gate. The audit found
2,502 such name-days and confirmed the proposed code mechanism: every row
carried the old position-life `unresolved_action` latch. Eighty-four rows traced
to a flag origin followed by a later print. It also ruled out the document's
conditional data-rebuild triggers: no observed held session had a false
retrospective action mask, no inferred-term/DISMES defect appeared on an
observed cell, and the evaluator used the retrospective rather than the
decision-known alignment. The store and coverage contract were therefore not
rewritten. This choice follows the addendum's explicit conditional, avoids a
result-equivalent rebuild, and preserves the sealed V3 source identity.

The ledger change adopts the source document's accounting argument in full.
Uncertainty is recomputed for the current session and no longer transfers or
latches with a position. It can suppress only a new entry. A positive observed
close fills an ordinary exit, risk trim, or terminal liquidation even if the
current action term is unresolved. A booked cash receivable/payable remains an
accounting claim but does not lock the associated shares. The evaluator now
rejects anything except the manifest-labelled retrospective outcome/accounting
arrays; the decision-known alignment remains confined to model features. The
union exposure gate remains unchanged, with two new daily components exposing
current unresolved-claim inventory and stale-mark inventory separately.

The suggested rebuild was not implemented because its stated trigger did not
fire. The suggested conditional merge and Round-1/Round-2 execution were also
not implemented because their stated prerequisite did not pass. The replay
reused all 16 sealed score panels and proved every non-ledger field bit-identical,
but only F1 momentum newly entered the gross band. F1 and F2 inverse-volatility
remained at `1.786278` and `1.741195`, and 9 of 16 books still exceeded the
unchanged 2% daily unresolved/stale bound. F3 momentum and the F3 blend remained
the largest failures at `10.4513%` and `10.9910%` of NAV. In every book the two
new components equal the union: current inferred-action uncertainty on held
inventory occurs on exactly the same no-print sessions as the stale mark. Thus
the old latch was a genuine defect, but removing it demonstrates that the
remaining gate failure is real stale/no-print exposure rather than an exit that
was forbidden when a later print existed.

The implementation is commit `eca09d6851074e79d7ffb7f1d4e9a11c022c3a02`.
Ruff, compilation, and all 861 tests passed. The diagnostic and replay roots,
respectively, are
`D:\quant-data\b3\processed\model_runs\v2_pass4e_unresolved_diagnostic_24abe56_20260906T222757Z`
and
`D:\quant-data\b3\processed\model_runs\v2_development_acceptance_pass4e_eca09d6_20260906T225238Z`.
Their log-inclusive inventory SHA-256 values are
`0f93b5e660c7701cfec392b3a92ff25fdb7eaccd212c270deffa91c0d07ed709`
and `49d6b2e0d7bbcbbdd4402b676a9e5891c81004f325928d440cd985c28cc1b534`.
No target, fold, seed, candidate roster, cost, gross/name/net cap, leakage rule,
identity rule, or protected-access boundary changed. The branch remains the
review surface; `main` and deployment intentionally remain unchanged.
Two final provider reads at `2026-09-06T22:57:42.9989658Z` and
`2026-09-06T22:57:45.8155995Z` both returned zero instances, confirming that
the pass neither used nor left paid compute running.

## 16. Pass-4h registered continuation: eligibility hysteresis and gross labels

Pass 4g subsequently proved all direct entry, fill, slot-consumption, and cap
signatures D1--D4 clean. F2 inverse-volatility's shortfall was 94% sizing and
only 6% occupancy, so its low gross is not repaired: it becomes the explicit
`gross_underdeployed` label. The only P0 signature was transient eligibility
loss followed by prompt re-eligibility inside the retention band.

Pass 4h therefore adds one bounded policy rule: an existing position holds
through at most five consecutive ineligible sessions, resetting on eligibility
or closure. Session six instructs `ineligible_hold_exhausted`; entries remain
eligible-only, and the ten-session no-print settlement path retains precedence.
No K, buffer, cap, expiry, sizing, trim, cost, borrow, settlement, clock,
identity, or leakage rule changes.

The 1.8--2.2 gross band is now a reporting label rather than a standalone stop.
Departures are `gross_underdeployed` or `gross_overdeployed` when the exact
decomposition is present, D1--D5 are zero, and mean gross stays inside the hard
1.5--2.25 interval. A signature or hard-bound breach still stops. The 2%
stale/unresolved bound and 15% settlement-incidence label are unchanged. This
contract was registered before its replay and before any Round-1 score.

## 17. Final disposition for the proposing reviewers

This section is the compact explanation to feed back to the LLMs that proposed
the refactors. Sections 1--16 above retain the evidence and decisions made before
the usage reset; this section consolidates them with Pass 4h and Round 1.

### What was implemented

| Area | Final choice | Why |
| --- | --- | --- |
| Identity | Permanent security identity, dated ticker attributes, audited same-ticker ISIN succession, and no filename-wide identity assignment | This preserves history without merging unrelated listings and makes every transition inspectable. |
| Information clock | Consumer-specific clocks and masks: publication-lagged sidecars, decision rows, target/outcome accounting, and execution each use their own causal boundary | The former all-purpose action mask silently mixed feature availability, economic truth, and execution. |
| Economic returns | Shareholder wealth uses observed endpoints and explicit action terms; raw-price cross-session fields also reject unit changes; same-session scale-free fields do not inherit an unnecessary action guard | This keeps splits and resolved actions economically meaningful without double masking. |
| Store | Family-at-a-time float32 memmaps, float64 only within bounded computation, row-wise rank-gauss, streamed targets, and one adjusted-OHLC memmap | This removed the full-panel peak-memory failure without changing values at research precision. The accepted store peaked at 7.138950 GiB. |
| Sidecars | Unconditional contemporaneity reconstruction, name-clustered liquidity-stratified coverage audit, and explicit source-missing capabilities | Leakage is guarded by reproducible availability. Composition is reported with support-aware inference. Missing archives never become zero-valued observations. |
| Model boundary | Every tensor reaching the model is finite; invalid cells are zeroed under masks; masked reductions use `where`; zero-history and empty-fast-patch cases are structural cases | This fixes NaN propagation at the loader/model contract instead of dropping failed predictions in the evaluator. |
| Ledger | Stateful pending orders and fills, partial risk trims, 20% absolute-net cap, same-auction slot reuse after a complete exit order, `K_eff` for small universes, current-session action uncertainty, ten-session labelled terminal settlement, and five-session transient-ineligibility hold | Each rule followed a sealed diagnosis and addresses a distinct accounting or churn defect without changing scores or inventing fills. |
| Acceptance | D1--D5 exact defect signatures; hard mean-gross interval 1.5--2.25; 1.8--2.2 as a label; mean stale/unresolved below 2%; settlement incidence separately labelled | The decomposition showed that modest gross drift can be a truthful consequence of fixed entry sizing, while signature breaches remain real engineering failures. |
| Research hygiene | Fresh immutable roots, hash-bound provenance, separate build/acceptance/freeze identities, no completed-candidate retry, false protected-access flags, and no deployment change | Operational recovery remains distinguishable from a result-changing retry. |

### Suggestions deliberately not implemented in full

- Verified corporate-action terms and a verified auction schedule were not
  claimed because those archives do not exist in the available development
  data. The pipeline accepts only the explicit
  `inferred_cotahist_dismes_v1` / `reconstructed_v1` tier and propagates those
  labels to every artifact. This is the shortest honest route to a number, not
  a substitute for later source acquisition.
- The proposed wealth-index bridge across raw-series restarts was rejected.
  The pre-change audit found zero F1/F2/F3 evaluation loss from those restarts;
  a bridge would manufacture unobserved continuity. Exact endpoints remain
  required, and restart incidence is recorded.
- Universe hysteresis, resize-to-target, routine rebalancing, altered K/buffer,
  and entry reach-down were deferred. The signed gross decomposition and D1--D4
  tests did not justify them, and adding them would change the registered
  strategy rather than repair a defect.
- The old position-life unresolved-action latch was removed, but the store was
  not rebuilt for that pass. The sealed diagnosis proved the current
  retrospective accounting arrays were already correct and only the ledger
  consumer was wrong.
- The legacy v1 fast checkpoint is not a canonical Round-2 input. Under the rev-2
  chronology audit it represents contaminated transfer history. Round 2 is
  registered with fresh native fast weights for every arm; no compatibility
  fallback remains.
- Options, rebalance, events, and fundamentals were not fabricated merely to
  make rung D rectangular. The sealed store explicitly records them as source-
  missing. Rung D consumed the materialized lending and oddlot groups and
  retained the unavailable capabilities as zero-dimensional absences. This is
  different from imputing a missing observation and is enforced by the store
  capability manifest.
- The official-validation and permanently spent test windows were not opened.
  The current claims are development-window claims only.
- Round 2 was not started. Pass 4h explicitly requires a user go-ahead after
  Round 1 and a disposable one-job smoke before any registered GPU trajectory.

### Pass-4h and Round-1 evidence

Pass 4h passed all exhaustive acceptance conditions. The accepted replay at
`D:\quant-data\b3\processed\model_runs\v2_development_acceptance_pass4h_384fd81_20260907T021757Z`
has manifest SHA-256
`97a1ed757e87e1e19f2c46999437b31f436470f2dae49d2ce0c5782635607f8f`.
All 16 reused score panels kept non-ledger fields bit-identical; D1--D5 were
zero; every mean gross lay inside 1.5--2.25; every stale/unresolved mean was
below 2%; and protected access was false/false. F2 inverse-volatility is
properly labelled underdeployed rather than silently rescaled.

Round 1 is sealed at
`D:\quant-data\b3\processed\model_runs\v2_round1_81fe0cb_20260907T023339Z`.
Result and complete-inventory SHA-256 values are
`ca39f11340f13956dca11da7fb6b3a8fa0a38f8a79cb558387d81aff26bf57a0`
and `b5084c817fc478c2759d962af12e282c292cade6f8e715aebed290ba5500ce08`.
The complete audit verifies 696 files, 33 score manifests with 66 array
payloads, 18 model manifests with 450 LightGBM models, and all 88 access-flagged
JSONs.

The GBDT ladder's pooled primary ICs were A/B/C/D =
0.052871/0.055264/0.053795/0.032725; headline net excess was
6.435/5.430/5.364/1.866 bps/day. The fixed preference rule keeps A and B,
drops C and D because each later step has both a negative IC point delta and a
negative economics point delta, and designates `b_intraday` as the Round-2
parent. The naive inverse-volatility floor has higher IC (0.062256) but is a
control, not a ladder candidate. The data-span preview is intentionally
non-binding: fine/decay/uniform ICs were 0.055264/0.053459/0.053680.

### Prepared Round-2 handoff; not executed

Before a future freeze, the exact sealed Round-1 root and the 4.531-GiB V3
store must be copied byte-for-byte to the `brazil-rv-east3` NFS namespace and
their result/inventory/store hashes reverified. No legacy fast checkpoint is
passed. On an explicitly authorized GH200, with a clean checkout at the then-
current commit, the prepared sequence is:

```bash
cd /home/ubuntu/Brazil-RV/quant/b3-quant
export BRAZIL_RV_DATA_ROOTS=/home/ubuntu/Brazil-RV/quant/b3-quant/research/configs/v2/data_roots.lambda_us_east_3.json
ROUND2_ROOT=/lambda/nfs/brazil-rv-east3/quant-data/b3/processed/model_runs/v2_round2_<commit>_<utc>

uv run --project research python -m brazil_rv.v2.research_rounds freeze-round2 \
  --round1-root /lambda/nfs/brazil-rv-east3/quant-data/b3/processed/model_runs/v2_round1_81fe0cb_20260907T023339Z \
  --store /lambda/nfs/brazil-rv-east3/quant-data/b3/processed/v2_daily_store_8021e42_20260906T202315Z \
  --cdi /lambda/nfs/brazil-rv-east3/quant-data/b3/interim/external/cdi_sgs12_v2_development_20260903T190000Z/daily_cdi.parquet \
  --cdi-sha256 a60147d598ffabea13a64228e3ec3f18beee7956b8ddca9e0d725cdc63250d23 \
  --experiment52-cdi /lambda/nfs/brazil-rv-east3/quant-data/b3/processed/model_runs/execution_c0_52_e380cd7_20260827T164738Z/cdi/daily_cdi.parquet \
  --experiment52-cdi-sha256 38d7934ebb6849b1e20310a81130f6faa68f4618c8edb0d046185067c83da2ea \
  --output-root "$ROUND2_ROOT" --max-parallel 4

uv run --project research python -m brazil_rv.v2.research_rounds write-round2-plan-smoke --output-root "$ROUND2_ROOT"
uv run --project research python -m brazil_rv.v2.run_many \
  --plan "$ROUND2_ROOT/round2_plan_smoke.json" \
  --manifest "$ROUND2_ROOT/round2_smoke_launcher_manifest.json"
```

That generated plan is exactly Arm A, F1, seed 11, one epoch, maximum one
process, and no score output directory. Only after its completed manifest is
finite and the root still contains no smoke `scores/` directory may the three
Stage-P inputs be materialized:

```bash
uv run --project research python -m brazil_rv.v2.research_rounds write-round2-plan-p --output-root "$ROUND2_ROOT"
uv run --project research python -m brazil_rv.v2.run_many \
  --plan "$ROUND2_ROOT/round2_plan_p.json" \
  --manifest "$ROUND2_ROOT/round2_stage_p_launcher_manifest.json"
```

The Stage-P plan contains seeds 11/29/47, native fresh fast weights, no external
sidecars because the designated parent is `b_intraday`, at most three active
jobs under the registered max-parallel four, and the exact 20-epoch/patience-3,
lookback-60, eight-pair, lambda-zero contract. No plan or Round-2 root was
created during Pass 4h.
