# V2 multiday pipeline and execution audit

Audit date: 2026-09-08. Baseline: `b6a0da3abf0095e266bc0d60c53ce9fcf99dbd75`,
which matched GitHub `main` when the audit began.

## Conclusion

There are credible restrictions on what the model can learn and monetize, but
the largest immediate problem is the validity of the economics. The hedge uses
a normalized feature as an economic beta, and two execution paths let information
from after 15:45 affect trades attributed to that decision. Fix these before
interpreting another portfolio sweep. Their aggregate effect could have either
sign; this audit does not establish that the reported edge is understated.

The strongest measured model restriction is missing input information: **eight
of the twenty current intraday scalar features have zero valid active-name
observations throughout 2024**. Another has only 21 valid name-days out of 46,627.
This is a source-unit/support problem, not evidence that those features lack alpha.

Changes accompanying this report remove unused code and obsolete rejection
machinery. They do **not** change the accepted model, ledger policy, feature
semantics, or frozen scores. The defects below remain repair work; an audit finding
is not a backtest correction or a measured improvement.

## Scope and evidence

Read the supplied catchup, current project context, registered research definitions,
and implementations spanning source assignment, universe, schedule, corporate
actions, features, normalization, store, data adapters, targets, model, loss,
training, GBDT, scoring, selection, evaluation, lending, BOVA11, and the ledger.
The attachment's proposed Round 3 was treated as background, not a request to run it.

Canonical store examined:
`D:\quant-data\b3\processed\v2_daily_store_8021e42_20260906T202315Z`.
Manifest SHA-256:
`deb9ca8449c5b9a83bf25ac19218069c006e183361b6f6ba836717ade63b491b`.
The manifest identity was verified. Array payloads were read through the store's
date authorization for **2024 inputs only**, without rehashing every large array.
Additional evidence consists of the store's 2024 source-quality audit and one
dated PETR4 M1 source sample. No target/score arrays, new performance results,
official validation, or test-period payloads were accessed for this audit.
Manifest, date-axis, and source-audit metadata describe the full store; those
metadata reads are distinct from opening held-out model or outcome arrays.

[Machine-readable evidence](v2_multiday_audit_evidence.json) records the input
coverage, synthetic execution demonstrations, source example, and baseline identity.
The synthetic examples use `_run`/`_config` from
`research/tests/test_v2_stateful_ledger.py`; they are not estimates of real P&L impact.

## What the current system actually does

| Component | Current contract |
|---|---|
| Identity and universe | Permanent ISIN and dated M1 source assignments; causal daily liquidity/activity/price/history eligibility. The broad store has 933 historical names, not 933 simultaneous constituents. |
| Decision and outcome | One 15:45 snapshot; completed daily information through t-1 and completed same-day minutes before 15:45. Five close(t)-to-close(t+H) horizons: 1, 2, 3, 5, 10 sessions. |
| Slow branch | 60 sessions by 32 base daily features, masks and ages, encoded by a width-64 GRU. Optional sidecar capabilities are separate from what a selected arm enables. |
| Fast branch | Fresh native TCN over up to 69 five-minute patches, seven channels with masks. Accepted M1 assignments cover 158 names; the current active-name fast-presence rate is about 60% in 2024. |
| Fusion | Current intraday scalars, slow state, gated fast state and pooled cross-sectional slow mean/std; two residual fusion blocks. Fast/pool gate biases start at -2. |
| Learning | Soft-Spearman multihead objective, SAM-AdamW; optional to-close/persistence objectives are inactive in the current parent. Stage P uses long daily history; Stage F adds intraday inputs. |
| GBDT | Per-horizon LightGBM regressors over the shared current scalar view, selected by daily IC; seed rank ensembles. The designated Round-1 parent is `b_intraday`. |
| Development evaluation | Three chronological half-year folds: H2 2023, H1 2024, H2 2024. Fit, 10-session purge, 55-session selection, 10-session purge, evaluation. |
| Trading signal | Rank aggregation over D1/D2/D3/D5. D10 is excluded from the primary book and primary IC. D5-only is already a comparator. |
| Headline book | 30 slots per side, 2x target equity gross, five volatility strata, retention buffer, 4 bps per side, constructed borrow, CDI cash accounting and a BOVA11 beta hedge. |

Round 2's recorded neural Arm B has neutral IC 0.0247766 and net excess 2.4040
bps/day. The equal-rank network/GBDT ensemble has IC 0.0265307 and net excess
1.2261 bps/day. These are inherited development results, not recomputed here;
their uncertainty and the execution findings below prevent a stronger claim.

## Findings requiring repair before new economics claims

### 1. The BOVA11 hedge consumes rank-Gauss scores as beta coefficients — P1

`feature_spec.feature_specs` declares `beta_60` as `rank_gauss`.
`build_store.consume_slow_feature` applies that transform to `slow_values`.
`validate_pipeline._evaluation_inputs` extracts this transformed column and
`evaluate.evaluate_scores` supplies it as `beta_60` to the ledger. The ledger
then computes `sum(signed_equity_notional * beta_60)` and hedges its negative.

A rank score cannot serve as an economic regression coefficient. With synthetic
short/long notionals -1/+1 and actual betas 1.0/1.2, the hedge is -0.20 NAV.
Using their two-name normal scores -0.67449/+0.67449 instead produces -1.34898
before the cap and **-0.60 NAV after it**. Nearly half of valid active-name stored
beta features are negative in 2024, as the rank transformation implies.

There are two related defects. The evaluator drops `slow_valid` for this input,
so invalid zero-filled feature cells become apparently finite betas. Also, the
raw feature beta is estimated against a median-equity return, the realized-beta
diagnostic uses an equal-weight equity benchmark, and the hedge trades BOVA11.
Even recovering the raw feature beta would not align those benchmarks.

Repair: provide a separately named, lagged, raw beta against the actual hedge
return series, with validity and source identity. Keep its model normalization
separate. The current store does not retain the raw beta alongside `slow_values`,
so use a hash-bound derived sidecar or a new store; do not invert the ranks or
silently change an existing manifest. Reconcile hedge notionals and exposure
diagnostics before re-evaluating economics.

### 2. Hedge sizing uses the same future close at which it fills — P1

In `execution/stateful_ledger.simulate_stateful_ledger`, equity intentions are
created before close observations. The hedge block runs **after equity fills and
mark updates**, builds planned exposure from the resulting shares/current marks,
then sizes and trades the hedge at that same day's BOVA11 close. Hedge activity
also bypasses the equity `IntendedOrder`/`Fill` records.

Synthetic demonstration: keep signals, starting references, beta, and BOVA11 price
fixed. Change only the first day's later long-equity close from 10 to 12.
The day-zero equity intentions are identical, but hedge target changes from
-0.20 to -0.44 NAV. A 15:45 order cannot be contingent on those later realized
marks/fills unless a separate, later execution decision is explicitly modeled.

Repair: freeze hedge intentions using decision-visible marks and planned equity
orders, and account for subsequent fill imbalance. Alternatively model a later
hedge decision with its own executable price and timestamp. Include the hedge
in intended-versus-realized reconciliation. Add a same-day future-mutation test.

### 3. Retrospective inferred actions alter historical intended orders — P1

The builder explicitly stores retrospective action arrays for outcome/accounting.
`validate_pipeline._evaluation_inputs` requires this alignment. In the ledger,
however, `has_action` is processed before order construction: it cancels orders,
converts positions, and adjusts decision reference prices by share/cash terms.

`infer_cotahist_action_terms` derives terms from the event day's close/activity and
DISMES and timestamps them at 23:59:59 local time. Retrospective alignment removes
the decision-time availability restriction. That is appropriate for later outcome
reconstruction, but not for determining what a 15:45 decision knew.

Synthetic demonstration: two names, unchanged signals and past prices of 10,
day-one DISMES change. Changing the later close from 10 to 9.6 changes inferred
cash from 0 to 0.4. Although both terms are marked available at 23:59:59, the
15:45 long order's reference changes from 10 to 9.6 and quantity from 0.1 to
0.1041667. This is an end-to-end decision leak, not merely a documentation issue.

Repair: separate decision-known terms/references from retrospective realized
accounting. Use the existing decision-known alignment for decisions, with a
well-defined treatment of unknown terms; preserve retrospective terms only for
realized economics. This needs a joined builder/evaluator/ledger causality test,
because tests that mutate prices while holding inferred terms fixed miss it.

### 4. Intraday feature coverage collapses at the source-unit/support boundary — P1

The 251 sessions in 2024 contain 46,627 active name-days, averaging 185.76 active
names (range 173–198). Measured valid fractions:

| Feature(s) | Active-name availability |
|---|---:|
| Overnight return; its 5/20-session sums; overnight-minus-intraday; its 20-session mean | 0% each |
| Lagged final-30-minute return share; lagged close/VWAP deviation | 0% each |
| 15:45 volume relative to prior-20-session median | 0% |
| 20-session intraday-return sum | 0.045% (21 name-days) |
| 5-session intraday-return sum | 25.53% |
| Current intraday return | 54.63% |
| Current VWAP deviation; lagged last-hour volume share | 23.00%; 21.91% |
| 1/5/20-session intraday realized volatility | 60.03%; 50.80%; 41.73% |

`intraday_features.replace_daily_close_anchors` requires an exact observed final
continuous-session M1 close and an absolute log mismatch to COTAHIST <=0.005.
The existing 2024 audit reports 29,750 mismatches among 33,261 comparable name-days
(89.44%), with median absolute log mismatch 0.12214. After active membership,
only 0–17 names per day pass the consistency mask; **no day reaches the 20-name
normalization support threshold**. This directly explains the collapse of the
close-anchored fields. As a concrete source example, PETR4 on 2024-01-02 has a
17:54 M1 close of 27.35 and COTAHIST close of 37.78. This establishes incompatible
levels, without assuming an unverified explanation of the broker's adjustment method.

Separately, `stream_intraday_from_assignments` sets activity validity from printed
observations, and the scheduled feature builder requires complete activity
prefixes for VWAP/volume aggregates. The relative-volume feature requires complete
prefix support across the current session and twenty prior sessions. Rolling
returns also require every constituent daily endpoint to be valid. Those
conditions compound missingness and are then subject to the same 20-name
cross-sectional threshold. The exact modal opening minute can change by date,
making endpoint and prefix definitions particularly consequential.

Repair source units first. Investigate a causal M1/COTAHIST unit bridge or an
internally consistent M1 close-to-open measure, preserving corporate-action and
identity boundaries. Do not relax the price-unit tolerance to manufacture valid
overnight returns. For activity, distinguish a proven no-trade minute from a
source outage before assigning zero volume; never fill missing OHLC. For rolling
summaries, compare an explicitly defined minimum-support estimator with its
coverage/age metadata. Rebuild derived inputs and retrain for any semantic change.

## Model and research choices that may suppress useful signal

These are hypotheses to test chronologically after the validity repairs, not
automatic changes justified by the current P&L ranking.

| Finding | Why it matters | Small useful comparison |
|---|---|---|
| D10 receives one fifth of the head-average loss but is absent from primary selection/book | Shared representation optimizes a horizon not rewarded by the deployed research objective. It could regularize beneficially or dilute D1–D5. | Five-head baseline versus D1–D5-only loss, with comparable updates and fixed data. |
| Heavy characteristic neutralization followed by volatility quotas and a beta hedge | Targets already residualize on volatility/beta/liquidity characteristics; the portfolio imposes further factor restrictions. This deliberately excludes some predictable return components. | Distinguish residual-alpha prediction from implementable total-return prediction; evaluate one target/book change at a time with matched risk. |
| Rank-Gauss removes levels and cross-sectional dispersion | Magnitude and common regime information cannot be recovered just by pooling ranks. Common-state return/volatility/dispersion arrays already exist, but are audit-only. | Add only those three causal common-state fields under a registered ablation. Keep absolute price scale out. |
| Stage P transfers more than genuinely trained intraday parameters | The fast encoder is excluded, but current-feature projections and fusion/gate parameters can be transferred from a phase without their real intraday inputs; transferred parameters receive 0.3x learning rate. | Transfer slow/pooled components only, reset intraday-dependent paths; inspect gate activation and gradients by branch. |
| Fast/pool gates begin near 0.119, and intraday inputs are sparse | Negative gate bias plus missing inputs and slower adaptation can make the model underuse the new branch. This is not proof of collapse in the trained checkpoint. | Measure gate distributions by fold and presence; change initialization only if supported. |
| Intraday-selected architecture/training defaults remain shared with v1 | V2 imports TCN architecture and optimizer/SAM/patience constants from `modeling.contract`; the original tuning problem had another horizon and sample structure. | Bounded tests of SAM strength, learning rate, and stopping stability after repairing inputs. Avoid a large architecture search. |
| Rank correlation weights the whole cross-section, while the book trades tails and pays asymmetric costs | A higher mean IC need not imply better returns for the selected names or cheaper shorts. | Report tail spread, tail stability, realized holding time, turnover, and borrow by score bucket; then compare a tail-aware objective if warranted. |
| Candidate designation is IC-first, not a joint persistence/economics optimizer | `_weighted_candidate_designation` selects the eligible IC leader; economics overrides only with a positive paired interval and an IC interval including zero. Persistence is reported but is not an input to the decision. | State the rule accurately. Register a different decision criterion before using new results; do not retroactively switch to the current P&L winner. |
| GBDT and neural objectives have different implicit date weights | GBDT fits pooled security rows; neural loss normalizes cross-sections per date/head. Larger-universe dates can carry more GBDT weight. | Test per-date sample weighting for GBDT on the same feature/support panel. |

The native fast branch also inherits the accepted 158-name XP source coverage.
Keeping the broad daily universe is correct, but missing-fast status can carry
provider/collection selection effects as well as ordinary data quality. Existing
survivor/delisted coverage audits are useful diagnostics, not proof of historical
provider availability. Compare fast-present/absent performance with matched causal
liquidity/history support before generalizing fast-branch results to the whole
universe. Do not narrow historical eligibility to today's surviving M1 names.

Stage P's internal split also excludes 70 sessions, versus a maximum label horizon
of 10. Shared historical lookback observations alone are not future leakage.
An interval-based purge may recover usable training data, but this is a smaller
priority than inputs or economic correctness. Keep actual label-boundary purges.

## Execution policy: where edge might be lost, and where it may be overstated

### Selection, retention, and execution timing

The five volatility strata receive six long and six short slots each. Strong
signals cannot freely consume unused capacity in another stratum. Retention
also uses current stratum ranks, so crossing a volatility boundary can cause
turnover without a comparable change in alpha. The book uses slot-based entry
notionals and rank retention rather than an expected-return-minus-cost trade rule.
Borrow cost is charged after selection instead of entering the selection objective.

Test the current stratified policy against global ranks with a soft volatility
constraint and matched gross/net/name limits. Compare D3/D5 or a calibrated
holding-horizon blend against the current equal D1–D5 ranks. Do not prescribe a
five-day minimum hold without testing signal decay: retention can help costs
but can also preserve a deteriorating signal. D5-only already exists in evaluation.

**Buffer sensitivity is confounded by changing capacity.** `_scaled_group_bands`
scales entry quota as well as retention width when a stratum is small. With five
strata of 34 names, k=30 and buffers 0/30 produce six entries per stratum; buffer
60 produces five. Thus a purported wider-buffer comparison can change 30 slots
to 25. The actual 2024 universe gets as small as 173 names. Keep k/gross/eligible
support fixed for a clean buffer comparison or explicitly report the capacity change.

Pending entries can survive for three sessions without refreshing an unheld
name's original signal. A fresh opposite signal need not cancel that old intent.
Test daily cancellation/re-ranking with separate accounting for filled holdings;
otherwise stale entries can monetize a signal that no longer exists.

The 15:45 snapshot is followed by a materially later close. This may discard
short-lived fast alpha, but executing earlier changes the tradable return target.
Compare close execution with a predeclared, actually executable post-decision
schedule only after obtaining appropriate price/volume data. Moving to a later
decision likewise requires rebuilding the feature snapshot and respecting dated
auction cutoffs. A COTAHIST close and fixed 4 bps charge are not proof of auction
capacity, fills, or executable prices at arbitrary size.

### Borrow, cash, and hedge instrument

The constructed lending source uses the latest recent rate or a cross-sectional
75th percentile; a past positive balance can keep a name eligible until updated.
Before the first causal rate event it uses a flagged 2% rate and makes names
shortable. These are explicit modeling assumptions, not historical locates.
They can be pessimistic for unavailable expensive rates and optimistic about
short availability. Removing the constraints would not establish a better strategy.

The ledger accrues annual borrow linearly as `rate / 252` and reprices from the
daily constructed panel. B3 specifies compounded annualized rates on a 252-business-
day basis and contract attributes; published daily rates are aggregates and may
carry the last published rate when no trade occurs. Contract-basis accounting and
renewal assumptions therefore need to be explicit. At 100% annual, the linear
one-day charge is 39.68 bps versus 27.54 bps for `(1 + rate)^(1/252) - 1` on the
same principal. This demonstrates potential overcharging, especially for expensive
shorts, without estimating the portfolio effect. Use dated historical fee terms
rather than applying today's fee schedule to earlier years.
[B3 securities-lending contract and rate methodology](https://www.b3.com.br/pt_br/produtos-e-servicos/emprestimo-de-ativos/informacoes-8AE490C964B9A9BC0164CCE5F2C9746B.htm).

BOVA11 shorts currently use the configured 2% floor when no instrument-specific
rate panel is supplied; they do not receive the equity borrow fee treatment.
After fixing beta and timing, compare unhedged, corrected BOVA11, and a separately
specified futures hedge. Futures require their own basis, financing, rolls, margins,
and variation accounting; they are not a costless replacement.

Full CDI remuneration of short proceeds is already in the current headline,
with sterile proceeds as a comparator. Do not count adding it as a new improvement.
The debit spread defaults to zero. These simplified cash assumptions should be
tested against the intended financing arrangement before making an executable claim.

### Economic reporting and terminal assumptions

`ledger_configurations` builds `cost_*_borrow_*` cases from a different strategy:
uniform borrow, no volatility balancing, and no beta hedge. Those cases are not
a clean cost stress of the headline. Add a headline-preserving cost grid when
repairing evaluation; retain a differently labeled strategy comparator only if useful.

Each development fold starts fresh and liquidates at its end. At 2x equity gross
and 4 bps per side, a full opening/closing round trip is roughly 16 bps per fold,
or 0.13 bps/day over 126 sessions, before hedge effects. Boundary drag exists,
but that scale alone does not explain a multi-bps/day difference. Use a causal
warm-start or longer contiguous evaluation only under a documented fold contract.

After ten sessions without a print, terminal inventory can be settled at a
declared mark instead of an observed executable price, and settled names remain
blocked even if they later print. The last-mark headline and haircut comparator
do not establish actual terminal wealth. Review incidence and per-name attribution
before crediting the P&L. This can overstate recovery and also suppress subsequent
opportunity; it is not uniformly conservative.

## Research speed and code cleanup

Useful future speed work, without changing the information set:

- `DatePairBatchSampler` visits interior dates twice even when persistence weight
  is zero. A one-date sampler needs an update-count/microbatch comparison so that
  faster training is not confused with halving training exposure.
- Slow-branch computation includes all 933 historical names although only about
  186 are active on a typical 2024 date. Compact current active names while
  retaining each one's causal history and a stable pair-wise identity mapping.
- The dataset reads seven target-family payloads, including reporting diagnostics,
  for every training sample. Virtual characteristic-neutral targets are also
  recomputed on reads. Separate training payloads from reporting and cache only
  within the authorized date/target contract. No speedup is claimed without profiling.

Removed in this change after checking call sites:

- Unused stateless `execution/daily_swing.py` and its dedicated tests.
- Retired fixed-grid intraday builder and its exclusive return/volatility/skew/
  spread helpers. Applicable causality, action-boundary, anchor, and mask tests
  now exercise the active scheduled implementation. The negative-covariance
  spread test now exercises its active moment-based estimator.
- Unused V2 block-parity split/mask/stitch helpers and tests; current chronological
  folds and active seed-ensemble code remain.
- `preset_jobs`, which only raised an obsolete refusal, plus its unusable CLI
  switches and historical phase-name rejection. `run_many` takes a plan and manifest.
- Unused alternate GBDT feature wrapper, unused date lookup, and masking entries
  for sample keys no longer produced.
- Enumerated legacy store-schema and array-name rejection lists. The single
  current-schema boundary, required arrays, identity/shape/mask checks, and source
  hashes remain because mixing incompatible artifacts can silently corrupt research.

V1 is an explicitly retained research program. Its active implementation, shared
layers, and explicitly selectable historical diagnostics were not treated as dead
code merely because V2 does not use them in the current parent. Optional source
producers also remain where they serve a current supported path. Raw data,
canonical derived data, paid compute, historical results, and GitHub were not modified.

## Recommended order of work

1. Repair economic beta, hedge timing, and decision/action separation with focused
   future-mutation and accounting fixtures. Preserve existing results as historical.
2. Reconcile M1 price units and define usable activity/rolling support. Measure
   coverage before training; do not infer feature utility from all-missing inputs.
3. Re-evaluate saved development predictions under the corrected ledger where
   compatible, identifying which prior comparisons survive. Keep pure cost tests
   separate from policy changes. Feature/target changes require retraining.
4. Run a small registered set of horizon/retention/borrow-aware policy comparisons,
   then model ablations for D10, common state, transfer scope, and optimizer settings.
   Keep the current chronological split discipline and held-out boundaries.

## Cleanup verification

The code/test cleanup changes 16 files: 62 inserted lines and 1,256 removed,
for **1,194 net lines removed**. Documentation is counted separately.

From the repository root, using the existing Python 3.12 research environment:

- `uv run --project research --no-sync pytest research/tests -q`:
  **890 passed** in 359.29 seconds, including the production-axis memory fixtures.
- After the final small test/comment and error-message edits,
  `uv run --project research --no-sync pytest research/tests/test_v2_intraday_features.py research/tests/test_v2_run_many.py -q`:
  **27 passed**.
- `uv run --project research --no-sync ruff check research/src research/tests`:
  passed.
- `uv run --project research --no-sync python -m compileall -q research/src/brazil_rv/v2 research/src/brazil_rv/execution`:
  passed.
- `git diff --check`: passed. Removed-symbol search found no remaining source/test
  references. AST comparison confirms every retained intraday function/class is
  unchanged; the feature production path was not rewritten by the cleanup.
- Synthetic execution demonstrations and the stored input-coverage evidence
  reproduced the reported findings. These demonstrations are separate from the
  passing existing acceptance suite: its success does not resolve the audit defects.

The pass-5 verification reran the full suite: 889 passed, with one stale error-message
assertion failing after all of that acceptance fixture's substantive checks passed.
After correcting that assertion, its complete acceptance fixture passed (61.37 s).
Eight additional identity/replay fixtures passed. Ruff and `git diff --check` passed.
No research campaign or saved-score economics replay was run for this cleanup.
