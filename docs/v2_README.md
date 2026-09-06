# Brazil-RV v2 multi-day research pipeline

Brazil-RV v2 is the current one-decision-per-session research pipeline. At
15:45:00 `America/Sao_Paulo` it produces cross-sectional forecasts for
1, 2, 3, 5, and 10 B3 equity sessions. Historical intraday experiments remain
archival evidence; their clocks, tensors, checkpoints, and adjusted-return
semantics do not define this pipeline.

This document is the canonical implementation contract. It describes research
plumbing, not a live trading system, and it does not authorize access to sealed
official-validation or test dates. A fresh real-data rebuild and the acceptance
matrix are still required before results from this contract can support a
research claim.

Where earlier fix-pass notes conflict, the full multi-day refactor semantics
documented here win. Compatible tactical requirements—bounded memory,
decision-prefix causality, revision-safe as-of state, explicit order/fill
separation, and common-population evaluation—remain mandatory.

The schema transition is intentionally incompatible. See
[v2_MIGRATION.md](v2_MIGRATION.md) before opening or resuming any earlier v2
artifact.

## End-to-end contract

The only current path is:

`validated raw observations and revisions -> canonical decision snapshots -> model adapters -> forecasts -> intended orders -> fills -> signed-share/cash accounting -> evaluation`

The priority order is information and economic correctness, then model
interfaces and bounded resources, then controlled research. A finite score or
PnL array is not evidence that identity, timing, actions, or missing outcomes
were handled correctly.

## Decision clock and row meaning

- The decision is exactly 15:45:00 in `America/Sao_Paulo`, converted to UTC
  with historical timezone rules.
- A canonical feature row `t` means information available by that decision.
  Daily market inputs in that row end at `t-1`; publications genuinely
  available by the decision may be current on `t`. The store applies those
  source lags once, and every consumer reads row `t` without another shift.
- Today's market prefix contains only completed bars ending by the decision.
  The bar starting at 15:45, including its open and observation flag, is not an
  input.
- Public data uses its actual release or receipt timestamp and declared
  latency. Date-only publications follow a documented conservative
  availability rule and next-session lags use the exchange calendar.
- Historical rows are historical decision snapshots. Later revisions,
  identities, actions, prices, and source records cannot rewrite them.
- Full-session lagged summaries use the dated continuous/auction schedule.
  There is no fixed historical close and no legacy 405-minute full-session
  assumption.

The session axis is a versioned B3 equity calendar covering warm-up, holidays,
exceptional sessions, and dated session boundaries. Missing source sessions
remain on that axis and are reported as source incompleteness; the number of
observed names is a coverage diagnostic, not a calendar generator.

The decision-time reference price, later entry/fill price, and target
entry/exit marks are separate quantities. Until verified auction marks exist,
targets and close-proxy execution use the explicitly named
`daily_last_trade_close_proxy`.

## Raw data, identity, and actions

Accepted COTAHIST and M1 observations must have supported units and timestamps,
finite positive prices, consistent OHLC bounds, and nonnegative activity.
Exact duplicate economic records may collapse; conflicting duplicates and
structural parse/audit failures prevent promotion. Invalid observations and
their reasons belong in audit output, not in accepted model arrays.

Permanent security identity is the B3 ISIN. Ticker is a dated attribute, and
security, issuer, ISIN, and ticker are not interchangeable. Same-ticker
adjacency may propose a link but never accepts one automatically. A
predecessor/successor conversion requires contractual ratio/cash terms,
effective date, first-known timestamp, and evidence in the verified allowlist.
Each source row is assigned to exactly one security/date segment.

Corporate-action economics use one contractual representation:

- shares received per prior share;
- cash entitlement per prior share;
- effective/ex-date and payment session;
- predecessor/successor claim mapping;
- decision-time availability, source evidence, and resolution status.

Splits change units, distributions create cash receivables/payables, and
payment clears the claim without creating a second gain. `DISMES` and
price/quantity jumps remain diagnostics; realized price ratios and
market-median residuals are not accepted action terms. Unsupported rights,
spin-offs, multi-claim mergers, or unmapped conversions remain unresolved.

A coherent shareholder-wealth OHLC path supplies return features and the
lagged Yang-Zhang scale. For a supported action on session `u`, with
`q_u` post-action shares and `d_u` cash per prior share:

`W_X,u = W_C,u-1 * (q_u * X_u + d_u) / P_C,u-1`, for
`X in {O, H, L, C}`.

This index uses a declared reinvest-at-close convention for features and risk.
The holding target and ledger instead retain distributions as cash or claims;
the conventions are named separately.

Shareholder-wealth history is never bridged across a missing interval. A
return whose endpoint interval spans a wealth-chain restart is invalid by
construction. Each store manifest reports
`wealth_chain_restarts_active_name_days` by calendar year so this latent data
quality condition is visible without inventing observations.

## Point-in-time universe

A name is eligible on decision session `t` using only information available
before `t` when it has:

- trades on at least 15 of the prior 20 exchange sessions;
- prior-20-session median BRL volume of at least R$2,000,000, with verified
  no-trade sessions counted as zero;
- a last observed prior close of at least R$1.00; and
- at least 60 sessions of observed history.

Observed-history age is not claimed to be exchange listing age and includes a
left-censor flag. Eligibility is independent from observation, feature,
forecast, outcome, execution, borrow, and terminal-status masks. Future
outcome availability never changes the universe or an order formed at `t`.

## Immutable daily store

The current store schema is `BRAZIL_RV_V2_DAILY_STORE_V3`. A build writes to
a new staging root, validates it, and promotes only a complete store. Arrays
are uncompressed and memory-mappable. The manifest binds the date, security,
slow/current/fast and horizon axes; source paths and hashes; clock and calendar;
FeatureSpec order and hash; identity/action mappings; target definitions;
array shapes, dtypes, and hashes; coverage; and peak build memory.

Important core arrays are:

| Family | Arrays | Meaning |
|---|---|---|
| Membership/source | `active`, `observed`, raw OHLC/activity | decision-time eligibility and canonical raw observations |
| Action economics | `shareholder_wealth_{open,high,low,close}`, `shareholder_wealth_valid`, `action_*` | coherent wealth coordinates and contractual action terms/status |
| Slow | `slow_values`, `slow_valid`, `slow_age_sessions`, `slow_timestep_valid` | 32 daily features, per-feature validity/age, and calendar/padding state |
| Current | `intraday_values`, `intraday_valid`, `intraday_age_sessions` | 20 scalar within-day or explicitly lagged full-session features with validity/age |
| Native fast | `fast_patch_values`, `fast_patch_valid`, `fast_patch_mask`, `fast_present` | compact seven-channel five-minute prefix |
| Risk/targets | `target_scale_sigma`, `target_*` | lagged risk scale and independent primary/shareholder/price/to-close families |

Slow windows are assembled lazily for 20, 60, or 120 sessions; they are not
duplicated on disk. Native fast tensors are read only for names and dates where
the branch is present.

### Mask semantics

Masks are independent contracts, not interchangeable readiness flags:

| Store/sample mask | Contract |
|---|---|
| `slow_valid` / `slow_feature_mask` | validity of each slow or enabled-sidecar value |
| `slow_age_sessions` / `slow_feature_age_sessions` | exchange sessions since the most recent usable source observation; `-1` means unknown/left-censored |
| `slow_timestep_valid` / `slow_history_mask` | a real elapsed per-name calendar timestep versus left padding; a genuine missing market observation remains a timestep |
| `intraday_valid` / `current_feature_mask` | validity of each current scalar value |
| `intraday_age_sessions` / `current_feature_age_sessions` | current-feature source age under the same sentinel contract |
| `fast_patch_valid` | per-channel validity inside each five-minute patch |
| `fast_patch_mask` | real contiguous prefix patches versus sequence padding |
| `fast_present` | the name has a permitted native prefix; slow-only samples remain valid when false |
| `active` / `active_mask` | decision-time entry universe, independent of future labels |
| `target_valid`, `target_shareholder_valid`, `target_price_valid` | independent outcome support for each target family and horizon |

An invalid tensor payload is stored/passed as zero only with a false aligned
mask. Economic zero remains valid. Changing an invalid payload or adding left
padding must not change a prediction; inserting a real missing calendar
session may change state and age as intended.

## Feature preprocessing

`FeatureSpec` is the ordered semantic contract for every model field. It
records source family, unit, timing rule, formula/support, validity,
age/staleness treatment, transform, and version. The manifest derives feature
order and dimensions from these specifications.

The default transforms preserve type:

- continuous cross-sectional relative states use tie-aware rank-Gauss over at
  least 20 valid active names, clipped to +/-3;
- binary flags remain 0/1;
- fractions in [0,1] map to [-1,1], while already signed values retain their
  sign;
- session ages use a capped log1p transform and retain raw age/censor status;
- standardized signed descriptors use their declared model-only clipping;
- annual loan rates use `clip(asinh(x / 0.01), -5, 5)`; and
- rebalance timing ramps preserve their signed timing multiplier.

No global learned scaler is fitted during store construction. If a later
registered variant uses a fitted scaler, it is trained inside each fit window,
frozen and hashed, and never sees selection or evaluation.

The 32 slow features cover exact-session shareholder-wealth returns and
momentum, Yang-Zhang volatility, higher moments and extremes, beta/residual
risk, liquidity/activity, price shape, observed-history state, and
past-only leave-one-out peer clusters. `log_adjusted_close` is not a model
feature. Rolling reducers keep their documented support rules; exact endpoint
returns never become partial sums.

The 20 current scalars contain same-session completed-prefix information and
explicitly named lag-one full-session summaries. Current endpoints use the
last completed observed price plus age. Lagged closing summaries use the
actual dated session endpoint and disclose whether an auction is supported.
No close-derived action classification may mask the same day's input.

Optional lending, events, options, odd-lot, rebalance, and fundamentals
sidecars are publication-ordered as-of snapshots. Revisions may affect later
decisions but cannot rewrite earlier ones. A missing or irreversibly transformed
source field stays unavailable; a similar proxy is not relabelled as the
requested field. Sidecar values, masks, and exchange-session ages join the
slow snapshot sequence, and GBDT receives the same canonical scalar fields,
ages, and validity information.

## Native five-minute fast stream

The default fast path is native to the current clock and works without any v1
directory or checkpoint. It uses completed five-minute blocks from the dated
continuous open to the decision, normally 69 blocks for a 10:00-15:45 prefix.
There are no 12 artificial compatibility patches.

Its seven channels, each with separate validity, are:

1. adjacent endpoint log return divided by `s5`;
2. block log high/low range divided by `s5`;
3. signed close location;
4. relative same-clock volume against the prior 20 sessions;
5. observed-minute fraction;
6. elapsed-session fraction; and
7. last observed price age as a fraction of scheduled continuous minutes.

`s5 = sigma_asof(t) * sqrt(5 / scheduled_continuous_minutes_t)`, using the
same positive-scale guard as the daily target. Returns do not bridge missing
endpoints or sessions. Exact OHLC channels require their stated within-block
support, and price and activity validity remain separate.

Only present names are collated. A six-block, 64-wide causal TCN encodes their
real prefix and scatters 64-wide states back to the broad security axis.
Absent fast data uses an explicit learned missing state and presence flag.
Native weights are freshly initialized by default.

An isolated `legacy_v1_contaminated` adapter may be used only for explicitly
labelled historical diagnostics with complete ancestor chronology. A file
hash proves identity, not temporal admissibility. Contaminated or unknown
ancestry is not eligible for a clean comparison or official evaluation.

## Targets

All horizon targets use `H in {1, 2, 3, 5, 10}`, entry after the decision on
`t`, and exit on `t+H`. Entry and exit marks are the declared daily
last-trade close proxy until a verified auction source is introduced.

### Gross shareholder holding family

Entry wealth buys the claim at close(`t`). Contractual conversions and cash
entitlements over `(t, t+H]` are applied exactly once; distributions stay as
cash/receivables rather than being discretionarily reinvested. A verified
endpoint can remain valid despite a missing intermediate quote. An unresolved
action chain or unknown terminal claim remains invalid.

`target_shareholder_simple_return`, `target_terminal_wealth`, and
`target_terminal_loss` preserve the economic result.
`target_shareholder_midrank` is the tie-aware rank of the raw holding return.
A known zero terminal wealth is a valid -100% outcome and ranks at the bottom.

### Price-return family

The price family measures the same economic share claim after contractual
split/unit conversions but excludes cash distributions. It has its own
`target_price_simple_return`, `target_price_midrank`, and
`target_price_valid`; it is not a substitute for shareholder wealth.

### Primary training target

For positive terminal shareholder wealth, let
`R_i,t,H = log(terminal_wealth_i,t,H)`. Known zero wealth participates as
negative infinity in the cross-sectional order statistic. With the risk scale
already available as of the decision:

`z_i,t,H = clip((R_i,t,H - median_j R_j,t,H) / (sigma_i,t * sqrt(H)), -5, 5)`.

Median subtraction occurs before name-specific volatility scaling.
`target_primary` is the tie-aware midrank of supported `z`;
`target_normalized_residual` retains `z`; and
`target_normalized_cross_section_valid` reports whether the median itself
was finite. A known zero-wealth name maps to the lower clipped value when the
median is finite, while its exact loss remains in the shareholder arrays.

Synthetic market-neutralized cash flows are absent from the default target and
headline PnL paths. Every target family has an independent mask, and requested
fit/selection/evaluation capabilities clear a target unless its full endpoint
interval stays inside that exact window.

The sixth, to-close head is an optional future-outcome auxiliary. It never
enters the decision input. Its default loss weight is `0.0`; the only
registered enabled weight is `0.2`, with its target, mask, coefficient, and
gradient path explicit.

## Neural model and objective

The reference model shares weights across names and has no security, ticker,
or sector embedding:

1. Zero invalid slow payloads, concatenate per-feature masks, project to width
   64, apply LayerNorm, and run a one- or two-layer 64-wide GRU. Calendar
   padding does not advance state.
2. Zero invalid current scalar payloads, concatenate their masks, and project
   to a normalized 64-wide state.
3. Encode the compact native prefix into a 64-wide fast state, or use the
   explicit absent state.
4. Pool slow states across active names as mean plus dispersion. Gate the fast
   and pooled states, concatenate slow/current/fast/pool states and the
   presence flag, then project to width 128.
5. Apply two residual LayerNorm/SwiGLU blocks and emit D1/D2/D3/D5/D10 plus
   to-close scores.

The daily objective is the equal mean of five per-date soft-Spearman horizon
losses. Persistence is optional. The to-close term contributes nothing at its
default zero weight. Complete date cross-sections, or complete adjacent-date
pairs when persistence is enabled, remain intact during microbatch
accumulation. With SAM, perturbation and update occur once per effective batch.

LightGBM trains one regressor per horizon on the same canonical last-step slow,
current, sidecar, mask/age information, using NaN only for invalid numeric
cells. It does not receive the extra native minute sequence.

## Stage chronology

Every stage uses the same decision-row meaning and slow history through
`t-1`:

| Stage | Fit population | Fast stream | Model selection |
|---|---|---|---|
| P | 2010-01-04 through 2021-07-30 | absent | final 10% chronological holdout, preceded by a 70-session embargo |
| F | 2021-08-16 through each fold's fit boundary | native where present | 10-session purge, 55-session selection window, 10-session purge, then evaluation |
| J | authorized P and F fit segments together | absent on P, native where present on F | the registered F selection window; optional 756-session half-life weighting |

P therefore cannot gain same-day close information merely because M1 is
absent. Native F/J fast weights are freshly initialized, including when a
clean P checkpoint initializes the compatible non-fast parameters.
Raw-Patience and final EMA (decay 0.995) checkpoints bind the store, feature
schema, stage, fold, dates, lookback, configuration, and input chronology.

## Selection and evaluation

Chronology is always `fit -> purge -> selection -> purge -> evaluation`.
Block parity is not a selection or evaluation mechanism. Each fold/seed yields
one selected model and one continuous evaluation path.

Primary selection and evaluation freeze `P = {D1, D2, D3, D5}`:

1. On each date, intersect entry eligibility, positive finite risk scale,
   outcome validity, and finite score validity across all four heads.
2. Require at least 20 common names.
3. Compute every head's Spearman correlation on that same population.
4. Define the daily primary as the equal mean only when all four correlations
   are defined; otherwise record an undefined reason.
5. Define fold primary as the equal mean of defined daily-primary values.
   Pooled primary concatenates actual fold dates before averaging, so unequal
   fold lengths are not silently equal-weighted.

D10 is a separate all-five-horizon diagnostic with its own common population.
It never governs primary checkpoint selection. To-close is auxiliary only.
Reports keep possible/used dates, outcome support, score-support loss, and name
counts rather than using `nanmean` across changing heads.

Evaluation reports the median-adjusted/scaled primary IC, shareholder-return
Rank-IC, price-return Rank-IC, and raw shareholder top-minus-bottom spread by
horizon. Paired candidate/baseline comparisons use exactly common dates,
names, score masks, and outcome masks. Moving-block bootstrap samples preserve
20-session order and fold boundaries.

The reference economic signal is the tie-aware rank average of D1/D2/D3/D5;
D10 is excluded. A separately labelled D5-only diagnostic may be run after
engineering acceptance. Results include total and per-session return spreads,
score persistence, deployed gross/net, turnover, actual holding ages, fills,
cost/borrow/funding attribution, action-affected PnL, and unresolved exposure.

Economics uses immutable intended orders created from the decision snapshot,
then later fill observations. A missing fill cannot cause a hindsight
replacement. The persistent ledger carries signed shares, free/restricted
cash, receivables/payables, pending orders, costs, financing, borrow, and
valuation status. Eligibility loss requests an exit but does not fabricate a
sale. A held name that has no print for ten consecutive sessions is settled at
its last mark on the tenth session under the explicitly labelled
`last_mark_after_10_sessions` development convention; the ordinary cost is
applied and the name is never reopened. A parallel 30% adverse settlement-price
scenario is reported separately and never used as the headline result. Net
performance is compared with compounded all-cash equity. Insolvency stops
trading and remains in the report.

Ledger utilization is gated per evaluation on mean daily marked notional in
positions that are stale inside the settlement grace period or exposed to an
explicit unresolved action term, strictly below 2% of NAV. An unobserved
session alone is not an unresolved claim. Settlement count, notional,
contemporaneous NAV share, and later-print count are mandatory per-book
diagnostics. Cumulative settlement notional above 15% of contemporaneous NAV
labels that book `economics_unresolved`. Any terminal unresolved inventory
retains count, notional, and the nonexclusive causes `no_terminal_print`,
`settlement_grace_sessions`, `unresolved_action`, and `prior_pending_exit`;
the terminal snapshot is not itself the utilization gate.

Undefined comparisons fail the not-worse guard. An all-cash result can be
well-defined but is not evidence of deployment feasibility.

## Development windows and sealing

F1-F3 are chronological development folds and may be used for model decisions.
Each has fit, 10-session purge, 55-session selection, 10-session purge, and
evaluation segments. Label endpoints are clipped centrally so no horizon
crosses a segment, purge, or sealed boundary.

Official validation is 2025-01-02 through 2025-12-30 and requires a repository
preregistration before evaluation access. It cannot be used for training or
selection. Dates from 2026-01-02 onward are held-out test dates and are refused
by the current code.

Store capabilities authorize dates before mapping model or target arrays.
Access ledgers derive official/test access from actual requested dates; callers
cannot self-declare those flags.

## Operations

Use Python 3.12 through `uv`. Store, action, validation, and run outputs must
be new roots; raw archives remain immutable. A representative store build is:

    uv run --project research python -m brazil_rv.v2.build_store --cotahist-root <parsed-cotahist-root> --cotahist-raw-root <raw-cotahist-root> --cotahist-parse-audit <parse-audit> --session-schedule <versioned-session-schedule> --isin-links-allowlist <verified-links-csv> --implementation-commit <full-current-git-sha> --actions <verified-action-bundle>/corporate_actions.parquet --m1-assignments <m1-assignments> --output-dir <new-v2-store>

Repeat `--sidecar GROUP=PARQUET` for intentionally materialized sidecars.
`--minute-npz` may point to a pre-aligned native M1 bundle, but the assignment
provenance remains required.

A single canonical trajectory can be exercised with:

    uv run --project research python -m brazil_rv.v2.train --store <v2-store> --output-dir <new-run-dir> --stage F --fold F1 --seed 11

The native default requires no fast checkpoint. The legacy checkpoint flags
are only for the explicit contaminated diagnostic adapter.

The development-only integration driver is:

    uv run --project research python -m brazil_rv.v2.validate_pipeline --store-root <v2-store> --cdi-path <development-cdi> --cdi-sha256 <sha256> --experiment52-cdi-path <reference-cdi> --experiment52-cdi-sha256 <sha256> --output-root <new-validation-root> --device cuda

Diagnostic session limits do not constitute full-scale acceptance. Record the
exact command, commit, configuration, store manifest hash, source hashes, and
tests actually run.

## Engineering acceptance

Before registered research, the revised path must demonstrate:

- raw quality, official calendar, decision-prefix causality, contractual
  actions, identity conversions, and known/unknown terminal outcomes;
- revision-safe as-of sidecars and timezone/publication correctness;
- typed-feature semantics, consumer parity, mask/padding invariance, and a
  native tiny fit/score with no v1 assets;
- order-before-fill behavior and hand-reconciled signed-share/cash/action
  ledger paths, including failed exits and insolvency;
- exact common-population evaluation and fold-preserving comparisons;
- populated production-axis build and real-scale adapter memory below the
  declared 8-GiB working-set target; and
- explicit rejection of stale stores, checkpoints, scores, and resume roots.

Unavailable source capability is reported as unsupported, not imputed into a
passing result. Only after these checks and a fresh immutable build should
development research be preregistered and run.
