# Experiment 54 — conditional edge, latency decay, and maker feasibility

Frozen before any Experiment-54 forward-return, fill, frontier, or positioning
number exists. This is discovery-only analysis over Folds C/A/B. It does not run
the execution simulator, train a model, change a policy, access official
validation, or access the permanently spent held-out test.

## Immutable sources

The freeze stage must hash-verify the completed Experiment-53 root and its exact
Experiment-52 source. It reuses Experiment 52's discovery prediction wrappers,
causal ADV20/minute-capacity profiles, lagged spread schedule, daily sigma, and
CDI series. It streams the canonical identity-bounded raw M1 grids only through
`TRAIN_END`, materializes high/low/close for the same 309 discovery dates, and
requires raw open/observed arrays to equal the Experiment-52 cache exactly.
Observed bars must have finite positive OHLC, `high >= max(open, close)`, and
`low <= min(open, close)`. No label mask participates.

## Event and state contract

An event is one `(fold, date, refresh, security_id)` with all three causal ranks
valid, a nonzero equal-weight blended centered rank, a valid prior-refresh rank,
and finite positive prior ADV20, lagged full spread, and causal daily sigma. The
first refresh is excluded because `|delta rank|` has no prior-refresh value; it
is counted in the exclusion record rather than assigned an invented delta.

The state is exactly:

- blended-rank decile;
- `|delta rank|` quintile;
- entry into either of the outer two rank deciles since the prior refresh;
- prior-ADV20 tercile;
- time of day: first 60 minutes of the prediction refresh schedule, middle, or
  final 60 minutes of that schedule;
- all-three-head sign agreement;
- lagged-full-spread tercile; and
- causal-daily-sigma tercile.

All empirical edges are linear quantiles computed once from eligible Fold-C
state attributes during `freeze`, before any outcome is evaluated. The exact
numeric edges and `searchsorted(..., side="right")` rule are frozen and reused
unchanged for A/B. Tail events are rank deciles `0/1/8/9`. Direction is short
below zero and long above zero.

## Part A — gross edge, latency ladder, and taker frontier

Horizons are exactly `15/30/60/120` minutes. A point entry at minute `e` exits
at that session's observed close at `e + horizon - 1`. The four entries are:

- decision open: `e = refresh` (IC-native reference only);
- next open: `e = refresh + 1` (taker-real);
- mean of every observed open from `refresh + 1` through `refresh + 10`, with
  exit `refresh + 10 + horizon - 1`; and
- the analogous complete 30-open mean, with exit
  `refresh + 30 + horizon - 1`.

Every required entry open and the exact exit close must be observed; otherwise
that event-entry-horizon is unavailable. Signed gross alpha is
`direction * (exit / entry - 1) * 10_000` bps. The latency table reports each
entry and its mean decline from decision-open alpha.

Conditional tables report count, dates, events/day, mean, median, and fractions
above fixed `4.5/7/10` bps hurdles. They also report fractions clearing each
event's own three cost scenarios: `2 + 0.25*full_spread_bps` (half-spread),
`2 + 0.5*full_spread_bps` (measured taker), and
`2 + 1.0*full_spread_bps` (conservative). These extra columns disambiguate the
fixed hurdle labels without changing them.

For each fold, horizon, and fixed threshold, the taker frontier uses same-fold
state-cell mean next-open gross edge. This is deliberately optimistic oracle
selection at the state-cell level, never event-outcome selection. Only cells
whose mean exceeds the threshold may trade. Expected event net is that mean
minus the event's measured taker cost; nonpositive expected-net events are not
required by a maximum. At each refresh, events are sorted by expected net then
permanent slot and allocated up to the smaller of 10% of the causal prior
minute-notional profile and `5% * 2.0 * R$10m = R$1m`, with total gross capped
at `2.0 * R$10m`. The upper bound has no neutrality, persistence, impact, or
round-lot constraint. Daily and fold-mean NAV bps are retained.

For that decision only, each fold's frontier is its maximum over the four
registered horizons; no horizon is chosen after seeing the result. The frozen
decision is verbatim: if the 7-bps taker frontier is below 8 bps/day of NAV on
every fold, taker execution at R$10m is CLOSED for the learned-policy stage and
any neural policy targets maker actions. If it clears 8 bps/day on at least two
folds, taker actions remain viable and their reward hurdle is all-cash CDI. Any
other pattern is recorded as inconclusive; no new rule is invented.

## Part B — conservative maker feasibility

Only tail events participate. The last observed close strictly before the
decision is the reference. Limits are that close and, separately, buy below or
sell above it by one half of the lagged half-spread (one quarter of the lagged
full spread). Waits are exactly `5/15/30` minutes. A quote posted after the
decision fills at its limit only when an observed subsequent bar in offsets
`1..wait` has `low < limit` for a buy or `high > limit` for a sell. Touches do
not fill. The first strict-through bar is the fill minute.

At each `15/30/60/120` horizon, market post-fill alpha uses the fill-bar open
and an exact close `horizon` minutes later. Exact limit alpha uses the same exit
and the limit price. Price improvement is exact limit alpha minus market
post-fill alpha, so their sum cannot double-count the limit. Filled net edge is
exact limit alpha minus the 2-bps fee. Adverse-selection gap is conditional
mean market post-fill alpha minus unconditional next-open event alpha in the
same frozen state cell and matched horizon.

An unfilled quote crosses at the exact open `wait + 1` minutes after decision,
holds for the matched horizon, and pays its own measured taker cost. A composite
event is retained only when its realized filled or fallback path and exact exit
are observed. The conditional table reports quote/composite counts, fill rate,
time to fill, adverse selection, improvement, filled and fallback net edge, and
composite net edge. Through-price fills are explicitly conservative and are not
a queue model.

For every fold, horizon, wait, and limit variant, the maker frontier uses the
same allocator and caps as Part A, replacing taker state means with same-fold
state-cell composite-net means and retaining only positive means. It reports
daily and fold-mean NAV bps beside the 8-bps/day framing. It has no automatic
decision rule; interpretation is the user's call.

## Part C — informational construction comparison

The next-open Part-A table alone supplies expected gross edge and measured
taker cost. Long-short outer-decile tails allocate at most one NAV gross to
each side; long-only tails allocate at most one NAV gross to longs and retain
cash under the existing 50%-of-gross margin convention. Both use the same
participation and 5%-of-configured-gross name caps and deterministic rank-
strength priority. The table reports expected net NAV bps/day and the long-only
residual CDI credit. Fold equal-weight market returns provide the measured
intraday drift component and its daily variance under beta approximately one
on deployed long gross. Drift is a displayed component of long returns, not a
second PnL credit; beta variance is reported in bps-squared rather than mixed
dimensionally into return without a preregistered risk-aversion coefficient.
This is construction information only.

## Required outputs and interpretation

One immutable root retains source/input hashes, raw-OHLC verification, frozen
buckets, event exclusions, the latency and conditional tables, taker daily and
fold frontiers, the exact taker decision, maker conditional/daily/fold
frontiers, positioning comparison, definition statements, result, logs, and a
full final audit. No post-score bucket, threshold, horizon, wait, state, cost,
frontier, policy, schedule, or interpretation change is permitted.

Expected in advance: 15–30-minute edge decays faster than 120-minute edge; the
7-bps taker frontier is near or below 8 NAV bps/day; maker fill rates may be
40–70%, and a 2–6-bps adverse-selection gap may leave tail edge. These are
expectations only. A weak maker composite informs the next user decision and
does not itself authorize RLP work, a live system, or a tradeability verdict.
