# Experiment 58 — swing feasibility screen

Frozen before any Experiment-58 score exists. This is a CPU-only analysis of
the existing ten-seed, four-head OOF predictions on the canonical 716 TRAIN
sessions. It performs no training, reads neither official validation nor the
permanently spent test, and authorizes no deployment change.

The model and its inputs were designed for intraday horizons. A positive
multi-day result is therefore a floor for a future daily-native program; a
negative result closes the swing question only for these existing signals.

## Immutable inputs

- Experiment-56 Section-B `result.json`, OOF prediction archive, reference,
  execution manifest, and resolved feature store.
- Experiment-57 Stage-0 `result.json`, `frozen_design.json`, designated
  dollar-neutral `replay_daily.parquet`, CDI series, and causal TRAIN market
  cache. Part 0 may aggregate only columns already present in that retained
  daily report; it may not replay the teacher.
- The canonical TRAIN date and permanent-security axes resolved by those
  manifests. Raw M1 is streamed only through `iter_discovery_equity_grids`,
  which is bounded at `TRAIN_END` and enforces accepted identity/date segments.

Every path and byte identity is written into `frozen_design.json`. The run must
execute at that file's exact repository commit.

## Part 0 — retained intraday attribution

Use only designated, dollar-neutral Experiment-57 Stage-0 daily report rows.
For C/A/B and their pooled union, report mean gross PnL, spread cost, fee cost,
CDI earned, net PnL, all-cash CDI, net excess over all-cash CDI, and turnover,
all in NAV bps/day. NAV is the report's frozen BRL 10,000,000 scale. This is a
table pull and aggregation, not a simulation.

## Part A — multi-day signal measurement

The OOF archive's four heads are `30m`, `60m`, `120m`, and `to_close`. For each
head, form an end-of-day signal from the final archived refresh and a
last-hour signal by averaging archived refresh ranks whose cutoff is within 60
minutes (inclusive) of the final cutoff. Re-midrank the latter cross-sectionally
per date. Also report cross-sectionally re-midranked equal means of the four
heads for both refresh variants. No seed is selected or reweighted: the source
archive is the already-frozen ten-seed ensemble.

Two context signals are built from exact observed daily closes: negative
five-session close return (reversal) and positive twenty-session close return
(momentum), each cross-sectionally midranked on the signal date. They are
report-only and never enter Part B or C.

For D in `{1,2,3,5,10}`, the target at TRAIN date t is exact close(t+D) over
exact close(t), divided by the causal stored sigma(t) times `sqrt(D)`, then
cross-sectionally median-removed and midranked. The median removal is retained
explicitly even though it cannot change ranks. A name is valid only when its
signal-date membership/readiness, both exact close endpoints, finite positive
sigma, and signal are valid. No stale close substitutes for an endpoint.

Report daily cross-sectional IC and block-10 moving-block 95% intervals for
all TRAIN and separately for canonical C/A/B; IC decay by D; consecutive-date
rank autocorrelation; and equal-weight top-minus-bottom decile raw return in
bps per planned holding day. Persistence pairs must be consecutive on the
canonical TRAIN axis and, for C/A/B, both dates must lie in that window.

## Part B — declared coarse auction feasibility

The exact eight cells are:

- signal: final-refresh `to_close` head or final-refresh equal four-head mean;
- K: 15 or 30 names on each side;
- rank-movement band: 0 or 0.3 on the `[-1,1]` rank scale.

At close t, require exact closes at t and t+1 and signal-date activity. For a
zero band, rank on the current signal. For band 0.3, a name whose current and
previous-session signals are both valid keeps its previous-session rank unless
the absolute change strictly exceeds 0.3; new or previously invalid names use
their current rank. Select exactly K highest and K lowest effective ranks,
equal weight each side, long gross 1 and short gross 1. A date with fewer than
2K eligible names is all cash. Holdings earn close(t)-to-close(t+1) returns.
Turnover is `sum(abs(w_t-w_{t-1}))`; the first entry is from cash and the final
book is liquidated at the last close, with that terminal turnover charged to
the final holding interval.

Every cell is reported at per-side turnover costs `{2,4,7}` bps and annual
short-borrow rates `{0.02,0.04}` using 252 sessions. CDI for the interval is
the pinned rate on its exit session. With the existing 0.5 margin fraction,
CDI earns on `max(1 - 0.5*gross, 0)` NAV. Daily excess over all-cash CDI is
therefore gross return minus turnover cost and borrow minus CDI on the margin
line. Report all TRAIN and canonical C/A/B block-10 intervals, annualized net
Sharpe, turnover/NAV/day, and `2*sum(gross)/sum(turnover)` average holding days.
Liquidity-tercile attribution uses causal signal-date ADV ranks and includes
name-level gross, turnover cost, and borrow, excluding unallocable CDI.

There is no mechanical gate. The headline asks whether any cell's all-TRAIN
block-10 lower 95% bound is above zero at 4 bps per side, separately at each
borrow assumption. The full sensitivity grid remains primary evidence.

## Part C — patient-entry bound

Informational analysis uses the final-refresh `to_close` and equal four-head
mean signals, K 15 and 30 tails, without the Part-B band. On signal date t,
place next-session long/short limits at close(t), or inside by half of the
lagged half-spread: buy `close*(1-0.25*full_spread)` and sell
`close*(1+0.25*full_spread)`. The spread is the causal schedule/fallback value
already frozen for date t.

Wait either the first 60 minutes or the full next session. A buy fills only if
an observed low is strictly below its limit; a sell fills only if an observed
high is strictly above its limit, at the limit price. For the composite, an
unfilled order crosses at the exact observed minute-60 close or next-session
close respectively; missing exact fallback prices remain missing. For every D
in `{1,2,3,5,10}`, exit at exact close(t+D). Report fill rate, unconditional
tail alpha, conditional-on-fill alpha, their adverse-selection gap, and the
limit-then-taker composite alpha, in directional bps per planned holding day.
No Part-C result selects a Part-B cell.

## Outputs and audit

One fresh commit-bound root must retain the frozen design, daily market
summaries, signal/target arrays, Part-0 table, all Part-A daily and aggregate
tables, full Part-B daily/grid/liquidity tables, Part-C table, result, copied
operational logs, and a recursive SHA-256 inventory. All result and audit files
must state `official_validation_accessed=false` and `test_accessed=false`.
No post-score cell, horizon, cost, borrow, band, signal, or interpretation rule
may be changed or added.

## Expectations recorded in advance

Part 0 is expected to identify cost and churn as most of the approximately
-54 bps/day teacher loss. Daily-scale IC around 0.02–0.05 with slow decay is
plausible but genuinely unknown. At IC 0.04, 4 bps per side, and a two-to-three
day effective holding period, roughly +3–7 NAV bps/day over CDI is plausible.

