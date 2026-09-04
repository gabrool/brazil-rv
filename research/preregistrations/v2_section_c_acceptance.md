# V2 Section C acceptance contract

Status: frozen before the next real-data rebuild and before any Section D score.

The target-validity gap between names that end within the panel and names whose
continuation identity reaches the final panel year remains capped at 10
percentage points.  Every internally derived feature family remains subject to
an unconditional 5-point validity-gap cap: slow return, slow volatility, slow
liquidity/state, peer, intraday, and COTAHIST classification-mask availability.

External `options`, `lending`, `oddlot`, `rebalance`, `events`, and
`fundamentals` families use both of these checks instead:

1. Rebuild the complete validity tensor independently from the raw archive's
   publication-lagged availability coordinate, including the upstream D+1
   availability date for daily archives, and require exact equality.
2. Within each pooled quartile of causal prior-20-session mean BRL volume among
   observable family rows, cap the eventual-survival validity gap at 5 points.
   Retain coverage by survival group and quartile.  The pooled unstratified row
   is diagnostic rather than binding.

The known options diagnostic is an unstratified 5.4220916184-point gap with
delisted-name validity above survivor validity.  It is a liquidity-composition
signature and passes the stratified gate; it is not evidence of availability
leakage.

Same-ticker COTAHIST ISIN changes link only when the old ISIN's final
observation is immediately followed on the next market session by the new
ISIN's first observation.  Ticker reuse after a gap and concurrent or returning
ISINs do not link.  The root continuation identity determines eventual
survival.  The successor inherits only predecessor feature rows strictly before
its first session, and the daily universe carries the same causal prior history
without simultaneously activating both identities.  Every qualifying link is
an immutable store audit row.

After a fresh local rebuild, both gate families and all Section C audit tables
must pass before the registered full-F1 GPU pipeline validation may start.
Official validation and the permanently spent test remain inaccessible.  Stop
at the first failure and do not run Section D.
