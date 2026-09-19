# Original US inputs and decision-clock audit

The 19 preserved source responses reconcile all 71,135 bounded 2010–2024 bars and
69,029 normalized return rows exactly. Original OHLC, volume and adjusted close
match the source projections. Each return requires positive observed endpoints and
volume, with adjacency on the EWZ source-session axis. No missing observation is
silently converted to a zero return or a longer-horizon move.

The audit independently constructs historical US regular and scheduled early
closes using New York time and DST. All stored availability timestamps agree.
There are 261 symbol-days whose close precedes the same-date B3 15:45 decision.
Pre-NYSE Suzano retains 2,249 original rows and 656 returns with unknown availability;
the audit preserves this boundary instead of inventing an OTC closing time.

A separate original-return-to-model audit checks EWZ's one- and five-session shocks
through the first eligible B3 decision, including holidays and source age. Both
fields match all stored float32 values, masks and ages: 568,567 and 568,064 active
valid cells, respectively. Fourteen early-close dates are correctly available on
the same B3 date. Five-session shocks require five linked returns, and a holiday
may carry known state with an older age without adding another return observation.
ADR-relative returns have a separate paired prior-B3/US endpoint contract; a same-day
US early close does not make the same-day B3 close available.

`ops/audit_us_sources.py` reads only the bounded timestamp and indicator arrays,
without decoding present-day market-price metadata. It verifies the source response
hashes and compares the original fields independently. It took 1.43 seconds on CPU.
`ops/audit_ewz_decisions.py` independently reconstructs and checks the two stored
fields. The run pointer binds both reports and their source identities.

These are latest preserved vendor snapshots, not proven contemporaneous vintages.
Historical correction and rounding revisions remain unknown. A common multiplicative
adjustment cancels in log returns, which does not prove arbitrary revisions harmless.
Adjusted vendor levels cannot establish a contemporaneous cash-price ADR premium or
historical conversion ratio. No such premium is admitted. Archives begin in 2010;
this audit does not create an earlier US warm-up or a survivor-free ADR roster.

No source, accepted model store, feature coordinate or old fit changed. Other
cross-market feeds, financial publication/revisions and denominators, full identity
propagation and corporate wealth/labels still require Stage B work. Exact source
reconstruction is not evidence of forecasting skill or economic improvement.
