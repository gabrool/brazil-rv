# Offline execution backtest

This package turns causal cross-sectional prediction scores into an intraday
position replay. It is research infrastructure, not a broker simulator or live
order-management system.

## Data flow

```text
raw member scores + store-derived causal PIT mask + explicit refresh minutes
                         |
                 causal cross-sectional ranks
                         |
       forward-fill to the 405-minute B3 session grid
                         |
          BandPolicy -> neutral/capped projection
                         |
      next-open participation-capped fills and carried demand
                         |
 open-to-open PnL - half-spread - fees + CDI -> daily report
```

The current prediction archive uses refresh indices `15, 20, ..., 285` on the
session-relative minute grid. `write_discovery_prediction_manifest` creates the
small `B3_EXECUTION_PREDICTION_ARCHIVE_V1` wrapper around a canonical prediction,
reference, and source-run manifest. It records the prediction array key, hashes,
store/axis identities, split, and explicit refresh minutes. The loader verifies
those minutes against each canonical sample's `equity_cutoff_index`; it never
infers `decision_idx * 5`. A future model with one refresh per minute uses the
same interface with denser canonical cutoffs. No score is backfilled before the
first refresh, so the book remains flat until an action can be formed. An action
at minute `t` fills no earlier than the observed open at `t + 1`, one minute more
conservative than the alpha label's entry-open timing.

## Configuration

| Field | Default | Meaning |
|---|---:|---|
| `nav_brl` | R$10m | Independent starting NAV for each replay day |
| `gross_target` | 2.0 | Gross exposure before the close taper |
| `participation_rate` | 10% | Maximum share of causal minute-of-day notional |
| `name_cap_fraction_of_gross` | 2.5% | Name cap as a fraction of gross target BRL |
| `adv_cap_fraction` | 5% | Second name cap as a fraction of prior ADV20 |
| `fee_bps` | 2 bps | Per-side fee on traded notional |
| `max_spread_bps` | 75 bps | Date/name universe exclusion |
| `min_adv_brl` | R$1m | Prior-ADV20 universe floor |
| `taper_minutes` | 30 | Linear target taper into the final open |
| `force_spread_multiplier` | 2x | Half-spread multiplier for terminal residuals |
| `margin_fraction_of_gross` | 0.5 | Margin line in `max(NAV - fraction*gross, 0)` |

Every config has a canonical JSON SHA-256. Report inputs must also be named by
SHA-256, and the report writer emits an atomic JSON file plus a checksum sidecar.

## Research contracts

- The action grid is every session minute. Score cadence is metadata, not a
  simulator constant.
- Prediction ranks are rebuilt from raw scores using activity derived directly
  from the hash-bound store membership/readiness arrays. Existing metric archives
  ranked with `label_mask` are unsuitable because that mask knows future
  label-endpoint availability. The loader also binds every sample/date/decision
  row to the canonical sample index, rejects dates after `TRAIN_END`, and
  requires a completed source run on the same exact discovery fold with matching
  selection-window date/sample hashes. Aggregate OOF archives are rejected until
  a canonical materializer can bind every constituent fold and prove per-sample
  fit-window exclusion. This prevents in-sample or official-selected weights from
  being relabeled as discovery predictions.
- `causal_liquidity` emits before updating the current session: ADV20 and each
  minute-of-day median use only prior sessions. Observations remain in the arrays;
  exclusions are masks.
- `iter_discovery_equity_grids` is the real-data bridge. It reuses the canonical
  accepted source assignments, splits shared physical XP files by permanent
  `security_id` and accepted dates, and streams exact M1 opens/closes/real volume
  only through the training end. It never opens official-validation or test rows.
- The Experiment-49 same-quarter Roll schedule is not causal for execution.
  `lagged_quarter_spreads` hash-verifies it and uses only the previous completed
  quarter. `causal_roll_spreads` supplies the same Roll formula from strictly
  prior M1 sessions when the lagged schedule has no name-quarter estimate.
- Fills are signed notionals at the next observed open. Charging half the full
  spread on absolute notional is accounting-equivalent to buying above and
  selling below that open. Fees are charged once; spread is not double-counted.
- Fixed participation caps produce monotone, arithmetic convergence toward the
  target. Unfilled demand is carried. Per-name caps apply to projected targets;
  an exogenous price jump may temporarily move a realized holding beyond its
  target cap, after which it is reduced subject to the same participation limit.
  This avoids pretending a risk limit can create nonexistent liquidity. A target
  retains its last feasible projection when a transient rank-validity mask lacks
  enough signed cap capacity for exact neutral gross; the hard projection is
  never relaxed. Targets taper after hard projection; any terminal residual is
  flattened at the configured spread multiplier and counted. Terminal
  liquidation ignores the ordinary participation, liquidity-profile, and
  spread-universe gates. If the
  final minute has no open, it uses that name's last observed session open and
  spread; this explicit end-of-session approximation is counted as a forced
  fill and never changes the observation mask.
- A missing open permits neither a mark nor a fill. An existing holding keeps its
  last observed notional and realizes the cumulative open-to-open return only
  when a later observed open arrives; the observation mask remains false and no
  synthetic bar is created. This causal state consumes only the last past
  observation. Terminal liquidation fails if the session has no prior observed
  priced open for an existing holding.
- Cash receives the supplied per-session CDI return through one explicit margin
  formula. The repository currently has no canonical daily CDI execution series,
  so callers must supply and hash-record one rather than substituting a DI quote.
- There is no impact model or round-lot rounding. At the intended roughly R$10m
  NAV, realism is represented by participation and projected-target caps.
- Use float64 for final currency reporting. Float32 remains supported for future
  differentiable optimization and performance smoke runs; the standalone report
  records float values only after enforcing its 1e-8 accounting identity.

The implemented baseline is `BandPolicy`. Neural-policy training, optimizer
scaffolding, OOF refits, cluster penalties, parameter tuning, and experiment
runners are intentionally absent until a registered research question needs
them. The discovery-fold prediction loader rejects OOF, official-validation,
and test archives; the held-out test is permanently spent.
