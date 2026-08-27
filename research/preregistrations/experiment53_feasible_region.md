# Experiment 53 — feasible-region map

Frozen before any Experiment 53 backtest score exists. This is a discovery-only,
CPU-only map of hand execution policies, not a live policy or trading system.

## Inputs and interpretation

The completed, hash-audited Experiment 52 root supplies the exact discovery
Folds C/A/B prediction wrappers, canonical store, real M1 replay, causal ADV20
and minute liquidity, one-completed-quarter-lagged Roll schedule with causal
fallback, sigma, and dated CDI. Official-validation and held-out-test predictions
remain outside the input contract.

A measured net-positive cell is an existence proof and lower bound for a future
learned policy. If all cells are negative, the result is a verdict on this hand-
policy family only and must never be described as evidence that the alpha is
untradeable. The selected C1 and existing C0 are both retained as references.

## Policy and pre-score Amendment A53.1

At each refresh, `ConcentratedPolicy` selects the K strongest and K weakest
names by the equally blended centered 30/60/120-minute ranks. Raw targets are
equal at `gross_target/(2K)`. A prior selected name exits only after crossing
`ceil(1.5K)` (expanded to include a capacity-completed selected set), and normal
per-name band semantics otherwise apply. The band threshold is
`band_base*sigma_i + c*full_spread_i`, algebraically identical to
`(band_base + c*full_spread_i/sigma_i)*sigma_i` without division at zero.

Amendment A53.1 was accepted before any score or launch:

1. Experiment 53 alone uses `name_cap_fraction_of_gross=0.05`.
2. If either selected side has insufficient summed `min(name cap, ADV cap)`
   capacity, selection extends in deterministic next-rank order until the
   inclusive side target is reached. Each refresh records
   `selection_extended_count`; exit hysteresis applies to extended members.
3. A cell is ineligible for C1 if mean deployed gross on any fold is below 50%
   of its configured gross target. It remains fully reported.
4. The grid, variants, readouts, and all other rules remain unchanged.

The optional top-half-ADV universe is a date/name mask built from causal prior
ADV20. The spread-schedule multiplier is part of the hashed configuration.

## Frozen grid and variants

Exactly 48 cells are the Cartesian product:

- K: `10`, `20`, `40`;
- band base: `0.5`, `1.5`;
- cost scale c: `0`, `1`;
- gross target: `1.0`, `2.0`;
- universe: full or top-half prior ADV20.

Every cell uses equal horizon blending and Experiment 52 defaults except the
explicit fields above and A53.1's 5% name cap. Every cell-fold runs measured
(2-bps fees, 1.0x spread), frictionless (zero fees, 0.0x spread), and half-
spread (2-bps fees, 0.5x spread). No cell or parameter may be added after any
score exists.

## Readouts and C1

All 432 standard reports record annualized net Sharpe, daily cost drag, and net
excess over an all-cash CDI benchmark. Liquidity-tercile tables attribute
turnover, spread cost, gross PnL, and pre-CDI net PnL. Per-round-trip tables
compare gross alpha and cost by tercile. Capacity-extension counts are retained
for every refresh.

The requested Roll/tick sanity is informational and cannot alter the schedule.
The Experiment 52 input contract contains no causal historical quoted-tick
archive. The MT5 historical `spread` field is prohibited as a market spread and
the catalogue is a current snapshot. The program therefore emits the per-name
lagged-Roll table and distribution with the tick ratio explicitly unavailable;
it does not fabricate a proxy or expand the frozen input contract.

C1 uses measured variants only. Cells failing the 50% deployed-gross guard on
any fold are excluded. For each heldout fold, remaining cells rank by mean
annualized net Sharpe on the other two folds. Most rotation wins selects C1;
a tie uses higher mean heldout Sharpe, and an exact residual tie fails closed.
All 144 measured cell-fold results remain reported.

No neural policy, training, maker model, schedule edit, post-score cell,
official-validation read, test read, prediction deployment change, or live
interface is permitted.
