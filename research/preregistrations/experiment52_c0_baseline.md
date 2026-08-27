# Experiment 52 — C0 band-policy baseline

Frozen before any execution-backtest number exists. This is the first
end-to-end net-economics measurement under the audited offline execution layer.
It uses discovery folds only and is not a live-trading model.

## Inputs

- Scores: the Experiment-41 Stage-C store-v2 prune-R2 seeds 11/29/47 on
  Folds C/A/B. Each seed uses its already-frozen bidirectional opposite-parity
  Raw Patience-3 epochs. Scores are tie-aware ranked over the causal
  membership-and-readiness mask, then uniformly averaged across seeds. Each
  fold is wrapped through `write_discovery_prediction_manifest` and reloaded
  through its discovery-only guard.
- Market: permanent-security real M1 grids through `TRAIN_END`, streamed by
  `iter_discovery_equity_grids`. ADV20 and same-minute capacity use
  `causal_liquidity(lookback=20)` and emit before consuming the current day.
- Spreads: the exact hash-pinned Experiment-49 Roll schedule, lagged one
  completed quarter. Missing security-quarter values fall back to
  `causal_roll_spreads(lookback=60)` using prior sessions only.
- CDI: Banco Central do Brasil SGS series 12, retrieved once as JSON, converted
  from daily percent to fractional daily return, stored as Parquet, and
  hash-pinned with the source URL and retrieval timestamp.
- Sigma: `PRICE_VOL_REFERENCE * sqrt(405) * exp(vol_regime)`, where
  `vol_regime` is the store's causal per-name `equity_slow` field. This is a
  dimensionless daily-return scale compatible with portfolio weights. The
  stored field's existing clipping is preserved rather than reconstructed from
  later raw observations.

## Frozen grid

Exactly 12 cells are run on each of Folds C/A/B:

- band: `0.0`, `0.5`, `1.0`, `2.0`;
- horizon blend: equal `(1/3,1/3,1/3)`, 30-minute-only `(1,0,0)`, and
  front-loaded `(0.5,0.3,0.2)`.

Every other `ExecutionConfig` field remains at its default: R$10m NAV, gross
2.0, participation 10%, name cap 2.5% of gross, ADV cap 5%, fees 2 bps,
30-minute close taper, and the existing spread/liquidity guards. No other cell,
parameter, retry, or post-score configuration may be added.

Each cell also runs once with `fee_bps=0` and full spreads exactly zero. All
other configuration, prices, liquidity, scores, dates, and CDI inputs remain
unchanged. This is the frozen frictionless decomposition.

## Readouts and C0 rule

The standard execution report is written for every measured and frictionless
cell-fold run. The 36 measured summaries additionally record mean and sample
standard deviation (`ddof=1`) of daily net PnL, net-to-gross ratio, daily spread
plus fee drag in bps of initial NAV, and annualized net Sharpe
`sqrt(252) * mean / std`.

For each held-out fold, cells are ranked by their mean annualized net Sharpe on
the other two folds. The top cell's held-out result is recorded. C0 is the cell
with the most rotation wins; a win-count tie is broken by the higher mean
held-out annualized net Sharpe across all three folds. If that exact tie remains,
the program fails rather than adding an unregistered tie-break.

This experiment designates a reference; it has no promotion gate and does not
change the deployed prediction recipe. Every cell-fold result is reported even
if all are net-negative.

## Access and non-goals

Official-validation and held-out-test predictions are never wrapped or loaded.
The test is permanently spent from Experiment 51. No neural policy, training,
OOF refit, split creation, tuning, execution-metric comparison to the capacity
document, live broker interface, or deployment change is permitted. All tensors
remain on CPU.
