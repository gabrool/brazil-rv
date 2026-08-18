# Brazil-RV project context

Last verified: 2026-08-18.

## Purpose and current research state

Brazil-RV is an offline research system for predicting 30-, 60-, and 120-minute
cross-sectional equity ranks from B3 M1 data. A sample is one eligible trading
date and one of 55 five-minute decisions over a fixed 158-slot point-in-time equity
axis.

The accepted incumbent is the peer-free, full causal time-of-day normalized,
width-64 causal TCN trained uniformly with soft Spearman and SAM-AdamW. The best
exact validation result verified on 2026-08-18 is seed-11 IC **0.041972**. The
gap-pairwise hybrid loss and residual-state cross-equity attention in current HEAD
are completed, rejected research candidates rather than promoted defaults.

Read [RESEARCH_HANDOFF.md](RESEARCH_HANDOFF.md) for the exact architecture,
campaign chronology, results, NFS artifact identities, interpretations, current
HEAD caveat, and next recommended experiments.

## Source and write boundaries

- Python 3.12; use `uv` and the `research/` project.
- Raw data under `quant-data/b3/raw/**`, canonical source archives, and
  `Trading/**` are immutable.
- Derived stores belong under `quant-data/b3/interim/**` or
  `quant-data/b3/processed/**`.
- Resolve canonical pointer files at runtime and record resolved identities in
  output manifests. Never hard-code a timestamped source when a valid pointer is
  available.
- Equity identity is permanent `security_id`/ISIN plus bounded source-assignment
  dates. Ticker is only a dated attribute.
- Monthly point-in-time membership is the eligibility contract.

## Current feature-store contract

The peer-free store contract is `M1_FEATURES_PIT_CAUSAL_TOD`: 1,248 dates, 1,228
eligible dates, 55 decisions, 158 equities, 26 dynamic channels, 32 slow channels,
three horizons, seven local contexts, and eight global contexts. It has no
human-prior or peer arrays.

Equity normalization uses an equity-wide 30-minute causal relative-variance TOD
profile with a 20-session-equivalent prior and `[0.25, 4.0]` bounds. Each training
date emits before updating; the profile freezes after 2024-06-28. Context series
retain their semantic causal transforms rather than receiving the equity overlay.

The full store built on persistent Lambda NFS is recorded in
`RESEARCH_HANDOFF.md`. The Windows local canonical pointer may still identify the
old V4 store; verify the pointer and schema in the execution environment before
training.

Core causal rules:

- History ends strictly before the decision; the entry bar is excluded.
- Label entry is `open[T]`; exit is exact `close[T+h-1]` within the session.
- Missing OHLC is never interpolated and stale prices are not label endpoints.
- Fitted scalers, volatility/TOD state, and other stateful transforms use only the
  information available at their historical timestamp.
- Training, validation, and test identities are immutable and audited.

## Splits and test policy

- Training: 2021-08-16 through 2024-06-28, 716 dates.
- Validation: 2024-07-08 through 2025-06-30, 244 dates.
- Held-out test: 2025-07-07 through 2026-07-17, 259 dates.

Embargo dates are not selection data. Training, early stopping, and experiment
selection use training and validation only. Test data may be opened only through
an explicit standalone held-out evaluation. The 2026-08-18 campaigns all record
`test_accessed=false`.

## Accepted model contract

- One shared per-instrument width-64 causal TCN.
- Five-minute patches, 69 patches, kernel 3, dilations `(1, 2, 4, 8, 16, 32)`.
- Six residual LayerNorm/SwiGLU blocks and final-state readout.
- Projected 32-field slow state.
- Fixed context-plus-masked-equity-mean/dispersion gated fusion.
- `WIN$` and equity `beta_to_WIN` masked; WDO, five DI contexts, ZT, and ZN active.
- All three horizons trained jointly.
- Accepted objective: soft Spearman, temperature 0.50.
- Uniform training dates; SAM-AdamW rho 0.125; effective batch 512.
- Maximum 20 epochs and three-epoch early stopping.
- No peer/classification inputs and no cross-equity attention in the incumbent.

Hard Spearman is the primary selection metric. It is averaged across decisions
within each date and horizon, then equally across validation dates and horizons.

## Current source-tree status

The repository deliberately retains the exact rejected hybrid-loss and
residual-attention experiment implementation at the 2026-08-18 snapshot. Therefore
the generic training entry point currently describes that research candidate and
is not the accepted incumbent command. See `RESEARCH_HANDOFF.md` before running or
removing it. Historical runs remain reproducible through recorded commits and
immutable manifests; current checkpoint loading is strict.

The old human-prior/peer, alternate model-family, routing, multiscale, attribution,
probe, overlay, and V-numbered experiment systems were deleted. Do not recreate
compatibility shims for them.

## Environment and operations

From the repository root:

```powershell
uv sync --project research --no-default-groups --group preprocessing
uv run --project research python -m brazil_rv.preprocessing.build
uv run --project research python -m pytest research/tests/test_modeling_training.py `
  research/tests/test_modeling_data.py research/tests/test_intraday_normalization.py
```

`ops/lambda-gh200.ps1` is the only Lambda watcher/launcher. Launch requires
explicit billing acknowledgement, transfers a verified Git bundle, and leaves the
instance running. It never starts training or terminates a paid host. Confirm
provider state and exact instance identity before launch or termination.

## Limitations and authority

The system does not model order-book state, bid/ask spread, queue position,
slippage, costs, or live execution. Historical MT5 spread and tick volume are not
market microstructure. Raw absolute prices, tickers, identifiers, news, and
unapproved technical indicators are not model features.

When statements conflict, prefer immutable sources and canonical pointers, then
executable code and tests, then this document, then the detailed handoff.
