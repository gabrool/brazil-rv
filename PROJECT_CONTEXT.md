# Brazil-RV Project Context

Last verified: 2026-08-13.

## Purpose

Brazil-RV is an offline research system for cross-sectional prediction from B3 M1 data. The research unit is one eligible trading date and one of 55 five-minute decisions. Models predict 30-, 60-, and 120-minute targets over a fixed 158-slot point-in-time equity axis.

The current repository supports:

1. Building or loading the canonical feature store.
2. Training a current neural model on the full eligible universe.
3. Evaluating a current-format run.
4. Testing direct routing alternatives.
5. Validation-only stock/time/opening attribution.

## Sources and write boundaries

- Python: 3.12
- Environment manager: `uv`
- Research project: `research/`

Raw data under `quant-data/b3/raw/**`, canonical source archives, and `Trading/**` are immutable. Derived stores belong under `quant-data/b3/interim/**` or `quant-data/b3/processed/**`. Resolve canonical pointer files at runtime rather than hard-coding timestamped output directories.

Equities are keyed by permanent B3 security identity and bounded source-assignment dates. Tickers are dated attributes. Monthly point-in-time membership is the eligibility contract; current constituents and survivors must never be substituted historically.

## Feature-store and causality contract

The external store identifier `M1_FEATURES_INTRADAY_DI_MASKED_CONTEXT_HUMAN_PRIORS_V4` identifies the existing immutable layout. It contains 1,248 dates, 1,228 eligible dates, 55 decisions per eligible date, 158 equity slots, 26 dynamic channels, 32 slow channels, three horizons, seven local contexts, eight global contexts, and selected-peer sidecars.

Core causal rules:

- Model history ends strictly before decision time; the entry bar is excluded.
- Label entry is `open[T]`; exit is `close[T+h-1]` within the permitted session.
- Missing OHLC observations are never interpolated or invented.
- Stale prices are not exact-horizon label endpoints.
- Every fixed grid carries observed masks.
- Stateful features consume only completed prior sessions or minutes available by their represented timestamp.
- Global bars become available only after their completed minute.

The local contexts are `WIN$`, `WDO$`, `DI1F27`, `DI1F28`, `DI1F29`, `DI1F31`, and `DI1$N`. Global contexts are `ES.v.0`, `NQ.v.0`, `ZT.v.0`, `ZN.v.0`, `CL.v.0`, `HG.v.0`, `6E.v.0`, and `6M.v.0`.

## Splits and held-out policy

- Training: 2021-08-16 through 2024-06-28
- Validation: 2024-07-08 through 2025-06-30
- Held-out test: 2025-07-07 through 2026-07-17

Embargo dates are not model-selection data. Training, early stopping, routing selection, opening thresholds, and other model choices use training and validation only. Test data is opened only by an explicit standalone evaluation with `--split test`. Attribution is validation-only and has no split option.

## Current model and input policy

The incumbent is a width-64, full-receptive-field, SwiGLU causal TCN with `context_pooled` fusion, all 32 slow fields, selected sector/subsector peer features, soft Spearman at temperature 0.50, SAM-AdamW at rho 0.125, at most 20 epochs, five-epoch early stopping, and seed choices 11, 29, or 47.

The canonical context policy is direct:

- `WIN$` is masked.
- Equity `beta_to_WIN` is zeroed.
- Global non-rate contexts are masked.
- `ZT.v.0` and `ZN.v.0` remain active.
- `WDO$`, all five DI inputs, and the two US-rate inputs keep their causal masks.

Selected peer state contains six fields: selected-peer 15/60-minute return differences, selected-peer 15/60-minute ranks, and two validity flags.

Default routing is `slow=late_only, macro=late_only`. Direct slow or macro alternatives are `late_only`, `early_concat`, `film`, and `early_concat_film`. Isolated slow FiLM remains available and is neutral at initialization because its final heads are zero-initialized. Transformer, TCN, and MLP are current neural families.

## Commands

From the repository root:

```powershell
# Full-universe incumbent
uv run --project research python -m brazil_rv.modeling.train

# Isolated slow FiLM
uv run --project research python -m brazil_rv.modeling.train --slow-routing film --seed 29

# Validation or explicit held-out evaluation
uv run --project research python -m brazil_rv.modeling.evaluate --run-dir <run-directory>
uv run --project research python -m brazil_rv.modeling.evaluate --run-dir <run-directory> --split test

# Validation-only attribution
uv run --project research python -m brazil_rv.modeling.analyze_stock_time_attribution `
  --run-dir <run-directory> `
  --output-dir <output-directory> `
  --cache-dir <optional-cache-directory>

# Lambda availability, launch, local safety tests, and credential deletion
.\ops\lambda-gh200.ps1 -Mode Notify -Notify
.\ops\lambda-gh200.ps1 -Mode Launch -IUnderstandBilling
.\ops\lambda-gh200.ps1 -SelfTest
.\ops\lambda-gh200.ps1 -ForgetStoredApiKey
```

Launch prints the instance ID, IP, exact SSH command, and persistent bootstrap log. Use that SSH command for `nvidia-smi` or log monitoring. Instance termination is deliberately manual in the Lambda console after verifying the printed instance ID; the watcher never starts training or terminates a paid instance.

## Evaluation and attribution

Hard Spearman is the primary validation and checkpoint-selection metric. It is averaged across decisions within each date and horizon, then equally across horizons. Top/bottom returns and turnover use raw returns.

Attribution performs validation inference once per current run or reads an explicitly requested simple prediction cache. It reports exact additive stock IC contributions, cross-sectionally normalized time-series rank skill, 5-minute and canonical 30-minute bins, horizon attribution, five-day moving-block confidence intervals, training-fitted overnight regimes, and causal opening-context completeness. Outputs are plain CSV, Parquet, and JSON.

Current checkpoints use one schema with strict PyTorch state-dict loading. Unique run directories prevent collisions, and checkpoints are published atomically. Historical run formats are intentionally unsupported.

## Operations

`ops/lambda-gh200.ps1` is the single notification/launch watcher. Launch requires explicit billing acknowledgement, refuses ambiguous matching instances, uses the explicit Brazil-RV SSH key and per-instance known-hosts file, transfers a verified Git bundle and bootstrap script, and leaves the instance running on success or failure.

The remote bootstrap verifies delivery, the filesystem mount, AArch64/GH200 visibility, the frozen dependency install, package import, CLI help, and Python compilation. It does not run the research test suite, compile a model, perform forward/backward/SAM work, start training, or terminate the instance.

## Limitations

The system does not model order-book state, true bid/ask spread, queue position, slippage, costs, live futures rolls, or live operational failures. Historical MT5 `spread` and `tick_volume` are not market microstructure. Raw absolute prices, tickers, identifiers, issuer embeddings, news, and unapproved technical indicators are not model features.

When statements conflict, prefer immutable sources and canonical pointers, then executable code and tests, then this document.
