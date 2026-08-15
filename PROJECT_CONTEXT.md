# Brazil-RV Project Context

Last verified: 2026-08-15.

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

The default project dependencies are the current modeling runtime. Local feature-store construction and source normalization use the explicit `preprocessing` group; tests and lint use `dev`, which includes preprocessing dependencies. Interactive notebook packages are not part of the current environment contract.

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

Embargo dates are not model-selection data. Training, early stopping, routing selection, opening thresholds, and other model choices use training and validation only. Test data is opened only by an explicit standalone evaluation with `--split test`. Attribution is validation-only and has no split option. The horizon-multiscale stage runner has no split option and never reads or evaluates the test period.

## Current model and input policy

The incumbent is a width-64, full-receptive-field, SwiGLU causal TCN with `context_pooled` fusion, all 32 slow fields, selected sector/subsector peer features, soft Spearman at temperature 0.50, SAM-AdamW at rho 0.125, at most 20 epochs, three-epoch early stopping, and seed choices 11, 29, or 47.

Production GH200 training uses effective batches of 512 as two ordered 256-sample loader/GPU microbatches. It compiles the model and soft-Spearman objective with the Inductor `default` mode, full graphs, and static shapes, while validation remains eager against the same current parameters.

The canonical context policy is direct:

- `WIN$` is masked.
- Equity `beta_to_WIN` is zeroed.
- Global non-rate contexts are masked.
- `ZT.v.0` and `ZN.v.0` remain active.
- `WDO$`, all five DI inputs, and the two US-rate inputs keep their causal masks.

Selected peer state contains six fields: selected-peer 15/60-minute return differences, selected-peer 15/60-minute ranks, and two validity flags.

Default routing is `slow=late_only, macro=late_only`. Direct slow or macro alternatives are `late_only`, `early_concat`, `film`, and `early_concat_film`. Isolated slow FiLM remains available and is neutral at initialization because its final heads are zero-initialized. Transformer, TCN, and MLP are current neural families.

The default TCN readout remains `final`, which uses only the final causal block state and strict-loads pre-readout checkpoints. Opt-in readouts are `shared_multiscale` (one global six-tap mixture), `horizon_multiscale` (one six-tap mixture per horizon), and `final_score_mlp` (a zero-initialized residual 3-to-2-to-3 score control). Training-only diagnostic controls are `--training-horizon {all,30,60,120}` and `--context-family-ablation {none,wdo,br_rates,us_rates}`; both default to the incumbent behavior.

## Commands

From the repository root:

```powershell
# Local feature-store and source preprocessing environment
uv sync --project research --no-default-groups --group preprocessing

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

# Resumable train/validation-only multiscale and diagnostic stage
uv run --project research python -m brazil_rv.modeling.run_horizon_multiscale_stage `
  --output-dir <persistent-stage-directory>

# Lambda availability, launch, local safety tests, and credential deletion
.\ops\lambda-gh200.ps1 -Mode Notify -Notify
.\ops\lambda-gh200.ps1 -Mode Launch -IUnderstandBilling
.\ops\lambda-gh200.ps1 -SelfTest
.\ops\lambda-gh200.ps1 -ForgetStoredApiKey
```

Launch prints the instance ID, IP, exact SSH command, and persistent bootstrap log. Use that SSH command for `nvidia-smi` or log monitoring. Instance termination is deliberately manual in the Lambda console after verifying the printed instance ID; the watcher never starts training or terminates a paid instance.

## Evaluation and attribution

Hard Spearman is the primary validation and checkpoint-selection metric. It is averaged across decisions within each date and horizon, then equally across horizons. Training-time validation uses decision-grouped batches, restores canonical sample order, and computes only objective loss plus this selection metric each epoch; the complete validation report is built once from the retained best-epoch observations. Standalone evaluation still builds the complete report normally. Top/bottom returns and turnover use raw returns.

Attribution performs validation inference once per current run or reads an explicitly requested simple prediction cache. It reports exact additive stock IC contributions, cross-sectionally normalized time-series rank skill, 5-minute and canonical 30-minute bins, horizon attribution, five-day moving-block confidence intervals, training-fitted overnight regimes, and causal opening-context completeness. Outputs are plain CSV, Parquet, and JSON.

The horizon-multiscale stage writes atomic resumable state plus run, probe, covariance, gradient, paired-bootstrap, gate-weight, and consolidated summary artifacts under its explicit output directory. Every resumable step strictly validates its complete artifact schema and provenance before reuse, including immutable feature-store identity, canonical fit/selection windows, material training settings, parameter counts, and strict checkpoint reconstruction. Frozen and OOF probe fitting is train-only; ordinary trained-arm comparison and paired context-retraining ablations use canonical validation; diagnostic evidence, trained evidence, controls, and negative-transfer outcomes remain separate hypotheses in the final conclusion. The runner never evaluates held-out test data.

Current checkpoints use one schema with strict PyTorch state-dict loading. Unique run directories prevent collisions, and checkpoints are published atomically. Historical run formats are intentionally unsupported.

## Operations

`ops/lambda-gh200.ps1` is the single notification/launch watcher. Launch requires explicit billing acknowledgement, refuses ambiguous matching instances, uses the explicit Brazil-RV SSH key and per-instance known-hosts file, transfers a verified Git bundle and bootstrap script, and leaves the instance running on success or failure.

The remote bootstrap verifies delivery, fast-forwards a clean nondivergent checkout from the uploaded bundle, verifies the filesystem mount and AArch64/GH200 visibility, installs only the frozen default modeling runtime with `--no-default-groups`, and checks package import, CLI help, and Python compilation. It does not install development, notebook, or preprocessing-only dependencies; run the research test suite; compile a model; perform forward/backward/SAM work; start training; or terminate the instance.

## Limitations

The system does not model order-book state, true bid/ask spread, queue position, slippage, costs, live futures rolls, or live operational failures. Historical MT5 `spread` and `tick_volume` are not market microstructure. Raw absolute prices, tickers, identifiers, issuer embeddings, news, and unapproved technical indicators are not model features.

When statements conflict, prefer immutable sources and canonical pointers, then executable code and tests, then this document.
