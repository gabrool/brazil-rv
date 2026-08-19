# Brazil-RV project context

Last verified: 2026-08-19.

## Purpose and current research state

Brazil-RV is an offline research system for predicting 30-, 60-, and 120-minute
cross-sectional equity ranks from B3 M1 data. A sample is one eligible trading
date and one of 55 five-minute decisions over a fixed 158-slot point-in-time equity
axis.

The accepted incumbent is the peer-free, full causal time-of-day normalized,
width-64 causal TCN trained uniformly with soft Spearman and SAM-AdamW. The best
recorded exact validation result is seed-11 IC **0.041972**. The rejected
gap-pairwise loss, continuous-target sidecar, and residual equity-attention branch
are absent from the current tree; their commits, manifests, and immutable artifacts
remain the historical reproduction contract.

The historical parent was reproduced on 2026-08-19 at commit `4067962` with
matched seeds 11/29/47. Best-IC deltas versus the immutable records were
`+0.0000053`, `-0.0000065`, and `+0.0000009`; every best epoch and stop epoch
matched. The internal folds then froze `final_ema_0995` as the trajectory rule.

Read [RESEARCH_HANDOFF.md](RESEARCH_HANDOFF.md) for architecture and campaign
history, exact results, artifact identities, and interpretations.

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

## Splits, discovery folds, and test policy

- Training: 2021-08-16 through 2024-06-28, 716 dates.
- Validation: 2024-07-08 through 2025-06-30, 244 dates.
- Held-out test: 2025-07-07 through 2026-07-17, 259 dates.
- Fold A: first 512 training dates fit, next 102 select.
- Fold B: first 614 training dates fit, final 102 select.

The two internal selection periods do not overlap. Both fit windows preserve an
effective batch of 512 distinct dates. Stored features are causal, but the TOD
profile adapted inside the historical training dates, so these are screening
folds rather than exact replicas of the officially frozen preprocessing regime.

The official validation split has already been consumed and is reserved for
sparse confirmation of stage winners. Embargo dates are not selection data. The
held-out test is the final lockbox and may be opened only through the explicit
standalone evaluator for an official-window run carrying a rule frozen on the
internal folds. Campaign drivers cannot request validation or test rows.

## Accepted model and trajectory contract

- One shared per-instrument width-64 causal TCN.
- Five-minute patches, 69 patches, kernel 3, dilations `(1, 2, 4, 8, 16, 32)`.
- Six residual LayerNorm/SwiGLU blocks and final-state readout.
- Projected 32-field slow state.
- Fixed context-plus-masked-equity-mean/dispersion gated fusion.
- `WIN$` and equity `beta_to_WIN` masked; WDO, five DI contexts, ZT, and ZN active.
- All three horizons trained jointly.
- Sole objective: soft Spearman, temperature 0.50.
- Uniform training dates; SAM-AdamW rho 0.125; effective batch 512.
- One fixed 20-epoch trajectory; no training-time early stopping.
- Raw checkpoint and raw/EMA validation predictions every epoch.
- EMA decays 0.98, 0.99, and 0.995.
- Frozen rule: final epoch EMA-0.995. Do not reselect epoch or EMA decay on the
  official validation split.
- Last-3/last-5 weight averages and raw-score prediction averages are constructed
  without retraining.
- Patience-3 and retrospective best epoch are diagnostic only.
- No peer/classification inputs and no cross-equity attention.

Hard Spearman is the primary selection metric. It is averaged across decisions
within each date and horizon, then equally across dates and horizons. Seed
ensembles uniformly average tie-aware within-sample/horizon ranks and never fit
ensemble weights.

The completed internal campaign at
`trajectory_discovery_e22dd67_20260819T134332Z` selected final EMA-0.995 with
fold-A/fold-B ensemble ICs `0.045309`/`0.050625` and mean `0.047967`, versus
final-raw `0.043416`/`0.049602` and mean `0.046509`. Paired EMA-minus-raw deltas
were positive on both folds (`+0.001892`, `+0.001024`) and at every horizon, but
their block-bootstrap intervals mostly included zero and time-of-day deltas were
mixed. Treat the rule as a deterministic variance-reduction choice, not a claim of
uniform statistical dominance. Patience-3 and retrospective-best scored higher
internally but remain diagnostic-only because they select epochs from the screening
windows.

## Current source-tree status

`modeling.train` is the canonical soft-Spearman trajectory entry point.
`modeling.run_discovery_campaign` runs exactly the two internal folds and seeds
11/29/47, then freezes one deterministic trajectory rule using their mean
three-seed ensemble IC. `modeling.analyze` strictly aligns observations and reports
member/ensemble IC, seed diversity, ensemble gains, paired date deltas, moving
block intervals at lengths 5 and 10, and horizon/time-of-day guardrails.

`modeling.evaluate` restores held-out evaluation without exposing it to campaign
drivers. It accepts only a completed official-window run with an internal-fold
selection file recorded in its manifest.

The old human-prior/peer, alternate model-family, routing, multiscale, attribution,
probe, overlay, V-numbered, recency-weight, hybrid-loss, target-sidecar, and
residual-attention experiment systems are not compatibility APIs. Historical
reproduction uses their recorded commits and immutable artifacts.

## Environment and operations

From the repository root:

    uv sync --project research --group dev
    uv run --project research python -m brazil_rv.preprocessing.build
    uv run --project research python -m brazil_rv.modeling.run_discovery_campaign --output-dir <new-campaign-directory>
    uv run --project research python -m brazil_rv.modeling.train --selection-window official --selection-rule-file <trajectory-selection.json> --seed 11
    uv run --project research python -m brazil_rv.modeling.evaluate --run-dir <completed-official-run-directory> --split test

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
