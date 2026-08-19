# Brazil-RV experiment log

Last updated: 2026-08-19

This is the chronological research record for the current experimentation program. Future sessions should append new experiments and preserve prior entries, including negative results. Every entry should identify the code commit, data source, split access, complete selection rule, seeds, artifact location, and result.

## Split and leakage ledger

| Split | Dates | Role in this session | Accessed? |
|---|---:|---|---|
| Training | 716, 2021-08-16 through 2024-06-28 | Parent fitting and source of the two internal discovery folds | Yes |
| Official validation | 244, 2024-07-08 through 2025-06-30 | Matched reproduction of the historical incumbent only | Yes, by the parent reproduction |
| Held-out test | 259, 2025-07-07 through 2026-07-17 | Final lockbox | No |

The official validation period had already been used repeatedly before this session and is not pristine. The trajectory discovery campaign did not load it. The held-out test remained inaccessible to campaign drivers and was not used for training, rule selection, or reporting.

The internal discovery folds were cut only from the 716 training dates:

| Fold | Fit window | Selection window | Samples |
|---|---|---|---:|
| A | First 512 dates, 2021-08-16 through 2023-08-31 | Next 102 dates, 2023-09-01 through 2024-01-31 | 28,160 fit; 5,610 select |
| B | First 614 dates, 2021-08-16 through 2024-01-31 | Final 102 dates, 2024-02-01 through 2024-06-28 | 33,770 fit; 5,610 select |

The two selection periods do not overlap. Both fit windows preserve the sampler's 512-distinct-date behavior. Stored features are causal, but the time-of-day profile continued adapting within these historical training dates. These folds are therefore screening folds, not exact replicas of an officially frozen preprocessing regime.

## Common model and optimization contract

Unless an entry says otherwise, experiments used the peer-free incumbent recipe below.

- Feature store: `m1_features_pit_causal_tod_20260818T151728490951Z`
- Feature contract: `M1_FEATURES_PIT_CAUSAL_TOD`
- Feature-store hash: `c90103b0f99e0017dc1303284a1ab61eca99106094227f5823ba718756d28a6b`
- Universe: 158 equities; 7 local and 8 global context series
- Targets: 30-, 60-, and 120-minute horizons
- Inputs: 26 dynamic features and 32 slow features
- Context screen: drop the WIN$ local input, neutralize slow `beta_to_WIN`, retain the global ZT/ZN rates inputs; WDO and the five DI inputs remain active
- Sequence representation: 5-minute patches, 130 values per patch, 69 patches
- Model: shared causal TCN, width 64, six residual blocks, kernel size 3, dilations 1/2/4/8/16/32, SwiGLU hidden width 24, fusion width 128, dropout 0.10, three outputs
- Parameter count: 277,379
- Sole training objective: soft Spearman, temperature 0.50
- Optimizer: SAM with rho 0.125 over AdamW
- AdamW: learning rate 0.0003, betas 0.9/0.95, epsilon 1e-8, weight decay 0.01
- Gradient norm clipping: 1.0
- Effective batch size: 512; loader and microbatch size: 256; evaluation batch size: 256
- Sampling: uniform dates, without date replacement
- Loader: 8 workers, prefetch factor 4
- Learning-rate schedule: 5% warmup and final learning-rate factor 0.10
- Numeric/runtime settings: float32, high matmul precision, compiled full-graph model with static shapes
- Seeds: 11, 29, and 47

Evaluation uniformly rank-averages member predictions within each sample and horizon, with tie-aware average ranks. IC is then calculated per sample; decisions are averaged within date and horizon, followed by equal averaging across dates and horizons. Ensemble weights are never learned from validation.

## Change 0 — Restore a trustworthy parent

Code merged on 2026-08-19 in commit `d5f651c9a8f6cc4ea0367833e84f856c978bff67` through PR #15.

- Restored soft Spearman as the sole canonical objective.
- Deleted the rejected hybrid loss, continuous-target requirement, residual-attention branch, campaign driver, and their dead tests/configuration.
- Removed target-scale sidecars from ordinary training and evaluation.
- Restored explicit held-out evaluation support for manual final evaluation while keeping held-out data inaccessible to campaign drivers.
- Removed other stale compatibility paths instead of preserving legacy code.

Historical reproducibility remains in recorded commits and immutable artifacts, not in compatibility code on current `main`.

## Experiment 1 — Matched peer-free parent reproduction

Purpose: verify that the cleaned parent reproduces the incumbent before changing checkpoint selection.

### Settings

- Historical recipe commit: `4067962`
- Data: all 716 training dates for fitting
- Evaluation: the 244-date official validation split
- Seeds: 11, 29, 47
- Maximum epochs: 20
- Early stopping: patience 3, minimum improvement 0.0001
- Selected state: best raw checkpoint observed before stopping
- Steps per epoch: 77
- Warmup steps: 77
- Held-out test access: none

Reusing official validation here was required for a matched, apples-to-apples reproduction of the historical incumbent. It was not used by the later discovery campaign.

### Results

| Seed | Historical IC | Reproduced IC | Delta | Best epoch | Stop epoch |
|---:|---:|---:|---:|---:|---:|
| 11 | 0.041972266 | 0.041977574 | +0.000005309 | 12 | 15 |
| 29 | 0.040481999 | 0.040475500 | -0.000006499 | 14 | 17 |
| 47 | 0.038463105 | 0.038464003 | +0.000000898 | 7 | 10 |

Conclusion: the peer-free incumbent was reproduced to within 6.5e-6 IC for every seed, with identical best and stop epochs. The cleaned implementation is a trustworthy parent.

Immutable artifact:

`/lambda/nfs/brazil-rv-east3/quant-data/b3/processed/model_runs/parent_reproduction_4067962_e22dd67_20260819T131142Z`

## Experiments 2 and 3 — Fixed 20-epoch checkpoint trajectories

Purpose: use one training trajectory per fold and seed to compare stopping and checkpoint-averaging policies without retraining for every policy.

### Settings

- Code commit used for the run: `e22dd67`
- Matrix: folds A/B times seeds 11/29/47, for six total trajectories
- Training length: exactly 20 epochs; no training-time early stop
- Fold A: 55 optimizer steps per epoch, 1,100 total steps, 55 warmup steps
- Fold B: 66 optimizer steps per epoch, 1,320 total steps, 66 warmup steps
- Saved after every epoch:
  - Raw model checkpoint
  - EMA states at decay 0.98, 0.99, and 0.995
  - Selection-fold predictions for the raw and all three EMA states
- EMA update after each optimizer step: `shadow = decay * shadow + (1 - decay) * current`; non-floating state is copied
- Total artifacts: 120 checkpoint files and 126 prediction archives, including constructed tail candidates

### Candidate definitions

- `final_raw`: raw epoch-20 checkpoint.
- `final_ema_098`, `final_ema_099`, `final_ema_0995`: the indicated EMA state after the final epoch-20 optimizer update.
- `last3_weight_average`: arithmetic average of raw checkpoint weights from epochs 18-20.
- `last5_weight_average`: arithmetic average of raw checkpoint weights from epochs 16-20.
- `tail3_prediction_average`: arithmetic average of raw predictions from epochs 18-20.
- `tail5_prediction_average`: arithmetic average of raw predictions from epochs 16-20.
- `patience3_raw`: replay the chronological raw validation ICs using patience 3 and minimum improvement 0.0001, selecting the stored best checkpoint when the rule stops. This is a predeclared stopping algorithm, not the 20-epoch oracle.
- `retrospective_best_epoch_raw`: per-seed argmax over all 20 raw selection-fold ICs. This is an oracle diagnostic and is never eligible for deployment selection.

Weight averages use float64 accumulation before casting floating tensors back to their original dtype; non-floating state comes from the final checkpoint in the averaging window.

### Cross-fold results

Each cell is the three-seed, uniformly rank-averaged ensemble IC.

| Rule | Fold A | Fold B | Mean |
|---|---:|---:|---:|
| `final_raw` | 0.043416330 | 0.049601613 | 0.046508972 |
| `final_ema_098` | 0.043521575 | 0.049681043 | 0.046601309 |
| `final_ema_099` | 0.043825966 | 0.049905499 | 0.046865732 |
| `final_ema_0995` | 0.045308520 | 0.050625481 | 0.047967001 |
| `last3_weight_average` | 0.043437668 | 0.049902230 | 0.046669949 |
| `last5_weight_average` | 0.043628097 | 0.050146187 | 0.046887142 |
| `tail3_prediction_average` | 0.043457892 | 0.049932086 | 0.046694989 |
| `tail5_prediction_average` | 0.043660588 | 0.050187198 | 0.046923893 |
| `patience3_raw` | **0.049575896** | **0.054144782** | **0.051860339** |
| `retrospective_best_epoch_raw` | 0.049381828 | 0.054144782 | 0.051763305 |

The initial implementation classified both `patience3_raw` and `retrospective_best_epoch_raw` as diagnostic and therefore selected `final_ema_0995` from the remaining rules. That classification was too conservative for Patience-3. The request explicitly called for testing Patience-3 versus fixed 20 epochs and reserved diagnostic-only status for retrospective best epoch. Patience-3 is a fully specified, reproducible rule once patience and minimum improvement are frozen.

Correct interpretation: Patience-3 was not rejected by the evidence. It is the leading numerical candidate, beating `final_ema_0995` by 0.003893338 mean IC and winning on both folds. It should be restored to selection eligibility and subjected to the same paired bootstrap and guardrail analysis before any official-validation stage run. Only the retrospective 20-epoch oracle must remain diagnostic-only.

Immutable artifact:

`/lambda/nfs/brazil-rv-east3/quant-data/b3/processed/model_runs/trajectory_discovery_e22dd67_20260819T134332Z`

## Experiment 4 — Initial deterministic-rule selection

Purpose: pick one rule from the internal folds without learning ensemble weights or tuning again on official validation.

### Implemented selection rule

For each candidate, calculate the three-seed ensemble IC separately on folds A and B, average those two values, choose the largest mean, and break an exact tie lexically by rule name. Under the initial, overly restrictive eligibility set, this selected `final_ema_0995` with mean IC 0.047967001.

Status: provisional and scientifically superseded by the Patience-3 interpretation above. No official-validation or held-out-test decision should rely on `final_ema_0995` as the winner until Patience-3 is made eligible and receives equivalent diagnostics.

## Experiment 5 — Measurement and guardrails for provisional EMA-0.995 winner

Purpose: compare the provisional selected candidate (`final_ema_0995`) with `final_raw` under strict alignment and paired uncertainty estimates.

### Measurement settings

- Strict alignment keys: `sample_id`, date, decision, targets, and masks
- Prediction aggregation: uniform, tie-aware rank averaging within each sample and horizon
- Reported ensemble diagnostics: member ICs, ensemble IC, pairwise seed diversity, and gains versus the mean and best member
- Paired unit: candidate-minus-parent IC per date after averaging decisions and horizons
- Uncertainty: moving-block bootstrap at block lengths 5 and 10
- Guardrails: horizon and time-of-day deltas
- Learned validation weights: none

### Results

| Fold | EMA-0.995 minus final-raw IC | Block-5 95% interval | Block-10 95% interval |
|---|---:|---:|---:|
| A | +0.001892190 | [-0.000126494, 0.003692874] | [-0.000015640, 0.003627169] |
| B | +0.001023868 | [-0.000086806, 0.001815572] | [-0.000078277, 0.001567743] |

Horizon deltas were positive in both folds:

| Fold | 30m | 60m | 120m |
|---|---:|---:|---:|
| A | +0.000718726 | +0.001486883 | +0.003470961 |
| B | +0.000658549 | +0.000887587 | +0.001525467 |

Time-of-day effects were mixed: 8 of 55 decision slots were negative on Fold A and 17 of 55 were negative on Fold B. Fold A ranged from -0.001888432 to +0.006398633; Fold B ranged from -0.001262664 to +0.003981136.

Pairwise seed correlations were 0.9085-0.9140 on Fold A and 0.9283-0.9321 on Fold B. The ensemble gain versus the mean member was +0.00104825 on Fold A and +0.00098071 on Fold B.

Conclusion: EMA-0.995 consistently improved the final raw checkpoint, especially at 120 minutes, but both moving-block intervals touched or crossed zero and TOD effects were not uniformly positive. These results do not answer whether EMA-0.995 is preferable to Patience-3 because Patience-3 was mistakenly excluded from this paired analysis.

## Open decision before the next stage

1. Make `patience3_raw` eligible while keeping `retrospective_best_epoch_raw` diagnostic-only.
2. Run the existing analyzer on Patience-3 versus the fixed parent using the already-saved predictions; no retraining is required.
3. If its paired bootstrap, horizon, TOD, and diversity diagnostics are acceptable, freeze the entire Patience-3 rule: monitor, minimum improvement 0.0001, patience 3, raw-state selection, and tie behavior.
4. Use official validation sparingly for the resulting stage winner. Do not use the held-out test until the final recipe is locked.

