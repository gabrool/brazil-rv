# Brazil-RV experiment log

Last updated: 2026-08-20

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

Initial interpretation, later superseded: the implementation classified both `patience3_raw` and `retrospective_best_epoch_raw` as diagnostic and selected `final_ema_0995` from the fixed rules. I incorrectly described this as unfairly excluding Patience-3 and treated its same-window IC as trustworthy evidence that it led.

Correction: Patience-3 is a fully specified deployable algorithm, but its checkpoint is selected on each 102-date window. Reporting IC on that same window gives it almost the same winner's-curse bias as retrospective best epoch. Its apparent +0.003893338 advantage over `final_ema_0995` was therefore not an apples-to-apples estimate of deployed value. Experiment 6 supplies the required out-of-half estimate.

Immutable artifact:

`/lambda/nfs/brazil-rv-east3/quant-data/b3/processed/model_runs/trajectory_discovery_e22dd67_20260819T134332Z`

## Experiment 4 — Initial deterministic-rule selection

Purpose: pick one rule from the internal folds without learning ensemble weights or tuning again on official validation.

### Implemented selection rule

For each fixed candidate, calculate the three-seed ensemble IC separately on folds A and B, average those two values, choose the largest mean, and break an exact tie lexically by rule name. This selected `final_ema_0995` with mean IC 0.047967001.

Status at this point: provisional pending an out-of-half comparison with validation-adaptive rules. Resolved by Experiment 6.

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

Conclusion: EMA-0.995 consistently improved the final raw checkpoint, especially at 120 minutes, but both moving-block intervals touched or crossed zero and TOD effects were not uniformly positive. The same-fold Patience estimate was selection-biased, so this experiment did not compare their deployed values; Experiment 6 does.

## Experiment 6 — Odd/even cross-fit of checkpoint and rule selection

Purpose: remove the winner's-curse bias from validation-adaptive checkpoint rules, add the missing EMA-0.995 Patience candidate, and test longer raw weight-average windows without retraining.

### Bias diagnosis

The original `patience3_raw` IC was calculated on the same 102-date window used to select its checkpoint. Its mean IC of 0.051860 was nearly identical to the retrospective-best oracle's 0.051763, including the same Fold-B ensemble result. It was therefore not a valid deployed-value comparison with fixed rules such as final EMA-0.995.

### Settings

- Analyzer commit: `30542282c29ad23943de1dc8ad9765a0eb88f9e2`
- Source campaign: `trajectory_discovery_e22dd67_20260819T134332Z`
- Retraining: none
- Each fold's 102 chronologically sorted selection dates was divided into 51 odd-position and 51 even-position dates.
- Direction 1 selected on odd dates and reported only on even dates; Direction 2 selected on even dates and reported only on odd dates.
- Patience was replayed independently for every seed using only the selecting parity. The selected checkpoint's predictions were placed only into the opposite parity before constructing the three-seed rank ensemble.
- `patience3_raw` monitored raw IC and restored the best raw checkpoint after three epochs without improvement greater than 0.0001.
- `patience3_ema_0995` monitored EMA-0.995 IC and restored the best EMA-0.995 checkpoint under the same patience rule.
- Fixed rules used the same state on both parities and received no per-fold checkpoint choice.
- Rule-level replay selected the highest selection-half ensemble IC and reported that selected rule only on the opposite half, in both directions.
- Prediction aggregation remained uniform, tie-aware rank averaging within sample and horizon. No ensemble weights were learned.
- Paired deltas used per-date candidate-minus-final-EMA-0.995 IC with moving-block bootstrap lengths 5 and 10 and 10,000 replications.
- Official validation and held-out test access: none.

The last-7 and last-10 candidates arithmetic-averaged raw checkpoint weights from epochs 14-20 and 11-20, respectively, with float64 accumulation. Their validation predictions required two post-hoc evaluation passes per run but no optimization steps. They were written to a new extension artifact; the source campaign remained immutable.

An initial extension attempt used seed-only filenames, which collided between folds. The process was stopped before accepting a report, the new invalid artifact was permanently deleted, and a regression test was added. The successful run keys every extension by fold and seed.

### Cross-fitted candidate results

| Rule | Fold A | Fold B | Mean |
|---|---:|---:|---:|
| `patience3_raw` | **0.048416261** | 0.050673275 | **0.049544768** |
| `patience3_ema_0995` | 0.047120095 | 0.049916235 | 0.048518165 |
| `last10_weight_average` | 0.045318709 | **0.050801712** | 0.048060210 |
| `final_ema_0995` | 0.045308520 | 0.050625481 | 0.047967001 |
| `last7_weight_average` | 0.044347388 | 0.050356302 | 0.047351845 |
| `tail5_prediction_average` | 0.043660588 | 0.050187198 | 0.046923893 |
| `last5_weight_average` | 0.043628097 | 0.050146187 | 0.046887142 |
| `final_ema_099` | 0.043825966 | 0.049905499 | 0.046865732 |
| `tail3_prediction_average` | 0.043457892 | 0.049932086 | 0.046694989 |
| `last3_weight_average` | 0.043437668 | 0.049902230 | 0.046669949 |
| `final_ema_098` | 0.043521575 | 0.049681043 | 0.046601309 |
| `final_raw` | 0.043416330 | 0.049601613 | 0.046508972 |

Cross-fitting reduced raw Patience's apparent advantage over final EMA-0.995 from 0.003893338 to 0.001577767 mean IC. EMA Patience did not improve on raw Patience. Last-10 was effectively tied with final EMA-0.995. Within the raw weight-average family, however, longer windows improved monotonically at every tested length.

The exact raw weight-average sequence was strictly monotone over every tested
window: last-3 `0.046669949`, last-5
`0.046887142`, last-7 `0.047351845`, and last-10 `0.048060210`. Comparing last-7
with final EMA-0.995 was informative for rule selection but was the wrong
comparator for the averaging-window trend. The trend had not saturated at last-10.

### Patience replay epochs

Epoch triplets are seeds 11/29/47 in order, shown as best checkpoint / stopping epoch.

| Fold and selecting parity | Raw Patience | EMA-0.995 Patience |
|---|---|---|
| A odd | 8/11, 5/8, 5/8 | 11/14, 10/13, 11/14 |
| A even | 10/13, 5/8, 9/12 | 14/17, 13/16, 14/17 |
| B odd | 4/7, 6/9, 4/7 | 8/11, 8/11, 8/11 |
| B even | 7/10, 9/12, 8/11 | 15/18, 12/15, 18/20 |

### Raw Patience versus final EMA-0.995

| Fold | Cross-fitted IC delta | Block-5 95% interval | Block-10 95% interval |
|---|---:|---:|---:|
| A | +0.003107741 | [-0.005447569, 0.011401605] | [-0.005303876, 0.011083898] |
| B | +0.000047794 | [-0.004546764, 0.003404376] | [-0.004811228, 0.002349003] |

| Fold | 30m delta | 60m delta | 120m delta |
|---|---:|---:|---:|
| A | -0.001166307 | +0.002453159 | +0.008036371 |
| B | -0.000247896 | -0.000978029 | +0.001369305 |

TOD effects were mixed: 25 of 55 slots were negative on Fold A, ranging from -0.014302 to +0.021623; 29 of 55 were negative on Fold B, ranging from -0.011852 to +0.013599. The result is driven most coherently by the 120-minute horizon and does not show uniform horizon or TOD dominance.

The fold result is best read as fold-dependent post-peak decline, not as
"Patience works on A but not B." On Fold A final raw ended at `0.043416330`
versus cross-fitted Patience `0.048416261`, so Patience recovered about
`0.0050`; on Fold B final raw `0.049601613` was already near Patience
`0.050673275`, leaving only about `0.0011` to recover. Relative to final EMA-0.995,
both folds concentrated the Patience gain at 120 minutes (`+0.008036371` and
`+0.001369305`) while 30-minute deltas were slightly negative. With only two
folds, the record cannot distinguish regime distance from the 512-versus-614-date
fit windows. That uncertainty is a reason to freeze the simple winner and stop
probing checkpoint selection.

### Cross-fitted rule-selection replay

| Fold and direction | Rule selected on first half | Opposite-half IC |
|---|---|---:|
| A odd to even | `patience3_raw` | 0.056957688 |
| A even to odd | `patience3_raw` | 0.039874834 |
| B odd to even | `patience3_raw` | 0.055943832 |
| B even to odd | `patience3_ema_0995` | 0.042812317 |

The combined out-of-half rule-selector IC was 0.048416261 on Fold A and 0.049378075 on Fold B, mean 0.048897168. Raw Patience was selected in three of four directions; EMA Patience was selected once.

### Decision

Freeze `patience3_raw`, the highest per-rule cross-fitted candidate: raw IC monitor, minimum improvement 0.0001, patience 3, maximum 20 epochs, restore best raw checkpoint. Retrospective best epoch remains diagnostic-only. This is a numerical selection, not evidence of established dominance: Fold B provided essentially no paired advantage over final EMA-0.995 and every paired block interval included zero.

The sparse official-validation stage may apply this already-frozen stopping algorithm but must not retune its patience, threshold, epoch, state type, or EMA decay. The held-out test remains sealed.

Immutable artifact:

`/lambda/nfs/brazil-rv-east3/quant-data/b3/processed/model_runs/trajectory_crossfit_3054228_20260819T161200Z`

## Experiment 7 — Patience-centered five-checkpoint weight averaging

Purpose: test the mechanistic combination missing from Experiment 6—use raw
Patience to locate the early peak, then reduce checkpoint variance by averaging
weights around that peak—without retraining or consuming another data split.

### Settings

- Evaluator commit: `381dcb7491b26f1e34d4ecdef75d0e5e291b5441`
- Source campaign: `trajectory_discovery_e22dd67_20260819T134332Z`
- Retraining and optimizer steps: none
- One predeclared candidate only; no window sweep
- For every fold, selecting parity, and seed, replay raw Patience-3 using only the
  51 selecting dates.
- Arithmetic-average five raw checkpoint state dictionaries centered on the
  selected best epoch: `[best-2, best-1, best, best+1, best+2]`. A boundary case
  shifts the five-wide window inside epochs 1-20; no observed replay required a
  boundary shift.
- Accumulate floating weights in float64 and cast back to their original dtype;
  copy non-floating state from the latest checkpoint in the window.
- Evaluate the averaged state once per unique window, but report its predictions
  only on the opposite parity. Repeat in both directions.
- Construct the three-seed ensemble by uniform tie-aware rank averaging within
  sample and horizon. Learn no ensemble weights.
- Official validation and held-out test access: none.
- Source artifacts remained immutable. The centered predictions and replay
  windows were written to a new extension artifact.

### Results

| Rule | Fold A | Fold B | Mean |
|---|---:|---:|---:|
| `patience3_raw` | **0.048416261** | **0.050673275** | **0.049544768** |
| `patience3_center5_weight_average` | 0.046655118 | 0.050385365 | 0.048520241 |
| `final_ema_0995` | 0.045308520 | 0.050625481 | 0.047967001 |

Centered-minus-raw-Patience was `-0.001761143` on Fold A and `-0.000287910`
on Fold B, mean `-0.001024527`. It was lower in every out-of-half direction:

| Fold and direction | Centered IC | Raw-Patience IC | Delta |
|---|---:|---:|---:|
| A odd to even | 0.054607525 | 0.056957688 | -0.002350163 |
| A even to odd | 0.038702710 | 0.039874834 | -0.001172124 |
| B odd to even | 0.055891940 | 0.055943832 | -0.000051892 |
| B even to odd | 0.044878789 | 0.045402718 | -0.000523929 |

Against final EMA-0.995, centered averaging gained `+0.001346598` on Fold A
but lost `-0.000240117` on Fold B. The block-5 intervals were
`[-0.006870319, 0.009388453]` and `[-0.005029775, 0.003178609]`; block-10
intervals were `[-0.006861055, 0.009247092]` and
`[-0.005313988, 0.002133605]`. Horizon deltas versus EMA-0.995 were
`-0.002189669/+0.000703642/+0.005525820` on Fold A and
`-0.000598913/-0.001190991/+0.001069554` on Fold B for 30/60/120 minutes.
The same 120-minute concentration remained, but smoothing diluted the raw
Patience result rather than improving it.

### Decision

Reject the centered five-checkpoint average and keep `patience3_raw` frozen.
The centered rule lost on both folds and all four honest evaluation halves, so no
additional window or centering sweep is justified. This closes the checkpoint-rule
line of investigation.

The rejected evaluator was removed from current HEAD under the project's
deletion-first rule. Historical reproduction uses evaluator commit `381dcb7` and
the immutable artifact below, not compatibility code:

`/lambda/nfs/brazil-rv-east3/quant-data/b3/processed/model_runs/trajectory_centered_crossfit_381dcb7_20260819T170100Z`


## Experiment 8 — Phase A representation screens

Purpose: test the six cheapest representation improvements from the research memo
on top of the frozen raw Patience-3 parent, with final EMA-0.995 retained as a
free secondary readout from the same trajectories.

### Settings

- Implementation and campaign commit: `732b1b0e7dd870d9ea210c7b2eb750a624f12fb7`
- Parent campaign: `trajectory_discovery_e22dd67_20260819T134332Z`
- Feature store and hash: unchanged from the common contract above
- Folds: the same 512+102 and 614+102 internal discovery folds
- Seeds: 11, 29, and 47
- Training: one fixed 20-epoch soft-Spearman SAM trajectory per fold and seed
- Primary readout: honest odd/even cross-fitted `patience3_raw`, selecting the
  stopping checkpoint on one parity and reporting it only on the other, in both
  directions
- Secondary readout: fixed final-epoch EMA-0.995 from the same trajectory
- Ensemble: uniform tie-aware rank averaging; no learned validation weights
- Paired unit: candidate-minus-parent IC by date after averaging decisions and
  horizons; 10,000-replicate moving-block bootstraps at block lengths 5 and 10
- Guardrails: 30/60/120-minute and all 55 decision-time deltas
- Parent pairing: exact member names, seeds, folds, observations, targets, and masks
- Stage-one compute gate: seed 29 on both folds; stop only when raw Patience delta
  was at most -0.003 on both folds and EMA delta was at most -0.002 on both folds
- All six candidates passed that deliberately broad harm gate, so the final count
  was 720 checkpoints: 6 candidates x 2 folds x 3 seeds x 20 epochs
- Official validation access: none
- Held-out test access: none

The candidates were implemented as follows. Every injection projection was
zero-initialized after parent construction, preserving the parent's shared-module
initialization and exact epoch-zero function.

- `decision_time`: sine/cosine decision-phase coordinates projected residually
  into each normalized equity state.
- `temporal_stats`: masked causal mean and standard deviation of the final TCN
  block, projected into the final state as
  `final_state + zero_projection(causal_mean, causal_std)`.
- `multi_depth_stats`: masked causal mean/std pools from blocks 2, 4, and 6,
  projected residually into the final state.
- `cross_section_max_min`: active-equity max/min states projected into the existing
  cross-sectional mean/dispersion pool.
- `learned_set_pool`: shared `phi` with width 16, active masked mean, and a
  zero-started projection into the existing pool.
- `conditional_bucket_means`: active-equity means in three fixed standardized
  buckets each for causal `beta_to_WDO` and cross-sectional realized-volatility
  rank, injected through a zero-started projection. No threshold was fit on a
  validation period.

### Primary `patience3_raw` results

All values below are candidate minus the matched trajectory parent. No primary
mean was positive.

| Candidate | Fold A delta | Block-5 95% | Block-10 95% | Fold B delta | Block-5 95% | Block-10 95% | Mean delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| Decision time | -0.000017458 | [-0.000066579, 0.000023393] | [-0.000066067, 0.000025086] | -0.000001586 | [-0.000025538, 0.000027084] | [-0.000024347, 0.000027698] | -0.000009522 |
| Temporal stats | +0.000098414 | [-0.001185126, 0.001529769] | [-0.001161095, 0.001660097] | -0.000307062 | [-0.001929435, 0.001120799] | [-0.001757769, 0.001142709] | -0.000104324 |
| Multi-depth stats | +0.000531513 | [-0.002742947, 0.004481381] | [-0.002839944, 0.005219267] | -0.000827167 | [-0.004261297, 0.001741108] | [-0.004201801, 0.001764999] | -0.000147827 |
| Cross-sectional max/min | -0.000497629 | [-0.002067675, 0.001231708] | [-0.002084246, 0.001227718] | +0.000101368 | [-0.000375884, 0.000442713] | [-0.000363012, 0.000369244] | -0.000198130 |
| Learned set pool | -0.000009072 | [-0.000028979, 0.000004408] | [-0.000030251, 0.000003328] | -0.000003349 | [-0.000024022, 0.000017019] | [-0.000022071, 0.000016529] | -0.000006211 |
| Conditional bucket means | -0.000362585 | [-0.001668679, 0.000988903] | [-0.001659114, 0.000904938] | -0.000035139 | [-0.000473808, 0.000372378] | [-0.000486874, 0.000328671] | -0.000198862 |

The candidate ensemble ICs for Fold A/Fold B were `0.048399/0.050672`,
`0.048515/0.050366`, `0.048948/0.049846`, `0.047919/0.050775`,
`0.048407/0.050670`, and `0.048054/0.050638` in table order. The matched
parent ICs were always `0.048416/0.050673`.

### Secondary final EMA-0.995 results

| Candidate | Fold A delta | Block-5 95% | Block-10 95% | Fold B delta | Block-5 95% | Block-10 95% | Mean delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| Decision time | -0.000179524 | [-0.000323778, -0.000025011] | [-0.000317016, -0.000013000] | -0.000088033 | [-0.000224618, 0.000051428] | [-0.000195851, 0.000052946] | -0.000133778 |
| Temporal stats | +0.001687701 | [-0.000821536, 0.004584018] | [-0.001136599, 0.005219924] | -0.001668903 | [-0.005149351, 0.000887101] | [-0.005186566, 0.001220284] | +0.000009399 |
| Multi-depth stats | +0.001407586 | [-0.004976762, 0.008640091] | [-0.005871645, 0.010369604] | -0.003357589 | [-0.009829727, 0.001296815] | [-0.009856910, 0.000986787] | -0.000975001 |
| Cross-sectional max/min | +0.001690664 | [0.000124882, 0.003093798] | [0.000165529, 0.003024779] | -0.000146385 | [-0.001771570, 0.000834316] | [-0.001649641, 0.000614895] | +0.000772140 |
| Learned set pool | +0.000001792 | [-0.000033089, 0.000027919] | [-0.000033031, 0.000023055] | +0.000009126 | [-0.000047985, 0.000060654] | [-0.000046444, 0.000059766] | +0.000005459 |
| Conditional bucket means | +0.001751860 | [0.000644237, 0.002792420] | [0.000648120, 0.002778574] | -0.000902497 | [-0.002484332, 0.000426722] | [-0.002191569, 0.000385227] | +0.000424682 |

The two apparently significant Fold-A EMA gains for max/min and conditional
buckets both reversed on Fold B. EMA was a free secondary readout, not the frozen
selection criterion; neither fold-dependent pattern justifies changing the recipe.

### Primary horizon and time-of-day guardrails

| Candidate | Fold A 30m / 60m / 120m | Fold B 30m / 60m / 120m | Fold A TOD range | Fold B TOD range |
|---|---|---|---|---|
| Decision time | +0.000008 / -0.000010 / -0.000051 | +0.000007 / +0.000031 / -0.000043 | [-0.000258, 0.000235] | [-0.000144, 0.000179] |
| Temporal stats | +0.000253 / +0.000036 / +0.000007 | +0.000116 / -0.000360 / -0.000678 | [-0.005854, 0.004085] | [-0.008505, 0.006066] |
| Multi-depth stats | +0.001275 / +0.000840 / -0.000520 | -0.000016 / -0.000926 / -0.001540 | [-0.010400, 0.009149] | [-0.012621, 0.007158] |
| Cross-sectional max/min | +0.000173 / -0.000476 / -0.001190 | +0.000102 / +0.000064 / +0.000139 | [-0.004606, 0.002280] | [-0.001083, 0.001213] |
| Learned set pool | -0.000008 / -0.000001 / -0.000017 | +0.000001 / -0.000018 / +0.000006 | [-0.000185, 0.000165] | [-0.000142, 0.000171] |
| Conditional bucket means | +0.000142 / -0.000327 / -0.000903 | -0.000021 / -0.000130 / +0.000046 | [-0.003131, 0.002451] | [-0.001303, 0.001073] |

No rejected candidate showed a coherent positive horizon or time-of-day profile.

### Primary ensemble diagnostics

Member triplets are seed 11/29/47. Correlation is the mean of the three pairwise
prediction Spearman values. Gains are ensemble minus mean member / best member.

| Candidate | Fold A member ICs | Fold B member ICs | Mean correlation A / B | Gain A vs mean / best | Gain B vs mean / best |
|---|---|---|---|---|---|
| Decision time | .048470 / .047536 / .045711 | .050001 / .050433 / .048800 | .9212 / .9368 | +.001160 / -.000071 | +.000927 / +.000239 |
| Temporal stats | .048091 / .047469 / .046312 | .048410 / .049084 / .051471 | .9151 / .9425 | +.001224 / +.000424 | +.000711 / -.001105 |
| Multi-depth stats | .046954 / .046720 / .047574 | .048171 / .049436 / .047840 | .8644 / .8988 | +.001865 / +.001374 | +.001364 / +.000410 |
| Cross-sectional max/min | .048966 / .046652 / .045421 | .050162 / .050652 / .049113 | .9344 / .9421 | +.000906 / -.001047 | +.000799 / +.000122 |
| Learned set pool | .048516 / .047533 / .045689 | .050036 / .050411 / .048829 | .9214 / .9368 | +.001161 / -.000109 | +.000911 / +.000259 |
| Conditional bucket means | .049334 / .046391 / .045859 | .050020 / .050383 / .048951 | .9398 / .9389 | +.000859 / -.001280 | +.000854 / +.000255 |

Multi-depth pooling did create the intended extra seed diversity, but its primary
IC still fell on average. More identical seeds are not the remedy for any of these
representation changes.

### Decision

Reject all six candidates as standalone changes. Raw Patience-3 remains the
canonical parent. Do not spend the repeatedly consumed official validation split
on any of these non-winners, and keep the held-out test sealed.

The rejected variant branches, variant plumbing, campaign driver, and their tests
were removed from current HEAD under the deletion-first rule. Exact reproduction
uses implementation commit `732b1b0` and the completed immutable campaign:

`/lambda/nfs/brazil-rv-east3/quant-data/b3/processed/model_runs/phase_a_732b1b0_20260819T180348Z`

The manifest records `status=completed`, all six three-seed completions,
`official_validation_accessed=false`, and `test_accessed=false`. The generic strict
observation-level analyzer remains in current source for future experiments.

The campaign ran on Lambda instance `df8326b7265845bf8285546d9018ed86` in
`us-east-3`. After artifact and documentation checks, termination was accepted and
a subsequent provider query reported the exact ID absent.

Final source verification: Ruff and full Python syntax compilation passed. Before
the building reset, implementation commit `732b1b0` passed all 192 research tests
and every candidate passed compiled BF16 real-store forward/backward smoke checks
on the GH200. After the reset, the local full-suite rerun could not collect because
Windows Application Control blocked `torch.dll` with `WinError 4551`, including in
a fresh isolated `uv` environment. The deletion-first model/training files were
therefore also byte-compared with the previously tested parent and matched exactly;
the retained analyzer matched the 192-test-passing campaign commit exactly. Re-run
the full suite after the machine policy is cleared.

## Experiment 9 — Phase A adapter autopsy and diversity ensembles

Purpose: test the claim that the near-zero decision-time and learned-set results
came from dead zero-start paths, and test whether the decorrelated multi-depth
members add value when uniformly pooled with the parent. This used only saved
checkpoints and predictions; there was no training.

### Settings

- Source parent: `trajectory_discovery_e22dd67_20260819T134332Z`
- Source Phase A campaign: `phase_a_732b1b0_20260819T180348Z`
- Folds and seeds: Fold A/Fold B and seeds 11/29/47
- Primary readout: separately replayed odd/even cross-fitted `patience3_raw`
- Secondary readout: fixed final-epoch EMA-0.995
- Candidate ensembles: uniform within-sample/horizon tie-aware rank average of
  parent-3 plus multi-depth-3 (six members), and parent-3 plus multi-depth-3 plus
  temporal-stats-3 (nine members)
- Parent comparator: the original parent three-member rank ensemble
- Learned ensemble weights: none
- Paired inference: candidate-minus-parent daily IC, 10,000-replicate moving-block
  bootstrap at block lengths 5 and 10, with horizon and TOD guardrails
- Official validation and held-out test access: none

### Adapter autopsy

The proposed double-zero dead-lock did not occur. In the historical code, learned
set `phi` was standard-initialized and only its final projection was zeroed; its
pooled value already entered the existing nonlinear shared fusion. Both candidate
paths learned substantially:

| Path | Epoch-20 final-projection L2 range | Epoch-1-to-20 delta-norm range | Prediction Spearman vs matched parent |
|---|---:|---:|---:|
| Decision-time state adapter | 0.319-0.355 | 0.314-0.356 | 0.999134-0.999440 |
| Learned-set final projection | 0.490-1.107 | 0.487-1.105 | 0.999626-0.999890 |

Learned-set `phi` weights also moved materially. Raw-prediction RMSE relative to
the parent prediction standard deviation was about 3.6%-7.6% for decision time
and 2.1%-4.1% for learned set. The paths were active, but their rank-relevant
effects were tiny.

### Diversity-ensemble results

| Readout and ensemble | Fold A delta | Block-5 95% | Block-10 95% | Fold B delta | Block-5 95% | Block-10 95% | Mean delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| Patience: parent + multi-depth | +0.001236751 | [-0.000263872, 0.003034151] | [-0.000389246, 0.003377175] | +0.000284322 | [-0.001438356, 0.001526562] | [-0.001402278, 0.001524120] | +0.000760536 |
| Patience: parent + multi-depth + temporal | +0.000942357 | [-0.000448536, 0.002593614] | [-0.000558944, 0.002855912] | +0.000229962 | [-0.001370314, 0.001442489] | [-0.001310741, 0.001471717] | +0.000586159 |
| EMA: parent + multi-depth | +0.002562444 | [-0.000611509, 0.005889654] | [-0.001062603, 0.006706125] | +0.000738824 | [-0.002360407, 0.002914785] | [-0.002325517, 0.002708976] | +0.001650634 |
| EMA: parent + multi-depth + temporal | +0.002398249 | [-0.000493044, 0.005519021] | [-0.000928059, 0.006254678] | +0.000134722 | [-0.002977608, 0.002310791] | [-0.003016961, 0.002328570] | +0.001266486 |

The six-member Patience ensemble scored `0.049653012`/`0.050957597` versus
parent `0.048416261`/`0.050673275` on Fold A/Fold B. Adding temporal statistics
diluted the gain under both readouts. All intervals included zero, but the
six-member direction was positive on both folds and both predeclared readouts.

### Decision

Keep parent+multi-depth as the sole Phase A diversity-ensemble candidate for the
next sparse official-validation confirmation. Do not add temporal statistics and
do not claim statistical establishment from two folds. Raw Patience-3 on the
parent architecture remains the canonical trajectory parent for subsequent
representation experiments; the diversity ensemble is a secondary recipe, not a
replacement training objective or checkpoint rule.

Immutable output:

`/lambda/nfs/brazil-rv-east3/quant-data/b3/processed/model_runs/phase_a_autopsy_d237998_20260820T111500Z`

## Experiment 10 — Corrected decision-time shared-fusion rerun

Purpose: isolate the remaining plausible routing criticism by replacing the
historical uniform state shift with a nonlinear shared-fusion decision context,
while removing the accidental RNG-stream shift caused by candidate construction.

### Settings

- Runnable experiment commits: `9828f7219efbda1cb3d9aef89217423bd7e65feb`
  and checkpoint-provenance fix `b8d955a71a0c6a20be0861d4a6bfd2330d1da65b`
- Parent campaign: `trajectory_discovery_e22dd67_20260819T134332Z`
- Candidate input: sine/cosine decision phase
- Candidate path: standard-initialized `2 -> 16 -> 16` GELU embedding; zero-only
  final `16 -> 128` projection added to the shared mean/dispersion context before
  the incumbent nonlinear fusion
- Parent start: exact shared-module weights and exact epoch-zero predictions
- RNG control: adapter construction inside `torch.random.fork_rng`, verified to
  leave the post-construction RNG state identical to the parent
- Gradient smoke: after 10 soft-Spearman optimization steps, both the final
  projection and upstream embedding changed; the full research suite passed
  188/188 on the exact experiment commit
- Training: Fold A/Fold B, seeds 11/29/47, fixed 20-epoch SAM trajectories;
  120 checkpoints total. The three seeds ran in isolated processes within each
  fold; each retained independent sampler and PyTorch RNG state.
- Reporting: frozen odd/even cross-fitted raw Patience-3 primary and final
  EMA-0.995 secondary, paired to the matched parent with the canonical analyzer
- Official validation and held-out test access: none

### Results

| Readout | Fold A delta | Block-5 95% | Block-10 95% | Fold B delta | Block-5 95% | Block-10 95% | Mean delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| `patience3_raw` | -0.000001432 | [-0.000022193, 0.000017576] | [-0.000025859, 0.000018291] | -0.000008622 | [-0.000029098, 0.000016326] | [-0.000028747, 0.000018578] | -0.000005027 |
| `final_ema_0995` | -0.000000824 | [-0.000031588, 0.000022928] | [-0.000030386, 0.000020209] | -0.000000765 | [-0.000046207, 0.000039875] | [-0.000043633, 0.000031953] | -0.000000795 |

Patience candidate ICs were `0.048414829`/`0.050664653` versus parent
`0.048416261`/`0.050673275`. Primary 30/60/120-minute deltas were
`+0.000016572/+0.000005019/-0.000025886` on Fold A and
`-0.000014128/-0.000007322/-0.000004417` on Fold B. TOD ranges were
`[-0.000169979, +0.000182903]` and `[-0.000214485, +0.000189598]`.

The corrected path was unquestionably active. Epoch-20 final-projection norms
ranged `0.299-0.992`; upstream embedding weights also moved materially on every
fold/seed. Nevertheless, both readouts reproduced the parent to numerical-noise
scale on both folds.

### Decision

Reject decision-time embedding conclusively. The original and corrected routes
both produce null results, while checkpoint norms and the 10-step assertion rule
out dead gradients. Do not spend official validation on it. The rejected adapter,
variant plumbing, campaign driver, and candidate-specific tests were removed from
current HEAD; reproduction uses the commits and immutable artifact:

`/lambda/nfs/brazil-rv-east3/quant-data/b3/processed/model_runs/decision_time_fusion_b8d955a_20260820T113924Z`

The completed manifest records 120 checkpoints, exact commit `b8d955a`,
`official_validation_accessed=false`, and `test_accessed=false`.

## Experiment 11 — Phase B immutable auxiliary-target sidecar

Purpose: implement and audit the target-decomposition contract before any GPU
training. This sidecar was separate from the canonical feature store and was
never required by ordinary parent training or evaluation.

### Contract

- Implementation commits: `a04d63e` with the validation-truncation correction in
  `15471e8`.
- Causal beta: stored pre-neutralization `beta_to_WIN`, emitted before the daily
  beta update.
- Market component: the exact future WIN return from decision-bar open to the
  same close endpoint used by the equity label.
- Endpoint rule: both exact WIN endpoints must be observed; stale prices are not
  accepted.
- Residual: `equity_return - beta_to_WIN * WIN_return`, cross-sectionally
  median-centered, divided by the existing causal volatility scale times the
  square root of the horizon, then cross-sectional tie-aware midranked.
- Sign target: whether the equity return is above the contemporaneous
  cross-sectional median.
- Magnitude target: absolute median-centered return divided by the existing
  causal volatility scale times the square root of the horizon.
- Ordinary feature/label arrays were not mutated. The official model head and
  main soft-Spearman target were unchanged.

### Training-scope audit

The audit used only the 716 training dates, 2021-08-16 through 2024-06-28.

| Horizon | Main labels | Beta coverage | Exact WIN endpoint | Matched residual | Residual/main rank corr. | Mean absolute rank shift | Factor/main RMS | Positive sign rate | Mean magnitude |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 30m | 4,916,832 | 0.998564 | 0.999974 | 0.998538 | 0.960993 | 0.093272 | 0.567132 | 0.483645 | 0.503436 |
| 60m | 4,921,547 | 0.998564 | 0.999974 | 0.998538 | 0.960573 | 0.093347 | 0.573532 | 0.489913 | 0.480353 |
| 120m | 4,935,876 | 0.998567 | 0.999948 | 0.998515 | 0.959743 | 0.094033 | 0.581310 | 0.493688 | 0.448421 |

Mutation tests passed for post-exit invariance, exact-exit sensitivity, missing
endpoint masking, and beta emit-before-update. The audit SHA-256 is
`a2f1ef5cbc6fd3c293d6da8e2ded6f873bc9183fe3ba353a8a6d3d55766fb6c5`.
The immutable sidecar is:

`/lambda/nfs/brazil-rv-east3/quant-data/b3/processed/auxiliary_targets/phase_b_aux_15471e8_20260820T141500Z`

Its manifest records `official_validation_evaluated=false` and
`test_accessed=false`.

## Experiment 12 — Phase B auxiliary-target trajectories

Purpose: test residual-rank, sign, magnitude, and combined auxiliary supervision
both as standalone models and as diversity members.

### Settings

- Runnable campaign commit: `6b7b121adf465327dd17ca87c436fe9033fa7686`.
- Discovery folds: Fold A fit first 512 training dates and selected on the next
  102; Fold B fit first 614 and selected on the final 102. The selection periods
  do not overlap.
- Seeds: 11, 29, and 47.
- Training: one fixed 20-epoch SAM trajectory per fold/seed/candidate with the
  canonical optimizer, schedule, sampler, and parent RNG initialization.
- Matrix: four candidates times two folds times three seeds, 24 trajectories and
  480 raw checkpoint files.
- Main loss: canonical soft Spearman, unchanged.
- Single auxiliary: total weight 0.5. Residual used soft Spearman; sign used
  binary cross-entropy with logits; magnitude used smooth L1.
- Combined bundle: equal mean of the three auxiliary losses with the same fixed
  total weight 0.5, not three separately weighted losses.
- Auxiliary heads: zero-initialized weights and biases; construction preserved
  the parent RNG stream and base-module initialization.
- Primary readout: separately replayed odd/even cross-fitted Raw Patience-3.
- Free secondary readout: fixed final-epoch EMA-0.995.
- Ensembles: uniform within-sample/horizon tie-aware ranks. No weights were
  learned. Each candidate was measured standalone, parent-3 plus candidate-3,
  and parent-3 plus Phase-A multi-depth-3 plus candidate-3.
- Inference: paired per-date candidate-minus-parent IC with 10,000-replicate
  moving-block bootstrap at lengths 5 and 10, plus horizon and TOD guardrails.
- Official validation and held-out test: not accessed.

### Standalone results

| Candidate | Raw Patience Fold A | Raw Patience Fold B | Primary mean | EMA Fold A | EMA Fold B | EMA mean |
|---|---:|---:|---:|---:|---:|---:|
| Residual rank | +0.000418 | -0.001669 | -0.000625 | +0.001439 | +0.000988 | +0.001214 |
| Sign | -0.000936 | -0.000028 | -0.000482 | -0.000143 | +0.000201 | +0.000029 |
| Magnitude | -0.003143 | -0.002054 | -0.002599 | -0.003219 | +0.000101 | -0.001559 |
| Combined bundle | -0.000641 | -0.000061 | -0.000351 | +0.001055 | +0.001382 | +0.001218 |

Residual-rank Fold B was negative with block-5 95% interval
`[-0.002757, -0.000101]` and block-10 interval
`[-0.002260, -0.000286]`. Magnitude Fold A was also negative with intervals
`[-0.005935, -0.000331]` and `[-0.005508, -0.000627]`. Sign and combined were
null. No primary candidate improved both folds, so the EMA-positive residual and
combined patterns were retained only as the predeclared secondary readout and
could not change selection.

### Diversity results

| Candidate members | Raw parent+candidate mean delta | Raw full-stack mean delta | EMA parent+candidate mean delta | EMA full-stack mean delta |
|---|---:|---:|---:|---:|
| Residual rank | -0.000082 | +0.000620 | +0.000892 | +0.002108 |
| Sign | -0.000193 | +0.000557 | +0.000029 | +0.001569 |
| Magnitude | -0.000415 | +0.000452 | +0.000288 | +0.001811 |
| Combined bundle | -0.000029 | +0.000695 | +0.000849 | +0.002046 |

The pre-existing Phase-A parent+multi-depth baseline was `+0.000761` on the raw
primary readout. No Phase B full stack exceeded it and added value on both folds;
therefore no Phase B member was retained. Magnitude generated the most diversity
but lost too much standalone quality. The conditional common-component head and
recombination diagnostic were correctly skipped because residual rank did not win
the primary discovery screen.

The completed campaign is:

`/lambda/nfs/brazil-rv-east3/quant-data/b3/processed/model_runs/phase_b_6b7b121_20260820T145500Z`

All 24 trajectories and 52 analysis reports completed. The campaign manifest and
selection file record `official_validation_accessed=false` and
`test_accessed=false`.

## Experiment 13 — Three-epoch recency fine-tuning

Purpose: test one, two, and three recent-window epochs and the 50/50 full-history
plus fine-tuned rank ensemble from one trajectory per seed/fold/direction.

### Settings

- Source: the unchanged full-history parent. No auxiliary model qualified on both
  folds.
- Recent window: most recent 120 dates of each fit window.
- Learning rate: `0.00003`, one tenth of the parent rate.
- Epochs: three, saving after epochs 1, 2, and 3.
- Honest evaluation: select the source Raw Patience-3 checkpoint on one date
  parity, fine-tune it, and evaluate on the opposite parity; run both directions.
- Matrix: two folds times three seeds times two directions times three epochs,
  36 checkpoints.
- Comparisons: fine-tuned model alone and a fixed 50/50 rank ensemble with its
  full-history source.
- Official validation and held-out test: not accessed.

| Epoch | Fine-only Fold A / Fold B / mean | 50/50 Fold A / Fold B / mean |
|---:|---|---|
| 1 | -0.001260 / +0.001500 / +0.000120 | -0.000543 / +0.000882 / +0.000169 |
| 2 | -0.002511 / +0.002299 / -0.000106 | -0.001030 / +0.001544 / +0.000257 |
| 3 | -0.002737 / +0.002844 / +0.000053 | -0.000930 / +0.001843 / +0.000457 |

Every rule showed the same Fold-A-down/Fold-B-up split. For the nominal best,
the epoch-3 50/50 ensemble, Fold A's block-10 interval was
`[-0.001686, -0.000320]` while Fold B's was `[+0.000412, +0.003094]`.
The predeclared both-fold guardrail therefore selected `full_history`; no recency
checkpoint or ensemble was retained. The completed recency manifest and analysis
live inside the Experiment 12 campaign directory and record
`official_validation_accessed=false` and `test_accessed=false`.

## Experiment 14 — Sparse official confirmation of the final stage recipe

Purpose: after Phase B concluded, spend one official-validation read on the only
remaining stage-level recipe: the three parent members plus the three Phase-A
multi-depth diversity members. No Phase B member or recency model qualified.

### Settings

- Multi-depth implementation: immutable commit `732b1b0`.
- Frozen selection rule: Raw Patience-3 from
  `trajectory_crossfit_3054228_20260819T161200Z/trajectory_selection.json`.
- Multi-depth training: all 716 training dates, seeds 11/29/47, fixed 20-epoch
  trajectories, 60 checkpoints total.
- Parent comparator: the three matched official parent-reproduction runs from
  Experiment 1.
- Candidate: uniform rank average of parent-3 plus multi-depth-3. Comparator:
  uniform rank average of parent-3. No learned weights.
- Manual validation-only analyzer commit: `e33a122`; it exposed no test selector,
  required exactly the three allowed seeds and 244 validation dates, and verified
  completed manifests, feature-store identity, frozen rule, and test-clean flags.

### Result

| Recipe | Official validation IC |
|---|---:|
| Parent three-seed ensemble | 0.041639843 |
| Parent-3 + multi-depth-3 | 0.040495819 |
| Candidate minus parent | -0.001144024 |

The paired block-5 95% interval was `[-0.002990, +0.000614]`; block-10 was
`[-0.003185, +0.000623]`. Horizon deltas were
`-0.000135/-0.001324/-0.001974` at 30/60/120 minutes. TOD deltas ranged
`[-0.006786, +0.002753]`. The six-member ensemble gained more over its mean member
than the parent ensemble (`+0.002075` versus `+0.001334`), but the multi-depth
members were only `0.036006-0.037464` IC versus parent members
`0.038464-0.041978`; diversity could not compensate for lower quality.

### Final Phase B decision

Reject the Phase-A six-member recipe on official validation. Retain the
three-seed parent with Raw Patience-3 as the sole canonical recipe. Do not open the
held-out test for this rejected stage. Official validation is consumed/closed
again; `test_accessed=false` throughout.

Official training and comparison artifacts are:

`/lambda/nfs/brazil-rv-east3/quant-data/b3/processed/model_runs/phase_a_official_732b1b0_20260820T201500Z`

Deletion-first cleanup removed the rejected Phase B sidecar/training/recency
plumbing, auxiliary heads, one-use official-confirmation driver, and specific
tests from current HEAD. Exact reproduction remains in commits
`a04d63e`, `15471e8`, `6b7b121`, `e33a122`, and the immutable NFS artifacts.

The paid GH200 instance was `5e9201fcd5b6436cbdd3be9fe9ee4524` in
`us-east-3`. After all artifacts, tests, commits, and GitHub updates were complete,
Lambda accepted termination and reported `terminating`; a subsequent provider
inventory poll returned zero matches for the exact ID. Persistent results remain
on the attached `brazil-rv-east3` NFS filesystem.


## Experiment 15 — Official-validation staleness profile

Purpose: determine whether the gap between the stronger internal discovery folds and
the canonical parent's `0.041640` official-validation IC is consistent with
post-training staleness.

### Settings

- Source: the three stored parent-reproduction validation observation files; no
  model training or new prediction generation.
- Scope: all 244 already-consumed official-validation dates.
- Statistic: daily primary IC from the uniform three-seed rank ensemble, quarterly
  means, H1-2025 minus H2-2024, and a linear IC slope on calendar days since the
  2024-06-28 training end.
- Inference: 10,000-replicate moving-block bootstrap at block lengths 5 and 10.
- Selection: diagnostic for retraining cadence only. It was not available to the
  Stage 2 architecture campaign.
- Held-out test: not accessed.

### Result

| Period | Mean daily IC |
|---|---:|
| 2024 Q3 | 0.040053 |
| 2024 Q4 | 0.045006 |
| 2025 Q1 | 0.045948 |
| 2025 Q2 | 0.035552 |

H1-2025 minus H2-2024 was `-0.001779`, with block-5 interval
`[-0.021436, +0.018751]` and block-10 interval
`[-0.022423, +0.018695]`. The fitted slope was slightly positive,
`+0.000131` IC per 100 calendar days. The late Q2 weakness is real in the point
estimate, but neither the half-year comparison nor the slope supports a systematic
staleness claim. Do not change retraining cadence from this diagnostic; interpret
the official stage winner against the observed quarterly variability.

Artifact:

`/lambda/nfs/brazil-rv-east3/quant-data/b3/processed/model_runs/next_stage_c0d0598_20260820T225000Z/d1_staleness`

## Experiment 16 — Stronger residual-target gate and corrected immutable sidecar

Purpose: determine whether target decomposition becomes meaningfully different
after less-shrunk and multi-factor residualization, before spending GPU time.

### Settings

- Short WIN beta: five-session EWMA half-life, no clipping, no variance floor.
- Two-factor target: stored causal pre-neutralization WIN and WDO betas times exact
  future WIN/WDO returns.
- Three-factor target: WIN + WDO plus the ready-contract mean exact DI basis-point
  level change against the stored causal `beta_to_DI1F28`.
- Exact endpoint rule: decision-bar open to the equity label's matching close;
  both endpoints observed, no stale prices.
- Residual transformation: subtract factor component, cross-sectionally
  median-center, apply the existing causal volatility/horizon normalization, then
  tie-aware midrank.
- Audit scope: the 716 training dates only. Residual arrays are zero/masked on
  every non-training date; causal C3 tilt exposures alone extend through validation.
- Gate: train exactly one candidate only if the lowest residual/main rank
  correlation is at most `0.90`.
- Held-out test and official-validation targets: not accessed.

### Audit result

| Residualization | Aggregate rank corr. | 30m / 60m / 120m matched coverage | Factor/main RMS, 30m / 60m / 120m |
|---|---:|---|---|
| Short unclipped WIN | 0.938757 | 0.9985 / 0.9985 / 0.9985 | 0.611 / 0.618 / 0.625 |
| WIN + WDO | 0.912874 | 0.9981 / 0.9980 / 0.9979 | 0.858 / 0.869 / 0.884 |
| WIN + WDO + DI level | 0.861604 | 0.8491 / 0.8462 / 0.7807 | 1.124 / 1.138 / 1.151 |

The three-factor target passed the gate and was selected for exactly one
three-seed candidate. Its horizon correlations were
`0.864586 / 0.860850 / 0.859186`, and mean absolute rank shifts were
`0.18797 / 0.19064 / 0.19129`. All mutation tests passed: post-exit changes were
invariant, missing exact exits masked the factor, and the short beta emitted
before the current update.

The first immutable artifact, built under `c0d0598`, incorrectly required three
simultaneous fixed-DI endpoints and was rejected before training. It remains
immutable for forensic reproducibility. Commit `3b60ac9` corrected the contract
to the mean over all endpoint-ready fixed contracts with at least one required,
and also sealed all residual targets at the training boundary.

Corrected artifact:

`/lambda/nfs/brazil-rv-east3/quant-data/b3/processed/auxiliary_targets/next_stage_3b60ac9_20260820T233000Z`

The same sidecar records C3's causal DI-tilt exposure at `99.8555%` readiness
among active training equity-date cells.

## Experiment 17 — Patience plus EMA rank blend

Purpose: test whether the complementary horizon/trajectory profiles of Raw
Patience-3 and EMA-0.995 improve the canonical parent without retraining.

### Settings

- Source: saved per-epoch predictions in
  `trajectory_discovery_e22dd67_20260819T134332Z`.
- Per fold/parity/seed: select Raw Patience-3 on one date parity; rank-average its
  out-of-half predictions 50/50 with either final EMA-0.995 or EMA-0.995 at the
  selected epoch; repeat both directions.
- Ensemble: uniform rank average of the three seed members.
- Inference: paired daily delta with 10,000-replicate block-5/10 bootstrap.
- Retention: strictly positive primary delta on both discovery folds.
- Official validation and held-out test: not accessed.

| Blend | Fold A delta | Fold B delta | Mean | Retained |
|---|---:|---:|---:|---|
| Raw Patience + final EMA-0.995 | -0.000123 | +0.001292 | +0.000585 | No |
| Raw Patience + selected-epoch EMA-0.995 | -0.002978 | -0.000598 | -0.001788 | No |

The final-EMA blend was mildly additive only on Fold B; its intervals included
zero on both folds. The selected-epoch EMA variant was significantly harmful on
Fold A. Reject both and retain Raw Patience-3 unchanged.

Artifact:

`/lambda/nfs/brazil-rv-east3/quant-data/b3/processed/model_runs/next_stage_c0d0598_20260820T225000Z/r1_rank_blend`


## Experiment 18 — Gated three-factor residual auxiliary

Purpose: train the one auxiliary candidate authorized by Experiment 16's
predeclared correlation gate.

### Settings

- Target: the selected WIN + WDO + ready-DI-level residual rank from the immutable
  `3b60ac9` sidecar.
- Main objective and official head: canonical soft Spearman, unchanged.
- Auxiliary objective: residual-rank soft Spearman at fixed weight `0.5` through a
  separate zero-initialized head.
- Initialization: parent modules and RNG stream matched exactly; only the final
  auxiliary head was zero-initialized.
- Matrix: folds A/B × seeds 11/29/47, one fixed 20-epoch SAM trajectory each,
  120 raw checkpoint files.
- Primary: odd/even cross-fitted Raw Patience-3. Secondary: fixed final
  EMA-0.995.
- Inference: paired daily deltas with 10,000-replicate block-5/10 bootstrap plus
  horizon and TOD guardrails.
- Official validation and held-out test: not accessed.

| Readout | Fold A delta | Fold B delta | Mean | Positive both folds |
|---|---:|---:|---:|---|
| Raw Patience-3 | +0.000286 | -0.000452 | -0.000083 | No |
| Final EMA-0.995 | +0.001731 | +0.001879 | +0.001805 | Yes |

For the primary readout, Fold A's block-5/10 intervals were
`[-0.001055, +0.001882]` and `[-0.001083, +0.001861]`; Fold B's were
`[-0.001532, +0.001311]` and `[-0.001133, +0.001168]`. The stronger
residualization therefore remained null under the frozen primary rule. The
secondary EMA pattern is directionally coherent with Experiment 12's residual
readout, but its fold intervals also cross zero and it cannot change selection.

Decision: do not retain this as a standalone candidate. Because the D2 gate opened,
the attached plan predeclared it as a Stage 3 diversity member regardless of its
standalone result; that stack inclusion remains fixed and does not authorize an
EMA reselection.

Analysis artifact:

`/lambda/nfs/brazil-rv-east3/quant-data/b3/processed/model_runs/next_stage_3b60ac9_20260820T233000Z/phase_c/analysis/residual_auxiliary`


## Experiment 19 — C1 compressed global-risk state

Purpose: test whether compressed causal summaries from global instruments add value
after full masked ES/NQ/CL/HG streams failed.

### Settings

- Inputs: causal ES 30-minute normalized return, ES 30-minute realized-volatility
  log ratio, HG 30-minute normalized return, and 6M 30-minute normalized return.
- Route: standard-initialized width-16 encoder into a zero-initialized projection
  added to the existing shared fusion hidden state.
- Parent preservation: exact at initialization, including parent module values and
  RNG stream.
- Matrix/readouts/inference: the standing six-trajectory discovery protocol,
  cross-fitted Raw Patience-3 primary, final EMA-0.995 secondary, paired
  block-5/10 bootstrap, horizon/TOD guardrails.
- Additional guardrail: low-versus-high daily ES volatility, split at each fold's
  median causal ES volatility state.
- Official validation and held-out test: not accessed.

| Readout | Fold A delta | Fold B delta | Mean | Positive both folds |
|---|---:|---:|---:|---|
| Raw Patience-3 | -0.000029 | +0.000001 | -0.000014 | No |
| Final EMA-0.995 | -0.000090 | +0.000148 | +0.000029 | No |

The primary high-ES-vol deltas were `-0.000081` on Fold A and `-0.000008` on
Fold B; the adapter did not earn its keep in the regime where the mechanism
predicted its benefit.

Checkpoint autopsy rules out a dead path. Epoch-20 final adapter norms were
`1.338–1.528` across all six runs, and the standard-initialized upstream encoder
weights and biases also moved. The unusually small prediction deltas are therefore
a genuine functional null after training, not a zero-init lock.

Decision: reject C1 and do not spend official validation on it.

Analysis artifact:

`/lambda/nfs/brazil-rv-east3/quant-data/b3/processed/model_runs/next_stage_3b60ac9_20260820T233000Z/phase_c/analysis/compressed_global_risk`


## Experiment 20 — C2 low-rank factor mixer, K=4

Purpose: test cross-equity interaction through a crosstalk-safe low-rank mixer.

### Settings

- Four learned factor queries attend over masked equity states.
- Source state: masked causal temporal mean plus the existing slow projection.
- Per-equity loadings: softmax loadings computed from each equity's own smoothed
  state.
- Route: zero-initialized final projection adds the mixed factor state to the fast
  final equity state; the fast state is otherwise untouched.
- Matrix/readouts/inference: the standing six-trajectory protocol with cross-fitted
  Raw Patience-3 primary and final EMA-0.995 secondary.
- Extension gate: K=8 and set-pool×mixer require primary mean at least `+0.001`
  and neither fold negative.
- Official validation and held-out test: not accessed.

| Readout | Fold A delta | Fold B delta | Mean | Positive both folds |
|---|---:|---:|---:|---|
| Raw Patience-3 | -0.000095 | -0.003950 | -0.002022 | No |
| Final EMA-0.995 | -0.000072 | -0.000545 | -0.000308 | No |

The primary Fold B result was significantly negative: block-5 interval
`[-0.005820, -0.001470]` and block-10 interval
`[-0.005381, -0.001585]`. Fold A was null. The fixed EMA readout softened the
damage but was also negative on both folds.

Decision: reject K=4. The extension gate failed by a wide margin, so K=8 and the
set-pool×mixer stack were correctly skipped with zero additional training.

Analysis artifact:

`/lambda/nfs/brazil-rv-east3/quant-data/b3/processed/model_runs/next_stage_3b60ac9_20260820T233000Z/phase_c/analysis/factor_mixer_k4`


## Experiment 21 — C3 causal DI-curve tilt exposure

Purpose: test whether the contemporaneous cross-equity residual-loading structure
contains signal that the existing sequence representation does not expose directly.

### Settings

- Feature: causal projection of each equity's available short beta vector onto the
  contemporaneous fixed-DI level direction from the immutable `3b60ac9` sidecar.
- Availability: `99.8555%` among active training equity-date cells; unavailable
  cells were masked rather than imputed.
- Route: the scalar, equity-varying exposure enters the existing slow-state fusion
  through a zero-initialized final projection.
- Matrix/readouts/inference: the standing six-trajectory protocol with cross-fitted
  Raw Patience-3 primary, final EMA-0.995 secondary, paired block-5/10 bootstrap,
  and horizon/TOD guardrails.
- Official validation and held-out test: not accessed.

| Readout | Fold A delta | Fold B delta | Mean | Positive both folds |
|---|---:|---:|---:|---|
| Raw Patience-3 | -0.000005 | +0.000002 | -0.000001 | No |
| Final EMA-0.995 | -0.000032 | -0.000025 | -0.000028 | No |

The primary block-5/10 intervals were `[-0.000029, +0.000020]` and
`[-0.000029, +0.000021]` on Fold A, and `[-0.000018, +0.000026]` and
`[-0.000017, +0.000027]` on Fold B. These are numerical nulls, not evidence of
a dead adapter: final projection norms were `0.593–0.707` across all six runs.

Decision: reject C3. With C1–C3 all null, the plan's predeclared gate opens the
width/regularization controls C4 and C5; it does not reopen any feature extension.

Analysis artifact:

`/lambda/nfs/brazil-rv-east3/quant-data/b3/processed/model_runs/next_stage_3b60ac9_20260820T233000Z/phase_c/analysis/di_tilt_exposure`

## Experiment 22 — C4 width/regularization capacity screen

Purpose: test the last cheap capacity axis after all three initial Phase C
representations failed the retention rule.

### Settings

- Architecture: width `96` and fusion width `192`, versus the parent's `128` and
  `256`; all other architecture and training choices frozen.
- Regularization: AdamW weight decay `0.02`, exactly twice the parent setting.
- Scope: one predeclared candidate, no width or decay sweep.
- Matrix/readouts/inference: the standing six-trajectory protocol with cross-fitted
  Raw Patience-3 primary, final EMA-0.995 secondary, paired block-5/10 bootstrap,
  and horizon/TOD guardrails.
- Official validation and held-out test: not accessed.

| Readout | Fold A delta | Fold B delta | Mean | Positive both folds |
|---|---:|---:|---:|---|
| Raw Patience-3 | +0.000751 | -0.002182 | -0.000716 | No |
| Final EMA-0.995 | -0.009056 | -0.007347 | -0.008201 | No |

The primary Fold A block-5/10 intervals were `[-0.001181, +0.002560]` and
`[-0.000885, +0.002599]`; Fold B's were `[-0.003839, +0.000364]` and
`[-0.003276, +0.000357]`. The fold reversal was most pronounced at 120 minutes:
`-0.000116` on Fold A versus `-0.004031` on Fold B. The fixed EMA readout was
uniformly harmful and cannot rescue the candidate.

Decision: reject C4. The one-candidate screen supplies no reason to sweep width or
regularization.

Analysis artifact:

`/lambda/nfs/brazil-rv-east3/quant-data/b3/processed/model_runs/next_stage_3b60ac9_20260820T233000Z/phase_c/analysis/capacity_96`

## Experiment 23 — C5 competitive market-state feature gate

Purpose: test the final predeclared architecture candidate, a MASTER-style
competitive multiplicative gate conditioned on compressed causal market state.

### Settings

- Conditioning state: the same four causal compressed global-risk inputs audited
  in C1.
- Gate: standard-initialized width-16 encoder followed by a zero-initialized output
  projection; softmax competition over feature channels at fixed temperature `2`.
- Route: multiplicative dynamic gating of input features, with exact parent output
  at initialization.
- Matrix/readouts/inference: the standing six-trajectory protocol with cross-fitted
  Raw Patience-3 primary, final EMA-0.995 secondary, paired block-5/10 bootstrap,
  and horizon/TOD guardrails.
- Official validation and held-out test: not accessed.

| Readout | Fold A delta | Fold B delta | Mean | Positive both folds |
|---|---:|---:|---:|---|
| Raw Patience-3 | +0.000024 | -0.000134 | -0.000055 | No |
| Final EMA-0.995 | +0.000283 | +0.000274 | +0.000279 | Yes |

The primary Fold A block-5/10 intervals were `[-0.000255, +0.000221]` and
`[-0.000288, +0.000230]`; Fold B's were `[-0.000892, +0.000741]` and
`[-0.000691, +0.000775]`. Every secondary interval also crossed zero. The EMA
pattern is directionally coherent but secondary-only and too small to override the
frozen rule. Final output-projection norms were `0.798–0.968`, and the upstream
encoder also moved in every run, ruling out a dead path.

Decision: reject C5. No Phase C candidate survived the both-fold primary rule.
The completed discovery campaign contains 36 trajectories and 720 raw epoch
checkpoints; its manifest is complete and records no official-validation or test
access.

Campaign artifact:

`/lambda/nfs/brazil-rv-east3/quant-data/b3/processed/model_runs/next_stage_3b60ac9_20260820T233000Z/phase_c`

## Experiment 24 — Sparse official confirmation of the next-stage stack

Purpose: spend the plan's single official-validation read on the fixed stack that
survived the prior gates.

### Settings

- Stack composition fixed before access: canonical parent seeds 11/29/47 plus
  full-history stronger-residual-auxiliary seeds 11/29/47. No Phase C candidate
  qualified, and R1 was null.
- Candidate training: official 716-date fit window, one fixed 20-epoch SAM
  trajectory per residual member, 60 raw checkpoints total.
- Selection: frozen Raw Patience-3 from the immutable selection-rule artifact.
- Ensemble: uniform within-sample/horizon tie-aware rank average of all six members;
  no learned weights.
- Comparison: matched parent-3 reproduction on the 244-date consumed official
  validation split, paired daily delta with 10,000-replicate block-5/10 bootstrap.
- Held-out test: not accessed; the driver exposes no test control.

| Recipe | Official validation IC |
|---|---:|
| Canonical parent-3 | 0.041639843 |
| Parent-3 + stronger-residual-3 | 0.042142944 |
| Candidate minus parent | +0.000503100 |

The paired block-5 interval was `[-0.000390, +0.001351]`; block-10 was
`[-0.000437, +0.001336]`. Horizon deltas were positive but small:
`+0.000608 / +0.000549 / +0.000353` at 30/60/120 minutes. The six-member
ensemble gained `+0.001371` over its mean member but remained `-0.000192` below
its best member. Parent/residual cross-family prediction correlations ranged from
`0.8737` to `0.9529`, providing real diversity but not enough evidence of a
repeatable recipe gain.

D1 found no material H1-2025 deterioration relative to H2-2024, so there is no
staleness result that changes the static interpretation or justifies a
walk-forward exception.

Decision: reject the stack as an accepted recipe change. It did not clear either
paired interval, so `held_out_test_read_justified=false`; the single held-out test
remains sealed. Raw Patience-3 parent-3 remains canonical.

Official artifact:

`/lambda/nfs/brazil-rv-east3/quant-data/b3/processed/model_runs/next_stage_official_921dd3a_20260821T085500Z`

### Deletion-first cleanup

After the null official result, all next-stage-only source and tests were removed.
Canonical `research/src` and `research/tests` are byte-for-byte identical to
accepted pre-experiment commit `a91c068`; exact reproduction uses commits
`c0d0598`, `3b60ac9`, and `921dd3a` and the immutable artifacts above. The
cleaned code passed 185 research tests plus 24 collector invariants, and Ruff.

### Paid-instance termination

The paid GH200 instance was `c6e81d007b354af98eaeec598902543c` in
`us-east-3`. Termination was requested only after both campaign manifests were
complete, all results were recorded, cleanup commit `09c0d12` passed 209 tests
and Ruff, and GitHub `main` was updated. Lambda accepted termination, reported
`terminating`, and the exact ID was then absent in two consecutive provider
inventory checks. Persistent experiment artifacts remain on the
`brazil-rv-east3` NFS filesystem.

## Experiment 25 — EMA residual-member stack rule reanalysis

Purpose: act once on the replicated interaction between auxiliary regularization
and the fixed checkpoint rule by isolating the saved residual members' readout.

### Settings

- Training: none. All predictions came from Experiment 18's six discovery
  trajectories.
- Candidate: parent-3 cross-fitted Raw Patience-3 plus residual-3 fixed final
  EMA-0.995.
- Comparator: the identical parent-3 Patience members plus residual-3 cross-fitted
  Raw Patience-3 members.
- Patience construction: select each member's raw epoch on one odd/even date
  parity and report only on the other, in both directions.
- Ensemble: uniform tie-aware within-sample/horizon rank average; no learned
  weights.
- Predeclared gate: candidate-minus-comparator IC at least `+0.001` on each fold
  individually. Only a passing gate could open the saved official predictions.
- Inference: paired daily delta with 10,000-replicate block-5/10 bootstrap plus
  horizon and TOD guardrails.

| Fold | EMA-member stack IC | Patience-member stack IC | Delta |
|---|---:|---:|---:|
| A | 0.049103342 | 0.048750860 | +0.000352482 |
| B | 0.053006013 | 0.050705923 | +0.002300089 |
| Mean | 0.051054677 | 0.049728392 | +0.001326286 |

Fold A's block-5/10 intervals were `[-0.002179, +0.002805]` and
`[-0.002249, +0.002854]`. Its 30/60/120-minute deltas were
`+0.001770 / +0.000576 / -0.001288`. Fold B was convincingly positive:
block-5 `[+0.001152, +0.003937]`, block-10 `[+0.001553, +0.003871]`, with
positive deltas at all three horizons.

Decision: the exact gate failed because Fold A gained only `+0.000352`, below
`+0.001`, despite the positive two-fold mean. The driver therefore never opened
the official artifacts. Its manifest records `official_validation_accessed=false`
and `test_accessed=false`. This resolves the repeated EMA signature honestly:
the effect is real in Fold B but not robust enough across both discovery periods
to support another consumed-validation read. The model-side program is closed.

Artifact:

`/lambda/nfs/brazil-rv-east3/quant-data/b3/processed/model_runs/ema_residual_stack_84ae363_20260821T114900Z`

### Deletion-first cleanup

The one-use analyzer and its specific tests were removed after the gate failed.
Exact reproduction uses implementation commit `84ae363` and the immutable artifact,
not compatibility code on current HEAD. Canonical source/tests returned exactly to
pre-experiment commit `68c6301`; 185 research tests, 24 collector invariants, and
Ruff passed.

### Paid-instance termination

The paid GH200 instance was `3985e78591e349549f7c99971c86fa9e` in
`us-east-3`. Termination was requested only after the immutable manifest was
complete, cleanup commit `7a6590e` passed verification, and GitHub `main` was
updated. Lambda reported `terminating`; two subsequent provider inventory checks
confirmed zero matches for the exact ID. The artifact remains on persistent
`brazil-rv-east3` NFS.

## Experiment 26 -- Storage, repository, and challenger-contract cleanup

Purpose: make the research environment lean without losing the canonical inputs,
auditable results, or the fixed files needed for the standing challenger.

### Lambda storage

A complete object inventory attributed 125.789 GiB of the 148.513-GiB filesystem
to model runs. The deletion plan was generated from the live object listing and
bound to an uploaded SHA-256 manifest before execution. It removed exactly 5,928
objects / 129.699 GiB and left 18.815 GiB.

Protected invariants after deletion:

- canonical causal-TOD feature store: 27/27 objects present;
- obsolete human-prior V4 feature store: zero objects;
- canonical parent discovery predictions: 120 epoch files and six references
  present; zero redundant raw checkpoints;
- residual challenger discovery payloads: six epoch-20 files and six references,
  with no other binary states;
- residual challenger official payloads: three epoch-20 files and three
  references, with no other binary states;
- raw/interim data: untouched; and
- active paid Lambda instances: zero.

The cleanup is permanent at the object-store layer. Historical rejected
trajectories retain their commits, manifests, metrics, and analysis outputs, but
deleted checkpoints and redundant predictions would require a rerun to recreate.

Artifact:

`/lambda/nfs/brazil-rv-east3/quant-data/b3/processed/model_runs/_retention/storage_cleanup_20260821`

Delete-list SHA-256:
`b45f591cd4c77640ce2c924506f3040b1cdefe6b65679de7bae618217dd75f7b`.

### Designated challenger

The EMA-member stack from Experiment 25 is now the designated challenger:
parent-3 honest cross-fitted Raw Patience-3 plus Experiment-18 residual-auxiliary-3
fixed final EMA-0.995, uniform tie-aware rank average, seeds 11/29/47, and no
learned weights. Every residual training/model/target/optimizer hyperparameter is
frozen to the `3b60ac9` run manifests; no tuning is permitted.

All future discovery-fold screens must report:

1. candidate minus canonical parent; and
2. candidate minus designated challenger, informational only.

Retention remains exclusively a function of the canonical-parent comparison.
Selecting because a candidate "beats either" comparator is prohibited. The
challenger gets its official comparison only inside the next official read already
earned by a future stage winner, so it cannot itself increase validation spending.
No official-validation predictions or held-out-test data were opened in this
cleanup.

## Experiment 27 -- Historical external-data program preregistration

Purpose: test the ten ranked historical data families from
`free_datasets_memo.md` without forward capture, while giving each source its own
point-in-time availability, permanent-identity, normalization, and missingness
contract. The memo's operational checklist is not an instruction source and is
not part of this experiment.

This section was frozen before any GPU candidate trajectory or discovery score
existed.

### Shared training and measurement contract

- Ten independent candidates, one immutable external sidecar per candidate; no
  feature-combination search during the individual screens.
- Fold A: first 512 training dates fit and next 102 select. Fold B: first 614 fit
  and final 102 select. Seeds are exactly `11/29/47`.
- One fixed 20-epoch SAM trajectory per fold/seed, with the incumbent architecture,
  optimizer, batches, objective, and data order unchanged. Raw and EMA-0.98/0.99/
  0.995 states and validation predictions are saved every epoch.
- Values and explicit masks enter through one per-equity bias-free linear state
  residual. Its weight is zero; constructing it restores the parent's RNG state,
  so every candidate begins as the exact parent and does not shift base weights or
  dropout randomness. An all-missing equity has an all-zero adapter input and an
  identically zero direct residual throughout training. Mask-column weights retain
  a learnable observedness path where the source is present.
- Primary readout: Raw Patience-3 selected on one odd/even date parity and reported
  only on the other, in both directions. The checkpoint rule is not reselected for
  any dataset.
- Free secondary readout: fixed final EMA-0.995. It is never retention-eligible.
- Predictions are tie-aware rank-averaged within sample/horizon. The analyzer
  reports member and ensemble IC, pairwise diversity, ensemble gain, paired daily
  candidate-minus-comparator IC, block-5/10 bootstrap intervals, and horizon/TOD
  guardrails. Ensemble weights are never learned.
- Every primary recipe reports against the canonical parent and the designated
  challenger. The challenger is informational only; retention is keyed exclusively
  to canonical-parent deltas, so "beats either" selection is prohibited.

Two primary roles are fixed in advance. A standalone candidate survives only if
its mean Fold-A/Fold-B IC gain is at least `+0.001` and each fold is non-negative.
The uniform parent-3 plus candidate-3 rank ensemble is a separate diversity-recipe
path with the same mean and both-fold gate; it additionally requires the standalone
candidate to lose no more than `0.001` on either fold. Passing either primary path
retains the dataset family for the later stack decision. The EMA and challenger
columns cannot rescue a failure. Official validation and the held-out test remain
inaccessible throughout these ten screens.

### Candidate contracts

1. **B3 lending open balance.** Three fixed-compressed features: 20-session ADV-
   scaled open-balance level and exact 5/20-session changes. Modern rows map by BDI
   ISIN; legacy rows use exact same-position-date COTAHIST ticker-to-ISIN mapping.
   A bulletin dated D is end-of-day data available at the next B3 open. Complete
   tables permit observed zero; missing/incomplete bulletins stay masked. Historical
   rates, fees, new-loan flow, and utilization are unavailable and are not tested.
   The retrievable BDI archive begins on 2022-03-21 (first usable sidecar date
   2022-03-22); 157 earlier requested sessions are explicit unmigrated endpoint
   failures, so Fold-A fit contains a real coverage-regime change rather than an
   invented zero history.
2. **SHFE ferrous/pulp.** Ten prior-only robust features from rebar, HRC, and pulp:
   same-contract 1/5-session returns, product curves, and HRC-minus-rebar spread.
   The contract is chosen with prior-session open interest. Exact Shanghai
   publication timestamps determine the first usable B3 session; steel and pulp
   values map only to six bounded permanent IDs. DCE iron ore and the live night
   session are unavailable historical scope and are not tested.
3. **COTAHIST options activity.** Eight fixed transforms covering option/stock
   quantity and turnover, put/call quantity and trades, near-expiry share,
   quantity-weighted moneyness, prior-20 quantity surprise, and exact lag-5 ratio
   change. COTAHIST's option row carries the underlying cash ISIN; D observations
   become usable next session. Per-series OI, covered/uncovered quantity, and IV
   are not inferred when their historical source payload is absent.
4. **CVM RAD events.** Five decision-level states: ITR/DFP, material-fact,
   market-communication, and IPE shareholder-notice events within five sessions,
   plus bounded log trading-minute age since the latest event. The shareholder-
   notice field is only a corporate-action proxy; the stopped optional expansion
   means provent and OPA categories are not claimed. Exact RAD receipt time activates
   at the first canonical decision strictly after receipt. Historical FCA plus exact
   same-date COTAHIST maps issuer/share classes; pending events and duplicated
   price-drift features are excluded.
5. **B3 odd-lot activity.** Eight next-session features: fixed transforms of odd-
   lot volume/trade shares, exact lag-5 changes, prior-20 median/MAD surprises,
   regular-versus-odd average trade value, and close ratio. Regular and odd COTAHIST
   rows join by exact ISIN. Buyer/seller imbalance is not present and is not
   inferred.
6. **B3 index rebalance.** Twenty-one decision-level fields for IBOV, IBXX, and
   SMLL: current weight, preview delta/add/delete/pressure, pre-effective ramp, and
   post-effective reversal. Historical official attachments activate at their
   exact archived HTTP timestamp; preview weights become current only at the
   effective-session open. The recovered archive contains 13 releases and 291 state
   dates beginning 2023-05-02; dates before that remain masked and are not called a
   complete 2021--2024 rebalance history. Permanent identity and prior ADV/close
   inputs are causal. MSCI is excluded because no clean free historical archive/
   license was found.
7. **CCEE PLD power state.** Eleven fixed, prior-only level/spread/range/change/
   surprise/floor/cap features plus five audited power-role masks. The complete D
   price curve is published D-1 at 20:00 and is usable from D open. ONS load/EAR
   files are excluded because current annual files are ex-post revised and no
   archived BDO-vintage contract was available.
8. **CVM structured fundamentals.** Nine fixed-clipped features: TTM margin, ROA,
   leverage, sales/assets growth, accruals, filing age, financial-sector flag, and
   consolidated-basis flag. Exact RAD versions, REAL/MIL scale, cumulative-ITR
   differencing, and historical FCA identities are audited. Exact receipts update
   at the first decision strictly after delivery; date-only fallbacks wait until
   next session. B/M and E/P are excluded because a clean issuer market-cap
   denominator across units/share classes is unavailable.
9. **Regular trade activity.** Six strictly-prior median/MAD surprises over 20/60
   observations for trade count, BRL per trade, and shares per trade. Shares-per-
   trade state resets only at a distribution-number change with a causal 25% price-
   unit discontinuity. D becomes usable next session. Historical after-hours
   metadata exists, but the official token currently returns zero-byte bodies, so
   after-hours activity is not tested or imputed.
10. **ADR overnight.** Four pair-specific fixed features for 18 audited ADR/local
    mappings: adjusted 1/5-session return, ADR-minus-EWZ return, and prior-only
    robust residual surprise. Only the last completed 16:00 New York close strictly
    before B3 10:15 is usable, with IANA DST handling. EWZ is not broadcast to
    unrelated names. No intraday, after-hours, parity, or FX claim is made.

The pre-GPU unified source-frame audit passed all ten candidates:

| Dataset | Cadence | Normalized rows | Feature count | Mapped scope |
|---|---|---:|---:|---|
| Lending | Daily | 63,647 | 3 | 146 securities; 447 valid bulletins |
| SHFE | Daily | 4,050 | 10 | 6 bounded steel/pulp securities |
| Options | Daily | 82,533 | 8 | 141 securities |
| RAD events | Intraday | 6,072,440 | 5 | 145 overall / 143 training securities |
| Odd lot | Daily | 258,845 | 8 | 505 source IDs; exact 158-axis join later |
| Index rebalance | Intraday | 2,526,262 | 21 | all 158 canonical IDs where applicable |
| CCEE PLD | Daily | 43,408 | 16 | 16 mapped / 13 active power names |
| Fundamentals | Intraday | 5,212,350 | 9 | 143 securities |
| Regular trade activity | Daily | 101,867 | 6 | 149 securities |
| ADR overnight | Daily | 13,068 | 4 | 18 audited ADR/local pairs |

All normalized frames must pass the generic materializer before training: exact
canonical feature-store identity and date/equity hashes, exact no-fill joins,
finite valid values, exactly-zero invalid values, permanent `security_id`, and
explicit masks. Daily arrays are `[date, 158, feature]`; intraday arrays are
`[date, 158, 55, feature]`. Sidecar and campaign manifests record source and array
hashes. No candidate may access official validation or test.

### Pre-score adapter correction

The first paid-instance preflight at repository commit `c7dbff6` was stopped
before any fold/seed trajectory completed and before any discovery analysis or
score existed. Its program root was
`external_data_c7dbff6_20260821T154100Z`. A live checkpoint-norm audit confirmed
that the external path woke, but also exposed a missing-row confound in the
preflight implementation: the zero-started linear projection had a trainable
bias, so after the first update an equity with `[values=0, masks=0]` received the
learned bias. This affected most cells for sparse sources such as SHFE, PLD, and
ADRs and tested generic residual capacity in addition to the source.

The preflight was rejected without interpretation. The corrected preregistered
adapter is bias-free, making the direct residual identically zero for every
all-missing equity while preserving learned mask effects for observed rows. New
tests require exact parent initialization/RNG, nonzero observed-feature gradient
flow, no bias checkpoint key, and exact missing-row nullity after assigning
nonzero adapter weights. No partial checkpoint or prediction from the rejected
preflight may enter the ten candidate screens.

## Experiments 28--37 -- Historical external-data discovery results

The preregistered ten-candidate program completed at repository commit
`7e535ac6a0b010ef6d9437d688f0ed40f172bdee`. All 60 fixed 20-epoch SAM
trajectories completed: ten candidates x two discovery folds x three seeds. The
strict result extractor accepted every campaign and analysis artifact, found no
contract errors, and verified `official_validation_accessed=false` and
`test_accessed=false` throughout.

Program artifact:

`/lambda/nfs/brazil-rv-east3/quant-data/b3/processed/model_runs/external_data_7e535ac_20260821T161800Z`

The strict consolidated JSON/Markdown summaries are frozen under
`_ops/final_results/` with SHA-256
`a670a1b3cd23257c8d96551bcfed07261348582f96103500852a59ea7d3e2627`
and `7a81599b4d528bff51bbd80c8dacc6d160accd9aea048252b75f4d0f18a5220b`.

Canonical feature-store metadata SHA-256:
`c90103b0f99e0017dc1303284a1ab61eca99106094227f5823ba718756d28a6b`.
Every run manifest records the bias-free zero-start external adapter; sampled
epoch-1 checkpoints confirmed a weight-only adapter with no bias key. Full member
ICs, all Patience replay directions, diversity matrices,
block-5/10 intervals, EMA/challenger comparisons, horizon/TOD guardrails, source
coverage, and adapter diagnostics remain in each immutable `analysis/<candidate>`
directory. The compact table below is the frozen program decision. Deltas are
candidate minus the canonical Raw-Patience-3 parent unless explicitly marked
informational.

| Exp. | Dataset | Standalone A | Standalone B | Mean | Parent+candidate A | Parent+candidate B | Mean | EMA standalone mean (info) | Standalone vs challenger mean (info) | Retained |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 28 | B3 lending | +0.000107 | -0.001113 | -0.000503 | +0.000060 | -0.000501 | -0.000220 | +0.000196 | -0.002013 | No |
| 29 | SHFE ferrous/pulp | +0.000265 | -0.000693 | -0.000214 | +0.000273 | -0.000236 | +0.000018 | +0.000599 | -0.001724 | No |
| 30 | Options activity | +0.000020 | +0.000494 | +0.000257 | +0.000149 | +0.000372 | +0.000261 | +0.000365 | -0.001253 | No |
| 31 | CVM RAD events | -0.002418 | +0.001158 | -0.000630 | -0.000977 | +0.000738 | -0.000119 | -0.000212 | -0.002140 | No |
| 32 | Odd-lot activity | -0.002593 | +0.000407 | -0.001093 | -0.001153 | +0.000250 | -0.000452 | -0.000434 | -0.002603 | No |
| 33 | Index rebalance | -0.000595 | -0.000131 | -0.000363 | -0.000152 | -0.000009 | -0.000080 | -0.000949 | -0.001873 | No |
| 34 | CCEE PLD | -0.001113 | +0.000735 | -0.000189 | -0.000185 | +0.000511 | +0.000163 | -0.000206 | -0.001699 | No |
| 35 | Fundamentals | -0.000695 | +0.001305 | +0.000305 | -0.000101 | +0.000954 | +0.000426 | -0.000448 | -0.001205 | No |
| 36 | Regular activity | +0.000080 | -0.000691 | -0.000305 | +0.000049 | -0.000260 | -0.000106 | +0.000303 | -0.001815 | No |
| 37 | ADR overnight | +0.001044 | -0.000757 | +0.000144 | +0.000725 | -0.000111 | +0.000307 | +0.000701 | -0.001366 | No |

No standalone or diversity recipe passed the frozen gate. Therefore no dataset
family was retained, no official-validation artifact was opened, and the
designated challenger's bundled official comparison was not triggered. This is a
recipe-level decision for the exact tested features, availability, adapter, and
training contract. Unavailable subfeeds explicitly excluded in Experiment 27
remain untested rather than rejected.

### Experiment 28 -- B3 lending open balance

The standalone Fold-A/Fold-B deltas were `+0.000107` and `-0.001113`, with
block-10 intervals `[-0.000040, +0.000223]` and
`[-0.001696, -0.000239]`. The six-member parent-plus-candidate deltas were
`+0.000060` and `-0.000501`, with block-10 intervals
`[-0.000019, +0.000124]` and `[-0.000783, -0.000069]`.
Standalone 30/60/120-minute deltas were
`+0.000060 / +0.000128 / +0.000134` on Fold A and
`-0.000796 / -0.000843 / -0.001701` on Fold B. Final EMA-0.995 was
positive on both folds (`+0.000182 / +0.000210`) but informational. The
primary paths failed, driven by the negative Fold-B result; the family was
rejected. Interpret Fold-A fit with the documented March-2022 coverage seam.

Sidecar manifest SHA-256:
`d543a9bc613d04b27c7765bce57ac6c953b540d9769ba6d22d0abd1afded2a5a`.

### Experiment 29 -- SHFE ferrous and pulp

Standalone deltas were `+0.000265` on Fold A, block-10
`[-0.000828, +0.002025]`, and `-0.000693` on Fold B, block-10
`[-0.002030, +0.000502]`. Parent-plus-candidate deltas were
`+0.000273` and `-0.000236`. Final EMA-0.995 produced a Fold-A gain of
`+0.002065` but reversed to `-0.000866` on Fold B. The candidate was
rejected on both primary paths. This result covers only the six mapped
steel/pulp names and does not test iron ore.

Sidecar manifest SHA-256:
`97afbd6d25b88de91316960064308de1f3d82bd95bf5cf5b94017c3b2d106b65`.

### Experiment 30 -- COTAHIST options activity

This was the only candidate with positive primary deltas on both folds for both
roles: standalone `+0.000020 / +0.000494` and parent-plus-candidate
`+0.000149 / +0.000372`. Their means, `+0.000257` and `+0.000261`, were
well below the predeclared `+0.001` threshold. Block-10 standalone intervals
were `[-0.000810, +0.000969]` and `[-0.000765, +0.001488]`.
The standalone effect increased with horizon on Fold B
(`+0.000214 / +0.000429 / +0.000839`) but remained informational. The
family was rejected. This does not reject option OI, IV, or covered/uncovered
position data, which were not retrievable and were not tested.

Sidecar manifest SHA-256:
`55e99e9529f6776f6b610f0ce2a482ea94be1cebf6b06dad916ea2117597a6eb`.

### Experiment 31 -- CVM RAD event state

Standalone deltas flipped from `-0.002418` on Fold A, block-10
`[-0.004804, -0.000288]`, to `+0.001158` on Fold B, block-10
`[+0.000243, +0.002038]`. The parent-plus-candidate recipe reduced but did
not remove the reversal: `-0.000977 / +0.000738`. The standalone loss grew
with horizon on Fold A (`-0.000502 / -0.002207 / -0.004543`) while all
Fold-B horizons were positive. With only two folds, no regime explanation is
assigned. Both gates failed and the family was rejected.

Sidecar manifest SHA-256:
`3ff942984ce6e1b6c2a9e5bff3e9ae95e92422e3f5c3b8954afc13b3fd8a71c3`.

### Experiment 32 -- B3 odd-lot activity

Standalone deltas were `-0.002593` on Fold A, block-10
`[-0.004542, -0.000630]`, and `+0.000407` on Fold B, block-10
`[-0.000245, +0.001240]`. Parent-plus-candidate deltas were
`-0.001153 / +0.000250`; the Fold-A standalone loss also violated the
diversity path's `-0.001` loss guardrail. Both primary paths failed and the
family was rejected. Buyer/seller imbalance was absent from COTAHIST and remains
untested.

Sidecar manifest SHA-256:
`9a214b4c0b17ab0a42ec4fd191bc41654a3987c9912a32e056dda384bd95f444`.

### Experiment 33 -- B3 index rebalance state

The primary results were near-null and non-positive: standalone
`-0.000595 / -0.000131` and parent-plus-candidate
`-0.000152 / -0.000009`. Final EMA-0.995 had a large opposing-fold pattern
(`-0.004406 / +0.002508`) and could not affect retention. Both paths failed
and the family was rejected. The result applies only to the sparse 13-release,
291-state-date archive beginning in May 2023, not to a complete 2021--2024 B3
or MSCI history.

Sidecar manifest SHA-256:
`9bc917a89e2d0d6473fcb576e81346c17cb77f7f525b7b613ae6424bec47cf85`.

### Experiment 34 -- CCEE PLD power state

Standalone deltas were `-0.001113` on Fold A and `+0.000735` on Fold B;
parent-plus-candidate deltas were `-0.000185 / +0.000511`. All three
standalone horizons were negative on Fold A and positive on Fold B. The
two-fold means were `-0.000189` standalone and `+0.000163` diversity, so
both paths failed and the family was rejected. ONS load/EAR vintages were not
available under a causal historical contract and remain untested.

Sidecar manifest SHA-256:
`7c4908b2aad10cadf820c9073262fb457a3c4376e4527efb6da60adcb5de17ca`.

### Experiment 35 -- CVM structured fundamentals

Standalone deltas were `-0.000695` on Fold A and `+0.001305` on Fold B;
parent-plus-candidate deltas were `-0.000101 / +0.000954`. Fold-B gains
rose with horizon, but the Fold-A sign was not preserved. Means were only
`+0.000305` and `+0.000426`, and both paths failed. The family was rejected
for this exact feature set; B/M and E/P were deliberately not constructed and
remain untested.

Sidecar manifest SHA-256:
`7e7d9326fbd73789a608d14fdd72eca2bdaec29a6234ca53e6f6d9e25661498e`.

### Experiment 36 -- Regular-session trade activity

Standalone deltas were `+0.000080` on Fold A and `-0.000691` on Fold B;
parent-plus-candidate deltas were `+0.000049 / -0.000260`. Final EMA-0.995
was modestly positive on both folds (`+0.000291 / +0.000315`) but was a fixed
informational readout. Both primary paths failed and the family was rejected.
Historical after-hours payloads were unavailable and were not tested.

Sidecar manifest SHA-256:
`fb45132b1fea1a95d7ce5605b319188db0355d85c8e6bb3a5367c72e7cd26bde`.

### Experiment 37 -- ADR overnight

ADR overnight was the strongest primary standalone on Fold A (`+0.001044`,
block-10 `[-0.000391, +0.002310]`) but reversed on Fold B (`-0.000757`,
block-10 `[-0.002171, +0.000906]`). Parent-plus-candidate deltas were
`+0.000725 / -0.000111`. Final EMA-0.995 was positive on both folds
(`+0.000779 / +0.000622`, mean `+0.000701`) and the EMA stack was also
positive on both (`+0.000529 / +0.000566`), but neither was retention-eligible.
Both primary paths failed and the family was rejected. FX/parity and intraday or
after-hours ADR information were not tested.

Sidecar manifest SHA-256:
`f2b195a5fef384243daff3635e1fff0d892e21ae4a5b4f542f5dc43eb08adb1f`.

### Program decision

The ten screens produced no canonical-primary stage winner. The nearest robust
primary was options activity, positive on both folds but only about one quarter
of the required mean gain. Several sources had an A/B sign reversal, so isolated
fold or horizon wins are not promoted. Final EMA-0.995 showed positive two-fold
deltas for lending, options, regular activity, and ADRs; this recurring
informational pattern may motivate a separately preregistered future question,
but it does not revise any Experiment-28--37 decision and does not reopen the
frozen checkpoint rule inside this program. Raw Patience-3 remains the canonical
recipe, the designated challenger remains informational, and the next official
read is unspent.

### Artifact cleanup and paid-instance termination

Cleanup began only after the strict 10/10 extraction was frozen, GitHub `main`
contained the results, the program manifest was complete, and no training or
analysis process remained. The accepted-program checkpoint plan was bound to
plan ID `dabdea079ff844b699b6809ee4f49993` and inventory SHA-256
`8d9c50d88f9457718f6cbd0a4aee7ca02d3b86725b552cfc7e672a1721e8f307`.
It removed exactly 1,022 unselected intermediate `epoch_XX.pt` files totaling
4,630,878,034 bytes. The postcheck verified all delete targets absent and all
2,298 retained files against their original SHA-256 hashes. Retained payloads
include 178 final/Patience-selected checkpoints, all 1,260 per-epoch/tail
prediction archives, all 60 validation references, and every history, manifest,
analysis, sidecar, and strict result summary.

The rejected bias-confounded preflight used separate plan ID
`55ef7ff7745543898c27ba4a12bd51e6` and inventory SHA-256
`33d174de79cb6211796ef9c84fd962c293189a84bf91199faf4392dc6d3105b9`.
It removed exactly 24 partial checkpoints, 24 partial prediction archives, and
two partial validation references totaling 1,177,912,896 bytes. Its abort record,
program/campaign/run manifests, logs, scheduler records, and histories remain as
12 hash-verified audit files totaling 74,918 bytes. These removals are permanent;
the rejected partial binaries were never valid experiment results.

Total cleanup was 1,072 files / 5,808,790,930 bytes. Persistent NFS reported
76 GiB used after cleanup. Paid GH200 instance
`c40d3ea383f84ea89780612a7aaaeeec` in `us-east-3` was terminated only after
these postchecks. Lambda accepted the exact-ID request as `terminating`; two
subsequent provider inventory reads returned zero matches for that ID.

## Experiment 38 -- Kronos K0 zero-shot kill-test preregistration

Status at registration: no Kronos inference score, momentum-control score,
parent same-scope score, or ensemble result exists. This section is frozen before
any score is computed. The source specification was supplied on 2026-08-22 as
`k0_killtest_spec.md`, SHA-256
`396cbc970e9f63ec57f286176122d95902f1ffa17214e24359fe4584dee3b6f5`.

K0 is inference only. There is no training, fine-tuning, gradient update,
context/DI/slow-feature injection, official-validation access, held-out-test
access, or K1 run. The user explicitly requires that the K0 decision rules be
registered but that K1 not be launched from this run.

### Frozen implementation and scope

- Upstream repository: `https://github.com/shiyu-coder/Kronos`, commit
  `67b630e67f6a18c9e9be918d9b4337c960db1e9a`. Only `model/` may be imported.
  The fine-tuning, CSV fine-tuning, examples, and WebUI trees are prohibited.
- Checkpoints are pinned before download: Kronos-small revision
  `901c26c1332695a2a8f243eb2f37243a37bea320`, Kronos-base revision
  `2b554741eca47781b64468546e77fef3e85130e6`, and shared tokenizer revision
  `0e0117387f39004a9016484a186a908917e22426`. Every downloaded file is hashed.
- The upstream predictor and model internals remain unmodified. Models and
  tokenizer run in evaluation mode. The upstream fine-token RoPE/cross-attention
  behavior is accepted as shipped.
- Only Fold A's 102-date selection window and Fold B's disjoint 102-date
  selection window are accessed. The fixed decision grid is
  `{0, 10, 20, 30, 40, 50}` and the horizons are 30/60/120 minutes. Kronos-small
  runs this full scope. Kronos-base runs the full scope if its 200-context
  projected time is no more than 24 GPU-hours; otherwise its scope is fixed to
  `{0, 20, 40}`. This throughput rule is result-independent.
- Each context contains the last 512 five-minute bars through the bar closing at
  the decision minute, with no bar beginning at the decision included. Five-minute
  aggregation, synthetic bars, point-in-time membership, exact security identity,
  and the `80%` full-context / `95%` last-24 coverage thresholds follow the supplied
  K0 specification. Groups with fewer than 30 eligible+labeled equities are skipped.
- The raw XP schema has no financial-volume or minute-VWAP field. The sidecar
  therefore stores OHLC plus summed `real_volume` and omits `amount`; the shipped
  predictor deterministically synthesizes amount as volume times mean OHLC. This
  rule is fixed before inference.
- Inference settings are `T=0.6`, `top_p=0.9`, `top_k=0`, `sample_count=5`,
  `pred_len=24`, and `max_context=512`. The horizon score is mean predicted close
  at index `H/5 - 1`, divided by the last context close, minus one.
- Momentum control is fixed as `close[T] / close[T-60 minutes] - 1`, using context
  close indices `-1` and `-13` on the same synthetic-aware K0 bars.
- The first 100 eligible contexts in chronological date/decision/security order
  form the bf16-versus-fp32 audit. Identical per-context seeds are used. bf16 is
  adopted for a model only if every evaluable group's absolute Spearman-IC change
  is below `0.001`; unsupported or non-finite bf16 falls back to fp32.
- The first 200 eligible contexts in the same order form the throughput audit.
  Throughput cannot change the small scope and can change base only through the
  predeclared 24-hour rule above.
- Stable per-context seed is the first unsigned 63 bits of SHA-256 over
  `(model_name, trade_date, decision_idx, security_id)`. The shipped
  `predict_batch` has one process-global RNG and no per-row generator. To preserve
  the exact per-context seed without patching upstream internals, K0 calls
  `predict_batch` with one equity context and five internally batched samples.
  This deliberately forgoes cross-equity batching. A context rerun resets its
  seed and must reproduce its score array bit-for-bit.
- Primary metrics use the existing `sample_level_spearman_ic` and
  `primary_validation_score` conventions. The three-seed parent is reconstructed
  from the immutable trajectory artifacts with the frozen bidirectional
  odd/even Patience-3 epochs and re-ranked on the same K0 mask. Parent correlation,
  same-scope parent IC, horizon/TOD tables, and uniform parent-plus-Kronos rank
  ensemble block-5/10 intervals are informational only.

### Preregistered decision rule

Let `IC_best` = the better of the two models' mean-of-folds primary IC.

- **Kill** (family rejected, program over) if `IC_best < 0.015`, OR if
  `IC_best ≤ momentum-control IC` on the mean of folds. Rationale recorded in
  advance: fold contamination biases upward, so failing on favorable ground is
  decisive.
- **Proceed to K1** (fine-tune-as-encoder program; separate preregistration) if
  `IC_best ≥ 0.015` AND mean score-parent correlation `< 0.5`. Record explicitly
  that K0 passing proves nothing (optimistic bias); K1's confirmatory evidence
  must come from post-2024-06 data (the bundled official read) or forward
  walk-forward.
- **Park** (no K1, revisit only with new evidence) if `IC_best ≥ 0.015` but
  correlation `≥ 0.5` -- a redundant signal at parent-quality-minus is not worth
  the compute.
- The informational ensemble delta cannot rescue a kill and cannot trigger K1 by
  itself.

For this execution, "Proceed to K1" means only that the registered K0 rule labels
the family eligible for a separately preregistered future experiment. It does not
authorize K1 work or another model/data read on the paid instance.

### Leakage register

Kronos weights are a non-point-in-time artifact relative to the fold dates:
pretraining extends through 2024-06 and includes B3 **daily/weekly** bars -- i.e.,
the daily-scale outcomes of these exact tickers during both fold windows are
inside the pretrained weights. B3 intraday bars are not in the corpus, so
memorization of intraday paths is not possible, but daily-scale foreknowledge and
global cross-market intraday patterns keep the bias direction positive. Hence the
asymmetric decision rule. Any future confirmatory claim requires post-2024-06
evaluation windows.

The immutable result entry must record the final artifact path and
`official_validation_accessed=false`, `test_accessed=false`. Score arrays and
manifests survive cleanup; K1 is explicitly outside this run.

### Pre-score runtime correction

The first GH200 model call stopped before returning a prediction or producing any
Kronos, momentum, parent, or ensemble score. Although upstream `predict_batch`
documents timestamp inputs as `DatetimeIndex or Series`, its shipped
`calc_time_stamps` implementation calls the pandas `.dt` accessor, which a
`DatetimeIndex` does not expose. The wrapper now passes identical naive-local
timestamps as pandas `Series`. No upstream file, model setting, context, mask,
scope, seed, metric, or decision rule changed. The incomplete run is resumed only
after this correction is committed and its tests pass.

The resumed precision preflight was then stopped before any score array, metric,
or result was persisted because Pandas/NumPy emitted one deprecation warning per
context when a `Timedelta` was added to an otherwise correct timestamp. No
preflight prediction value was inspected. Future timestamps are now constructed
as the exact same `last_close_ns + 5-minute * [1..24]` integer nanoseconds before
conversion to a Series. This is a runtime/logging correction only; it changes no
timestamp or experimental decision.

The first serial pass then established actual single-context throughput while
persisting only independently seeded, unanalyzed score rows. No score value or
metric was inspected. GPU utilization was approximately 45%, so the process was
stopped to introduce execution-only sharding; `1,939` completed context rows were
retained from its memory-mapped partial artifact.
Eight isolated processes now receive disjoint context ordinals. Every worker still
calls the unchanged upstream `predict_batch` with exactly one equity context and
five samples, resets the same stable per-context seed, and writes a private array;
no cross-equity RNG stream is introduced. The coordinator rejects overlaps and
out-of-scope writes, merges only completed worker masks, and bitwise-reruns the
first/middle/last merged contexts through the single-process predictor. Existing
serial scores are retained and included in that audit. Worker count and CUDA MPS
state are operational metadata only and cannot change scope, precision, sampling,
metrics, or the registered decision.

### User-directed model-scope override before analysis

At `2026-08-22T16:56:20Z`, after Kronos-small inference and its bitwise merge
audit had completed but before any Kronos-small score value or metric was
inspected, the user narrowed the research question to whether Kronos-small works.
The user explicitly waived Kronos-base because the paper reports similar results
for the two sizes. This result-independent instruction supersedes the original
two-model scope: K0 will analyze Kronos-small only, and `IC_best` in the frozen
kill/park/eligibility rule is therefore the Kronos-small mean-of-folds IC.

Kronos-base had passed its fp32 and throughput preflights and begun the full
six-decision pass. It was stopped with `6,751` disjoint worker contexts complete,
zero contexts merged into the coordinator array, no base metric computed, and no
base or small score value inspected. Those incomplete base arrays are excluded
from analysis and deleted. All other fixed settings, masks, comparators, leakage
register, decision thresholds, and the prohibition on K1 remain unchanged.

### K0 result -- Kronos-small rejected

The completed immutable run is:

    /lambda/nfs/brazil-rv-east3/quant-data/b3/processed/model_runs/kronos_k0_3f93b26_20260822T134800Z

Its immutable bar sidecar is:

    /lambda/nfs/brazil-rv-east3/quant-data/b3/processed/model_runs/kronos_k0_bars_3f93b26_20260822T134400Z

Kronos-small used the full six-decision scope and fp32; bf16 was unsupported by
the shipped path on this stack. The 200-context serial projection was `9.3201`
GPU-hours. Eight exact-seed MPS workers completed the fixed `158,205` contexts in
`9,039.18` seconds after resuming `1,939` serial rows. The coordinator rejected
overlap and out-of-scope writes and bitwise reproduced the first, middle, and
last merged contexts. Fold A supplied `79,484` eligible contexts and Fold B
`78,721`; mean synthetic fractions were `0.004132` and `0.007562`.

| Metric | Fold A | Fold B | Mean of folds |
|---|---:|---:|---:|
| Kronos-small primary IC | 0.008843 | 0.018551 | 0.013697 |
| Matched 60-minute momentum IC | -0.013233 | -0.018841 | -0.016037 |
| Same-scope parent-3 IC | 0.045239 | 0.047483 | 0.046361 |
| Kronos-small / parent rank correlation | 0.128711 | 0.138826 | 0.133768 |
| Parent-3 + Kronos-small IC | 0.044389 | 0.048590 | 0.046489 |
| Stack delta versus parent | -0.000849 | +0.001106 | +0.000128 |

The informational stack delta was not stable: Fold A block-5 and block-10 95%
intervals were `[-0.002691, +0.001375]` and
`[-0.002531, +0.001097]`; Fold B intervals were
`[-0.000521, +0.003391]` and `[-0.000288, +0.003406]`. All include zero.
Kronos-small IC was positive at every horizon in both folds, but the entire Fold
A profile was weak and the mean remained below the registered floor.

The decision is **kill**. Kronos-small's mean-fold IC `0.013697` is below the
predeclared `0.015` threshold. Its favorable comparison with the negative
momentum control and its low parent correlation cannot override that independent
condition; the informational ensemble cannot rescue a kill. Because the fold
windows are optimistically contaminated by Kronos pretraining, failure on this
favorable ground rejects the zero-shot Kronos-small family for the current
program. K1 was not run and is not authorized by this result.

The final score array SHA-256 is
`d06613122d7fc0b4b05b86dd052ea3847775e6951f5018f3dea3fcafc0f6b739`;
the completion-mask SHA-256 is
`0111fa78692e86c4799e8e6a13887a9583ef0dea2b94103735d21ad65c8a4e07`.
The final artifact audit matched the fixed coverage mask, confirmed finite
in-scope values and exact zero out of scope, and found the expected 2/6/12 rows
in the fold summary, horizon table, and time-of-day table. The manifest records
`official_validation_accessed=false`, `test_accessed=false`, and
`k1_started=false`.

Deletion-first cleanup then removed the unmerged Kronos-base directory, all
private completed-worker score arrays, the pinned upstream clone, the three
Kronos HuggingFace cache directories (`502 MiB` total), the CUDA MPS runtime,
and the local workspace clone. The completed run (`9.0 MiB`) and bar sidecar
(`199 MiB`) are retained read-only; operational logs remain outside the run.
The paid Lambda instance `c0aef7522bf64fe0899e8703027668db`
(`gpu_1x_gh200`, `us-east-3`, IP `192.222.50.94`) accepted termination and was
confirmed absent from the provider's active-instance inventory at
`2026-08-22T17:05:28Z`.

## Experiment 39 -- P0/P1 feature-program preregistration

Status at registration: no P0 mixed-stack score, channel-attribution score, P1
feature IC, F3 trajectory, or F4 result has been computed. The research input is
`research_memo_v2.md` dated 2026-08-22. Per the user's explicit scope, **P0.2
(the Kronos K0 closure diagnostic) is omitted completely**. It will not be run,
partially computed, or used to select a P1 feature.

### P0.1 mixed-state stack

Two and only two uniform tie-aware rank ensembles are registered on Fold A and
Fold B:

1. parent-3 cross-fitted Raw Patience plus final-EMA-0.995 members from residual
   auxiliary, combined auxiliary, options, lending, regular activity, ADR, and
   market-gate families (24 members total); and
2. parent-3 plus residual, options, and ADR EMA members (12 total).

Every family contributes exactly seeds 11/29/47 and receives no fitted weight.
Both variants report paired block-5/10 deltas versus the canonical parent and the
designated challenger. Retention remains keyed only to the canonical parent:
mean Fold-A/B delta at least `+0.001` and each fold non-negative. If both pass,
the higher mean delta (lexical tie-break) is the discovery finalist. The family
criterion was itself derived from these folds, so a pass is explicitly discovery
evidence rather than confirmation and cannot spend official validation alone.

The 2026-08-21 retention cleanup removed the combined-auxiliary and market-gate
binary predictions, and all parent raw checkpoints needed by P0.3. Missing
trajectories will be reproduced only at their recorded commits and frozen
hyperparameters. Parent reproduction must match the retained cross-fit
predictions/recorded stop epochs before attribution. Combined and market-gate
reproductions must match their immutable recorded fold/readout summaries before
entering P0.1. This operational rerun is not a new model search.

### P0.3 channel attribution (F1)

For each of the 26 incumbent dynamic channels and 32 slow fields, the field is
zeroed only on the 158 equity inputs; local/global context inputs and sequence
history masks are unchanged. Each seed uses the checkpoint selected on one
odd/even selection-date parity and reports the ablation only on the opposite
parity, in both directions. Seeds are uniformly rank-averaged before the IC drop
is calculated. Report overall and 30/60/120-minute parent-minus-zeroed IC drops
on both folds with moving-block intervals.

A field is `dead` only when its overall drop is non-positive on both folds and at
least two of three horizon drops are non-positive on each fold. It is `keep` when
mean drop is at least `+0.00025` and neither fold is negative; all other fields are
`suspect`. Only `dead` fields may be removed in F3. This conservative rule favors
false retention over deleting weak alpha.

### P1 F2 feature screen

The causal library contains 19 candidates: same-30-minute returns lagged 1/5/20
sessions; 15-minute VWAP reversal and its volume sign-flip interaction; 1/5-day
signed semivariance and realized skewness; high-attention open-gap fade; late-day
market momentum times stored pre-neutralization WIN beta; interval, cumulative,
and first-30-minute relative volume; EDGE spread, intraday Amihud, their reversal
interactions; and trailing-20-session overnight-minus-intraday return.

Three memo items are deliberately not mislabeled as tests. Current sector labels
are not substituted for an immutable point-in-time sector history, so
sector-demeaned reversal is unavailable. The session ends its decision grid at
14:45, so an after-15:00 signal is structurally unavailable. The incumbent
already has causal same-minute 20-session robust volume normalization, so P1
tests only incremental interval/opening variants rather than duplicating it.

All transformations use exact accepted permanent-security dates. History ends
strictly before the decision, no stale price is an endpoint, prior-session
normalizers consume only prior observations, and current cross-sectional robust
scales consume only the contemporaneously available cross-section. EDGE follows
the estimator authors' published pseudocode.

F2 is frozen to the first 407 training dates (through 2023-03-31), leaving every
F3 selection date unseen. For each feature, compute IC on two chronological F2
halves and its maximum absolute within-sample correlation with the 58 incumbent
equity fields. Eligibility requires the same IC sign in both halves and at least
`0.001` absolute IC in each. Incremental score is
`min(abs(half ICs)) * (1 - max_existing_corr^2)`. Greedily take at most eight,
at most two per family, rejecting a candidate correlated at least `0.85` with an
already selected feature. F3 runs only if at least six survive.

### P1 F3/F4

F3 trains one bias-free, zero-start sidecar candidate containing the frozen F2
shortlist while zeroing only F1-dead incumbent fields. It uses seeds 11/29/47,
the frozen 20-epoch SAM trajectory, and cross-fitted Raw Patience primary. The
new Fold C fits 407 dates through 2023-03-31 and selects the 105 dates from
2023-04-03 through 2023-08-31. Its 512-date effective batches necessarily draw
fit dates with replacement; Fold A/B retain their existing contracts. Both the
standalone candidate and the uniform parent-3+candidate-3 diversity path are
reported. Each may pass only with three-fold mean delta at least `+0.001` and
every fold non-negative; the diversity path also requires standalone loss no
worse than `-0.001` on every fold. Final EMA and P0 mixed-state additions are
secondary/informational and cannot retain the candidate.

F4 runs only if a primary F3 path passes. Each selected sidecar feature is then
zeroed (value and observedness mask) inference-only on the frozen cross-fit
states. A feature survives when the full-minus-ablated mean is positive and at
least two folds are non-negative. Exactly one reduced sidecar is retrained across
the same three folds/seeds. It is promotion-eligible only if it independently
passes the F3 gate, loses no more than `0.0005` mean IC versus the full F3 recipe,
and loses no more than `0.001` on any fold. No official validation or held-out
test access is authorized by this registration.

### Source reproduction and operational repair

The missing parent, combined-auxiliary, and market-gate trajectories were
reproduced at their frozen settings before P0 was scored. The original source
stage exited cleanly when `combined_fold_b_11` suffered a transient child-process
failure. No score from the incomplete Fold-B parent was used. A fresh Fold-B
repair reran all nine family/seed trajectories without reusing partial artifacts;
Fold A was taken only from the completed original runs and Fold B only from the
completed repair.

The accepted assembly reproduced every parent cross-fit selected/stopped epoch
exactly. Parent prediction-rank similarities were `0.999930-0.999952`, maximum
absolute member IC drift was `0.0000264`, and ensemble IC drift was
`-0.0000068` on Fold A and `+0.0000029` on Fold B. Reproduced combined and
market-gate final-EMA metrics differed from their recorded values by at most
`0.0000274`. These are within the preregistered cross-instance tolerance and the
assembly was accepted before P0.1/P0.3. P0.2 remained absent. The exact report is:

    /lambda/nfs/brazil-rv-east3/quant-data/b3/processed/model_runs/p0_p1_27aa0d0_20260822T194900Z/validated_sources/validation_report.json

### P0 results

P0.1 rejected both fixed uniform stacks against the canonical parent:

| Variant | Members | Fold A | Fold B | Mean | Gate |
|---|---:|---:|---:|---:|---|
| all listed families | 24 | `-0.001242` | `+0.001709` | `+0.000233` | fail |
| residual + options + ADR | 12 | `-0.000272` | `+0.002278` | `+0.001003` | fail |

Against the informational designated challenger, the same A/B deltas were
`-0.001929/-0.000624` and `-0.000960/-0.000055`, respectively. The smaller stack
met the mean threshold only by reversing sign across folds, so it was not
retained. No weights were learned. The initial summary write encountered only a
NumPy-boolean JSON serialization error after all immutable analyses had finished;
commit `8f46124` fixed serialization, and the summary was reconstructed from those
unchanged analyses without recomputation or selection changes.

P0.3 classified 14 of 58 incumbent equity fields `keep`, 32 `suspect`, and 12
`dead` under the conservative two-fold/horizon rule. The dead fields were:

- dynamic 11 `realized_vol_30m_log_ratio`;
- dynamic 20 `market_dispersion_15m`;
- dynamic 25 `cross_section_volatility_rank_30m`;
- dynamic 14 `session_range_position`;
- dynamic 12 `realized_vol_60m_log_ratio`;
- slow 15/16 `observed_fraction_5d`/`observed_fraction_20d`;
- slow 27 `weekday_cos` and slow 28 `month_end_proximity`;
- slow 10 `realized_vol_20d_log_ratio`;
- slow 13 `median_daily_dollar_volume_20d_log_scale`; and
- slow 18 `dollar_volume_cross_section_rank`.

Only those fields were zeroed in the P1 candidate. They were not deleted from the
canonical parent, because the downstream joint candidate did not pass.

### P1 results

F2 selected eight of 19 causal candidates on the first 407 dates:

| Feature | Half-1 IC | Half-2 IC | Max existing correlation | Incremental score |
|---|---:|---:|---:|---:|
| `vwap_reversal_15m_cs` | `+0.014392` | `+0.012306` | `0.6861` | `0.006514` |
| `late_market_momentum_beta` | `-0.013616` | `-0.005298` | `0.0594` | `0.005280` |
| `overnight_minus_intraday_20d_cs` | `-0.016134` | `-0.005889` | `0.5887` | `0.003848` |
| `signed_semivariance_1d` | `-0.005317` | `-0.011427` | `0.6551` | `0.003035` |
| `edge_spread_60m_cs` | `-0.003758` | `-0.007268` | `0.7230` | `0.001793` |
| `vwap_reversal_volume_flip` | `+0.001791` | `+0.006833` | `0.3185` | `0.001610` |
| `amihud_30m_cs` | `-0.004782` | `-0.007855` | `0.8150` | `0.001606` |
| `hks_same_interval_return_lag5` | `+0.001569` | `+0.002473` | `0.0959` | `0.001555` |

F3 then trained the predeclared eight-feature, twelve-field-pruned candidate on
Fold C/A/B. Cross-fitted Raw Patience produced:

| Primary path | Fold C | Fold A | Fold B | Three-fold mean | Gate |
|---|---:|---:|---:|---:|---|
| standalone candidate | `-0.000568` | `+0.000576` | `+0.001054` | `+0.000354` | fail |
| parent-3 + candidate-3 | `-0.000015` | `+0.000711` | `+0.000920` | `+0.000539` | fail |

The Fold-C block-5/block-10 95% intervals were
`[-0.003293,+0.001523]`/`[-0.003493,+0.000994]` standalone and
`[-0.001387,+0.001045]`/`[-0.001486,+0.000779]` for the diversity stack. Fold A
intervals were `[-0.001465,+0.003064]`/`[-0.001705,+0.003128]` and
`[-0.000344,+0.001972]`/`[-0.000525,+0.002044]`; Fold B intervals were
`[-0.000595,+0.002739]`/`[-0.000178,+0.002425]` and
`[+0.000096,+0.001722]`/`[+0.000298,+0.001580]`. Thus the only locally positive
interval was Fold B's ensemble, while the unseen Fold C prevented retention and
the three-fold means remained below `+0.001`.

Final-EMA-0.995 was also negative overall: standalone C/A/B deltas were
`-0.000026/-0.001814/-0.002199` (mean `-0.001346`), and the parent-plus-candidate
deltas were `+0.000350/-0.000098/-0.000520` (mean `-0.000089`). Adding the P1
members to the P0 mixed-state informational comparator reduced IC by `-0.000601`
on Fold A and `-0.000515` on Fold B. Neither secondary readout altered the
decision.

Because neither primary path passed, F4 was invoked exactly once and wrote
`status=not_run`, reason `F3 primary gate did not pass`; no ablation or reduced
retraining was launched. P0.2 was never run. Official validation and the held-out
test remained sealed throughout.

### Final decision and artifacts

Reject P0.1 and the joint P1 feature/pruning recipe. Preserve P0.3 and F2 as
diagnostic evidence, not as permission to modify the canonical recipe. Raw
Patience-3 on the unchanged parent architecture remains canonical, and the
designated challenger policy is unchanged. The immutable program root is:

    /lambda/nfs/brazil-rv-east3/quant-data/b3/processed/model_runs/p0_p1_27aa0d0_20260822T194900Z

Key summary SHA-256 values are `04d67ff1...293dca0` (P0.1),
`f920e404...041d280` (P0.3), `b05c8cc5...3294ed` (F2),
`a0ab1490...8886ee0` (F3), and `3130d4a1...a384b21` (F4). Exact implementation
history is commits `5b6b5d4`, `1b63661`, `27aa0d0`, and `8f46124`; rejected
experiment-only plumbing is removed from current HEAD rather than retained as
compatibility code.

After the summaries and repository commit were secured, an exact hash inventory
identified only redundant raw checkpoints and per-epoch validation predictions.
The reviewed plan deleted 1,183 individual `.pt`/`.npz` files totaling
28,569,479,747 bytes (26.607 GiB). It retained 154 binary artifacts totaling
3,664,530,222 bytes: every observation reference, epoch 20 raw/EMA container,
and each checkpoint/prediction epoch selected by an honest opposite-parity
Patience replay. Sidecars, manifests, histories, diagnostics, analyses, summary
JSON, source archives, and canonical data were outside the delete set. Hash and
post-deletion checks passed; the program root is 4.8 GiB after cleanup. The
immutable audit is:

    /lambda/nfs/brazil-rv-east3/quant-data/b3/processed/model_runs/p0_p1_27aa0d0_20260822T194900Z/_cleanup/20260823T024900Z

The cleanup-plan SHA-256 is
`9d820e71259c92607ed7c28e969ed8f7e54d0175d40df4c1fc77843af0caacae`;
the result SHA-256 is
`da2b91140683cbd179e60ea0ffc94345598b9c3a4c8dcaa59025ad17398d5e1d`.

Paid Lambda instance `b3eac682796a4e1ea7912422a81f0e85`
(`gpu_1x_gh200`, `us-east-3`, IP `192.222.51.153`) accepted termination after
the results commit was pushed. It was confirmed absent from the provider's
active-instance inventory at `2026-08-23T02:59:07Z` and again at
`2026-08-23T02:59:27Z`.

## Experiment 40 — final features-only closure and P2 strong-source screens (preregistered 2026-08-23)

Status at registration: no Experiment-40 trajectory, fold score, gate, official
validation score, or held-out-test score has been computed. The user fixed the
scope to the final P1 feature-confound screen followed by P2 from
`research_memo_v2.md`; no later memo stage or bundled official read is authorized
in this experiment.

### Final P1 feature screen

The primary closure candidate is exactly the frozen eight-feature F2 sidecar from
Experiment 39, trained with every incumbent parent channel present. There is no
P0.3 pruning, no feature reselection, and no change to the sidecar adapter. A
second fixed confirmatory candidate contains only
`late_market_momentum_beta` and `hks_same_interval_return_lag5`, the two selected
features whose maximum incumbent correlations were approximately `0.059` and
`0.096`. It is declared before either closure result and cannot be chosen after
observing the eight-feature score. These are the feature program's final tests;
failure closes the program rather than authorizing further feature combinations.

Both candidates use Fold C/A/B and seeds 11/29/47, one 20-epoch SAM trajectory
per fold/seed, the bias-free zero-start residual sidecar, and bidirectional
odd/even cross-fitted Raw Patience-3 primary. Final EMA-0.995 remains a free
secondary read. Each candidate reports standalone and uniform parent-3 plus
candidate-3 rank ensembles. A path passes only when its three-fold mean delta
versus the canonical parent is at least `+0.001` and every fold is non-negative;
the diversity path additionally requires standalone delta no worse than
`-0.001` on every fold. The designated challenger is informational on Fold A/B
only and cannot retain a candidate.

### P2 source contracts

Three candidates are screened independently under the identical three-fold
training/readout/gate contract. No combined P2 bundle is selected from these
results.

1. **B3 registered lending rates and flows.** Parse exact ISIN rows from the
   official chapter-05 BDI registered-loans table. The features are taker-fee
   level, exact five-B3-session fee change, registered share flow divided by
   trailing-20 observed cash-share volume, and its exact five-session change.
   Report D first becomes available on the next observed B3 session. A missing
   accepted ISIN in a complete table is an observed zero registered flow but has
   no fee. The legacy free BDI archive is retained and audited, but it contains
   balances rather than the rate table; the exact free rate history begins
   `2023-07-10` and has a documented source gap from `2023-07-21` through
   `2023-08-24`. Changes mask across that gap. Zero-byte historical bodies from
   the nominal legacy CSV endpoint are not treated as data.
2. **B3 listed-equity option open interest.** Use the final timestamped
   BVBG.086 Price Report for D and the same-date final BVBG.028 instrument master.
   Map option instrument ID to the master's explicit underlying instrument ID,
   then to the same-date cash instrument's exact ISIN and accepted permanent
   `security_id`; never parse option ticker prefixes. Features cover OI/cash ADV,
   exact one-session OI change, put/call OI, near-expiry OI share, OI-weighted
   absolute log moneyness, and prior-only OI surprise. Final report D is first
   used next B3 session. An active master series absent from the complete final
   report contributes zero OI. The free Price Report does not expose the separate
   covered/uncovered split, and historical `DerivativesOpenPositionFile` tokens
   returned zero-byte bodies on the unrestricted machine, so covered/uncovered
   and PIN are explicitly not tested or fabricated.
3. **DCE iron ore.** Use contract-specific daily settlement and OI histories,
   selecting the return contract only from prior-session OI and never splicing
   contracts. Features are 1/5-day same-contract settlement returns, one-day OI
   change, curve slope, and fixed producer/steel exposure masks for six exact
   accepted B3 ISINs. Continuous `I0` is prohibited. The old official DCE public
   route returned HTTP 412/timeouts and the current official API requires
   credentials unavailable in this session, so the frozen free source is the
   contract-specific Sina mirror, disclosed as unofficial. DCE session D closes
   around 04:00 Sao Paulo and is used at the first B3 session on or after D; all
   robust normalization uses prior observations only.

All normalized source frames use explicit value masks, exact-zero invalid values,
permanent identities, immutable manifests/hashes, and exact assignment to the
canonical model axes. No official validation or held-out test access is allowed
during Experiment 40.

### Completed result (2026-08-23)

The program completed all 45 preregistered trajectories: five candidates, three
chronological folds, seeds 11/29/47, and 20 epochs per fold/seed. The primary
readout remained bidirectional odd/even cross-fitted Raw Patience-3. The table
reports candidate-minus-canonical IC in Fold C/A/B order; `stack` is the fixed
uniform parent-3 plus candidate-3 rank ensemble.

| candidate | standalone C/A/B | standalone mean | stack C/A/B | stack mean | retained |
| --- | --- | ---: | --- | ---: | --- |
| full eight features, no pruning | `-0.000028 / +0.000875 / -0.000312` | `+0.000179` | `+0.000174 / +0.000613 / +0.000005` | `+0.000264` | no |
| fixed late-market/HKS pair | `-0.000016 / +0.000027 / -0.001433` | `-0.000474` | `-0.000010 / +0.000031 / -0.000660` | `-0.000213` | no |
| B3 lending rates/flows | `+0.000001 / +0.000125 / -0.000876` | `-0.000250` | `+0.000005 / +0.000171 / -0.000331` | `-0.000051` | no |
| B3 listed-equity option OI | `+0.000156 / +0.000470 / +0.001132` | `+0.000586` | `+0.000080 / +0.000346 / +0.000774` | `+0.000400` | no |
| DCE iron ore | `-0.000139 / -0.001702 / +0.000344` | `-0.000499` | `+0.000044 / -0.000620 / +0.000211` | `-0.000122` | no |

No path met the frozen mean `>= +0.001` and non-negative-every-fold gate. The
orthogonal pair was materially negative on Fold B: both its block-5 and block-10
95% intervals excluded zero. The options-OI candidate was the strongest P2
result and was positive on all three folds, but its mean was only `+0.000586`;
all three fold intervals included zero. It is retained as evidence that the
source was implemented successfully, not as a promoted model input. Final
EMA-0.995 was also sub-gate: its standalone three-fold means were approximately
`+0.000480`, `+0.000024`, `+0.000806`, `+0.000757`, and `+0.000145` in table
order. These secondary observations do not override the primary decision.

The full-eight failure resolves the F3 feature/pruning confound, while the fixed
near-orthogonal pair's failure closes the feature program: the selected features
do not provide enough incremental signal beyond the unchanged parent. Reject all
three P2 candidates under the registered gate. Do not combine the P2 sources or
use the designated challenger to rescue them after observing these results.

### Source acquisition and immutable artifacts

The missing option source was acquired correctly from an unrestricted Windows
host. The immutable official BVBG.086/BVBG.028 archive contains 1,154 complete
daily PR/IN pairs from `2019-11-01` through `2024-06-28` (15 GB); its manifest
SHA-256 is
`53ec0cdfba7c0f6eff5a7cbdf2bdfce258e750af8864d28cb737668940bd7d85`.
Accepted permanent-identity bounds begin `2021-07-19`, so the normalized builder
used that date through `2024-06-28`, retaining about 20 business days of causal
warmup before the model window while preserving the earlier raw archive. It
emitted 735 complete daily pairs, 95,045 rows, and 142 permanent IDs with no
duplicate keys, future availability, non-finite valid data, or nonzero invalid
data. Its Parquet SHA-256 is
`669ecd96c865d48324cf6ac414a68ec6bcccb9a66c0f55dc159e4a92cf28eeb5`.
The normalized lending and DCE Parquet SHA-256 values are respectively
`6e2cc7c3ae950b17ce6e9d7f5243272ce947567c92df0d7ad08abf7d5c3b0262`
and `6cbd6e08e57afcd86f4a2b5ae8dfabe983994f42a32102bd2bc262f60a574940`.

The immutable completed program is:

    /lambda/nfs/brazil-rv-east3/quant-data/b3/processed/model_runs/experiment40_final_feature_p2_0b6ff68_20260823T075000Z

Its `program_manifest.json` SHA-256 is
`495e69adf98b190a697c17d620a4ccb04dc02f4f05c13e4ce039ce0caee871ae`;
`program_summary.json` is
`db7ba8028f6955b2e753a4ed5272b792fd9f72fdc7325e5970a779d77814331c`.
A recursive audit read 216 JSON artifacts, 55 manifests, all 45 twenty-epoch
histories, and all five screen summaries with zero contract errors. Every
repository identity matched commit `0b6ff68a64276fff53c770b49b1ab9db64120e4b`.
Official validation and the held-out test remained sealed throughout.

A reviewed hash-bound cleanup removed only 1,595 redundant candidate checkpoint
and per-epoch prediction files (38,738,372,551 bytes, 36.078 GiB). It retained
all 45 observation references, epoch-20 raw/EMA containers, and every checkpoint
and prediction selected by either honest parity replay: 295 binary artifacts
totaling 7,027,551,029 bytes. Sidecars, histories, manifests, diagnostics,
analyses, summaries, and source archives were outside the deletion set. The
program root is 6.7 GiB after cleanup. The immutable plan/result are under
`_cleanup/20260823T170000Z`; their SHA-256 values are
`7d9ad2a6ad6ee15c007decfbb3e822a3a495143fafb2648a1969f0973daa4af4`
and `c95e997ee3522fd6d8725c4184f771c8d46b32870cb792192f68056daf888dc4`.
Postchecks found zero planned files remaining and zero damaged retained files.

After the result commit was pushed and persistent artifacts were rechecked,
Lambda accepted termination of exact paid GH200 instance
`95098103c2da4ffcb8e9d10a4ac7704c` (`gpu_1x_gh200`, `us-east-3`, IP
`192.222.58.49`). The instance was then absent in two consecutive provider
inventory reads. No paid Experiment-40 host remains running.

## Storage maintenance — Lambda object-store cleanup round 2 (2026-08-23)

A complete S3-compatible object inventory was taken after Experiment 40. The
bucket contained 18,970 objects / 109,637,262,343 bytes (102.108 GiB). Inventory
SHA-256:
`2a0567a0db123267ad8d42e56b8d10fbb704d8d06b121b1d74019eeb02617d14`.

The plan was derived from the completed Experiment-27 program manifests,
screen summaries, two-direction Patience-3 replays, and whole-fold trajectory
diagnostics. It retained the union of final epoch 20, both honest parity-selected
epochs, and whole-fold Patience-3 for every fold/seed. This produced 178 retained
prediction epochs across the 60 runs. The already-pruned 178 matching checkpoint
containers and all 60 validation references were also required to remain.

The exact applied plan removed 6,323 objects / 48,739,734,061 bytes
(45.392 GiB):

- 1,022 unselected frozen per-epoch predictions / 43,483,279,280 bytes;
- 60 redundant tail prediction bundles / 2,552,841,960 bytes;
- 5,211 ephemeral cache objects / 751,448,837 bytes; and
- 30 rejected-preflight sidecar objects / 1,952,163,984 bytes.

The rejected sidecar tree was deleted only after both old and accepted copies of
all arrays were streamed and their SHA-256 values matched the common manifest
hashes. Their manifests were semantically identical after excluding only
`created_at_utc`. The accepted sidecar tree remains complete.

The cleanup plan ID is `34b332f028db43a1b06465e713deb97c` and its SHA-256 is
`eda1cb77d361ac0a8fa8b5e00460aff71477812a29e63f9f46ded7535937b64b`.
The postcheck SHA-256 is
`cd68b22bc23a891a73eab6c1b75cc61e82e524dae02439153fb37cf965b23ce6`.
Both immutable records are stored under:

    quant-data/b3/processed/model_runs/_retention/storage_cleanup_20260823_round2

The external-data program shrank from 2,312 objects / 52.032 GiB to 1,230
objects / 9.158 GiB. The independent post-cleanup inventory contains 12,649
objects / 60,899,434,233 bytes (56.717 GiB), SHA-256
`20f8fa4258e914f2a7731bfd0cee42d809fc34488c5f0a2716a28c6e7d9ecce6`.
Set comparison against the original inventory found zero unexpected removals,
zero planned survivors, zero retained-object size/ETag changes, and exactly two
expected audit additions. Raw and interim sources, the canonical feature store,
parent/challenger artifacts, program results, and Experiments 39--40 were outside
the deletion set. No paid Lambda instance was active.

## Experiment 41 — incumbent feature removal and retrained pruning confirmation (preregistered 2026-08-23)

Status at registration: no Experiment-41 correlation table, ablation score,
removal set, retrained trajectory, official-validation score, or held-out-test
score exists. The objective is one decisive KEEP/REMOVE verdict for each of the
58 incumbent equity inputs (26 dynamic channels and 32 slow fields), followed by
at most two retrained candidates. No new feature, adapter, store rebuild,
official-validation access, held-out-test access, or change to the frozen
official-read lineup is authorized.

### Frozen inputs and Stage-A definitions

The input store is the unchanged canonical PIT-causal store at
`m1_features_pit_causal_tod_20260818T151728490951Z`. Stage A uses only training
dates `2021-08-16` through `2024-06-28`, respects the active-equity mask, and
uses only model-visible intraday minutes 0--284. All Spearman calculations use
average ranks for ties and require paired observed data.

- Slow/slow correlation is the mean of per-date active-equity cross-sectional
  correlations.
- Dynamic/dynamic correlation is computed two ways: the mean per-(date,
  decision) active-equity cross-sectional correlation and the mean per-equity
  time-series correlation. The signed value with larger absolute magnitude is
  retained; exact ties prefer the cross-sectional value.
- Dynamic/slow correlation is the mean per-(date, decision) active-equity
  cross-sectional correlation, broadcasting the dated slow value across that
  date's visible decisions. There is no slow-field time-series alternative.

An undirected edge exists at fixed `abs(rho) >= 0.80`; connected components,
including singletons, are frozen in a hash-bound Stage-A artifact before the
first Stage-B evaluation. The six additional semantic tests are fixed by
incumbent indices: slow beta fields 20--25; dynamic 10--12 plus slow 9--11;
dynamic 15 plus slow 15--16; slow 26--29; dynamic 16--21; and dynamic 24 plus
slow 12--14 and 18. Correlation components and semantic sets remain independent
tests even when they overlap; candidate removals are their union, subject to the
preview gate.

### Stage-B ablation and assembly contract

Stage B is inference-only. It uses the honestly opposite-parity Raw Patience-3
parent checkpoints and never changes a weight. A targeted input is set to zero
only for the 158 equity streams; context/global streams and observation/history
masks remain untouched. The three seeds are rank-averaged. Reports contain
parent-minus-ablated IC by Fold C/A/B and horizon, with paired block-10
intervals. The Experiment-39 P0.3 Fold-A/B single
ablations are imported unchanged; all 58 Fold-C singles are new.
The prior inventory-bound cleanup removed unselected A/B prediction epochs.
Therefore A/B Patience directions are loaded from Experiment 39's sealed
`validated_sources/validation_report.json`, whose historical and reproduced
selected/stopped epochs match exactly; the runner must not recompute selection
from missing files. Fold C continues to use its retained frozen replay metadata.

The group rules are fixed as follows. A set is group-dead when its mean joint
drop is at most zero and at most one fold is positive. A materially alive set
(mean joint drop at least `+0.00025`) receives one representative-sufficiency
test. Its representative maximizes the three-fold single-field drop; exact ties
prefer the field with fewer missing observations, then the shorter canonical
feature name as the simpler semantics, then lower global index. A set is
representative-sufficient when the residual drop after retaining that
representative is at most `+0.00025`; otherwise every member is KEEP.
The canonical loader has no per-field missing mask: every active equity receives
a finite value for all 58 fields, while missing-bar state is carried by the
dedicated observed/fraction channels. Consequently the missing-fraction
tie-break is zero for every incumbent field and is still recorded explicitly;
the name-length and global-index tie-breaks resolve any remaining exact tie.
Singletons enter R1 only when they satisfy the frozen Experiment-39 P0.3-dead
rule on both A/B evidence and their new Fold-C single drop is non-positive.

R1 is the union of removals from group-dead sets and eligible singletons. R2 is
R1 plus non-representatives from representative-sufficient sets. A field may be
proposed by one overlapping set while another set says KEEP; the proposal still
enters the preview because the decisive safeguard is the joint preview followed
by retraining, not an undeclared veto. Each preview must have mean cost at most
`+0.00025` and every-fold cost at most `+0.0005`. When R1 fails, it is rebuilt
from empty by adding fields in ascending single-drop order and stops before the
first failing addition. R2 starts from the final R1 and considers only its extra
fields in the same order, preserving nesting. This is the fixed interpretation
of the requested "smallest single-field drop first" greedy walk-back. Final R1,
R2, their preview metrics, and the full provenance of accepted/rejected
additions are frozen and hashed before Stage C begins.

### Stage-C decisive retraining and selection

Exactly two candidates are permitted: prune-R1 and prune-R2. Each keeps the
parent architecture, loss, sampling, SAM optimizer, initialization/RNG, and
20-epoch training contract unchanged, while zeroing its frozen equity fields in
the loader from epoch zero. Each runs seeds 11/29/47 on Fold C/A/B: 18 total
trajectories at hard maximum parallelism two. Bidirectional cross-fitted Raw
Patience-3 standalone candidate-minus-canonical IC is primary. Final EMA-0.995,
the uniform parent-plus-pruned rank ensemble, block intervals, horizons, and
decision-time slices are diagnostic only.

A candidate is non-inferior when its three-fold primary mean is non-negative and
no fold is below `-0.0005`. It meets the numerical improvement threshold only
when the mean is at least `+0.0005` and every fold is non-negative; paired
block-5/10 intervals determine how strongly that improvement is supported and
must accompany any claim. A non-negative sub-threshold outcome is parity.
If both are non-inferior, prefer R2 when its mean is within `0.00025` of R1;
otherwise choose the higher mean. If only one is non-inferior, it wins. The
winner defines a store-v2 feature specification for the next-generation parent,
but does not silently replace the canonical recipe. If neither is non-inferior,
all 58 fields remain and the conclusion is "redundant but load-bearing under
retraining — do not remove." No third subset or post-score rescue is allowed.

The immutable program must retain the Stage-A matrix/table, complete Stage-B
set-by-fold/horizon drop matrix and intervals, hashed R1/R2 definitions and
walk-back trace, all Stage-C histories and analyses, a rule-attributed verdict
for every field, hashes/manifests, and sealed-data flags. Cleanup is allowed only
after the final decision is recorded and must be inventory-bound; selected
checkpoints, analyses, manifests, and decision artifacts are required survivors.
The experiment-closing cleanup initially retains all prediction archives; a
later explicitly authorized global storage cleanup may reduce them to the exact
epoch-20/cross-fit/whole-fold Patience union only after an immutable plan and
full-bucket postcheck are written.

### Completed result (2026-08-24)

Stage A clustered the 58 incumbent fields under the frozen training-only
correlation contract. Stage B then froze nested candidates of 16 fields for R1
and 24 fields for R2. Their inference-only parent-minus-ablated preview means
were respectively `-0.001899` and `-0.002240`; both passed the fixed preview
gate on every fold. These favorable inference ablations were treated only as
candidate-construction evidence, not as the feature-removal decision.

All 18 Stage-C trajectories completed. Prune-R1 produced Raw Patience-3
candidate-minus-parent deltas of `+0.000741/-0.000812/+0.002965` on Fold C/A/B,
mean `+0.000965`. It failed non-inferiority because Fold A breached the fixed
`-0.0005` floor. Prune-R2 produced `+0.000898/-0.000216/+0.002978`, mean
`+0.001220`, and passed non-inferiority. Its paired block-10 95% intervals were
`[-0.001939,+0.002712]`, `[-0.001908,+0.000963]`, and
`[+0.001665,+0.004946]` on C/A/B. It did not meet the separate numerical
improvement rule because Fold A remained slightly negative. The diagnostic
parent-plus-prune-R2 stack was positive on all three folds
(`+0.000794/+0.000273/+0.001959`), while final EMA-0.995 was mixed and
subordinate to the primary read.

Under the preregistered selection rule, prune-R2 is the winner and defines the
next store-v2 feature specification. Remove these 24 equity inputs:

- dynamic: `return_60m_normalized`, `realized_vol_30m_log_ratio`,
  `session_range_position`, `cross_section_return_rank_15m`,
  `cross_section_volume_rank`, and `cross_section_volatility_rank_30m`;
- slow return/liquidity/coverage: `overnight_gap_normalized`,
  `previous_close_to_close_return_normalized`,
  `previous_open_to_close_return_normalized`,
  `median_daily_real_volume_20d_log_scale`,
  `median_daily_dollar_volume_20d_log_scale`,
  `daily_dollar_volume_regime_20d`, `observed_fraction_5d`,
  `observed_fraction_20d`, and `dollar_volume_cross_section_rank`;
- slow betas: `beta_to_WIN`, `beta_to_DI1F27`, `beta_to_DI1F28`,
  `beta_to_DI1F29`, and `beta_to_DI1F31`; and
- slow calendar: `weekday_sin`, `weekday_cos`, `month_end_proximity`, and
  `quarter_end_proximity`.

The other 34 fields remain. In particular, the preview walk-back proposed but
did not remove `volume_surprise`, `market_median_return_15m`,
`market_median_return_60m`, `market_breadth_15m`, and
`cross_section_return_rank_60m`; their verdict is KEEP. The selected
specification is for a future rebuilt parent and does not silently change the
current canonical recipe or official-read lineup. Official validation and the
held-out test remained sealed.

The immutable Stage-A/B source program is:

    /lambda/nfs/brazil-rv-east3/quant-data/b3/processed/model_runs/feature_removal_d5b5e1f_20260823T224100Z

It ran at commit `d5b5e1f5b56fcdb02b6a76ff37363ac841cc4e6e` and froze the
Stage-A table and Stage-B sets at SHA-256
`e05ebb1357b45db11f962818d80d50e2f12006335aaf3a7389a30adab38b0ade`
and `4673a418745847c1a23210f6ed4f5513c19f76ceb7ebd216c9de7ff18e72ca54`.
After those sets were frozen, the original parent retained 72.5 GiB of compiled
Stage-B CUDA state and caused the first Stage-C worker launch to fail before a
trajectory completed. The failed root and partial manifests were preserved.
Stage C was rerun in a fresh isolated process/root, reusing the exact frozen-set
hash and none of the partial Stage-C files:

    /lambda/nfs/brazil-rv-east3/quant-data/b3/processed/model_runs/feature_removal_stage_c_repair_d5b5e1f_20260823T232938Z

The repair summary and store-v2 specification SHA-256 values are
`1070ecfadb99eef42d224b8eacc0ef31fc8e0e08ecc6a7e39aa5153e57fb18b8`
and `08c04de3396fdc31d67b6baeabab1fea80cfd137d55bf2a1aef4ee69d1a34b72`.
The final audit checked all 58 verdicts, 18 twenty-epoch histories, 360 original
checkpoint containers, all 360 prediction archives, 24 analyses, 67 JSON
artifacts, exact store/commit/definition hashes, and sealed-data flags with zero
errors. Its SHA-256 is
`b423d0f76c7e5f4e7aa88de1f8706ef01e678bd3f87be32828729c5b1443006b`.

A reviewed inventory-bound cleanup deleted only 315 unselected checkpoint
containers (1,420,508,533 bytes). It retained all 360 prediction archives and
the union of epoch 20, both cross-fit Patience-selected epochs, and whole-fold
Patience-selected epochs: 45 checkpoints. Every retained artifact was rehashed
after deletion. The cleanup plan/result SHA-256 values are
`0bca5cd90abb99f8294baaf947bd024dc387e2ebff9c4fb157a4b31f5be3c9d6`
and `0f5242e759665f4e2bfe2b0ba3a6c0ec8045a57cd61cfeeb1c6c3d9bb843b8ea`;
both are under `_cleanup/20260824T030400Z` in the repair root. The canonical
runner now executes Stage B in an isolated process so compiled inference state
cannot consume Stage-C worker memory on an exact rerun.

A later explicitly authorized persistent-storage cleanup used a fresh complete
object inventory and the frozen keep-epoch sets from that first cleanup. It
removed 315 unselected per-epoch prediction archives plus 18 redundant
`tail_candidates.npz` bundles, exactly 333 objects / 14,309,636,868 bytes
(13.327 GiB). It retained the 45 selected/final prediction archives, their 45
matching checkpoints, all 18 observation-alignment references, all histories,
24 analyses, manifests, summaries, field verdicts, the store-v2 specification,
and both sealed-data flags. The Stage-C repair root fell to 2,574,433,193 bytes.

The cleanup plan ID is `2b3dd77f79e34dfa92bd4b015fe73e10`; plan and postcheck
SHA-256 values are
`f6c0c1d07088a3829671f43faaf9c06b3b42a9e72eacc0d056b94324a90bc947`
and `fd6eaa3c2d71dccb37ee2d1aa039c7483133d776c7fdb13f6ab13fae044af88e`.
Both immutable records are under:

    quant-data/b3/processed/model_runs/_retention/storage_cleanup_20260824_round3

An independent post-cleanup inventory contained 12,941 objects /
63,474,753,532 bytes (59.115 GiB), SHA-256
`46a11d4fdfa9c29ad9ca97fa7f78a863cc0b250c2df3cf18f2fa95f3e05145ec`.
Comparison against the source inventory found exactly the 333 planned
removals, zero planned survivors, zero unexpected removals, zero retained-object
metadata changes, and exactly the two expected audit additions.

After result commit `8f7871c77f379efc4fb18df8d9e824d8fc692687` was pushed and
the persistent artifacts were rechecked, Lambda accepted termination of exact
paid GH200 instance `e975b774f5834e0fa265d11bbbef680f` (`gpu_1x_gh200`,
`us-east-3`, IP `192.222.50.236`). It was absent from two consecutive provider
inventory reads at `2026-08-24T03:11:31Z` and `2026-08-24T03:12:02Z`; the
account then had zero active instances.

## Experiment 42 — correlation-conditioned R3 and full options program (preregistered 2026-08-24)

Status at registration: no Experiment-42 correlation table, inference-ablation
score, R3 definition, R3 trajectory, full-options source, option-candidate
trajectory, fold score, official-validation score, or held-out-test score
exists. This is one immutable program with two independent decisions. Both use
the selected Experiment-41 prune-R2 candidate as their sole fold comparator;
the historical 58-field parent is not a comparator. Official validation and
the held-out test remain sealed throughout.

### R3 contract

Stage B-prime recomputes the Experiment-41 correlation estimators and the fixed
`abs(rho) >= 0.80` graph on only the 34 prune-R2 survivors, using all 716
training dates, active/ready equities, and model-visible minutes. The old
semantic groups are intersected with the survivors and remnants of at least two
fields remain tests. Prune-R2's retained bidirectional Raw Patience-3 replay
epochs are loaded from its hash-bound Stage-C analyses; Stage B-prime zeros the
24 prune-R2 fields in the evaluation loader before any new inference ablation.

All 34 survivors receive three-fold single-field ablations. A singleton is dead
only when its overall parent-minus-ablated drop is non-positive on every fold
and at least two of three horizon drops are non-positive on every fold, exactly
matching the Experiment-39/41 dead rule. Every new correlation component and
semantic remnant is jointly ablated. A set is group-dead when its mean joint
drop is at most zero and at most one fold is positive. A materially alive set
(mean joint drop at least `+0.00025`) receives one representative-sufficiency
test. The representative maximizes the three-fold single drop; ties prefer
lower missing fraction, shorter canonical name, then lower global index. The
remaining members qualify only when their residual joint drop is at most
`+0.00025`.

The union of qualified new fields receives one joint preview on prune-R2. It
passes only when mean cost is at most `+0.00025` and every fold cost is at most
`+0.0005`. On failure, the set is rebuilt from empty in ascending single-drop
order and stops before the first failing addition. If no new removal survives,
R2 is the correlation-conditioned frontier and no R3 trajectory runs. Otherwise
exactly one prune-R3 candidate runs 20 epochs for seeds 11/29/47 on folds C/A/B
with at most two processes. R3 replaces R2 only when Raw Patience-3 mean delta
is non-negative and no fold is below `-0.0005`; a numerical improvement claim
also requires mean at least `+0.0005` and every fold non-negative. Final
EMA-0.995 and the uniform prune-R2-Patience plus R3-EMA stack are informational.
There is no R4, alternate removal set, or post-score edit.

### Full-options source and candidates

The optional F2 trim is not used. This avoids another selection layer and fixes
the full candidate to all 14 named fields: the unchanged six Experiment-40 OI
fields; option/cash quantity ratio, put/call quantity ratio, and prior-20 option
trade-count surprise; and five IV fields (prior-20 ATM-IV robust z, one- and
five-session ATM-IV changes, OTM-put skew, and ATM-IV minus realized-20 spread).
Every invalid cell is masked and exactly zero. Every D-dated value is first
available on the next observed B3 session with no filling.

COTAHIST option rows are joined by exact same-date option ticker to the
BVBG.028 instrument master. Option type, expiry, and strike must agree, and the
master's explicit underlying instrument ID must resolve to the same accepted
cash ISIN recorded by COTAHIST. Ticker prefixes are never used. IV inputs require
at least 20 contracts, at least three trades, 5--45 calendar days to expiry, and
positive close premium (otherwise positive average premium). The risk-free rate
is the same-session final annual-percentage quote of the shortest non-expired
fixed-maturity DI contract, divided by 100. A qualifying same-strike/expiry
call-put pair supplies `F = K + exp(rT) * (C-P)` using median premiums per side;
otherwise the exact cash close is the forward proxy. Discounted-forward
Black-Scholes is inverted by 64-step bisection on volatility `[0.05, 3.0]`.
ATM is `abs(log(K/F)) <= 0.10`; OTM put is
`-0.25 <= log(K/F) <= -0.05`. ATM IV is the median and requires at least two
solved ATM series; the skew leg is the median solved OTM-put IV. American-style
contracts are approximated as European and dividends are ignored, both recorded
as known approximations.

Prior-20 level/surprise fields use only already-emitted observations and robust
median/MAD scaling clipped to five and divided by five. One- and five-session
changes require the exact prior B3 session; realized volatility uses the 20
cash-close returns ending at D and is annualized by `sqrt(252)`. Fixed transforms
are: log-ratio/tanh scale four for option/cash quantity, scale three for put/call
quantity, tanh scale `0.25` for IV changes and skew, and tanh scale `0.50` for the
IV-realized spread.

Exactly two bias-free zero-start sidecar candidates run on the prune-R2 field
mask: opt-full (all 14) and opt-IV (the five IV fields), each seeds 11/29/47 on
folds C/A/B for 20 epochs. Standalone bidirectional Raw Patience-3 is primary;
final EMA-0.995 and the predeclared uniform prune-R2-Patience plus candidate-EMA
mixed state are secondary. A path passes only when its three-fold mean is at
least `+0.0005`, at least two folds are strictly positive, no fold is below
`-0.0005`, and the pooled paired daily delta across the three non-overlapping
fold windows has a 10-session moving-block 10,000-replication 90% interval that
excludes zero. If both standalone candidates pass, the higher mean advances. A
mixed state is eligible only if both standalones fail; if both mixed states pass,
the higher mixed mean advances. No third subset exists.

A passing fold screen registers, but does not execute, a future official-read
arm. Its preparation record requires the identical source contract through
2025-06-30, full-716-date members on the final store-v2 specification, and the
already frozen read lineup. This program neither acquires post-training data nor
accesses official validation/test. It retains all prediction archives,
selected/final checkpoints, histories, analyses, sidecars, source diagnostics,
hash manifests, field verdicts, and sealed-data flags until a reviewed
inventory-bound cleanup is written after the decisions.

### Completed result (2026-08-24)

Stage B-prime proposed seven further removals from the 34 prune-R2 survivors.
Its frozen walk-back accepted only `realized_vol_60m_log_ratio`,
`realized_vol_20d_log_ratio`, and `vol_of_vol_20d`. The accepted three-field
inference ablation passed its preview gate with parent-minus-ablated deltas of
`-0.000797/-0.001842/+0.000368` on Fold A/B/C (mean `-0.000757`). This was only
candidate-construction evidence. The single retrained R3 candidate added
`+0.000496/-0.001669/-0.000260` Raw Patience-3 IC on A/B/C, mean
`-0.000478`. It failed both non-inferiority and improvement, so R3 is rejected,
R2 remains the selected 34-field store-v2 specification, and the frozen hard
stop forbids R4 or a post-score removal edit.

The full 14-field options candidate added
`-0.000410/+0.001526/-0.000073` standalone on A/B/C, mean `+0.000348`.
It failed the mean, fold-count, and pooled-uncertainty checks; its pooled
10-session moving-block 90% interval was `[-0.000684,+0.001367]`. The
predeclared parent-plus-options mixed state had mean `+0.000994` but failed the
per-fold floor on Fold A (`-0.000676`) and its pooled interval also included
zero (`[-0.001501,+0.003399]`).

The five-field IV candidate was materially negative standalone:
`-0.001134/-0.002320/-0.000563` on A/B/C, mean `-0.001339`, with pooled 90%
interval `[-0.001953,-0.000696]`. Its mixed state added
`-0.001200/+0.000215/+0.001763`, mean `+0.000259`, and failed the mean,
per-fold, and uncertainty gates. Neither options path advanced; the options
family is parked, no third subset was run, and the future-read preparation
record is `not_applicable`. Optional F2 trimming was not used.

The causal full-options source contains 95,045 dated security rows for 142
permanent IDs from 2021-07-19 through 2024-06-28. It used the exact four
COTAHIST archives, official BVBG instrument masters, the retained option-OI
source, and fixed DI risk-free inputs. The one manifest-declared unpublished
instrument-master date, 2023-12-08, remains invalid for activity and IV fields;
no adjacent-day master or invented identity was substituted. The source output
SHA-256 is
`6a0cff033fb48a3b190ba49389e173c385ee1df0211a335e81665e9ec2af5686`.

All 27 twenty-epoch trajectories completed under exact repair commit
`9a05b1d620d51672956d02765dcebcd65292715a`. The original preregistered root
failed before any score because the immutable BVBG manifest correctly records
`IN231208.zip` as not published. The causal pre-score repair was tested with 269
local and seven instance tests; its audit SHA-256 is
`9b1f234799b11714c3fa9ed1c63433fceb64a5acd0fc7a1d3b4339b3de10e967`.
The completed program is:

    /lambda/nfs/brazil-rv-east3/quant-data/b3/processed/model_runs/r3_options_9a05b1d_20260824T053052Z

Program, R3-summary, options-summary, source-manifest, and original artifact-
inventory SHA-256 values are respectively
`888a1cef6c1488365db3870aa434127767c36ee544a82b8deed58dab7382f91d`,
`a058efc42b66be5117b13c044c774cf531ba57df8294823121bfd3877d6df452`,
`118746188fe2cc5d61c9f7dfad7a7a68173c433b6ff11979c957907d4cf4dafb`,
`d03afc6c75fff445bac4d57df09b72d373f459e1edc08d4e15af679c20711509`,
and `852ae14cc959c65c3786ac9fef16462600c5f3407858ba4664653c8a3b7bf210`.
The final audit checked all 27 histories, 540 checkpoints, all 540 prediction
archives, exact inputs/decisions/hashes, and sealed flags with zero errors; its
SHA-256 is
`752036376719d08faab6bea69003283de40d882391e352223645f955d23f5a8c`.

The reviewed inventory-bound cleanup removed only 467 redundant checkpoint
containers (2,113,294,685 bytes). It retained every one of the 540 prediction
archives and the exact union of epoch 20, bidirectional cross-fit Patience
epochs, and whole-fold Patience epochs: 73 checkpoints. No retained artifact
changed. Cleanup plan ID is
`2b03829993933bd421898bc3488f8a051cfd6bee7c3cd9c018834e75817440e8`;
plan/postcheck SHA-256 values are
`13e7f8203a47b5c53ddf12dbafd322855ac3abae70be16a54040086e3dc61d7a`
and `34fd29c1887d066fa247f15737b17256bfb7f6fd30808b8d2a32a3ad2ce4fb68`.
The post-cleanup inventory has 875 program artifacts / 25,374,165,801 bytes,
SHA-256
`dbc5304c79cd27ca7112e6b6b98029311314eb03eda6176a3347f20115e746eb`.
Official validation and the held-out test remained sealed throughout.

After result commit `41e54c9654ecb3ab3dd98e83b084763661a8c25c` was pushed and
the retained artifacts were rechecked, Lambda accepted termination of exact
paid GH200 instance `e2cf2e517d9541ac93cac3906fc5c0e4` (`gpu_1x_gh200`,
`us-east-3`, IP `192.222.50.236`). It was absent from two consecutive provider
inventory reads at `2026-08-24T12:16:44Z` and `2026-08-24T12:17:04Z`; the
account then had zero active instances.

## Experiment 43 — official-validation read and conditional 10-seed expansion (preregistered 2026-08-24)

Status at registration: no Experiment-43 trajectory, prediction, analysis,
promotion decision, supplementary measurement, or deployed declaration exists.
The exact implementation commit is
`a441307f23cbc058f17fcbce5f102cb7a84d4c05`. The complete immutable contract is
`research/preregistrations/experiment43_official_read.md`; this section and that
file are frozen before training any arm or opening any official-validation
prediction. This is official-validation access event 3. The held-out test is
not authorized and remains sealed.

The comparator is the stored Experiment-1 parent-3 Raw Patience-3 ensemble.
Arm 1 is that parent-3 plus the stored Experiment-24 residual-3 final-EMA-0.995
members. Arm 2 is the Experiment-41 prune-R2 loader mask retrained on all 716
training dates at seeds 11/29/47 with matched official-monitor Raw Patience-3.
Both use uniform tie-aware rank averaging. The only promotion evidence is each
arm's paired 244-date block-10 95% interval versus canonical: lower bound above
zero means supported; one supported arm is promoted, two supported arms resolve
by higher official IC, and no supported arm leaves canonical deployed.

Only a promoted arm opens the fixed seven-seed expansion at seeds
61/79/97/113/131/149/167. The 10-seed form deploys if its official IC is at
least the promoted three-seed IC minus `0.0005`; otherwise the three-seed form
deploys. No options arm, hybrid, extra subset, fold screen, retuning, new data,
test read, post-score edit, or second official read is permitted. At most two
processes run concurrently. All prediction archives and deployed selected/final
checkpoints are retained, results and the validation-access ledger are pushed,
and the exact paid instance is terminated and verified absent twice.

### Completed result (2026-08-24)

All three preregistered store-v2 official-monitor trajectories completed. The
canonical parent-3 ensemble scored `0.041639843`. The stored six-member
challenger scored `0.042093822`, a `+0.000453978` delta, but its paired block-10
95% interval was `[-0.000684798,+0.001534552]`. The store-v2 ensemble scored
`0.043235373`, a `+0.001595530` delta, with positive 30/60/120-minute deltas of
`+0.001408910/+0.001476892/+0.001900788`; its paired block-10 95% interval was
`[-0.000294105,+0.003434960]`. Both point estimates were positive, but neither
interval excluded zero. The frozen decision therefore supports no arm,
canonical parent-3 Raw Patience-3 remains deployed, and the conditional
seven-seed expansion was correctly not run.

The completed program is:

    /lambda/nfs/brazil-rv-east3/quant-data/b3/processed/model_runs/official_read_c04ea91_20260824T140900Z

The exact preregistration SHA-256 is
`a30c0bc20565439b7ff3628c2a4576618698e387cf593eadbdc5fed47a6ba938`.
The challenger analysis, store-v2 analysis, promotion decision, deployed recipe,
validation-access ledger, and completed program-manifest SHA-256 values are
`f3442dd80a2da8c9f1989c19f2c4a9c08317f54dde012d11ce71d0636b8c2f29`,
`53f9082775a90ac5d28b5573b827c518b7e4c409aecbde2273e02c18d9343652`,
`80ef2a79ba6755983e125946211292dbc4ace0f1f8ac1f126917a9449d9924ea`,
`b5a4f78b3a054822893e6849c16abc4fc14ca7cf2add421bfdec7f4617be6b6d`,
`c06f618f16f01269fb8d0334591bb59aa0454df64e64781d2313268bbb9661ce`,
and `6dbe8314262dd61c0dfda835055282644aed83053149be57398b86673da9ff85`.
The implementation used preregistration commit `c04ea917` plus pre-score
operational path repair commit
`0c05100e9ca527eab3421bba9128d15e6a0fc573`; the repair only resolved the
immutable Experiment-41 specification at its actual `stage_c` location and
occurred before any official prediction was opened.

All three 20-epoch histories completed, all 63 official prediction archives
were retained, every required-output hash passed, and every JSON audit found
`test_accessed=false`. Because none of the newly trained store-v2 members was
deployed, the reviewed cleanup removed exactly their 60 checkpoint files /
270,571,140 bytes while preserving every prediction archive. Cleanup plan and
postcheck SHA-256 values are
`f35fd302f27db7e6c88e8cc4801a5794fce051e4163386b7584a0afa7592534f`
and `49fbcb0a93563d9fbbd44b9279327b203df5fd2691306fc577a52a0f41ab6ea8`;
the retained inventory SHA-256 is
`12135cbd84c61afe95722c5dab8106d0feb8e6acc7a035e62770b918ad0fee08`.

After result commit `880a66e40b1295cc31ea0b51e392caee5a9eef4c` was pushed and
the persistent program was rechecked, Lambda accepted termination of exact paid
GH200 instance `c2da7efd0ab645178a847aad8fdf12c8` (`gpu_1x_gh200`,
`us-east-3`, IP `192.222.50.160`). The instance was absent from consecutive
provider inventory reads at `2026-08-24T15:34:25Z` and
`2026-08-24T15:34:54Z`; the account then had zero active instances.

## Experiment 44 — ensemble science E1/E2 (preregistered; completed)

The full immutable protocol is
`research/preregistrations/experiment44_ensemble_science.md`. Before any new
combination score or trajectory, Experiment 43 Amendment A1 was frozen at
`research/preregistrations/experiment43_amendment_a1.md`. A1 preserves the
historical Experiment-43 decision under its original gate, introduces separate
non-inferiority and superiority tracks, and requires an exact store-v2
full-window reproduction before store-v2 becomes the standing comparator.

Experiment 44 is one discovery-only session: inventory and hash-freeze the
retained prediction roster, run the fixed E1 rank-combination rules, manufacture
exactly 45 E2 store-v2 members, rerun the fixed combination rules, and name at
most one future official-read arm. Every E1/E2 candidate is complexity-adding
and must add at least `+0.001` mean IC with no negative held-out fold and paired
block-5/10 support. Official validation and the held-out test are sealed; no E3,
HPO, deployment change, or post-score rule expansion is authorized.

### Completed result (2026-08-25)

Amendment A1's separate full-window reproduction did not satisfy its exact-
match condition. The three input manifests matched after excluding provenance,
runtime, and outcome fields, but none of the 60 epoch prediction archives was
byte-identical to the Experiment-43 source. The resulting IC differences were
small, consistent with operational GPU nondeterminism, but the frozen rule was
exact: A1 failed and the historical Experiment-43 deployment decision remains
unchanged. The immutable failure-audit SHA-256 is
`b0ef52109aa3b0e1516a7db0f7ef05bfc9542a844b0328c2fc68792439e66238`.
The E1/E2 protocol explicitly did not depend on A1, so the discovery program
continued without changing its comparator or any scored rule.

E1 materialized and hash-verified the frozen archive roster before computing
scores. None of its 17 fixed-grid cells passed. Its honest greedy path added
`+0.001544028/+0.000254077/+0.001099957` on held-out Fold C/A/B, mean
`+0.000966020`, missing the frozen `+0.001` complexity-adding threshold.
The three predeclared hygiene rules also failed. E1 therefore closed the
uniform-weighting-only hypothesis without registering an arm.

All 45 frozen E2 trajectories completed: nine date-block bagged members, nine
feature-subspace members, and 27 horizon specialists, each with Raw Patience-3
and final EMA-0.995 archives. No one of the 20 E2 fixed-grid cells passed. The
E2-only honest greedy path added
`+0.001332262/+0.000535001/+0.000134179` on C/A/B, mean `+0.000667147`, and
failed. The predeclared full-pool greedy path over E2 plus the primary archive
roster passed: `+0.001739170/+0.000767959/+0.000721738` on C/A/B, mean
`+0.001076289`. Every fold was nonnegative, the mean exceeded `+0.001`, and
the paired block-5/10 intervals were reported as required. Exactly one arm,
`e2_plus_archive`, is registered for a future official-read preregistration.
Its fold-specific cross-fit compositions and repeat weights are frozen in the
analysis. This is not a deployment decision and does not authorize an official
read by itself.

The completed program is:

    /lambda/nfs/brazil-rv-east3/quant-data/b3/processed/model_runs/ensemble_science_8dff0be_20260824T174700Z

Frozen-design, pre-score source-map, E1-analysis, completed-analysis, E2-
training-manifest, and E2-member-catalogue SHA-256 values are respectively
`e0e991079e88f146534a24f3fd94de545f8d773de4547fc49d395d984523d8d9`,
`6f7f419dbe5c60ea0b55cf358d23195b23b1086c037847ebc49638cd58634c60`,
`29fcb869475e0c124b38b5a485d0d6b92f2de6cb97619a12120717e204003cd5`,
`b933e740034cf6499d4569d98ebdb4242c697ee6664c44cafcbc9c606aa913b7`,
`0a33c92f60d26e402685b62dca6201f722bd8d2f537c8859d5b4f486565098c1`,
and `538dbd3cea54e066b7d7af59900a847cdd4e465faea97af1ed286afaa71cb157`.
The audit verified all 90 selected-state E2 catalogue records, their three
unique observation references, source-run manifests, and hashes. A separate
complete preservation inventory rehashed all 945 epoch/tail prediction
archives and all 45 per-run reference archives. Its SHA-256 is
`b8a8d21600f2655f2869343dfa20564c00b573de43fe70cd50e6b1ea7bdaaa67`.
Recursive JSON access checks found `official_validation_accessed=false` and
`test_accessed=false` throughout Experiment 44.

The reviewed inventory-bound cleanup removed exactly the 900 E2 per-epoch
checkpoint files / 4,060,129,980 bytes after recording every path, size, and
SHA-256. It preserved and rehashed all 945 prediction archives plus all 45
per-run references, and left zero E2 checkpoints. Cleanup plan and passing
postcheck SHA-256 values are
`e471da7822e7463e2b66001a1e7bb35167e2cc20062011d28a50d19678c4d4cc`
and `45d74a3d9cd9fd02850d20a889c0107ec3c0aa7f54d5974c77447787ecef5ebc`.
The exact evidence is under the program root's `_cleanup/20260825T045600Z`
directory.

After result commit `ffbb5b672ebd982ad3ebe748558f81bfb22e00d1` was pushed and
the retained archives were rechecked, Lambda accepted termination of exact paid
GH200 instance `74ecb13e8b704ffcad890a5930ee74fd` (`gpu_1x_gh200`,
`us-east-3`, IP `192.222.50.236`). Consecutive provider inventory reads at
`2026-08-25T05:16:27Z` and `2026-08-25T05:16:47Z` confirmed the exact instance
absent and the account at zero active instances.

## Experiment 45 — consolidation read (preregistered 2026-08-25)

Status at registration: no Experiment-45 trajectory, official prediction,
analysis, promotion decision, or deployment declaration exists. The complete
immutable contract is
`research/preregistrations/experiment45_consolidation_read.md`; this summary and
that file are frozen before training any member or opening any official-
validation prediction. This is official-validation access event 4. The
held-out test is not authorized and remains sealed.

The sole comparator for decisions is the retained Experiment-43 store-v2
three-seed prediction ensemble at official IC `0.043235373`; canonical parent-3
is reference-only. Arm 1 freshly realizes store-v2 at the exact ten frozen
seeds under matched official-monitor Raw Patience-3. Its fresh three-seed
result has an informational `+/-0.0015` reproduction guard: a breach completes
measurement but halts every deployment declaration. Otherwise ten seeds deploy
iff their IC is at least fresh-three IC minus `0.0005`; fresh three deploys if
not.

Arm 2 applies the blind Experiment-44 consensus rule exactly: repeat count at
least two for non-comparators, the three comparators always present, repeat-
count weights, the frozen total/gain/lexical cap ordering, weighted tie-aware
rank averaging, and specialist horizon coverage. Full-window realization is
capped at 20 new trajectories beyond Arm 1. Raw Patience-3 members use the
official monitor; final EMA-0.995 members use fixed 20-epoch training with only
their final read. Arm 2 is supported only when its paired block-10 95% lower
bound versus the retained comparator is strictly above zero; support
supersedes Arm 1. No hybrid, new arm, post-score edit, second read, or test read
is permitted.

At most two training processes run concurrently. Every prediction archive and
analysis is retained. All evaluated checkpoints remain through the pushed
deployment declaration, after which only inventory-reviewed non-deployed
checkpoints may be removed. The deployed measured members' selected and final
checkpoints remain retained until a future deployment supersedes them. Results,
hashes, and the validation-access ledger are pushed before the exact paid
instance is terminated and verified absent twice.

### Experiment 45 result (completed 2026-08-25)

The frozen program completed official-validation access event 4 at exact
implementation commit `75956f56b38aacfe92c57975fa34552b37a97b3c`. Its
immutable root and frozen-design SHA-256 are:

    /lambda/nfs/brazil-rv-east3/quant-data/b3/processed/model_runs/consolidation_read_e2eb713_20260825T105134Z
    05e0eebfffe95fc870e3e8a1138689aebe5e0595a210333d03a60cd15f1ebe2b

All 18 frozen trajectories completed: the ten Arm-1 store-v2 seeds and the
eight exact Arm-2 member realizations. The completed artifact retains all 298
official prediction archives, 18 observation references, all analyses, and
the 284 pre-cleanup checkpoints. The validation-access ledger is completed for
event 4 and records `test_accessed=false`; the held-out test was never opened.

The reproduction sanity check passed. Fresh store-v2 seeds 11/29/47 scored
`0.043239944968` versus the retained Experiment-43 comparator at
`0.043235373213`, a delta of `+0.000004571755`, well inside the frozen
`+/-0.0015` band. The ten-seed ensemble scored `0.043718770472`, adding
`+0.000478825504` versus the fresh three-seed ensemble and satisfying the
frozen deployment rule (`fresh-three - 0.0005`). Arm 2 scored
`0.043916831114`, a point delta of `+0.000681457900` versus the comparator,
but its paired block-10 95% interval was
`[-0.000853731696, +0.002373780406]`. Because the lower bound did not exceed
zero, Arm 2 was not supported. The measured ten-seed store-v2 recipe is
therefore deployed; no hybrid or post-score candidate was evaluated.

The deployment declaration retains the selected and final checkpoints for all
ten measured deployed members: 20 files / 90,194,220 bytes, with every path,
size, and SHA-256 recorded. Key completed SHA-256 values are:

- promotion decision: `28f185d32aa9948d8569abaa68255c789bb56d0b0daac284b9f427ac5dd7224b`;
- deployed recipe: `d4729dc4e614e0edd5118ba5ed5b7bc92f69ca2faceab4a09d0559115e5c4058`;
- validation-access ledger: `c03e00433c03fcb0ecf9b543e596563e3089fb59f2583103c25b32a6aeebd33f`;
- completed program manifest: `64a69be9345a0d9cc4a2a506f9f295fd50f9a2bcc210793458adfa9429cda34e`;
- Arm-1 fresh-3/comparator analysis: `5bf50739d89cacd7f84eddecf2cb20baf417dc48cd0cc9a28ead6595547d2b34`;
- Arm-1 fresh-10/comparator analysis: `e1ef80298fc2fe00934e5c6271b186cc348ac50875cee0b17ec59e6f36cb73d7`;
- Arm-1 fresh-10/fresh-3 analysis: `2fcc845da5d48a7900c20c89c1840dd32d0a775ab5d47109851e2ebe4bd98e0f`;
- Arm-2/comparator analysis: `90de369e1b87a9fb70fac313c9fe972bc39b745b371bb3c8753621adb6b183b7`.

After every frozen score had been produced, the two 30-minute specialist jobs
hit a JSON-only finalization failure because their deliberately untrained
60/120-minute metrics were NaN and strict JSON forbids NaN. The operational
repair used only their already-saved reference and selected prediction
archives, encoded exactly the eight undefined metrics per run as JSON `null`,
and completed the member manifests without reopening an official source or
changing any score, candidate, rule, or weight. Its audit SHA-256 is
`28f64727d2d6bb949fe3a0d0bd5b461ec14f9b0c51db7393c5ca7c7bfe864ce4`.

The independent completion audit rehashed all 730 manifest-bound outputs /
32,334,740,807 bytes, validated all 18 member contracts and access flags,
rechecked the 20 deployed checkpoints, found no temporary or mismatched file,
and passed with SHA-256
`52239ea7db0b0051cfdb8c25bb17ff1c64a89560f0a41ff6226dfd928ac618e7`.

After result commit `d09da270f26303ef16f3cc586b5cdbf0699186e5` reached
GitHub, the reviewed non-deployed-only cleanup removed exactly 84 checkpoint
files / 370,484,520 bytes from the eight Arm-2 realization jobs. It retained
all 298 prediction archives, all 18 references and analyses, and all 200
checkpoints / 901,942,200 bytes belonging to the ten deployed measured jobs,
including every one of the 20 required selected/final checkpoints. The exact
cleanup plan SHA-256 is
`f7ec92ed63c837b28539c75e48890ee77ec378dcba42de65db6fb0aef06ee19b`.
The postcheck rehashed all 646 retained manifest-bound outputs /
31,964,256,287 bytes and passed at SHA-256
`22e44bde683993387e60f7347bc16ea437fc8e426fb8f792be3e693eec0bf37a`.
Both artifacts are under `_cleanup/20260825T164320Z` in the program root.

The attached registration requested instance termination, but the user
explicitly superseded that operational step for this run. Exact paid GH200
instance `d0ebcd5f7dbb44dc99370080df7b47cc` remains active for the immediately
following experiment; it must not be left idle after that work.

## Verification V1 — Negócio a Negócio archive depth and participant-code coverage (frozen 2026-08-25)

Status at registration: no endpoint, UI, or archive probe has been made for
Verification V1. The unrestricted local Windows host is the execution
environment. This exact frozen specification has SHA-256
`3fff572e79969cc3d64c9156e6cd1d4f505e3b94e5ff8b612f9ec16b1273c1a3`.
It has no GPU, model, official-validation, or held-out-test implication and
does not modify the frozen Experiment-45 program.

### Frozen Verification V1 specification

#### Purpose

Decide, with body-verified evidence, whether B3's free daily trade-by-trade
file ("Negócio a Negócio", `{date}_NEGOCIOSAVISTA.txt`) is retrievable
historically — especially back to February 2023, when the
`CodigoParticipanteComprador`/`CodigoParticipanteVendedor` columns were added.
This single fact decides whether the D2 broker-flow sidecar is researchable
now (fold windows are covered from 2023-02 onward) or must wait on forward
capture. It also formally closes the MT5-L1 question (Appendix).

#### Ground rules

1. HTTP status is never evidence. A 200 with a zero-byte or stub body is EMPTY;
   a 403/404 from a proxied environment is BLOCKED, not ABSENT. Only a parsed
   body with plausible content proves existence; only the same negative result
   from an unrestricted machine and the official UI suggests absence.
2. Before any probe, fetch a known-good B3 market-data page and confirm direct
   egress with no proxy interception. If blocked, halt without running the
   ladder.
3. Use no more than one request per two seconds, a standard browser User-Agent,
   two retries on transient errors, approximately 40 requests maximum, and
   stop early once the depth boundary is established.

#### Outcome classes

- FULL: the ZIP contains the dated `NEGOCIOSAVISTA` text file with a plausible
  full market day (at least one million rows or at least 30 MB compressed), a
  parseable header, and both participant-code columns.
- PARTIAL: a full-day file without the participant columns.
- EMPTY: HTTP success with a zero-byte, stub, or malformed body.
- DENIED: 4xx/5xx from the unrestricted machine.
- BLOCKED: DNS, TLS, proxy-page, or other network-level failure.

#### Procedure

1. Probe `https://arquivos.b3.com.br/apinegocios/tickercsv/{YYYY-MM-DD}` for
   the two most recent completed B3 sessions. If both are DENIED, perform UI
   route discovery before concluding anything.
2. Probe in order 2026-01-15, 2025-06-02, 2024-08-01, 2024-01-15,
   2023-08-01, 2023-03-01, 2023-02-06, and negative control 2022-06-01.
   A holiday advances to the next session. Mixed FULL and DENIED results are
   bisected to the earliest retrievable date with at most 12 extra requests.
3. If endpoint probes fail, inspect `https://arquivos.b3.com.br/bdi`, record
   the Negócio a Negócio date-picker range, capture the exact working route,
   headers, and token flow for the newest and oldest UI-permitted dates, then
   rerun the ladder on that route.
4. For every FULL/PARTIAL file, record compressed/uncompressed size, row count,
   distinct instruments, the full header, and participant-column flag. Check
   PETR4 or VALE3 for positive price/quantity, session-hour trade times, and,
   during the code era, greater than 90% nonempty buyer and seller participant
   codes. Preserve ZIP bytes unmodified with SHA-256 manifests.

#### Predeclared conclusions and outputs

- A: FULL reaches March 2023 or earlier. Declare D2 RESEARCHABLE, record the
  earliest FULL date, project backfill size from mean compressed size times
  approximately 870 sessions, and produce only a polite rate-limited,
  resumable, manifest-hashed backfill plan.
- B: recent FULL but old DENIED/EMPTY, confirmed by the UI range. D2 is not
  researchable now and is shelved while capture accrues.
- C: nothing retrievable anywhere including the UI. Record all probes and
  distinguish ABSENT from BLOCKED.

For A or B, install a Windows scheduled daily capture job which downloads the
latest session each evening through the verified route, rejects empty output,
updates the SHA-256 manifest, and logs failures loudly. The final immutable
verification artifact records the probe table, working URL patterns, UI range,
earliest FULL date, negative control, conclusion letter, capture status, and
every retained SHA-256. Append a dated result summary here. No full backfill is
authorized.

The optional D1 closure may query the existing current-broker MT5 terminal
only, with unlimited max bars, for PETR4/VALE3/BOVA11 M1 from 2018-01-01 and
record the earliest returned bar. No account creation or further D1 work is
authorized.

### Verification V1 result (2026-08-25)

Direct egress was confirmed before the ladder: the known-good B3 market-data
page returned HTTP 200 directly from `104.18.43.121`, with 42,542 legitimate
HTML bytes and body SHA-256
`36520c545aae28d557b1497632e6971b91aeafa04d92735a54270984815e3ae5`.
The legacy `arquivos.b3.com.br/apinegocios/tickercsv/{date}` route returned
zero-byte HTTP 404 responses for both recent completed sessions, so it was not
treated as archive evidence.

Official `/bdi` UI inspection exposed the current no-token GET route
`https://drp.b3.com.br/rapinegocios/tickercsv/{date}`. The live date picker
spanned `2026-07-15` through `2026-08-25`. Seventeen rate-limited date probes
stayed inside the frozen 40-request budget. The historical ladder from
`2026-01-15` through the `2022-06-01` negative control returned HTTP 200 with
zero bytes, hence EMPTY rather than FULL. Boundary bisection established
`2026-07-29` as the earliest verified FULL date; `2026-07-27` was EMPTY and
`2026-07-28` was zero-byte HTTP 404 / DENIED. This is predeclared Outcome B:
the endpoint is a recent rolling window, not a deep historical archive. D2 is
therefore not researchable now and is shelved without forward capture under
the user's direct instruction.

Three downloaded ZIPs were preserved byte-for-byte under the immutable
`quant-data/b3/raw/b3_negocios/archives` root. Their date / compressed bytes /
row count / distinct-instrument count / participant-code coverage / SHA-256
records are:

- `2026-07-29` / `80,559,926` / `9,023,458` / `15,278` /
  `0.996820731` / `4b7c269ffc2b95eafa191ec57a083c2d7a73e330d683456d7667570e85a7aaef`;
- `2026-08-04` / `80,901,806` / `8,803,482` / `15,876` /
  `0.997777357` / `d13d329ae163e23712a73f2805ac998b505d94051af2e0807e4767a260878824`;
- `2026-08-24` / `90,482,379` / `9,948,893` / `17,468` /
  `0.997779853` / `e874f8c4fb5750060d497c00b374e6ca6ffa1b292585d1ab1da6e4a4a286adaf`.

All three archives had zero malformed rows, both participant columns, and
passing PETR4/VALE3 positive-price, positive-quantity, parseable-time, and
session-range checks. The raw archive manifest SHA-256 after adoption and a
no-download existing-archive validation was
`cb45a4cc8403e92fcae1b00dfe4b0e369b3d27fdb13a10719addb0fabbf0b520`.

No recurring capture task is installed. A scheduled task was initially
registered while following the attached specification, then immediately
unregistered before any scheduled execution when the user clarified that only
the history check was wanted. The optional current-XP-terminal D1 probe also
made no data read: the terminal already had `MaxBars=100000000`, but did not
expose a Python IPC connection (`-10004`, `No IPC connection`); it was closed
without changing the account or terminal setting.

The single immutable verification artifact is
`C:\Brazil-RV\transfer\verification_v1\verification_v1_report.json`, 7,419
bytes, SHA-256
`efbe1ab047fe7c2c38e5a035676131038a98af095ea07ee39399190274069d53`.
Verification V1 did not access model data, official validation, or the held-out
test, and it did not modify Experiment 45's frozen program.

## Experiment 46 — cross-equity given-graph structure (frozen 2026-08-25)

Status at registration: no Experiment-46 target or retained fold prediction
has been opened. The full frozen specification is
`research/preregistrations/experiment46_cross_equity.md`. This discovery-only
program builds one fixed monthly peer graph, runs the N0 decomposition and
rotated neutralization screen on retained store-v2 Fold C/A/B comparators,
screens exactly eight F-peer fields on dates through 2023-03-31, and spends
nine GPU trajectories only if at least two fields survive. Official validation
and held-out test remain sealed. Nothing in Experiment 46 changes the deployed
Experiment-45 store-v2 ten-seed baseline. The active GH200 is retained after
completion by explicit user instruction.

### Experiment 46 result (completed 2026-08-25)

Part 0 froze 54 monthly peer-graph snapshots from 2022-02-01 through
2026-07-01. The cluster sizes ranged from 3 to 51 and adjacent-month adjusted
Rand index averaged `0.5404`. PETR3/PETR4 were mutual top-one peers in all 54
snapshots. ELET3/ELET6 were not mutual top-one peers; this was a preregistered
diagnostic only and did not alter the graph. Monthly-graph and graph-audit
SHA-256 values are
`99286f3a1ec1df2955014f82ccdab6e29889d09ca0150c0dfebec767007cac40`
and `033dfcdaa7f41de3da93875c3ed9ab3cefaa6c312613f562fad4e32cb35cf341`.

N0 rejected every deployment neutralization transform. Its rotated held-out
mean paired delta was `-0.001826197`, with block-10 95% interval
`[-0.003539513,-0.000382781]`. Stability improved by `+0.010052757`, with
interval `[+0.007026385,+0.012897310]`, but the frozen joint IC guardrail
failed. The held-out rotations chose beta `lambda=0.5` for Folds A/B and
liquidity `lambda=0.25` for Fold C. The cluster cross-group component had
positive IC on all three folds, so T-peer's conditional registration did not
trigger. No deployment transform or future T-peer arm was registered. The N0
analysis SHA-256 is
`95627832e0c49df78975c001c5b8db2b995d0ff970998f32679b15669e09a5d9`.

F2 selected exactly `peer_relative_return_60m`,
`peer_relative_return_1d`, `peer_mean_return_60m`, and
`peer_dispersion_60m`, permitting the frozen F3 run. The F2 table and selected
sidecar-manifest SHA-256 values are
`e9c6c7e440a681497adae94d456cf1bc8674f3ea736dad714104b8169b719a7d`
and `127630d2df2f1eb125c5e638abfbcfb1521e02ca8f1a6c83f16ce931318ae077`.

All nine F3 trajectories completed. Standalone Raw Patience-3 deltas versus
the store-v2 parent were Fold C/A/B
`+0.000054345/+0.000347304/-0.000881236`, mean `-0.000159862`.
The pooled block-5 and block-10 95% intervals were respectively
`[-0.000709435,+0.000386293]` and
`[-0.000717352,+0.000396371]`. The predeclared parent-plus-candidate path had
Fold C/A/B deltas `+0.000032783/+0.000262657/-0.000319044`, mean
`-0.000007868`. Both frozen gates failed. F-peer earns no future official-read
arm and does not change deployment. Per the frozen contract, its nine final
EMA-0.995 members remain eligible for a future ensemble-pool registration;
they were not rescored as a pool here.

The first post-training analysis stopped before a Fold-A verdict because the
standing designated-challenger helper asserted that the supplied retention
parent must equal the challenger's legacy embedded parent. All nine prediction
trajectories were already complete and untouched. Operational repair commit
`f565933cb61eea789254eb900f3d46263ff8dcf3` separated that informational
challenger identity check from Experiment 46's frozen hash-verified store-v2
retention parent and added an analysis-only resume. Ruff and nine targeted
tests passed on Windows and Linux. No trajectory was retrained and no
prediction, comparator, replay, fold, interval, or gate changed.

The immutable program root is:

    /lambda/nfs/brazil-rv-east3/quant-data/b3/processed/model_runs/cross_equity_46_affea34_20260825T170000Z

The completed F3 summary and campaign-manifest SHA-256 values are
`4d9ac84ffad135038f0b57141df2672e347eb63d37d8cacb31251dabf862f455`
and `7354e7bffab6ae4e6cadbb698393ad3a239b9b36599930b2460027a16f1f44d0`.
The independent pre-cleanup audit hashed all 552 artifacts /
9,806,864,144 bytes, validated all nine 20-epoch histories, 180 prediction
archives, 180 checkpoints, frozen identities, exact gates, registrations, and
access flags, and passed with no errors. Inventory and completion-audit
SHA-256 values are
`cbf5696f3c2784fa8be032ea0fdec953e4855ec8de274a63d78d1a2b33b784ee`
and `2f82faef622882768af07aeec9e3e750b195c5f49c5929f9344f5da52ab2b74b`.
Official validation and the held-out test remained sealed throughout.

After result commit `e325be9` reached GitHub, the reviewed checkpoint cleanup
removed exactly 156 redundant epoch checkpoints / 705,303,300 bytes. It
retained all 180 prediction archives and the exact union of epoch 20,
whole-fold Raw Patience-3, and both honest odd/even cross-fit Raw Patience-3
epochs for every trajectory: 24 checkpoints / 108,508,200 bytes. Every run
manifest, history, analysis, graph/sidecar artifact, and final EMA-capable
epoch-20 state remains. Cleanup-plan, retained-inventory, and passing-postcheck
SHA-256 values are
`ada7872e3cf20a4729de37dc6ed03c660e20d800035b026e8fa800aeb1d0bc22`,
`1eddbf56876fa7d9746d54268bc4d24c656766f409133d917cb3327a532adcc2`,
and `68ab557488c49895821c62a38045581c5daa4d696b579e64848e30dc42a97427`.
The immutable evidence is under `_cleanup/20260825T194800Z` in the program
root.

Per the user's explicit override, paid GH200 instance
`d0ebcd5f7dbb44dc99370080df7b47cc` remains active for the immediately
following experiment and must not be terminated at Experiment-46 closure.

## Experiment 47 — final bounded HPO + structural sweep (frozen 2026-08-25)

Status at registration: no Experiment-47 trajectory, fold prediction, or
score exists. The full frozen specification and pre-score implementation
resolutions are in
`research/preregistrations/experiment47_hpo_sweep.md`. This discovery-only
program runs exactly 16 preregistered cells on seed 29 and Folds B/C, then
conditionally confirms at most three qualifiers on seeds 11/29/47 and Folds
C/A/B. Stage 2 compares only with the immutable Experiment-41 Stage-C
store-v2 three-seed comparator. Official validation and the held-out test stay
sealed; no deployment change or ensemble-pool scoring is permitted.

The attached contract closes HPO and architecture axes permanently for this
generation after completion. It also ends the standing keep-alive override:
after results, artifact audit, reviewed checkpoint cleanup, and result push,
exact paid instance `d0ebcd5f7dbb44dc99370080df7b47cc` must be terminated and
verified absent twice.

### Experiment 47 result (completed 2026-08-26)

All 59 frozen discovery trajectories completed: 32 Stage-1 trajectories and
the maximum 27 Stage-2 trajectories. Stage 1 advanced exactly `R7` (VAL),
`R1` (SIMP), and `P2` (SIMP). Their two-fold mean primary deltas were
`+0.000159592`, `+0.000390868`, and `-0.000229203`, respectively.

In Stage 2, R7 produced Fold A/B/C deltas
`+0.000434758/+0.000918349/+0.000245795`, mean `+0.000532967`.
Both pooled supporting intervals had positive lower bounds, but the frozen VAL
gate still failed because the mean was below `+0.001`. R1 produced Fold A/B/C
deltas `-0.000442724/+0.001436183/+0.000737590`, mean `+0.000577016`,
and passed the SIMP gate. P2 produced
`-0.000922042/-0.000197612/-0.000425118`, mean `-0.000514924`, and
failed. Therefore R1 is the frozen training-recipe specification for future
work. No future official-read arm was registered, the ensemble pool was not
scored, and deployment did not change. HPO and architecture axes are closed
permanently for this generation.

The immutable program root is:

    /lambda/nfs/brazil-rv-east3/quant-data/b3/processed/model_runs/hpo_sweep_47_b637561_20260825T202142Z

Frozen-design, Stage-1 screen, Stage-2 confirmation, result, and completed
program-manifest SHA-256 values are
`4ba6919bbe2a27c80e9e3c9d60d9ec73f1bbbc0e005c5ef4ab84a8490b5adc99`,
`25f19b88a92f7bd922f9011bcb615bc3c4e242578a5be6faf8de3394d0741787`,
`5d36b859577f58cd7f4b49edbd3996fe5c0c2e59e4bf6e3f048462babb76fafe`,
`464c2a213a0953f1ec68eb2869825a9418a3458aaa7dd3262bf197b18a844b1a`,
and `55818633a4559bc5b4584f73dc06deee043be857c7bd559d27608616e63e7a5f`.

The final audit validated all 59 run manifests, 1,180 history epochs, 1,180
prediction archives, 1,180 pre-cleanup checkpoints, 78 paired-analysis JSONs,
11 frozen source hashes, all screen/confirmation/result gates, and 288 explicit
access-ledger flags. It hashed all 2,973 pre-cleanup artifacts / 60,105,765,909
bytes. Official validation and the held-out test remained sealed throughout.
Pre-cleanup inventory and final-audit SHA-256 values are
`50d06b52d874fa9173be37fe2ccbe696a52542261d60e91169b71c6eea59ac6a`
and `168fc4c7870826a6044de69b841544341d4022b275fff0d45de4d3e8187067ea`.

The reviewed inventory-bound cleanup removed exactly 1,027 redundant epoch
checkpoints / 4,603,571,101 bytes. It retained every prediction, manifest,
history, readout, analysis, and the exact union of epoch 20 and candidate/control
honest odd/even cross-fit selected checkpoints: 153 checkpoints. Cleanup-plan
and post-cleanup inventory SHA-256 values are
`e43fa2bb7f826b4bcb1e5dac90b97e0714638b24167039e13301ff1f6c2a56bb`
and `94187fa46da00001ee72779e2a931c4586b72ab7d97dbeba85cd48c43138087d`.
The immutable audit evidence is under `_audit/20260826T070331Z` in the program
root.

Result-log commit `6369688` reached GitHub before shutdown. The exact paid
GH200 instance `d0ebcd5f7dbb44dc99370080df7b47cc` was then terminated. Provider
inventory confirmed the ID absent independently at `2026-08-26T07:08:37Z`
and again at `2026-08-26T07:08:47Z`; no paid instance was left idle.

## Experiment 48 — next-generation recipe and 15-minute head (frozen 2026-08-26)

Status at registration: no Experiment-48 target, trajectory, fold prediction,
or score exists. The full frozen specification is
`research/preregistrations/experiment48_nextgen.md`. This discovery-only
program first decomposes the retained Experiment-41 comparator's 30-minute
signal into exact independently ranked 0→15 and 15→30 minute legs. It then
runs exactly nine R1-depth-4, temperature-1.00 trajectories on seeds 11/29/47
and Folds C/A/B. Only if the frozen leg gate passes does it run nine further
four-head trajectories with a hashed development-only 15-minute target
sidecar. The incumbent three-horizon metric remains the sole adoption gate;
the new 15-minute IC is measured separately.

The exact canonical store, Experiment-41 cross-fit archives, Experiment-47 R1
archives, and causal target-scale source are hash-bound in the runtime frozen
design before any Experiment-48 score. At most two training processes are
permitted. No pool scoring, new specialist, weighting search, official
validation, held-out test, deployment change, or read registration is
authorized. Exact paid GH200 instance
`f766c33a775344d394ec0bdc915fca6d` must be terminated only after the complete
artifact audit, reviewed checkpoint cleanup, result log, commit, and push,
then verified absent twice.

### Experiment 48 result (completed 2026-08-26)

Part A replicated the early-realization claim on all three discovery folds.
Fold C/A/B leg-1 versus leg-2 IC was
`0.039540720/0.021964317`, `0.048528015/0.028513916`, and
`0.049717261/0.031518314`; the paired differences were
`+0.017576404/+0.020014099/+0.018198947`. Every block-10 lower 95% bound
was positive (`+0.013223676/+0.016300907/+0.010434877`), so the frozen
Part-C gate passed 3/3 folds.

Part B combined R1 depth 4 with temperature 1.00. Its Fold A/B/C deltas versus
the archived Experiment-47 R1 trajectories were
`+0.000198613/+0.000944777/+0.000125564`, mean `+0.000422984`.
The pooled block-5 and block-10 lower 95% bounds were both positive
(`+0.000104390` and `+0.000109832`). It passed the frozen non-inferiority
gate, making `R1_T1.0` the next-generation three-head training recipe.

Part C added the fixed equal-weight 15-minute head. Its three-horizon Fold
A/B/C deltas were `+0.000819706/-0.000864794/-0.000089351`, mean
`-0.000044813`. Fold B crossed the fixed `-0.0005` floor, so both the primary
and superiority gates failed. The final next-generation specification is
therefore `R1_T1.0_three_head_30_60_120`; the dedicated 15-minute model remains
a registered future option but was not built. The new 15-minute head itself
measured IC `0.054073317/0.051798396/0.038488832` on Fold A/B/C, versus its
30-minute head's `0.046053040/0.054633680/0.038939226`.

No pool was scored, no official-validation or held-out-test data was accessed,
no read was registered, and deployment remained the Experiment-45 ten-seed
store-v2 recipe. The immutable program root is:

    /lambda/nfs/brazil-rv-east3/quant-data/b3/processed/model_runs/nextgen_48_e5c1983_20260826T132551Z

Frozen-design, completed program-manifest, result, Part-A, Part-B, Part-C, and
15-minute target-sidecar manifest SHA-256 values are
`def9ff3229e66825df4ef6856130edd2a12f599ed299ef30a9aa5b93de0192bd`,
`33d5f400d4b775f842bd8237f91ba01bb9570ff39095657ea03603d30239002f`,
`7d91e63e50c164ede14a75ef02097563a7d1bb58540d6a8bc57a709e43869a0d`,
`ce0479466decd19e6c345f87c72c913005dbd61c3df3a5b5ed8df78662317855`,
`04b975228e0a8656e4a005f49d725ce7c3b30d072d8ca28030d6f9c1f1a13c19`,
`a7aab4061e43c7f7adb9ee9627e455b96073769056599a2deb8527233c5880c8`,
and `a4cad0382fca44e5b07783f03cefafd2e9378ffd487530655388142002202bb7`.

The final audit verified all 18 run manifests, 360 history epochs, 360
both-state epoch prediction archives, 360 pre-cleanup checkpoints, 290 frozen
source hashes, every decision gate, and 124 explicit access flags. It hashed
873 pre-cleanup artifacts / 21,033,562,827 bytes. The reviewed inventory-bound
cleanup removed exactly 313 redundant checkpoints / 1,235,370,343 bytes and
retained every prediction, history, manifest, reference, analysis, plus the
exact union of epoch 20 and honest odd/even-selected checkpoints: 47
checkpoints. Pre-inventory, cleanup-plan, post-inventory, and final-audit
SHA-256 values are
`0272e2ec37361bdd48b5aafac89429f0a677a1d2eac905fdfbf9234034fdb1ec`,
`8134ae5ff852d3e2e7a13736300ff490f3017fd3b73233af98bae502ad8b6ab3`,
`9daf9fa584f05386287a408204f213008277bb28e4af1c3502d8db4ea6d16a87`,
and `4b43baa7b1760ca22cc10c5f48d084e7d4976154d155d6a714d1539915db38d3`.
The passing audit evidence is under `_audit/20260826T162730Z` in the program
root. Every access record remains `official_validation_accessed=false` and
`test_accessed=false`.

Result/context commit `8fc7cde` reached GitHub before shutdown. The exact paid
GH200 instance `f766c33a775344d394ec0bdc915fca6d` was then terminated after
its identity, attached filesystem, and termination action were reverified.
Provider inventory confirmed the exact ID absent independently at
`2026-08-26T16:33:18Z` and again at `2026-08-26T16:33:28Z`; the account had
zero active instances and no paid host was left idle.

## Experiments 49 and 50 — economics robustness and read event 5 (frozen 2026-08-26)

Status at registration: no Experiment-49 derived number exists and no
Experiment-50 trajectory or official prediction has been accessed. The complete
frozen contracts and pre-score implementation resolutions are in
`research/preregistrations/experiment49_economics.md` and
`research/preregistrations/experiment50_nextgen_read.md`.

Experiment 49 is a zero-GPU audit of the archived Experiment-48 four-head and
store-v2 comparator discovery predictions. It measures quarterly Roll spreads,
recomputes IC under exactly open-to-open and adjacent-close-midpoint labels,
localizes IC by spread tercile, and runs the fixed top-80 executable 15m/30m
books. Its three-part KEEP/DROP rule fixes Experiment 50's head count before any
official result exists. Price-only Roll estimates may extend through 2025-06-30;
official model predictions and the held-out test remain sealed.

Experiment 50 is official-validation access event 5. It contains exactly one
ten-seed R1-depth-4, temperature-1.00 next-generation arm and the immutable
deployed Experiment-45 ten-seed comparator. Deployment requires the paired
block-10 lower 95% bound to be at least `-0.0005`, inclusive. The optional 15m
official score is decision-neutral. No second read, pool, alternate arm,
weighting search, or test access is allowed. After complete artifact audit,
reviewed checkpoint cleanup, result-log commit, and push, the exact paid instance
must be terminated and verified absent twice.

### Experiment 49 result (completed 2026-08-27)

The zero-GPU economics audit completed against all 18 frozen discovery
prediction runs. The 15-minute signal passed the bounce-robustness condition
on all three folds, but its executable net return was negative on every fold
and the combined 15m/30m book improved net Sharpe on zero folds. The frozen
verdict is therefore `DROP`, fixing Experiment 50 to the three-head
30/60/120-minute realization. The dedicated 15-minute option remains
registered but unbuilt. Official validation and the held-out test remained
sealed throughout Experiment 49.

The immutable Experiment-49 program root is:

    /lambda/nfs/brazil-rv-east3/quant-data/b3/processed/model_runs/economics_49_77411e4_20260827T004334Z

Frozen-design, result, verdict, completed-manifest, inventory, and final-audit
SHA-256 values are
`ad4ee72cdc59dd7b7ea7d89542d7af6c81d89d40f42d258946d92024881e82c3`,
`d954762eaf578b403531d1d9ee84f0e42b63b994cc8ff8f9324603c158de6e00`,
`d7603cf1f0ea5a05ef951bfd3ceb15334cb7e79d7d5780984ef6da14959e5221`,
`1f548748add7cfef44870d52fc75485211d9e7bffca6562560f6148192f2795e`,
`83f465d7fc7e861ee49712503611c5f12f2f5d0a07de568b659d44923c0dece7`,
and `9c0a3e2e722598e54a7fdd72b40f49c5691a49da9db1a6b044be0afb921613ec`.
The passing audit covers 18 source prediction runs, 63 retention rows, 2,528
spread-schedule rows, 189 spread-tercile rows, nine portfolio summaries, and
19 inventoried files / 345,089,669 bytes. Pre-score operational repair commit
`0922a1f` only left post-schedule dates unmaterialized and resumed the exact
interrupted analysis; it did not alter frozen measurements or gates.

### Experiment 50 result (completed 2026-08-27)

Official-validation access event 5 ran exactly the ten frozen three-head
next-generation seeds. The candidate uniform ensemble scored
`0.043588809`; the deployed Experiment-45 store-v2 comparator scored
`0.043718770`. The candidate-minus-comparator paired block-10 lower 95% bound
was `-0.001054764`, below the inclusive `-0.0005` non-inferiority floor.
The frozen decision is `RETAIN_EXPERIMENT45`: the next-generation recipe is
not deployed, no alternative or hybrid was evaluated, and the standing
deployment remains the Experiment-45 store-v2 ten-seed ensemble.

The immutable Experiment-50 program root is:

    /lambda/nfs/brazil-rv-east3/quant-data/b3/processed/model_runs/nextgen_read_50_77411e4_20260827T004334Z

Conditional-design, result, analysis, deployment-decision, deployed-recipe,
completed-manifest, access-ledger, final-inventory, and final-audit SHA-256
values are
`27dd2a7e8cebbdf40d94308bb6e7251e49fc837d9a67e90413bffd4c455f3b64`,
`e21571de6bebd04ec8e168036597e077ecc7c4f512002c27cc55328c8a843c28`,
`b722fc42cd50b063ac7ed6dd257697b4f011162ef7fa2adaedf629dc16e5b4b7`,
`8189066f5dc6fdf3d7489c20853cf0404b095f1241c5505aea34ec0c041aa3f4`,
`1555d37882296f6d08e7bb67904bc8f8a2b5c3b64f2b5e5b1bf248f2e33f3ee8`,
`a7c77d421dbc1eec84c98335241019146210d1d800f6a7087b9ff7ac2c379b9e`,
`1ceb731411acdca2f2cbfd0e06a420c0eccfaf009c1899f4f4d79dd246f2037a`,
`c4b837c273c94abab5448f01686f99744b29ea6b986f19bcc9dd6d06dd779d4b`,
and `aca83d6d84c458b7e1db19645bc411942f6e8ef8530e86bda040a62ca2c116fa`.

The final audit verified all ten completed manifests, 200 history epochs, 210
official prediction archives, the exact comparator sources, all analyses and
decision hashes, and `test_accessed=false`. It inventoried 304 files /
22,028,877,436 bytes. Reviewed cleanup removed 180 redundant candidate
checkpoints / 710,259,660 bytes and retained every prediction, history,
manifest, analysis, plus exactly 20 candidate selected/final checkpoints.
Post-score audit repair commit `1efe9c5` corrected only the terminal location
of the immutable sealed-test flag (`run_provenance.test_accessed`); it did not
rerun training, access predictions again, recompute scores, or change the
deployment decision. Its provenance SHA-256 is
`c839c226715bb736f5b71d81444fae4cc17c64aa367b3925e03fceae34d22523`.
The event-5 ledger records `official_validation_accessed=true`; every final
artifact records `test_accessed=false`, so the held-out test remains sealed.

Result/context commit `1683dfb` reached GitHub before shutdown. The exact paid
GH200 instance `2407fd931c3f47b7825bf6538617571d` was then terminated only
after the immutable artifacts, audit, cleanup evidence, checkpoints, logs, and
result push were secured. Provider inventory confirmed the exact ID absent at
`2026-08-27T03:54:38Z` and again at `2026-08-27T03:54:57Z`; the account had
zero active instances and no paid host was left idle.
## Experiment 51 — first and only held-out test read (frozen 2026-08-27)

Status at registration: the held-out test has never been accessed. The full
immutable specification is
`research/preregistrations/experiment51_test_read.md`; this registration and
the implementation are committed before any test-period target or prediction
is opened. Experiment 51 authorizes the first and only test-access event. On
completion the test is spent forever; hypotheses from this read require new
forward data.

The only measured object is the exact Experiment-45 deployed store-v2 recipe:
ten retained Raw Patience-3 selected checkpoints at seeds
11/29/47/61/79/97/113/131/149/167, the 34-field input, and uniform
within-sample/horizon tie-aware rank averaging. The deployed-recipe file is
bound to SHA-256
`d4729dc4e614e0edd5118ba5ed5b7bc92f69ca2faceab4a09d0559115e5c4058`;
all member manifests and the 20-file deployed checkpoint inventory must be
rehash-verified before test access. There is no comparator, training,
selection, new weighting, or deployment change.

The only analyses are the preregistered IC, per-horizon IC, staleness,
period-difficulty, time-of-day, monthly, member, and ensemble diagnostics. The
daily bootstrap uses exactly 10,000 moving-block replications at block lengths
5 and 10 with base seed 20260827. The paired staleness statistic is frozen as
H2 minus H1: test dates 1–129 are paired by ordinal position with dates
130–258; date 259 remains in all full-period statistics and is excluded only
from this equal-length paired diagnostic. A negative block-10 interval that
excludes zero therefore indicates retraining before live use.

Quarterly difficulty uses exactly three label-return summaries, computed the
same way for test and the already-retained official-validation reference:
mean population cross-sectional standard deviation over valid
sample-horizon cells; median sample standard deviation over valid
equity-horizon groups; and mean valid-equity count over sample-horizon cells.
No execution statistic—including Sharpe, costs, turnover, or a portfolio
book—is computed. No analysis may be added after a test number is visible.

### Experiment 51 result (completed 2026-08-27)

The first and only held-out test event completed at exact implementation commit
`6bca6d3e4fa9cbd1123156210561bf9d024438ec`. Before access, frozen-design
SHA-256
`0bdf9e1efcf2b4d20cbc952b1d1335d978150faaacc89166eeb832ad865bd1ea`
rehash-verified the Experiment-45 deployed recipe, all ten source manifests and
selected checkpoints, all 20 retained selected/final checkpoints, and the
store-v2 identity. The immutable read root is:

    /lambda/nfs/brazil-rv-east3/quant-data/b3/processed/model_runs/test_read_51_67f43b3_20260827T130349Z

The deployed ensemble scored test IC `0.040345936073`, Band A. Block-5 and
block-10 95% intervals were respectively
`[0.033209566211, 0.047739292104]` and
`[0.033313199285, 0.047668906133]`. Per-horizon test ICs were
`0.040721247903` at 30 minutes, `0.039138347910` at 60 minutes, and
`0.041178212408` at 120 minutes.

Quarterly IC was `0.044357109` in 2025Q3, `0.029403024` in 2025Q4,
`0.037295333` in 2026Q1, `0.045224692` in 2026Q2, and `0.064826747` in the
partial 2026Q3. The daily-IC slope versus days since the 2024-06-28 training
end was `+0.000026511743` per day. The frozen paired H2-minus-H1 difference
was `+0.012885851390`; its block-10 interval
`[-0.007851288044, +0.023472244463]` included zero. The preregistered
staleness rule therefore does not indicate deterioration or a
retrain-before-live policy. Difficulty, TOD, monthly, ten member-IC, 45 member-
correlation, and ensemble-gain tables are retained exactly as preregistered.
No execution metric was computed and no model, recipe, weighting, or deployment
state changed.

Two bounded pre-access operational issues were score-neutral. The first SSH
freeze command stopped before Python because `uv` was not on the non-interactive
shell path; no output root or ledger existed. The source verifier then correctly
stopped before root creation because the historical Experiment-45 inventory
uses field `bytes`, not `size_bytes`. Repair commit `6bca6d3` bound that exact
historical schema and added a targeted test. Both issues occurred before any
test access. After scores, only existing logs, the independent completion
audit, and terminal hash inventories were added; no score, analysis, store
read, or prediction changed.

The terminal audit passed over 27 artifacts / 358,294,145 bytes, including all
ten member prediction archives, the ensemble archive, test reference,
predeclared analyses, preregistration/interpretation text, and launcher,
bootstrap, freeze, and run logs. Key SHA-256 values are:

- program manifest: `19b87ed6d23a9d38b9defaa923a91e280e7af47d684625ae6c36cbd1a99c3cbf`;
- artifact inventory: `04a9c367089c6c1c5146cf8464521dc43200f4b26d07220ff3cabe5ef8d61f33`;
- result: `23fa452636245170659a7889181f1285c3779d47cb3644236f88113359d88fa8`;
- analysis: `013a490ae8c4a0b860edb1db4b84f4c47322785a9ae26726472bbb1f34a63e16`;
- test-access ledger: `5ad518caf94f02c4b8b9b02ed5f6afd4a78110002819ca942ab9bdb9d41bf517`;
- completion audit: `99c06a406fd6cc3eed7f1d6adb76defbb5047f0bad1887b41f2c7a83c36626a8`.

The ledger records `test_accessed=true` and `test_spent_forever=true`. The
current evaluator rejects any further test request. The held-out period is now
permanently spent; future hypotheses require new forward data.

Result/context commit `23a100750397e903aa8ef394af2c12b0906e30fd`
reached GitHub before shutdown. Exact paid GH200 instance
`a6c710df4bfa48c9a40f64ee4b7e85c4` was terminated only after every retained
artifact, log, audit hash, result, and the spent-test policy were secured and
pushed. Provider inventory confirmed that exact ID absent at
`2026-08-27T13:17:35.6228739Z` and again at
`2026-08-27T13:17:57.0202321Z`; both reads showed zero active instances.

## Storage maintenance — Lambda object-store cleanup round 4 (2026-08-27)

The cleanup used the filesystem's direct S3-compatible endpoint, so the GH200
capacity watcher was stopped before launch and no paid instance was created.
The reviewed plan was generated at repository commit `0851256`, uploaded before
deletion, and dry-run against every exact key. Its only deletion scope was
unselected per-epoch validation predictions and tail-candidate bundles in the
closed Experiment-42, 43, 44, 45, 46, 47, 48, and 50 program roots.

The plan removed exactly 3,521 objects / 183,169,695,704 bytes. Per root, the
removed object/byte counts were:

- Experiment 42: 494 / 21,227,317,082;
- Experiment 43: 57 / 5,801,362,338;
- Experiment 44: 1,008 / 47,013,357,888;
- Experiment 45: 190 / 19,337,874,460;
- Experiment 46: 165 / 7,090,371,894;
- Experiment 47: 1,086 / 46,790,695,194;
- Experiment 48: 331 / 16,570,842,388;
- Experiment 50: 190 / 19,337,874,460.

The retained-state counts for those same roots were 73, 6, 90, 20, 24, 153,
47, and 20, totaling 433 selected/final or honest-selection prediction
objects. Every matching retained checkpoint, manifest, history, observation
reference, analysis, decision, access ledger, and cleanup authority remains.
Experiment 44's 90 hash-frozen member states remain intact. Experiment 45
retains the exact deployed ten-member selected/final prediction and checkpoint
inventory; its non-deployed arm archives were not part of this deletion. The
complete Experiment-51 test event was explicitly excluded.

The postcheck rebuilt the selection-derived keep sets from each prior cleanup
authority, found every required retained object, and found zero remaining keys
eligible under the round-4 plan. Raw/interim, canonical feature-store, external
sidecar, auxiliary-target, and Experiment-51 sizes remained at
18,224,073,078 / 287,164,841 / 9,890,909,028 / 1,952,163,984 / 517,990,485 /
358,303,408 bytes. The final bucket, including the plan and result evidence,
contains 16,399 objects / 103,355,305,165 bytes.

The immutable record is under
`model_runs/_retention/storage_cleanup_20260827_round4`. Plan, delete-list, and
result SHA-256 values are
`1bd8bb8f99f6821066ad5c03f51ca420d40c9960ee6b2fa3bbb881d3d0d9e7b1`,
`e262d1fc2430e2d9d0e031a54e21fe40ae2c6ae89be2f5185070db15c8f01f02`,
and `0b59fa9611a5e098cbd337c63e38ee8cfef941e415dc7f0f258092dca7643eba`.
The removed unselected archives are not recoverable in place; reproduction uses
the recorded commits, manifests, histories, retained states, and canonical
derived data. Provider inventory at `2026-08-27T14:31:41.9870434Z` contained
zero active instances.

## Execution layer implementation (2026-08-27)

The offline execution-backtest layer was implemented from repository base
commit `2f82d3a8f7b6b9b59a99926a5e04d2f5f981ddb2`. Implementation commit
`0d933ef83ef31348eac001d20b8ba4151cbbe2ec` contains the execution package,
tests, module README, and durable project-context contract. It adds an explicit
score-refresh interface that accepts the current `15,20,...,285` session-minute
indices without deriving cadence from decision ordinals, and accepts a future
dense one-minute refresh grid through the same contract. The layer contains
causal raw-score ranking, discovery-only archive verification, identity-safe M1
streaming, strictly lagged liquidity and Roll inputs, exact differentiable
neutral/gross/cap projection, a deterministic no-trade-band policy,
participation-capped next-open replay, close taper and counted terminal fills,
CDI/cost accounting, and hash-bound JSON reporting.

The implementation deliberately excludes live/broker interfaces, policy
training, OOF refits, split generation, tuning, neural scaffolding, cluster
penalties, impact, and round lots. These are not needed to answer the current
offline replay question. Aggregate OOF archives are also rejected until a
canonical materializer can prove constituent-fold exclusion per sample. No
experiment or training run was performed. No
production feature store, spread schedule, CDI series, cluster sidecar,
prediction archive, official-validation data, or held-out-test data was opened;
all implementation tests used synthetic temporary fixtures. Consequently there
are no production input hashes to record for this entry. A real execution run
must hash-record its store, source and execution-wrapper manifests, discovery
prediction/reference, lagged spread schedule, and explicit dated CDI series
before replay.

The focused execution suite passed 37 tests, including accounting and golden
cost arithmetic, causal future-mutation guards, cap/carry/taper/forced-close
behavior, exact projection and gradcheck, masked-name gradient isolation, CDI,
band-policy behavior, archive/hash/access rejection, and end-to-end policy
gradient flow. An additional 21 Experiment-49/50 and modeling data/training
regression tests passed. Ruff check, Ruff format check, and Python compileall
also passed on the Windows development host. A float32 synthetic CPU replay of
250 days × 405 session minutes × 150 names completed in 1.496 seconds and
finished flat without forced fills on that fixture; this is a smoke measurement,
not a production-data runtime guarantee.

## Experiment 52 — C0 band-policy baseline (preregistered 2026-08-27)

Status at registration: no Experiment-52 market cache, execution report,
cell-fold readout, frictionless decomposition, rotation table, C0 designation,
or net-economics number exists. The official-validation and held-out-test
prediction sets are outside the input contract; the test remains permanently
spent after Experiment 51.

Experiment 52 is the first end-to-end measurement under the offline execution
layer. It consumes only the completed Experiment-41 Stage-C store-v2 prune-R2
seeds 11/29/47 on discovery Folds C/A/B. Each seed uses its already-frozen
bidirectional opposite-parity Raw Patience-3 states. Per-seed raw predictions
are tie-aware ranked over the causal store membership-and-readiness mask and
uniformly averaged, then each fold is bound through the execution discovery
manifest guard. No label mask participates in execution ranking.

The real M1 bridge, causal liquidity with a 20-session lookback, one-completed-
quarter-lagged Experiment-49 Roll schedule, prior-session Roll fallback, and a
hash-pinned BCB SGS series-12 CDI artifact are the only market inputs. Sigma is
the stored causal `vol_regime` field converted to dimensionless daily-return
units as `PRICE_VOL_REFERENCE * sqrt(405) * exp(vol_regime)`; the canonical
stored clipping is retained.

The grid contains exactly 12 cells: bands `0.0/0.5/1.0/2.0` crossed with equal,
30-minute-only, and `0.5/0.3/0.2` horizon blends. Every other execution setting
remains at its default. Every measured cell-fold run receives one frictionless
counterpart with zero fees and zero full spreads. Daily standard deviation uses
`ddof=1`. Annualized net Sharpe is `sqrt(252) * mean/std`, and daily cost drag is
spread plus fees divided by date count and initial NAV, expressed in bps.

For each held-out fold, cell ranking uses mean annualized net Sharpe on the other
two folds. C0 is the cell with the most rotation wins; a tie uses higher mean
held-out Sharpe across all three folds. An exact residual tie fails closed.
This is a reference designation with no promotion gate. All 36 measured results
must be reported even if all are net-negative.

The canonical preregistration is
`research/preregistrations/experiment52_c0_baseline.md`. The implementation is
`brazil_rv.execution.experiment52`; its targeted preflight passed 42 tests plus
Ruff and compile checks. CPU execution only is permitted. No neural policy,
trainer, OOF materializer, added cell, post-score parameter change, official or
test prediction read, live interface, or prediction deployment change is in
scope.

Pre-score operational repair (2026-08-27): the first production replay stopped
before writing any measured or frictionless report because a legitimately
unobserved M1 open interrupted an existing position. The failed immutable root
is `execution_c0_52_cbc9bd5_20260827T162810Z`; it contains only the frozen
design, wrapped discovery predictions, causal market inputs, CDI artifact, and
the zero-score failure log. Official-validation and test access both remained
false. Repair commit `b466cd639fbc1c5bc7799865c8ae7205cc8ced3e` makes the existing next-open
contract usable on sparse real M1 paths without creating a bar or a fill: an
unobserved minute carries the last observed position notional, permits no fill,
and realizes the cumulative return only when the next observed open arrives.
Terminal liquidation still requires an observed final open. This is causal,
changes no frozen cell, source, parameter, readout, or designation rule, and
passed all 42 focused execution/Experiment-52 tests plus Ruff and compile
checks before any score existed.

The refrozen replay then reached the registered end-of-session flattening step
without any report having been written and found that some held names lacked an
open in the exact final M1 slot. Operational repair commit
`a30685620e92031db05b5493847f59b5c928aada` applies the predeclared forced-close
rule at each name's last observed session open and spread when the final slot is
unobserved. It still applies the frozen 2x spread multiplier, ignores the
ordinary participation cap only for that counted terminal fill, never creates
an observed bar, and fails if a held name has no prior observed priced open.
The registered 30-minute taper, 12-cell grid, inputs, metrics, and C0 rule are
unchanged. The focused suite passed 43 tests plus Ruff and compile checks before
any measured or frictionless report existed; official-validation and test
access remained false.

After 44 reports had been written, the replay reached one transient
rank-validity cross-section whose signed cap capacity could not support the
exact neutral gross target. The hard projection correctly failed rather than
relaxing gross or caps. Post-score operational repair commit
`e36b8bc9fdd95072d0b30ccd45159fbad217015c` retains the last feasible projected
target only for such an infeasible refresh and adds hash-verified report resume;
the hard constraints, target grid, and every completed number are unchanged.
Before resume, all 44 JSON reports and 44 checksum sidecars were inventoried at
SHA-256 `b67dce2da2823733ee0d0836c292a31954d8d51ac69774066a243571be520df5`.
The resumed program reloaded those reports only after verifying their config,
input map, payload, and sidecar hashes. The focused suite passed 45 tests plus
Ruff, format, and compile checks. The final manifest records the exact
operational repair commit; official-validation and test access remained false.

Experiment 52 completed with all 36 measured and 36 frictionless cell-fold
reports. The frozen rotation rule designated `band_2p0__blend_equal` as C0:
band `2.0` with the equal `(1/3,1/3,1/3)` horizon blend. It won all three
two-fold rotations. Its measured annualized net Sharpes on Folds C/A/B were
`-21.085379`, `-12.272587`, and `-14.234936`. All 36 measured cells were
net-negative; the least-negative cell-fold net PnL was `-R$4,218,309.36` and
the most negative was `-R$18,474,688.45`.

Across the three C0 fold windows, gross PnL was `R$2,908,868.36`, while spread
cost was `R$12,816,563.05`, fees were `R$5,306,864.17`, CDI was
`R$304,193.79`, and measured net PnL was `-R$14,910,365.07` on
`R$26,534,320,864.54` turnover with 16,052 counted forced fills. The exact
frictionless counterpart was positive by `R$3,372,629.98` in aggregate; its
fold Sharpes were `0.157567`, `4.863886`, and `4.965943` on C/A/B. This is the
registered execution reference, not a deployment or prediction-model change.
The result plainly attributes the sign reversal to the measured execution-cost
and turnover structure.

The immutable completed root is:

    /lambda/nfs/brazil-rv-east3/quant-data/b3/processed/model_runs/execution_c0_52_e380cd7_20260827T164738Z

Frozen-design, program-manifest, market-input-manifest, cell-summary, rotation,
C0-designation, and final-audit SHA-256 values are respectively
`5906da60ec6d1179f20ed368f0e519973c173fc968ee7467936d08e8c1687f41`,
`cc7d84a0ff52420221f1eb13f574f4857109cb08396e5b28cfb4f547771cd21e`,
`72acb435973015199eb7fb3b85d41afc75ebc117457fc231a817637e5df14d77`,
`a05f62d33379616fba1ddcae8aa3020b52a1891023ea5f3ea8911c881d016e6d`,
`124e619871f848ff582b238a963740cd6058d25ba1a63cff9c94beba7dfa54da`,
`d18b97df11891b95e5e6b2ae226155e5569bdd6341d618c7bd624655b8e4de5f`,
and `b7591bb6464da90aa3324dced14ebf04e2e29245043b2719b76fb75bbdbd061f`.
The final audit passed over 182 files / 405,197,467 bytes, including every
prediction wrapper, causal market/CDI artifact, report and checksum, repair
inventory, and operational log. C0 does not access official-validation or test
predictions; both flags are false.

Result/context commit `e4c7736583ce6390eed37b1c953eaf17f52c990e` reached
GitHub before shutdown. Exact paid GH200 instance
`356e43e74aa14abd84ad5bca30f70212` was then terminated only after the final
audit and retained root were secured. Provider inventory confirmed that exact
ID absent at `2026-08-27T17:10:06.4015869Z` and again at
`2026-08-27T17:10:24.4091678Z`; the Lambda account contained zero active
instances on both reads.

## Experiment 53 — feasible-region map (preregistered 2026-08-27)

Status at registration: no Experiment-53 frozen root, backtest report, cell-
fold-variant readout, liquidity decomposition, per-trade table, rotation result,
or C1 designation exists. No paid instance is active. Experiment 53 is CPU-only
and reuses the exact hash-audited Experiment-52 discovery prediction and market
inputs; official-validation and held-out-test predictions are outside the input
contract.

The frozen grid contains exactly 48 concentrated equal-blend cells: K
`10/20/40`, base band `0.5/1.5`, cost scale `0/1`, gross `1.0/2.0`, and full or
top-half prior-ADV20 universe. Every cell-fold runs measured, frictionless, and
half-spread variants, for 432 standard reports. The spread multiplier is part of
the config hash. Required outputs also include all-cash-CDI excess, liquidity-
tercile attribution, round-trip economics, and per-refresh selection-extension
telemetry.

Pre-score Amendment A53.1 is frozen: Experiment 53 alone uses a 5%-of-gross name
cap; insufficient selected-side name/ADV cap capacity completes deterministically
with the next-ranked names; every refresh records the extension count; exit
hysteresis applies to the completed set; and a cell deploying less than 50% of
its configured gross on any fold is ineligible for C1 but remains fully
reported. The original 48 cells and every other rule are unchanged.

C1 uses the Experiment-52 measured-only two-fold rotation rule after the
deployment guard. Any net-positive cell is an existence proof and lower bound
for a future learned policy. An all-negative map is a verdict on this hand-policy
family only and may never be described as evidence that the alpha is
untradeable. C0 and C1 are both retained.

The frozen Experiment-52 contract has no causal historical quoted-tick archive.
Consistent with the raw-data contract, MT5's historical `spread` field is not
repurposed as a market spread and the current catalogue snapshot is not treated
as historical. The requested informational Roll sanity therefore emits per-name
lagged-Roll values and explicitly unavailable tick ratios without changing the
schedule or inventing an input. The canonical preregistration is
`research/preregistrations/experiment53_feasible_region.md`.

Experiment 53 completed from frozen implementation/preregistration commit
`00cb38a6f414198d87c24553f8eed8af78214093`. All 48 cells ran on discovery
Folds C/A/B under measured, frictionless, and half-spread contracts: 432
standard reports and 432 summary rows. The final audit passed with
`official_validation_accessed=false` and `test_accessed=false`. Amendment
A53.1 was preserved exactly. Capacity completion was active in 28 of 144
measured cell-folds, spanning 131,890 refreshes and 417,722 added-name events;
all per-refresh records were retained. The Roll diagnostic covered 309 dates
and 158 securities, recorded the causal lagged-Roll distribution, and correctly
left the quoted-tick ratio unavailable because no permitted historical quoted-
tick archive exists in the frozen inputs.

The preregistered gross guard left 36 cells eligible and 12 ineligible. All
eight cells that were net-positive on every measured fold were top-half-ADV
K=40 cells excluded by that guard, and no cell beat all-cash CDI on every fold.
The measured-only rotation designated
`k40__band1p5__c1p0__gross1p0__universe_full` as C1 with three wins. Its mean
deployed-gross fractions were `0.832879`, `0.826737`, and `0.855144` on A/B/C,
but its measured Sharpes were `-7.188525`, `-8.505339`, and `-14.026724`.
Across those folds it produced `R$1,428,362.47` gross PnL, paid
`R$5,488,091.32` spread and `R$2,149,913.71` fees, earned `R$815,557.00` CDI,
and finished at `-R$5,394,085.57` net and `-R$6,804,325.37` versus all-cash
CDI on `R$10,749,568,569.19` turnover. C1 is therefore retained only as the
feasible hand-policy reference; it is not promoted or deployed, and C0 remains
the baseline comparator.

The immutable completed root is:

    /lambda/nfs/brazil-rv-east3/quant-data/b3/processed/model_runs/execution_feasible_53_00cb38a_20260827T184000Z

Frozen-design, program-manifest, cell-summary, C1-designation,
selection-extension, result, and log-inclusive final-audit SHA-256 values are
respectively
`54344ff68ea55da1a13f6d3cb879a1154edb3eeaafb6839358fd1951ad22aecb`,
`e80c5cccf8b0e92b202aeff23cc0ec41472e7f588dd6236a6d6838709e510b6a`,
`eb44cb0b9e650a877ad9b4876e45a9c7f7af1854326dc56f7ef66e21b37158c0`,
`db61aa52e2b854bf5bc7709d9e3846e531bcd87c301857fe64ab11483fea1d72`,
`e7bb3369ef52759036c00fbf25defc002a30a68c9eb6e65c24d79c761131bb03`,
`4c1cb2e73a6608bd4e06ada9213ca744a7acb1da4356b2cb012e4da6509a164b`,
and `bc5811f118cabae6514e889709a5b801551ebdd91f2b3094183107ad9d43f513`.
The audit covers 2,606 files / 23,794,843 bytes, including every report,
readout, analysis, telemetry table, manifest, checksum, and both operational
logs. No prediction recipe or deployment changed.

Result/context commit `6862f5c044518f126bec56a1d991112057700a8c`
reached GitHub before shutdown. Exact paid GH200 instance
`21d5542246d144978478284e10837a22` was terminated only after all required
artifacts, analyses, hashes, and logs were secured. Provider inventory found
that exact ID absent and zero active instances at
`2026-08-27T19:14:50.9818266Z` and again at
`2026-08-27T19:15:18.8786125Z`.

## Experiment 54 — conditional edge, latency, and maker feasibility (preregistered 2026-08-27)

Status at registration: no Experiment-54 bucket, forward-edge, fill, frontier,
positioning, decision, or result number exists, and provider inventory contains
no paid instance. This is a CPU-only, discovery-only pure-analysis program. It
must not run the simulator, train a model, alter C0/C1 or the deployed prediction
recipe, or access official-validation or the permanently spent held-out test.

The immutable inputs are the completed Experiment-53 root and its exact
Experiment-52 source. The program reuses only the three discovery prediction
wrappers, causal ADV20/minute-capacity profiles, lagged spreads, daily sigma,
and CDI. It streams identity-bounded raw M1 only through `TRAIN_END`, requires
raw open/observed values to reproduce the Experiment-52 cache, and verifies
observed high/low/close before any outcome analysis.

Fold C alone freezes rank deciles, absolute-rank-change quintiles, and causal
ADV20/spread/sigma terciles; A/B reuse those numeric edges unchanged. The state
also includes tail entry, prediction-grid first/middle/final hour, and all-head
sign agreement. The first refresh is excluded rather than assigned an invented
rank change. Part A measures signed gross edge at `15/30/60/120` minutes from
the decision open, next open, and complete 10/30-minute mean-open convergence
entries. Exact entry observations and exact session-close endpoints are
required. Conditional tables retain fixed `4.5/7/10`-bps hurdle clearance and
each event's half-spread/measured/conservative lagged cost clearance.

The taker frontier is an optimistic same-fold state-cell oracle, never an event-
outcome oracle. It uses next-open cell means, each event's measured taker cost,
10% causal minute capacity, a 5%-of-gross name cap, R$10m NAV, and gross at most
2 with no neutrality or persistence constraint. At the 7-bps threshold, the
maximum over the four registered horizons is compared with 8 NAV bps/day on
each fold. Below 8 on every fold closes taker actions for learned policy; at
least two folds at or above 8 keeps them viable with all-cash CDI as hurdle;
the remaining pattern is inconclusive.

Part B tests tail-only limits at the last observed predecision close and one
half of the lagged half-spread better, waits `5/15/30` minutes, and fills only
on a strict low-through/high-through. Filled paths pay 2 bps; unfilled paths
cross at the exact open after the wait and pay measured taker cost. The maker
frontier uses positive same-fold conditional composite-net means with the same
sizing caps. It has no automatic decision rule. Part C reports an analytic
long-short versus long-only-plus-cash comparison, with drift shown once and
long-only beta variance reported in bps-squared rather than subtracted using an
unregistered risk-aversion coefficient.

No bucket, state, horizon, wait, threshold, cost, sizing rule, decision,
schedule, or interpretation may change after a forward outcome is computed.
The canonical executable preregistration is
`research/preregistrations/experiment54_edge_maker.md`.

### Experiment 54 completion

Exact frozen pre-score implementation/preregistration commit
`047e2d2c3989175d46f1decee04247270b449556` reached local `main` and GitHub
before any forward return or fill was computed. Windows Ruff/compile and 38
focused tests passed; the exact Linux checkout repeated Ruff, compile, and the
same 38 tests before freeze. Exact paid GH200 instance
`e02aac505c884d129c8118577e671ed3` at `192.222.51.247` ran only the registered
CPU analysis. Frozen-design SHA-256 was
`b358144752049d149305f4f7ecfc2021c1d224a176716efc4267b88e166018e7`.

The state exclusions retained 757,148, 734,479, and 727,940 eligible events on
C/A/B respectively; first-refresh and other registered exclusions were
recorded rather than imputed. Across folds, mean signed decision-open gross
edge was `0.893317/1.233609/1.602609/2.292508` bps at
`15/30/60/120` minutes. Moving to the next observed open reduced those means by
`0.198878/0.225159/0.216071/0.219717` bps. Every registered entry/horizon and
conditional state table was retained.

At the frozen 7-bps taker hurdle, the best registered horizon was 120 minutes.
The optimistic same-fold state-cell frontier measured `9.141462` expected NAV
bps/day on A, `7.944564` on B, and `10.429961` on C. Because A and C cleared
the inclusive 8-bps/day rule, the exact decision is `VIABLE`: a future learned
policy may retain taker actions and uses all-cash CDI as its reward hurdle.
This is explicitly an oracle feasibility bound rather than realized simulator
PnL, acceptance, deployment, or a C0/C1 replacement.

The best maker frontier was Fold A, 120-minute horizon, 15-minute wait, and the
half-half-spread-improved limit at `3.917194` expected NAV bps/day. The same
schedule produced `3.739960` on B and `3.856317` on C. Strict-through fill
rates ranged `64.3%–66.8%` at 5 minutes, `76.1%–78.2%` at 15 minutes, and
`81.9%–83.7%` at 30 minutes. No maker schedule reached the 8-bps/day framing;
the preregistration provides no automatic maker verdict. The Part-C
construction comparison was retained as information only and authorized no
policy or deployment change.

The immutable root is:

    /lambda/nfs/brazil-rv-east3/quant-data/b3/processed/model_runs/execution_edge_maker_54_047e2d2_20260827T201700Z

It retains raw high/low/close verification, exact frozen buckets, exclusions,
48 latency rows, 276,881 taker-conditional rows, 36 taker frontiers and daily
tables, 163,440 maker-conditional rows, 72 maker frontiers and daily tables,
24 positioning rows, decisions, definitions, manifests, and both operational
logs. Frozen-design, result, program-manifest, and log-inclusive final-audit
SHA-256 values are respectively
`b358144752049d149305f4f7ecfc2021c1d224a176716efc4267b88e166018e7`,
`f5bcdfbfd654be29e1c9c758ee0276d9eb386eefd1290b5dadb7be5e11939fb1`,
`129dcfc09273d923948ea2c2dcf8e929885629776b2fc34b53d150ca1764aa86`,
and `d040f12b0a43356a53c1d03070da5a4d7c9f24f1f8cb4054dd73d3251e37c364`.
The final audit passed over 21 artifacts / 495,173,023 bytes with
`simulator_run=false`, `official_validation_accessed=false`, and
`test_accessed=false`. Per the user's explicit keep-alive override, exact
instance `e02aac505c884d129c8118577e671ed3` is not terminated after Experiment
54 and is reserved for the immediately following experiments.

Result/context commit `6eac1962386aede3345d98fcf4a0cfc5886ac6c0`
reached GitHub, and the instance checkout was fast-forwarded to that exact
commit. A direct provider inventory read at `2026-08-27T20:28:00.1318316Z`
returned exactly one instance, exact ID `e02aac505c884d129c8118577e671ed3`,
with status `active` at `192.222.51.247`. It remains ready by explicit user
request; no termination was attempted.

## Learned-policy package Part 1 — code-only implementation (2026-08-27)

Before any real policy trajectory or research score, the execution layer was
extended with the registered Part-1 mechanisms: a sign-preserving bounded
neutral/cap projector that never scales gross up; a schema-hashed causal policy
state built inside the minute scan; volume-weighted entry-price and position-age
tracking; a zero-initialized shared per-name neural policy; direct net-PnL-above-
all-cash-CDI optimization with AdamW default and opt-in SAM/checkpointing; and a
frozen five-block/five-session-purged TRAIN split manifest with exact C/A/B policy
evaluation slices. Legacy Band/ConcentratedPolicy targets retain their original
exact-gross path.

This is an implementation registration, not an experiment result. Validation
used synthetic unit and simulator fixtures only. No canonical prediction,
feature-store, official-validation, or held-out-test artifact was opened; no OOF
member was refit or materialized; no real learned policy was trained; no
Experiment-55 sidecar, score, gate, or trajectory was produced. The OOF archive
loader remains closed to aggregates until the separately registered constituent-
fold provenance chain is implemented and manufactured. The held GH200 was not
used, modified, or terminated for this code-only work.

## Learned-policy OOF manufacture (preregistered 2026-08-27)

Status at registration: no OOF trajectory, held-out prediction, calibration, or
real policy score exists. The exact five-block/five-session purged TRAIN manifest
is frozen before training. Five folds times seeds
`11/29/47/61/79/97/113/131/149/167` use the deployed store-v2 depth-six,
temperature-0.50 recipe for fixed 20 epochs with no monitor and no held-out
evaluation during training. Epoch-20 raw is archived and final EMA-0.995 is the
member. Ten members are causally rank averaged per held-out fold and assembled
into a 716-date archive whose loader independently verifies every source fold,
fit exclusion, run/file hash, and exact coverage. A frozen C/A/B calibration
compares monitor-free OOF EMA ranks with the exact Experiment-41 Patience
comparator. At most two training processes run. Checkpoints may be removed only
after archive verification; predictions, references, histories, manifests,
calibration, and audits remain. The executable registration is
`research/preregistrations/oof_prediction_manufacture.md`.

## Experiment 55 — horizon-conditioned to-close head (preregistered 2026-08-27)

Status at registration: no to-close target sidecar, trajectory, fold score,
conditional edge, or gate result exists. After OOF manufacture, a TRAIN-only
sidecar freezes exact standard-entry-to-final-close labels, median removal,
causal `sigma*sqrt(H)` scaling, and per-group midranks. The candidate adds three
zero-initialized full-width readouts combined by `[1,H/405,sqrt(H/405)]`, retaining
exact three-head and RNG parity at epoch zero. The deployed store-v2 specification
runs on C/A/B with seeds 11/29/47 and standard four-head-selected cross-fitted
Patience; final EMA-0.995 is secondary.

The three-head guardrail requires mean delta at least zero and every fold delta
at least `-0.0005`. To-close IC is reported overall and by morning/middle/late
third. Economics reuses the exact Experiment-54 buckets, costs, and capacity
allocator; adoption requires the guardrail plus at least `+2` incremental NAV
bps/day versus the best existing frontier on at least two folds. Adoption affects
execution inputs only. If it fires, exactly 50 four-head OOF refits run under the
already frozen no-monitor protocol; otherwise they are skipped. No retry, new
bucket, head, state, threshold, cost, weight, official-validation read, or test
read is allowed. The executable registration is
`research/preregistrations/experiment55_to_close.md`.

### Experiment-55 pre-score compile repair (2026-08-28)

The first Experiment-55 launch stopped all nine scheduled cells at epoch zero,
before a trajectory, prediction, or score existed. PyTorch full-graph compilation
correctly rejected a defensive tensor-dependent Python branch in the new
to-close basis. The branch only rechecked the canonical decision-position range
already fixed by the dataset contract; removing it leaves the registered
`[1,H/405,sqrt(H/405)]` arithmetic unchanged. A focused full-graph regression
test now exercises the four-head forward path. The failed root and operational
log are retained as provenance, and the repaired program must freeze into a
fresh commit-bound root before training.

### Pre-score OOF freeze repair (2026-08-27)

The initial OOF startup attempts stopped before training a model or reading any
score: the first during freeze before a design was written and the second during
job construction after a valid design freeze. The date-axis constructions had
called Polars `DataFrame.sort()` without its required column argument. The
bounded operational repair reads the already-selected `trade_date` Series
before sorting; it changes no date, fold, embargo, seed, recipe, archive, gate,
or access rule. The incomplete pre-score roots are excluded and the repaired
program must freeze into a fresh root. Full Ruff and the 394-test suite passed
before the repaired freeze; official validation and the spent test remain
unopened.

### Post-training, pre-calibration OOF materialization repair (2026-08-28)

All 50 frozen trajectories completed their exact 20 epochs and wrote their
hash-bound raw/final-EMA prediction archives before the first materialization
attempt. That attempt stopped before writing the aggregate archive, calibration,
research score, cleanup, or Experiment-55 artifact because it treated the
store's absolute `date_idx` values as zero-based positions in the TRAIN-only
date tuple. The canonical TRAIN slice has 716 dates indexed `20..735`, so the
date-coverage assertion failed without consuming a score.

The bounded operational repair maps `date_idx` to `trade_date` explicitly and
uses those same absolute indices when selecting the frozen C/A/B calibration
dates. It changes no fold, embargo, seed, recipe, member prediction, rank
averaging, weight, target, gate, or access rule. The 50 completed trajectories
are reused without retraining; a changed code commit is accepted only after all
50 hash-valid source runs from the frozen commit exist. The aggregate source
manifest, result, and audit record the frozen and operational-repair commits
separately. A focused nonzero-index regression test protects the repaired axis
contract; official validation and the permanently spent test remain unopened.

The first repair replay then wrote the aggregate candidates atomically but
stopped during their independent loader verification, still before calibration
or any score. The loader bounded the canonical date table above by `TRAIN_END`
but omitted the frozen `TRAIN_START`, thereby including the store's 20 pre-TRAIN
dates and rejecting the required 716-date fold contract. The same bounded repair
now applies the exact closed TRAIN interval in that independent verifier. This
changes no archive value or research rule; the unverified aggregate candidates
are deterministically overwritten and reverified from the 50 immutable runs.

That verifier then passed on the next replay. The immediately following
calibration-slice constructor stopped before loading a comparator or computing a
score because its separate canonical date-table read had the same omitted lower
bound. It now uses the identical closed TRAIN interval. This completes the
date-axis repair consistently across materialization, independent archive
verification, and the preregistered calibration slice construction.

## Learned-policy OOF manufacture — completed (2026-08-28)

The repaired replay completed without retraining any finished trajectory. All
50 frozen fold/seed runs completed exactly 20 epochs (1,000 total); raw and
final-EMA-0.995 archives, histories, references, manifests, exact fit
exclusions/embargoes, and per-sample source-fold proof were retained. The
causal rank-average archive covers exactly 716 TRAIN dates (`date_idx 20..735`)
and 39,380 samples. Independent loader verification passed before calibration
or checkpoint cleanup. The 50 epoch-20 checkpoints were then removed only
after their prediction archives were hash verified.

OOF final-EMA rank IC versus the exact Experiment-41 Patience comparator was
`0.037489` versus `0.036936` on C, `0.042963` versus `0.048181` on A, and
`0.049915` versus `0.053639` on B. These are calibration readouts, not a new
deployment decision. The immutable root is:

    /lambda/nfs/brazil-rv-east3/quant-data/b3/processed/model_runs/learned_policy_oof_53859b8_20260827T220239Z

Its log-inclusive final-audit SHA-256 is
`2b023ca3368fd06cc2098298c73704b4571a08516acdb93364280dd3d3525451`.
The audit covers 211 retained files / 3,259,772,887 bytes, all 50 manifests,
1,000 epochs, aggregate and source hashes, exact coverage/provenance, loader
verification, calibration, cleanup inventory, and both operational logs.
`official_validation_accessed=false` and `test_accessed=false` throughout.

## Experiment 55 — completed, not adopted (2026-08-28)

After the recorded epoch-zero compile failure, the fullgraph-compatible repair
froze at commit `3b69ba8be11cc0308f96923b337e6382ed4dd1e4` into a fresh root. Exactly
nine C/A/B-by-seed trajectories completed 20 epochs each (180 total), producing
180 epoch prediction archives. The TRAIN-only to-close sidecar, zero-impact
three-head guardrail construction, Experiment-54 economics binding, cleanup,
access flags, and every retained artifact hash passed independent audit.

The to-close head showed positive capability: overall IC was `0.050540`,
`0.074533`, and `0.066367` on C/A/B. At the frozen 7-bps threshold, incremental
expected NAV edge versus the best Experiment-54 three-head frontier was
`+10.833850`, `+8.752321`, and `+9.168938` bps/day respectively, so all three
folds passed the economics gate. Adoption nevertheless failed the prior
prediction guardrail. Three-head IC deltas versus the exact comparator were
`-0.000270` on C, `-0.001212` on A, and `+0.000409` on B; Fold A breached the
frozen `-0.0005` floor. Therefore `adopted_for_execution_layer=false`, the
conditional 50-run four-head OOF extension was correctly skipped, and no
deployment recipe changed.

The immutable completed root is:

    /lambda/nfs/brazil-rv-east3/quant-data/b3/processed/model_runs/experiment55_to_close_3b69ba8_20260828T055100Z

Frozen-design, result, and log-inclusive final-audit SHA-256 values are
`7f298d2aada74e9d0d8c655d45b732ebbc8c265d9534c45fdf6f5b92c4b96b51`,
`3101985765fcba52dbc193514ffc8b882b5604c741ec81f065b0292256afac93`,
and `21dd4f4ab1abb22e1445f2d8d1c5fcb3667952701620e68c6ba63183c4989cbd`.
The final audit covers 243 retained files / 11,176,806,611 bytes, nine
manifests, 180 epochs, 180 prediction archives, six analysis tables, the target
sidecar, checkpoint cleanup, both operational logs, and a complete file hash
inventory. `official_validation_accessed=false` and `test_accessed=false`;
the permanently spent held-out test was not read again.

## Experiment 56 — four-head OOF and learned execution policy (preregistered 2026-08-28)

Experiment 56 is registered before any new forward outcome, conditional edge,
OOF calibration, policy objective, or evaluation score exists. The exact
contract is `research/preregistrations/experiment56_policy.md`. The program has
three separately frozen, commit-bound sections.

Section A records the standing research decision that the Experiment-55
to-close head is adopted as an execution-layer input on its own capability and
economic evidence. This does not rewrite the historical Experiment-55 result
and does not change the deployed Experiment-45 prediction recipe. The retained
Experiment-55 candidate archives will be replayed through the exact
Experiment-54 state, cost, and capacity contracts. The four-head total frontier
allows exactly one deterministic best expected-net horizon per name/refresh,
preventing horizon double counting. Sections B/C abort only if that total is
below the best threshold-7 three-head frontier on at least two folds.

If the gate proceeds, Section B trains exactly 50 monitor-free four-head OOF
trajectories: five purged TRAIN folds by the ten frozen seeds, 20 epochs, final
EMA 0.995, maximum two processes. Section C then trains exactly 18 NeuralPolicy
runs on OOF ranks: C/A/B by lambdas 0.02/0.10 by seeds 11/29/47. Each evaluation
window uses only dates ending before a five-session embargo; the last floor-20%
of those dates selects the checkpoint and all earlier dates fit it. Patience is
10 under the exact bps objective `mean(net - all-cash CDI) - lambda *
population_std(net)`, with a 100-epoch operational ceiling. A window's lambda
is chosen by mean selection objective across all three seeds. Graduation is
strictly and only positive pooled daily net excess over all-cash CDI after
averaging the three designated-policy seed replicas by date.

The exact paid GH200 claimed for this program is
`16d1f1be8d0d4261a5583f27d4cd3ff0` at `192.222.51.247`, attached to
`brazil-rv-east3`. It was bootstrapped at commit
`4042723555365c6807c0459b7a146e0c1f0c5fd8`; no training or Experiment-56 score
has yet run. Official validation and the permanently spent held-out test remain
inaccessible. After all required roots, results, diagnostics, histories,
designated checkpoints, logs, and hashes are secured and pushed, only this
exact instance will be terminated and its absence verified twice.
