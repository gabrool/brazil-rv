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

Per the user's explicit override, paid GH200 instance
`d0ebcd5f7dbb44dc99370080df7b47cc` remains active for the immediately
following experiment and must not be terminated at Experiment-46 closure.
