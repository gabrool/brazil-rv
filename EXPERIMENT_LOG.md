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
