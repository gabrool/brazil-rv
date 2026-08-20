# Brazil-RV research handoff

Last verified: 2026-08-18.

This is the current-session handoff for a new researcher or LLM. Read
`AGENTS.md` and `PROJECT_CONTEXT.md` first, then use executable code, immutable
feature/run manifests, and canonical pointers as the final authority. Some older
Markdown in Git history describes systems that were deliberately deleted.

## Executive state

- The accepted incumbent family is the peer-free, full causal time-of-day (TOD)
  normalized, width-64 causal TCN trained uniformly with soft Spearman and
  SAM-AdamW. It has no cross-equity attention.
- The best exact validation score verified in this session is **0.041972** for
  seed 11. The project-level incumbent discussed at the start of the session was
  approximately 0.0415. Nothing reached 0.05 or 0.06.
- The gap-weighted pairwise hybrid objective was tested and rejected at seed 11:
  IC **0.037294**, delta **-0.004678** from the matched soft-Spearman control.
- Residual-state cross-equity attention on top of that hybrid parent was also
  rejected at seed 11: IC **0.034091**, delta **-0.003204** from its parent.
- No held-out test observations were loaded. Every campaign and report records
  `test_accessed=false`.
- The original 21-run campaign was deliberately curtailed after weak early
  evidence. It did **not** complete 21 runs.
- Implementation work through commit `a91815d08c282a2f1018c9e22a7db3731f104c55`
  was tested and used for the final two-run campaign. Later documentation commits
  do not change those reported runs.

## Accepted incumbent architecture

The model accepts one date/decision sample containing a fixed 158-slot
point-in-time equity axis plus named context instruments.

```text
Each instrument's causal 5-minute patches (69 x 130)
                         |
        shared width-64 causal temporal encoder
     kernel 3, dilations 1/2/4/8/16/32, 6 blocks
        LayerNorm + SwiGLU + residual in each block
                         |
       final temporal state + projected 32-D slow state
                         |
              64-D state per instrument

Equity state ---------------------------------------+
15 fixed-slot context states -----------------------+--> gated fusion --> 3 scores
masked equity mean and dispersion ------------------+
```

Important details:

- Five one-minute rows form each patch, so the patch input width is
  `5 * 26 = 130`.
- The nominal temporal receptive field is 127 patches, larger than the 69-patch
  input. The encoder already covers the complete history window.
- TCN weights are shared across equities and contexts, but each instrument is
  encoded independently. The TCN itself does not learn stock-to-stock mixing.
- The readout is the final causal state. A linear slow-state projection is added
  before state normalization.
- Context-plus-pooled fusion concatenates the named context states, masked equity
  mean, and masked equity dispersion; a gated residual MLP conditions each equity
  state on that shared vector.
- The accepted no-attention model has 277,379 trainable parameters and produces
  scores for 30-, 60-, and 120-minute horizons.

The context screen is fixed rather than an experiment switch:

- Active local contexts: `WDO$`, `DI1F27`, `DI1F28`, `DI1F29`, `DI1F31`, and
  `DI1$N`.
- `WIN$` is masked and equity `beta_to_WIN` is zeroed.
- Active global contexts: `ZT.v.0` and `ZN.v.0`.
- `ES.v.0`, `NQ.v.0`, `CL.v.0`, `HG.v.0`, `6E.v.0`, and `6M.v.0` are present in
  the store but masked by the accepted model policy.

## Features, normalization, targets, and splits

The current peer-free full-TOD store built and audited on the GH200 is:

```text
/lambda/nfs/brazil-rv-east3/quant-data/b3/processed/features/
  m1_features_pit_causal_tod_20260818T151728490951Z
```

Identity:

- Contract: `M1_FEATURES_PIT_CAUSAL_TOD`
- Metadata SHA-256:
  `c90103b0f99e0017dc1303284a1ab61eca99106094227f5823ba718756d28a6b`
- 1,248 dates, 1,228 eligible dates, and 67,540 date/decision samples.
- No human-prior or peer arrays exist in the schema.

The Windows workspace's local canonical pointer may still identify the old V4
store. The promoted full-TOD pointer was written on the persistent Lambda NFS
filesystem. Resolve and verify the pointer in the environment where an experiment
will run; do not silently use the local V4 pointer with current code.

Equity normalization is causal full TOD normalization:

- 30-minute bins.
- The profile is estimated from unclipped legacy-normalized equity close moves.
- Twenty-session-equivalent shrinkage prior centered on relative variance one.
- Relative-variance bounds `[0.25, 4.0]`.
- Each training date emits its profile before updating it.
- The profile freezes after 2024-06-28 for validation, embargo, and test dates.
- Only equities receive the TOD overlay.

Other series are normalized according to their semantics:

- Price-like equities and WDO use log moves divided by causal volatility.
- Fixed DI quote changes are converted to basis points and divided by causal
  rate-change volatility; valid prior rate level and exact expiry distance live
  only in applicable slow fields.
- `DI1$N` is session-local and does not create an absolute level, overnight chain,
  or fabricated selected maturity.
- Global futures use log returns, completed-Globex-session causal volatility,
  causal robust volume state, and roll/mapping-change masks.
- Volume surprises use trailing robust median/MAD state. Missing bars are never
  interpolated; masks carry availability.

For equity `i` and horizon `H`, the continuous precursor to the rank target is:

```text
z_i = (log(exit_close / entry_open) - contemporaneous CS median)
      / (causal equity sigma_i * sqrt(H))
```

The stored learning target is the centered cross-sectional midrank of `z_i`,
independently by date, decision, and horizon. Model history ends strictly before
decision time, the entry bar is excluded, and labels cannot cross the permitted
session boundary.

Splits:

| Split | Dates | Date count | Samples |
|---|---|---:|---:|
| Train | 2021-08-16 through 2024-06-28 | 716 | 39,380 |
| Embargo 1 | between train and validation | 5 | 275 |
| Validation | 2024-07-08 through 2025-06-30 | 244 | 13,420 |
| Embargo 2 | between validation and test | 4 | 220 |
| Held-out test | 2025-07-07 through 2026-07-17 | 259 | 14,245 |

The primary metric is mean daily cross-sectional Spearman IC: average decisions
within each date/horizon, then average validation dates and the three horizons
equally.

## Training contract for the incumbent

- Objective: soft Spearman, temperature 0.50.
- Recency: uniform over all 716 training sessions.
- Seeds used for matched campaigns: 11, 29, and 47.
- Optimizer: SAM-AdamW, SAM rho 0.125, learning rate `3e-4`, AdamW betas
  `(0.9, 0.95)`, epsilon `1e-8`, weight decay 0.01.
- Effective batch: 512, composed from two ordered 256-sample loader batches.
- Date-stratified sampling.
- Maximum 20 epochs, early-stop patience 3, minimum IC improvement `1e-4`.
- PyTorch Inductor compilation uses default mode, full graphs, and static shapes;
  validation is eager.

## What was implemented and removed in this session

The cleanup reduced the session diff by roughly 28,700 deleted lines. It removed:

- Human-prior acquisition, classifications, static peer construction, peer arrays,
  peer loader/model paths, and related audits/tests.
- Old attribution/probe, conflict-OOF, feature-variant, normalization-overlay,
  horizon/multiscale, and stage-validation machinery.
- Transformer and MLP model families, alternate TCN configurations, routing modes,
  readout variants, single-horizon branches, rank Huber, and other completed
  experiment switches.
- V-numbered normalization implementations and the old 21-run driver after the
  campaign direction changed.

The remaining core preserves point-in-time membership, security/source identity,
entry-bar exclusion, session-safe labels, causal/frozen fitted state, raw-source
immutability, feature-store identity, atomic run artifacts, and test isolation.

Portability fixes made during remote execution include workspace-path resolution
between Windows and Linux/NFS, repository identity resolution from the source root,
and support for security slots that are inactive through the development period
when building the temporary target-scale sidecar.

## Experiment chronology and actual run count

### Original PIT-clean campaign

The planned campaign had 21 run specifications: three legacy controls, three
full-TOD uniform controls, twelve recency runs, and three attention runs.

What actually happened before the campaign was curtailed:

- Three clean legacy controls were reused from the persistent campaign at
  `pit_clean_core_campaign_858b372`; they were not retrained in this session.
- Three peer-free full-TOD uniform controls, seeds 11/29/47, completed under
  `pit_clean_core_campaign_4067962`.
- At the last recorded original-campaign checkpoint, scientific progress was 6/21
  when reused controls were included, but only three new runs had completed.
- The first `exp_504` recency attempt was stopped while incomplete after weak early
  results. All remaining recency specifications were skipped. It must not be
  counted as a completed result.
- A first non-residual uniform-TOD attention attempt showed declining/stagnant
  validation behavior while training loss fell. The requested three-seed attention
  sweep was not completed and is not promotable evidence.

The expensive feature construction was legitimate but much slower than estimated:
it made a two-pass 146-source full-TOD store, then audited it. That work was shared
preprocessing, not repeated baseline training.

The exact full-TOD seed-11 soft-Spearman result used in all later comparisons was:

| Metric | Value |
|---|---:|
| Primary IC | **0.0419722656** |
| 30-minute IC | 0.0460368119 |
| 60-minute IC | 0.0409836943 |
| 120-minute IC | 0.0388962908 |
| Best epoch / epochs completed | 12 / 15 |
| Runtime | 661.3 seconds |

The other two uniform-control manifests remain under the persistent source
campaign. Their exact values were not copied into this Git checkout, so do not
invent them; read their immutable run manifests after mounting the NFS store.

### Hybrid loss experiment

The motivation was that soft Spearman treats near-ties and large target separations
similarly. The tested loss was:

```text
L = L_soft_spearman + 0.25 * L_gap_pairwise
pair_weight(i,j) = min(abs(z_i - z_j), 1.0)
```

The pairwise logistic temperature was 0.50. The exact continuous `z` values were
reconstructed from immutable raw-return/median arrays and exact causal equity
sigmas in a development-only sidecar. The sidecar ended at validation end and had
SHA-256 `46031721f696c1d55c1d9285caacaedb2177051fa670f4a4c3028a455b9c7245`.

Interpretation: no new inference feature is required for this objective. The gap
only changes gradient weights. The model can benefit only when existing causal
inputs predict the large separation; the loss cannot manufacture absent signal.

### Residual-state attention experiment

The candidate was designed after the first attention attempt appeared to overfit:

- Take final 64-D equity states after slow-state addition.
- Pre-normalize them and subtract the active cross-sectional mean.
- Apply one four-head, bias-free self-attention layer to those residualized states.
- Mask inactive equities as keys/values and zero their outputs.
- Add the result residually to the original, non-residualized equity state.
- Preserve the original fixed context-plus-pooled fusion afterward.
- Zero-initialize the attention output projection so epoch-zero behavior exactly
  matches its no-attention parent.
- Add no ticker, security, sector, positional, or classification embeddings and no
  separate feed-forward block.

This is a coherent permutation-equivariant equity mixer. The common component is
not discarded: it stays on the residual path and enters pooled/context fusion.

### Final two-run campaign results

Campaign:

```text
/lambda/nfs/brazil-rv-east3/quant-data/b3/processed/model_runs/
  hybrid_loss_residual_attention_a91815d
```

| Arm, seed 11 | Best epoch | Epochs | Primary IC | IC 30 / 60 / 120 | Delta from parent |
|---|---:|---:|---:|---:|---:|
| Existing soft-Spearman full-TOD control | 12 | 15 | **0.041972** | .046037 / .040984 / .038896 | - |
| Hybrid base | 13 | 16 | 0.037294 | .043617 / .036109 / .032158 | **-0.004678** |
| Hybrid + residual attention | 8 | 11 | 0.034091 | .039608 / .032327 / .030338 | **-0.003204** |

The hybrid base took 745.4 seconds. Attention took 489.3 seconds. The systemd
service completed successfully with zero restarts. The campaign report records
`test_accessed=false`.

Decision: do not run seeds 29/47 for these exact candidates. The losses are large
enough on the screen seed that neither deserves promotion. Retain the full-TOD
soft-Spearman no-attention family as incumbent.

## Conceptual conclusions from the session

### Role of the TCN

The TCN is a shared causal temporal encoder, not inherently a denoiser or a
cross-asset model. Locality, dilation, causality, residual learning, and sharing
weights across instruments are its sample-efficiency biases. Denoising is learned
only if it helps the objective. Stock-to-stock interaction begins in cross-sectional
features, pooled fusion, or an explicit post-encoder mixer.

### RevIN

Whole-input RevIN is not a natural fit. RevIN assumes a forecasting output that can
be returned to the same physical units after per-instance normalization. Brazil-RV
outputs dimensionless cross-sectional scores and mixes semantically different
channels: returns, volume surprises, ranks, masks, calendars, rate levels, and slow
state. A single window mean/variance would erase useful regime information and mix
structural zeros into statistics. A future ablation could normalize only masked
price-move channels, but that would not be ordinary RevIN and is not currently a
top-priority experiment.

### Why macro instruments were not put in equity attention

Equities are an exchangeable set; DI tenors, WDO, ZT, and ZN are named and
heterogeneous. Their identity and maturity matter. They already condition every
equity through fixed-slot nonlinear fusion. If context processing is revisited,
use a tenor-aware DI curve encoder or equity-query/context-key cross-attention,
not undifferentiated self-attention over equity and macro tokens.

## Highest-value next research order

1. **Learned low-capacity set pooling.** Apply a small shared transform to equity
   states, masked-mean the result, and feed that learned market summary back to
   each equity. This is an O(N), lower-variance extension of the existing fixed
   mean/dispersion pool.
2. **Common/idiosyncratic auxiliary decomposition.** Keep the actual rank target,
   but add a causally defined factor/common and residual auxiliary objective. Do
   not reproduce a large paper architecture blindly; control beta/factor estimation
   error and preserve both components for the final raw-return ordering.
3. **Low-rank factor mixer.** Learn a small number of pooled latent factors and
   per-equity loadings instead of another full N-by-N attention matrix.
4. **Causal ModernTCN-lite screen.** Match inputs, loss, fusion, optimizer, width,
   and approximate parameter count. Use left-only padding. This is medium value
   because the incumbent already has a receptive field larger than the input.
5. **Causal ConvTimeNet channel-independent screen** only after the above. Ensure
   deformable patch offsets can never sample future positions. Do not begin with
   the channel-dependent version across equities.

Use the accepted soft-Spearman incumbent as the parent for new experiments, not
the rejected hybrid objective. Start with one matched seed; promote to seeds
11/29/47 only after a meaningful improvement.

## Current HEAD status

The current source tree is the restored peer-free soft-Spearman parent plus the
internal-fold trajectory measurement layer. Historical hybrid and attention
results above remain authoritative records, not executable compatibility paths.

- `modeling.train` runs a fixed 20-epoch soft-Spearman SAM trajectory and records
  raw, EMA, weight-average, and prediction-average candidates.
- Rejected hybrid loss, continuous-target sidecar, residual attention, recency
  weighting, and their campaign driver have been deleted from current code.
- The rejected Patience-centered checkpoint-average evaluator is likewise absent;
  its recorded commit and immutable artifact preserve exact reproduction.
- Exact historical reproduction still uses the recorded commit
  `4067962f6bb6748a530814d10e20dfc865a385c7`, immutable store identity, and run
  manifest.
- The official validation split is reserved for sparse stage-winner checks; the
  held-out test remains accessible only through the standalone frozen-rule evaluator.

## Completed restoration and trajectory screen (2026-08-19)

The peer-free incumbent was retrained from historical commit
`4067962f6bb6748a530814d10e20dfc865a385c7` before the new recipe was used. The
matched reproduction is stored at:

    /lambda/nfs/brazil-rv-east3/quant-data/b3/processed/model_runs/parent_reproduction_4067962_e22dd67_20260819T131142Z

| Seed | Immutable IC | Reproduced IC | Delta | Best / stopped epoch |
|---:|---:|---:|---:|---:|
| 11 | 0.041972266 | 0.041977574 | +0.000005309 | 12 / 15 |
| 29 | 0.040481999 | 0.040475500 | -0.000006499 | 14 / 17 |
| 47 | 0.038463105 | 0.038464003 | +0.000000898 | 7 / 10 |

All best and stopped epochs matched. Every run used soft Spearman, uniform dates,
the full causal-TOD store hash
`c90103b0f99e0017dc1303284a1ab61eca99106094227f5823ba718756d28a6b`, and
recorded `test_accessed=false`.

The fixed-trajectory discovery campaign from commit
`e22dd671305f30069ff2da4aafc50c1eb521cb51` is stored at:

    /lambda/nfs/brazil-rv-east3/quant-data/b3/processed/model_runs/trajectory_discovery_e22dd67_20260819T134332Z

It contains six completed runs, 120 epoch checkpoints, and 126 prediction files.
Fold A fit 512 dates through 2023-08-31 and selected on the next 102 dates through
2024-01-31. Fold B fit 614 dates through 2024-01-31 and selected on the final 102
training dates through 2024-06-28. These remain screening folds because the stored
causal TOD profile adapted inside the historical training dates.

| Rule | Fold A ensemble IC | Fold B ensemble IC | Mean |
|---|---:|---:|---:|
| Final raw | 0.043416 | 0.049602 | 0.046509 |
| Final EMA-0.98 | 0.043522 | 0.049681 | 0.046601 |
| Final EMA-0.99 | 0.043826 | 0.049905 | 0.046866 |
| **Final EMA-0.995** | **0.045309** | **0.050625** | **0.047967** |
| Last-3 weight average | 0.043438 | 0.049902 | 0.046670 |
| Last-5 weight average | 0.043628 | 0.050146 | 0.046887 |
| Tail-3 prediction average | 0.043458 | 0.049932 | 0.046695 |
| Tail-5 prediction average | 0.043661 | 0.050187 | 0.046924 |
| Patience-3 raw (same-window, selection-biased) | 0.049576 | 0.054145 | 0.051860 |
| Retrospective best raw (diagnostic) | 0.049382 | 0.054145 | 0.051763 |

The same-window Patience result above is not a deployed-value estimate: each seed's
checkpoint was selected and reported on the same 102-date window.

The corrective artifact is:

    /lambda/nfs/brazil-rv-east3/quant-data/b3/processed/model_runs/trajectory_crossfit_3054228_20260819T161200Z

It selected on odd dates and reported on even dates, then reversed the roles. Raw
Patience scored `0.048416`/`0.050673`, mean `0.049545`, versus final EMA-0.995 mean
`0.047967`. EMA-0.995 Patience scored `0.048518`; last-10 and last-7 raw weight
averaging scored `0.048060` and `0.047352`. The raw weight-average sequence was
strictly monotone across last-3/5/7/10 (`0.046670`, `0.046887`, `0.047352`,
`0.048060`) and had not saturated at the longest tested tail. The outer rule
replay chose raw Patience in three of four directions and EMA Patience once, with
mean out-of-half IC
`0.048897`. Raw Patience-3 is frozen: minimum improvement `0.0001`, patience three,
maximum 20 epochs, restore best raw checkpoint. Its Fold-B paired advantage over
final EMA-0.995 was effectively zero and all paired block intervals included zero,
so this is a numerical freeze rather than established dominance.

The fold contrast reflects different post-peak declines: Fold A final raw fell
about `0.0050` below cross-fitted Patience, while Fold B final raw was only about
`0.0011` lower. Both folds placed the coherent benefit at 120 minutes and were
slightly negative at 30 minutes. With two folds this cannot be attributed to
regime distance versus fit-window length, so checkpoint selection was frozen
rather than probed further.

One predeclared no-retraining refinement then averaged five raw checkpoints around
each parity-selected Patience peak. It scored `0.046655`/`0.050385`, mean
`0.048520`, versus raw Patience mean `0.049545`; centered-minus-raw-Patience was
`-0.001761` on Fold A and `-0.000288` on Fold B and was negative in all four
out-of-half directions. The centered rule was rejected and no window sweep was
run. Official validation and test were not accessed. Historical reproduction is:

    evaluator commit 381dcb7491b26f1e34d4ecdef75d0e5e291b5441
    /lambda/nfs/brazil-rv-east3/quant-data/b3/processed/model_runs/trajectory_centered_crossfit_381dcb7_20260819T170100Z

The evaluator was removed from current HEAD under the deletion-first rule. Raw
Patience-3 remains the frozen trajectory rule, and the checkpoint-rule line is
closed.

The strict paired analyzer compared the selected rule with final raw. Fold-A and
fold-B deltas were `+0.001892` and `+0.001024`. Moving-block 95% intervals were
`[-0.000126, 0.003693]` / `[-0.000016, 0.003627]` at block lengths 5/10 for fold A
and `[-0.000087, 0.001816]` / `[-0.000078, 0.001568]` for fold B. Horizon deltas
were positive at 30/60/120 minutes: `+0.000719/+0.001487/+0.003471` in fold A and
`+0.000659/+0.000888/+0.001525` in fold B. Time-of-day deltas were mixed: 8 of 55
were negative in fold A (range `-0.001888` to `+0.006399`) and 17 of 55 in fold B
(range `-0.001263` to `+0.003981`).

EMA-0.995 seed-prediction correlations ranged 0.909-0.914 in fold A and
0.928-0.932 in fold B. Uniform rank ensembling gained `+0.001048` and `+0.000981`
versus the mean member, respectively; no ensemble weights were learned. Official
validation was not accessed by this campaign, and the held-out test remains sealed.

## Code map

- `research/src/brazil_rv/preprocessing/build.py`: self-contained full-TOD store.
- `research/src/brazil_rv/preprocessing/intraday_normalization.py`: causal TOD
  profile and equity dynamic correction.
- `research/src/brazil_rv/modeling/data.py`: sidecar-free loader, masking, patches,
  and the two expanding internal screening folds.
- `research/src/brazil_rv/modeling/layers.py`: causal residual TCN block.
- `research/src/brazil_rv/modeling/model.py`: peer-free shared TCN and fixed
  context-plus-pooled fusion.
- `research/src/brazil_rv/modeling/engine.py`: compiled soft-Spearman/SAM training
  and eager validation.
- `research/src/brazil_rv/modeling/trajectory.py`: EMA, tail averaging, checkpoint,
  frozen raw-Patience, and frozen-rule helpers.
- `research/src/brazil_rv/modeling/train.py`: one fixed 20-epoch trajectory with
  raw and EMA artifacts at every epoch.
- `research/src/brazil_rv/modeling/analyze.py`: strict alignment, uniform rank
  ensembles, paired bootstraps, guardrails, and fixed-rule baseline selection.
- `research/src/brazil_rv/modeling/crossfit.py`: bidirectional odd/even checkpoint
  and rule selection plus immutable last-7/last-10 prediction extensions.
- `research/src/brazil_rv/modeling/run_discovery_campaign.py`: exact two-fold,
  three-seed internal screen; it cannot access official validation or test.
- `research/src/brazil_rv/modeling/evaluate.py`: standalone validation/test
  evaluation for official runs carrying an internally frozen rule.

## Operational handoff

The 2026-08-19 parent reproduction and trajectory screen used Lambda instance
`de1f90e39e204d1aa10f6a00677ad0f4` at `192.222.59.14`. After the NFS artifacts
and lockbox flags passed audit, termination was accepted and a subsequent provider
query no longer listed the instance. Persistent results remain on the
`brazil-rv-east3` NFS filesystem. The older final two-run campaign used instance
`1408116f8e794a4baa1962d512e80d6c`; its host state is historical and must not be
inferred from this record.

Do not evaluate the held-out test split, overwrite raw data, mutate immutable
feature stores, or update a canonical pointer until the corresponding audit passes.

## Completed Phase A representation campaign (2026-08-20)

Campaign commit `732b1b0e7dd870d9ea210c7b2eb750a624f12fb7` tested six
zero-start residual candidates on the two internal folds. Each candidate completed
both folds, seeds 11/29/47, and 20 epochs: 120 checkpoints per candidate and 720
total. Primary reporting used separately replayed odd/even cross-fitted raw
Patience-3 for candidate and parent; final EMA-0.995 was the free secondary
readout. Prediction ensembles were uniform rank averages and learned no weights.

| Candidate | Patience Fold A / Fold B / mean delta | EMA-0.995 Fold A / Fold B / mean delta |
|---|---|---|
| Decision time | -.000017 / -.000002 / -.000010 | -.000180 / -.000088 / -.000134 |
| Temporal stats | +.000098 / -.000307 / -.000104 | +.001688 / -.001669 / +.000009 |
| Multi-depth stats | +.000532 / -.000827 / -.000148 | +.001408 / -.003358 / -.000975 |
| Cross-sectional max/min | -.000498 / +.000101 / -.000198 | +.001691 / -.000146 / +.000772 |
| Learned set pool | -.000009 / -.000003 / -.000006 | +.000002 / +.000009 / +.000005 |
| Conditional bucket means | -.000363 / -.000035 / -.000199 | +.001752 / -.000902 / +.000425 |

All six primary means were non-positive. The two significant-looking Fold-A EMA
effects for max/min and conditional buckets reversed on Fold B. Horizon and TOD
guardrails were mixed, and no candidate warranted official-validation access.
Reject all six standalone candidates; raw Patience-3 remains the parent.

The completed campaign is stored at:

    /lambda/nfs/brazil-rv-east3/quant-data/b3/processed/model_runs/phase_a_732b1b0_20260819T180348Z

The manifest is `status=completed`, references the causal feature-store hash
`c90103b0f99e0017dc1303284a1ab61eca99106094227f5823ba718756d28a6b`, and records
`official_validation_accessed=false` and `test_accessed=false`.

Deletion-first cleanup removed the six candidate implementations, generic variant
plumbing, `modeling.phase_a`, and candidate tests from current HEAD. The experiment
commit and immutable artifact preserve exact reproduction. The strict analyzer's
observation-level comparison entry point was retained because it is generally
useful for future cross-fitted campaigns.

The Phase A instance was `df8326b7265845bf8285546d9018ed86` in `us-east-3`.
After results and repository state were safely recorded, termination was accepted;
a subsequent provider query reported the exact instance ID absent. Persistent
campaign results remain on the `brazil-rv-east3` NFS filesystem.

Final verification passed Ruff and full Python syntax compilation. Before the
building reset, commit `732b1b0` passed all 192 research tests and compiled BF16
real-store smoke checks for all six candidates on the GH200. The post-reset local
full-suite rerun did not collect tests because Windows Application Control blocked
`torch.dll` with `WinError 4551`, including from a fresh isolated `uv` environment.
This is an environment-policy failure, not a test assertion. As a compensating
check, every deletion-first model/training/test file byte-matched the previously
tested parent and the retained analyzer byte-matched commit `732b1b0`. Re-run
`uv run --project research pytest` after the Windows policy is cleared; do not
represent the post-reset attempt as a passing suite.

## Phase A autopsy, diversity ensemble, and decision-time closure (2026-08-20)

A checkpoint autopsy showed that the near-zero decision-time and learned-set
scores were not dead-adapter artifacts. Across Fold A/Fold B and seeds 11/29/47,
the historical decision-time projection ended at L2 `0.319-0.355`; the learned-set
final projection ended at `0.490-1.107`, and both learned-set `phi` layers moved.
Candidate/parent prediction Spearman remained `0.999134-0.999440` for decision
time and `0.999626-0.999890` for learned set. Learned set had already used
standard `phi` initialization, zero-only final projection, and the incumbent
nonlinear shared fusion. Both paths were active but contributed almost no new
cross-sectional ordering.

Saved predictions then supported two no-training diversity ensembles:

| Uniform rank ensemble | Patience Fold A / Fold B / mean delta | EMA Fold A / Fold B / mean delta |
|---|---|---|
| Parent-3 + multi-depth-3 | +.001237 / +.000284 / +.000761 | +.002562 / +.000739 / +.001651 |
| Parent-3 + multi-depth-3 + temporal-3 | +.000942 / +.000230 / +.000586 | +.002398 / +.000135 / +.001266 |

Every direction was positive, but every fold-level block interval included zero.
Adding temporal members diluted the six-member pool. Retain parent+multi-depth as
the sole Phase A diversity candidate for sparse official-validation confirmation;
do not learn weights, do not add temporal members, and do not treat it as proven
until that confirmation. Raw Patience-3 on the parent remains the base for new
representation experiments. The immutable reanalysis is:

    /lambda/nfs/brazil-rv-east3/quant-data/b3/processed/model_runs/phase_a_autopsy_d237998_20260820T111500Z

The remaining decision-time routing objection received one corrected rerun. The
candidate used a standard-initialized `2 -> 16 -> 16` GELU decision embedding and
a zero-only `16 -> 128` projection into shared mean/dispersion context before the
existing nonlinear fusion. Adapter construction preserved the parent's RNG state;
exact parent weights and predictions matched at epoch zero. A 10-step rank-loss
test confirmed both the projection and upstream embedding changed. The exact
experiment commit passed all 188 research tests on the GH200.

| Readout | Fold A delta | Fold B delta | Mean |
|---|---:|---:|---:|
| Cross-fitted raw Patience-3 | -0.000001432 | -0.000008622 | -0.000005027 |
| Final EMA-0.995 | -0.000000824 | -0.000000765 | -0.000000795 |

All bootstrap intervals included zero; horizon/TOD deltas were at noise scale.
The final projection reached L2 `0.299-0.992` and both embedding layers moved in
every run. Decision time is therefore an active, route-corrected null and the line
is closed without official-validation access. The campaign contains six completed
20-epoch trajectories and 120 checkpoints:

    implementation 9828f7219efbda1cb3d9aef89217423bd7e65feb
    provenance fix b8d955a71a0c6a20be0861d4a6bfd2330d1da65b
    /lambda/nfs/brazil-rv-east3/quant-data/b3/processed/model_runs/decision_time_fusion_b8d955a_20260820T113924Z

Its manifest is `status=completed`, records the exact final commit, and has
`official_validation_accessed=false` and `test_accessed=false`. Deletion-first
cleanup removed the rejected adapter, variant plumbing, driver, and specific
tests; exact reproduction uses the commits and immutable artifact. The generic
analyzer now supports candidate and parent ensembles with different member counts
without weakening strict alignment or uniform-rank requirements.

The paid instance for these follow-ups was
`d09de0143ed64f2f929f117e1b68727d` in `us-east-3`. Termination and provider
absence must be recorded only after they are verified.
