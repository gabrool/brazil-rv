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
- Exact historical reproduction still uses the recorded commit
  `4067962f6bb6748a530814d10e20dfc865a385c7`, immutable store identity, and run
  manifest.
- The official validation split is reserved for sparse stage-winner checks; the
  held-out test remains accessible only through the standalone frozen-rule evaluator.

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
  diagnostic early-stop, and frozen-rule helpers.
- `research/src/brazil_rv/modeling/train.py`: one fixed 20-epoch trajectory with
  raw and EMA artifacts at every epoch.
- `research/src/brazil_rv/modeling/analyze.py`: strict alignment, uniform rank
  ensembles, paired bootstraps, guardrails, and trajectory-rule selection.
- `research/src/brazil_rv/modeling/run_discovery_campaign.py`: exact two-fold,
  three-seed internal screen; it cannot access official validation or test.
- `research/src/brazil_rv/modeling/evaluate.py`: standalone validation/test
  evaluation for official runs carrying an internally frozen rule.

## Operational handoff

The final two-run campaign used Lambda instance
`1408116f8e794a4baa1962d512e80d6c` at `192.222.50.69`. After the campaign completed,
the host stopped answering SSH during this documentation pass. Do not assume it is
still running or terminated; check Lambda state before launching or billing another
instance. Persistent results belong on the `brazil-rv-east3` NFS filesystem.

Do not evaluate the held-out test split, overwrite raw data, mutate immutable
feature stores, or update a canonical pointer until the corresponding audit passes.
