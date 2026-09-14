# Post-data stages A–C: training, relational learning and matched financial screen

**Work in progress. Stages A and B are accepted; Stage C is running.** This document will be completed with the matched financial screen and recovery evidence. Calibration selection IC is not the screen's financial result.

## Purpose and fixed research boundary

The user authorized A–C of [the post-data plan](v2_POST_DATA_RESEARCH_PLAN.md): repair the training/selection machinery, calibrate optimization on fit/selection data, then compare fresh models on matched development folds. The [registration](../research/preregistrations/v2_post_data.md) translates that plan into executable choices. The optional LLM postmortem informed the prior review; it was not treated as an authority over the repository or evidence.

The financial run is frozen at training commit `368fbf7`, root `v2_post_data_368fbf7_20260914T004400Z`, and frozen-design SHA-256 `d5fecc8496dda7fc0a2b2b51a3e7642bfb2399d7670bc80dcc8391f699f87ed0`. Later report commits must not silently replace that training implementation.

Inputs resolve through [the accepted data pointer](v2_data_inputs.json): store `v2_data_store_2d59b9d_20260913T211700Z`, manifest `61149d43a23fc55fcd92b43aaa4e74d747fbdf62f852bedbbbdb2cf18281f3cc`. The preceding [data repair](v2_DATA_REPAIR.md) and [readiness verification](v2_next_stage_readiness.json) remain the input acceptance. This program preserves all 568,815 active stock-days, the full 60-session history, PIT identity/eligibility, input availability, target masks, splits and book policy. The 73 protected arrays remain unchanged. No raw source is modified.

There is no forward capture, 2025/2026 consumer read, deployment, automatic seed extension or Stage D–F follow-up. All evaluated history is repeatedly studied development data, not a fresh holdout. Historical source/calendar/action-term and unresolved economic qualifications remain in force.

## Stage A: implemented behavior

### Selection and optimization

The candidate trainer now evaluates complete selection dates from epoch 1 and selects their best raw checkpoint. Improvement must exceed .0001 mean common-population D3/D5/D10 IC; ties retain the earlier checkpoint. It stops after five consecutive checks without improvement, with a 60-epoch ceiling. Warm-up/cosine scheduling is fixed to the original 60-epoch-equivalent update budget, rather than changing with the eventual stopping time. P parents use the same selection-aware procedure. Unconditional late weight averaging is removed from this trainer.

The fresh incumbent S0 retains its separate accepted training/selection recipe. S0_common measures the effect of the proposed recipe on that same network. Candidate comparisons therefore do not obtain a weak baseline simply by forcing the incumbent onto unproven training settings.

The common candidate recipe uses AdamW, SAM rho .125, peak LR 1e-4, dropout .1 and weight decay .01. New candidate optimizer routing recognizes all module-owned biases, including GRU bias_ih/bias_hh, and normalization parameters as no-decay parameters. TabM factors have an explicit decay policy. The incumbent's historical routing remains part of its separate control recipe.

ASAM uses `epsilon = rho * T²g / ||Tg||`, with `T = abs(w) + .01` for adaptive weights and unit metric for bias/norm parameters. This mixed metric keeps zero-initialized outputs perturbable; it is not a claim of exact whole-network scale invariance. Parameters, optimizer moments, ranking loss and sensitive reductions stay FP32. Two-pass updates reuse the same batch and stochastic RNG and restore the exact original weights before the descent step, including failure paths. Numerical SAM and ASAM radii are not directly comparable measures of function-space strength; the diagnostics measure their actual effects.

Transferred parameters use .3 times the F LR. A family encoder with neither a valid value nor a known age anywhere in P's active fit population is explicitly unexposed and receives full F LR. Partially exposed and age-only-exposed encoders retain their learned mappings; shared fusion parameters remain transferred. This is not a blanket exclusion of late-arriving datasets or a deletion of cold columns. Parent graph, input, seed, store and preprocessing contracts must match exactly.

Every fit retains epoch histories, epoch weights, selected weights and selected optimizer state. Interrupted candidate fits can resume a bound epoch with RNG/optimizer state; compatible B fits can later attach C scores without repeating optimization. Exact B manifest receipts are retained before score attachment updates a run manifest.

### Useful, bounded diagnostics

Each epoch records training loss, SAM loss gap, gradient-clipping frequency, complete selection IC and a fixed 64-date era-spanning clean-fit probe. Full clean-fit IC is measured at selected and terminal states. These fit/selection differences compare different chronological populations; they are not pure estimates of overfitting, and zero gap is not a required success criterion.

Two fixed fit dates support restorative module probes at initialization, epochs 1/3 and selection: activation scale, weight/gradient/update norms, actual SAM score movement, FiLM gamma/beta, and peer-bypass score sensitivity. Eight attention queries from a populated historical slice provide logit scale, entropy relative to valid-key count and head dispersion. Per-head score spread and unique-score fractions expose flat/quantized outputs. Probes restore weights, optimizer, gradients, mode and RNG; hooks do not enter compiled training. No full attention matrices or activation histories are saved. Bypass sensitivity is not causal feature importance, and low attention entropy is not a success criterion.

The pooled-context attention initialization is now consistent with the explicit peer module's Xavier policy. This corrects an inconsistency; it does not prove that historical attention was disabled. The obsolete native-fundamentals builder, duplicate construction/specification/test branches and stale fixed-budget launch entrypoints were removed. Shared source-audit helpers and the original Git history remain available. Current magnitude metadata describes the accepted smooth transform; sealed historical manifests retain their bytes.

### Efficiency and acceptance

[CPU evidence](v2_post_data_cpu_engineering.json) covers actual P/F2/F14 inputs for S0/C1/TE/TL. [CUDA evidence](v2_post_data_cuda_engineering.json) verifies full 16-date, 256-slot, 60-session F14 batches: BF16 forward differences were .57–.76% of FP32 score standard deviation, with finite adaptive updates. Compiled TE/TL update timings were about 23.5/18.1 ms versus eager 49.8/38.7 ms. C1 GRU gained nothing from compilation, so C1 and S0_common use eager BF16; TE/TL use compilation. The incumbent retains its accepted compiled BF16 path. First compilation costs remain visible.

The candidate trainer materializes each finite fit/selection population's canonical FP32/bool model tensors once on the GPU. Epochs gather exactly the registered dates instead of repeatedly reconstructing overlapping histories on the CPU. [GPU cache acceptance](v2_post_data_cuda_cache_engineering.json) proves exact tensor equality for P/F2/F14, including reordered gathers and masks/ages. F14 cached gathering took .69 ms versus 228 ms for canonical collation/transfer. This is an input-stage result, not a whole-fit speedup. No lookback, name, feature, precision of cached inputs or model width was reduced.

Targeted acceptance covers adaptive perturbation/restoration, RNG-preserving diagnostics, bias/LR routing, early selection, exact score attachment, preprocessing and cache parity. Earlier test groups overlap and should not be added into an artificial total. Stage A was checked against the reference plan after implementation and again after the cache amendment. Financial jobs run four processes on the GH200; independent C book folds also run in four processes while every individual ledger remains chronological.

## Stage B engineering: what learned, what failed

The [complete attempt record](v2_post_data_engineering_attempts.json) preserves all **54 attempts**, rosters and source hashes. All 178 engineering files, totaling 44,516,079 bytes, were recovered and hash-verified locally. Final confirmation weights are retained for additional readouts without retraining.

These are full TE models with the actual rank loss and SAM/ASAM updates, 60 historical steps, 32 recipient slots plus 16 donor slots, variable active universes, masks and missing histories. Every update receives 16 new generated dates, and evaluation uses 128 independent generated dates. Queries, keys, messages and regimes are supplied as observable channels; no hidden relation IDs are supplied to the model.

| Test | Result and interpretation |
| --- | --- |
| Initial 512-update SAM .125 bracket | The original “own” fixture passed but encoded only current information; it did not establish history retrieval. Dynamic peer, lagged and context tasks failed. |
| Short optimizer and persistent-message controls | Changing the initial optimizer/LR or making messages persistent did not resolve short-budget failures. These failed attempts remain in the record. |
| Original dynamic peer, ASAM .2, 4096 updates | IC .7783/.7827 for seeds 11/29; own-only .0204/.0107. Uniform interaction .6860/.7819 also worked, so this task alone did not establish learned-attention superiority. |
| Corrected true own-history task | Observed recipient values at t−21 and t−41 gave IC .9951/.9968. Seed11 also passed with an FP32 final head and full FP32. |
| Original lagged/context, unlabelled donors | Failed under the tested budgets/settings. FP32, a current-core bypass, lower rho, changed LR and ordinary AdamW did not resolve the lagged failure. These are still failures. One flat-output AdamW attempt also exposed a JSON NaN failure; the harness now retains null IC and its defined-date count. |
| Jointly supervised lagged, ASAM .2, 4096 updates | **.9475/.9557** versus uniform −.0110/.3438; own-only −.0278 on seed11. |
| Jointly supervised context, ASAM .2, 8192 updates | **.8601/.8573**, versus uniform .0131/.0135. Current-message regime .7587/.7541; historical-message regime .9616/.9513. Both regimes learned on both seeds. The shorter 4096-update own-only control was −.0002 on seed11. |

The joint-supervision diagnosis preserves **every input and every recipient target exactly** and adds only donor stocks' own observed-signal labels. Evaluation still uses recipients only. Its own-only control bypasses both stock mixers while preserving donor inputs and their own supervision. This isolates the difficulty of learning source histories solely through indirect recipient gradients from joint temporal learning. The supervision and budget amendments were made after failures and are explicitly disclosed, not retroactively presented as the initial protocol.

The accepted conclusion is conditional: the full architecture and adaptive rank-training procedure can learn own history, cross-stock history and context-conditioned relations under these tested conditions. The original unlabelled history tasks did not pass. This does not certify learning from unlabelled macro histories, prove financial alpha, validate every optimizer recipe, or establish that attention is universally superior. Financial labels are unchanged, and financial optimizer choice is determined by the registered fit/selection comparison, not synthetic IC.

## Financial roster and interpretation of contrasts

| Cell | Inputs and architecture | Parameters |
| --- | --- | ---: |
| S0 | Fresh incumbent slow GRU, accepted recipe | 120,902 |
| S0_common | Same S0 network, proposed common recipe | 120,902 |
| C1_slow | Slow GRU and characteristic trunk | 1,297,475 |
| TE_slow | Temporal attention → per-time stock attention → temporal pooling | 1,327,107 |
| TL_slow | Matched temporal attention → pooling → stock attention | 1,327,107 |
| C1_family | GRU plus cleaned per-name families, FiLM off | 1,417,859 |
| C1_all | Same, with common-state FiLM | 1,459,651 |
| TE_family | Early attention plus per-name families, FiLM off | 1,447,491 |
| TE_all | Same, with common-state FiLM | 1,489,283 |

All use the complete 60-session window and 32 slow fields. Rich models additionally receive 72 per-name fields across ten families. FiLM-on arms receive 44 common fields: 41 common cross-market fields plus three availability diagnostics. Values, masks and known ages follow the accepted preprocessing contract. “FiLM off” is a registered contribution contrast, not a proposal to remove common data from every model.

S0 and S0_common retain five trained horizon heads (1/2/3/5/10), while the characteristic C1/TE/TL networks train 3/5/10. Financial selection and the principal paired comparison use common D3/D5/D10 populations for every cell. Thus **TE versus TL is a matched timing comparison**, whereas S0 versus C1 is a broader model-package comparison, including capacity/readout and auxiliary-head differences. It must not be described as a pure attention or pure encoder ablation. No head/target experiment was silently added during this run.

Stage B calibrates TE_slow and C1_all on F2/F14, seeds 11/29, under SAM .125, ASAM .2/.5, SAM .125 with LR3e-4 and SAM .05. A transfer-LR multiplier 1.0 bridge is limited to F14/seeds 11/29 and cannot win the four-fit recipe comparison. Each lane uses one selected SAM .125 P parent per seed across its F recipes. This isolates F optimization conditional on those parents; it is not a study of ASAM pretraining.

Stage C uses F2/F6/F10/F14 and seeds 11/29/47: twelve F fits per cell. TE/TL variants share the attention lane's selected recipe; C1 variants share the rich-GRU choice. S0_common retains common SAM .125, and S0 retains its own recipe. The complete A–C financial roster requires 27 P fits and 144 distinct F fits; eight B fits can attach C scores without retraining. No fourth seed or extra development fold is silently added.

Calibration deliberately uses early and late development selection periods, then maps one recipe across the screen. This is a retrospective development comparison, not a claim that the tuned recipe would already have been chosen at each early historical trading date. Features, preprocessing and fitted weights remain bounded to each fold's permitted history; global research selection is a separate source of optimism that a four-fold screen cannot eliminate.

## Stage B financial result

All 4 P and 44 F runs completed; none hit the 60-epoch ceiling. The complete runner took 29m20s, including initialization, preprocessing, caching, compilation, fitting, selection, diagnostics and sealing. Audit and recovery time are additional. [Calibration choices](v2_post_data_calibration_choice.json), [full trajectories and probes](v2_post_data_calibration_diagnostics.json) and [selected-state precision](v2_post_data_calibration_precision.json) provide the machine-readable evidence.

| F recipe | TE_slow mean selection IC | C1_all mean selection IC |
| --- | ---: | ---: |
| SAM .125 | .023561 | .030038 |
| ASAM .2 | **.030505 — selected** | .033728 |
| ASAM .5 | .031099 | **.036144 — selected** |
| SAM .125, LR3e-4 | .022148 | .034140 |
| SAM .05 | .026627 | .027907 |

Each mean weights the same four fold/seed fits equally. The declared .001 tie band chooses ASAM .2 for attention, even though .5's point estimate is slightly higher. The C1 choice is .5. These are development selection outcomes, not proof of financial improvement or architecture superiority.

For selected attention ASAM .2, mean full clean-fit IC was .068825 at selection and .088723 at termination. For selected C1 ASAM .5 it was .056219/.070841. Across all 48 P/F trajectories the largest selected/terminal full-fit IC was .132271/.153556. The earlier .8-type fit trajectory is not reproduced within this bounded program; this does not isolate ASAM as the only cause or establish that every form of overfitting has been solved. P selections varied from epoch4 to34. F2 selected attention at19/15 and rich GRU at13/22; F14 selected at1/2. Weak F14 selection remains a real diagnostic finding.

The transfer multiplier1.0 bridge has only two F14 fits. For attention, its mean .009080 is below matched SAM .125/.3 transfer's .010812. For C1 it is .006472 versus .006674. It does not support increasing transfer LR and cannot enter the four-fit recipe ranking.

Selected temporal/peer parameters receive nonzero updates. Attention peer-bypass score movement was .431–1.036 of clean score standard deviation across the four selected fits; sampled peer entropy/log-key-count was .760–.973. C1 FiLM gamma RMS was .187–.692. These indicate active routes, not useful financial contribution. Whole-trajectory gradient clipping was at most8.01% under selected TE ASAM .2 and .214% under selected C1 ASAM .5, rather than persistent clipping of every step.

BF16 score ties prompted a bounded precision readout on all selection dates of the eight selected calibration fits. Score-rank correlation against FP32 exceeded .99984; centered RMS error was .246–.759% of cross-sectional score standard deviation, and the largest selection-IC difference was .000285. An FP32 final head also made only small differences. Eager-versus-registered compiled BF16 selection differences for TE were below .0001. Keep the frozen precision: the observed ties do not justify changing the financial model. The diagnostic's initial dataset-close typo and subsequent successful correction are retained in the log; no training run was changed.

Stage B was verified against the reference plan before Stage C. C uses six isolated jobs, within the measured roughly10.44GB-per-F14-fit allocation and97.9GB GPU capacity, to reduce preparation/compilation gaps. This is a concurrency choice with unchanged batches, histories, names, parameters and recipes.

## Stage C financial results and closure

Pending completion of Stage C. Final sections will include paired IC and book economics, seed/fold and omission sensitivity, resource timing and complete artifact recovery.

The economic rebind already passed: all four hedge-beta arrays are exactly unchanged under the repaired-store provenance. Ledger costs, borrow scenarios, execution policy and terminal qualifications remain fixed. C will report per-head and actual book-composite alignment, neutral/scaled/shareholder/price targets on matched populations, and persistence. Paired uncertainty uses within-fold 20-session moving blocks, a 60-session sensitivity, 10,000 draws and seed 20260913. Legacy helper readouts with other head definitions are auxiliary and cannot replace the explicit traded-head comparison. The intervals are nominal development diagnostics, not multiplicity-adjusted confirmation.
