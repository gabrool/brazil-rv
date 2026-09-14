# Post-data stages A–C: training, relational learning and matched financial screen

**All numerical work for A–C and the user-requested ASAM .5 extension is complete.** This review separates engineering, calibration and matched financial evidence; artifact recovery and shutdown are recorded at the end.

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

Targeted acceptance covers adaptive perturbation/restoration, RNG-preserving diagnostics, bias/LR routing, early selection, exact score attachment, preprocessing and cache parity. Earlier test groups overlap and should not be added into an artificial total. Stage A was checked against the reference plan after implementation and again after the cache amendment. Stage B financial jobs ran four processes on the GH200; Stage C uses six. Independent C book folds run in four processes while every individual ledger remains chronological.

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

Stage B was verified against the reference plan before Stage C. C uses six isolated jobs, within the measured roughly 10.44 GB per F14 fit and 97,871 MiB (95.6 GiB) GPU capacity, to reduce preparation/compilation gaps. This is a concurrency choice with unchanged batches, histories, names, parameters and recipes.

## Stage C financial results and closure

The user subsequently requested ASAM .5 attention in addition to .2. The
[explicit extension](../research/preregistrations/v2_post_data_attention_asam50.md)
adds TE_slow, TL_slow, TE_family and TE_all at .5 on the same four folds and
three seeds. It preserves identical parents and every other training setting.
Four B fits are reusable, so this adds 44 new F fits and no parents: the combined
program now has 27 distinct P fits, 188 distinct F fits and 13 scored C cells.
The original calibration choice and original screen remain intact. Both radii,
their paired differences and the additional development-search qualification
will be reported; no winner is chosen in advance.

An actual-plan acceptance compared all 48 extension jobs with their .2 sources:
every parent/training argument was preserved, only rho changed among recipe
parameters, output paths were distinct and the original plan hash was unchanged.
Four targeted readout tests passed, including preservation of original score
sources. The extension runs behind the original GPU screen at concurrency six
and can overlap its CPU book readouts. A separate clean checkout supplies the
extended evaluation code while the training checkout remains frozen.

## Stage C results: original screen and the .5 extension

All 27 parents and 188 distinct F fits across B/C/extension completed. Unsuffixed TE/TL cells use ASAM .2; the `_asam50` suffix identifies .5. All C1 cells use their selected ASAM .5 recipe. The matched C comparison contains 156 scored F fits, 13 cells and 52 three-seed books. [Combined results](v2_post_data_screen.json), [original-screen results](v2_post_data_screen_original.json), [all C trajectories and module probes](v2_post_data_screen_diagnostics.json), and [the explicit radius amendment](v2_post_data_attention_amendment.json) retain the evidence.

The principal IC is the equally weighted D3/D5/D10 head IC, calculated on the same eligible, outcome-valid names and averaged over finite evaluation dates. All cells score all 89,298 active evaluation stock-days; the shared three-head outcome population contains 81,654 stock-days on 461 of 501 sessions. The remaining 40 dates are the ten terminal dates per fold without complete within-fold D10 outcomes. They remain in book accounting. This is not a loss of input observations.

| Cell | Primary IC | Net excess, bps/day | Turnover / NAV per day |
| --- | ---: | ---: | ---: |
| S0 | 0.018849 | -0.028 | 25.79% |
| S0_common | 0.022241 | +2.861 | 16.21% |
| C1_slow | 0.016782 | +2.265 | 25.52% |
| C1_family | 0.021357 | +1.804 | 26.04% |
| C1_all | 0.022809 | +0.571 | 24.31% |
| TE_slow | 0.017069 | -0.421 | 23.43% |
| TE_slow_asam50 | 0.017760 | +0.260 | 21.84% |
| TL_slow | 0.019645 | +2.682 | 21.60% |
| TL_slow_asam50 | 0.019966 | +2.203 | 20.82% |
| TE_family | 0.023499 | -1.165 | 20.39% |
| TE_family_asam50 | 0.022119 | -0.512 | 20.98% |
| TE_all | 0.027103 | +1.440 | 19.60% |
| TE_all_asam50 | 0.021804 | -0.196 | 19.84% |

The strongest observed IC is **TE_all at ASAM .2: .027103**, versus incumbent S0 .018849 and C1_all .022809. The richer inputs now contribute under the repaired data/training package. This is evidence against the earlier claim that every richer candidate necessarily loses. It does not identify which individual repair caused the improvement, establish an optimal architecture, or warrant a deployment designation.

### Paired contrasts and the radius question

Intervals below are nominal paired 95% date-block intervals with the registered 10,000 draws, seed 20260913 and within-fold boundaries. They do not correct for the repeated research history or the many comparisons.

| Contrast | IC difference, 20-session blocks | 60-session sensitivity |
| --- | --- | --- |
| TE_all_minus_S0 | +0.008254 [+0.000857, +0.015935] | +0.008254 [+0.000739, +0.013826] |
| C1_all_minus_S0 | +0.003960 [+0.000448, +0.009701] | +0.003960 [+0.000924, +0.007119] |
| TE_family_minus_TE_slow | +0.006430 [+0.000434, +0.012274] | +0.006430 [+0.000899, +0.011030] |
| TE_all_minus_TE_family | +0.003604 [+0.001375, +0.007319] | +0.003604 [+0.002007, +0.005549] |
| TE_slow_minus_TL_slow | -0.002576 [-0.006984, -0.000313] | -0.002576 [-0.006238, -0.001463] |
| TE_slow_asam50_minus_TL_slow_asam50 | -0.002206 [-0.006123, +0.000374] | -0.002206 [-0.005022, -0.000556] |
| TE_slow_asam50_minus_TE_slow | +0.000691 [-0.000721, +0.001942] | +0.000691 [-0.000696, +0.000849] |
| TL_slow_asam50_minus_TL_slow | +0.000321 [-0.003288, +0.003122] | +0.000321 [-0.003229, +0.001096] |
| TE_family_asam50_minus_TE_family | -0.001380 [-0.003685, +0.000478] | -0.001380 [-0.003533, +0.000133] |
| TE_all_asam50_minus_TE_all | -0.005299 [-0.008442, -0.003107] | -0.005299 [-0.007441, -0.003470] |

ASAM .5 gives small, uncertain improvements to slow early/late attention. It reduces TE_family IC modestly and reduces TE_all IC by .005299, with both block intervals below zero. Keep .2 as the stronger observed rich-attention setting in this comparison. Numerical rho is not itself a universal measure of functional regularization; the same radius can affect different input/graph combinations differently.

**Early mixing did not beat matched late mixing on slow inputs.** At .2, early-minus-late IC is −.002576; at .5 it is −.002206. The .5 20-session interval spans zero while its 60-session sensitivity is below zero. The successful rich TE_all arm does not isolate timing: no equally rich TL_all arm was included. A future matched rich early/late comparison is the clean way to answer that remaining question, rather than claiming this screen proved that early mixing is necessary.

### Chronological and seed stability

| Cell | F2: 2018 H2 | F6: 2020 H2 | F10: 2022 H2 | F14: 2024 H2 |
| --- | ---: | ---: | ---: | ---: |
| S0 | 0.003286 | -0.011517 | 0.026463 | 0.056305 |
| S0_common | 0.011512 | -0.009166 | 0.023141 | 0.062758 |
| C1_slow | -0.015734 | -0.009901 | 0.025337 | 0.065882 |
| C1_family | 0.000429 | -0.005343 | 0.032250 | 0.057062 |
| C1_all | 0.003547 | -0.008715 | 0.030712 | 0.064667 |
| TE_slow | 0.000732 | -0.019236 | 0.025683 | 0.060163 |
| TE_slow_asam50 | 0.004327 | -0.019243 | 0.024886 | 0.060240 |
| TL_slow | 0.006455 | -0.019427 | 0.025165 | 0.065536 |
| TL_slow_asam50 | 0.006735 | -0.018949 | 0.024533 | 0.066684 |
| TE_family | 0.011777 | -0.010009 | 0.033461 | 0.058067 |
| TE_family_asam50 | 0.011307 | -0.011763 | 0.030925 | 0.057331 |
| TE_all | 0.015898 | -0.011576 | 0.034798 | 0.068549 |
| TE_all_asam50 | 0.006092 | -0.011439 | 0.030543 | 0.061139 |

The evaluation windows are 2018-07-02–2018-12-28, 2020-07-01–2020-12-30, 2022-07-01–2022-12-29 and 2024-07-01–2024-12-30. Every model has negative F6 IC. This shared era failure is a substantive unresolved generalization issue; increasing rho does not remove it.

TE_all .2 improves on S0 in F2/F10/F14 and is essentially tied slightly below it in F6. Its individual-seed ICs are .025148/.026910/.026581; two-seed omission ICs are .027552/.026835/.026392. All three individual comparisons and all three matched omissions exceed S0. This is useful seed robustness inside these four development windows, not four independent historical replications.

S0_common has competitive .022241 ensemble IC and the highest net point estimate, but seed47 IC is only .005918 versus .024130/.022092 for seeds11/29. That inconsistency remains visible in the retained results. Do not conclude that its new recipe universally improves the incumbent from the ensemble headline alone.

### Training trajectories and module diagnostics

| Candidate | Mean selected full-fit IC | Mean terminal full-fit IC | Largest trajectory clipping fraction |
| --- | ---: | ---: | ---: |
| S0_common | 0.03247 | 0.03575 | 0.00% |
| C1_slow | 0.05014 | 0.05668 | 0.35% |
| C1_family | 0.04716 | 0.06071 | 0.43% |
| C1_all | 0.04884 | 0.06190 | 0.21% |
| TE_slow | 0.05385 | 0.07330 | 15.90% |
| TE_slow_asam50 | 0.05481 | 0.06706 | 18.70% |
| TL_slow | 0.05757 | 0.08018 | 36.92% |
| TL_slow_asam50 | 0.05745 | 0.07104 | 22.89% |
| TE_family | 0.05601 | 0.08378 | 22.80% |
| TE_family_asam50 | 0.05091 | 0.07279 | 7.45% |
| TE_all | 0.05714 | 0.09049 | 27.69% |
| TE_all_asam50 | 0.05422 | 0.07877 | 26.92% |

Across the C and extension candidate trajectories, the largest selected/terminal full-fit IC is .137724/.149094. Including B, the maxima are .132271 for B-selected and .153556 for B-terminal states. The old .8-type fit pathology is not reproduced. This is the combined consequence of the repaired representation, selected parents, LR/transfer treatment, optimizer and stopping protocol; it is not an isolated causal proof that ASAM solved overfitting.

Attention still shows more clipping than C1 in some runs, and terminal fit rises after selection. TE_all .2 reaches a 27.69% whole-trajectory clipping fraction in its most affected fit; TL_slow .2 reaches 36.92%. Those are reasons to retain the optimization diagnostics, not evidence that the gradients vanished or grounds for changing this completed comparison. Selected temporal/peer and FiLM probes, score resolution, functional perturbations and all epoch curves are retained in the diagnostic JSON. Activity or bypass sensitivity is not proof of incremental alpha.

One safety ceiling was reached: S0_common F2 seed11 selected epoch58 and stopped at60. Selection rose from .030091 at58 to .030187 at60, less than the frozen .0001 improvement threshold relative to the selected state. The full patience horizon was therefore censored by the ceiling. This limits the common-recipe control and deserves a declared future continuation if needed; no extra epochs were added after seeing evaluation outcomes. All other new candidate trajectories stopped under patience. The incumbent retains its separate accepted logging/selection/averaging policy.

### Objective, composite score and economic interpretation

TE_all .2 improves the neutral target but does not improve every return view: same-population shareholder-return IC is .04672 versus S0 .04677, while neutral IC is .02710 versus .01885. The added information appears more useful for the selected idiosyncratic ranking task than for improving this unneutralized aggregate statistic. This does not prove neutrality is optimal or that macro/fundamental inputs lack value.

TE_all head ICs are D3 .02474, D5 .02323 and D10 .03334. The actual registered composite achieves .02269/.02567/.03174 against those same horizon targets; S0 achieves .01863/.01963/.01727. One/five-session composite persistence is .91618/.86996 for TE_all, .91520/.84004 for S0, and .95439/.90884 for S0_common. These use the **actual book signal**, not an old four-head default.

The book policy remains `theta=1`, horizons 3/5/10, equal sizing and `buffer_per_quintile=9`. All hedge-beta arrays were exactly unchanged under the repaired-store provenance. No cost, borrow or execution-policy change was used to rescue a model.

TE_all versus S0 net excess is +1.468 bps/day, with paired 20-session interval [−2.287, +6.294]; the economic advantage is not established. TE_all versus TE_family is +2.606 [+.637, +5.169] bps/day, but these are nominal development contrasts. TE_all .5 versus .2 is −1.636 [−4.724, +.392] bps/day. Better head IC does not guarantee a better concentrated, cost-bearing portfolio. Initial and terminal trades enter turnover; the retained interval note warns that non-circular blocks underweight boundary spikes.

All 52 books pass the existing bounds and carry economics_unresolved=false **under the frozen settlement convention**. That is not a certificate of exact historical liquidation prices. Several books require terminal settlement, and the source, action-term, calendar, borrow and settlement assumptions remain consequential. The per-book settlement notional and haircut scenario are retained in the results; this screen does not establish tradable profitability.

### Final diagnostic correction, throughput and closure

The final review caught an auxiliary composite mismatch: the initial alignment report called a helper with the historical 1/2/3/5 default, making composite/persistence diagnostics undefined for three-head candidates and mismatching the incumbent book signal. The books themselves already used the registered policy. The canonical alignment now calls the same policy-specific signal as the ledger. Five targeted tests passed, including a three-head masked fixture and a smoothed policy. [The repair proof](v2_post_data_alignment_repair.json) verifies all 52 books, score files, accepted records, daily readouts and seed results exactly unchanged; only composite IC/persistence and available-to-book diagnostic populations were refreshed. Previous result/alignment receipts are preserved inside the recovered run. Primary model IC and every reported economic result above are unchanged by this repair.

B’s complete 4 P + 44 F run took 29m20s. The remaining-parent C phase took 17m06s including four reuses; the original 108-score screen took 65m30s; its four-fold book/report pass took 5m07s. The added 48-score .5 phase took 19m15s with four reused fits, followed by roughly 6m52s for its combined readout pass. Different rosters and compilation reuse make those durations incomparable as isolated architecture speedups. Recovery and final verification overlapped computation where possible and are additional work.

The reference plan was revisited after A, B and the completed C numerical screen/extension. A implements the intended contracts; B records both failed and conditionally successful engineering plus selection-only calibration; C now supplies the matched timing, family, FiLM, radius, seed, objective-alignment and unchanged-book evidence. No Stage D–F experiment was launched.

The next justified research questions are a matched rich early/late comparison, the shared F6 generalization failure and targeted parent/fusion or objective follow-ups informed by these curves. The present evidence supports retaining TE_all .2 as a promising development candidate alongside S0/S0_common and C1_all, while preserving S0 as the established comparator. It does not authorize automatic promotion or a broader search.

### Recovery and instance closure

[Recovery verification](v2_post_data_recovery.json) confirms **6,796 final files / 18,571,092,654 uncompressed bytes** recovered in lossless local archive chunks across C: and D:. Every archive and every member was SHA-256 verified; applying the chunks in recorded order reproduces the exact [final remote inventory](v2_post_data_final_inventory.json). The financial root remains on persistent Lambda storage. This is a verified archive recovery, not a claim that all models were also unpacked locally. The published result copies match that final inventory exactly. The separately recovered engineering evidence contains 178 files / 44,516,079 bytes.

To make room for recovery, an abandoned 1.27 GB synthetic memory-test fixture was identified against its generating test and synthetic price/volume/trade formulas, then removed. Canonical inputs and research checkpoints were preserved. The cleanup receipt is retained with the recovered operations evidence.

Exact-instance shutdown is the remaining operational closure step and will be recorded after provider confirmation.
