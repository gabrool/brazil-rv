# Research decision after the portfolio-objective experiments

The user subsequently authorized all five foundation workstreams. Implementation
and outcome-independent gates are in [the registration](../research/preregistrations/v2_foundation.md),
with [progress](v2_foundation_progress.md) and [canonical run pointer](v2_foundation_run.json).
The assessment below remains the rationale; later results belong in the progress
record and combined report.

2026-09-18, updated after the user's input-selection, architectural-synthesis, variance-reduction and scaling requests. Assessment and staged experimental design; financial runs still require a frozen implementation/registration. The completed financial source is 33086be; its final review and recovery were published in 193fe4d. No new model fitting, portfolio replay or held-out access was performed for this assessment. Descriptive forecast/account comparisons are in [the evidence file](v2_next_research_evidence.json); the new read-only input and parameter census is in [the model-design audit](v2_model_design_audit.json).

## 1. Decision

**Pause broad end-to-end P&L, execution-controller and checkpoint-selector searches. Continue research on incremental predictive information and how the model represents it.** Preserve the corrected allocator derivative and all accounting/efficiency improvements. Do not replace ranking training or the established references with the unsuccessful continuation candidates.

There is one bounded exception: the older learned-controller conclusion deserves a corrected-gradient check, because a demonstrated numerical defect affected that branch. This is a narrowly scoped closure experiment, not justification for another large policy search.

The highest-value *alpha* question is now: **Can the existing rich datasets add stable information beyond a strong predictor when their contribution is learned separately, without having to relearn or disturb the whole predictor?** The most urgent *validity* work is resolving the material settlement and financing assumptions. These are different priorities: one seeks better predictions; the other establishes what existing backtest returns mean.

The user's follow-up adds a prerequisite foundation pass: identify the useful input roster, measure variance reduction, disentangle existing architectural components, and test where capacity belongs. **Do this before new datasets, new feature ideas or new history/attention mechanisms.** Section 9 specifies that pass. The old-controller retry and section 4.4's new history experiment are deferred behind it; they are not part of its first GPU wave.

## 2. What the experiments establish

| Finding | Evidence | Decision consequence |
|---|---|---|
| Direct utility continuation did not earn promotion | Attention screen +.843 net bps/day; ten-fold confirmation -.177, nominal 40-session interval [-2.467, 2.166]; only one of three seeds improves utility | Close this registered recipe. The interval is too wide to prove equivalence or universal inferiority |
| Economic checkpoint selection was less reliable here | Confirmation net: rank/IC 5.323 vs rank/economic 5.005; utility/IC 5.490 vs utility/economic 4.828 bps/day | Retain IC selection as the working reference. Do not choose the best selector after seeing evaluation results |
| Auxiliary return supervision was also insufficient | Earlier confirmation: attention +.086 bps/day, C6 -.056; intervals span zero | Do not assume another generic loss substitution is the highest-value next step |
| Learned timing has not demonstrated a dependable edge | Both original and stronger-shrinkage supervised opportunity probes fail their chronological predictive gates | Bad historical halves are not, by themselves, observable signals telling the model to hold cash |
| Dollar neutrality and market neutrality are different constraints | Prior continuous C6: 4.000 neutral vs 6.707 flexible-net; attention: 4.497 vs 5.937 | Keep flexible net with tight beta as a policy candidate; do not describe it as learned market timing |
| The allocator backward had a real defect | Wrong-sign multiday derivatives; repaired active-face derivative agrees with historical finite differences | Qualify earlier gradient-trained controller failures; retain their unchanged forward books as historical results |
| Attention is a functioning candidate | Post-data work demonstrated conditional relational learning and a stronger rich-input IC screen; later accounts have positive development returns | No basis to discard attention or insist it must dominate C6 |

Primary evidence: [direct objective review](v2_PORTFOLIO_OBJECTIVE.md), [Phases 1–3](v2_DECISION_PHASES123.md), [opportunity study](v2_OPPORTUNITY_PORTFOLIOS.md), [post-data A–C](v2_POST_DATA_ABC.md).

The latest uninterrupted neutral accounts remain around **4.23–4.60 bps/day above CDI**, rather than the 1–2 bps/day seen on the four screen folds. The earlier roughly six-bps results had more flexible dollar exposure. Different dates, account boundaries and exposure constraints explain much of that apparent discrepancy. They must not be used interchangeably.

For the latest continuous economic-selector comparison, utility training gives 4.323 vs ranking's 4.226 bps/day, but BRL/CDI Sharpe falls from .972 to .871 and BRL maximum drawdown worsens from -12.61% to -13.50%. A small positive full-history mean does not rescue the failed confirmation or establish better risk-adjusted performance.

## 3. What remains unproven

There is no established single explanation such as “attention is broken,” “IC is the wrong loss,” or “the model just needs more capacity.” The latest training has verified economic gradients into the encoder, exact account reconciliation and preserved inputs. Passing those checks establishes implementation capability, not statistical learnability.

The direct-objective experiment is deliberately narrow: an already trained ranking representation, a smooth-rank anchor plus a cardinal correction, twelve continuation epochs, truncated 32-session credit assignment and a fixed neutral allocator. It does not test all decision-focused learning, training from scratch, a separately learned market forecast, a new stateful controller or intraday intervention. The forecaster itself does not receive actual holdings as part of its representation; the allocator/account handles inventory. Calling it a complete learned execution agent would overstate what was tested.

An economic objective is closer to the final goal, but may be harder to estimate. Many stock observations share the same date and market shock; five-day labels overlap. Hundreds of thousands of stock-days are not hundreds of thousands of independent observations about whether a whole market regime is favorable. The portfolio reward also mixes selection, financing, common exposure, costs and a few unusual events. This is a plausible explanation for unstable learning/selection, not a proven attribution of the failed result.

Similarly, marginal predictive relevance of a dataset does not establish incremental, tradable information after controlling for existing inputs, dated availability and the prediction target. Low correlation between input columns does not guarantee independent information about future returns. Conversely, an individually weak feature can matter through interactions. The next study must test joint incremental information, not reject every family that lacks standalone IC.

Attention and C6 are not a pure architecture ablation: families, fusion, width and regularization differ. Descriptive checks on their sealed Phase-3 neutral forecasts give mean daily cross-sectional correlation **.701** between their average horizon ranks; the two neutral accounts' excess-return correlation is **.755**, over 1,738 sessions. This suggests potentially useful diversity, but does not establish that blending improves net returns. Correlations are not an investment result or a selection criterion.

## 4. Ranked next work

### 4.1 Resolve the economically material accounting assumptions

This is the first priority for confidence in the numbers, not a promise of additional alpha.

- Reconstruct the largest unresolved historical dispositions from official issuer/B3/CVM documents, starting with the already identified TRPL, CIEL, BRML, ALSO and CPLE-unit cases. Establish permanent-identity succession, exchange ratios, cash consideration, effective dates and payment dates where applicable. A later-known settlement amount may be an accounting outcome; it must not enter an earlier trading decision.
- Correct derived accounting only where evidence supports a correction. Keep the full security population. Do not replace missing contractual evidence with an executable last-quote exit or discard the affected name.
- Quantify how much unresolved events can change both total returns and **candidate-minus-control differences**. A common assumption need not cancel: models hold different positions.
- Determine feasible cash, collateral, short-proceeds and debit-rate treatment for the intended trading setup. Until then retain explicitly labelled scenarios; neither full CDI remuneration nor zero remuneration becomes “truth” by preference.
- Recheck the flexible-net advantage after the accounting work and compare risk/exposure as well as daily bps. Some gain remains in the rank-only comparison, but a common residual intercept also contributes. Do not attribute all of it to stock-selection skill.

This is a bounded source audit, not another indefinite general data-cleaning project. Stop the initial pass once the dominant cases have verified terms or explicit unresolved bounds. Record remaining uncertainty and whether it can reverse the comparison. The accepted numerical feature store need not be reopened absent new evidence of an input defect; an actual target/return correction may require a separately bound store and targeted reruns.

### 4.2 Test incremental signal with cached predictions and small models

This remains a high-priority test of existing information, now integrated with section 9's input and model foundation pass. It can begin on CPU while document retrieval proceeds; promotion still depends on credible accounting. It is a **revised test after data repair**, not a claim that residual trees have never been tried.

The [Round-7 CPU diagnostic](v2_ROUND7.md#supporting-cpu-evidence) already tested chronological residual trees on the older inputs: S0 composite IC .03186, score-only tree .01605, all-family tree .02038. The all-family tree improved on its tree control but failed to beat S0. That is relevant negative evidence. The implementation used fixed 500-round, 31-leaf trees without prior-selection early stopping or correction-strength selection. Its score-only deterioration makes unconditional replacement by a fully applied correction a questionable baseline. The separate linear lagged-peer probe also failed. Neither result should be erased by calling a new probe novel.

The justification for one bounded revisit is the subsequently repaired family data, the stronger current C6 anchor, explicit linear/nonlinear comparison and prior-selected correction strength with **zero correction** as a genuine candidate. Do not simply repeat the old tree settings under a new name. If those changes still fail, close the current-state corrector branch rather than continuing to tune it.

Use C6 as the strong reduced-input anchor and TE_all as the rich reference. First report the existing causal blend correctly: it was already tested, its chosen attention weight varied widely by fold, and its auxiliary-minus-neutral gain was uncertain. Do not advertise “try ensembling” as untouched ground.

Proposed compact comparison:

1. Existing causal C6 and TE_all forecasts, with the same dates, input population, calibration and allocator.
2. One fixed equal-weight combination of their forecasts before joint allocation, if the artifact inventory confirms this exact account is absent. This measures diversification without fitting another unstable blend selector. It is not the arithmetic mean of separate book returns.
3. A strongly regularized shared linear/ridge correction to the C6 prediction, using the accepted additional stock fields, masks/ages and a compact interaction between common context and the base signal.
4. One bounded nonlinear correction using the **same inputs and targets**, such as shallow histogram-boosted trees. This supplies a distinct way to find nonlinear interactions without immediately training another large temporal network.

Use the existing three primary horizons and target definition for the first incremental-prediction comparison. Keep the prediction-loss question fixed. Fit residuals only against genuinely chronological out-of-fit base predictions in compatible target units; raw rank scores cannot simply be subtracted from bps labels. If early out-of-fit history is unavailable, begin after adequate history exists and restart all matched controls on those same dates. Never substitute in-sample baseline residuals for speed.

All scalers, residual calibration, shrinkage and tree settings must be determined from earlier fit/selection periods with matured labels and the required purge. Include an unchanged-base option and a small predeclared correction-strength set. Selection of zero is an informative failure to improve, not evidence of a successful new model. Include the matched score-only correction so apparent gains cannot be attributed merely to recalibrating C6. Equalize date contributions; keep every eligible security and explicit missingness. Evaluate all added families jointly at least once. Small grouped follow-ups may locate a surviving contribution; a full leave-every-column-out search is unnecessary.

Report incremental ranking, calibration where applicable, paired portfolio net/utility, turnover and stability across periods. Do not infer tradable value from standalone IC or a feature-importance chart. If the nonlinear correction works where the linear one does not, that supports an interaction hypothesis. If both fail, it weakens this specific tested residual approach, not every possible use of those datasets. If a simple branch succeeds while the rich network fails, representation/fusion becomes a much better-supported next diagnosis.

Choose hyperparameters on prior selection only. Use the existing four-fold screen and remaining-fold confirmation convention for comparable development evidence; those folds are heavily reused research history, not fresh untouched tests. Freeze the practical improvement threshold and multiplicity treatment before scoring. Do not keep adding probes until one happens to pass.

### 4.3 Close the corrected-controller question cheaply

The new end-to-end experiment used the repaired derivative; the older conditional/stateful controllers did not. These are different models and cannot substitute for each other's reruns.

Re-run the existing synthetic behavioral admission under the repaired allocator, keeping its teacher distribution, gates and budget. Start with the small conditional controller. If it passes, allow **one** matched four-fold CPU screen against its coherent static benchmark, reusing the exact sealed forecasts and fixing every difference except the derivative/source amendment. The old conditional model is deterministic; do not manufacture three identical seed replications. Preserve the actual three-seed requirement for any stochastic MLP engineering test.

Do not automatically launch the stateful MLP's financial campaign merely because one synthetic seed improves. Its admission must pass consistently first. No new controller layers, feature sweep or loss search belongs in this closure pass. If a corrected controller still fails the declared chronological screen, close this branch until new evidence of predictable opportunity appears.

This has high value for correcting our scientific conclusion and low expected compute cost relative to retraining encoders. It is not currently the leading bet for a large financial improvement, given the separate supervised timing probes that also failed without using the broken allocator gradient.

### 4.4 Choose one representation experiment from the probe result

Two substantive unresolved questions remain from original stages D/F:

**Transfer versus fresh training.** Many attention continuations select epoch zero: 35/42 ranking runs and 28/42 utility runs under IC selection. This establishes that further changes often fail the original selection criterion; it does not prove bad pretraining. If probes suggest the raw rich inputs add information that the frozen representation misses, compare a properly selected compatible parent against a fresh start under the same repaired data, architecture, loss and declared tuning budget. A fresh model must receive an adequate learning budget; twelve continuation epochs would be an unfair ceiling. A third slow-core parent is optional only if the first contrast specifically implicates late-supported families. No parent-by-optimizer-by-fusion cross-product.

**Market/event history before more architecture depth.** TE_all preserves the stock history, but the current auxiliary-family/common-context interface is a current-state encoding, followed by FiLM; it is not a general temporal sequence of all those datasets. Some current fields already contain changes or summaries, so this is not a claim that the model has zero historical macro information. The untested question is whether explicit release-aware evolution adds information that those snapshots miss.

If the evidence favors this route, add one compact sequence of existing dated common-market/release observations while retaining the full sixty-session stock history. Encode the common sequence once per date, shared across stocks. Compare a pooled-history FiLM control with stock-to-common-history cross-attention using the **same** history, so additional information is not confused with the fusion mechanism. Preserve each observation's real availability and mask; do not forward-fill an observation and mark it newly observed. Expand history only if that separately registered question warrants it. No automatic shortening of stock lookback or exclusion of names for speed.

My default order is the cheap incremental-information probe first. Then test additive fusion if the probe succeeds; otherwise use the coverage/representation diagnostics to choose between transfer and a genuinely missing temporal input. There is currently insufficient evidence to launch all three GPU campaigns.

## 5. What I would defer or stop

| Avenue | Disposition |
|---|---|
| Another broad direct-P&L/hybrid/Sharpe-loss sweep | Pause. Requires a new diagnosed failure mechanism, not just a different loss name |
| Economic checkpoint maximization | Keep diagnostic; do not promote. Stable checkpoint averaging could be a later variance-reduction test, but cannot be chosen retrospectively from evaluation returns |
| More SAM/ASAM radii, larger attention models or a Mamba rewrite | Defer until a probe identifies an optimization/capacity bottleneck; existing results do not justify an undirected sweep |
| Learning market timing from the same context with a larger controller | Defer. The regularized supervised opportunity study already supplies relevant negative evidence |
| More seeds or epochs for every losing candidate | Stop as a default response. More seeds improve measurement, not the underlying expected edge; epochs need evidence of useful learning |
| Multi-period trade planning/horizon aggregation | Conditional. Measure forecast decay and horizon disagreement first. Costs are material, but lower attention turnover has already shown that cheaper trading alone need not improve net returns |
| Mandatory sector neutrality or removing the hedge | Do not promote. Existing tests provide no net-return case; sector taxonomy limitations and stock/hedge coherence still matter |
| Broad new dataset collection or longer history | Defer until marginal coverage/information is established. More old dates with absent new fields do not automatically teach those relationships |
| Intraday risk intervention | Remains a separate future safety/execution problem; no forward capture or immediate implementation |
| Repeating a complete generic data audit | Do not reopen without a specific defect. Resolve the identified accounting/source cases and preserve accepted preprocessing |

## 6. Efficient execution and decision rules for the next program

Start with frozen forecasts, existing metadata and CPU probes. Do not repeat pretraining just to inspect whether new fields contain incremental information. Reuse only checkpoints/caches with identical data, parent, preprocessing and recipe contracts.

For a surviving neural candidate, use one strong reference, one substantive change, the existing three seeds and four screen folds. A complete candidate cell is twelve F fits; additional matched controls or parents are necessary only when their contract changes. Confirm only survivors. Keep exact compact names, full histories, AMP appropriate to the RTX 2060, compiled paths where measured faster and FP64 accounting. GPU-hour estimates should follow one representative measured fit, not an arbitrary promise based on network size.

Separate findings into implementation correctness, predictive generalization and economic value. A correct derivative passes the first; a useful residual predictor may pass the second; only matched net/risk results under credible assumptions address the third. Report all attempted cells and preserve adverse periods. Do not select a stop rule, ensemble weight or model using the known losing halves.

The reused 2010–2024 development history supports research comparisons, not a fresh investment-performance claim. Nominal block intervals do not account for the entire accumulated search. Keep 2025/2026 consumers unopened and forward capture disabled under the standing contract. Any eventual independent validation needs a separate explicit decision.

## 7. How this changes the older plan

Original A–C are complete. Broader confirmation (E) was performed for the actual later survivors; it is not a reason to confirm every old cell. Parts of F—return supervision, economic selection and direct utility—have now been tested and do not warrant an automatic repeat. D's transfer/additive-fusion questions and F's explicit market/event histories remain open. The proposed incremental-information probe gives those remaining questions an evidence-based order.

The repaired-gradient controller check is a justified exception to closing old branches: it addresses an observed numerical defect. The independent supervised timing failures remain relevant negative evidence and are not invalidated by that defect.

Keep C6 as the reduced-input reference, TE_all as the working rich-attention candidate and the historical S0 comparator in the research record. No claim that one architecture is universally best follows from this review. The current accepted baseline need not be replaced simply to make the next experiment possible.

## 8. Literature context and limits

The portfolio method itself is not rendered obsolete by the controller failures. Boyd and coauthors separate expected-return/risk forecasting from efficient constrained trading and explicitly do not solve the forecasting problem; this supports investigating forecast quality before adding planning complexity. Their later treatment also emphasizes handling forecast uncertainty. Neither establishes our data's attainable alpha. Sources: [Multi-Period Trading via Convex Optimization](https://stanford.edu/~boyd/papers/cvx_portfolio.html), [Markowitz Portfolio Construction at Seventy](https://web.stanford.edu/~boyd/papers/markowitz.html).

Decision-focused losses remain a legitimate research class. SPO+ research gives calibration/risk results under specified assumptions and studies portfolio allocation, but it does not establish that our path-dependent, cost/borrow/settlement account will improve under direct empirical P&L training. The current negative result and that literature are compatible. Source: [Risk Bounds and Calibration for a Smart Predict-then-Optimize Method](https://arxiv.org/abs/2108.08887).

These papers motivate testable designs; our chronological data and matched economics determine whether a particular design is useful here. The next recommendation is driven primarily by our own completed results and remaining input/representation questions.

## 9. Foundation pass requested by the user

### 9.1 Input selection: use evidence, not an all-data default

TE_all currently includes all ten admitted auxiliary families, whereas C6 includes only fundamentals and magnitudes. This is not every raw dataset: the intraday fast branch is off, and input masks/availability still apply. Auxiliary families enter the characteristic fusion path **after** temporal/peer attention. They are not individual feature tokens in the current stock-attention matrix. Irrelevant inputs can still harm learning, but adding a family does not directly enlarge the stock-attention token count or prove that attention is uniquely vulnerable.

The original all-family configuration was a legitimate research arm. Keeping it indefinitely as the presumed best input set would be unjustified. We should select a useful roster and retain the all-family model as a comparison, not require every future attention candidate to consume everything.

Fresh read-only findings from the accepted 2010–2024 store:

- All 145 slow/auxiliary source fields were inspected on the 568,815 active stock-days. No valid observation is nonfinite, and no field with observed values is globally constant. This is a numerical-support census, not certification of predictive value or every source record. The three additional common diagnostic inputs are not included in this 145-field count.
- **`options:uncovered_call_share` has zero valid values and zero known ages.** Exclude it from the next model input contract. It carries no varying information here. Preserve the immutable source schema and sealed checkpoints; removing its constant encoded channels may require absorbing their affine contribution into the first-layer bias for an exact checkpoint bridge. Do not silently reinterpret existing checkpoint dimensions.
- `options:put_call_oi_log_ratio` has only 442 valid active stock-days and `delta_oi_to_volume_1` only 409. Their observed-series counterparts have much broader coverage. Investigate the exact distinct dates, information overlap and reliability before retaining an expensive claim of learned usefulness. Low support alone is not an automatic deletion rule.
- Microstructure contains two daily fields: average trade size and after-hours volume share. They have 565,501 and 253,025 valid active stock-days respectively; this is not the old intraday prediction branch. Removing that family does not mean eliminating every liquidity-related quantity in the slow input, risk model or execution account.

Round-6 evidence makes microstructure a **first removal candidate**: paired net -1.443 bps/day, interval [-3.049, .177], and IC -.000489, interval [-.002757, .002799]. But masking it at inference slightly *reduced* IC in that fitted model on average (+.001575 model-minus-masked, interval crossing zero). These observations do not prove uselessness in the repaired rich model. Inference reliance, marginal family gain, and the effect of retraining without the family are different estimands.

Practical disposition of existing families:

| Family/group | Starting treatment | Reason |
|---|---|---|
| Fundamentals + magnitudes | Retain as the core; audit fields within them | C6's joint and magnitude-removal diagnostics show reliance; no case for dropping them wholesale |
| Common-market context | Retain in the first compact attention comparison | Post-data TE_all vs TE_family improved IC by .003604; removing FiLM/context while changing width would discard an observed benefit and confound the experiment |
| Microstructure | First matched removal ablation | Weak original marginal results; broad enough support for a meaningful test |
| Odd-lot, sector-derived alpha fields, rebalance | Secondary joint-removal shortlist, then resolve a surviving group only if needed | Weak/uncertain original marginal evidence, different coverage, possible interactions |
| Lending, events, options, stock-specific cross-market fields | Uncertain; use conditional/group evidence | Earlier screens disagree across learners/eras; do not delete based on one preliminary metric |
| Fully absent uncovered-call field | Exclude from the next input contract | Demonstrated absence of both value and age information |

Do not require a positive standalone p-value from every retained field. That would reject interacting predictors and repeat the earlier excessive data-suppression mistake. Conversely, do not let “it might interact” protect every field forever. A family that can be removed without practically material loss across matched seeds/periods, or whose removal improves results, need not remain in the working model. Predeclare an equivalence margin; failure to reject zero alone is not equivalence.

Model-input removal is separate from removing data needed for universe eligibility, borrow costs, risk estimates, corporate actions or truthful reporting. For example, removing sector alpha features does not authorize deleting sector metadata. Missing/age flags require their own semantics: an absent numerical value can still have an informative known age; constant numerical values can coexist with informative changing coverage.

### 9.2 Feature selection without a 145-column search

Use a hierarchy: documented structural redundancy/absence first, family groups second, feature clusters within implicated families third. Inspect fit-period support, distinct dates, revision/timing provenance, scale and correlations. Preserve economically distinct rank/magnitude representations; high correlation is not exact duplication. Inspect known-age versus explicit age fields rather than assuming they are duplicates.

Use existing selected checkpoints for group interventions only to prioritize retraining. Mask through the proper value/valid/age interface; never replace a field with a valid zero and call that feature removal. Ordinary permutation can create implausible combinations when fields are correlated; group/context-preserving interventions are supporting diagnostics, not a deletion oracle. The issue is discussed in [Conditional permutation importance revisited](https://pmc.ncbi.nlm.nih.gov/articles/PMC7362659/).

Final feature decisions require matched training with the proposed roster. Selection uses preceding training/selection data, never picking the subset that happens to win all later evaluation dates. Report the full input roster, retained support and every tested deletion. No new universe/liquidity/staleness gate is introduced.

### 9.3 Architecture: compare components, then synthesize

Fresh graph construction confirms these counts, excluding the later added portfolio-objective head:

| Module | C6 parameters | TE_all parameters |
|---|---:|---:|
| Historical encoder | 24,960 | 33,600 |
| Early peer attention plus temporal pooling | Not present | 20,992 |
| Final pooled cross-stock context | 24,704 | 590,592 |
| Feed-forward trunk | 37,376 | 593,664 |
| Whole model | **127,814** | **1,489,283** |

The attention model is 11.65 times larger, but **79.5% of its weights are in the final pooled-context and trunk modules**. Temporal plus early peer modules are only 3.7%. Those percentages describe parameter allocation, not runtime: per-time stock attention still performs work proportional to the historical stock-pair count.

This strengthens a concrete hypothesis: some excess flexibility may be in the final fusion/readout rather than the relational encoder. It is not evidence that those parameters are all harmful. The first synthesis candidate should retain full 60-session temporal/early peer attention, its nonlinear family adapters and useful common-state conditioning, while testing a more compact post-pooling context/trunk. Do not remove FiLM, change the feature roster and alter the objective in that same comparison.

For attribution, construct matched graph contrasts on one frozen roster. Use the same input projection, temporal pooling, adapters, context/readout width, head supervision, parent policy and budget. Compare GRU sequence states with temporal attention, then the presence/absence of early peer interaction. A complete two-by-two is justified only if needed; first run the nearest informative matched pair and add the missing contrasts when an interaction remains ambiguous. A GRU-plus-early-peer design is a legitimate synthesis candidate: recurrent encoding and cross-stock attention are not mutually exclusive.

The existing C6-versus-TE package comparison must remain labelled as such. Also control the five supervised C6 horizons versus three TE horizons. A common-five-head bridge preserves C6's auxiliary supervision; compare it with current TE's three-head control before crediting any difference to the encoder. Evaluation remains on D3/D5/D10. This is a supervision contrast, not proof that five heads are better.

SAM .125 versus ASAM .2, different parent inputs and differently tuned budgets are additional confounders. Use a shared recipe for causal component contrasts, plus a small equal-budget selection-only adjustment if optimization diagnostics show one graph is disadvantaged. Do not give a favored architecture a larger search and call the result a pure architectural effect. Input-specific or shape-changing parents must be compatible and explicitly trained/adapted; deleted-input P exposure cannot be silently described as never having seen that input.

### 9.4 Variance reduction deserves high priority

There is direct, if historical, project evidence: the earlier S0 raw-versus-EMA diagnostic had evaluation mean-seed IC .026315 versus .027890, a +.001576 difference. It was not a current matched promotion experiment, and its checkpoint/initialization context matters. It nevertheless makes variance reduction a better-motivated candidate than another arbitrary architecture rewrite.

Prioritize:

1. **Measure the existing ensemble benefit first.** Use archived out-of-fit forecasts to compare the mean performance of individual seeds, all three pairs, and the existing three-seed ensemble. Do not choose the best seed or pair using evaluation outcomes. Shared pairs are dependent, not independent replications. Report prediction stability and economics, not only IC. Different score aggregation may require a consistently prior-fitted calibration.
2. **One local checkpoint-average comparison.** On saved compatible trajectories, compare raw selection with a fixed trailing average of up to three saved checkpoints ending at the selected epoch. Do not force averaging over a late overfit tail. Restrict averaging to the same graph, parameter alignment, preprocessing and trajectory. This is checkpoint averaging, not reconstructed update-level EMA. If original F epoch states are absent, do not substitute economic-continuation states while claiming to test the original model.
3. **Track true EMA during already-needed new fits.** One registered averaging horizon, measured in training progress rather than blindly identical update counts, avoids another sweep. Update the shadow only after successful clean optimizer steps, never SAM-perturbed weights or skipped AMP steps. Select raw and EMA on the same earlier selection window, retain both trajectories, and report separate fixed-strategy comparisons. Extra evaluation has a cost, but no second training pass is needed.
4. **Retain cross-architecture diversity as a cheap diagnostic.** A fixed equal-weight forecast blend can complement the already tested adaptive blend, if that exact account does not already exist. It is not a substitute for determining each component's value.

Weight averaging is supported by [SWA](https://arxiv.org/abs/1803.05407) and a systematic [EMA study](https://arxiv.org/abs/2411.18704), but their results do not establish improvement in this financial task. Do not average unrelated independently initialized model weights: forecast averaging and parameter averaging are different operations. SAM and averaging can be complementary but are not guaranteed to be so.

Do not automatically add three more seeds, bootstrap away many training dates, or smooth live predictions across days. More seeds incur nearly proportional training cost; historical prediction smoothing can delay response to new information. Internal parameter-efficient ensembles such as TabM are a reserve candidate: the repository already tried B10 unsuccessfully under the older problematic training/data regime. That result is qualified, but does not make a rerun the first choice. Measure temporal/seed variance before adding that complexity.

### 9.5 Scaling: separate encoder size from final readout size

We do **not** yet have a clean empirical scaling curve for these current models. The 128k-versus-1.49m comparison changes too many other things. Neither “larger is better” nor “we have too little data for a larger model” follows directly.

Graph-only counts on the unchanged full input roster:

| Attention configuration | Parameters | What changes |
|---|---:|---|
| Current | 1,489,283 | 64-wide temporal representation; 256-wide context/trunk; three blocks |
| Readout width 128, three blocks | 518,019 | Only the post-pooling width family is reduced; temporal/peer history unchanged |
| Readout width 128, two blocks | 468,227 | An additional depth change; test separately after width |
| Temporal width 96, original readout | 1,568,899 | Temporal/peer capacity increased; original final network retained |

These are parameter counts, **not speed benchmarks or fitted results**. In particular, the compact variant removes about 65% of parameters without deleting history, names or datasets, but may still lose useful predictive capacity. Its benefit must be measured. Widening temporal processing from 64 to 96 adds only about 5.3% total parameters, yet can noticeably increase expensive sequence computation.

First compare the existing readout width with the compact width. Then, if justified, test one wider encoder (64 to 96) with the winning readout held fixed, and one depth change separately. C6 should receive its own one-axis width/depth check if its fit/selection curves suggest a bottleneck; do not force an equally large readout on it just to equalize parameter counts.

At fixed data dimensions, GRU matrix work grows roughly with hidden-width squared and sequential depth; attention projection/MLP work also grows roughly quadratically in width, while attention mixing grows linearly in width and quadratically in sequence/name counts. These are operation-count tendencies, not wall-clock forecasts. Preserve every eligible name and the 60-session window. Measure real RTX 2060 step time, peak memory, epochs to selection and total fit time. Compare both performance per experiment and performance per unit compute, without truncating the larger model's training prematurely just to equalize elapsed minutes.

### 9.6 Fast, staged delivery and stopping points

1. **Inventory and confidence work:** source/financing investigation, the completed input/parameter census, archived ensemble diagnostics and checkpoint availability. No neural fitting is required to decide the shortlist.
2. **Input wave:** exclude the demonstrably empty option field from the next compatible model contract, then test microstructure removal and at most one predeclared grouped removal. Keep C6's core roster unless its field audit identifies a concrete issue. An inference-only mask is insufficient for accepting removal.
3. **Stability and structure wave:** compare the one averaging rule using saved states where available; track EMA during necessary new fits; test the compact attention readout as one isolated capacity contrast. These do not form a Cartesian product with every candidate roster.
4. **Conditional follow-up:** only then run the nearest GRU/attention component bridge or the one-axis encoder/depth experiment that the diagnostics justify. Choose one synthesis candidate, then confirm it with the fixed protocol.
5. **Incremental rich-data probe:** use the retained roster in section 4.2's small residual study. It may share the input audit and earlier CPU preparation, but final comparisons must use the frozen candidate roster. New market/event histories and novel architecture mechanisms wait until the foundation pass is assessed.

Limit each GPU wave to **two new candidate cells**, each normally four folds by three seeds (24 new F fits maximum before a review), plus only required matched controls and compatible parents. Reuse controls only when their contract is actually identical. This is a cap per wave, not a promise that all later waves will run. A preliminary two-fold engineering/selection check can detect failed optimization; it cannot certify an alpha improvement or delete a family as universally useless.

Input selection and width/depth are model selection, even when called cleanup. Before outcomes, freeze practical noninferiority/improvement margins, eligible folds, prior-only tuning and the nominal paired uncertainty procedure. Fewer fields/weights can be preferred when a declared practical-equivalence criterion is met; a wide confidence interval spanning zero does not prove no loss. Preserve failed attempts. After each wave, return to this document and explicitly record what the evidence permits next.

The only additional categories needed are **experimental comparability** (heads, parents, preprocessing, optimizer and selection policy) and **regime/coverage stability**. They protect interpretation of the user's six categories. They do not authorize a new general data rebuild, a broad optimizer search, held-out access or forward capture.
