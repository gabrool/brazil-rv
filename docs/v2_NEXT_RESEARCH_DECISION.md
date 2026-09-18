# Research decision after the portfolio-objective experiments

2026-09-18. Assessment and recommended next program, **not a registration or authorization to launch another campaign**. The completed financial source is 33086be; its final review and recovery were published in 193fe4d. No new model fitting, portfolio replay or held-out access was performed for this assessment. A small descriptive comparison of existing forecasts/accounts is recorded in [the evidence file](v2_next_research_evidence.json).

## 1. Decision

**Pause broad end-to-end P&L, execution-controller and checkpoint-selector searches. Continue research on incremental predictive information and how the model represents it.** Preserve the corrected allocator derivative and all accounting/efficiency improvements. Do not replace ranking training or the established references with the unsuccessful continuation candidates.

There is one bounded exception: the older learned-controller conclusion deserves a corrected-gradient check, because a demonstrated numerical defect affected that branch. This is a narrowly scoped closure experiment, not justification for another large policy search.

The highest-value *alpha* question is now: **Can the existing rich datasets add stable information beyond a strong predictor when their contribution is learned separately, without having to relearn or disturb the whole predictor?** The most urgent *validity* work is resolving the material settlement and financing assumptions. These are different priorities: one seeks better predictions; the other establishes what existing backtest returns mean.

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

This is my highest-priority next alpha experiment. It can begin on CPU while document retrieval proceeds; promotion still depends on credible accounting. It is a **revised test after data repair**, not a claim that residual trees have never been tried.

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
