# Brazil-RV: research priorities for improving the multiday model

This is a research proposal for independent LLM review, based on the repository at commit `d6679e1331239fa21d866e10f7170691597edbf4` and primary sources checked on 10 September 2026. It distinguishes measured results, implementation facts, and proposed experiments. It does not amend a registration, authorize a new holdout read, or claim that an untested idea works.

## Recommendation

The strongest program combines **better conversion of forecasts into trades**, **a small amount of genuinely new information**, and **targeted tests of how the existing model learns**. A larger neural network is not the first investment I would make.

My first five priorities are:

1. Use the frozen S0 forecasts to test whether cost-aware entry, retention, and sizing can preserve more of their predictive value. First establish a continuous-book comparison and resolve the known economic-data limitations sufficiently for this test.
2. Test conservative combinations of S0 with the existing slow GBDT and momentum forecasts. The current evidence suggests materially different ranking and portfolio behavior; additional seeds of the same network do not address that kind of diversity.
3. Build a small, point-in-time earnings and announcement dataset. Signed information about a new earnings release is a more promising addition than another transformation of the same price path.
4. Test whether the reduced fine-tuning learning rate unnecessarily anchors recent models to 2010–2016 pretraining. Then, only if warranted, test one recency treatment.
5. Add a very small set of global market and economic inputs through company-specific exposures. A currency, commodity, or rate shock matters differently to an exporter, importer, bank, and utility; a common scalar added identically to every score cannot express this.

The first two are attractive because much of the expensive work already exists. The third offers the strongest plausible new-information upside, but publication-history reconstruction is a substantive research task. The fourth is unusually easy to isolate. The fifth has a clear economic mechanism but needs a restricted feature budget.

This ordering is a judgment about expected value, not a numerical probability forecast. There is no defensible basis for assigning precise success percentages or expected IC gains to these unrun experiments.

| Priority | Experiment family | Chance of a useful improvement, relative to this shortlist | Potential material benefit | Main cost | First test |
| --- | --- | --- | --- | --- | --- |
| 1 | Forecast-to-trade calibration and costs | Medium–high, conditional on reliable execution data | High for net returns; does not create forecasting alpha | Ledger/calibration work | Frozen scores; one cost-aware entry/retention rule |
| 2 | Diverse forecast combination | Medium | Moderate; potentially substantial for stability and net returns | Low; saved scores and CPU replay | Two fixed blends, no weight search |
| 3 | Earnings and timestamped announcements | Medium, conditional on usable historical vintages | High for incremental information | Data engineering | Event timing plus signed seasonal earnings surprise |
| 4 | Transfer and adaptation | Medium | Moderate, possibly large if old pretraining is restrictive | One matched neural arm | Fine-tuning multiplier 1.0 versus 0.3 |
| 5 | Global inputs with heterogeneous exposures | Medium | Moderate–high | Small data build and one arm | FX, oil, and rates/risk context with causal exposures |
| 6 | Preserve selected feature magnitudes | Medium | Moderate | Small preprocessing change and one arm | Add four magnitude channels beside existing ranks |
| 7 | Public index changes and lending on S0 | Medium–low to medium, depending on coverage | Moderate, concentrated in relevant periods | Mostly existing adapters; data audit | One compact family at a time |
| 8 | Smaller/simple model and selective attention | Uncertain | Moderate–high if the current representation is limiting | New training and graph validation | Simple comparator before one residual attention block |
| 9 | Target-estimation noise and economic calibration head | Conditional on diagnostics | Moderate–high | Label analysis and controlled retraining | Keep evaluation target fixed; change one training objective |
| 10 | Fund-flow pressure from public holdings | Medium–low before an availability audit | High if genuinely incremental | Substantial point-in-time engineering | Publication-valid coverage and pressure study first |
| 11 | Temporal memory or additional history | Low–medium | Uncertain | More training/data work | 120-session context before pre-2010 expansion |

An archived-EMA comparison is a cheap optional screening experiment, rather than a major research thesis. Electricity-sector data is a credible later specialist experiment. Neither should displace the first five priorities.

## 1. What the model actually is

### Accepted parent and information set

The working V2 parent is **S0**, following Round 4. It uses the slow daily stream and removes the current/intraday projection from the network. This is different from an intraday model whose fast values are merely set to zero. Its 15:45 decision uses completed daily slow observations through the prior session. Its five predictive horizons are D1, D2, D3, D5, and D10; the current primary metric averages D3/D5/D10 neutral-target IC. Here, IC means a cross-sectional rank correlation, not a return.[^r4][^model]

The slow stream already contains **32 features**, including returns through 252 sessions, 12–1 momentum, several volatility measures, idiosyncratic volatility and beta, volume/liquidity measures, history/missingness information, and causal cluster-relative features. The sequence length is 60 sessions. Therefore, “add momentum,” “add volatility,” and “give it a year of history” are not accurate descriptions of missing capabilities: year-long summaries already exist, although a year-long sequence does not.[^features]

Most continuous features receive a cross-sectional rank-Gaussian transformation. Each feature also supplies validity and age information. A width-64 GRU encodes each security, followed by name-dependent gated cross-sectional mean/dispersion pooling and a small residual prediction network. The model already shares information across stocks, but does not use explicit pairwise stock attention.[^features][^model]

The training recipe already includes SAM-AdamW, dropout, three matched seeds, and a historical pretraining stage. Pretraining uses 2010–2016 data; expanding fine-tuning windows begin in July 2016. Transferred parameters receive a learning-rate multiplier of 0.3. An EMA with decay 0.995 is maintained and archived, but raw Patience checkpoints supply the accepted predictions. “Add SAM,” “add ensembling,” or “implement EMA” would duplicate existing work.[^training]

The primary labels are not raw stock returns. They scale shareholder returns by causal volatility and horizon, remove cross-sectional characteristic effects involving volatility, beta, and liquidity, then rank the residual. The label transformation uses realized cross-sectional outcomes, which is legitimate label construction; those outcomes must never enter historical features. This neutralization does not explicitly neutralize momentum.[^targets]

### Current evidence

Round 4 uses 14 half-year development folds spanning 2018–2024, with 1,738 sessions for economics and 1,598 defined sessions for the pooled primary IC. Undefined label dates are not zero-IC observations. The table below uses the current full-calendar economic convention. GBDT/control results come from the matched Round-4 CPU replay, not the much narrower earlier Round-1 comparison.[^r4][^cpu]

| Model/control | Primary D3/D5/D10 neutral IC | Net excess, bps/day | Important interpretation |
| --- | ---: | ---: | --- |
| S0 | 0.026591 | 4.936 | Accepted research parent |
| Previous `fast_off` neural parent | 0.023481 | 4.201 | Matched parent for Round-4 ablations |
| Slow GBDT, `a_slow` | 0.017440 | 4.616 | Much lower IC, similar net point estimate |
| Momentum 12–1 | 0.025134 | 2.627 | Close to S0 on aggregate IC, weaker economic point estimate |
| Five-session reversal | -0.000493 | -1.584 | No case for blindly adding a reversal expert |
| Twenty-one-session reversal | -0.013047 | -3.906 | Negative in this registered control |

These are **levels**, not paired significance tests between every listed candidate. The existing S0-minus-fast_off paired IC gain is **0.003110**, with a 95% block interval of **[0.001522, 0.005129]**. The paired net gain is **0.735 bps/day**, with **[-0.234, 3.130]**. S0's own net interval, **[1.174, 8.989]**, does not establish that it economically beats the other models.[^r4]

The leave-one-seed-out S0-minus-fast_off IC gains are approximately 0.003081, 0.003191, and 0.002995. Their corresponding net gains are 0.256, 2.239, and -0.519 bps/day. Thus the relative ranking result is more stable than the incremental economic result. These overlapping two-seed panels are sensitivity checks, not three independent replications.[^audit]

S0's mean cross-sectional correlation with momentum is approximately **0.726**; its diagnostic momentum-residual IC is approximately **0.0154**. There is evidence of information beyond a simple momentum ranking, but substantial shared structure. The reported extreme-momentum spread attribution is not a causal decomposition of total portfolio P&L, and should not be described as “70% of the strategy's profit is momentum.”[^r4]

The highest-value question is consequently not just “can IC rise?” It is also: **which incremental forecasts remain useful after position persistence, risk constraints, borrow, and trading costs?** The GBDT's lower IC and similar economic point estimate make that question especially relevant. This is a motivation for further comparison, not proof that its forecasts diversify S0.

### Experiments that must inform the prior

Round 4 already tested horizon reweighting (H), score-persistence regularization (P), lending (L), and three common daily state variables (C). None replaced S0. P also failed one unchanged occupancy bound in its seed-sensitivity audit and was excluded under the disclosed A4.1 amendment. Its very persistent forecasts reduced turnover without improving the selected outcome.[^r4]

These arms were tested against **fast_off**, not as additions to the newly selected S0. That leaves a scientifically motivated interaction test possible, particularly lending, whose available-data subset had a positive IC result. It does not justify automatically rerunning every unsuccessful arm on a new parent.

The earlier intraday program also screened lending, SHFE commodities, options activity, CVM filing events, odd lots, index changes, CCEE electricity prices, fundamentals, and ADR information; none of those exact recipes passed its gate. Some requested fields were unavailable rather than tested. Those results lower the prior for an indiscriminate feature dump, but do not settle the value of a different information set, publication treatment, or multiday horizon.[^context]

## 2. Fix the measurement bottlenecks without blocking all research

Better data can make an apparent edge smaller. That is useful: the objective is a reproducible improvement in an economically meaningful model, not a higher number produced by favorable assumptions.

| Issue | Why it matters | What it should block |
| --- | --- | --- |
| Daily last-trade close as execution proxy | A 15:45 decision does not establish that the daily last trade was an attainable fill | Strong execution/capacity claims and promotion of timing-sensitive policies |
| Missing BOVA11 marks in 2019H2 | Terminal settlement closes the ledger but cannot reconstruct 92 missing hedge-return sessions | Unqualified conclusions from affected economics |
| Inferred corporate-action terms | Bad adjustments can create false features, labels, and wealth changes | Affected label comparisons; repair before treating such gains as alpha |
| Forced liquidation at each half-year boundary | Can distort turnover, holding periods, and cost-aware policy comparisons | Promotion of a new multiday holding policy without a continuous-book readout |
| Placeholder borrowing rates before observed coverage | A short-side gain may depend on favorable cost assumptions | Unqualified full-history short economics; require explicit cost scenarios |
| Missing publication vintages for a proposed dataset | A valid observation date is not necessarily a valid availability date | That dataset's backtest, not unrelated experiments |

The repaired store is still development-grade: its calendar is reconstructed and action terms are inferred from COTAHIST. The official B3 historical files themselves are unadjusted for dividends and other corporate actions. More raw price history does not solve this problem automatically.[^r4][^b3]

A practical first pass is a **risk-weighted data audit**: inspect the largest absolute labels, largest ledger contributions, action transitions, delistings, share-class/identity changes, and known missing hedge segments. Do not drop an observation because it loses money or is hard to explain. Require independent source evidence for a correction; preserve the original sealed result and re-baseline comparisons consistently.

For continuous-book replay, carry inventory across scheduled model updates, change forecasts when the new model becomes available, and retain one chronological wealth path. Do not invent scores for fold-end dates where existing arrays intentionally omit predictions: audit saved score coverage first, then recover only missing predictions from eligible frozen checkpoints if necessary. Keep the original fold-reset readout as a historical comparator. Labels must still respect their registered evaluation boundaries.

This can proceed alongside input and training research. There is no reason to suspend a clean prediction ablation while every historical execution ambiguity is investigated. There is a reason to avoid selecting a complex trading policy on an unresolved ledger.

## 3. First experiments, specified narrowly

### E1. Convert ranked forecasts into economically selective trades

**Hypothesis.** A fixed-slot, equal-notional ranking book discards useful information about whether a proposed replacement is worth its cost. The current S0 forecast may support better after-cost decisions even if its IC is unchanged.

The existing policy already uses unsmoothed D3/D5/D10 scores, equal sizing, and a buffer of nine per volatility quintile. R3.1 selected it in sample after a substantial sweep. Another generic buffer or smoothing grid would repeat research and increase selection bias.[^execution]

The distinct experiment is to estimate the **expected benefit of changing the current position**, including the alternative of continuing to hold it. A newly attractive stock should not automatically replace a still-useful holding when the expected difference is smaller than entry, exit, financing, and borrow costs. Dynamic trading research motivates this treatment of signal persistence and costs; it does not demonstrate that a particular implementation will work in Brazil.[^gp][^boyd][^frontier]

**First test:** keep frozen S0 scores, the universe, risk caps, and current equal notional sizing. Replace only entry/retention selection with one conservative trade-versus-hold rule. Estimate a heavily pooled, monotone mapping from score and holding age to expected residual shareholder return using strictly earlier, purged out-of-fold predictions. Use a small number of predeclared score buckets; do not fit a high-capacity second model.

Audit the calibration sample before calling this a cheap replay. Saved outer-fold scores begin too late to calibrate the earliest fold from earlier outer folds; a short internal selection panel may also be inadequate. Missing early calibration predictions require explicitly budgeted inner chronological fits, or a predeclared simple fallback with separate coverage reporting. Never borrow later folds to fill that gap. Prefer calibrating next-session residual returns by current score/holding age, then valuing a fixed planned holding path, rather than learning realized exit times from the policy being optimized.

A rank is not a return in basis points. Subtracting a cost in basis points from a raw rank is dimensionally meaningless. The calibration must put expected marginal holding return and marginal implementation cost in the same units and use the same planned holding horizon. Existing holdings and new entries have different incremental costs. D3, D5, and D10 returns overlap; they are not three independent cash flows to add together.

Initially, restrict the calibration to score, holding age, and a few causal risk/cost quantities. Preserve gross/net/name/hedge limits and compare at matched risk and capital. A rule that simply invests less should also be compared with a correspondingly scaled baseline; a rule that takes more gross should not receive credit for extra leverage alone. If the fixed minimum gross makes the intended abstention test impossible, preregister a separate risk-matched comparison instead of quietly changing the bound.

**Readouts:** paired net return, gross return, actual cost attribution, turnover including model-update dates, drawdown, long/short contributions, concentration, participation, and score decay conditional on holding age. Separate observed-rate periods from placeholder-rate history. Use an explicitly labeled more adverse cost scenario, not only the frozen headline cost model.

**Advance only if** the gain is not explained by reduced risk, favorable terminal handling, or a handful of untradeable names. If equal-sized trade selection works, test smooth confidence-based position sizing next. Directly differentiating a neural network through portfolio optimization is a later experiment, after this simpler comparison establishes a benefit.

### E2. Diversify forecasts, rather than automatically adding more seeds

**Hypothesis.** A different forecasting family may contribute errors or holding behavior that more initializations of the same GRU do not.

Use existing S0, slow-GBDT, and momentum score panels. Align their eligible populations, horizon definitions, dates, and cost conventions before combining them. Re-rank within the common decision cross-section so differences in output scale do not determine the blend. Keep missingness explicit; do not let “both available” silently choose a different universe for only one candidate.

**First test:** preregister exactly two fixed score recipes: 75% S0 plus 25% slow GBDT, and 75% S0 plus 25% momentum. The 25% allocation is a deliberately conservative proposed design choice, not a measured optimum. Use the existing three-seed S0 ensemble; no new neural fit is required if the saved panels satisfy the contract.

Report the matched S0 baseline, pairwise score/error dependence, residual predictive contribution, and the economic readouts from E1. A lower standalone IC does not disqualify a diversifier, but high standalone IC also does not guarantee a useful blend. Momentum is already a major part of S0's information set, so its marginal benefit may be small.

If a blend helps, compare one fixed blend of the **resulting portfolio targets** as a separate follow-up. Blending scores and blending books are not equivalent because ranking thresholds, costs, and holdings are nonlinear. Do not assume a portfolio blend is free of crossing or execution costs.

Do not tune a dozen weights. A learned blend is justified only after a fixed combination works; fit any weights on nested chronological predictions with strong shrinkage and a simple fallback. Prior Round-3 ensemble results are relevant but were not this exact S0 comparison. A reversal expert is lower priority given the current negative controls.[^cpu][^r4]

### E3. Add signed earnings information and event timing

**Hypothesis.** Price/volume history cannot fully reconstruct what a newly published financial report says. The information could affect several following sessions, particularly when the surprise interacts with company size, liquidity, and existing price trends.

The current events adapter maps filing age, but `standardized_unexpected_earnings` is unsupported. Most named fundamental fields are also unsupported: market capitalization, book-to-market, and gross profitability are not supplied merely because their names appear in the schema. S0 consumes none of these sidecar families. This is a genuine missing-information opportunity.[^sidecars]

**Data feasibility first:** link CVM issuer identity to dated B3 security identities; recover the actual public release/receipt time and report version. CVM ITR archives include annual files from 2011, but the files are updated for resubmissions. A ZIP downloaded today is not proof of what an investor knew at the time. The IPE registry and original filing documents may help reconstruct publication history; that needs verification on representative historical issuers before promising a complete dataset.[^cvm-itr][^cvm-itr-directory][^cvm-ipe]

**First feature set:** first-publication indicator/age; seasonally comparable earnings change; revenue change; and one accrual or cash-conversion measure only where genuinely comparable. Construct an initial surprise relative to a simple same-quarter-prior-year expectation, scaled by information available before publication. This is an accounting-based surprise, not analyst-consensus surprise. Do not claim access to free historical analyst consensus that has not been found.

Quarterly statements may contain year-to-date flows. Convert them to standalone quarters using versions available at the event time; annual-minus-nine-month Q4 construction must respect the publication of both inputs. Handle consolidated versus individual accounts consistently. Banks and nonfinancial companies need different accounting semantics; missing comparability is not a zero surprise. Avoid backfilling restatements into earlier decisions.

Start with a small residual event expert whose contribution is zero when no usable event information is available. Compare timing-only with timing-plus-signed-information so any gain is not misattributed to earnings content. Use a fixed short post-publication activity window or one predeclared decay; event sampling for estimation must be reweighted back to the intended decision distribution. Keep the base book eligible on non-event dates.

**Readouts:** common-population IC/net, event-window IC/net, number of distinct issuers and announcements, yearly coverage, first-publication versus restatement behavior, and sector/size concentration. Cluster event diagnostics by issuer and announcement; multiple dates and share classes around one report do not supply independent events.

Latin American research, including Brazilian firms, reports post-announcement drift and motivates the hypothesis. Its longer event windows and use of analyst information do not establish a free-data, D3–D10, after-cost strategy. The experiment must demonstrate that bridge itself.[^pead]

Broad static value/profitability factors should follow the event experiment, not be assumed to be the strongest short-horizon addition. Their slower information cycle may be more useful as conditioning variables than as daily standalone forecasts.

### E4. Test whether old pretraining restricts adaptation

**Hypothesis.** The current low learning rate on transferred parameters may retain useful structure, or may keep recent models too close to an old regime. The code establishes the mechanism; it does not establish a defect.

**First test:** S0 with fine-tuning's transferred-parameter learning-rate multiplier changed from 0.3 to 1.0. Keep features, pretraining checkpoints, fitting dates, losses, optimizer settings other than this multiplier, seeds, and selection rule fixed. This can reuse matching S0 pretraining; it does not require manufacturing a different architecture.[^training]

Inspect paired gains by chronological era, training/selection behavior, and seed sensitivity. Avoid concluding that later folds improve simply because they are larger or easier: all comparisons are paired within fold.

If adaptation improves consistently, the next experiment is one predeclared recency weighting, for example a two-year half-life within each legally available fitting window. That half-life is a proposal, not an optimum. Compare effective date weights and training exposure. A sampling change that also changes the number of optimizer updates has two mechanisms unless the comparison controls or explicitly separates them.

A randomly initialized model trained on recent data is a useful later diagnostic, but “no pretraining” also removes old data exposure. To isolate transfer mechanics from data availability, distinguish a recent-window-from-scratch arm from an all-available-history-from-scratch arm. Do not run that full matrix initially.

There are two additional inherited choices worth diagnosing before expanding this family. Stage-P checkpoint selection still uses D1/D2/D3/D5, whereas Stage-F selection uses D3/D5/D10. Also, models are refreshed at the half-year fold boundaries. If prediction quality decays with time since fitting, one scheduled mid-fold refresh is a more direct test than a larger architecture. Fit it only on earlier, fully matured labels and preserve the unchanged first-half predictions. This adds training; it is not a free replay. A Stage-P horizon-alignment test or a quarterly-refresh test should be a separate follow-up, not bundled with the learning-rate change.[^training]

**Stop** if apparent improvement is confined to an isolated regime or if adapting faster trades away robust performance for a fragile recent fit. More old data and more recency are competing treatments, not changes that should be combined before either has been tested.

### E5. Add global context through company-specific exposures

**Hypothesis.** The model lacks some economically relevant shocks that occur outside an individual Brazilian stock's daily path. Their value is mainly in how different companies respond.

A common shock added to every score as the same scalar leaves the ranking unchanged. Useful mechanisms include conditioning the network or interacting the shock with a dated sector classification or a causal, shrinkage-estimated stock exposure. For example:

`currency change × past FX sensitivity`,
`oil-price change × past commodity sensitivity`,
`rate/risk-state change × past rate sensitivity`.

These exposures must use only trailing returns and public information, with no full-sample fit or present-day sector map. Shrink estimates toward broad groups when individual histories are short. Fit transformations inside the historical training information set.

**First test:** no more than three external drivers, with a small fixed number of lags and exposure interactions. A sensible starting set is BRL exchange-rate change, oil-price change, and a rate/global-risk variable. Prefer already acquired, publication-audited sources where possible. The old repository contains external-data work on commodities and ADRs, so audit canonical archives before commissioning a duplicate collector.[^context]

BCB provides PTAX bulletin data; Cboe provides daily VIX history; EIA provides historical oil spot prices; FRED/ALFRED can supply selected rate/macro series with explicit vintage handling. These are different measurements: PTAX is a reference-rate bulletin, VIX is an index, and an EIA spot assessment is not an executable oil-futures price. Use their release timing and documented meaning.[^ptax][^vix][^eia][^alfred]

At 15:45 Brazil time, the same-date US close may not yet exist. Use an earlier available close or a timestamped intraday observation, with historical daylight-saving calendars. A one-day lag is not enough for every publication series; lag from actual availability, not from the economic reference date.

The Round-4 C arm already tested prior market return, median volatility, and return dispersion. The new hypothesis is **external shocks plus heterogeneous exposures**, not another undifferentiated market-state append. Separate the external-data contribution from additional model capacity with a matched small residual branch.

**Advance** only if the gain appears across more than one company or exceptional episode and improves conditional stock selection after accounting for broad sector/market exposures. A strategy that merely acquires oil or dollar beta should be identified as such.

### E6. Retain useful magnitudes beside cross-sectional ranks

**Hypothesis.** Rank normalization is robust but loses economically relevant distances. The most volatile name can be much more volatile than its neighbors on one date and only slightly more volatile on another, while receiving the same normalized rank. Similar losses occur for return shocks and liquidity.

**First test:** retain the existing 32-feature representation and add four causal magnitude channels, such as a one-session return scaled by prior volatility, absolute volatility, log traded value, and an economically interpretable market beta. Use audited physical quantities, not an attempted inversion of rank-Gaussian values. Absolute prices remain unsuitable features; invariant return, risk, and liquidity quantities are different.

Estimate clipping and normalization from training data or an explicitly causal rolling history, never the full panel. If a rolling transformation is used at decision t, document its inputs through t-1. Preserve missingness and outlier indicators rather than treating missing data as an ordinary zero.

This is a narrow test of lost information. Replacing every rank with raw values, adding a new architecture, and changing targets simultaneously would not answer it. Compare active populations exactly and ensure the new channels do not merely reveal differences in source coverage.

**Advance** if the improvement is stable across magnitude/risk regimes and remains after costs. If it only recreates a low-volatility or liquidity tilt, report that explicitly. The previous C result lowers the prior for a large gain from common magnitudes alone, but did not test these individual-security channels.[^features][^r4]

### E7. Reuse the most relevant existing sidecars selectively

**Index changes.** Public index previews and scheduled effective dates can identify demand that is absent from a stock's own path. The repository already has explicit preview additions/deletions, signed weight changes, pressure, pre-effective ramp, and post-effective reversal fields for IBOV, IBXX, and SMLL. Audit publication-date coverage and actual valid values before treating these as ready-made multiday alpha.[^sidecars]

The first test should use one compact published-pressure signal and event age on S0, not append all 21 correlated fields. Keep a zero contribution outside supported event windows. The effective future index membership must never become today's eligibility rule; only already announced changes can be used as information. Do not compare a preview strategy against a baseline retrospectively restricted to eventual constituents.

**Lending.** The L arm's informative subset starts in 2022, and its IC gain against fast_off is 0.001553 with interval [0.000543, 0.003041]. The all-fold and economic evidence is weaker. This is sufficient motivation for a single S0-plus-lending interaction test, not proof of a broadly persistent effect. Separate loan demand, rate, and missing-coverage behavior, and retain all-fold and available-era readouts. Borrow inputs used for execution costs are conceptually distinct from lending inputs used to predict returns.[^r4]

**Priority decision:** if the index archive has broad, credible publication history, test it before importing a new complex dataset. If coverage is poor, prefer the already informative lending test. Do not run both merely because their adapters exist. The old intraday rejection is contextual evidence, while the specific multiday mechanism and available coverage determine the new test.

Options and odd-lot activity are lower priority. Some options fields in the V2 schema, including implied-volatility/skew fields, have no source mapping. A mapped name also does not establish the economic meaning or quality of the underlying measurement. An implied-volatility surface requires much more than attaching a column with that name.[^sidecars]

### E8. Test a simple representation before selective attention

**Hypothesis.** The GRU's temporal bottleneck or coarse pooling may be limiting. Alternatively, the model may be learning mostly static nonlinear relationships from features that already summarize long histories. Those explanations suggest different improvements.

**First comparator:** a small residual MLP using the latest available slow feature vector and the same missingness information, trained under the same chronological targets and selection contract. The existing slow GBDT is also a required comparator. A shallow regularized linear/additive model is a cheap diagnostic if its features and target align exactly.

If the simple model approaches the GRU, a bigger temporal model has a weak immediate case. The useful route may be reduced estimation variance, a blend, or better inputs. General tabular benchmarks support taking strong simple models seriously, but are not evidence about this particular return sequence.[^trees]

If temporal or relational information is demonstrably useful, test **one** of the following next, chosen before seeing its results:

- Attention pooling over the already historical GRU states, to retrieve relevant earlier observations instead of using only the last state.
- One small residual cross-stock attention block, allowing a name to retrieve relevant peers instead of receiving only pooled mean/dispersion.

Do not combine both in the first arm. Match parameter count where feasible, or include a comparably sized non-attention control. Preserve the stock-permutation behavior: permuting securities should permute their forecasts, not change their economic meaning. Mask unavailable names; learn correlations or relationships only from information available at that date. A permanent ticker embedding is not a substitute for point-in-time identity handling.

Set Transformer and MASTER provide concrete architectural ideas for interactions among a set of entities and stock-specific attention. Their results are external motivation, not a transferable performance guarantee. S0 already has peer features and gated pooling, so the experiment must justify **richer interactions**, rather than claim to introduce cross-sectional information for the first time.[^set][^master]

### E9. Reduce target noise without changing the answer being graded

**Hypothesis.** Daily cross-sectional neutralization, volatility scaling, and a pure ranking loss may discard useful magnitude information or introduce estimation noise. The size of that problem has not yet been established.

Start with diagnostics: eligible names versus regression degrees of freedom, leverage and influence of extreme outcomes, clipping frequency, group occupancy, label changes near eligibility thresholds, and the stability of residual ordering. The implementation changes from a simpler characteristic regression to volatility/beta buckets when enough names are present. This is a concrete place to inspect, not an automatic bug.[^targets]

If unstable neutralization is material, compare one shrunk/simplified **training-label** neutralization with the current version. Continue grading on the unchanged registered neutral target, raw/shareholder-return diagnostics, and identical economics. Otherwise a new target can appear better merely because it is easier to predict.

An alternative follow-up is one small robust magnitude head beside the ranking head. Predict a volatility-scaled residual shareholder return and use it for training regularization or separately validated calibration. Keep the ranking target and main readout unchanged. This differs from the already tested H arm, which only changed horizon weights.

Do not automatically remove large moves, downweight loss-making stocks, or call all market/sector variation “noise.” Common movements may be tradable through heterogeneous exposures; removing them can discard signal. Conversely, retaining a market premium is not evidence of stock-selection alpha. Any risk-residual label must use causal exposure estimates and receive both neutral and economic evaluation.

Adjacent multi-day labels overlap. Treating every stock-date-horizon row as independent exaggerates sample size. Sampling more evenly across dates may improve optimization, but subsampling away observations or changing the loss weights is itself an experiment. Keep full cross-sections for a rank loss and account for the realized date weights.

### E10. Fund-flow pressure from public holdings

**Hypothesis.** Redemptions and subscriptions can create predictable demand for securities held by investment funds. The effect may persist for several sessions and differ from ordinary momentum or fundamental information.

CVM daily fund reports contain net asset value, subscriptions, and redemptions. Monthly portfolio disclosures identify holdings, subject to confidentiality and publication delays. Official catalogues link long archives, but today's historical files also contain revisions and delayed filings. A holdings reference month is not its public-availability month.[^fund-daily][^fund-holdings]

Build an initial pressure measure only from **last publicly available holdings** and **already published flows**, scaled by the equity's causal liquidity. Do not use the later realized portfolio to infer what the fund must have traded earlier. Retain the age of the disclosed holdings; an old composition is not a known current position.

First deliver a coverage/timing table, then test whether the measure has incremental conditional predictive information in a few prespecified holding windows. Include flow continuation and reversal as distinct hypotheses with a fixed sign/timing rationale; do not retrospectively choose whichever window works.

Evidence on mutual-fund fire sales supports the mechanism, but does not settle whether publicly delayed Brazilian data is early enough to trade. This ranks below earnings primarily because recovering the historical information set and mapping funds/holdings is harder. If representative data show that publication delay consumes the entire effect, stop before building a full pipeline.[^flows]

### E11. Longer memory and more history

There are three different interventions:

1. Longer input context for each prediction.
2. More historical observations in a training window.
3. New markets/assets for representation pretraining.

They should not be sold as one generic “more data” improvement.

**Longer context:** compare 120 versus 60 completed sessions with the same features, target, and training recipe. This adds the trajectory of older information; it does not introduce the first annual momentum feature. If daily long context is too costly, a later two-resolution design can retain recent daily data and older weekly summaries, with causal aggregation and explicit missingness. Do not use centered filters.

**Earlier Brazilian history:** official COTAHIST history extends back to 1986, but the current model already uses 2010–2016 pretraining and later expanding fine-tuning. Pre-2010 prices bring action, identity, liquidity, and regime problems; nominally more rows may contribute less relevant information than a better use of existing data. Establish an improvement from learning-window choices before paying for a large earlier-history reconstruction.[^b3][^training]

**Cross-market pretraining:** it may supply more independent episodes, but point-in-time universes, corporate actions, market structure, currencies, and calendars are a substantial new contract. Publicly downloadable current-stock histories are not a survivor-free global equity dataset. This is a later program, not the next high-value experiment.

Do not use the reserved 2025 performance window as an inexpensive extension of training while presenting it as an independent confirmation year. Any eventual transition from validation data to training data needs an explicit new evaluation contract.

## 4. Free-data opportunities and their actual limitations

“Free” here means that an official public source was identified, or that the repository already records a relevant acquisition. It does not mean that a clean historical point-in-time table, redistribution rights, and a tested adapter already exist. This review inspected source documentation and repository evidence; it did not download or audit every historical data file.

| Dataset | Frequency / verified history boundary | Proposed use | Availability and quality constraint | Priority |
| --- | --- | --- | --- | --- |
| CVM ITR financial statements + IPE filings | Quarterly statements and timestamped documents; ITR directory lists 2011 onward | Earnings/revenue surprise and event age | Annual files are revised; recover first public versions and account semantics | High, feasibility-gated |
| B3 published index changes | Event-driven; V2 has mapped IBOV/IBXX/SMLL fields | Anticipated passive demand around effective dates | Verify historical announcement availability and valid coverage; never use eventual membership early | High if existing archive passes |
| B3 lending | Daily; V2 useful balance coverage begins in 2022, rate coverage in 2023 | Crowding, short demand, financing | A short recent sample; costs and alpha inputs have different roles | Medium–high, targeted |
| BCB PTAX bulletins | Several bulletins per day / daily history; exact usable span needs audit | Currency shocks and issuer exposures | Reference-rate construction and publication time; not an executable FX quote | High within E5 |
| Cboe VIX | Daily closing history from 1990 | Risk-state conditioning | Same-date US close may be unavailable at the Brazil decision | Medium within E5 |
| EIA oil spot prices | Daily historical series | Oil-linked issuer response | Assessment/publication lag and series-specific coverage; not a futures return | Medium within E5 |
| FRED/ALFRED rates and macro | Daily or release-frequency, depending on series | Global rates/risk and release surprises | Use vintage options where available; verify timestamps and underlying-series terms | Medium, few series |
| BCB Focus expectations | Statistics have daily reference dates; official public dissemination is weekly | Changes in expected inflation/rates/FX | Do not expose each reference-date estimate before its weekly publication | Later; limited short-horizon novelty |
| CVM fund daily reports + monthly holdings | Catalogues link daily-report history since 2000 and holdings since 2005 | Flow pressure mapped to stocks | Late/revised reports and confidential holdings; historical public versions essential | High upside, later engineering |
| ONS reservoir/energy data | Daily stored energy; hourly hydraulic observations | Utility-specific supply and hydrology surprises | Hourly data can have gaps and later corrections; exact archival availability needs audit | Specialist, later |
| CCEE PLD | Hourly from 2021; official dataset also provides weekly 2001–2020 history | Electricity-price exposure | Day-ahead setting and publication timing; weekly and hourly regimes are different | Specialist, later |
| CVM FRE company structure | Annual/event-updated reference filings | Dated share counts, industry, capital structure, exposures | Resubmissions and limited catalogue windows; reconstruct share-class/issuer state | Enabling data, selectively |
| SHFE/ADR sources recorded in the old program | Existing historical acquisition, coverage not re-audited here | Commodity and overseas lead/lag | Resolve canonical archives; verify time zones, rolls, corporate actions, and prior feature definitions | Conditional reuse before new collection |

Sources for the catalogue: CVM ITR/IPE/FRE and fund documentation; BCB PTAX and Focus; Cboe; EIA; FRED/ALFRED; ONS; CCEE; and the repository's feature/source records.[^cvm-itr][^cvm-itr-directory][^cvm-ipe][^cvm-fre][^fund-daily][^fund-holdings][^ptax][^focus][^vix][^eia][^alfred][^ons][^ccee][^context][^sidecars]

The energy data are especially relevant to Brazil, but a common PLD increase does not have a uniform positive effect on every electricity company. Generation mix, contracted sales, hydrological exposure, retail obligations, and regulation can change the sign. Without defensible issuer exposures, a large hourly energy ingestion is likely to be an expensive weak feature. Start with a few documented exposures and daily information available at the decision; hourly data alone does not imply that an hourly forecasting model is needed.

Free-data projects should begin with a small acceptance table: earliest usable publication, annual issuer/name coverage, missingness, revisions, economic units, identity mapping, and the source observations consumed at each decision. These checks prevent leakage and false dataset claims; they are not a request for production monitoring or generic stale-input machinery.

## 5. Variance reduction: useful distinctions

There are at least four different kinds of variance here, and they need different treatments.

| Source of variation | Suitable intervention | What does not solve it |
| --- | --- | --- |
| Random optimization | Matched seed ensembles, possibly archived EMA | Adding unrelated data without controlling the fit |
| Model misspecification | Different model families, better inputs, better inductive structure | Many near-identical seeds |
| Economic outcome noise | More independent historical episodes, paired block inference, stable portfolio policies | Counting overlapping horizons or names as independent samples |
| Search/selection noise | Fewer predeclared candidates, nested calibration, independent final evaluation | Repeatedly inspecting the same development history |

For equally variable errors with common pairwise correlation rho, the variance of a K-member mean is proportional to `rho + (1-rho)/K`. This simple illustration explains why diversity can matter more than another three seeds. It is not an estimate of this model's error correlation, and near-equal IC across seeds does not itself prove that correlation is high.

**Archived EMA:** the training code already saves `final_ema.pt`. A fixed final-EMA versus raw-Patience comparison on S0 can therefore avoid new training if all required checkpoints and prediction inputs are present. It still needs inference and evaluation. Do not search EMA decay rates or average arbitrary missing epochs. Final EMA and raw Patience use different trajectory summaries, so report the exact selection rules.[^training]

The earlier V1 checkpoint-averaging experiment lost on both folds and its checkpoint investigation was closed. This is a reason to keep an optional V2 test small, not to reopen the old V1 campaign. General weight-averaging research supplies motivation, not evidence that averaging SAM-trained financial models will help.[^context][^swa]

**Feature and model simplification:** reducing redundant inputs or using a smaller model can reduce estimation variance, especially with only a few thousand substantially dependent decision dates. Use a coherent block ablation informed by information content, rather than removing features based on a full-sample importance ranking and then calling the result out-of-sample. Keep feature masks and source integrity protections.

**Checkpoint-selection noise:** Stage-F's 55-session selection window contains strongly overlapping multiday outcomes. If training trajectories show unstable peak selection, a fixed longer selection window or a less adaptive checkpoint rule is a possible later test. A longer window also removes recent observations from fitting, so the tradeoff must be measured. Do not introduce another checkpoint grid merely because an averaged trajectory looks smoother after the fact.[^training]

**Uncertainty-based sizing:** ensemble disagreement may be useful as a calibration input, but a three-seed ensemble can share the same blind spot. Before reducing positions based on disagreement, show that higher disagreement predicts worse realized calibration on earlier data, beyond volatility and liquidity. Otherwise the rule may merely reject difficult but profitable stocks.

## 6. Portfolio structure and intraday intervention

If E1 succeeds, two structural portfolio tests become worthwhile.

First, replace rigid counts per volatility quintile with a small, transparent set of continuous risk budgets. Fixed counts can require weak trades in one bucket while rejecting stronger candidates in another. But the current occupancy rules are part of the registered contract: changing them requires a new experiment with explicit gross, name, sector, factor, liquidity, and borrow controls. A prior failed occupancy check is not permission to remove the check retrospectively.

Second, test a causal shrunk covariance/factor risk estimate so highly correlated positions are recognized as overlapping bets. Compare at matched volatility, gross exposure, and capital; retain the simple current hedge benchmark. More holdings are not automatically more independent breadth, particularly for related share classes or companies with the same commodity exposure. The first implementation should remain an offline research allocator, not a live portfolio-management framework.

These are plausible ways to obtain more value from the same forecasts. They also carry greater evaluation complexity than a fixed-score blend, which is why I would not start by jointly optimizing forecasts, risk, leverage, and execution.

Intraday intervention remains useful even though the broad fast input stream did not demonstrate multiday forecast value. Maintain a separation between the slow expected-return estimate and a separately evaluated decision to reduce exposure, defer an order, or react to a verified event. The highest-priority uses are executable-price/cost measurement and conditional risk response, with all-day P&L and missed-opportunity cost recorded.

Do not reintroduce the full native fast stream under a new name. A future fast experiment should state a narrow hypothesis, such as whether a public issuer event or large market shock makes the outstanding slow forecast conditionally stale. Evaluate it on predetermined event definitions with information available at the intervention time. A stop-loss can reduce drawdown and still reduce expected return; assess both. Correct risk handling need not generate an IC improvement to be useful, but should not be reported as new forecasting alpha.

## 7. Ideas I would defer

| Idea | Why it is not first |
| --- | --- |
| Automatically expand from three to six or more seeds | Reuses the same information and model family; current ranking choice already survives all three omissions |
| Broad persistence/smoothing search | Already studied in R3.1 and Round 4; less turnover did not reliably produce a better model |
| Add many conventional technical indicators | Most summarize information already present; raises search flexibility without a clear new information source |
| Large Transformer/GNN or general-purpose time-series foundation model | More capacity and complex pretraining provenance before establishing a representation bottleneck |
| Cross-market or decades-earlier pretraining | High data-contract cost; uncertain transfer; current old pretraining has not yet been isolated |
| Generic news sentiment or retrospective LLM stock scoring | Historical text coverage and release timing are difficult; a pretrained model may encode later company outcomes |
| Synthetic price paths or unconstrained data augmentation | Can break action, mask, cross-stock, and market-state relationships; no demonstrated shortage of this kind of training variation |
| Broad hyperparameter/Bayesian optimization | Consumes development evidence and compute before strong economic hypotheses are exhausted |
| Automatic target outlier deletion | Can delete genuine crashes, delistings, and exactly the risks the model must handle |
| Make the universe more liquid and call the new result alpha | Changes the population; use a matched restricted baseline and separate original-trade attribution |
| Remove borrowing/impact constraints to expose more “edge” | Changes the economic question; report gross predictability and implementable economics separately |

Text research is not permanently ruled out. A small timestamped event taxonomy or extraction of explicit financial facts is a more controlled starting point than asking a modern LLM whether a historical named company was a good investment. Any model used for historical NLP requires an explicit temporal-contamination assessment.

## 8. A research protocol that is fast enough to use

### Protect the remaining independent evidence

The 2018–2024 folds have already been inspected repeatedly. A new walk-forward experiment on them is still **development research**, even if every individual prediction is chronologically valid. A clean implementation does not undo human or LLM selection based on previously observed results.

The official 2025 performance read remains unspent under the current contract. Historical 2026 test evidence is already spent. Earlier integrity-only later-payload reads were disclosed; “no held-out performance evaluation” must not be inflated into a literal claim that no later payload was ever accessed. This review did not fit or evaluate on either period.[^context][^r4]

Keep the reserved performance read closed while selecting these experiments. Freeze a short final candidate list and its decision rule before requesting any future independent evaluation. A genuinely new prospective period can provide further evidence, but only relative to a documented model/decision freeze. Do not invent fresh folds by relabeling used dates.

Multiple-testing research is directly relevant to this program. Adding a candidate, a different horizon, a subset, a weighting scheme, or a portfolio variant creates additional selection opportunities. A nominal 95% interval after a broad search is not a 95% guarantee about the chosen winner.[^testing]

### Use two kinds of improvement criteria

For a **forecast experiment**, primary evidence is a paired change in the existing D3/D5/D10 neutral IC, with consistent raw/shareholder-return and after-cost diagnostics. For a **portfolio experiment with frozen forecasts**, primary evidence is net improvement at matched risk/capital; an IC gain is neither required nor expected.

As proposed planning thresholds, an absolute IC gain around **0.002** is about 7.5% of S0's current IC, while **0.5–1.0 bps/day** after costs could be economically meaningful. These are not forecasts or already accepted gates. At 0.5 bps/day, simple multiplication by 252 sessions gives 1.26 percentage points of annual NAV return before compounding; feasibility depends on the actual risk/cost/capital convention. Set the final materiality thresholds before scoring each experiment.

A promising point estimate is a screening result. Promotion needs uncertainty, robustness, practical magnitude, and a defensible multiple-comparison treatment. If the evidence is imprecise, label it inconclusive rather than increasing seeds or trying subsets until a desired sign appears. For a registered family with multiple formal acceptance tests, specify a family-level adjustment or a fixed hierarchical testing order; selection rules should not be invented after the tables are seen.

### Keep fitting and calibration chronological

1. Reuse S0 checkpoints, scores, and immutable features when the experiment leaves their contract unchanged.
2. Fit any score-to-return mapping, ensemble weights, risk model, scaler, or feature selection on legally earlier data. Predictions used to train a second-stage calibrator must themselves be out-of-fold; in-sample neural predictions are not a safe shortcut.
3. Purge training/selection/calibration boundaries for overlapping labels. Add any necessary information embargo explicitly; do not treat a mature date's incomplete D10 outcome as already known.
4. Apply the same evaluation population and label masks to paired candidates. Show full-calendar economics, full development IC, and predeclared availability subsets together.
5. Reuse the registered paired 20-session block bootstrap with fold boundaries for comparable screening results. For finalists, add a predeclared longer-block sensitivity and an era/concentration check. The existing turnover interval underweights fold-boundary spikes; do not use that interval as precise evidence for an execution gain.
6. For neural finalists, retain matched seeds 11/29/47 and all three leave-one-seed-out diagnostics. A fixed three-seed design gives useful optimization sensitivity; it does not provide six-seed evidence or independent market histories.
7. Log every attempted arm and its intended decision. No secret failed variants, no unreported feature-selection sweep, and no post-result promotion-rule changes presented as preregistered.

### Spend compute on hypotheses, not repeated bookkeeping

Round 4 measured about **111 minutes for 42 parent fine-tuning fits** on its GH200 setup, while 210 arm fits took about **9.04 hours**. The twelve pretraining runs took about 41 minutes. These are reference measurements, not promises for a different graph, patience trajectory, or host.[^r4]

A single comparable neural arm means 14 folds × 3 seeds = 42 fine-tuning fits, plus matching pretraining if required. At the prior throughput, a rough training budget is around two hours per arm, with uncertainty and additional inference, CPU evaluation, and recovery time. Graph-changing arms can cost more. Use measured short runs to refine the estimate before committing a paid session.

For cheap saved-score experiments, compute the compact IC/spread/cost screen first. Run expensive full ledger and diagnostic projections only for the fixed viable roster, while preserving whatever checks protect material correctness. Cache immutable inputs and deterministic outputs; do not recompute identical books for each textual report. If a screen filters candidates, that filtering is part of the recorded selection procedure.

For a larger neural change, a one-seed, predetermined multi-era screen can reject a plainly bad idea quickly. It is exploratory and cannot promote the candidate. A surviving candidate must complete the matched 14-fold, three-seed comparison; do not retain only favorable screen folds or change the hyperparameters in response without recording another experiment. For the especially cheap E4 multiplier test, going directly to the complete paired comparison may be cleaner.

More GPU utilization does not imply shorter total runtime if CPU replay or data transport is the bottleneck. Profile those separately. Precision changes, batching, and concurrency are engineering optimizations with their own parity checks, not sources of alpha. Paid sessions should have a bounded queued workload, followed by verified recovery and exact-instance termination under the existing operations contract.

### Suggested first campaign

| Stage | Work | Maximum initial candidate scope | Decision |
| --- | --- | --- | --- |
| A | Audit dominant economic errors; establish continuous replay and timing limitations; verify available score/checkpoint coverage | No alpha search | Decide which economic comparisons are credible |
| B | E2 fixed blends; optional fixed archived-EMA comparison; calibrator feasibility for E1 | Two blends, at most one EMA rule | Retain only candidates with a clear paired benefit or informative diversification |
| C | E4 fine-tuning multiplier; E6 selected magnitudes | Two separate neural arms | Determine whether current training/representation is restrictive |
| D | Earnings publication/version feasibility and a small E3 event experiment | Timing control plus one signed-content candidate | Establish actual new-information value |
| E | Choose E5 global exposures or E7 existing sidecar based on source readiness | One family, one compact arm | Avoid launching a broad external-data tournament |
| F | Combine independently useful changes, one interaction at a time | One proposed combined parent | Verify additivity, risk, economics, and seed sensitivity |

Stages A and the data-feasibility work can overlap. This is a budgeted sequence, not an instruction to run every listed experiment. Re-rank after substantive evidence. Do not let data engineering for an unavailable filing archive block inexpensive completed-score comparisons.

If only three research bets can be funded, I would choose **cost-aware use of S0**, **diverse saved-score combinations**, and **point-in-time earnings information**. If the execution audit is not yet ready, substitute the **fine-tuning multiplier test** as the immediate third experiment while the earnings data work continues.

## 9. Questions for the reviewing LLM

Review this proposal against the pinned repository evidence. Treat all experiment designs and rankings as proposals, not established facts or authorization to execute them.

1. Which of the first five priorities has the weakest causal mechanism or the least credible path to a material gain? Propose a better replacement and explain its cost.
2. Does any recommendation duplicate an implemented feature, tested arm, or previously closed experiment? Distinguish V1 intraday evidence from the exact V2 S0 setting.
3. Are the proposed cost-aware calibration and blending procedures valid at each historical decision timestamp? Identify any hidden use of in-sample predictions, overlapping future labels, or future publication versions.
4. Does the dataset shortlist exaggerate what is freely available point-in-time? Identify the minimum availability evidence needed before each dataset can be tested.
5. Which changes could raise reported IC or P&L without improving economically meaningful stock selection? Check leverage, population changes, terminal treatment, risk-factor exposure, and target redefinition.
6. Is the event experiment sufficiently different from the existing filing-age field and old intraday screens? Would a seasonal accounting surprise be timely enough at D3–D10 without analyst consensus?
7. Is one residual attention block a better use of compute than more informative data or the simple-model comparator? Explain the missing interaction it would learn.
8. Is the screening and promotion procedure proportionate to 14 dependent development folds and a repeatedly used research period? Suggest a specific limited candidate budget and family-level decision rule.
9. Return a revised top-five list, the single first experiment, concrete rejection criteria, and the strongest argument against your own recommendation. Do not return an unranked catalogue of model types or datasets.

## Sources and repository evidence

Repository links are pinned to the reviewed commit, so later changes cannot silently alter the evidence behind this proposal. External sources motivate methods or establish dataset availability; none proves a Brazil-RV improvement. All external pages below were checked on 10 September 2026. Numerical results in this document come from repository artifacts, not external studies.

[^r4]: Brazil-RV, [Round 4: execution and development evidence](https://github.com/gabrool/brazil-rv/blob/d6679e1331239fa21d866e10f7170691597edbf4/docs/v2_ROUND4.md). Includes model levels, paired comparisons, informative subsets, limitations, training timings, A4/A4.1, and final S0 designation. The [screening JSON](https://github.com/gabrool/brazil-rv/blob/d6679e1331239fa21d866e10f7170691597edbf4/docs/v2_round4_screening_report.json) supplies the underlying summary projection.

[^audit]: Brazil-RV, [Round-4 seed-audit report](https://github.com/gabrool/brazil-rv/blob/d6679e1331239fa21d866e10f7170691597edbf4/docs/v2_round4_seed_audit_report.json). Full and leave-one-seed-out decision/effect evidence.

[^cpu]: Brazil-RV, [Round-4 CPU replay result](https://github.com/gabrool/brazil-rv/blob/d6679e1331239fa21d866e10f7170691597edbf4/docs/v2_round4_cpu_replay_result.json). Values are from `readouts.<candidate>.pooled.primary_neutral_target_ic` and `headline_net_excess_bps`, not legacy metrics or the former three-window slice.

[^context]: Brazil-RV, [PROJECT_CONTEXT.md](https://github.com/gabrool/brazil-rv/blob/d6679e1331239fa21d866e10f7170691597edbf4/PROJECT_CONTEXT.md). Current accepted state, immutable boundaries, historical V1 external-data/checkpoint decisions, and held-out-access qualifications. Its historical V1 passages must not be mistaken for the current V2 execution specification.

[^features]: Brazil-RV, [feature contract](https://github.com/gabrool/brazil-rv/blob/d6679e1331239fa21d866e10f7170691597edbf4/research/src/brazil_rv/v2/contract.py), [feature transforms](https://github.com/gabrool/brazil-rv/blob/d6679e1331239fa21d866e10f7170691597edbf4/research/src/brazil_rv/v2/feature_spec.py), and [feature construction](https://github.com/gabrool/brazil-rv/blob/d6679e1331239fa21d866e10f7170691597edbf4/research/src/brazil_rv/v2/features.py).

[^model]: Brazil-RV, [V2 model implementation](https://github.com/gabrool/brazil-rv/blob/d6679e1331239fa21d866e10f7170691597edbf4/research/src/brazil_rv/v2/model.py).

[^training]: Brazil-RV, [V2 training implementation](https://github.com/gabrool/brazil-rv/blob/d6679e1331239fa21d866e10f7170691597edbf4/research/src/brazil_rv/v2/train.py) and [Round-4 registration JSON](https://github.com/gabrool/brazil-rv/blob/d6679e1331239fa21d866e10f7170691597edbf4/research/preregistrations/v2_round4.json). Pretraining/fine-tuning boundaries, transferred-parameter learning rates, checkpoint artifacts, and selection settings.

[^targets]: Brazil-RV, [V2 store and characteristic-neutral targets](https://github.com/gabrool/brazil-rv/blob/d6679e1331239fa21d866e10f7170691597edbf4/research/src/brazil_rv/v2/store.py).

[^sidecars]: Brazil-RV, [V2 sidecar adapters](https://github.com/gabrool/brazil-rv/blob/d6679e1331239fa21d866e10f7170691597edbf4/research/src/brazil_rv/v2/sidecars.py). `ARCHIVE_COLUMN_MAP` distinguishes supported fields from fields with no source mapping.

[^execution]: Brazil-RV, [R3.1 execution sweep](https://github.com/gabrool/brazil-rv/blob/d6679e1331239fa21d866e10f7170691597edbf4/docs/v2_R31_EXECUTION_SWEEP.md). The current horizon/buffer/sizing choice was selected in sample.

[^gp]: Gârleanu, N., and Pedersen, L. H., [Dynamic Trading with Predictable Returns and Transaction Costs](https://www.nber.org/papers/w15205), working paper revised 2013; published in the Journal of Finance. Motivation for trading according to expected returns, signal decay, existing holdings, and costs.

[^boyd]: Boyd, S., et al., [Multi-Period Trading via Convex Optimization](https://web.stanford.edu/~boyd/papers/cvx_portfolio.html), Foundations and Trends in Optimization, 2017. A transparent framework for return/risk/trading/holding-cost tradeoffs; does not solve forecasting.

[^frontier]: Jensen, T. I., Kelly, B., Malamud, S., and Pedersen, L. H., [Machine Learning and the Implementable Efficient Frontier](https://academic.oup.com/rfs/advance-article/doi/10.1093/rfs/hhag022/8524346), Review of Financial Studies, 2026. Motivation for evaluating net performance at comparable risk and integrating trading costs into portfolio decisions.

[^b3]: B3, [Historical equity quotations](https://www.b3.com.br/pt_br/market-data-e-indices/servicos-de-dados/market-data/historico/mercado-a-vista/cotacoes-historicas/). Official history from 1986; historical prices are not adjusted for dividends or other corporate actions.

[^cvm-itr]: CVM, [ITR open-data catalogue](https://dados.cvm.gov.br/dataset/cia_aberta-doc-itr). Structured quarterly statements and resubmission/update behavior.

[^cvm-itr-directory]: CVM, [ITR annual data directory](https://dados.cvm.gov.br/dados/CIA_ABERTA/DOC/ITR/DADOS/). Lists 2011 and later annual archives; file existence and recent modification dates do not establish original publication vintages.

[^cvm-ipe]: CVM, [IPE document catalogue](https://dados.cvm.gov.br/dataset/cia_aberta-doc-ipe). Public-company periodic/eventual document records.

[^cvm-fre]: CVM, [FRE reference-form catalogue](https://dados.cvm.gov.br/dataset/cia_aberta-doc-fre). Structured company information with resubmission/update considerations.

[^pead]: [Post-Earnings Announcement Drift in Latin America](https://www.scielo.br/j/rbgn/a/FB4tKLFBHggTVRDhpLygFWB/?lang=en), Revista Brasileira de Gestão de Negócios, 24(3), 2022, DOI 10.7819/rbgn.v24i3.4193. Regional event-study evidence, including Brazil; not a replication of this strategy or proof of tradable short-horizon effects.

[^ptax]: Banco Central do Brasil, [PTAX: all daily bulletins, API documentation](https://www.bcb.gov.br/conteudo/dadosabertos/BCBDepin/gnastportal-dados-abertostaxas-de-cambio---todos-os-boletins-diarios.pdf). Official reference-rate bulletin service.

[^vix]: Cboe, [VIX historical data](https://www.cboe.com/tradable_products/vix/vix_historical_data). Official daily closing history from 1990.

[^eia]: U.S. Energy Information Administration, [Spot Prices for Crude Oil and Petroleum Products](https://www.eia.gov/dnav/pet/pet_pri_spt_s1_d.htm). Daily historical oil/product series; publication timing and assessment definitions must be respected.

[^alfred]: Federal Reserve Bank of St. Louis, [FRED API real-time periods](https://fred.stlouisfed.org/docs/api/fred/realtime_period.html). Defaults describe today's information; real-time options retrieve historical vintages where available.

[^focus]: Banco Central do Brasil, [Market expectations dataset](https://dadosabertos.bcb.gov.br/en/dataset/expectativas-mercado). Daily-reference survey statistics and weekly public dissemination.

[^fund-daily]: CVM, [Daily fund reports](https://dados.cvm.gov.br/dataset/fi-doc-inf_diario). Fund assets, subscriptions, redemptions and historical/report-update conventions.

[^fund-holdings]: CVM, [Fund portfolio composition and diversification](https://dados.cvm.gov.br/dataset/fi-doc-cda). Monthly holdings, confidentiality, delayed disclosure, and historical resources.

[^flows]: Coval, J., and Stafford, E., [Asset Fire Sales (and Purchases) in Equity Markets](https://www.hbs.edu/ris/download.aspx?name=Asset+Fire+Sales+in+Equity+Markets.pdf), Journal of Financial Economics, 86, 2007. Primary evidence for fund-flow-induced price pressure; not evidence of timely tradability from the proposed Brazilian public data.

[^ons]: ONS, [Daily stored energy by reservoir](https://dados.ons.org.br/dataset/ear-diario-por-reservatorio) and [Hourly hydraulic reservoir data](https://dados.ons.org.br/dataset/dados_hidrologicos_ho). The hourly source explicitly notes gaps, corrections, and publication updates.

[^ccee]: CCEE, [Historical hourly PLD dataset](https://dadosabertos.ccee.org.br/dataset/pld_horario). Hourly archives from 2021 and weekly history for 2001–2020; day-ahead price setting.

[^trees]: Grinsztajn, L., Oyallon, E., and Varoquaux, G., [Why do tree-based models still outperform deep learning on typical tabular data?](https://arxiv.org/abs/2207.08815), NeurIPS 2022. Broad tabular benchmark evidence, not a financial-sequence result.

[^set]: Lee, J., et al., [Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks](https://proceedings.mlr.press/v97/lee19d.html), ICML 2019. Architectural motivation for interactions among unordered entities; the stock-prediction output must preserve per-security equivariance.

[^master]: Li, T., et al., [MASTER: Market-Guided Stock Transformer for Stock Price Forecasting](https://arxiv.org/abs/2312.15235), AAAI 2024. Motivation for selective temporal/cross-stock interaction and market conditioning.

[^swa]: Izmailov, P., et al., [Averaging Weights Leads to Wider Optima and Better Generalization](https://arxiv.org/abs/1803.05407), UAI 2018. Weight-averaging motivation; its optimization setting does not establish a gain for this SAM/EMA recipe.

[^testing]: Harvey, C. R., Liu, Y., and Zhu, H., [... and the Cross-Section of Expected Returns](https://academic.oup.com/rfs/article-abstract/29/1/5/1843824?login=false), Review of Financial Studies, 2016. Multiple-testing concerns in asset-pricing research; the experiment-specific procedure still needs to be registered.
