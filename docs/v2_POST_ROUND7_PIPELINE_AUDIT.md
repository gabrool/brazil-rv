**Post-Round-7 pipeline audit — 13 September 2026**

Prepared for independent LLM review. This is a diagnosis of the completed experiments, not a new experiment result or permission to reinterpret a losing candidate as a winner. The audit starts from commit `02013fdb249101d321eed9569cdeb2474df1e36f`. Machine-readable measurements, source hashes, and reproduction details accompany this document in [the audit evidence](v2_post_round7_audit_evidence.json).

**Main finding**

There are real problems to repair. The strongest explanation in the saved training evidence is **severe overfitting, compounded by a checkpoint/budget policy that excluded the useful early part of training**. The richer models are capable of fitting the training relationships. They do so much more strongly than the simpler model, but that performance fails to transfer to later dates.

The data audit separately establishes actual fundamental-data errors, remaining unnecessary rank-support exclusions, inconsistent input scaling, and a misleading compression of fundamental freshness information. These are actionable findings, even though we have not established how much each contributes to the performance gap. The confirmed source errors affect a relatively small number of observations in the examples traced; they cannot, by themselves, explain every losing architecture. Slow-only architectural variants also deteriorate.

My earlier Round-7 review should have foregrounded the training/selection gap and early calibration peak. Its discussion of possible undertraining was insufficiently informed by the saved curves. This audit corrects that interpretation. A0 remains the valid winner of the completed protocol; the protocol does not establish that richer architectures or additional data are intrinsically unsuitable.

**What was inspected**

The audit resolved the canonical Round-7 store through `docs/v2_round7_inputs.json`, followed its source-family ancestry, inspected all 152 per-security scalar channels and their masks/ages, read 336 saved R-recipe fine-tuning histories and the 12 calibration histories, and ran frozen-checkpoint gradient/input interventions. It traced four anomalous issuers into 137 own-version CVM documents and independently inspected relevant filing pages. It also reviewed feature transformations, target construction, data access, model assembly, SAM, checkpoint selection, and the two papers' experimental contracts.

The store has 3,717 dates through 2024, 933 historical identities, and at most 243 active names. The 152-field count includes slow features, sidecars and native fundamentals; the separate three common diagnostic fields are additional. This audit used development history only. No new fit, forward capture, GPU instance, held-out 2025/2026 access, canonical source mutation, or sealed-score replacement occurred.

The existing targeted model/data suite passed 59 tests; an additional targeted source/transform/store suite passed 123 tests. All 152 audited fields had zero non-finite values or negative ages among valid active observations. Passing these checks supports their specific invariants. It does not certify the economic truth of vendor observations or filing tables.

**1. The richer models learn the training set and then generalize poorly**

The following are equal-fit averages over the same four screen folds, F2/F6/F10/F14, and seeds 11/29/47: twelve fits per cell. Both columns use the saved clean evaluation-mode D3/D5/D10 IC readout, including the common-head eligibility requirement. Training IC is measured on the full training loader, with dropout off. These are final-epoch diagnostics, not the official tail-averaged out-of-sample score panels.

| Cell | Meaning | Clean training IC | Selection IC |
| --- | --- | ---: | ---: |
| A1 | S0, slow inputs, recipe R, rho .125 | .0743 | .0367 |
| A2 | S0, slow inputs, recipe R, rho .05 | .1006 | .0348 |
| A3 | S0, all inputs, recipe R, rho .05 | .2217 | .0376 |
| B1 | Characteristic model, slow inputs, rho .125 | .1831 | .0311 |
| B3 | Characteristic model, slow inputs, rho .05 | .2763 | .0214 |
| B4 | Characteristic model, all inputs, rho .05 | **.7689** | **.0045** |
| B5 | B4 with rho .125 | .6771 | .0150 |
| B8 | B4 with Pearson-on-ranks loss | .7676 | .0074 |
| B9 | All inputs, final cross-stock attention | .7766 | -.0020 |
| B11 | All inputs, AdamW fine-tuning | .8110 | .0056 |
| GL | GRU, late peer interaction | .7968 | .0046 |
| GE | GRU, early peer interaction | .7914 | .0077 |
| TL | Temporal attention, late peer interaction | .8110 | .0038 |
| TE | Temporal attention, early peer interaction | .8076 | .0070 |

This is inconsistent with a general inability to optimize or discover relationships in-sample. It is consistent with memorization, unstable conditional relationships, excessively flexible interactions, and/or fitting source/coverage artifacts. The curves do not identify the relative contribution of those mechanisms. In particular, excellent training fit does not prove the learned relationships are economically meaningful.

The added current-state fields give the network distinctive combinations of issuer characteristics, reporting ages and market regimes. These can help identify recurring training observations even without an explicit security-ID embedding. Adjacent 60-day histories and overlapping multi-day labels are highly dependent. Hundreds of thousands of stock-days are not hundreds of thousands of independent market regimes.

The fixed 60-epoch pretraining also warrants revision. Across three `c1_all` parents, final clean training IC averages .6992 and selection IC .0066, versus .0167 selection IC at epoch one. This does not prove a particular earlier parent is best for transfer, but it rules out treating longer pretraining as an automatically helpful response.

**2. The calibration rule excluded the best early selection region**

The twelve B4 calibration fits used 60 epochs. Their mean selection IC peaked at epoch **3**, but the registered budget rule only admitted budgets **20, 25, …, 60**, comparing trailing-five means within that late range. It selected 20.

| Calibration epoch | Mean clean training IC | Mean selection IC |
| --- | ---: | ---: |
| 1 | .0837 | .00771 |
| 3 | .2881 | **.01191** |
| 5 | .4717 | .00260 |
| 10 | .6808 | .00406 |
| 20 | .8151 | -.00263 |
| 60 | .9041 | -.00116 |

This is a design flaw in the experiment, not a failure to execute its registration. [The rule](../research/src/brazil_rv/v2/round7.py) explicitly takes its maximum over epochs 20–60. [Training](../research/src/brazil_rv/v2/round7_training.py) then averages only the final quarter of the chosen budget. Tail averaging cannot recover a useful state abandoned much earlier.

There are two important qualifications. First, epoch three under a 60-epoch warm-up/cosine schedule is not equivalent to epoch three under a newly shortened schedule. Second, its .0119 selection IC is not evidence of beating A0. Selecting and rescoring archived early checkpoints would be a new, post-hoc development diagnostic, not a replacement for the sealed Round-7 result.

The fastest useful next experiment is therefore **checkpoint-policy analysis using saved histories and checkpoints**, before another broad fitting campaign. Apply a declared selection-only rule that permits early checkpoints, keep evaluation labels out of checkpoint choice, and compare it with the existing tail rule on matched fits. Freeze the rule before producing additional evaluation scores. Subsequent training should allow early stopping and separate the learning-rate schedule from an arbitrary minimum epoch budget.

**3. Actual fundamental-data errors survive the existing cleaning**

These are source inconsistencies propagated into accepted derived fields. The original filings were preserved correctly; the problem is accepting numerically parseable tables without sufficient semantic reconciliation.

| Case | Accepted derived observation | Independent source evidence | Finding |
| --- | --- | --- | --- |
| Auren, `BRAUREACNOR9`, 5 May 2023 | Earnings yield 184.834984; log market cap 16.57235 | Own ITR 125769, reference 31 March 2023: capital table says 1,000,000 shares; note 19.1 explicitly lists 1,000,000,000 shares with a shareholder-total reconciliation | **Confirmed 1,000-fold capital-count error.** The accepted HTML parser returns the smaller count. With the count corrected, the same earnings/price arithmetic would give roughly .184835. |
| BK Brasil, `BRBKBRACNOR4`, March–May 2018 | Revenue-growth ratio 1,279.311839 | Own DFP 70549, reference 31 December 2016: note 24 gives consolidated revenue 1,393,284 in thousands of reais. Parsed prior revenue is only 1,393,284 reais; 2017 revenue is 1,783,838,000 reais | **Confirmed account-currency scale error.** The corresponding annual growth arithmetic is about .280312, not 1,279.311839. |
| Três Tentos, `BRTTENACNOR0`, February 2024 | Earnings yield as high as 119.670139; book/market 696.390067 | Own DFP 134143: note 23 identifies 498,298 thousand issued shares; accepted parsed non-treasury count is 498,163 shares | **Confirmed order-of-magnitude capital inconsistency.** Reconcile treasury units in the same filing before issuing an exact corrected count. |
| Odontoprev, `BRODPVACNOR4`, July–October 2021 | Gross-profit/assets 250.841777 | Document 106749 carries assets 2,020,126 and first-half gross profit 550,995; adjacent filings' accounts are approximately 1,000 times larger. The TTM bridge mixes those scales | **Strongly supported unit-error candidate**, not yet independently certified from that filing's explanatory notes in this audit. |

The inspected PDFs' physical page numbers are Auren 2/57, BK Brasil 94, and Três Tentos 2/120. Their original archive paths and SHA-256 hashes are in the evidence. Official document viewers: [Auren 125769](https://www.rad.cvm.gov.br/ENET/frmGerenciaPaginaFRE.aspx?NumeroSequencialDocumento=125769&CodigoTipoInstituicao=1), [BK Brasil 70549](https://www.rad.cvm.gov.br/ENET/frmGerenciaPaginaFRE.aspx?NumeroSequencialDocumento=70549&CodigoTipoInstituicao=1), [Três Tentos 134143](https://www.rad.cvm.gov.br/ENET/frmGerenciaPaginaFRE.aspx?NumeroSequencialDocumento=134143&CodigoTipoInstituicao=1). The Auren and BK Brasil notes were visually checked against the parsed tables, not inferred just from a surprising ratio. None of these document IDs appears in the accepted correction dispositions used for those errors.

Threshold-based footprints on active stock-days are: BK Brasil growth above 100 on **36** days; Odontoprev gross profitability above 100 on **63** days; Três Tentos earnings yield above 10 on **8** days across two episodes; Auren earnings yield above 10 on **1** day. These counts identify obvious extremes, not the complete propagation footprint: bad accounts can contaminate later TTM, growth, accrual and SUE calculations without leaving an extreme final ratio.

Across all active issuers, four crude triage screens find 154 earnings-yield observations above absolute 10, 144 book/market observations above absolute 100, 427 gross-profitability observations above absolute 10, and 36 revenue-growth observations above absolute 100. They overlap and **are not a count of proven errors**. Distress, restructurings and small denominators can produce legitimate extremes.

Required repair: resolve exact own-filing currency and quantity units, including treasury quantities and consolidated/parent accounting basis; apply documented corrections at their original availability; rebuild affected dependent features in a new derived store. Check identities, balance-sheet relationships, fiscal-period bridges and approximate 1,000-fold discontinuities. Use these as review triggers, not universal deletion thresholds. Preserve unaffected fields when one account cannot be resolved. Native clipping and rank transforms limit magnitude; neither makes a wrongly ordered observation correct.

**4. Useful observations are still being removed by rank-support requirements**

[Feature transformation](../research/src/brazil_rv/v2/feature_spec.py) masks a ranked field when fewer than 20 eligible names have that field. Re-aligning the accepted physical source families to the canonical date/ISIN axis gives:

| Field | Physically available active stock-days | Lost in transformed channel | Share lost |
| --- | ---: | ---: | ---: |
| Lending utilization proxy | 16,779 | 9,747 | **58.09%** |
| Option/stock volume, 20 sessions | 251,487 | 21,352 | 8.49% |
| Put/call volume, 5 sessions | 233,984 | 20,335 | 8.69% |
| Put/call OI log ratio | 442 | 442 | **100%** |
| Change in OI / volume | 409 | 409 | **100%** |
| Name-minus-sector return, 5 sessions | 206,994 | 11,848 | 5.72% |
| Sector momentum | 187,951 | 10,721 | 5.70% |

All these measured losses occur on dates with fewer than 20 supported names. The lending percentage is for utilization, not for all lending data. Separately, uncovered-call share has no usable observations at source; removing a transform gate cannot create those observations.

Round 7 already restored many fundamental observations through native channels. The old ranked fundamental channels still lose roughly 13–14% of valuation observations, but that is **not** an additional loss of all that information from B4: the native fields recover most of it. Native book/market specifically logs only positive values; 1,990 physically available non-positive observations have no native log representation. The old ranked channel may still contain some of them. A signed representation or explicit negative-equity channel is preferable to treating economically meaningful negative equity as an ordinary missing measurement.

Required repair: use field-appropriate physical or robustly scaled representations for sparse fields, retaining value/mask/age. Keep a rank channel when useful, with its own support flag. Sparse partial option OI must remain explicitly partial; it must not be relabeled as complete exchange OI. Restoring support is not proof of predictive value and should not justify weakening identity or availability requirements.

**5. Input conditioning and freshness semantics need repair**

The current preprocessing is inconsistent across field types. [Round-7 preprocessing](../research/src/brazil_rv/v2/round7_preprocessing.py) uses fit-only median/IQR scaling and ±5 clipping for native fundamentals and the split common state. Magnitudes receive percentile clipping but retain their physical units. Per-security cross-market interactions and some event-day values have no comparable scaling before the first linear map.

Examples from valid active observations: five-session foreign flow multiplied by log volume has median 10.115 and 99th percentile 134.433; expected-filing day distance has median 32 and 99th percentile 90. Other shock/exposure fields are orders of magnitude smaller. Magnitudes mix daily volatility around hundredths with log traded value around tens. Value channels share the first projection with masks and bounded ages.

Layer normalization after a mixed projection cannot reconstruct dimensions overwhelmed before that projection. This establishes a conditioning risk, not proof that small fields are ignored: the intervention tests show meaningful sensitivity to several families.

There is also a pretraining-to-fine-tuning coordinate change. Scalers are re-estimated on the new fit window, while learned weights are copied unchanged. In B4 F14 seed 11, the native earnings-yield scale is 1.979 times its parent's scale; growth is 1.985 times, and gross profitability .608 times. This is fit-isolated and not leakage. It does mean the same learned coefficient sees a different numerical meaning. A stable causal transform, or a mathematically compatible first-layer remapping where possible, should be tested. Clipping complicates exact remapping. In that F14 training window, native ±5 clipping affects approximately 0–3% per field; there is no evidence that this clip alone destroys most native observations.

Fundamental age has another semantic problem. TTM/growth/SUE age often represents the **oldest dependency's availability**, not the time since the newest filing updated the statistic. The shared age encoder caps at 252 sessions. Among valid native observations, ages exceed that cap for 68.3% of earnings yield, 71.8% of gross profitability, 85.1% of accruals, 95.4% of growth and 99.94% of SUE. A newly updated SUE statistic can therefore have the same saturated age encoding as a much older one.

Required repair: distinguish update age, reference-period age, and dependency history where they carry different information. Keep long required histories for TTM/SUE; do not delete observations for exceeding an arbitrary freshness threshold. This is an encoding correction, not a request for stale-input rejection.

**6. Coverage and historical pretraining are not equivalent to having all datasets**

This table averages valid feature cells over all active stock-days in the full development store, including dates before a source exists. It includes flags and correlated channels; it is not a measure of independent information or conditional feature quality.

| Family | Fields | Mean valid-cell coverage | Main remaining issue |
| --- | ---: | ---: | --- |
| Slow | 32 | 97.11% | Mostly ranks; magnitude and regime information are only partly retained elsewhere |
| Cross-market | 65 | 80.35% | Mixed units, noisy estimated exposures, timing/revision limits of free archives |
| Events | 7 | 53.57% | Counts and timing, not event content; raw day-distance scaling |
| Ranked fundamentals | 12 | 42.48% | Source unit errors, support gating, basis/period semantics |
| Native fundamentals | 9 | 36.44% | Same underlying source errors, age saturation, positive-only log book/market |
| Lending | 7 | 16.42% | Late source coverage and utilization gate; balance is not a live borrow quote |
| Magnitudes | 4 | 99.34% | Clipping without common input scale |
| Microstructure | 2 | 71.95% | After-hours history absent from P; source/session distinctions matter |
| Odd-lot | 2 | 99.89% | High coverage is not proof of incremental multi-day signal |
| Options | 7 | 20.70% | Partial/absent OI and unnecessary rank-support loss |
| Rebalance | 2 | 11.48% | Limited dated preview history; pressure proxy also contains price drift |
| Sector | 3 | 33.25% | Late dated mapping and rank-support loss |

Pretraining ends on 30 June 2016. All lending fields, all sector fields, both rebalance fields, after-hours share, and several iron/pulp/foreign-flow/OI channels have zero valid P observations. A joint all-input graph was indeed pretrained, but it cannot learn relations involving data that do not exist in that era. Those channels effectively enter later fine-tuning without useful prior training on their values.

A further code-level loss of information occurs when the latest known filing lacks usable assets/equity: `fundamental_state` returns missing fields even when an older complete filing is already public. This audit did not quantify its full population footprint. Preserve separately identified last-known valid facts and expose the incomplete/new-filing state when appropriate; never fill from a later revision. Correct handling must be per field and accounting basis, not an indiscriminate forward fill.

**7. Non-stationarity, signal-to-noise, and missing representations**

Heteroscedasticity is partly addressed already: labels use causal volatility scaling; slow features are largely cross-sectional ranks; native fundamentals use robust fit-window scaling. It would be incorrect to claim the pipeline has no normalization. But ranks discard cardinal distance and common market variation, while unscaled physical interactions reintroduce unstable units. The objective can remove exactly the style component a new variable predicts.

The right response is a typed input contract: dimensionless returns/shocks where justified; sensible signed transforms for ratios; fit-only robust scales for continuous state; explicit binary/category/missingness representations; distinct common-state inputs; and unambiguous source/update ages. Use a historical decision's information set for every parameter estimate. Avoid full-sample detrending, centered filters, revised-data backfills, and smoothing that erases event shocks. Per-security normalization of a cross-sectional valuation feature can remove useful between-company information and should not be applied indiscriminately.

The model currently receives full temporal history primarily for the 32 slow fields. Added fundamental/event/market sidecars are current snapshots, sometimes already containing a few engineered windows. It cannot reconstruct arbitrary publication-response paths or long cross-market lead/lag relations absent from those snapshots. FiLM provides current common-state conditioning; it does not manufacture missing market history. The early-peer extension changes interaction within the supplied slow history, not this input-history boundary.

This is a real representational limitation, but expanding all sidecars to dense histories before solving overfitting would be premature. High-value candidates are causal changes/surprises and short release histories for a small number of cleaned families, or a compact common-market sequence with sufficient observed history. Their rationale is to supply information currently absent, not merely add indicators.

Prior univariate relevance and weak raw correlations are not proof of conditional incremental alpha. Signals may share a latent exposure, work in another horizon/universe, disappear after target neutralization, or reflect errors/coverage effects. Date and issuer dependence also inflate apparent sample size. Re-run incremental tests after repairs, controlling for A0 and the existing slow information on the same eligible population, with temporal blocks and issuer-aware checks. Treat those as screens, not deployment evidence.

**8. Architecture and SAM: what the tests establish**

Frozen CPU probes used A1/B4/B5/B9/B11/GE, F14 seed 11, and four complete training-date cross-sections. Gradients reached core/family encoders, fusion, context and the trunk. GE's temporal-peer component also received gradients. Missing rebalance fields on those four dates correctly had no value contribution. This does not certify every state or mixed-precision training step, but it rejects a universal disconnected-sidecar explanation.

Zeroing one family's values while preserving masks/ages changes B4's pooled scores substantially: approximate score correlations are .50 for events and .84–.89 for cross-market fields across inspected rich models. Native/ranked fundamentals also matter; odd-lot is almost inert in this bounded intervention. These are distribution-shifting sensitivity probes, not causal feature importance or new IC results.

Testing rho .05 was reasonable. Making .05 the common setting for the entire pathway extension was insufficient to establish that each architecture was well regularized. With all inputs, B5 at .125 fits less strongly and has better final selection IC than B4 at .05. But B5's official screen IC is **.005324**, below B4's **.008553**. Stronger SAM is therefore a candidate repair to test with checkpoint selection, not an already proven winner. In the slow-only controls, reducing rho also increases training fit while selection deteriorates. AdamW fine-tuning overfits strongly as well; removing SAM is not the leading recommendation.

Rho is a radius in parameter space, not a universal function-space regularization strength. Changing architecture and input scales changes its effect. The [SAM paper](https://arxiv.org/abs/2010.01412) defines neighborhood-based optimization; [ASAM](https://arxiv.org/abs/2102.11600) addresses sensitivity to parameter rescaling. This supports checking perturbation effects, not inferring an ideal rho from another architecture.

The frozen end-state probes at rho .05 preserve score correlations around .97–.99 in the richer models while increasing loss. They do not show a catastrophic global learning block at those checkpoints. They also do not settle early optimization dynamics or validate an optimizer from a synthetic test alone.

No particular projection, GRU or attention mechanism is proven to be the fundamental blocker. The rich model's nonlinear family encoders and residual trunk can fit complex relations. A current-state bottleneck and scale imbalance remain plausible; the dominant observed issue is transfer to later data. Increasing width/depth or replacing GRU with Mamba before correcting this would be poorly motivated.

**9. Objective and selection alignment need controlled examination**

[The target](../research/src/brazil_rv/v2/store.py) is not simply a stock's forward return. It subtracts the cross-sectional median, divides by causal volatility times square-root horizon, clips, residualizes against ten volatility groups, five beta groups and linear log ADV when support permits, and then ranks the residual. The nonlinear design starts at 40 names; smaller supported sections use a linear fallback.

This is a coherent idiosyncratic relative-value objective, but it deliberately removes some return variation predicted by fundamentals, commodity exposures or macro conditions. A dataset can forecast total returns or risk and add little to this target. Removing neutralization merely to improve headline IC would change the research question. Diagnose raw, scaled and neutral outcomes together, including exposures and costs, before deciding whether neutrality belongs in the target, portfolio construction, or both.

The differentiable rank loss gives each date/head a ranking objective over available names. Selection averages common-population D3/D5/D10 IC. The executed portfolio uses a composite score, position limits, persistence, borrow and costs. Average head IC is not mathematically identical to composite-score IC or net economic utility. Per-head training masks also differ from the common-head selection mask. B8 changes the surrogate to Pearson on ranks but retains the same underlying targets; its overfitting means it does not isolate target neutrality as the cause.

Start by reporting head-level and composite IC on identical populations, return/exposure decompositions, and marginal cost/turnover from existing score panels. Then test one loss/target change at a time on a stable training recipe. Avoid a noisy differentiable backtest or direct Sharpe objective as the first remedy.

A bounded label audit found one latent issue: exact ties in risk characteristics can be split across neutralization buckets by security ordering. A deliberately tied synthetic example changes 467/500 labels under permutation. However, **58,481 valid labels across 68 sampled real development dates had zero permutation difference**. This is a low-priority robustness defect, not evidence that it explains Round 7.

**10. Source timing and quality findings that remain bounded**

The reviewed code preserves several important safeguards: dated security identity and eligibility; exclusion of the entry bar; local target-window masks; fit-only scalers; explicit missingness; release-aware market snapshots; and distinction between current information and retrospective accounting. The compact security axis retains eligible names and does not crop the universe. I found no new general future-data leakage or security-axis mismatch in those reviewed paths.

Free-source provenance still limits certainty. US vendor historical prices can be revised; common multiplicative adjustment factors can cancel in returns, but that does not prove all corporate-action corrections or rounding revisions are harmless. Foreign-flow availability remains a receipt bound where an earlier publication was not established. Asia settlements and US closes need their actual local clocks and historical holidays; generic blanket extra lags are not a sound substitute. DI uses an end-of-day decision-equivalent availability bound, not a measured publication timestamp. These are disclosed source assumptions, not newly demonstrated errors.

Options' partial OI, issuer-class market-cap pricing, and rebalance price-drift contamination should remain explicit. Borrow balances/rates do not prove a historical stock-specific executable borrow offer. The inherited corporate-action/terminal valuation contract still leaves unresolved economics in some replay books. Those limits weaken precise net-alpha claims, but cannot explain a large failure already visible in pre-execution IC.

**11. The papers do not create an either-they-or-we contradiction**

MASTER uses Chinese CSI300/CSI800 data from 2008–2022, Alpha158 stock features, eight time steps, a five-day target and 63 index features. It tunes learning rates and model sizes and uses early stopping within 40 epochs; its selected learning rate is 1e-5. Its top-30 portfolio evaluation and target contract differ from our neutral multi-horizon, cost/borrow-constrained book. Its results motivate architectural hypotheses, not a promise of superior Brazil performance. Its early-stopping practice is directly relevant to the flaw identified here. [MASTER](https://arxiv.org/html/2312.15235v1)

HIGSTM reports CSI500/800/1000 experiments with 3,159 dates, 16-step inputs and 45 stated features, including prices, volumes, valuation, share counts and trade-size money flows. Crucially, its appendix defines stock lists using **31 December 2023 constituents**. I infer survivor conditioning from that historical-universe description; it differs from our dated eligibility. The paper also reports overfitting when hidden dimensions become too large. Its results do not establish that increasingly complex models should win here. We should neither replicate endpoint constituent selection nor borrow its headline IC as a target. [HIGSTM](https://arxiv.org/html/2503.11387v1)

Neither comparison independently validates our preprocessing. Conversely, finding defects here does not prove a particular literature architecture will outperform after repair.

**Recommended sequence, balancing quality and speed**

1. **Repair demonstrated data/encoding defects before the next full experiment.** Reconcile exact CVM unit/count conflicts, propagate corrections through fiscal bridges, restore sparse physical channels, establish per-field scaling, and separate update age from dependency age. Rebuild only affected derived families in a newly bound store. Preserve raw sources and report changed values/support by field, issuer and era. Re-run source mutation, availability and identity checks. Do not introduce blanket stale-data filters.
2. **Diagnose checkpoint selection with archived models on their original bound inputs.** Declare one early-checkpoint rule using selection curves, compare it with tail averaging on matched existing fits, and label all new evaluation readouts post-hoc development evidence. This requires inference, not new pretraining. It can establish whether selection policy explains part of the gap; it cannot establish generalization of a repaired data pipeline.
3. **Run a small matched optimization experiment on the repaired inputs.** Keep A0 as a control; compare A1 (S0 slow), A3 (S0 all inputs) and one rich candidate, initially B4. For the rich model, a small learning-rate/rho grid such as 1e-4 versus 3e-4 and .05 versus .125 is more informative than another architecture sweep. Allow early selection, keep full eligibility/history, and use the same seeds/folds/policy across comparisons. Check learning curves before full confirmation. These values are proposed trial points, not selected optima.
4. **Separate pretraining from architecture.** On a bounded matched subset, compare compatible pretrained and fresh-start rich models under the same corrected fine-tuning policy. If pretraining remains useful, select its duration appropriately and address channels absent in P using only each fold's permitted past. Never extend a historical parent's information into its future evaluation era.
5. **Test incremental learning around a strong baseline.** A small residual/additive branch around a frozen A0, initialized to preserve A0's prediction, asks whether cleaned extra information adds value without forcing the network to relearn the established signal. Any residual-target construction must use appropriately cross-fitted baseline predictions, not optimistic in-sample teacher errors. Compare with a simple regularized current-state model as a representation control.
6. **Only then expand temporal inputs or change the objective.** Add a small number of cleaned event/market histories where information is actually absent, or run a matched target/loss ablation. Keep each change identifiable. Confirm a stable improvement across additional chronological folds before spending another large full-architecture budget.

The first two steps are primarily source processing and inference. A fair performance answer for the corrected pipeline will still require fresh candidate fits; a CPU audit cannot certify new alpha. Reuse A0 scores where its inputs, targets and comparison contract are demonstrated unchanged, rather than refitting an identical control. Existing checkpoints and completed Round-7 results remain useful controls and should be retained.

**Reproduction and limits**

Run from the repository root with the research environment:

```powershell
uv run --project research python research/scripts/audit_round7_inputs.py
uv run --project research python research/scripts/audit_round7_cvm_outliers.py
uv run --project research python research/scripts/audit_round7_gradients.py
```

These bounded diagnostic scripts resolve the canonical store/CVM pointers and explicitly name the sealed run roots being audited. Their outputs go to `D:/quant-data/b3/interim/round7_postmortem_20260913`. Full source account traces and extracted filing text stay outside Git; the committed evidence contains compact numeric findings and source hashes. The source audit records unsupported original XML layouts rather than pretending those documents were fully reconstructed; existing accepted CSV accounts remain identifiable. It is not an exhaustive verification of every filing or market observation.

The repairs and proposed experiments above have **not** been applied to canonical data or model recipes in this audit. The delivered changes are diagnostic scripts, evidence and this review. We have identified concrete defects and a much better-supported failure mode; we have not yet measured the improvement from fixing them.
