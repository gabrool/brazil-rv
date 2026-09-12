# Multi-day model design diagnosis after Round 6

Prepared 2026-09-12 for independent LLM review. Code audited at
`a43d4907c73f9ed838056a2104d609696744367a`; Round-6 neural implementation was
frozen at `0bf8130b73fe6b207c6b6f62603bf383b8f82e1f`. This is an investigation
and a proposed research order, not a new experiment registration or a change
to the accepted comparator. S0 remains the working comparator.

**My assessment is that the model has several consequential, insufficiently
tested design choices, but Round 6 does not establish that its architecture is
fundamentally broken.** The strongest next investigation is whether the training
objective and inherited optimizer prevent useful adaptation, followed by whether
the representation supplied to the network preserves the information we want it
to learn. A larger attention model is a later, conditional experiment.

This assessment uses the implementation, the complete Round-6 readout, all 714
completed fine-tuning histories, and a new inspection of 126 selected checkpoints:
all 14 folds and three seeds for fundamentals, magnitudes and C6. The
[checkpoint evidence](v2_model_design_checkpoint_audit.json) records source hashes,
methods and individual measurements. This audit performed no training, model
inference, store-payload reads or held-out-data reads. No GPU was launched.

**What the results actually establish.**

| Model | Primary IC | Net excess, bps/day | Paired IC gain over S0, nominal 95% interval |
| --- | ---: | ---: | --- |
| S0 | 0.025616 | 4.508 | Reference |
| Fundamentals | 0.028187 | 5.830 | +0.002571 [-0.002937, +0.008394] |
| C6: fundamentals + magnitudes | 0.027807 | 6.071 | +0.002191 [-0.003369, +0.008098] |
| Latest-row MLP | 0.025267 | 3.462 | -0.000348 [-0.003534, +0.001377] |

The fundamentals point estimate is approximately a 10% relative IC improvement
and a 29% relative net improvement. Those are potentially material changes, but
the uncertainty includes no improvement. The full panel prefers fundamentals,
whereas one seed-omission panel prefers C6. This makes the replacement decision
inconclusive; it does not establish that all additional information was useless.
The net estimates also retain the economic-data qualifications described below.
[Source: complete Round-6 results](v2_ROUND6.md).

There are three reasons the earlier data screens do not imply a large neural gain.
First, the Round-5 positive tree screens were against an `a_slow` GBDT with IC
0.017440, whereas Round-6 S0 has IC 0.025616. Additional value relative to the
weaker model need not remain incremental relative to the stronger model. Second,
low raw feature correlation does not establish independent information about
future returns conditional on the existing model. Several features can measure
the same latent driver through different nonlinear transformations. Third,
source quality, sensible economic definitions and plausible annual signs do not
constitute independent predictive confirmation. The screens involved multiple
candidates on development history. Only magnitudes, oddlot and options had
positive nominal paired intervals in those earlier tree screens; it would be
incorrect to describe every added family as having established predictive value.
[Source: Round-5 information screens](v2_ROUND5_DATA.md).

We also have evidence that the network uses the added inputs. Invalidating both
C6 families after training reduces IC by 0.005976 [0.000744, 0.012227]. Invalidating
magnitudes in the magnitudes-only model reduces IC by 0.002628
[0.000983, 0.004855]. These are reliance measurements, not the improvement obtained
by retraining with the family. The backbone can adapt around the new inputs;
removing them afterward can damage that adapted model more than their original
incremental benefit. Such ablations also remove validity and age information,
so they do not isolate numeric content from coverage effects.
[Source: Round-6 inference attribution](v2_ROUND6.md).

**What is actually implemented.**

| Component | Audited implementation | Consequence |
| --- | --- | --- |
| Slow input | 60 sessions of 32 fields, with masks and ages; 128 encoded inputs projected to width 64 | Daily rows contain derived lookbacks, not just raw daily OHLCV |
| Temporal encoder | One-layer GRU, width 64 | Compact sequence model; no internal GRU dropout in this configuration |
| Cross-sectional context | Mean and dispersion of active names' slow states, with a learned gate | Coarse learned peer information from the slow branch |
| Added families | Current decision-row values, validity and age; zero-initialized linear maps added after pooling | No learned sidecar history or direct peer-sidecar aggregation |
| Shared trunk | Width 128, two residual SwiGLU blocks with inner width 48, dropout 0.1 | Nonlinear own-name interactions are possible |
| Heads and loss | D1/D2/D3/D5/D10, equal loss weights of 0.2 | 40% of the explicit objective concerns D1/D2 |
| Selection and trading | D3/D5/D10 | Short heads are auxiliary tasks, although not weighted as such |
| Optimization | SAM radius 0.125; AdamW; base LR 0.0003; inherited parameters use LR multiplier 0.3 | New inputs adapt against a more slowly changing pretrained backbone |
| Checkpoint | Raw Patience; evaluation every two epochs; patience three checks | EMA is maintained but is not the selected scoring checkpoint |

Implementation references, pinned to the audited revision:
[model and forward path](https://github.com/gabrool/brazil-rv/blob/a43d4907c73f9ed838056a2104d609696744367a/research/src/brazil_rv/v2/model.py#L250),
[decision-row sidecar loading](https://github.com/gabrool/brazil-rv/blob/a43d4907c73f9ed838056a2104d609696744367a/research/src/brazil_rv/v2/data.py#L1074),
[loss](https://github.com/gabrool/brazil-rv/blob/a43d4907c73f9ed838056a2104d609696744367a/research/src/brazil_rv/v2/losses.py#L126),
[optimizer](https://github.com/gabrool/brazil-rv/blob/a43d4907c73f9ed838056a2104d609696744367a/research/src/brazil_rv/v2/train.py#L248).

The 933 historical security identities are no longer a reason to suspect wasted
model capacity: the amended run used compact axes containing every PIT-active
name. BF16, FP32 loss arithmetic and unique-date batches were tested before the
matched refit. These repairs changed computational efficiency without dropping
eligible observations. They should be retained.

**Finding 1: objective alignment deserves the first controlled test.**

The current objective gives D1 and D2 a combined 40% weight even though checkpoint
selection and the portfolio use D3/D5/D10. This is a verified mismatch in emphasis,
not proof of a defective loss. Auxiliary short horizons might improve the shared
representation; they might instead pull it toward short-lived effects that do
not help the traded horizons. Equal nominal weights also do not imply equal
gradient influence.

The direct test is current weights versus `(0, 0, 1/3, 1/3, 1/3)`, holding labels,
architecture, selection metric and portfolio fixed. Measure per-head gradient
norms and cosine similarities on shared parameters using fixed fit-window
batches. Repeated opposition between short- and long-horizon gradients would
support interference as a mechanism. We do not currently log those measurements.
Gradient interference is a documented multi-task optimization problem, but its
presence here must be measured. I would test removing the unnecessary tasks
before introducing gradient-surgery machinery.
[Primary reference: Yu et al., Gradient Surgery](https://arxiv.org/abs/2001.06782).

There is a second alignment issue: training averages each head's loss on its own
valid population, while selection evaluates the declared heads on common support.
This is not inherently wrong; using additional legitimate labels can help.
Measure the size and characteristics of the population difference before adding
another restriction. Do not discard valid observations merely to make masks look
identical.

The loss itself deserves a subsequent, separate comparison. It standardizes
predicted scores within a date, forms soft ranks through all pairwise sigmoids
at temperature 0.5, and correlates those ranks with the target ranks. This is a
smooth approximation to the evaluation statistic, rather than a forecast of
return magnitude. It may be suitable, but its inherited temperature and gradients
have not been justified by the multiday experiment. A cheap control is Pearson
correlation of predictions with the same target ranks, without soft-ranking the
predictions. This retains the existing labels and evaluation while removing the
quadratic pairwise operation. It is a different surrogate and must earn adoption;
it is not mathematically identical to Spearman. Inspect rank saturation and
gradient scale before sweeping temperatures or adding a more complicated loss.
[Source: soft-rank implementation](https://github.com/gabrool/brazil-rv/blob/a43d4907c73f9ed838056a2104d609696744367a/research/src/brazil_rv/modeling/engine.py#L69).

**Finding 2: SAM is a high-value suspect, but 0.125 is not demonstrably excessive.**

SAM perturbs weights in the first gradient direction and takes the optimizer
gradient at those perturbed weights. The current implementation restores weights
and reuses the dropout RNG correctly between passes. I found no basis for blaming
an obvious broken SAM update. The concern is whether this regularization is
helpful for the present task and parameterization. A radius selected for the
intraday network is not automatically appropriate for a GRU with newly added
zero-initialized projections and a different cross-sectional loss distribution.

Fixed-radius sharpness is sensitive to weight parameterization. This is a reason
to test SAM here, not evidence that a particular radius must fail. I would compare
true AdamW without SAM against the current radius first, and test an intermediate
radius such as 0.025 only if the initial result warrants it. I would not introduce
ASAM as the first remedy.
[Primary reference: Kwon et al., ASAM](https://arxiv.org/abs/2102.11600).

This test has unusually good research value because it can improve both accuracy
and speed: removing SAM removes one forward/backward pass per optimizer update.
The update computation can approach a twofold reduction, but compilation,
validation, loading and scoring prevent assuming a twofold end-to-end speedup.
Use matched date exposure, optimizer-update opportunities and checkpoint rules;
report runtime separately.

An implementation detail matters: the present SAM function rejects `rho <= 0`.
A real no-SAM control needs a straightforward single-pass AdamW update path.
An arbitrarily tiny radius would still pay for two passes and would not be a
clean implementation of the proposed speed experiment.

Other regularizers are weaker first suspects. Dropout is 0.1 in the residual
blocks, not recurrent dropout throughout the GRU. Persistence and to-close loss
weights are both zero. Weight decay is 0.01, including LayerNorm weights under
the present parameter routing; that routing can be cleaned up in a separately
recorded optimizer change, but I would not infer severe underfitting from its
presence. Increasing the inherited-parameter LR multiplier to 1.0 was already
tested and did not show a clear IC improvement. Neither repeating that experiment
nor turning off every regularizer at once would be my next step.

**Finding 3: preprocessing removes some of the information under discussion.**

Most numeric fundamentals, including valuation ratios, profitability, leverage,
revenue growth and SUE, are transformed to contemporaneous cross-sectional
Gaussian ranks. This preserves order and limits outlier influence, but removes
cardinal distances and absolute thresholds. A median-ranked earnings yield does
not tell the model whether earnings are positive or negative. A large accounting
surprise and a small one can have the same rank on different dates. Ranking SUE
also removes some of the scale meaning created by its own historical
standardization.

These are actual information losses before learning. More attention or more
epochs cannot reliably recover information that the supplied representation
does not identify. The highest-value representation test is therefore to retain
the current ranks and add a small, specified set of robust native values or
economically meaningful sign/threshold channels. Use fit-only clipping and
scaling, with independently preserved masks and receipt-time ages. Avoid raw
unscaled accounting quantities and broad indicator expansion.

There is also a support rule: a rank-transformed field with fewer than 20 valid
active names is marked invalid for that whole cross-section. That is sensible
for a stable rank estimate but can suppress legitimate sparse observations.
Quantify the lost field-name-date coverage first. A separately scaled native
channel could retain usable sparse values without pretending that a rank on five
names is comparable to a rank on 200. This is preferable to blindly removing
the rank-support condition.

The same criticism does not apply to every feature. Common macro states and
the four magnitude fields explicitly retain native representations. Age, binary
and bounded-fraction fields have distinct transforms. Current data is not being
universally rank-normalized into oblivion. Financial/nonfinancial accounting
semantics also remain relevant: an explicit financial-sector flag helps, but
pooled ranks alone do not make unlike accounting ratios economically equivalent.
[Sources: fundamental definitions](https://github.com/gabrool/brazil-rv/blob/a43d4907c73f9ed838056a2104d609696744367a/research/src/brazil_rv/v2/feature_spec.py#L416),
[typed transforms and support rule](https://github.com/gabrool/brazil-rv/blob/a43d4907c73f9ed838056a2104d609696744367a/research/src/brazil_rv/v2/feature_spec.py#L861).

**Finding 4: the linear sidecar is not inherently a narrow projection, but the
learned maps are strongly concentrated.**

Fundamentals supplies 12 fields, expanded to 36 columns including validity and
age, then projected to 128 dimensions. Magnitudes supplies 12 encoded columns
to 128. C6 combines 48 encoded columns. A linear map that expands these inputs
does not necessarily discard nonlinear information: a later nonlinear network
can learn interactions from an injective linear encoding. The shared residual
trunk is nonlinear. The inner width of 48 is the residual transformation width,
not a hard 48-dimensional limit on the entire representation.

Nevertheless, additive mixing into an already trained latent state gives new
inputs a particular and possibly unhelpful learning route. The checkpoint audit
found a concrete signature worth investigating:

| Selected checkpoints, 42 per model | Median stable rank of the full family map | Median share of squared value-weight norm in its largest singular direction |
| --- | ---: | ---: |
| Fundamentals: fundamentals map | 1.093 | 95.5% |
| Magnitudes: magnitudes map | 1.160 | 93.5% |
| C6: fundamentals map | 1.072 | 96.4% |
| C6: magnitudes map | 1.159 | 94.1% |

Stable rank is the squared Frobenius norm divided by the squared largest singular
value. All three models have positive aggregate sidecar gradient norms in every
logged epoch of every fit. Numeric-value weights are nonzero. These branches
learned; they are not disconnected. Their weight matrices nevertheless emphasize
one direction strongly.

This is **not** evidence that 95% of predictive information was lost. Weight SVD
depends on input coordinates and scaling; masks and ages contain redundancies;
a scalar ranking task may appropriately favor one linear combination; and small
weight directions may still affect predictions through the trunk. We have not
measured covariance-weighted activations, prediction Jacobians or nonlinear
interaction strength. These qualifications are essential when interpreting the
[checkpoint audit](v2_model_design_checkpoint_audit.json).

The discriminating architecture test is a small nonlinear family encoder followed
by late concatenation with the slow representation, or a small residual prediction
head with access to both representations. Use one specified version, not a large
architecture search. Keep the same input information first so the comparison
isolates the learning route. If preserving the initial parent function, initialize
only the new branch's final output projection to zero; do not zero every layer
of a multilayer branch and then wonder why it cannot learn.

**Finding 5: the network lacks some temporal and relational inputs entirely.**

The added families arrive only as current decision-row scalars. Some already
contain historical summaries, changes or surprise measures, so it would be wrong
to say they contain no historical information. But the model does not receive
their sequence in the way it receives the 60-session slow branch. It cannot
generally reconstruct the previous accounting state, the sequence of revisions,
or changing event intensity from the latest value and its age alone.

For slowly changing fundamentals, a concise event representation is likely a
better first test than repeating the same filing over 60 daily rows: current
value, previous publicly known comparable value, causal change and time since
receipt. For rapidly changing market-side families, a short masked history is
more plausible. These are separate experiments; constructing a temporal encoder
over missing information would not fix the omission.

The peer limitation is also explicit. Cross-sectional pooling happens before
sidecars enter. No learned route carries another security's contemporaneous
fundamental vector into a stock's prediction. Precomputed ranks and sector
features can already contain limited peer information, so this is not complete
cross-sectional isolation. However, learning which peers' valuation, earnings
or events matter requires more than the mean and dispersion of slow states.

Start with PIT sector-relative representations or simple pooled sidecar summaries
and matched same-information controls. If these help, a small attention block
across eligible securities becomes a justified candidate. Distinguish that from
attention across feature tokens or across dates: these solve different problems.
All peer construction must use historical membership and contemporaneously known
sector assignments.

Common macro information has another constraint: adding the same number to every
stock's score cannot change a cross-sectional ranking. Such information helps
stock selection through differing exposures or interactions with company states.
The current nonlinear trunk can represent those interactions, but its doing so
is an empirical question. For that specific mechanism, a small conditioning
branch that lets macro states modulate stock features is a clearer test than
simply adding more common scalars. Shared predictors can also be useful for risk
management without improving a cross-sectional stock-selection target.

A strong small MLP and a GBDT remain useful controls. The Round-6 MLP's latest
row already contains return, momentum and volatility lookbacks; its similar IC
does not prove history is irrelevant. It does suggest that a large temporal
architecture upgrade is not yet supported by a demonstrated temporal premium.
Broader tabular benchmarks motivate serious simple controls and show no
universally superior neural-versus-tree choice; they do not tell us which model
will win on Brazilian equities.
[Primary references: Gorishniy et al.](https://arxiv.org/abs/2106.11959),
[McElfresh et al.](https://arxiv.org/abs/2305.02997).

**Finding 6: the target intentionally removes substantial economic structure.**

The target is not simply a future excess return. For each date and horizon, it
subtracts the cross-sectional median realized return, divides by causal stock
volatility times the square root of the horizon, clips at plus/minus five, then
residualizes against ten volatility groups, five beta groups and linear log ADV
when support permits. Smaller supported populations use a linear fallback.
The final labels are ranks of those residuals. This is a deliberate definition
of stock-specific performance after removing selected risk characteristics.
[Source: characteristic-neutral targets](https://github.com/gabrool/brazil-rv/blob/a43d4907c73f9ed838056a2104d609696744367a/research/src/brazil_rv/v2/store.py#L85).

The portfolio uses equal sizing, volatility-quintile construction and a BOVA11
hedge; it does not implement exactly the same ten-volatility-group,
five-beta-group and liquidity projection. Thus the prediction objective and the
realized economic objective are related but not identical. Useful raw-return
information associated with risk or liquidity may be intentionally excluded
from the headline target. A family can be informative about market returns and
still contribute little to this more restrictive target.

I would examine this after the simpler objective and optimizer tests, with a
separate target registration: compare the current target with a less aggressively
neutralized, robust return target while reporting BOTH the original neutral IC
and risk-matched net economics. Preserve portfolio constraints, explicitly report
factor exposures, and do not call increased systematic risk improved stock
selection. Merely changing the target to obtain a larger IC is not progress.
Directly optimizing a backtest Sharpe or adding a complicated portfolio loss is
premature while the economic-data defects remain.

**Finding 7: checkpoint selection is a more credible remaining convergence issue
than the blanket epoch cap.**

Fundamentals and C6 both have median selected epoch eight, maximum selected epoch
36 and no fit reaching the 60-epoch cap. Across all 714 fits, 15 reached 60,
including seven MLP fits. This argues against raising the ceiling everywhere as
the main remedy. It does not prove the models have fitted enough.

The selection window is only 55 sessions, with overlapping D3/D5/D10 outcomes.
Those are not 55 independent observations at every horizon. Checkpoint evaluation
every two epochs and stopping after three non-improving checks can respond to a
noisy local curve. Moreover, warmup and cosine decay are parameterized by the
60-epoch maximum: warmup lasts approximately three epochs, so an epoch-eight
checkpoint is selected early in the decay schedule. Changing the cap alone also
changes the learning-rate trajectory and would confound a convergence test.
[Sources: split contract](https://github.com/gabrool/brazil-rv/blob/a43d4907c73f9ed838056a2104d609696744367a/research/src/brazil_rv/v2/splits.py#L35),
[schedule and checkpoint setup](https://github.com/gabrool/brazil-rv/blob/a43d4907c73f9ed838056a2104d609696744367a/research/src/brazil_rv/v2/train.py#L2096).

Before changing checkpoint rules, measure clean fit-window IC and selection IC
using the same evaluation mode and metric, plus dropout-on/off loss and the SAM
perturbation gap on fixed fit batches. The stored training loss is a dropout-on
first-pass soft-rank loss across five heads; selection is a hard-rank statistic
on three heads and another population. Comparing `1 - training_loss` directly
with selection IC cannot diagnose underfitting.

If clean fit performance is weak and optimization continues improving, test a
fixed update budget and independently specified LR decay on development folds.
If clean fit performance is strong but held-out development performance is weak,
more capacity or less regularization is a weaker recommendation. A separately
registered checkpoint-averaging test could address selection variance, but final
EMA is not an apples-to-apples substitute for the selected raw checkpoint.
The historical intraday averaging rejection is evidence about that earlier
contract, not proof about every multiday averaging rule.

**Known data and evaluation defects must remain separate from architecture claims.**

False inferred corporate actions include PETR3 on 2020-03-09 and EMBR3 on
2020-03-12. Three apparent security endings arise from excluded B3 cash categories
for AMER3, OIBR4 and SEQL3. These can corrupt labels, wealth continuity or
settlement economics. Round 6 retained its frozen parent targets and economic
contract, so these defects cannot be repaired by silently rewriting its artifacts.
Before making a new economic promotion claim, repair and version the relevant
derived contracts and re-establish matched controls. The present audit does not
quantify how much of any model difference those defects explain.

The registered non-circular block bootstrap also underweights fold endpoints.
Its turnover intervals demonstrably do not center on the displayed sample means.
Audit boundary weighting before relying on small paired differences in a new
decision. Preserve the old report and show any sensitivity as a new analysis.
Neither a noisy interval nor a conservative promotion rule is evidence that the
physical network cannot learn. Likewise, sparse early CVM identity and historical
borrow coverage reduce actual information; a feature name in a schema does not
mean it supplies a long, dense learning history.
[Source: Round-6 qualifications](v2_ROUND6.md).

**The research order I recommend.**

The immediate goal should be to distinguish optimization limits, information
loss, weak conditional signal and evaluation noise. Do not change SAM, targets,
preprocessing and architecture simultaneously and then attribute a gain to the
network.

| Order | Experiment or diagnostic | Evidence it would provide |
| --- | --- | --- |
| 0 | Resolve known label/economic defects for the next registered contract; measure clean fit/selection behavior, head gradients and sidecar activations | A trustworthy comparison and a distinction between underfitting and weak generalization |
| 1 | Factorial: current SAM versus AdamW; five-head versus D3/D5/D10-only loss; both S0 and fixed C6 | Whether inherited training suppresses performance, and specifically suppresses the incremental value of new data |
| 2 | Ranked fundamentals plus a small native/sign representation, with original representation as control | Whether useful information is lost before the model sees it |
| 3 | Small nonlinear late-fusion branch with identical inputs | Whether the present additive route limits learnable interactions |
| 4 | Current loss versus a simple non-pairwise rank-target surrogate | Whether the optimization surrogate costs accuracy or unnecessary compute |
| 5 | Causal event changes/short sidecar histories, then cheap peer context | Whether missing temporal or relational information is the bottleneck |
| 6 | Less aggressive target neutralization with risk-matched economics; attention only against strong same-information controls | Whether the desired economic signal differs from the target, or richer relations justify architectural complexity |

A practical first neural screen is four calendar-spaced development folds, for
example F2/F6/F10/F14, using all three existing seeds. Freeze those folds for
calendar coverage before running new variants; they are not selected because a
candidate previously won there. The eight combinations of two optimizers, two
head-weight recipes and two input sets require 96 F cells. If the input/label
contract remains exactly compatible, 24 current-recipe S0/C6 cells already exist,
leaving 72 new F fits. If repairs change that contract, refit those 24 controls:
the screen then requires all 96. Do not reuse incompatible scores to save time.

Only a promising recipe advances to the remaining ten folds. Confirming that
recipe on both S0 and C6 adds 60 F fits, while the four screen folds are reused.
That is 132 new F fits with compatible controls, or 156 with repaired controls,
rather than another 714-fit family campaign. A null screen can end the optimizer
investigation early. These are experiment counts, not runtime promises; saved
data loading, compilation and single-pass updates should reduce cost further.
Representation experiments are a later bounded stage, not hidden additional
arms in this budget.

Compatible existing P checkpoints make this an efficient test of fine-tuning
changes. It would **not** establish the effect of removing SAM or short-head
losses from pretraining: those inherited P weights still encode the old recipe.
If a finalist earns an end-to-end claim, perform matched fresh P/F validation
only for that finalist and its necessary control. Those additional P fits are
outside the F-screen budget above.

Keep selection windows, purges, population rules, execution, seed pairing and
date exposure explicit. Compare candidate-minus-parent gains as well as absolute
performance: if AdamW improves S0 and C6 equally, it improves the model without
establishing that SAM was suppressing the new datasets specifically. Predeclare
advancement and economic criteria before fitting. The repeatedly examined
2018–2024 folds remain development evidence even when rerun; another pass over
them is not independent confirmation. The protected 2025/2026 boundary remains
unchanged.

For a cheap conditional-information control, fit a small tree or MLP using the
same new fields and a genuinely out-of-time parent score. Generate training
parent scores through chronological cross-fitting, with all labels matured and
purged before each evaluation period. Never train a residual corrector on the
evaluation labels or calibrate it using the full stitched evaluation panel. If
such a control extracts incremental value that the neural branch misses, the
case for changing the branch becomes substantially stronger. Without that
evidence, adding complexity could simply fit development noise more effectively.

**Questions for an independent reviewer.**

1. Does the proposed SAM-by-horizon factorial isolate the most plausible inherited
   constraints, including the distinction between F-only and end-to-end changes?
2. Which ranked fundamental fields lose economically essential thresholds, and
   what is the smallest causal representation that restores them?
3. What additional activation or Jacobian evidence would make the concentrated
   sidecar weight maps a meaningful learning diagnosis rather than a harmless
   consequence of a scalar output?
4. Does the target remove risk exposure we genuinely intend to exclude, or does
   it materially diverge from the selected portfolio's economic objective?
5. Are checkpoint noise and data defects better explanations of the apparent
   ceiling than insufficient representational capacity?

My preferred first change to test is the optimizer-and-horizon factorial. My
preferred subsequent representation change is a small nonlinear late-fusion
branch supplied with carefully preserved fundamental magnitudes and event
changes, introduced through separate controls. I would require evidence of a
relational benefit before moving to a larger attention architecture.
