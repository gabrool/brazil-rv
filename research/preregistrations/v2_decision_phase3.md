# Phase 3: controlled economic auxiliary objective

This amendment fixes the objective and matched comparison before new financial
fits/evaluation. Phase 2 completion does not require a winning controller before
this separately authorized phase can proceed.

## One change within each architecture

TE_all retains full 60-session temporal attention and early peer mixing, all
accepted families, FiLM and ASAM .2. C6 retains its slow GRU, pooling and additive
fundamentals/magnitudes projections, with SAM .125. Use the selected neutral P
parents for seeds 11/29/47. TE weights/config/preprocessing load exactly. Adapt
the S0 P parent to C6 explicitly: copy all shared tensors exactly, initialize
only the new family projections at zero, retain the same store/slow coordinates,
and fit the previously unseen family scalers only on F fitting dates. This is an
explicit cold-family extension of S0, not a claim of all-family pretraining.

For each architecture, compare fresh neutral-only and neutral-plus-economic
fits using the same training implementation, batch order, schedule, regularizer,
parents and seeds. The common trainer uses learning rate 1e-4, transferred
multiplier .3, a 60-epoch fixed schedule and selection patience five. New family
projections and the auxiliary head receive full learning rate. This changes C6's
historical training recipe, so its old fits are descriptive references only;
the fresh matched neutral control is mandatory. No lookback/name pruning.

## Target and loss

Add one linear economic head to the final shared trunk, evaluated in the same
encoder pass. Its target is the mean daily five-session shareholder residual:

`y = (shareholder_return_H5 - CDI_H5 - beta_at_decision * BOVA_excess_H5) / 5`.

Use the already sealed shareholder/action economics and causal resolved beta.
No outcome enters predictive inputs. An auxiliary label is valid only where
its original shareholder mask, active name and benchmark endpoints are valid
and all six dates belong to the permitted F target window. Missing auxiliary
labels never remove a name or its ranking loss. Divide by the fit-only,
equal-date root mean square target; keep zero as zero and preserve magnitudes.
Do not rank, clip or cross-sectionally demean this target.

Loss is the existing equal-date neutral ranking loss plus **0.25 times**
equal-date/member Huber loss, delta one, in those scaled units. Initialize the
economic head to zero without changing the matched RNG sequence. The encoder
therefore receives its auxiliary gradient after the head begins learning.
Record the scale and export the head in actual daily-return units. Test fit
isolation, endpoint masks, ISIN alignment, unchanged neutral initialization and
nonzero auxiliary gradients reaching both encoders.

Use BF16 autocast, FP32 loss/optimizer, compiled model/loss and complete unique
date caches. The head is a tiny additional matrix, not another history pass.
Keep selection on the same neutral IC for both variants to isolate the objective
change; record economic calibration and utility separately. A null result does
not rule out changing checkpoint selection or end-to-end decision training.

## Fixed policy and chronological comparison

Use the coherent **equal-rank** calibrated QP from Phase 1 for both variants.
It improved TE materially without establishing harm to C6 and avoids the
unstable opposing three-horizon coefficients. This is fixed before Phase 3
results; no Phase 2 evaluation-selected policy is substituted.

Four-fold screening does not provide a full history of new out-of-fit forecasts.
Therefore estimate each architecture's shared rank-to-return mapping using its
existing sealed chronological forecast cache on each fold's permitted past.
Apply that same mapping to fresh neutral and auxiliary ranks. Never calibrate
with in-sample predictions from either new F fit or future evaluation outcomes.
Report this shared historical mapping assumption; evaluate cardinal head errors,
out-of-fit calibration slopes and tails separately. This experiment tests whether
the auxiliary improves the learned representation under a fixed decision rule,
not whether a newly optimized policy can exploit every auxiliary output.

Also compare a simple C6/TE convex rank blend. Choose its weight from
{0,.25,.5,.75,1} using the prior selection window and exact costs, with mapping
fit on the preceding fit window using the two sealed out-of-fit caches. Freeze
that weight for new evaluation scores and use identical weights for neutral
and auxiliary variants. Report whether the architectures have complementary
errors. This small blend is a diagnostic, not an additional architecture search.

Screen F2/F6/F10/F14 x seeds11/29/47: 48 new F fits. No automatic seed expansion.
Report each seed and the equally averaged rank ensemble, paired 20/40/60-session
circular intervals, neutral IC, calibration/tail errors, net above CDI, utility,
costs, exposures, fit/selection curves and settlement flags. Advance an economic
variant only with positive ensemble paired net and utility versus its matched
neutral control, positive utility in at least three folds and no seed mean net
or utility below -0.25 bps/day. Confirm survivors and their matched neutral
controls on the remaining ten folds. No real-result-driven retuning. New
continuous books require the full corresponding forecast history; do not splice
isolated fold P&Ls or fill missing variant forecasts with other models.

No 2025/2026 consumer reads, forward capture or Phase 4 work. After Phase 3,
return to the postmortem and complete the combined review and recovery record.

## CPU preparation and evaluation boundary clarification

Lambda authentication is unavailable and the user deferred GPU work. Complete
CPU engineering and preserve the remaining campaign for later. GPU acceptance
uses two epochs for each architecture/objective at F2/seed11, so compile startup
and the following epoch are reported separately. These disposable smoke fits
do not enter the financial comparison. Require finite updates, actual CUDA
tensor caches, compilation and identical matched populations before dispatch.

New score panels contain the registered evaluation dates only. Start each
matched fold book from cash on its first evaluation date, with the same final
liquidation rule for both objectives. Do not manufacture a burn-in using old
forecasts. Thus Phase 1/2 fold books are descriptive, not matched Phase 3 controls.
Use only labels whose full horizon remains inside that evaluation block for
rank/cardinal readouts. Continuous confirmation starts from cash at the first
development evaluation date and carries inventory across subsequent fold changes.

The blend coefficient is the TE weight. Select the highest prior-window utility;
exact ties prefer the smaller weight, an explicit deterministic convention.
Keep the original normalized rank coordinates when blending, without re-ranking
the convex combination. The rank ensemble averages the three seed midranks;
the economic head ensemble averages predictions in daily-return units. Freeze
the four screen weights/mappings on sealed old OOS forecasts before new GPU fits.

## Authorized local RTX 2060 execution amendment (2026-09-16)

The user authorized optimizing and running the remaining experiment on their
6 GiB RTX 2060. This supersedes the operational GPU deferral above. The complete
48-fit roster, original parents, repaired store, economic targets, 60-session
history, all eligible identities, unique-date effective batches, loss, SAM/ASAM
radii, schedule, stopping, selector and financial gates remain unchanged.

Use FP16 autocast with FP32 master weights, optimizer moments and losses on
Turing. Scale and unscale both SAM passes before computing the perturbation or
clipping. A scale overflow retries the same batch and dropout realization with
a smaller scale; it never skips an update or advances the learning schedule.
Persist scaling state for exact interruption/resumption. Both objectives use
the same runtime and precision. Original P parameters load unchanged in FP32.

Windows uses the supported PyTorch 2.6.0/cu126 and Triton-Windows 3.2.0.post21
pair; the GH200 environment retains PyTorch 2.13.0. Native Turing compilation
uses the default Inductor mode with fused forward/backward/loss. Exhaustive
reduction autotuning is excluded after a discarded fit-only probe spent several
minutes tuning its first backward. Use a short compiler-cache path to avoid
Windows path-length failures. No compiler failure silently falls back to eager.

Factor the canonical materialized histories by session and permanent security
identity. Store each history observation once, without quantization, and gather
the exact original full window for every sample. Decision snapshots and labels
retain their own date axis. The cache receives only the existing authorized
collator's tensors; it grants no broader data access. This replaces the expanded
overlapping-window cache while preserving actual CUDA caching. One persistent
GPU worker retains import/verified-file caches between independently seeded fits;
models, optimizers, RNG and compiler graph references reset per fit.

Before financial dispatch require: all input/parent/mapping hashes verified;
exact canonical cache equality; representative F2/F14 full-population, full-history
precision and actual two-pass throughput/memory checks; and the original four
two-epoch matched smoke fits with finite losses, unchanged update/population
counts and source-bound artifacts. Record first-epoch startup separately.
Retain full histories, module diagnostics, selected weights and epoch checkpoints.
Engineering probes consume fitting data only and do not select financial recipes.
