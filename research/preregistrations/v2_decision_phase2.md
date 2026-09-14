# Phase 2 implementation amendment

Scope: controller learnability and a bounded chronological comparison, following
the accepted Phase 1 report. Engineering acceptance is required before financial
fits. This document fixes the financial recipe before those fits; synthetic
attempts and numerical amendments remain disclosed rather than erased.

## Models and information

Use the coherent three-regressor benchmark-residual calibration as the fixed
starting model. Also report the stronger equal-rank deterministic comparison.
No Phase 1 evaluation-selected substitution of the registered reference.

The conditional calibration model produces three linear outputs from common
context c. Its stock preference is

mu_i = (1 + tanh(g0(c))) * calibrated_mu_i
       + 0.0003 * g1(c) * average_rank_i + 0.0001 * g2(c).

The stateful alternative contains this same direct conditional path plus a
shared two-layer 32-unit SiLU MLP. The MLP adds an unbounded residual in
0.0001-daily-return units. Its output layer and the conditional linear outputs
initialize at zero, exactly preserving the coherent optimizer at epoch zero.
The one-bp residual unit controls numerical conditioning; it is not an output
cap or a change to available names/history. All parameters train together.

Context preserves every field in the existing common-state partition: 41
shared cross-market fields, three common market diagnostics, explicit validity,
known age, three matured shadow statistics and validity, and cross-seed/
cross-horizon rank disagreement summaries. This is 178 common input channels.
Four additional common features describe the calibrated daily alpha
distribution: mean, standard deviation, maximum and minimum, smoothly asinh
scaled in bps. These are economic model predictions, not raw logit confidence.

The stateful MLP additionally receives the existing 13 per-name static inputs,
ten actual inventory/account-state inputs, per-name seed disagreement,
per-name horizon disagreement and calibrated alpha in smooth bps units.
Common arrays are stored once per date and broadcast only to active/held names.
The forecaster and its original features/history are unchanged.

Continuous static/common fields use fit-only median/IQR and asinh; binary,
bounded age and known-validity fields retain their meanings. Invalid values
remain zero plus explicit masks. No selection/evaluation statistics fit scalers.
The context cache verifies the five consumed source arrays and binds the
accepted manifest, forecast cache and seed artifact hashes. Repeated controller
fits read this small verified cache rather than scanning the original store.

Seed disagreement is the population standard deviation of the three separately
normalized seed ranks, averaged across D3/D5/D10. All 675 source score/identity/
mask/manifest members were verified. It is not cross-sectional score dispersion.

The shadow series uses a fixed demeaned equal-average-rank shadow portfolio and
five-session benchmark-residual shareholder returns. At decision t it may add
origin t-6, whose close(t-1) endpoint is now known. Its EWMA half-life is 60 valid
updates. The features are payoff, payoff volatility and accumulated EWMA mass.
Missing outcomes create no observation; the previously known state persists.
It updates even when the actual policy holds cash. Future-mutation tests enforce
this boundary. No training/calibration score is retrospectively inserted as an
out-of-fit predictor observation.

## Training and synthetic acceptance

Adam, learning rate .003; chronological 32-session truncated backpropagation,
with actual inventory carried across chunks and detached only between chunks.
Thirty-epoch ceiling, five-epoch selection patience after at least twenty
epochs, gradient norm cap one, and .01-bps minimum selection-utility improvement.
Epoch zero remains eligible. Select using the independent exact ledger on the
same prior 55-session selection window/purges as the preceding program.
Checkpoint and optimizer state resume at completed epoch boundaries.

The twenty-epoch minimum is an engineering response to the synthetic failure:
the structured MLP temporarily plateaus near epoch five, but resumes learning
the conditional behavior after epoch ten. This is fixed before real financial
fits. The thirty-epoch ceiling does not expand. CPU QP/account arithmetic remains
FP64 and the small neural component FP32. BF16/compile is reserved for the later
GPU forecaster: Python/sparse-QP state transitions dominate this controller.

Use independent synthetic fit/selection/evaluation dates, persistent noisy
cross-sectional signal, observable zero/useful/continuation states, adverse price
shocks and independently changing known borrow costs. All prices are observed;
corporate actions are identity. Test actual profitable trading, improvement on
the unconditional optimizer, lower zero-state exposure and different actions
for otherwise identical adverse inventory under continuation versus reversal.
Also check the trained model's sequential account against the exact ledger,
known transaction-cost increases against identical prior states, and the
materiality of terminal residuals. A gradient alone is not learning acceptance.

The initial unrestricted additive MLP failed this test; its evidence is retained.
The direct conditional-rank path and a smaller residual unit improve its
optimization. The original five-epoch patience also stopped the corrected model
before it learned, motivating the fixed minimum. Do not hide these engineering
attempts or count their evaluation paths as untouched financial data.

The synthetic stress exposed numerical forward/adjoint disagreement in a nearly
flat QP direction. Clarabel's objective gap tolerance is now 1e-12 rather than
1e-10; feasibility remains 1e-10. OSQP retains its 1e-8 tolerances and the same
strict forward-weight agreement bound, with a 100,000-iteration ceiling for rare
difficult cases instead of 20,000. Ordinary solves stop at convergence. No
financial constraint, cost or utility parameter changed. Recompute Phase 2
deterministic references using this same accuracy and compare them to Phase 1.

## Financial roster and decisions

Fit TE_all .2 and C6 on F2/F6/F10/F14. S0 remains a deterministic reference from
Phase 1; this is not a third large controller campaign. Keep seeds 11/29/47 for
the stateful model and the same three-seed frozen forecast input for every cell.

The linear conditional model has zero initialization, deterministic chronological
updates and no stochastic layers/sampler. Its controller seeds are exact aliases;
run one unique fit per arm/fold, record that fact, and do not claim three
independent replications. The stateful MLP has three distinct initializations.
This gives 32 unique financial fits: eight conditional and 24 stateful, rather
than repeatedly solving an identical deterministic optimization.

Cash/benchmark/learned fallback is chosen exclusively from prior selection
utility. Ties prefer cash, then benchmark, then learned. Evaluate the candidate
without fallback separately. Continuous fallback switches request actual
liquidation through the shared ledger, including costs and unavailable fills;
they never splice separate return series.

Use the predeclared screen gate: positive mean paired net and utility versus
benchmark, positive utility in at least three of four folds, and no material
seed reversal. Report the equal-rank comparison alongside it. Confirm qualifying
survivors on the remaining ten matching folds; otherwise stop this financial
branch. Report nominal paired 20/40/60-session circular intervals, seed results,
fit/selection trajectories, actual exposure/turnover and fallback choices. No
seed expansion or further real-result-driven controller tuning is authorized by
this screen. Phase 3 remains separately authorized even if this phase fails.
