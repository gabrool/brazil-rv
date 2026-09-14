# Cash-aware portfolio experiment and conditional attention follow-up

Completed 2026-09-14. The main experiment, rich-attention screen and remaining-fold
confirmation, artifact recovery and paid-instance shutdown are complete.

## What the main experiment establishes

Changing the portfolio controller did not produce a robust improvement over the
same forecasts traded by the legacy policy. None of the three forecast families'
new controllers passes the registered advancement rule. In particular, the
learned controller underperforms the deterministic optimizer in all three
families over the fourteen development folds. Do not promote it or start joint
encoder/controller training on the premise that this experiment succeeded.

Attention remains worth investigating: with the new optimizer, TE_all earns
2.6104 bps/day above CDI versus S0's -0.7926. However, repaired C6 remains stronger
at 4.6122 under that controller and 5.7564 under the legacy controller. These
results do not establish that richer architectures cannot learn, nor that the
old execution policy is optimal. They reject the claimed improvement for this
particular calibration/controller/training recipe.

The accounting implementation reconciles, but missing-price settlement is an
important economic assumption. Very small terminal residuals are numerical dust;
historical artificial settlements are a separate, economically material issue.
Keep the raw unresolved flags and the haircut sensitivity visible. None of the
figures below should be described as demonstrated executable live returns.

## Research contract and experiment inventory

The controlling documents are `research/preregistrations/v2_portfolio_policy.md`
and its implementation record, with the conditional extension in
`research/preregistrations/v2_portfolio_attention_followup.md`. The earlier
execution assessment explains the hypotheses. This report distinguishes those
hypotheses from measured outcomes.

- Accepted repaired daily input store, unchanged full 60-session histories and
  complete eligible populations. No 2025/2026 outcomes or forward capture.
- S0; TE_all with ASAM .2; repaired C6 with its original fundamentals+magnitudes
  roster and inherited S0 parent recipe. All use seeds 11/29/47.
- 102 new fold fits, 24 compatible reused fold fits, three C6 prelude parents,
  and nine prelude score panels. The new C6 parents serve the prelude; its fold
  fits preserve the registered inherited S0 transfer recipe.
- Three causal forecast caches; 126 learned policy fits; 42 family/fold readout
  jobs; three continuous-book readout jobs, each including the registered cost
  and funding scenarios. Main summary completed at 18:42:04 UTC on September 14.
- Four original screen folds F2/F6/F10/F14, ten remaining development folds,
  and all fourteen reported separately. The pooled evaluation has 1,738 sessions.
  The continuous book carries actual state through model switches.

The forecasts are an equal-seed ensemble of within-date/head cross-sectional
midranks. Individual forecasts are retained. Averaging the learned policies'
reported outcomes is descriptive; it is not the return of an implemented
ensemble controller.

Chronological policy fit/selection windows are expanding, with ten-session
purges and a 55-session selection window. Five-session calibration endpoints
must remain inside fit. Selection uses the exact ledger, including a causal
burn-in; no evaluation outcomes select an earlier policy. These are development
results: prior global architecture and regularization choices have already used
development evidence, so the remaining folds are not an untouched final holdout.

## What was implemented

The policy chooses an allocation at the existing 15:45 information cutoff,
using prior observed reference marks. The later close is a fill proxy. Intended
notional or a fraction of existing inventory is frozen before that fill; future
prices cannot choose the original order. Missing prints do not become invented
fills. Identity transitions, contractual actions, claims, lending, pending exits
and cash accounting remain explicit.

Cash can remain undeployed. Historical CDI accrues under the registered cash,
debit and short-proceeds conventions, and every strategy is compared with the
same all-cash CDI benchmark. Short-sale proceeds remain restricted. Under the
base remuneration assumption a neutral equity book can also earn funding
income; CDI is not a special reward assigned only to idle cash.

The new allocator jointly chooses stocks and optional BOVA11. Planned joint
gross is capped at 2.25 NAV, each stock at .05, hedge at .60, absolute signed net
at .05, and estimated absolute beta at .05. There are no minimum stock counts,
minimum gross, or volatility-bucket occupancy quotas. Missing fills and price
gaps can cause subsequent realized exposure drift; the policy requests causal
reductions rather than inventing compliance through fills.

The legacy control intentionally retains its original rank buffer, volatility
quota, wider net limit and post-hoc hedge. This is a comparison of controller
packages, not an isolation of the neural network with identical feasible sets.

The convex objective combines expected excess payoff, a causal diagonal-plus-
market-factor risk estimate, linear trading costs and short-holding costs.
Daily risk aversion is fixed at 5. The optimizer plans five sessions, multiplying
daily payoff/risk/borrow by five but charging immediate trading once, then
replans daily. This is a limited planning approximation, not a complete learned
future forecast path.

Fit-only, equal-date ridge calibration maps the three rank heads to five-session
raw shareholder return above CDI, expressed in daily units. Overlapping heads
are regressors for one payoff, not independent returns added together.

The learned model adds a residual preference to this calibration: two shared
32-wide SiLU layers, 1,857 parameters, zero final-layer initialization. Inputs
include the three ranks, causal score changes/validity, volatility, beta, borrow,
prior CDI, inventory weight/age/marked P&L, pending commitments and portfolio
cash/gross/net/beta. There is no ticker embedding, critic or value function.
The learned preference is a decision parameter, not a claimed calibrated return.

Training uses Adam at .001, maximum 30 epochs, patience five, .01 bp/day minimum
selection improvement and gradient clipping at one. Epoch zero is selectable.
The objective is daily net CDI excess minus 2.5 times causal portfolio variance.
Thirty-two-session truncated backpropagation carries financial state forward;
it detaches gradients rather than liquidating/resetting the portfolio. Exact
ledger selection separates surrogate training from financial evaluation.

## Engineering changes and the evidence they preserve

The first policy batch was rejected. Binary validity flags had been standardized
using almost-zero training variance, allowing an unseen transition to create a
huge numerical input. A percentage return relative to a nearly zero position's
cost basis also created unstable inventory derivatives. Validity flags now keep
their 0/1 units. Inventory movement uses signed marked P&L divided by NAV and
the stock-cap/volatility scale, with asinh. Tiny holdings are retained; no stock
observations or positions were dropped to hide the problem.

The rejected state-feature batch is archived under `rejected_policy_numerics_74c4021`.
It supplies no financial ranking. An unrelated output-schema inference failure
was fixed by inspecting all bounded audit records rather than assuming the first
integer-zero rows determined the entire column type. Low gross is no longer
treated as failure for an explicit cash-capable controller; legacy diagnostics
remain applicable to the legacy control.

Subsequent failures came from numerical QP convergence and adjoint requirements,
not new alpha evidence. Several intermediate solver attempts were superseded.
The final allocator at cf84b62 uses Clarabel at 1e-10 tolerances for the forward
solution. During training only, a fully solved OSQP workspace provides the native
adjoint for the identical QP, with checked primal agreement within 2e-6 NAV.
Inference uses no ADMM solve. Risk limits and economic objective were not relaxed.

Completed valid policy fits were preserved through explicit metadata migrations:
protected training/accounting sources were compared, original manifests and
checkpoint hashes archived, and model/optimizer tensors checked unchanged.
Current metadata identifies the accepted replay implementation; the preserved
origin identifies the implementation that actually trained each checkpoint.
This is numerical continuation, not a claim that every model was originally
trained under cf84b62. No migration imports the rejected state-feature batch.

Validation includes 107 targeted tests and actual-market accounting/gradient
acceptance. On the bounded 128-session engineering path, independent NAV differs
by at most 1.97e-11. The 32-versus-64-session gradient comparison has cosine .9033
and norm ratio .8099. Exact replay took .843 seconds; the 64-session training
path with 32-session chunks took 1.168 seconds. These are engineering timings,
not full program duration estimates.

GPU forecasters retain BF16, compilation, unique-date visits and compact active
axes. Independent jobs share six GPU lanes; CPU policy folds used twelve
single-threaded workers and continuous books three. Financial state stays FP64.
The sparse sequential policy was optimized for its measured CPU workload rather
than forcing the solver into a GPU/compile path that would not help it.

## Main financial results

The empty-start bridge reproduces the prior S0 and TE screen results exactly:
S0 IC .018849 and net -.027859 bps/day; TE IC .027103 and net 1.440401. This is
an important control: the common-burn-in policy tables below are not the same
boundary convention as those historical screen numbers.

Repaired C6 has screen IC .024503 and empty-start net 4.386270 bps/day. Its old
fixed-book accounting bridge was 6.789236; a fresh fit on repaired inputs is a
different experiment. The useful family combination survives, but the old
numerical maximum is not reproduced merely by rerunning its recipe.

Across all fourteen folds, mean neutral IC is S0 .026724, TE .026517, and C6
.029321. Thus the original four-fold attention IC advantage does not persist
as an all-fold average advantage over S0. This is evidence against extrapolating
the original screen, not evidence of an accounting regression or proof that
attention cannot work. Economic ordering also depends on which names/horizons
carry the information, sizing, costs and regime, so equal mean IC need not imply
equal portfolio return.

All numbers in the following tables are mean daily net return **above CDI**, in
basis points, under 4 bps per side and full registered proceeds remuneration.
Cash is zero above CDI. Rounded values are for readability; machine evidence
retains full precision.

| Family / window | Legacy | Optimizer | Learned seed mean |
| --- | ---: | ---: | ---: |
| S0, original four | -0.541 | -1.905 | -0.912 |
| S0, remaining ten | 5.164 | -0.342 | -2.277 |
| S0, all fourteen | 3.519 | -0.793 | -1.883 |
| TE_all, original four | 1.203 | 2.126 | 1.594 |
| TE_all, remaining ten | 4.998 | 2.807 | 2.392 |
| TE_all, all fourteen | 3.904 | 2.610 | 2.162 |
| C6, original four | 4.728 | 1.739 | 1.412 |
| C6, remaining ten | 6.173 | 5.776 | 3.990 |
| C6, all fourteen | 5.756 | 4.612 | 3.247 |

TE's positive screen increment over its legacy control does not persist in the
remaining ten folds. Only two original screen folds have positive paired utility
for TE's new policies; S0 and C6 each have one. The required count is three.
Thus no candidate passes even before resolving the economic caveats.

Paired utility differences over all fourteen folds, with nominal 95% intervals:

| Family | Optimizer minus legacy | Learned mean minus optimizer |
| --- | ---: | ---: |
| S0 | -3.891 [-7.010, 0.717] | -1.320 [-4.883, -0.112] |
| TE_all | -0.846 [-4.079, 3.569] | -0.678 [-3.661, -0.055] |
| C6 | -0.691 [-2.490, 4.085] | -1.542 [-4.066, -0.398] |

Intervals use 10,000 paired, within-fold 20-session block resamples. They are
nominal development intervals, not multiplicity-adjusted proofs. Nonetheless,
the learned policy's failure to improve on the simpler optimizer is consistent
across families. Selected epoch zero accounts for 14/42 S0, 17/42 TE and 17/42
C6 fits. Learning is not reliably improving selection, and the nonzero selected
updates do not translate into aggregate evaluation gains.

The continuous-book results support the same controller conclusion:

| Family | Legacy | Optimizer | Learned 11 / 29 / 47 |
| --- | ---: | ---: | ---: |
| S0 | 4.545 | -0.913 | -2.949 / -1.572 / -1.662 |
| TE_all | 4.386 | 2.724 | 1.676 / 2.336 / 2.117 |
| C6 | 5.635 | 4.693 | 3.112 / 2.126 / 4.050 |

Continuous and restarted-fold books answer different questions. Their difference
includes inherited inventory at boundaries; do not select whichever number is
more attractive or compare either directly with an old result having a different
ensemble, parent, repaired input or boundary convention.

## Costs, cash and funding

Allowing cash did not make the new policies reliably conservative. Their
continuous mean joint gross is roughly 2.06–2.25 NAV. S0 learned policies remain
near the gross ceiling despite negative out-of-sample return. An available cash
action does not guarantee that an estimated-return optimizer or learned policy
will correctly choose it.

C6 continuous daily trading cost rises from .637 bps under legacy to 1.387 under
the optimizer and about 2.02–2.26 under the learned policies. TE's corresponding
cost is .800, .905 and .989–1.087. S0's is .977, 1.997 and 1.93–2.09. Additional
turnover is one concrete mechanism of underperformance, especially for C6/S0;
it is not the sole explanation or a license to retune costs after results.

| Family / policy | 2 bps per side | 4 bps base | 8 bps per side | No proceeds remuneration |
| --- | ---: | ---: | ---: | ---: |
| S0 legacy | 3.720 | 3.519 | 2.676 | 0.241 |
| S0 optimizer | 0.229 | -0.793 | -2.835 | -4.093 |
| S0 learned mean | -0.853 | -1.883 | -3.948 | -5.172 |
| TE legacy | 4.325 | 3.904 | 3.144 | 0.741 |
| TE optimizer | 3.088 | 2.610 | 1.654 | -0.644 |
| TE learned mean | 2.772 | 2.162 | 1.116 | -1.035 |
| C6 legacy | 6.112 | 5.756 | 5.002 | 2.390 |
| C6 optimizer | 5.328 | 4.612 | 3.180 | 1.531 |
| C6 learned mean | 4.346 | 3.247 | 1.035 | 0.066 |

These are frozen-model replays, not refits. Changed realized costs/funding can
change later financial state and hence later decisions. Planned transaction cost
remains the registered 4 bps. Actual funding terms therefore matter considerably.
The base all-cash benchmark compounds to 69.56% over the continuous window; its
mean daily CDI is 3.0387 bps and its excess over itself is zero.

## Accounting closure versus settlement uncertainty

Across 753 non-aliased saved account paths, the maximum terminal stock exposure
is 1.965e-9 NAV. There are no terminal unpaid claims, hedge holdings or unpriced
nonzero holdings. An exact nonzero-share count can therefore flag hundreds of
residual positions while their combined economic exposure is negligible.

Separately, all fifteen continuous noncash base books were replayed with richer
settlement diagnostics. Every NAV path matches the saved result exactly; none
is insolvent and unresolved corporate-action name-days are zero. However,
cumulative settled notional is .383–.694 times contemporaneous NAV summed across
the seven-year window. That is not a simultaneous exposure or a realized loss,
but it exceeds the legacy cumulative .15 diagnostic threshold.

The existing 30% settlement-haircut sensitivity lowers final wealth by about
9.3%–20.8% relative to each base path. For legacy / optimizer, respectively:
S0 12.1% /16.8%; TE 11.0% /15.1%; C6 12.7% /9.4%. These are relative final-wealth
differences, not daily bps or percentage points of annual return. Between one
and four settled names subsequently print in each continuous book, further
showing why settlement at a last mark is an assumption rather than a fill.

Do not clear every unresolved flag merely because terminal positions are dust.
Conversely, do not call a seven-year cumulative settlement threshold an unclosed
cash-accounting discrepancy. Preserve both facts. Before stronger executable-
performance claims, inspect the dominant missing-price exits and whether source
recovery or a better evidenced settlement value can resolve them. Do not replace
missing historical prints with fabricated prices or retrospectively remove the
affected securities.

## Interpretation and next decisions

The decision-trained controller remains a research prototype. It can express
cash, retention, reversals and horizon combination, but this training recipe
does not learn them well enough to beat its initialization. The short selection
window, noisy realized utility, changing calibrations and approximate planning
horizon are plausible contributors, not separately proven causes. Stronger
regularization or longer training is not automatically the answer.

Another modeling assumption deserves an explicit future check: stock calibration
includes a fit-period raw excess-return intercept, whereas BOVA's preference is
fixed at zero excess alpha. This is not a numerical bug or a verified explanation
of the observed losses. It can, however, confound cross-sectional preference with
an estimated common return premium when stocks and the hedge are joint assets.
A coherent shared market-return reference is a more focused diagnostic than
immediately changing the supervised encoder objective. It was not varied in the
registered main comparison and has not been chosen using evaluation outcomes.

The rich early/late attention trigger was met because TE remained competitive
relative to S0 after changing controllers. The follow-up changed only stock-mixing
timing, with compatible new rich parents, matched recipes, three seeds and the
four original folds. It preserved the full history, inputs and capacity settings.
It compared exact legacy books as a common controller and checked all three IC
heads. A positive, sufficiently consistent screen was required before extending
that new arm to the remaining folds. This was an empirical architecture check,
not a claim that late or early attention is universally canonical.

The raw-return auxiliary objective is deferred because this experiment has not
isolated the encoder's objective as the dominant bottleneck. Joint encoder/policy
training is not triggered because the frozen learned policy did not improve
utility. The main results remained sealed throughout the timing experiment.

## Completed rich early/late attention comparison

Three compatible late-attention parents and all 42 late-attention fold fits were
completed, with three causal parent prelude panels. The initial twelve screen
fits were not rerun when the prewritten gate triggered the other thirty fits.
No history, family, eligible name or capacity setting was removed. The existing
temporal/peer architecture was configured for late timing; this was not a new
model implementation requiring a different input contract.

The screen's ensemble net increment was +.253 bps/day and utility increment
+.275, with three of four positive utility folds. Its nominal utility interval
[-2.503,4.184] included zero. This justified running the registered confirmation;
it did not justify declaring late attention superior.

Final paired **TL minus TE** results:

| Controller / window | Net excess difference | Utility difference [nominal 95% CI] |
| --- | ---: | ---: |
| Legacy, original four | +0.253 | +0.275 [-2.503, 4.184] |
| Legacy, remaining ten | +0.170 | +0.169 [-1.754, 2.895] |
| Legacy, all fourteen | +0.194 | +0.200 [-1.244, 2.563] |
| Optimizer, original four | -1.954 | -1.960 [-3.117, -0.392] |
| Optimizer, remaining ten | -0.153 | -0.151 [-1.711, 1.216] |
| Optimizer, all fourteen | -0.672 | -0.673 [-1.793, 0.442] |

The legacy comparison uses the common empty-start convention. The optimizer
comparison uses the registered common burn-in. Within each comparison, dates,
populations and accounting match; the re-created early-attention legacy daily
returns and ICs match the sealed reference. Readout work was bounded to the
evaluated dates without changing those metrics.

The small positive legacy ensemble difference is not stable across individual
seeds: all-fourteen paired net differences are -.597, +1.804 and -.987 bps/day
for seeds 11/29/47. The ensemble is its own nonlinear book, so its outcome need
not equal the mean of individual-seed books. Five of the ten remaining folds
have positive ensemble utility differences. The uncertainty and seed reversals
do not support a robust architecture improvement.

Mean neutral IC is also weaker for late attention outside the screen:

| Window | Early rich attention | Late rich attention |
| --- | ---: | ---: |
| Original four | .027103 | .027388 |
| Remaining ten | .026280 | .023739 |
| All fourteen | .026517 | .024791 |

With the optimizer, late attention earns 1.938 bps/day above CDI over all folds,
versus early's 2.610. Its continuous optimizer earns 2.029 versus 2.724, a -.695
paired difference. Late's all-fold net disadvantage persists under 2 bps costs
(-.525), 8 bps (-.967) and no proceeds remuneration (-.677). These stress results
were not used to choose a new model or tune its parameters.

The late continuous optimizer reconciles to 8.88e-16, with zero unresolved action
days and terminal residual notional 5.38e-10. It still inherits material historical
settlement uncertainty: cumulative settled notional .661 NAV over the window.
Its maximum realized joint gross 2.376, absolute net .113 and absolute beta .137
also illustrate that planned risk caps are not guarantees on exposures after
later proxy fills/price moves. Mean net is .0518 and mean beta .0312. The comparison
must retain that distinction rather than describe the book as perfectly neutral.
Among the empty-start legacy ensemble books, F12 is flagged for both timings,
and late F8 crosses the cumulative settlement threshold (.1593 NAV). These flags
remain visible; there is no claim that every economic caveat was resolved.

Decision: the positive legacy point estimates remain exploratory and controller-
dependent. Late timing does not improve the calibrated optimizer and does not
establish a robust advantage over early timing. Retain both as documented
comparators, not as a reason for another unregistered training expansion. The
experiment does not prove early attention is universally better, and it does
not establish that the temporal attention architecture itself is incapable.
No learned controller, joint training or new supervised objective is promoted.

## Evidence and completion checklist

The main source root is
`v2_portfolio_49d3c9a_20260914T143600Z` on persistent Lambda storage. It contains
`portfolio_comparison.json`, saved books/accounts, forecasts, selected policies,
training histories and every numerical migration receipt. The separate
`_settlement_audit` directory contains the exact-replay diagnostics. Accepted
engineering evidence is `docs/v2_portfolio_primal_adjoint_acceptance.json`.

Recovery is verified locally and on persistent storage: 11,464 required main
members and 1,829 attention members, across five local archives. This includes
every selected checkpoint, forecast, current book, training history, preprocessing
contract and numerical provenance needed to review the decisions and reproduce
scoring. The attention local packages total 865,091,652 bytes. Full intermediate
attention epoch snapshots additionally remain in a verified 3,982,625,033-byte
persistent archive; they are not needed to score the selected checkpoints.
Superseded numerical replay books and reproducible caches also remain verified
on persistent storage. No raw source was overwritten.

Primary evidence:

- [Main comparisons and settlement audit](v2_portfolio_policy_evidence.json).
- [Historical boundary/forecast bridge](v2_portfolio_bridge_evidence.json).
- [Attention screen, confirmation and optimizer comparisons](v2_portfolio_attention_evidence.json).
- [Main recovery receipt](v2_portfolio_policy_recovery.json) and
  [attention recovery receipt](v2_portfolio_attention_recovery.json).

All scientific work is complete. Local main, GitHub and the clean host checkout
matched commit `25443d75ab7f49ebd67f5316a6385a62cf84d623` before shutdown; the
verified Git bundle is retained locally and on persistent storage. Later closing
commits only record operational completion and the shutdown evidence.

The exact paid instance `ee625bd03bdc41fd9ba676de6e171511` was submitted for
termination at 20:02:23 UTC. Lambda inventories independently confirmed it absent
at 20:04:34.767 UTC and 20:04:43.167 UTC on September 14 (17:04 Brasília time).
The five-minute monitor is paused. See the [shutdown receipt](v2_portfolio_policy_shutdown.json).
No further experiment, forward capture or live deployment remains running.
