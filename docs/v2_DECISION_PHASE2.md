# Phase 2: controller engineering and chronological screen

**Complete. Neither conditional controller advances.** The MLP failed its
engineering admission criterion; the eight admitted conditional financial fits
all completed. No additional Phase 2 confirmation/continuous campaign is triggered.
See [the source-bound numerical results](v2_decision_phase2_results.json).

This phase tests whether a learned decision model can demonstrate the intended
behavior under the real allocator/account, then improve a bounded chronological
comparison. It uses the original sealed forecasts, not newly trained encoders.
The reference is the coherent benchmark-residual calibration from Phase 1;
equal-rank allocation and CDI cash are additional controls. The registered
candidate without fallback determines advancement.

## Actual account defects corrected

The synthetic checks found a material **state disagreement hidden behind tiny
positions**. The training account submitted stock changes below the exact ledger's
existing 1e-10-NAV order threshold. Their monetary value was negligible, but their
presence changed holding/age flags supplied to a stateful controller. Training
now applies the same existing order-intention threshold, without deleting actual
inventory or changing any eligible name.

A related reversal defect occurred when an opposite entry crossed a residual
too small to submit an exit. The exact ledger retained the old side's holding age.
Both accounts now start the new side's age and cost basis correctly. On a trained
144-session synthetic trajectory, maximum NAV disagreement fell from 1.186e-4
to 9.882e-12. A dedicated regression covers ignored intentions, actual residual
holdings and cross-side entries. The 22 targeted account/policy/allocation tests pass.

Nearly flat QP directions also required tighter forward-solver objective accuracy
and more adjoint iterations in rare difficult cases. Financial constraints, costs,
eligibility and the forward/adjoint weight-agreement bound are unchanged. Twenty-four
current Phase 2 reference books reproduce the corresponding Phase 1 books within
the already registered daily/aggregate bounds: maximum daily bps-field error
.0011512, maximum mean bps error .00001129. Earlier results remain sealed under
their actual source implementation; no claim of byte-identical solvers is made.

## What passed and what did not

The conditional model adjusts calibrated preference using three linear outputs
from common context: alpha scale, average-rank contribution and common residual
return. Inputs include causal market state, known ages/masks, actual cross-seed
rank disagreement and matured shadow outcomes. A shadow observation at decision
t may first use origin t−6, whose five-session payoff ends at close(t−1). It keeps
updating while the actual account is in cash. Fit-only median/IQR plus asinh
scaling does not remove observations. The fixed forecast input is the original
three-seed rank ensemble. The linear controller itself is deterministic across
initialization seeds, so one unique fit is sufficient; it is not three replications.

The stateful extension adds a shared two-layer, 32-unit SiLU residual using actual
holdings, age, marked P&L, costs and pending state. After joint training proved
unreliable, a bounded staged attempt froze a learned conditional base and trained
the zero-initialized stateful residual. All attempts and seed outcomes are retained.

The initial synthetic null was flawed: a systematic adverse-rank shock also hit
nominal zero-opportunity episodes. That is negative predictable alpha, not a
valid inactivity test. The corrected null contains independent zero-mean noise.
The original small fixture also supplied too few independent null episodes, so
the final engineering fixture has 1,920 sessions in balanced, randomized
16-session regimes: 1,152 fitting, 304 selection and 432 independent evaluation
sessions, with 16-session gaps. These were engineering corrections before real
financial outcomes, not changes to financial gates.

The final conditional model passed profitable independent trading, improvement
over unconditional allocation, inactive-regime exposure reduction, differentiated
continuation/reversal responses, paired cost sensitivity, trained-account parity
and terminal materiality. Its zero-opportunity average gross fell from 1.4202 to
1.0138, a 28.6% reduction. This is useful learning, **not perfect inactivity**.
Synthetic profitability is evidence of implementation capability, not financial alpha.

The stateful MLP did not pass its prerequisite across the three fixed seeds:

| Staged MLP seed | Selected epoch | Result |
| --- | ---: | --- |
| 11 | 0 | Retains the accepted conditional model; no incremental MLP learning |
| 29 | 10 | Only 16.8% zero-state gross reduction, below the predeclared 25% gate |
| 47 | 12 | Passes the behavioral checks |

Seed 29's failure is specifically an inactivity failure, not an inability to
trade profitably or respond to costs. The admission rule requires all three seeds
to pass. Rather than continue tuning the fixture/model, the branch was closed
before financial dispatch: **24 planned MLP financial fits were not run**. The
eight deterministic conditional fits remain the financial screen. The complete
16-attempt engineering evidence is in
[the engineering record](v2_decision_phase2_engineering.json); the precise recipe
and amendments are in [the registration](../research/preregistrations/v2_decision_phase2.md).

## Financial comparison

All eight registered fits completed. The process retains each fit's full
selection/training curve, selected epoch, exact candidate and fallback books,
fills/orders, costs, exposures, settlement flags and source hashes. The fallback
chooses cash, benchmark or learned policy solely from the earlier selection
window; its evaluation improvement alone cannot pass the learning gate.

The financial boundary is F2/F6/F10/F14, the same 55-session prior selection and
10-session purges as Phase 1, historical CDI cash and the same risk/cost account.
Every fit may run up to 30 epochs, with five-epoch patience after at least 20.
Earlier source commits on completed fits are preserved; later commits prepared
the independent Phase 3 readout or repaired launch/resume handling. They did not
change the registered financial model between folds.

All figures below are bps/day above CDI, including the stated costs and accounting.

| Forecast arm | Fold | Selected epoch | Conditional | Benchmark control | Equal-rank control |
| --- | --- | ---: | ---: | ---: | ---: |
| TE_all | F2 | 9 | −6.681 | −2.328 | −0.200 |
| TE_all | F6 | 0 | −3.974 | −3.974 | −6.162 |
| TE_all | F10 | 9 | −2.682 | −1.040 | +0.173 |
| TE_all | F14 | 19 | +2.537 | +8.900 | +9.643 |
| C6 | F2 | 2 | +7.321 | +4.734 | +3.027 |
| C6 | F6 | 3 | −2.825 | −3.852 | −4.450 |
| C6 | F10 | 20 | −3.528 | +0.270 | −0.135 |
| C6 | F14 | 0 | +9.415 | +9.415 | +9.960 |

| Contrast: conditional minus benchmark | TE_all | C6 |
| --- | ---: | ---: |
| Net, weighted by evaluation sessions | −3.086 | −0.067 |
| Nominal 95% interval, 40-session circular blocks | [−9.580, +2.517] | [−6.481, +5.724] |
| Utility | −3.366 | −0.530 |
| Utility interval | [−9.867, +2.251] | [−6.934, +5.260] |

The conditional absolute net above CDI is −2.658 for TE and +2.572 for C6. C6 is
essentially tied with its benchmark on net, but its additional risk worsens the
utility comparison. Its two improved folds do not satisfy the three-of-four rule.
TE does not improve any of the four folds. Neither meets positive paired net/
utility; the −.25-bps materiality floor also fails. For the deterministic
conditional model this is one controller's aggregate comparison, not evidence
about variation across independently trained controller seeds.

The past-only fallback chooses cash in TE/F14. That earns exactly CDI, missing the
conditional model's +2.537 bps/day and the benchmark's +8.900 in that period.
Its aggregate TE net contrast consequently falls to −3.729 bps/day. C6's fallback
matches its candidate books. These choices were made before evaluation; choosing
cash retrospectively only for bad periods would be hindsight leakage.

All primary paired intervals span zero. This is a failed advancement screen,
not a statistically established universal inferiority result. Twenty/sixty-session
sensitivities, all curves and decomposition fields are retained in the numerical
record. Settlement flags and terminal haircut sensitivities remain visible;
historical unobserved fills and the assumed financing terms are not resolved by
controller learning or account reconciliation.

Local import-memory and clean-worktree restart failures were recovered before
affected fits began. The old logs remain, and the final launcher verification
skips all eight completed manifests. No selected fit was discarded or refit to
improve an outcome. Financial training/account/calibration sources are unchanged
from the first admitted fit; later commits prepared the independent objective
experiment and its reports/continuation code.

## Interpretation and continuation

An admitted synthetic controller can still fail to forecast real changing
opportunity. In the synthetic problem the regime is explicitly observed. In the
financial problem, market state and matured shadow payoffs are noisy proxies;
strong past-window utility can fail in the next period. Separate optimizer/account
correctness, demonstrated behavioral learning, and chronological financial
generalization. Failure at one level is not proof that every ML decision method
must fail or that the new forecaster objective cannot help.

Phase 3 is independently authorized even if no controller advances. Its fixed
policy was chosen before these financial results. Lambda is deferred by the user;
the exact remaining objective experiment, compatible parents, full-data CPU
checks, hashes, source locations and restart procedure are preserved in
[the restart document](v2_DECISION_RESUME.md). No Phase 3 alpha result exists yet.
