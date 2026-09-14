# Portfolio implementation resolutions

These engineering details supplement the original registration before new
portfolio financial results. Forecast fits remain frozen to their separately
committed implementation and original recipes. No new architecture/data search
or change of development dates is authorized by these resolutions.

## Causal inputs and risk

The three ensemble midranks are normalized to [-1, 1] using each current
date/head population. A missing current or prior forecast has an explicit mask;
score changes require both. No target-validity filter controls entry eligibility.

Risk uses adjacent observed shareholder-wealth and BOVA returns with endpoints
t-60 through t-1. Gaps are not bridged. Economic beta retains the accepted
shrinkage/history/fallback contract. Residual variance around today's causal beta
is shrunk toward the current active-name median with twenty pseudo-observations.
The median requires twenty observed residual pairs per contributing name;
unsupported names retain eligibility and receive this risk prior. A 1e-8 daily
variance floor makes the QP strictly convex in portfolio weights. The optional
BOVA hedge has this numerical diagonal plus the market factor and zero assumed
excess alpha; no directional market forecast is invented. Initial risk warm-up
fallbacks (.02 stock residual and .015 market daily volatility) are fixed and
precede the actual policy sample, which has full market history.

Static controller inputs are three ranks, three valid score changes, current
and previous score validity, log daily volatility, beta, asinh daily borrow cost,
and prior published CDI value and validity. The archived CDI series starts on
2016-07-18: the first decision's prior-CDI feature is missing, with value zero and
validity false. Today's realized CDI is never substituted as a model feature.
The account still consumes today's CDI as its historical funding realization.

Static means/scales use fit stock-days only. Inventory features use fixed smooth
scales: weight/.05, log1p(age/20), asinh(marked adverse return/daily volatility),
pending commitment/.05, held flag, free and restricted cash fractions, joint
gross, signed net and beta. No stock is excluded by these transformations.
The MLP's final layer starts at zero, so epoch zero is exactly the calibrated
optimizer. Residual outputs have a .001 return-unit multiplier. The resulting
controller has 1,857 parameters; financial state, risk and constraints stay FP64.

## Chronology, boundaries and numerical acceptance

Policy calibration uses only fit entries whose five-session endpoint remains
inside fit. The ten existing purge sessions after fitting provide selection
burn-in; they are excluded from its utility. No pre-fit or overlapping-label
burn-in uses a model that could not yet have been trained. Evaluation uses the
ten sessions after selection for a common burn-in for all portfolio controls;
these are excluded from fold comparisons. A separate C6 old-policy empty-start
bridge preserves comparison with historical readouts. The continuous evaluation
book changes policies only at their evaluation boundaries and carries inventory.

Daily utility subtracts risk of the drifted inventory present before that day's
realization. Transaction costs are paid on realized fills. Training chunks carry
inventory, cash and claims; only the autograd graph is detached. The last session
of an exact evaluation/selection book closes inventory under the existing ledger;
training chunks and epochs do not receive a liquidation bonus or a last-mark
settlement windfall. Settlement uncertainty remains explicit in reporting.

Cost/funding sensitivities replay the frozen policy under realized 2/8 bps and
zero short-proceeds remuneration. Its planned trading-cost estimate remains
4 bps; the neural controller is not refitted or made aware of a hindsight cost
scenario. State and NAV respond normally to the changed realized expenses.
Seed-zero-residual selections reuse their mathematically identical optimizer
book, with explicit source hashes. Continuous books reuse this result only when
every fold selected epoch zero. Controller seeds are reported individually and
as a mean of daily outcomes, not represented as an implemented ensemble policy.

CPU policy jobs load each forecast family once per fold, fit the three seeds and
run the matched controls in that process. The initial concurrency is twelve,
with one BLAS/PyTorch thread per process; it may be adjusted from measured host
memory and throughput. GPU forecast jobs keep their independent six-job limit.
Paired intervals retain 10,000 draws and the original 20-session block sampling;
prefix sums avoid materializing repeated daily panels. A targeted numerical test
matches the established intervals to 1e-12 bps.

The three fresh C6 P trajectories and 102 new F trajectories are independent:
C6 F inherits existing S0 P, not the new C6 prelude P. A single six-worker GPU
queue runs these 105 fits, then scores the nine P prelude panels. This scheduling
change avoids leaving half the GPU fit slots unused during the three-parent
phase; it changes no model, seed, training window or forecast provenance.

The first integrated bounded fixture showed that a 1e-6 solver tolerance could
accumulate a joint-budget violation of 8.73e-6 NAV. Native OSQP now uses absolute
tolerance 1e-8 and relative tolerance 1e-8 in percent-NAV coordinates, retaining
the explicit 2e-6 NAV feasibility acceptance. Non-finite preferences are rejected
before solving. No capacity or economic risk limit is relaxed.

On 128 real 2016/2017 prelude market sessions with fixed synthetic preferences,
independent account/ledger NAV differs by at most 1.23e-10 NAV; cash and restricted
proceeds differ by less than 3.8e-9. Fit-only 32/64-session gradient cosine is .960
and norm ratio .861, passing the declared engineering bounds cosine>.8 and
ratio in (.5,2). This is accounting/gradient evidence, not an alpha result.
Local measured forward/backward cost is about 19ms per session including the
sequential account and QP; exact 128-day replay is 1.17s. Neither AMP nor compiling
this tiny CPU controller is justified by that workload. GPU forecasters retain
their original BF16 and compile settings. Full-fit timings remain to be measured.

Artifact: D:/quant-data/b3/interim/portfolio_policy_engineering/real_path/engineering_acceptance.json.
Any material actual-run discrepancy requires repair and a recorded decision
before economic conclusions; accounting checks are not replaced by optimization
success alone.
