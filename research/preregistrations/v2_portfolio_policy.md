# Cash-aware portfolio policy and repaired C6 bridge

Registered 2026-09-14 before any new fit. The user authorized the execution
reassessment and experiments, including cash, efficient implementation and paid
compute. This is development research; 2025/2026, forward capture, deployment
and intraday intervention remain excluded. Historical A–C results stay sealed.

## Order and scope

1. Accept the allocation gradients, causal intentions and independent accounting
   tests locally. Engineering changes may resolve test failures; record material
   contract amendments before looking at new financial results.
2. Refit C6 on accepted repaired inputs with its original fundamentals+magnitudes
   roster and inherited slow S0 P recipe. Preserve the old policy as control.
3. Build one chronological out-of-fit forecast cache for S0, TE_all ASAM .2, and
   repaired C6. Reuse compatible A–C parents and F2/F6/F10/F14 fits. Complete the
   other ten folds for S0/TE and all fourteen C6 folds, with seeds 11/29/47.
   No model-history truncation, active-name removal or lower-precision accounting.
4. Compare cash, incumbent rank-buffer policy, calibrated constrained optimizer,
   and a small inventory-aware learned policy on each frozen forecast family.
   Report the four original screen folds and remaining ten separately, plus
   all-fold and continuous-book results. Policy fitting/selection is expanding
   and chronological; evaluation outcomes never select earlier-fold policies.
5. Conditional follow-ups need a written result-dependent decision before fits:
   rich early/late attention if TE remains competitive after the policy change;
   a raw-return/neutral auxiliary objective if objective alignment remains the
   supported bottleneck; joint encoder/policy training only after the frozen
   policy demonstrates incremental utility. This is not a broad factorial.

## Forecast provenance and policy chronology

All new fits resolve docs/v2_data_inputs.json and bind its manifest; evaluation
uses the accepted repaired economic arrays. Existing checkpoints are reused only
with their exact input/preprocessing/parent contracts. New run manifests record
source hashes and the implementation commit.

P checkpoints, whose fit/selection information ends by 2016-06-30, generate an
out-of-fit prelude from 2016-07-18 through 2017's last session. S0 and TE reuse
compatible P. Three fresh C6-family P fits supply only this initial policy-cache
prelude; C6's fourteen F fits still inherit S0 P. This deliberate P-to-F change
in the forecasting process is recorded, including the C6 transfer difference.
The later cache concatenates the fourteen original F evaluation blocks. Each
date's source fit and selection labels must finish strictly before that date.

The forecaster is never refit inside a policy trial. Three forecast seeds form
one equally weighted ensemble of within-date/head midranks, matching the
accepted comparisons, with component scores retained. Policy seeds
11/29/47 measure controller initialization sensitivity; they do not multiply
forecaster training. The deterministic optimizer is fitted once per fold.

Policy windows use the original fold fit/10-session purge/55-session selection/
10-session purge/evaluation boundaries, restricted to available out-of-fit cache
dates. Raw return calibration endpoints must remain within their fit/selection
window. All feature transforms fit on policy fit dates only. A–C's global
architecture/radius choice remains retrospective development selection; neither
this cache nor chronological policy training turns it into a historical live
strategy. Results are development evidence, not an untouched holdout.

## Accounting and risk contract

The decision is the existing 15:45 snapshot, with prior observed marks; actual
fills use the later close proxy. Orders freeze notional or a held-position
fraction before that close. Keep action/identity transitions, cash claims,
missing-print valuation, shortability, and borrow accounting. A missing future
price is not a zero return and is not a decision-time exclusion.

Cash is an explicit residual asset, without minimum gross or stock counts.
Historical CDI accrues under the existing free-cash/debit/short-proceeds ledger
conventions. Cash benchmark earns the same CDI: its net excess is zero. Report
absolute returns, CDI excess, free cash, restricted proceeds, financing and
borrow separately. Restricted short-sale proceeds are not spendable cash.
Under remuneration 1/debit spread 0, a neutral stock book can also earn funding
income; CDI is not awarded only to the cash control or counted twice.

New controllers jointly allocate stocks and optional BOVA11 with planned gross
at most 2.25 NAV including the hedge, individual stock weight at most .05,
hedge absolute weight at most .60, absolute signed net at most .05 and estimated
absolute market beta at most .05. The legacy control retains its original risk
rules and post-hoc hedge; report that distinction. No evaluation-volatility
rescaling. Unfilled orders may cause realized drift outside planned constraints;
report it and request causal reductions, never invent fills. No sector/volatility
occupancy quotas are imposed on the new policies.

Use the original 4 bps per side for stocks and hedge, accepted archive lending
rates and registration fees, original short-proceeds remuneration 1 and debit
spread 0. These are research assumptions. Replay 2/8 bps and remuneration 0 as
sensitivities without refitting or choosing a winner from those sensitivities.
Do not claim live executable performance or capacity from normalized NAV.

## Allocation and learner

Use a convex allocation layer with diagonal idiosyncratic risk plus the causal
market factor, linear trading and short-holding costs, and the above constraints.
Use economic beta's existing causal history contract; estimate risk with past
returns only. Fixed daily risk-aversion coefficient 5; no evaluation tuning.
The optimizer plans a five-session holding interval: five times daily expected
excess return/risk/borrow, paying the immediate trading cost once. It replans
daily. The three overlapping forecast heads are explanatory variables for one
calibrated payoff, not independently summed returns.

The deterministic control uses fit-only ridge calibration of the three
cross-sectional score ranks to five-session shareholder return above CDI,
converted to daily units. Average loss by date, so larger universes do not
implicitly overweight dates. Ridge coefficient 1e-3 on standardized predictors.

The learned control starts from that calibration and adds a shared MLP with
two 32-wide SiLU layers. Inputs: three ranks, causal score changes and validity,
volatility, beta, borrow, own drifted weight, inventory age/adverse movement,
portfolio cash/gross/net/beta and pending commitment. The prior published CDI
return may also enter the cash context; the current session's realized accrual
is accounting-only. The output is a decision preference in return units,
not an asserted calibrated forecast.
Risk limits and actual costs cannot be learned away. No ticker embedding.

Train on non-overlapping daily net excess increments minus one half of 5 times
the causal portfolio variance. Use Adam at 1e-3, maximum 30 epochs, selection
every epoch, patience 5, minimum selection utility improvement .01 bp/day.
Retain epoch-zero calibrated policy as a selectable checkpoint. Clip gradient
norm at 1; no dropout/weight-decay search. Three matched seeds. Use chronological
32-session truncated-backprop chunks, carrying actual inventory/cash across
chunks and detaching gradients only; no chunk liquidation/reset. Selection
has causal burn-in and selects by exact-ledger net utility, not the surrogate
alone. Compare 64-session gradients on a bounded fit-only diagnostic; material
instability requires a recorded engineering resolution before financial claims.

Sparse native QP solving and its supported adjoint derivatives are preferred
to a hand-built dense differentiable solver. Financial state is FP64; neural
work may use FP32. Benchmark end-to-end sequential work, including backward
and exact replay; use AMP/compile only where they improve that workload without
changing financial calculations. GPU forecasters keep accepted BF16/compile,
unique dates and full stage-compact active axes.

## Acceptance and decisions

Tests cover all-cash CDI/self-financing, cost/borrow signs, price-gap intentions,
missing prints, actions/identity/claims, pending exits and reversals, constraints,
future mutation, out-of-fit provenance, finite-difference allocation gradients
and sequential gradient sensitivity. Differential accounting must reconcile on
controlled paths. Actual-data surrogate differences must be quantified; any
material discrepancy blocks financial conclusions until explained or repaired.

Rank policies by paired mean daily net utility; report net P&L and uncertainty
separately. Paired intervals resample contiguous 20-session blocks, preserving
same-date model pairing; report fold and seed sensitivity. Advancement requires
positive mean paired utility and net excess versus its same-forecast incumbent,
positive paired utility in at least three of four original screen folds and in
the remaining-fold aggregate, no unresolved accounting/feasibility failure,
and no material reversal across policy seeds. A nominal 95% paired interval
above zero supports stronger evidence; otherwise the result is exploratory.
No automatic 2025 read or replacement based solely on the best point estimate.

The continuous book carries inventory through model switches and liquidates
only at the final boundary. Artificial terminal settlements remain visible and
are never a training reward or a policy feature. Record incomplete/unresolved
books rather than conceal their exposure in a pooled mean.

Recover/hash all decision artifacts, forecasts and required checkpoints locally
and on persistent storage; commit/push the implementation and comprehensive
review. Terminate the exact paid instance after recovery and verify absence.
