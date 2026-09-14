# Cash-aware portfolio program: initial engineering acceptance

2026-09-14. Implements the locally testable foundation of the
[registered program](../research/preregistrations/v2_portfolio_policy.md).
No new financial research result is claimed by these tests.

The policy now replaces the ledger's rank/slot intentions through an observable
inventory/cash interface. The original rule remains the explicit comparator.
The new controller can vary every position, leave capital in cash and choose
the hedge jointly. Both policies use the same independent share/cash ledger:
action claims, segregated short proceeds, CDI/debit accrual, borrow fees,
observed close-proxy fills and labelled terminal settlement remain intact.
Unfilled reversal exits cannot become an assumed fill of the opposite position.
Rank-slot shortfall diagnostics are inapplicable to target portfolios and must
be omitted from their reports.

The training account differentiates through financial state while consuming
market realizations only after intentions are fixed. It matches the independent
ledger to 1e-12 absolute tolerance on controlled financing, gap, missing-print,
partial-fill/reversal, split, deferred-dividend, identity-conversion, stale
settlement and joint-hedge cases. Cash earns CDI once and has zero CDI excess.
Finite differences confirm inventory payoff gradients; detaching a chunk's
gradient preserves its financial state. Actual-data differential replay and
32/64-session gradient diagnostics remain required before financial acceptance.

Allocation uses fixed diagonal-plus-market risk, transaction/borrow costs and
joint gross/net/beta/name bounds, with optional cash. Sparse native OSQP solves
and its supported vector adjoint provide gradients to economic preferences and
inventory-dependent trade bounds. The adapter avoids the library's generic
torch wrapper's per-call full-machine thread pool and unnecessary risk-matrix
derivatives. Each autograd step retains its own solver state until backward.
See the primary [OSQP portfolio example](https://osqp.org/docs/examples/portfolio.html)
and [derivative interface](https://osqp.org/docs/interfaces/C.html).

A local 244-asset, 20-step synthetic allocation-only forward/backward diagnostic
took 55.352 ms/step initially and 20.865 ms/step after using percent-NAV solver
coordinates. This is an exact unit change, without fewer names, looser risk
limits or a smaller model. Both gradients and feasibility pass afterward.
These are solver microbenchmarks, not full sequential training speed claims.
Keep full account/replay/IO timings in the subsequent run report.

The forecast plan reuses 24 exact A–C fits, adds 102 F fits (30 S0, 30 TE_all,
42 C6), and three C6-family P fits solely for policy warm-up. The C6 F bridge
continues to inherit slow S0 P. Existing compatible S0/TE parents and the nine
parent score panels cover 361 initial out-of-fit dates before the fourteen
half-year blocks. The earliest missing fold and four C6 bridge folds receive
execution priority; this changes scheduling only.

Historical parents are restored from the sealed recovery inventory with exact
file hashes. Later scoring commits require unchanged inference dependencies,
as well as matching checkpoint/store/preprocessing contracts. A real S0 P
checkpoint successfully scored the first two prelude dates on CPU with full
60-session history and the compact active axis. Its diagnostic artifact is at
`C:/quant-data/b3/interim/portfolio_policy_engineering/S0_parent_2date/score_manifest.json`.
It is an engineering sample; the complete financial panels will be regenerated
on the registered compute run. Dates within P's information boundary are
rejected, and no 2025/2026 consumer read is involved.

Targeted acceptance comprises 121 tests: 77 standing ledger tests, one existing
action causality test, 15 allocation/account/policy tests, 11 existing score
tests, 14 training/resume tests and three plan/provenance tests. They passed in
bounded invocations; Ruff passes on every changed Python file. Windows checks
use OPENBLAS_NUM_THREADS=1 and OMP_NUM_THREADS=1 to avoid OpenBLAS's allocation
failure when constructing its default full-machine worker pool.

The remaining work is causal cache assembly, risk/calibration inputs, the small
policy fitter and exact selection/replay, development comparisons and continuous
accounting, result-dependent follow-up decisions, complete recovery and shutdown.
