# Direct portfolio-objective progress

Current registration: [v2_portfolio_objective.md](../research/preregistrations/v2_portfolio_objective.md).
Authorized implementation and experiments continue autonomously on the RTX 2060.

## Stage A

The 22 additional CPU replays completed under source e52be14; four sealed full-map
books were reused. Root: resolve [the run pointer](v2_portfolio_objective_run.json).
Closeout summaries include all financing/cost scenarios, annual contribution,
realized beta, issuer proxies and exact settlement records. Full net .05/.45
results remain 4.000/6.707 daily bps for C6 and 4.497/5.937 for TE_all above CDI.
With the intercept removed they are 4.466/5.823 and 4.506/5.225 respectively.
Intercept-only delivers .503/2.154 for both arms. Attribution is nonlinear: do not
add these books' P&L. Flexible exposure is partly an unconditional common residual
tilt, not evidence of conditional timing. Rank-only flexible exposure also gains.

At zero short-proceeds remuneration the full neutral/flexible books fall to
1.037/4.074 for C6 and 1.419/3.221 for TE. At 8 bps/side they are 3.513/6.172 and
3.958/5.370. A 3% debit spread produces 3.893/6.377 and 4.394/5.649. These are
separate accounting sensitivities, not verified brokerage terms or new defaults.
The primary allocator remains net .05/beta .05 as pre-registered.

Largest stale-mark settlements include TRPL, CIEL, BRML, ALSO and CPLE units.
The accepted store explicitly has inferred development-grade action terms and
zero established ISIN succession links. The events must not be silently treated
as observed executable fills or removed from the universe. Source-level event
inspection and compact review export are being finalized; no new financial
conclusion is promoted from these sensitivities.

## Stage B

New chronological account-gradient tests exposed an allocator backward defect:
on the eight-session known synthetic fixture, the old analytic derivative was
+.285666 while centered numerical perturbations gave -1.336235. The account-only
derivative agreed. Changing OSQP objective scaling changed the derivative despite
the mathematically identical optimization, so a single rescaling is not accepted
as a repair. The implementation now differentiates the exact active face using
the positive diagonal-plus-market Hessian and the small exposure constraint
system. The same Clarabel forward portfolios are preserved. Twenty-eight targeted
tests pass, including multi-day finite differences and the real small hedge
variance. Historical financial/QP gradient checks and local GPU admission remain.

The new preference adds an unranked cardinal return head to the differentiable
rank anchor. Training uses actual chronological accounts and clones state for
SAM/retries. It preserves the 60-session model history and every eligible name.
Implementation also separates selection from the inherited model's purged label
endpoints. The continuation budget and loss-scale rule are registered before runs.

## Stage C — not launched

Required: local GPU acceptance; freeze verified implementation; run the matched
72-fit screen (two arms, three objectives, four folds, three seeds), compare both
selectors on the same trajectories; confirm admitted pairs only; final economic
readouts, archive/audit and combined LLM review. No new pretraining or Lambda.
Re-read the registration after each stage. No held-out access or forward capture.
