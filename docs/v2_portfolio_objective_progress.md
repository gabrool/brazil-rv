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
as observed executable fills or removed from the universe. The source-level
inspection and [compact export](v2_portfolio_objective_closeout.json) are complete:
they bind last/next same-ISIN quotes and available action rows for the largest
events. Contractual successor/cash dispositions remain unverified. No new financial
conclusion is promoted from these sensitivities. Stage A is complete; its registered
freeze is the established neutral allocator, with the other arrangements as sensitivities.

## Stage B

New chronological account-gradient tests exposed an allocator backward defect:
on the eight-session known synthetic fixture, the old analytic derivative was
+.285666 while centered numerical perturbations gave -1.336235. The account-only
derivative agreed. Changing OSQP objective scaling changed the derivative despite
the mathematically identical optimization, so a single rescaling is not accepted
as a repair. The implementation now differentiates the exact active face using
the positive diagonal-plus-market Hessian and the small exposure constraint
system. The same Clarabel forward portfolios are preserved. Thirty targeted tests
pass, including multi-day finite differences, the real small hedge variance,
SAM account-state resets and label-mutation independence of neural preferences.
Three historical twelve-session gradient paths agree with numerical perturbations;
independent account NAV error is below 8e-13. All four representative local GPU
cases (C6/TE_all, F2/F14, seed 11) pass. Warm 32-session SAM steps take .765-.938
seconds and peak allocated GPU memory is 1.32-3.14 GiB. The smooth-rank interface
bridge is complete: mean absolute preference differences are .00560 bps for C6
and .01686 bps for TE_all on 64 initial F2 fitting dates. The temperature was not
selected from financial outcomes. All engineering checks finished before dispatch.
See [the compact evidence](v2_portfolio_objective_engineering.json). The engineering
source is 295f416; financial source 33086be preserves the identical training and
allocation mathematics. The first engineering attempt's donated-buffer error is
retained; separate forwards for gradient-norm measurements fixed it before fitting.

The new preference adds an unranked cardinal return head to the differentiable
rank anchor. Training uses actual chronological accounts and clones state for
SAM/retries. It preserves the 60-session model history and every eligible name.
Implementation also separates selection from the inherited neural model's purged
label endpoints. The old portfolio-calibration burn-in convention cannot simply
be reused for a warm-start neural model trained on those endpoints. This does not
assert that the old frozen-OOS controller calibration leaked. The continuation
budget and loss-scale rule are registered before runs.

## Stage C — financial work and audits complete

The screen completed 72 fits and admitted only TE_all utility-only. Confirmation
completed the registered sixty matched rank/utility fits on the remaining ten
folds. All 132 trajectories pass the final fit audit, including exact warm-start
weights, zero initial cardinal head, shared fit-only loss scale, artifact hashes
and both selectors reconstructed from the same saved trajectory. No zero-gradient
blocks or AMP retries occurred.

Confirmation fails: primary paired net -.177 bps/day, 40-session interval
[-2.467, 2.166], paired utility -.233 [-2.525, 2.110], only one positive seed.
Neither loss nor economic selector is promoted. The IC-selector diagnostic also
has an interval crossing zero. No additional C6/hybrid fits were run.

The supervisor exited successfully at 00:38:46 UTC September 18 after all fold
readouts and sixteen continuous accounts. Final verification passes for 318
forecast archives, 742 fold books, sixteen continuous books and eighteen panels.
Checks cover hashes, dates/ISIN axes, accepted eligible populations, ten-session
post-selection burn-in, 1,738-session continuous coverage, reconciliation,
recomputed benchmarks/Sharpes and reconstructed confirmation intervals/gate.
The complete LLM review is docs/v2_PORTFOLIO_OBJECTIVE.md; all confirmation cells,
stresses, seeds, continuous metrics and training diagnostics are exported in
docs/v2_portfolio_objective_results.json. The actual financial source remains
clean detached 33086be, not the later documentation commit.

Stages A/B/C have been checked against the registration. Financial limitations
remain: contractual settlement and financing assumptions, reused development
periods and warm-start continuation scope. Prior OSQP-gradient-dependent learned
controller failures remain qualified; their forward books are unchanged.

Recovery is complete: all 8,443 files in the 9.78 GB D-drive archive were read
back and verified against the SHA-256 inventory. The recovery receipt binds both
archive and inventory. Final documentation is committed/published with this state;
the heartbeat is paused after verifying GitHub. Do not launch more experiments.
No held-out consumer access, Lambda, forward capture or deployment.
