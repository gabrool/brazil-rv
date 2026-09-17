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

## Stage C — running

The matched 72-fit screen launched on 2026-09-17 at 13:28 UTC / 10:28 Brasilia.
Its frozen clean worktree is `C:/quant/brazil-rv-portfolio-objective-33086be`.
The source main branch may receive reports; do not modify this execution worktree.
Resolve the run pointer, then check `local_execution.json` and actual command lines
before any restart. `local_campaign.stdout.log` contains epoch progress;
`local_campaign.stderr.log` and `local_runner_exit.json` report failures. The
supervisor prevents sleep while running and restores the normal setting on exit.
Do not launch a duplicate. The existing 15-minute heartbeat now follows this task.

The first completed trajectories passed hash, matched-parent, zero-initial-head,
shared loss-scale and both checkpoint-selector audits. The audit command can be
repeated after all fits: `uv run --project research --no-sync python
ops/audit_portfolio_objective.py fits --root <resolved-root>`. Its receipt is
`completed_fit_audit.json`. Do not use early selection results to alter the design.
The input-population audit also verifies identical dates/ISIN axes and all 363,314
active stock-days in the economic cache's 2016-07-18 through 2024-12-30 interval.
Neither arm's cached score masks remove any accepted active observation. This is
the accounting interval, not the full 2010-onward store and feature-history span.

The supervisor compares both selectors on the same trajectories, admits candidates
using the economic selector alone, and runs confirmation only for admitted pairs.
`worker_complete.json` means financial readouts are finished, not that recovery and
review are complete. Remaining: verify the screen and any triggered confirmation,
all forecast populations/axes and hashes, paired conclusions, exact ledger and
benchmark readouts; complete the LLM review; archive/hash recovery on D; commit/push
and verify GitHub; pause the heartbeat. No new pretraining or Lambda. Re-read the
registration after each stage. No held-out access or forward capture.
