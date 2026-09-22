# Registration draft: post-hoc selection views, ensembles, smoothed parents and the short F schedule

Drafted 2026-09-22 by the cloud research session. **Status: drafted, not yet
authorized.** The drivers below bind this file by hash at freeze time; nothing
is frozen or executed until the user runs a freeze step. The evaluation boundary
of the September 21 continuation is inherited unchanged: the eight fixed
periods F2/F3/F6/F7/F10/F11/F13/F14, the six development reserves
F1/F4/F5/F8/F9/F12 unopened, no 2025/2026 consumer, no spliced continuous
account, no best-seed choice, no learned ensemble weights.

## A. Post-hoc selection views on saved forecasts (no fit)

Every completed F fit keeps all epoch weights, its per-epoch selection history and
the scores of the trainer's selected epoch. A view applies another rule to the
same saved history and scores the chosen epoch through the registered path on a
derived checkpoint that binds the sealed epoch file by hash. Rules:

- `raw`: the trainer's selection (the reference).
- `centre3`: the centre epoch of the best trailing three-epoch mean of the
  selection IC.
- `top3`: equal-weight rank average of the three best selection epochs.
- `around3`: equal-weight rank average of the raw selection and its neighbours.

Driver `ops/replay_forecast_variants.py`; library `brazil_rv.v2.forecast_variants`.
Books use the frozen matched-stopping evaluation inputs, dates, account,
equal-rank calibration and allocation (neutral 5% net; the flexible 45% policy is
a second frozen plan). Primary readout: paired fold-by-fold daily net-excess
deltas against the reference arm's existing R$10m books, circular block 40, 90
and 95 percent intervals, seeds and ensemble reported separately. IC is
diagnostic. Every variant is reported; none is adopted by this registration.

Retention rule for a selection rule: pooled 95 percent lower bound above zero on
the neutral policy for the three-seed ensemble, a majority of folds and seeds
positive, and no drawdown deterioration; then a single confirmation on the
flexible policy. Otherwise the raw selector stands.

## B. Rank-average ensembles of existing forecasts (no fit)

Equal-weight rank averages across stopping views (`p5+p20`), widths
(`TE_full_p20+TE_wide_p20`) and, once the common-model fits exist, families
(`C6+TE_full_p20`, `C6+GRU_early+TE_full_p20`). Same driver, same books, same
paired readout against the first member's arm. No weights are fitted; member sets
are fixed here. Retention as in A.

## C. Smoothed parent selection views and their children (GPU, parents reused)

On the six executed patience-20 attention parents, add the views `s3` (trailing
three-epoch smoothed score over the executed trajectory, centre epoch selected)
and optionally `s3p5` (patience 5 applied to the smoothed score). Each view binds
a sealed epoch file; a view equal to an existing view's epoch aliases that view's
completed child. Children use the unchanged child recipes, seeds, store and eight
periods. Driver `ops/extend_matched_stopping_views.py`; evaluation through A with
arms `TE_full_s3`, `TE_wide_s3` against `TE_full_p20`, `TE_wide_p20`. At most 48
child fits per view; execution waits for the current GPU queue.

## D. Short fully annealed F schedule (GPU, parents reused)

One factor: the child schedule, `schedule_epochs = epochs = 8` with patience 8,
against the 60-epoch cosine with patience 5. One or two cells per wave, parents
the patience-20 attention views (or trained common-model parents). Driver
`ops/run_trajectory_recipe.py`; evaluation through A with arm `<cell>_f8` against
`<cell>_p20` (or `<cell>`). Post-hoc views in A apply to these children too.

## E. Causal risk overlays on saved books (no fit)

`ops/evaluate_risk_overlay.py` on the saved continuous and period books: own-return
and BOVA11 volatility targeting with a 25 percent floor, rescale turnover charged
at 4 bps, paired circular block-40 intervals for mean and zero-rate Sharpe. Every
rule is reported; a rule that raises the zero-rate Sharpe with a lower bound above
zero on both families is confirmed by a ledger rerun at a daily gross cap equal to
the scale times the planned cap before any adoption.

## Boundaries common to A to E

Freeze before outcomes; preserve failed and skipped attempts; report all seeds
and folds with paired 20/40/60-session uncertainty (40 primary), currency-consistent
Sharpes, drawdown, turnover and holding information. No new capacity cell, no new
dataset, no held-out read, no recurring automation, no second GPU fit while a
worker owns the queue.
