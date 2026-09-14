# User-requested attention ASAM .5 extension

Authorized 13 September 2026 (America/Sao_Paulo), after Stage B calibration
and while the original Stage C screen was running. No Stage C evaluation
ranking had been inspected when this extension was specified.

Keep the original ASAM .2 attention screen and its calibration choice intact.
Additionally score ASAM .5 for TE_slow, TL_slow, TE_family and TE_all on the
same F2/F6/F10/F14 folds and seeds 11/29/47. This is 48 scored fits, of which
four TE_slow F2/F14 seed11/29 fits already completed in B can attach scores
without retraining: **44 new F fits, no new P fits**. The A–C total becomes
27 distinct P fits and 188 distinct F fits, with 156 scored C fits in 13 cells.

Use each corresponding .2 fit's identical selected SAM .125 parent, inputs,
preprocessing, full 60-session history, eligible stocks, architecture, loss,
LR/schedule, transfer multiplier, ASAM eta, selection/stopping and precision.
Only F ASAM rho changes from .2 to .5. This does not test ASAM pretraining.

Report paired .5-minus-.2 within all four variants, early-minus-late at .5,
family and FiLM increments at .5, and comparisons against the original S0
and C1 controls. Retain the original common D3/D5/D10 populations, three-seed
rank ensembles, individual/omission IC, 20/60-session paired date blocks,
10,000 draws and unchanged book economics. Include trajectory/module diagnostics.

This is an explicitly user-requested additional development comparison after
calibration, not a rewrite of the tie rule or an independent confirmatory
holdout. Report both radii, all outcomes and nominal uncertainty. No automatic
promotion, further radius search, new seeds/folds, D–F expansion, held-out
access or forward capture follows from the extension.

Reuse the frozen training checkout and keep aggregate fit concurrency at six;
queue the extra fits behind the original GPU screen and overlap original CPU
readouts where possible. A separate clean checkout can supply the extended
readout code. Preserve the original result and write the combined result
separately. Recovery, final report and exact-instance shutdown must include
the extension before this task is closed.
