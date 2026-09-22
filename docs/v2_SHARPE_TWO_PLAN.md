# From 1.88 to a zero-rate Sharpe above 2: diagnosis and entry point

Status: diagnosis with runnable tooling, first written 2026-09-21 and revised
2026-09-22 after the local handoff. The full ranked program, including the items
beyond the Sharpe-2 gap, is `docs/v2_RESEARCH_PROGRAM.md`; the registration
draft is `research/preregistrations/v2_posthoc_selection_and_ensembles.md`.
Nothing here adopts a model, reads a held-out consumer or changes a registered
result. Every number quoted from earlier work names its source document.

## The answer

Implementation changes come before any new feature or dataset, because they
change the ruler as well as the model:

1. Fix checkpoint selection. Today's rule makes the chosen weights largely a draw
   from selection-window noise, which is the most likely cause of results that
   reverse between data versions. The fix is evaluated as post-hoc views over the
   executed trajectories (no trainer change) and, on the GPU later, as smoothed
   parent views and a short annealed child schedule.
2. Evaluate paired. The four-fold economic screens cannot see a 0.6 bps/day
   effect; paired fold-by-fold deltas on common dates can.
3. Harvest what is already in the saved forecasts: ensembles across views, widths
   and families, and a causal volatility overlay. Each is cheap and estimated at
   a material fraction of the gap.

## The size of the gap, in the units the pipeline produces

The historical continuous C6 result (`docs/v2_FOUNDATION.md`,
`docs/v2_R31_EXECUTION_SWEEP.md`) is 6.71 bps/day above CDI at a zero-rate
Sharpe of 1.88 over 1,738 sessions. With CDI near 3 bps/day the absolute mean is
about 9.75 bps/day, which implies a daily standard deviation near 82 bps (13
percent annualised).

| Route to Sharpe 2.00 | Needed change |
|---|---:|
| Same volatility, higher mean | +0.6 bps/day (about +9 percent of the absolute mean) |
| Same mean, lower volatility | about -6 percent daily standard deviation |
| Two books of equal Sharpe and daily correlation .755, blended 50/50 | Sharpe x 1.067, the whole gap |

Those are small relative to the noise of any single experiment:

| Statistic | Approximate standard error |
|---|---:|
| Mean excess of one continuous book over about 1,600 development days | 2.0 bps/day |
| Paired difference of two books with daily correlation .90 | 0.9 bps/day |
| Paired difference of two books with daily correlation .95 | 0.65 bps/day |
| Four-fold net delta screens in the capacity waves | +/- 2 to 3 bps/day (`docs/v2_CAPACITY_RESULTS.md`) |
| Pooled daily IC mean over 1,598 days | .0046 |
| Mean IC over one 45-day selection window | .027 |
| Mean IC over the 160-day P-stage internal holdout | .0145 |

A change worth exactly the gap is a 0.3-sigma effect on one book and invisible
to the four-fold screens. Only paired comparisons of near-identical books, or
IC-primary tests pooled over all folds, resolve it.

## Diagnosis: the trajectory, not the architecture, dominates the differences

Round-7 selection scores each epoch by the mean common-population D3/D5/D10 IC
on a 55-session window whose D10 labels must stay inside the window, so at most
45 days decide a fit (`splits.py`, `round7_training.py`). With daily IC
dispersion near .18 that mean has a standard error near .027; per-epoch means in
saved histories differ by about .005 between neighbours. The rule picks whichever
epoch's noise draw was highest and stops after five non-improving epochs, which
at this noise level is close to a geometric stopping time. The audit quantifies
this on saved histories without touching evaluation labels:

```
uv run --project research --no-sync python ops/audit_selection_noise.py \
    --root <attention root>/fits --stage F --output docs/v2_selection_noise_F.json
uv run --project research --no-sync python ops/audit_selection_noise.py \
    --root <attention root>/parents --stage P --output docs/v2_selection_noise_P.json
```

Evidence already in the repository:

- The one-factor patience diagnostic (`docs/v2_ATTENTION_REVERSAL_DIAGNOSTICS.md`):
  the corrected TE_wide P seed 29 parent stopped after nine epochs and selected
  epoch 4 under patience 5; under patience 20 it ran 51 epochs and selected epoch
  31. Its F10 child improved by +7.05 bps/day above CDI, and the three-seed
  ensemble with only that seed replaced by +2.36 bps/day. The running matched
  stopping experiment is the general test of this mechanism.
- F-stage fits select at epochs 3 to 6 of a 60-epoch cosine schedule, so the
  selected child is a near-peak-learning-rate iterate. Fit-probe IC keeps rising
  (.046 to .082) while the selection mean falls (.042 to .023).
- Seed-pair forecast correlation is .777 for attention against .914 for C6
  (`docs/v2_FOUNDATION.md`); rank-averaging three seeds already added .474
  bps/day. That gain is trajectory noise being averaged out.
- Width and depth contrasts landed inside +/- 2 to 3 bps/day intervals on four
  folds, so "scaling does not help" is unproven either way. GBDT's IC of .017
  against .027 for S0 is a representation gap (no sequence encoder, no
  cross-sectional context), not evidence that fewer parameters win.
- The worst fold (F6, 2020 H2) is a universal loss with forecasts about .7
  correlated with momentum: a risk-control problem.

## Why there is no trainer change

An earlier draft added a smoothed selector to `TrainingRecipe`. It was withdrawn:
`verify_reused_inference_source` compares `round7_training.py` byte for byte with
each checkpoint's training commit, so any trainer edit would block post-hoc
scoring of every existing checkpoint from that checkout. Smoothing and patience
only change which saved epoch is selected, so both are expressed as labelled
views over executed trajectories, exactly as the matched stopping experiment
treats patience 5 against 20. `brazil_rv.v2.selection_rules` holds the view
rules; `brazil_rv.v2.forecast_variants` scores alternative epochs through the
registered path on hash-bound derived checkpoints.

## Tools in this change

| Path | Purpose |
|---|---|
| `research/src/brazil_rv/v2/selection_rules.py` | trailing smoothing, Newey-West mean SE, selection-noise summaries, raw-prefix and smoothed views, epoch view sets |
| `research/src/brazil_rv/v2/forecast_variants.py` | derived checkpoints, on-demand CPU scoring, panel loading, rank-average composition |
| `research/src/brazil_rv/v2/risk_overlay.py` | causal trailing volatility, scale rules, rescale-cost charging, paired circular block bootstrap |
| `ops/audit_selection_noise.py` | audit saved histories; no evaluation labels |
| `ops/replay_forecast_variants.py` | frozen plan, books and paired summaries for post-hoc views and ensembles |
| `ops/extend_matched_stopping_views.py` | smoothed parent views and their children wave |
| `ops/run_trajectory_recipe.py` | short fully annealed F schedule wave |
| `ops/evaluate_risk_overlay.py` | overlay rules on saved books |
| `research/tests/test_v2_selection_rules.py`, `test_v2_forecast_variants.py`, `test_v2_forecast_variants_integration.py`, `test_v2_risk_overlay.py` | coverage |

## What not to do now

- More width, depth or attention-variant waves before the selection rule is settled.
- Best-seed, best-fold or best-epoch choices of any kind.
- New datasets before the trajectory fix; their measured effect would reverse for
  the same reason wider attention did.
- Any GPU fit while the continuation owns the queue.
