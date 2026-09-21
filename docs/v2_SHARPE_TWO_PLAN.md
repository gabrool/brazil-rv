# Getting from 1.88 to a zero-rate Sharpe of 2: what to change before adding data

Status: research plan with runnable tooling, 2026-09-21. Nothing here adopts a
model, reads a held-out consumer, or changes any registered result. Every number
quoted from earlier work names its source document.

## The answer

Yes. There are implementation changes worth running before any new feature or
dataset, and they should come first because they change the ruler as well as the
model. In order of expected value per GPU hour:

1. Fix the checkpoint trajectory. The current stopping and selection rule makes
   the chosen weights largely a draw from selection-window noise. This is the
   most likely cause of the results that "do not make sense" (wider attention
   reversing between data versions, capacity changes inside noise, seeds
   disagreeing). Code: `TrainingRecipe.selection_smoothing`,
   `ops/run_trajectory_recipe.py`, `ops/audit_selection_noise.py`.
2. Change the evaluation protocol so a 0.6 bps/day improvement is detectable.
   Today's four-fold economic screen cannot see the whole gap to Sharpe 2. Use
   a paired, IC-primary comparison over all fourteen folds and reserve the
   economics for finalists on the continuous book.
3. Harvest what is already in the saved forecasts: more seeds, a fixed
   C6/attention blend, and a causal volatility-targeting overlay. Each is
   cheap, each is estimated at a material fraction of the gap, and none needs a
   new fit. Code: `ops/evaluate_risk_overlay.py` for the overlay.

Only after those three should features and datasets be added, and they should be
added under the new protocol. Model widening, deeper trunks, further attention
variants, and GBDT retries should stop until step 1 lands: the evidence below
says their measured differences were not resolvable.

## The size of the gap, in the units the pipeline produces

The current best continuous result (`docs/v2_FOUNDATION.md`,
`docs/v2_R31_EXECUTION_SWEEP.md`) is about 6.7 bps/day above CDI at a zero-rate
Sharpe of 1.88. With CDI near 3 bps/day the absolute mean is about 9.75 bps/day,
which implies a daily standard deviation near 82 bps (13% annualised).

| Route to Sharpe 2.00 | Needed change |
|---|---:|
| Same volatility, higher mean | +0.6 bps/day (about +9% of the absolute mean) |
| Same mean, lower volatility | about -6% daily standard deviation |
| Two books with equal Sharpe and daily correlation .755, blended 50/50 | Sharpe x 1.067, which is the whole gap |

Those are small numbers relative to the noise of any single experiment:

| Statistic | Approximate standard error |
|---|---:|
| Mean excess of one continuous book over ~1,600 development days | 82 / sqrt(1600) = 2.0 bps/day |
| Paired difference of two books with daily correlation .90 | 0.9 bps/day |
| Paired difference of two books with daily correlation .95 | 0.65 bps/day |
| Four-fold net delta screens used in the capacity waves | +/- 2 to 3 bps/day intervals (`docs/v2_CAPACITY_RESULTS.md`) |
| Pooled daily IC mean over 1,598 days | .0046 |
| Mean IC over one 45-day selection window | .027 |
| Mean IC over the 160-day P-stage internal holdout | .0145 |

The consequence is blunt: a change worth exactly the gap to Sharpe 2 is a
0.3-sigma effect on one book and is invisible to the four-fold screens. Only
paired comparisons of near-identical books, or IC-primary tests pooled over all
folds, can resolve it. That is why the evaluation protocol is itself on the list.

## Diagnosis: the trajectory, not the architecture, dominates the differences

### Selection resolves nothing between neighbouring epochs

Round-7 selection scores each epoch by the mean common-population D3/D5/D10 IC
on a 55-session window whose D10 labels must stay inside the window, so at most
45 days decide a fit (`splits.py`, `round7_training.py`). With daily IC
dispersion near .18 that mean has a standard error near .027. The per-epoch
means in saved histories differ by about .005 between neighbours. The rule
therefore picks whichever epoch's noise draw was highest, and then stops after
five non-improving epochs, which at this noise level is close to a geometric
stopping time unrelated to fit quality. The audit tool quantifies this on your
histories without touching evaluation labels:

```
uv run --project research python ops/audit_selection_noise.py \
    --root <stage C refit root>/fits --stage F --output docs/v2_selection_noise_F.json
uv run --project research python ops/audit_selection_noise.py \
    --root <stage C refit root>/fits --stage P --output docs/v2_selection_noise_P.json
```

Expect `median_spread_over_se` near or below one and several epochs within one
paired standard error of the maximum. If that is what the audit shows, every
architecture comparison to date was a comparison of noise draws.

### The evidence already in the repository points the same way

- The one-factor patience diagnostic (`docs/v2_ATTENTION_REVERSAL_DIAGNOSTICS.md`):
  the corrected TE_wide P seed 29 parent stopped after nine epochs and selected
  epoch 4 under patience 5; under patience 20 it ran 51 epochs and selected
  epoch 31. Its F10 child improved by +7.05 bps/day above CDI in that fold, and
  the three-seed ensemble with only that seed replaced by +2.36 bps/day. One
  stopping rule moved one fold by ten times the gap to Sharpe 2.
- F-stage fits select at epochs 3 to 6 of a 60-epoch cosine schedule, so the
  selected child is a near-peak-learning-rate iterate. Fit-probe IC keeps
  rising (.046 to .082) while the selection mean falls (.042 to .023), so the
  raw rule is choosing the most transient point of the trajectory.
- Seed-pair forecast correlation is .777 for attention against .914 for C6
  (`docs/v2_FOUNDATION.md`): the attention trajectory carries roughly twice the
  seed noise of the GRU trajectory, which is exactly the family whose results
  reversed between data versions. Rank-averaging three seeds already added
  .474 bps/day; that gain is the trajectory noise being averaged out, not new
  signal.
- Width and depth contrasts landed inside +/- 2 to 3 bps/day intervals on four
  folds (`docs/v2_CAPACITY_RESULTS.md`, `docs/v2_SCALING_INVESTIGATION.md`), so
  "scaling does not help" is unproven either way. The GBDT result is not
  evidence that fewer parameters win: GBDT has no sequence encoder and no
  cross-sectional context, and its IC of .017 against .027 for S0 is a
  representation gap, not a capacity gap.
- The worst fold (F6, 2020 H2) is a universal loss with forecasts about .7
  correlated with momentum. That is a risk-control problem, which no
  architecture change addresses and which the overlay work below does.

## Ranked plan

### R1. Trajectory recipe (run first; code ready)

One-factor change against the Stage C fits: identical cells, store, optimizer,
loss and selector population. Only the stopping and selection rule changes.

- P stage: patience 5 to 20; trailing-3 smoothed selection. The smoothed rule
  uses the trailing three-epoch mean of the selection IC to judge improvement
  and patience, and selects the centre epoch of the best window. Its standard
  error is about .027 / sqrt(3) = .016 on the F window and .008 on the P
  window, which starts to resolve neighbouring epochs.
- F stage: replace the 60-epoch cosine schedule with a fully annealed 8-epoch
  schedule (`schedule_epochs = epochs = 8`), no early stop, same smoothed
  selection. The selected child becomes a low-learning-rate iterate near the
  centre of the best window instead of a near-peak-LR draw. The transferred
  learning-rate multiplier of 0.3 stays.
- Fallback arm: `--keep-f-schedule` keeps the F schedule and patience and adds
  only the smoothing, in case the short annealed schedule under-adapts.

Implementation notes. `TrainingRecipe.selection_smoothing` defaults to 1, which
reproduces the historical rule bit for bit and leaves every frozen contract hash
unchanged (`recipe_contract` drops the field at its default). When it is above
one the trainer keeps the last `selection_smoothing` epoch states in memory,
writes `selected.pt` from the centre epoch with its optimizer state, records
`selection_score` and `selected_epoch` in `history.json`, and resumes
identically after interruption (`tests/test_v2_smoothed_selection.py`).

Run:

```
uv run --project research python ops/run_trajectory_recipe.py --freeze \
    --cell TE_full --cell C6 --seed 11 --seed 29 --seed 47 --all-folds
uv run --project research python ops/run_trajectory_recipe.py
```

Budget on the RTX 2060: 2 cells x 3 seeds x (1 P + 14 F) = 90 fits. Take the
per-fit seconds from the Stage C `refits.json` to budget; the P stages will run
longer than before because patience 20 lets them continue, and the F stages
will run at most eight epochs. Plan for one overnight run per cell pair.

Gate (registered in the plan file the freeze step writes): paired per-fold,
per-seed common-population evaluation IC delta against the Stage C control,
Newey-West lag 10 on the pooled daily differences over all executed folds; adopt
only if the lower 95% bound exceeds zero and at least two thirds of seeds and
of folds are positive. Economics are reported for the retained recipe only and
never used to choose it. No per-fold or per-seed recipe choice.

Expected effect. The patience diagnostic and the seed-correlation numbers say
the trajectory noise is worth several bps/day in the worst folds and a few
tenths of a bps/day on average. Even a quarter of the diagnostic's fold effect
averaged over folds would close the gap on its own. The smaller, more certain
benefit is that later comparisons stop reversing.

### R2. Evaluation protocol (run with R1; no code needed)

- Primary metric: paired daily IC differences, all fourteen folds, Newey-West
  lag 10, reported with the 95% interval. Secondary: the same for the R$10m
  neutral and flexible-net books, paired, on the continuous path only.
- Never screen on four folds again for effects of this size. Four folds cannot
  distinguish +0.6 bps/day from zero, and any adoption decision made on them is
  a selection on noise.
- Report the selection audit (above) for every wave, so that changes to the
  trajectory can be seen in the histories, not inferred from outcomes.
- Keep the one-factor discipline of the existing preregistrations: one
  contrast per wave, matched controls, no best-seed or best-fold choice.

### R3. Ensembling of saved forecasts (no new fit for the first step)

- Fixed equal-weight rank blend of the C6 and attention forecasts on the
  continuous flexible-net book. Book excess-return correlation is about .755
  (`docs/v2_R31_EXECUTION_SWEEP.md`), so two books of equal Sharpe blend to
  Sharpe x 1.067. The earlier .5 blend measured +.70 bps/day with an interval
  crossing zero (`docs/v2_FOUNDATION.md`); the paired protocol in R2 is what
  can tighten that interval.
- Six seeds instead of three for attention once R1 is in. With seed
  correlation .777, the non-shared forecast variance falls from .074 (three
  seeds) to .037 (six seeds), a quarter of the reduction that three seeds
  already bought. If the +.474 bps/day scales with noise variance removed,
  expect about +.1 bps/day. Cheap, and a rank average has no free parameter.

### R4. Causal volatility-targeting overlay (code ready; saved books only)

The F6 loss and the momentum crash are the largest single drag on the Sharpe,
and every model shares them. A scale in (0, 1] applied to the book's exposure
from information strictly before the day is the standard remedy. The evaluator
tests own-return volatility, an external market-stress series (BOVA11 daily
returns), and the minimum of both, with the rescale turnover charged at 4 bps:

```
uv run --project research python ops/evaluate_risk_overlay.py \
    --book <run>/continuous/C6_flexible_net/book.json \
    --book <run>/continuous/TE_full_flexible_net/book.json \
    --market-csv <path to BOVA11 daily returns csv> \
    --window 20 60 --target 0.08 0.10 0.12 --floor 0.25 \
    --output docs/v2_risk_overlay_readout.json
```

Report every rule; do not pick the best. The paired circular block bootstrap
(block 40) gives Sharpe-delta intervals against the unscaled book. A rule that
raises the zero-rate Sharpe with a lower bound above zero across both books is
then confirmed by a ledger rerun with a daily gross cap equal to the scale
times the planned cap. The needed effect is about -6% volatility at unchanged
mean, or any mix of lower volatility and preserved mean; volatility targeting
on equity long-short books with clustered volatility usually delivers a few
percent of Sharpe, so this is a plausible route to a third of the gap and a
large improvement in drawdown.

### R5. Momentum-exposure cap in the allocator (design only)

The allocator already accepts group net caps (`sector_net_cap` in
`AllocationConfig`, used by `opportunity_research.py`). A momentum-decile
group row with a net cap is the same machinery. It is second to R4 because it
changes the optimisation, so it needs a full ledger replay per setting, while
R4 is a post-hoc scale on saved books. Run it only if R4 shows that
market-stress scaling helps but leaves the F6-type losses.

### R6. Regularisation for the attention family (only under R1 and R2)

Attention's seed noise (.777) is the symptom of a family that fits the P stage
faster than the selection window can measure. Candidates in order: input-family
dropout during P (drop whole feature families per batch), a stronger SAM radius
for the temporal path, and a slower P learning rate. Each is a one-factor
contrast under the R1 recipe and R2 protocol. None is worth running under the
current selection rule.

### Then, and only then: features and data

Features and datasets make the model heavier and the fits slower, and under the
current protocol their effect would be measured with the same broken ruler.
After R1 and R2 the paired IC test over fourteen folds is the right instrument
for them, and the audit shows whether a new family changes the trajectory.

## What not to do now

- More width, depth, or attention-variant waves. The last ones were inside
  noise and the diagnostic explains why.
- Best-seed, best-fold, or best-epoch choices of any kind. Every such choice at
  the current noise level is a selection on noise and will reverse.
- New datasets before the trajectory fix. Their measured effect will reverse
  for the same reason wider attention did.
- GBDT retries as a capacity argument. Its gap is representational.

## Runbook

1. Audit the existing histories with `ops/audit_selection_noise.py` (P and F).
   Record the pooled numbers in the experiment log.
2. Freeze and execute `ops/run_trajectory_recipe.py` for TE_full and C6. Score
   the F fits with the existing replay tooling paired against the Stage C
   controls, IC first.
3. Run `ops/evaluate_risk_overlay.py` on the saved continuous books. Confirm any
   surviving rule with a ledger rerun.
4. Evaluate the fixed C6/attention rank blend on the continuous book under the
   paired protocol.
5. Add three attention seeds under the retained recipe; rank-average.
6. Only then start the feature and data waves, one family per wave.

## Files added in this change

| Path | Purpose |
|---|---|
| `research/src/brazil_rv/v2/round7_training.py` | `selection_smoothing` recipe field, centre-epoch selection, resume support, contract-preserving default |
| `research/src/brazil_rv/v2/selection_rules.py` | trailing smoothing, Newey-West mean SE, selection-noise summaries |
| `research/src/brazil_rv/v2/risk_overlay.py` | causal trailing volatility, scale rules, rescale-cost charging, paired circular block bootstrap |
| `ops/audit_selection_noise.py` | audit saved histories; no evaluation labels |
| `ops/run_trajectory_recipe.py` | freeze and execute the R1 wave against the Stage C plan |
| `ops/evaluate_risk_overlay.py` | evaluate overlay rules on saved books |
| `research/tests/test_v2_smoothed_selection.py`, `test_v2_selection_rules.py`, `test_v2_risk_overlay.py` | coverage for the above |
