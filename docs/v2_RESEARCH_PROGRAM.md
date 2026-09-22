# Research program: every high-value change, ranked

Written 2026-09-22 (UTC) by the cloud research session after reading the
2026-09-21 local handoff. This is the answer to "what will take the model to the
highest trustworthy Sharpe", not the shortest path to any particular number. It
ranks every change by expected value per GPU hour under the repository's own
discipline: one factor per wave, frozen before outcomes, paired evidence on the
eight fixed periods, nothing adopted on a point estimate.

## 0. Where things stand and what this session could and could not do

- The RTX 2060 is running the matched stopping experiment (attention widths 64/96,
  parent patience 5/20, seeds 11/29/47, eight periods), and the finite continuation
  will then verify forecasts and train the 54 C6/GRU common-model fits. **Nothing
  in this program starts a GPU fit while that queue is incomplete**; the new GPU
  drivers refuse to run until both queues report complete.
- This session ran in the cloud container without the data, the GPU or the local
  checkouts, so no experiment was executed here. What it delivers is tested code,
  a registration draft and this program, plus a one-click task card for the local
  machine. The local status commands are in `RESEARCH_HANDOFF.md` section 2.
- The evaluation boundary is inherited: folds F2/F3/F6/F7/F10/F11/F13/F14 only,
  the six reserves unopened, no 2025/2026 consumer, no spliced continuous account.
- The trainer is untouched. An earlier draft added a smoothed selector inside
  `round7_training.py`; that was withdrawn because the inference-provenance guard
  compares the trainer file byte for byte with the training commit, so any edit
  would have blocked post-hoc scoring of every existing checkpoint from main. All
  selection changes are therefore *post-hoc views* over executed trajectories,
  the same convention the running experiment uses for patience 5 versus 20.

## 1. What limits performance now

The numbers below are from the repository's own documents; the sources are named
in `docs/v2_SHARPE_TWO_PLAN.md`.

1. **Checkpoint selection resolves nothing between neighbouring epochs.** Selection
   uses at most 45 defined days; the mean IC's standard error is about .027 while
   neighbouring epochs differ by about .005. The patience-5 rule then stops on what
   is close to a random clock. The one-factor patience probe moved one fold by
   +7.05 bps/day for one seed and +2.36 for the ensemble. F children select at
   epochs three to six of a 60-epoch cosine schedule, at near-peak learning rate.
2. **Seed noise is large for attention.** Seed-pair forecast correlation .777 for
   attention against .914 for the GRU; the three-seed rank average is worth
   +.47 bps/day. Ensembling is the cheapest variance reduction available and is
   under-used: only seeds are averaged, never views, widths or families.
3. **The evaluation cannot see the effects being sought.** A single continuous
   book's mean has a standard error near 2 bps/day; the four-fold screens had
   intervals of +/- 2 to 3 bps/day. Only paired comparisons of near-identical books
   on common dates resolve a few tenths of a bps/day, which is why every driver
   here evaluates paired against a reference arm on the same dates and account.
4. **The largest drawdown is a shared factor event.** F6 (2020 H2) is a universal
   loss with forecasts about .7 correlated with momentum. No architecture change
   addresses it; a risk overlay or exposure control does.
5. **The wide-net policy carries most of the historical return.** 6.7 bps/day at
   45 percent net against about 3 bps/day at 5 percent net on the newer eight-period
   accounts. Whatever wins must be confirmed on the flexible policy, because the
   long tilt is part of the product.

## 2. The ranked program

Each item: what, why, expected effect, cost, gate, tool. "Paired" always means
fold-by-fold daily net-excess deltas against the reference arm's existing books on
the same dates and account, circular block 40, 95 percent interval.

### Tier 1: saved artifacts only, CPU, can run now

**1.1 Post-hoc selection views on every completed child** (`centre3`, `around3`,
`top3`). Why: item 1; the sealed epoch files already exist, so alternative
selections cost one CPU inference per epoch. Expected: +0.1 to +0.5 bps/day on
average with large fold dispersion; the `around3` and `top3` averages also reduce
seed noise. Cost: about 100 to 300 CPU scorings, minutes each. Gate: registration
A. Tool: `ops/replay_forecast_variants.py --freeze --policy neutral`, then
execute, then `--summarize`.

**1.2 Rank-average ensembles of existing forecasts**: stopping views (`p5+p20`),
widths (`64+96`), families (`C6+attention`, `C6+GRU+attention`). Why: item 2;
book excess-return correlation across families is about .755, so two books of
equal Sharpe blend to Sharpe x 1.067 and the earlier .5 blend measured
+.70 bps/day. Expected: +0.3 to +0.8 bps/day for the family ensemble, smaller
for views and widths. Cost: books only. Gate: registration B. Tool: same driver,
the family variants appear automatically once the common-model fits exist.

**1.3 Causal volatility-targeting overlay** on the saved period and continuous
books (own-return and BOVA11 rules, 25 percent floor, rescale cost charged).
Why: item 4; a shared-factor drawdown is exactly what a causal vol target trims.
Expected: -5 to -15 percent volatility at a smaller mean loss, a drawdown
reduction, plausibly the whole gap between 1.88 and 2.0 on its own. Cost:
seconds. Gate: registration E, then a ledger rerun with a daily gross cap. Tool:
`ops/evaluate_risk_overlay.py`.

**1.4 Selection-noise audit** of every history (P and F). Why: it turns item 1
from an argument into a measurement on the current fits and shows whether any
view can be resolved at all. Tool: `ops/audit_selection_noise.py --root <root>
--stage F`.

### Tier 2: GPU children only, parents reused (after the current queue)

**2.1 Smoothed parent views** `s3` (and conditionally `s3p5`) on the six executed
patience-20 parents, 48 children each with exact-reuse aliases. Why: the parent
epoch decides the child's starting point; the probe showed the child is highly
sensitive to it, and the executed trajectories already contain every epoch.
Expected: the largest single trajectory effect available, but the sign is not
known; it is the direct test of whether item 1 is a parent problem. Cost: up to
48 child fits, roughly four to six GPU hours. Gate: registration C. Tool:
`ops/extend_matched_stopping_views.py --freeze`, then execute when the GPU is free.

**2.2 Short fully annealed F schedule** (`schedule_epochs = epochs = 8`, patience
8) from the patience-20 parents, one or two cells. Why: children currently stop at
epochs three to six of a 60-epoch cosine, so the selected weights are a
near-peak-LR iterate; annealing gives the selector low-LR iterates to compare and
makes the post-hoc averages (1.1) meaningful. Expected: a reduction of seed
disagreement first, mean effect uncertain. Cost: 24 child fits per cell, each
short. Gate: registration D. Tool: `ops/run_trajectory_recipe.py --freeze --cell
TE_full`.

**2.3 Six seeds for the retained attention recipe.** Why: item 2; with seed
correlation .777 the non-shared variance falls from .074 (three seeds) to .037
(six). Expected: about +.1 bps/day, nearly free at inference. Cost: 3 more parents
and 24 more children per cell. Gate: must be registered as extra seeds (the
handoff forbids automatic extra seeds); the tooling is the existing drivers with
seeds 61/79/97.

### Tier 3: model-side one-factor contrasts, two cells per wave, only under the
paired protocol and after Tier 1 and 2 have settled the recipe

**3.1 LSTM temporal encoder** (already authorized). Expected: within noise of the
GRU unless the recipe change reveals a difference; run it as a matched cell.

**3.2 Input-family dropout for attention during P** (drop whole feature families
per batch). Why: attention fits the P stage faster than the selection window can
measure (fit-probe IC rising while selection falls), and its seed noise is twice
the GRU's; family dropout is the regulariser that targets exactly that. Expected:
lower seed noise, small mean gain. Cost: one cell, full P and F.

**3.3 Distillation of the winning multi-family ensemble into one student.** Why:
if 1.2 wins, a student trained on the ensemble's rank targets keeps most of the
gain at a third of the inference cost and can be ensembled again over seeds.
Expected: retains 60 to 90 percent of the ensemble gain. Cost: a new target view
(the teacher's ranks) plus one cell.

### Tier 4: target and loss

**4.1 Horizon-consistent target denoising.** The three heads learn the D3, D5 and
D10 characteristic-neutral midranks independently; the D3 rank is the noisiest
label and the three overlap. A registered rev-5 target view that averages the
neutral residual ranks over the three horizons with head-specific weights (for
example D5 = .25 D3 + .5 D5 + .25 D10) lowers label noise without changing the
allocator's horizon structure. Why: label noise is the dominant noise source in a
.03 IC problem; averaging overlapping labels is the cheapest denoiser. Expected:
+.001 to +.003 IC. Cost: a store target view (virtual, no rebuild) and full fits
for one cell. Gate: paired IC first, then books.

**4.2 Dispersion-weighted soft-Spearman.** Weight each date's loss by its
cross-sectional dispersion of the target's continuous precursor, because days with
wide spreads carry most of the P&L. Expected: small mean gain, better economics
per IC. Cost: a loss option and one cell.

**4.3 Tail-emphasis weighting** of the rank loss (more weight on the top and bottom
deciles the book actually trades). Expected: small; run only after 4.1 and 4.2.

### Tier 5: portfolio and risk, independent of the model

**5.1 Exact ledger version of the vol target** (daily gross cap = scale times the
planned cap) once 1.3 shows a survivor.

**5.2 Momentum-exposure control.** The allocator already accepts dated group caps
(`sector_net_cap` machinery); a momentum-decile group row is the same mechanism.
The sector caps did not raise net returns, and a momentum cap will trade some
alpha (forecasts are .7 momentum-correlated), so this is a drawdown control to
evaluate, not an expected Sharpe gain. A proper K-factor covariance in the QP is
the fuller version: the forward solve extends the diagonal-plus-market form with
K exposure coordinates and the adjoint's Sherman-Morrison inverse becomes a
Woodbury inverse; it needs causal factor exposures added to the policy cache.
Design only until 5.1 and 1.3 are read.

**5.3 Temporal smoothing of forecasts** (rank-EMA across days) as a post-hoc
variant rule. Why: reduces noise-driven turnover under 4 bps plus B3 charges.
Expected: +.1 to +.3 bps/day net if turnover is the binding cost. Cost: a rule in
the variant driver and books.

### Tier 6: features and data, last

New families and datasets make fits slower and the model heavier, and under the
current selection rule their effect would be measured with the same broken ruler.
After Tiers 1 and 2 the paired eight-fold protocol is the right instrument for
them: one family per wave, paired IC primary, books for finalists.

## 3. Evaluation protocol for every wave

- Paired daily deltas against the reference arm's existing books, same dates,
  account, calibration and allocation; circular block 40 primary with 20/60
  reported; seeds and ensemble separately; the eight folds pooled and per fold.
- IC is diagnostic. Selection-window statistics never decide adoption.
- Report every variant and every failed or skipped attempt; retain only on a
  lower bound above zero with a majority of folds and seeds positive.
- Confirm on the flexible policy after a neutral-policy win, never the reverse.
- No per-fold, per-seed or best-epoch choice of any kind.

## 4. Budget on the RTX 2060

| Item | Fits | Rough GPU time |
|---|---:|---:|
| 1.1 to 1.4 | none | CPU only, hours |
| 2.1 smoothed views, one view | up to 48 children minus aliases | 4 to 6 h |
| 2.2 short F schedule, one cell | 24 children | 1 to 2 h |
| 2.3 six seeds, one cell | 3 parents plus 24 children | 4 to 6 h |
| 3.x one cell | 3 parents plus 24 children | 6 to 10 h |

Per-fit seconds are recorded in every `refits.json`; budget from those rather
than from these rough figures.

## 5. Order of operations

1. Now, on CPU, in a checkout whose inference files match the training commit
   (main today): `ops/audit_selection_noise.py`, then registration A and B through
   `ops/replay_forecast_variants.py` on the neutral policy for the attention arms
   that have completed forecast groups; add the family ensembles when the C6/GRU
   fits land; `ops/evaluate_risk_overlay.py` on every saved book.
2. When both GPU queues report complete: registration C (`s3` children), then D
   (`TE_full_f8`), evaluated through the same variant driver.
3. Read the paired summaries together; retain at most one selection rule, one
   ensemble composition and one overlay; confirm on the flexible policy.
4. Only then Tier 3 and 4 waves, two cells each, and Tier 6 last.

## 6. What not to do

- No width, depth or attention-variant waves before the selection rule is settled.
- No extra seeds, folds or datasets without a registration line.
- No best-seed, best-fold or best-epoch choice; no learned ensemble weights.
- No GPU fit while the continuation owns the queue; no edits, pulls or
  reformatting under the pinned checkouts.
