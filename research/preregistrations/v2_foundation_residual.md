# Foundation residual probe: implementation specification

This specifies the already authorized bounded probe in section 4.2 of
`docs/v2_NEXT_RESEARCH_DECISION.md` and section 5 of `v2_foundation.md` before
its new outcomes. It includes the planned linear comparison as well as the tree.
Run only after the retained input roster is frozen. This is reused development
research, not a fresh independent test or a new dataset campaign.

## Information and controls

Use the accepted repaired store bound by the foundation design. Bind genuine
chronological C6 forecasts: original Phase-3 neutral F forecasts from 2018 onward,
with the verified C6 P-prelude forecasts for earlier OOS history. Verify the earlier
checkpoint/store/date/identity bindings; do not use fitted training predictions.
If that prelude cannot be verified, report the missing history and amend the
common comparison dates explicitly rather than backfilling in-sample scores.

Keep the original fold fit/selection/evaluation windows, purge/embargo and all
eligible names. Every label consumed by fitting must mature before selection;
every label consumed by selection must mature before evaluation. Fit transformations
on fit dates only. The model emits a forecast for every eligible name even if that
name lacks labels or additional fields. Missing labels limit loss observations,
not execution eligibility. Fit each horizon on its own valid-label population.

Predict corrections separately for D3/D5/D10. Convert the established neutral
target to cross-sectional midrank units [-1,1] on each date/head; subtract the C6
forecast in the same rank units. Never subtract a raw rank from a bps label.
Equalize the total fitting weight per date and normalize mean sample weight to one.
Use the corresponding C6 seed's genuinely OOS anchor for each of 11/29/47, then
form the normal three-seed rank ensemble. A deterministic tree seed of 29 is an
implementation setting, not an extra independent replication.

For each learner compare three things on identical dates/populations:

1. Unchanged C6 anchor.
2. Matched correction using only the three anchor horizon ranks.
3. Correction adding the frozen retained auxiliary roster, masks/known ages,
   common-state diagnostics and common-state-by-anchor-rank interactions.

Retain core fundamentals/magnitudes when in the frozen roster: the question is
incremental information beyond the forecast, not whether the neural model has
ever seen a field. No new hand-crafted price-path indicators or raw datasets.
Use fit-only median/IQR/asinh for continuous values, existing passthrough semantics
for bounded/rank fields, explicit value-valid indicators and bounded known ages.
Common scaler samples occur once per date, not once per security. Ridge and trees
receive identical columns; missing encoded values are zero with the same explicit
mask/age channels. Do not add a minimum-support stock or feature filter.

## Small fixed learners and selection

- Ridge: average weighted squared residual loss, unpenalized intercept and fixed
  coefficient penalty .1. Solve in float64. No ridge-penalty grid.
- Histogram tree: LightGBM regression, max depth 3, 7 leaves, learning rate .03,
  minimum 200 observations per leaf, L2 penalty 10, full feature/sample fractions,
  255 histogram bins, deterministic column-wise fitting with at most four threads.
  Maximum 500 rounds; early stopping after 30 rounds without improved equal-date
  selection IC of anchor plus full correction. Save the selected iteration.
- For both, choose a single correction strength from 0/.1/.25/.5/1 by mean
  D3/D5/D10 IC on the preceding selection dates. Exact ties prefer the smaller
  strength. Zero uses unchanged anchor forecasts exactly and is a real failure
  to improve, not a newly successful predictor.

The score-only and rich cells get the same learner settings, windows and shrinkage
selection. Do not refit on selection labels afterward. No extra feature, depth,
penalty or seed sweep follows a negative result. Cache common point-in-time inputs
once; reuse labels and source-bound anchor panels without reopening raw sources.

## Reporting and advancement

Use the foundation's four screen folds and registered paired IC/economic gates.
Report each rich cell against both unchanged C6 and its learner's score-only cell.
Incremental rich-data value requires improvement over the score-only control as
well as admission against C6; recalibration alone is not rich-data evidence.
Rank outputs feed the same prior-only C6 calibration and neutral cash-permitting
allocator as the anchor. Keep the economic selector fixed; do not choose shrinkage
from evaluation P&L. Report three Sharpes, turnover, drawdown, win/loss fractions,
seed/fold dispersion, selected strengths/iterations and train/selection gaps.

Confirm at most one admitted rich corrector on the other ten development folds,
preferring ridge when both satisfy the same practical gates. Use the existing
confirmation rules; do not count deterministic repeated runs as extra seeds.
If neither qualifies, close this bounded current-state residual branch. Failure
does not establish that every temporal representation of the data is useless.
