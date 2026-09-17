# Direct portfolio-objective experiment

Authorized 2026-09-17. This extends the completed decision/opportunity programs;
it does not reopen their results. Development dates stop at 2024-12-30. No forward
capture, held-out consumer access, Lambda launch or deployment.

## A. Close the bounded portfolio questions

Use the sealed Phase-3 neutral three-seed forecasts and previous causal economic
mappings. For C6 and TE_all, decompose the mapping into full, rank-only and
intercept-only preferences, under net caps .05 and .45 (beta cap .05 throughout).
Full books already exist and are reused. Add one full-map intermediate cap .25.
Stress the two full-map books with 8 bps/side costs, zero short-proceeds interest,
and a 3% annual debit spread separately. These are assumption sensitivities, not
candidate hyperparameters. Retain all securities and settlement flags. Inspect
the largest settlement events and yearly/issuer concentration and realized beta.

The primary objective comparison keeps the existing neutral allocator: gross
2.25, net .05, beta .05, stock .05, hedge .60, risk aversion 5 and five-session
planning. Keeping this established comparator prevents the already-observed
flexible-net gain from being credited to a new training loss. Flexible net is a
secondary frozen-policy sensitivity only. Cash remains available and earns CDI.
Unresolved settlement assumptions qualify financial conclusions; do not delete
affected names or label sensitivity scenarios as verified realized returns.

## B. Engineering and frozen training contract

Use the accepted repaired store through docs/v2_data_inputs.json and verified
Phase-3 neutral F checkpoints as warm starts, matched by arm/fold/seed. This is a
continuation experiment, not another pretraining campaign. The inherited F
checkpoint used only that fold's earlier fit and selection periods. Freeze its
preprocessing and original 60-session history, complete eligible population and
architecture. This saves repeated learning and isolates whether direct economic
fine-tuning adds value to the existing predictor. All three objectives start from
exactly the same weights; include epoch zero in both selectors.

The economic preference is the causal fixed calibration of differentiable
cross-sectional score ranks plus a zero-initialized, unbounded daily-return head
on the shared hidden representation. The additive return is never ranked away.
The smooth-rank bridge must be measured against the legacy hard-rank account;
the matched rank-only continuation is the causal control for this new interface.
Economic gradients must reach the shared encoder, not only the added head.

Compute negative realized daily excess-to-CDI utility plus the same causal risk
penalty (2.5 times daily variance) through the existing differentiable allocator
and share/cash account. Include transaction costs, borrow, restricted proceeds,
payments and actual inventory. Do not sum overlapping five-day labels as P&L.
Use chronological truncated backpropagation blocks with inherited account state;
no liquidation or cash reset at internal block boundaries. SAM's two passes and
AMP overflow retries must start from identical cloned account state and RNG.
Financial outcomes cannot enter a model input or decision before their timestamp.

Before financial fits, verify finite-difference/behavioral gradients, actual
encoder gradient, independent-ledger agreement, chronological state, and causal
input boundaries. Profile full-sized representative local RTX 2060 batches.
Reuse FP16 AMP with FP32 master weights/losses and FP64 finance, fused optimizer,
compiled neural/loss kernels and compact exact history caches. Do not truncate
history or discard names for speed. Record engineering-chosen block size, bounded
continuation budget and relative loss scale before evaluating financial outcomes.

## C. Matched screen and conditional confirmation

Three objectives: ranking control; ranking plus utility (primary candidate);
utility-only continuation (diagnostic candidate). Keep original SAM recipes:
TE_all ASAM .2, C6 SAM .125. Use seeds 11/29/47 and screen folds F2/F6/F10/F14.
Compare IC and economic-utility selectors on each identical saved trajectory;
selectors do not receive separate fits. The economic selector uses only the
original purged selection period and exact accounting. Earlier ties win. Report
all cells, both selectors and the unchanged warm start, including failures.

The implementation supplement must freeze the final fit budget, smooth-rank
temperature, head units, loss scaling and engineering acceptance before launch.
Neither fitting nor hyperparameter choices may use the screen evaluation dates.
Screen admission requires positive mean paired net and utility against the
matched rank control under the same selector, positive mean utility in at least
three of four folds, and at least two of three seeds. Confirm only admitted
arm/objective pairs on the remaining ten folds, with their matched rank controls.
Primary confirmation is the economic selector. Promotion requires positive lower
95% paired 40-session block intervals for net and utility, and positive mean
utility for at least two seeds. Show 20/60-session sensitivity and financing/cost
scenarios. These reused development periods are not a new untouched holdout;
nominal intervals do not correct the full historical research search.

Report daily net above CDI, risk-adjusted utility, IC, three Sharpes (BRL minus
CDI; USD-converted returns minus EFFR cash proxy; BRL minus zero), winning/losing
days, maximum drawdown, exposures, turnover/costs, halves and settlement flags.
Preserve artifacts, source/checkpoint hashes and a combined LLM review. Revisit
this registration at each stage completion and record remaining work.

## Implementation supplement, before financial fitting

Use 32 chronological sessions per gradient block, encoded in GPU microbatches
of at most 16 dates. All eligible names and all 60 history sessions remain.
Continuation ceiling: 12 epochs, minimum six; after that stop when neither IC
nor utility has improved for five epochs. Improvements are .0001 IC / .01 daily
bps utility; earlier ties win. The ceiling is for continuation from an already
selected F model, not a replacement 12-epoch training-from-scratch budget.
Inherited parameters use peak LR 3e-5; the new cardinal head uses 1e-4; preserve
original SAM/ASAM and AdamW decay routing. Warmup/cosine use the fixed ceiling.

Smooth-rank temperature is .1 in same-date standardized-score units. The cardinal
head is zero initialized, with one output unit equal to one daily basis point.
Hybrid utility weight is the median ranking/utility shared-encoder gradient-norm
ratio across three pre-specified initial fit blocks (first, middle, last full
block). Direct utility uses that same scale. This is fit-only unit calibration,
not selection of a loss weight on financial evaluation outcomes.

Start economic selection from cash on the first original selection date. The
preceding purge contains endpoints used by the inherited F ranking labels and
cannot be used as supposedly unseen inventory burn-in. Evaluation may use only
the post-selection embargo as unreported burn-in. This adapts the old portfolio
selection convention to the neural warm start; it does not imply the old
frozen-OOS calibration used those purge outcomes. Never carry fitting inventory
into selection.

The admission tests exposed a wrong-sign multi-day derivative with the old OSQP
adjoint despite correct forward portfolios. Replace that backward implementation
with a reduced active-face derivative of the same convex problem; retain the
Clarabel forward solve. Test the real hedge variance, inventory, no-trade regions,
caps, finite differences and independent accounts before financial launch.
The frozen old books and rank/Huber neural fits remain unchanged. Previous learned
controller failures are qualified by this newly identified backward-path defect.

Screen admission uses the economic selector, fixed in advance. The IC selector
and selector-minus-IC contrast are reported for every objective but do not form
an additional route through the screen. Include paired comparisons against the
unmodified warm start as well as the rank continuation. Independent fold accounts
must not be presented as one uninterrupted compounded history.
