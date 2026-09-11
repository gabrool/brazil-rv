# Round 6 compute and training-budget amendment

Authorized by the user on 2026-09-11 before new optimized outcomes. The user
explicitly requested active-name compaction, BF16, unique-date training, retention
of usable prior work, and examination of the epoch cap. The healthy GH200 remains
available during this bounded implementation and restart. No new data, future
capture, held-out access, portfolio changes or additional research arms are added.

## Population and arithmetic

Pack every PIT-active security on each decision date, including active names
without valid labels or portfolio positions. Retain each selected security's full
60-session historical input, irrespective of its eligibility on those earlier
dates. Keep original target construction and characteristic neutralization on
the immutable canonical population. Store an explicit per-date canonical ISIN
index and scatter outputs back to the full 933-name score axis. No name is capped
or truncated. The stage width is the maximum active count across its fit and
internal-selection dates, rounded up to 16. Scoring computes its own width; its
future counts do not determine training layout.

The accepted development store has at most 243 active names. A width of 192 would
drop eligible names on 704 dates. No valid canonical target cell is inactive.
Compaction tests must preserve histories, predictions, loss, gradients, and output
identity within FP32 rounding tolerance. Padding itself is not extra training data.

Use CUDA BF16 autocast, retaining FP32 model/optimizer state, rank loss and
cross-sectional moments. Compare FP32 dense, FP32 packed and BF16 packed compute
on identical fit-only data and initialization. BF16 is an arithmetic change,
not a claim of bit-identical optimization. A parent trained in FP32 may initialize
a BF16 fit when all structural and source/chronology identities match; record
both precisions in the transfer audit.

## Sampling and epoch budget

Without persistence regularization, shuffle all fit dates without replacement.
Balanced batches contain at most 16 dates; retain the remainder and visit each
fit date exactly once per epoch. Adjacent-date pairs remain only for the explicit
positive-persistence objective, which requires permanent-identity alignment.
Round 6 has zero persistence weight throughout.

For the 756-session decay arm, apply causal per-date exponential loss weights,
normalized over valid fit groups separately for each horizon. Uniform unique-date
batches divide by their date count; do not normalize weights inside a minibatch.
Fit-only endpoint eligibility is applied before computing normalization. This
preserves the intended weighted objective without replacement sampling duplicates.
Selection and evaluation remain unweighted.

Audit of 102 finished old F fits found 20 capped at epoch 20 and 15 selecting
epoch 20. Among capped runs, median final selection-IC gains over the best before
epoch 16 were 0.001400 (magnitudes), 0.001458 (oddlot), and 0.001814 (options).
These internal-selection curves support testing additional training; they do not
prove an out-of-sample gain or justify selecting any candidate arm.

The revised maximum is 60 unique-date epochs. About 40 unique-date epochs match
the old 20 paired epochs' maximum updates. Validation occurs every two epochs
(also at a shorter smoke's final epoch); Patience-3 counts these checks. Retain
the existing minimum improvement, SAM/AdamW settings, warmup/cosine formula and
raw selected checkpoint. The schedule spans the revised maximum update budget.
Early stopping therefore gets approximately the old update-scale patience, with
50% more maximum update capacity. Record per-epoch update counts and wall time.

## Validation and reuse

Before launching the full grid, run score-free compiled smoke checks and a
bounded training-quality bridge: magnitudes on F1, F12 and F14, seeds 11/29/47,
using the same sealed parent P checkpoints as the finished old magnitudes fits.
Use internal-selection curves only for this engineering comparison. Report mean
and individual selected-IC changes, epochs and optimizer updates, cap frequency,
and time. A mean loss exceeding 0.001 IC or any fold's three-seed mean loss
exceeding 0.003 requires investigation before full dispatch. Inspect gains after
40 unique epochs to assess the extra budget. These are engineering tolerances,
not promotion thresholds. Keep the bridge separate from the final experiment root.

Freeze a new clean implementation and run root after validation. Refit a matched
S0 and every candidate F panel under the same new recipe; the old sealed S0 F
scores are historical reference only. Existing immutable stores, source audits,
borrow sensitivities and compatible P initializations are retained. Completed
old F fits and economic readouts remain archived under their original design;
do not mix them into the revised comparison. Keep all original seed, fold,
Session-2 roster, attribution, economic, confirmation and shutdown requirements.

Recover and verify all completed old artifacts before retiring the stopped old
dispatch processes. Never resume their queued jobs. Terminate the exact paid
instance only after the intended work and required recovery are complete.
