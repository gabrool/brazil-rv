# Conditional rich-attention timing comparison

Recorded 2026-09-14 after the main portfolio comparisons and before any new fit.
This is the conditional architecture experiment authorized in the portfolio
registration, not an expansion across unrelated models or hyperparameters.

The trigger is met: over fourteen folds, TE_all's net CDI excess is 2.6104
bps/day with the calibrated optimizer versus S0's -0.7926, and 2.1624 with the
learned policy seed mean versus S0's -1.8835. TE remains a credible candidate,
although C6 is stronger and no new controller passes its own-legacy advancement
rule. This does not establish that early stock mixing is preferable to late
mixing when the rich inputs and FiLM are held constant.

Run TL_all: exactly TE_all's rich families, FiLM, temporal attention, full
60-session history, active universe and capacity settings; change peer_timing
from early to late. Use the same three seeds 11/29/47. Train three compatible
parents using the original TE parent recipe (sam125), then twelve F fits on
F2/F6/F10/F14 using the matched ASAM .2 recipe. Keep the existing 60-epoch ceiling,
selection logic, BF16 and compilation. Do not initialize late models from early
weights or reuse slow-only parents. Reuse the sealed TE_all forecasts as control.

Compare equal-seed rank-ensemble forecasts using the original empty-start legacy
book and identical repaired economic arrays, with paired daily net utility and
CDI excess, all three forecast-head ICs and fold/seed sensitivity. This isolates
timing under the common rich representation and inherited training recipe; it
does not demonstrate the optimal separately tuned recipe or optimal portfolio
controller for either architecture. Use the existing 10,000 paired 20-session
block bootstrap. The four windows have already informed research decisions;
call this a development screen, never independent confirmation.

Continue TL_all to the remaining ten folds and the causal optimizer comparison
only if screen aggregate paired net excess and utility versus TE_all are both
positive, at least three of four folds have positive paired utility, and no
material accounting issue invalidates the comparison. Otherwise stop this
bounded timing experiment and report the negative or inconclusive evidence.
No new learned-policy fit is justified solely by a favorable timing screen.

The raw-return auxiliary objective is deferred: the current evidence also shows
policy fitting/selection weakness and different risk/turnover choices, so it
does not isolate the supervised objective as the dominant remaining bottleneck.
The matched timing control answers an already missing comparison first.
Joint encoder/policy training is not triggered: all three families' learned
seed-mean policies underperform their deterministic optimizer over fourteen
folds, and none passes the registered own-legacy screen/confirmation rule.

Keep main results immutable. No 2025/2026 read, forward capture, history
shortening or input exclusion. Keep this instance for this authorized follow-up;
recover artifacts and terminate it when the program is complete.
