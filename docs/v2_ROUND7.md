# Round 7 and temporal pathway extension: combined research review

Status: both research programs complete; artifacts recovered and verified.

## Decision and interpretation

Both registered programs retain **A0**, the repaired-store S0 comparator. Original
Round 7 found no eligible confirmed improvement. The separately frozen pathway
extension found no qualifying stable extension gain. There was no access to the
2025/2026 consumer period, forward capture, or deployment.

This is evidence about the tested architectures, input representations, optimizer,
budgets and historical panels. It is not proof that richer data lack alpha, that
attention is intrinsically unsuitable, or that the old architecture is universally
optimal. In particular, the pathway engineering diagnostics exposed optimizer
sensitivity, which limits a capacity-only interpretation of the financial result.

## What was compared

Original Round 7 separated recipe changes, input changes and architecture changes.
A0 used the inherited S0 patience recipe, five heads and SAM radius 0.125. A1 used
the new fixed-budget recipe on S0 at the same radius; A2 reduced the radius to 0.05;
A3 added all families through the old linear sidecars. B1–B3 used the new
characteristic architecture with slow inputs at radii 0.125, 0.10 and 0.05. B4 added
all families at 0.05; B5 used 0.125. B6 removed the temporal branch, B7 restored five
equal heads, B8 used Pearson on rank targets, B9 used stock attention, B10 used a
TabM trunk with eight members, and B11 used single-pass AdamW as the diagnostic
optimizer comparison.

The new characteristic architecture combines a 60-session width-64 GRU, a latest
core-characteristic MLP, nonlinear family encoders, a width-256 joint state, FiLM
market conditioning, stock context and three residual SwiGLU trunk blocks. Its
usual heads predict D3/D5/D10. FiLM lets common market information change the
importance of stock-specific characteristics; it is not explicit attention over
oil/rate/security tokens. Calling this candidate "canonical" in the registration
expresses the design motivation, not an empirical guarantee or a literature consensus.

The extension kept the surrounding B4 architecture and compared:

| Cell | Temporal encoder | Peer interaction placement |
| --- | --- | --- |
| GL | GRU | Pool history, then mix stocks |
| GE | GRU | Mix stocks at each dated state, then pool history |
| TL | Temporal attention | Pool history, then mix stocks |
| TE | Temporal attention | Mix stocks at each dated state, then pool history |

All four use learned pooling. Temporal attention has four heads, positional
information and a width-128 feed-forward layer. Its bidirectional attention stays
inside the already historical decision window. Peer attention is masked, uses four
heads of width 16, and preserves the dated states for early interaction. GL/GE have
1,492,643 parameters; TL/TE have 1,501,283; B4 has 1,471,651.

GE versus GL is the cleanest tested placement contrast within the GRU family.
Comparisons with B4 also change the temporal pooling/peer construction. TL and TE
were screened but not confirmed; a full fourteen-fold temporal-encoder factorial
cannot be inferred from the confirmed GRU pair.

## Compute and selection contract

The repaired store contains 3,717 dates through 2024, 933 historical identities and
at most 243 active stocks. Compact date batches preserve eligible names and the
complete 60-session history. BF16 autocast retains FP32 loss/moments, and each epoch
uses unique dates. No security crop, lookback shortening or model-width reduction
was used to accelerate this comparison.

Twelve B4 calibration fits ran to 60 epochs. The registered selection-curve rule
selected **B=20** before screening. This was not a manually shortened runtime
budget. The recipe uses warm-up/cosine scheduling over the actual budget and uniform
averaging over its final quarter. Calibration checkpoints were not reused as B20
financial scores. The extension inherited this same budget and used fresh compatible
60-epoch pretraining. A limitation is that a common B4-calibrated budget need not be
individually optimal for every architecture; these are matched protocol comparisons.

Original screening used F2/F6/F10/F14 and seeds 11/29/47. Only A1 advanced; its
screen gain over A0 was 0.009520 primary IC, while the next cell, B1, was outside the
registered 0.002 advancement window. A1's advantage did not persist across the full
fourteen-fold confirmation. A0 was the confirmed leader. Three extra A0 seeds
(61/79/97) completed the six-seed panel; because candidate and comparator were both
A0, duplicate training was avoided. A zero self-comparison in the omission audit
must not be interpreted as evidence that A0 itself is unstable.

The extension screen advanced the GL/GE pair. Relative to B4, screen primary-IC
deltas were GL −0.003696, GE +0.004166, TL +0.000354 and TE −0.006101. Ninety new
confirmation fits completed GL, GE and the missing B4 control folds. There were no
duplicated original B4 confirmation fits, because B4 did not advance there.

GE was the best extension candidate but was ineligible against A0. Its confirmed
primary-IC difference was **−0.018301**, with a paired 20-date-block 95% interval
**[−0.031371, −0.005423]**. Its net difference was **−4.9598 bps/day**, interval
**[−9.6292, +0.5760]**. Leave-one-seed-out gains were not all positive. No extension
extra-seed confirmation was triggered. Improving on B4 did not mean improving on A0.

## Engineering corrections and their limits

The original compiler issue was resolved with dynamic compilation, float
specialization and max-autotune without CUDA graphs. The accepted engineering
suite passed before financial dispatch. B10's three pretraining fits subsequently
produced two training graphs because of a compiler tensor-size guard separating
full and remainder batches. An explicit amendment accepted exactly those three
bounded specializations; the failed launcher and original plan were retained.
All B10 screen fits met the original one-graph criterion. This was an engineering
criterion amendment, not a claim that the first gate passed or an independent
numerical-equivalence experiment. See [the amendment](v2_round7_b10_graph_amendment.md).

The pathway synthetic peer task exposed a real qualification. Xavier initialization
improved component-level learning, but full-model SAM runs remained near the
own-stock shortcut, while full-model AdamW GL/GE learned the synthetic relation.
The unseen-peer diagnostic was explicitly changed to mandatory supporting evidence
with no veto before financial fitting. Financial SAM remained 0.05. A failed
synthetic diagnostic was not relabeled successful. See
[the engineering evidence](v2_round7_pathway_engineering.md).

All four pathway GPU engineering checks passed, including masking/permutation and
precision checks. Gradient cosine similarity was at least 0.999715 and peak memory
was at most 2.528 GB in the engineering cases. Compile-inclusive concurrent timings
are not isolated architecture throughput benchmarks.

A readout stop identified a false-positive execution diagnostic: printed pending
entries blocked by the existing corporate-action permission mask were counted as
"unblocked." The correction changed the counter only. An exact GL/F14 replay changed
15 D1 diagnostic fields from one to zero; every other report field was identical.
Seventy-seven ledger tests passed. See [the correction](v2_round7_d1_correction.md).

A second readout correction separated experiment provenance from market-data
identity. Original and extension frozen-design hashes legitimately differ. Each
aggregate's exact experiment hash is now verified against its bound root and
reported separately before the unchanged strict data/label/execution comparison.
Stored reports remain intact. Three focused tests verify permitted distinct
experiment hashes, rejection of an incorrect experiment binding, and rejection of
different source data; the existing 43 evaluator tests also passed. Frozen training
checkouts were not changed; hash-bound external readout modules applied the fixes.

## Supporting CPU evidence

These diagnostics have no promotion weight and do not establish absence of neural
alpha. Across 42 repaired-anchor fits, mean clean fit IC was 0.046698 and selection
IC 0.038099. Evaluation mean seed IC was 0.026315 raw versus 0.027890 EMA; the paired
EMA difference was +0.001576 with Newey–West lag-10 SE 0.000751. All 42 head alignment
summaries were positive (mean 0.51587); the median SAM loss gap was 0.043708.

The chronological residual-corrector control covered 13 folds and 1,485 dates.
Composite IC was 0.031860 for S0, 0.016045 for the score-only tree and 0.020383 for
the all-family tree. The latter improved on the score-only tree but did not beat
S0. This is a specific corrector test, not evidence that the family information is
unusable. These composite values are not the official primary-IC endpoint.

The separate linear peer diagnostic covered F2–F14. Peer-plus-score minus score-only
composite IC was −0.004576, block interval [−0.010773, −0.001882]. The Newey–West
estimate gave SE 0.002429 and p=0.05958, so the uncertainty procedures disagree about
excluding zero. The feature map and linear estimator are narrow; this cannot veto
the neural architecture comparison or prove that cross-time peer effects are absent.

## Source-derived financial tables

All intervals below are paired 20-session-block 95% intervals. Net figures are bps/day under the frozen accounting conventions, including unresolved-economics labels; they are not assured realizable returns. Full metrics, seed values, momentum summaries and source hashes are in [the machine-readable results](v2_round7_results.json).

### original_screen

| Cell | Primary IC | Net bps/day | Unresolved folds |
| --- | ---: | ---: | --- |
| A0 | 0.018725 | 1.1423 |  |
| A1 | 0.028245 | 0.6678 |  |
| A2 | 0.023410 | 0.8063 |  |
| A3 | 0.020522 | 0.4780 |  |
| B1 | 0.023961 | 3.7154 |  |
| B10 | 0.007600 | -2.5196 |  |
| B11 | 0.004518 | -0.8754 |  |
| B2 | 0.021587 | 1.7139 |  |
| B3 | 0.015545 | -2.4643 |  |
| B4 | 0.008553 | -1.0582 |  |
| B5 | 0.005324 | -4.7988 |  |
| B6 | -0.001686 | -2.5502 |  |
| B7 | 0.004120 | -1.2220 |  |
| B8 | 0.012227 | 0.9981 |  |
| B9 | 0.002651 | 0.1839 |  |

### original_confirmation

| Cell | Primary IC | Net bps/day | Unresolved folds |
| --- | ---: | ---: | --- |
| A0 | 0.027216 | 3.9439 | F8, F9 |
| A1 | 0.025793 | 4.4304 | F4, F7, F8, F12 |

### original_six_seed

| Cell | Primary IC | Net bps/day | Unresolved folds |
| --- | ---: | ---: | --- |
| A0 | 0.026209 | 4.7530 | F7, F8, F9, F12 |

### pathway_screen

| Cell | Primary IC | Net bps/day | Unresolved folds |
| --- | ---: | ---: | --- |
| B4 | 0.008553 | -1.0582 |  |
| B9 | 0.002651 | 0.1839 |  |
| GE | 0.012719 | -0.2751 |  |
| GL | 0.004858 | -2.8203 |  |
| TE | 0.002453 | -3.2448 |  |
| TL | 0.008907 | 1.2147 |  |

### pathway_confirmation

| Cell | Primary IC | Net bps/day | Unresolved folds |
| --- | ---: | ---: | --- |
| A0 | 0.027216 | 3.9439 | F8, F9 |
| B4 | 0.007525 | -0.4097 | F4 |
| GE | 0.008915 | -1.0159 | F4, F7, F9 |
| GL | 0.005027 | -0.7036 | F4, F7, F9 |

The three-seed A1−A0 confirmation IC difference is −0.001422 [−0.006627, +0.004445]. Its screen advantage therefore did not justify replacing A0. The six-seed A0 panel is a different ensemble from the three-seed comparator used in pathway confirmation; do not subtract those unmatched headline values.

## Final continuous book

The designated six-seed A0 continuous development replay completed with one ledger initialization, thirteen model changes and no engineering gate failures. It reports 4.163758 net bps/day, annualized net-excess Sharpe 0.757888 and mean turnover 0.276228 NAV/day. Continuous minus fold-reset net is −0.589271 bps/day. Economics remain explicitly unresolved under the inherited valuation contract; this is a descriptive accounting scenario, not a clean executable-return claim. It has zero selection weight and does not change either decision.

## Input repair and preprocessing

Protected arrays were byte-identical. Retrospective labels/accounting changed only under the accepted action and continued-print repairs; historical decision-time wealth inputs stayed protected. Action metadata corroborates candidate events, not necessarily exact legal terms. Earlier foreign-flow publication could not be established, so the receipt bound was retained. No observations were invented to improve coverage.

Native fundamentals remove the minimum cross-sectional rank-support rule because they use fit-window median/IQR scaling and clipping instead of daily ranks. Validity and receipt age remain explicit. This admits sparse valid observations without fitting preprocessing on future data. Other ranked families retain their registered transforms; this round does not claim that every such transform is optimal.

| Native field | Valid active name-days | Previously suppressed |
| --- | ---: | ---: |
| earnings_yield_ttm | 133703 | 17150 |
| book_to_market | 136459 | 19170 |
| gross_profitability | 276620 | 477 |
| liabilities_to_assets | 303976 | 221 |
| accruals_to_assets | 263036 | 182 |
| revenue_growth_yoy | 240758 | 702 |
| sue | 236809 | 84 |
| log_market_cap | 140339 | 19354 |
| earnings_negative_flag | 133703 | 17150 |

The unchanged-signal accounting replay moved mean net from 4.508188 to 4.290994 bps/day (−0.217194). This isolates an accounting repair, not a new model gain. Full source limitations, coverage bindings and all target-array differences are in [input acceptance](v2_round7_inputs.json).

| Fold | Primary target cells changed | Target-validity cells changed |
| --- | ---: | ---: |
| F1 | 963 | 0 |
| F2 | 3106 | 0 |
| F3 | 0 | 0 |
| F4 | 0 | 0 |
| F5 | 9478 | 0 |
| F6 | 0 | 0 |
| F7 | 483 | 0 |
| F8 | 2297 | 0 |
| F9 | 681 | 0 |
| F10 | 887 | 0 |
| F11 | 21136 | 84 |
| F12 | 1255 | 0 |
| F13 | 5306 | 26 |
| F14 | 5862 | 26 |

## Observed fit durations

These are per-process wall times at the actual shared-instance concurrency, including compilation where performed; they are not isolated hardware benchmarks. Medians are over the twelve screen fits per cell.

| Program | Cell | Median minutes per fit |
| --- | --- | ---: |
| Original | A1 | 9.03 |
| Original | A2 | 7.16 |
| Original | A3 | 8.45 |
| Original | B1 | 7.69 |
| Original | B10 | 8.58 |
| Original | B11 | 8.24 |
| Original | B2 | 7.87 |
| Original | B3 | 7.21 |
| Original | B4 | 8.89 |
| Original | B5 | 8.99 |
| Original | B6 | 6.71 |
| Original | B7 | 8.45 |
| Original | B8 | 8.16 |
| Original | B9 | 8.57 |
| Pathway | GE | 11.02 |
| Pathway | GL | 9.60 |
| Pathway | TE | 8.10 |
| Pathway | TL | 7.79 |

## Continuous-book cost and borrow scenarios

| Scenario | Net bps/day | Economics unresolved |
| --- | ---: | --- |
| cost_2_borrow_balance | 4.844191 | True |
| cost_2_borrow_strict | 4.966110 | True |
| cost_2_borrow_open | 5.050747 | True |
| borrow_balance | 4.163758 | True |
| borrow_strict | 4.301529 | True |
| borrow_open | 4.482560 | True |
| cost_7_borrow_balance | 3.855786 | True |
| cost_7_borrow_strict | 3.930529 | True |
| cost_7_borrow_open | 4.051877 | True |
| sensitivity_buffer_0 | 2.122359 | True |
| sensitivity_buffer_2k | 4.540887 | True |
| comparator_sterile_proceeds | 1.095716 | True |
| comparator_uniform_borrow | 4.457522 | True |


## Provenance and review boundaries

The original training freeze was `8e00a2aa2edc9536e3472f82dbcf0073155387c2`;
the separate pathway freeze was `210922773c60ed349f382d8f60726c8c3ee86575`.
Both run roots, checkpoints, scores, reports and failed diagnostic evidence were
recovered from persistent NFS. Archive checksums and every extracted file were
verified: 12,771 original files and 5,388 extension files. The subsequently completed
continuous book and external readout harnesses were separately matched by SHA-256;
harness copies also reside on persistent NFS. See [recovery provenance](v2_round7_recovery.json).
The compact results retain source-report hashes; full results reside in the recovered
run roots recorded there. Frozen fit provenance remains distinct from the final
merged reporting and readout corrections.

The strongest next research question is why the richer B4 representation loses to
A0 under this training protocol. The GL/GE result does not justify simply promoting
early stock mixing. Before another broad architecture search, use a tightly matched
optimization/representation study: verify learning curves and perturbation scale,
then test a small number of causal input/optimizer changes with chronological
confirmation. The synthetic SAM sensitivity makes optimization a credible hypothesis,
but the financial AdamW screen did not demonstrate a winning replacement. Neither
fact alone settles the cause. Budget-specific undertraining, weak incremental inputs,
regularization and unfavorable interactions remain competing explanations.

This round establishes a research comparator, not production readiness. The retained
unresolved valuation scenarios and repeated development-panel use must accompany any
future economic or statistical claim. Original and extension screen selection are
explicit; the full fourteen-fold panel overlaps the screen and is not an untouched
holdout. No inference here consumes the barred 2025/2026 period.
