# Decision research: combined Phases 1–3 review

Completed 17 September 2026. Intended as a self-contained handoff for LLM review.

## Decision and scope

The authorized program is complete. Coherent return calibration and simpler
equal-rank allocation materially improved some frozen-forecast books. Neither
the admitted learned controller nor the economic-auxiliary forecasting candidate
qualified for advancement. Keep the established comparator; do not promote a new
architecture, objective or controller on these results.

This does **not** establish that attention, economic objectives or learned
execution cannot work. Phase 3 attention remains useful in absolute development
economics. Its auxiliary objective produced a small positive estimate, with
insufficient consistency and uncertainty spanning zero. The experiment tested
one specific representation-training change under a fixed allocation rule.

The program used development observations ending in 2024. No 2025/2026 consumer
read, forward capture, new pretraining or Phase 4 experiment was performed. No
Lambda instance was launched. All 168 Phase 3 financial fits ran on the local
RTX 2060, separately from four disposable engineering smoke fits.

Authoritative contracts: [original plan, section 13](v2_PORTFOLIO_DECISION_POSTMORTEM.md),
[Phase 3 registration](../research/preregistrations/v2_decision_phase3.md),
[canonical run pointer](v2_decision_run.json). Full machine-readable verified
results: [Phase 3 evidence](v2_decision_phase3_results.json).

## Phase 1: repair the forecast-to-portfolio interface

The prior stock-only calibration included common returns that a joint stock/BOVA
portfolio could hedge away. This made expected opportunity inconsistent with
the implemented portfolio. We tested coherent benchmark-residual calibration,
intercept diagnostics, equal-rank calibration, block-cluster uncertainty and
exposure shrinkage alongside cash and deterministic controls. Original forecasts,
eligibility and input observations were retained.

The completed roster contains 315 books across seven cells, three forecast arms,
fourteen folds and continuous accounts. All 90 original raw/cash controls
reproduced within registered numerical tolerances. No additional calibration
labels or stock-days were removed. Opposing multi-horizon coefficients were
diagnosed rather than forcibly constrained positive.

| Continuous net above CDI, bps/day | S0 | Early attention TE_all | C6 |
| --- | ---: | ---: | ---: |
| Coherent three-regressor benchmark | −0.363 | 2.535 | 4.972 |
| Coherent equal-rank control | 4.112 | 4.555 | 4.897 |

Equal-rank substantially helps S0/attention, but does not dominate C6. It became
the **predeclared common Phase 3 readout**, not a retrospective choice between
Phase 3 outcomes. Detailed uncertainty, attribution, exposure and historical
settlement assumptions remain in [Phase 1](v2_DECISION_PHASE1.md).

## Phase 2: establish controller learnability, then test real forecasts

Two actual state-account defects were repaired: sub-threshold order intentions
incorrectly changed holding state, and crossing a tiny residual into the opposite
side retained the old position's age. The training and exact accounts now agree;
trained synthetic NAV disagreement fell from 1.186e−4 to 9.882e−12. Actual
holdings were not deleted. Solver accuracy was tightened for nearly flat cases,
with the financial constraints and agreement tolerances retained.

The initial synthetic null mistakenly included predictable adverse shocks. It
was corrected before financial dispatch. The final independent-date fixture had
1,920 sessions, with explicit fit/selection/evaluation gaps and randomized regimes.
The conditional controller demonstrated profitable trading, cost sensitivity,
different continuation/reversal responses and 28.6% lower gross in signaled null
regimes. This proves bounded implementation capability, not financial alpha or
perfect inactivity.

The stateful MLP was not admitted: seed 29 reduced null-regime gross only 16.8%,
below the fixed 25% requirement. Seed 11 selected the unchanged conditional base;
seed 47 passed. All attempts are preserved. The 24 planned MLP financial fits
were therefore **not run**, rather than described as a negative real-data result.

Eight unique deterministic conditional fits completed on F2/F6/F10/F14. Repeated
initialization seeds are aliases for this controller, not independent evidence.
Inputs included known market context, actual ensemble disagreement, holding state,
costs and causally matured shadow outcomes. Shadow observations could first use
origin t−6 at decision t, because the payoff was known at close(t−1).

| Conditional minus coherent benchmark | TE_all | C6 |
| --- | ---: | ---: |
| Net above CDI, bps/day | −3.086 | −0.067 |
| Nominal 95% paired interval, 40-session blocks | [−9.580, 2.517] | [−6.481, 5.724] |
| Utility difference, bps/day | −3.366 | −0.530 |

Neither controller passed. Past-only cash/reference fallback did not rescue the
screen. There was no triggered Phase 2 confirmation. All intervals span zero;
this is not proof of universal ML-policy inferiority. See [Phase 2](v2_DECISION_PHASE2.md)
for the 16 engineering attempts and full selection/account evidence.

## Phase 3: exact matched experiment

Two models were compared against their own fresh neutral controls:

- **TE_all:** full 60-session temporal attention, early stock interaction,
  accepted feature families, FiLM, ASAM rho .2; 1,489,540 parameters including
  the auxiliary head. Its three original pretrained parents load unchanged.
- **C6:** slow GRU, pooling and additive fundamentals/magnitudes projections,
  SAM rho .125; 127,943 parameters including the auxiliary head. Its parents
  explicitly copy shared tensors from the seed-matched S0 parent and initialize
  new family projections at zero. This is not all-family pretraining.

The one treatment is neutral ranking loss plus .25 times an equal-date Huber
loss (delta one) on a single zero-initialized head sharing the encoder. The
target is `(shareholder_H5 − CDI_H5 − decision_beta × BOVA_excess_H5) / 5`.
Fit-only equal-date RMS scaling preserves zero and magnitudes. Missing auxiliary
labels never exclude a stock from ranking training. All six horizon dates must
lie in the authorized fitting window. The exported head is in daily-return units.

Both objectives use the same parent, seed, full populations, unique-date batches
of up to 16, learning rate 1e−4, transferred multiplier .3, fixed 60-epoch schedule,
five-check patience and **neutral-IC checkpoint selection**. C6's recipe differs
from its historical recipe, so historical C6 fits are not matched controls.

The accepted repaired store has 3,717 dates, 933 historical identities, at most
243 active names, 568,815 active stock-days and 145 scalar fields. All histories
remain 60 sessions. Stage padding follows actual populations, up to 256; it does
not cap the universe. Store manifest SHA-256:
`61149d43a23fc55fcd92b43aaa4e74d747fbdf62f852bedbbbdb2cf18281f3cc`.

The fixed equal-rank QP uses shared historical out-of-fit calibration for each
matched pair, historical CDI cash and identical costs/borrow/risk assumptions.
The auxiliary head does not directly set orders; the treatment tests whether it
improves the shared representation and hence ranking under this fixed policy.

Screen: 48 fits, F2/F6/F10/F14 × two models × two objectives × seeds11/29/47.
Both passed the original screen, triggering 120 fits on the other ten folds.
All 168 completed. No extra seed search or real-result-driven tuning occurred.
Each seed and the actually executed equal-rank ensemble are reported.

The diagnostic C6/TE blend uses TE weights from {0,.25,.5,.75,1}, selected only
on earlier old-OOS selection utility. Exact ties prefer lower TE weight. The
same weight is applied to each new objective, without reranking the convex
combination. The blend cannot independently trigger architecture advancement.

## Phase 3 results

All differences below are economic-auxiliary minus fresh neutral. Intervals are
nominal paired 40-session circular block intervals; 20/60 sensitivities are saved.
Screen and confirmation are reported separately, not combined to rescue a gate.

| Panel/model | Net difference, bps/day | 95% interval | Utility difference | Positive utility folds | Decision |
| --- | ---: | --- | ---: | ---: | --- |
| Screen TE_all | +0.282 | [+0.092, +0.476] | +0.281 | 4/4 | Confirm |
| Screen C6 | +0.078 | [−0.019, +0.184] | +0.077 | 4/4 | Confirm |
| Confirmation TE_all | +0.086 | [−0.046, +0.220] | +0.084 | 6/10 | Fails consistency |
| Confirmation C6 | −0.056 | [−0.227, +0.117] | −0.058 | 6/10 | Fails multiple gates |
| Confirmation diagnostic blend | +0.106 | [−0.045, +0.259] | +0.106 | 8/10 | Diagnostic only |

Confirmation required at least eight positive utility folds, positive net and
utility, and no seed mean below −.25 bps/day. TE seed net differences were
−.182/−.144/+.187; C6 were +.030/−.344/+.014. Thus C6 also failed seed robustness.
The positive attention screen was concentrated in F10 (+1.090 bps/day); the other
three screen gains were each below .028. The larger panel shows why confirmation
was necessary. The blend's favorable point estimate remains statistically uncertain
and does not change its registered diagnostic status.

| Ensemble neutral IC | Neutral | Economic auxiliary | Difference |
| --- | ---: | ---: | ---: |
| Screen TE_all | .027141 | .027302 | +.000161 |
| Screen C6 | .024168 | .024264 | +.000096 |
| Confirmation TE_all | .025294 | .025253 | −.000041 |
| Confirmation C6 | .030586 | .031069 | +.000483 |

The primary 40-session paired IC intervals include zero. C6's small IC gain did
not translate into an economic gain. This reinforces the distinction between
rank quality and portfolio outcomes; it does not identify a new pipeline bug.

Cardinal-head calibration is unstable across confirmation folds: slopes range
from −15.90 to +3.70 for TE and −4.34 to +5.11 for C6. Negative slopes and large
magnitudes argue against using the uncalibrated head directly for sizing. A large
slope can also reflect small prediction variance; it is not by itself proof of
causality failure. Per-fold RMSE, means, slopes and fixed-rank tail outcomes are
preserved in the machine-readable evidence. No fitting decision was changed using
these outcomes.

## Continuous accounts and economic limitations

Both screen survivors received all fourteen new forecast blocks. Sixteen
continuous books (two models × two objectives × three seeds plus ensemble) carry
actual inventory through model changes, with one initial cash account and one
final liquidation. They contain 1,738 sessions. No fold-P&L splicing or substituted
old forecasts. These accounts are descriptive because confirmation failed.

| Continuous ensemble | Neutral net above CDI | Auxiliary net above CDI | Paired difference [95% interval] |
| --- | ---: | ---: | --- |
| TE_all | 4.497 | 4.581 | +.085 [−.077, +.260] |
| C6 | 4.000 | 3.940 | −.060 [−.220, +.094] |

TE neutral/auxiliary mean gross is 2.022/2.023; C6 1.925/1.922. Trading costs are
.533/.527 bps/day for TE and .486/.481 for C6; borrow costs .733/.737 and
.691/.691 respectively. The auxiliary is not a learned cash-timing overlay.
These largely invested portfolios must not be described as solving regime timing.

All four ensemble accounts retain `economics_unresolved=true`. The saved terminal
haircut scenario reduces final NAV relative to initial capital by approximately
37.38/39.72 percentage points for TE neutral/auxiliary and 37.12/36.16 for C6.
This is a scenario difference over the full account, **not daily bps or a 37%
daily loss**. Historical stale/missing-price settlement is materially consequential.
The reported daily reconciliation errors below 1e−15 establish ledger consistency,
not historical executability. Gross/net/beta can drift beyond new-order limits
after returns or when exits cannot execute; those days remain explicitly reported.
Full CDI remuneration of short collateral and fixed costs remain assumptions.

Historical Phase 1/2 fold books had a different burn-in boundary. New matched
Phase 3 fold books begin in cash on their first new evaluation date and liquidate
at the boundary. Do not interpret a cross-program bps difference as purely a
model improvement. Fresh paired Phase 3 comparisons are the relevant evidence.

## Local GPU engineering and actual runtime

Native Windows uses PyTorch 2.6/cu126 and Triton-Windows 3.2.0.post21, compatible
with Turing. Linux/GH200 retains its separate runtime. Both local treatment and
control use FP16 Tensor Core autocast, FP32 weights/moments/loss, and gradient
scaling on both SAM passes. Overflow retries preserve the batch and RNG rather
than skipping an update. No financial fit required an overflow retry.

Exact session/security history storage replaces overlapping expanded windows.
Gathered tensors match the canonical collator. F2 attention cache fell from
1.204GB to .203GB without shortening history, pruning names or quantizing inputs.
Compiled forward/backward/loss, fused AdamW, foreach SAM and a persistent worker
reduce overhead. Full validation remains every epoch. Native default Inductor
mode avoids exhaustive reduction autotuning; no silent eager fallback exists.

Representative compiled updates took .115/.176 seconds for TE F2/F14 and
.137/.179 for C6. TE F14 eager was about .31 seconds. Prediction RMS error against
FP32 was at most .000450 of prediction standard deviation. These are engineering
step timings, not end-to-end campaign speedups. The full batch 16 × 256 names × 60
history fits on the 6GiB card. Recorded process peak CUDA allocation reached
about 2.253 GiB; retained allocator peaks are not isolated per-fit measurements.

Twenty-three targeted tests covered exact caches/populations, scaled SAM/ASAM,
same-batch overflow/RNG recovery, resume, causality/scaling and matched readouts.
Four two-epoch smoke fits passed before finance. Initial smoke attempts exposed
a missing parent byte-count descriptor and an integer-zero branch unsupported by
Torch2.6 symbolic tracing; both were fixed before financial dispatch. Failed
autotuning/path-length probes remain engineering evidence, not financial fits.

Financial source is frozen commit `b16732e83a9031571429d2b14c31eeda4bd54d74`.
There were 3,050 completed epochs: TE neutral/economic 585/587, C6 939/939.
Summed fit-process time is approximately 10.59 hours, excluding outside-fit startup,
readouts and recovery. Median selected epochs were 8 for TE and 14.5 for C6.
Only two C6 fits reached the 60-epoch cap; attention did not. Mean selected clean-fit
IC was about .075 for TE and .047 for C6. These diagnostics do not resemble the
previous .8 in-sample collapse, but neither guarantee absence of overfitting.

The supervisor ran from 16 September 14:24 UTC to 17 September 04:03 UTC, roughly 13h39m
including failures, verification and readouts. After 91 confirmation fits, CUDA
failed a 20 MiB allocation despite reporting 3.60 GiB free. The exact OS/driver cause
was not established. A fresh worker then correctly refused an empty incomplete
fit directory with no completed-epoch checkpoint. That directory and both failure
logs were preserved. All 91 completed fits were verified and reused; the failed
first-epoch fit restarted from its unchanged parent/seed. An external continuation
wrapper dispatched the 29 remaining jobs through the unchanged trainer, then all
registered readouts. No completed financial fit was discarded or silently rebound.

## Storage, verification and recovery

Lossless NTFS compression preserved input bytes. User-authorized cleanup on C/D
removed 19 obsolete directories and 8,658 redundant historical epoch copies,
recovering 63.746 GiB. Historical selected/explicitly referenced checkpoints,
scores/books and complete hash-verified Round7 recovery archives remain. Two
superseded intermediate repaired stores were removed; the accepted store and
immutable raw sources were retained. The user later freed additional C space.
Cleanup inventories, receipts and recovery-member references are preserved under
the run's `phase3/local_engineering/disk_cleanup` and workspace `audit-tools`.

[The reproducible completion audit](../ops/audit_decision_phase3.py) verifies all
168 fit manifests/checkpoints, score artifact hashes, canonical date/ISIN axes,
exact active masks on every trained horizon, finite predictions, development
access flags, economic fitting bounds, and 352 book file hashes/reconciliation.
It checked 5,366 bound artifact hashes. All 84 training pairs match except for
the auxiliary objective; all 14 mapping fit/selection windows precede evaluation.
Four smoke fits are excluded. This complements
the original 126-input hash verification and causal tests; it is not a claim that
free source data or settlement assumptions are perfect.

The [recovery record](v2_decision_complete_recovery.json) binds the complete local
archive and a per-file inventory. Every archived member is read back and hashed.
The unchanged original economic caches are bound by the original freeze rather
than duplicated. This is local recovery on D, not a claim of cloud upload.

## What this answers and what remains

Against the original plan: Phase 1 corrected and measured the economic interface;
Phase 2 tested actual behavioral learning and closed failed branches honestly;
Phase 3 completed matched objective screening, all triggered confirmation, causal
blend diagnostics, full-history accounts and recovery. No authorized Phase 1–3
financial work remains.

The next question should be narrow. The auxiliary head is poorly calibrated and
not directly consumed by allocation, so this result does not test end-to-end
decision learning or economic checkpoint selection. A future registered comparison
could test one of those interfaces, or the original Phase 4 parent/fresh question
if transfer diagnostics justify it. Context history/market-to-stock attention
remains another untested architectural hypothesis. Do not launch a joint
architecture/objective/optimizer/policy sweep or retune on these failed gates.

Priority is to distinguish an insufficient economic signal from an interface that
does not exploit it, while keeping fresh paired controls and settlement sensitivity.
Retain early attention as a research candidate; neither its promising absolute
performance nor the small blend effect establishes a new default. Phase 4 requires
a separate research decision, and held-out/forward-capture prohibitions remain.
