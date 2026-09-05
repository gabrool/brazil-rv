# v2 research registration — Round 1 (baselines, GBDT ladder) and Round 2 (data-span arms, network vs GBDT)

For the coding model. Freeze in `EXPERIMENT_LOG.md` (full text under
`research/preregistrations/v2_round1_round2.md`) before any Round-1 score is
computed. This is the first v2 research registration; Section C acceptance
is met at commit 23586d4 on the accepted store `v2_daily_store_98e9386_*`.

## Common protocol (applies to every v2 research experiment)

- Folds F1/F2/F3 exactly as defined in `brazil_rv.v2.splits`; fit windows
  with the 75-session embargo; block-parity cross-fit inside selection
  windows; sealed 2025/2026 windows untouched (no registration token is
  issued by this document).
- Readouts for every candidate, per fold and pooled: residual IC pooled
  over D ∈ {1,2,3,5} (primary statistical), per-horizon IC including D10,
  raw-return Rank-IC, persistence at 1 and 5 sessions (per parity model),
  decile spread in bps per holding session, and the swing-economics grid
  with the 4 bps / 2% cell as headline (primary economic). Block-20
  moving-block bootstrap intervals on daily series; paired daily deltas for
  every comparison.
- Seeds: networks 11/29/47 (rank-averaged); GBDT five seeds.
- Decisions are weighed on the four readouts; the "preference" rules below
  state defaults, never vetoes. Every number is reported regardless.
- Compute: CPU for Round 1; one GH200 session (4–6 concurrent trajectories)
  for Round 2, terminated and verified absent afterward.

## Round 1 — the number to beat, and the GBDT parent (CPU)

**R1.1 Baseline table.** Report the naive-signal evaluations (reversal 5,
reversal 21, momentum 12-1, the rank-gauss blend) on F1–F3 with the full
readouts. The full-scale integration runs may be reused only if their
manifests bind the current accepted store and code; otherwise recompute.
This table is the v2 floor and the first reference every later result is
read against.

**R1.2 GBDT ladder** (five seeds, F1–F3, early stopping on the selection
parity; each rung is the previous rung plus one group):
(a) slow feature library only (2010→ history, broad universe);
(b) + intraday-derived daily features (2021→, masked before);
(c) + lending sidecar;
(d) + odd-lot, options, rebalance, events, and fundamentals sidecars.
Paired deltas rung-vs-previous on all four readouts. Preference: keep a rung
if its pooled-IC delta is positive with an interval mostly above zero OR
its headline net excess improves; drop a rung that worsens both; keep
ambiguous rungs (features are cheap for GBDT) and flag them for the
network stage. The **GBDT parent** = the best kept rung by mean pooled IC
across folds, economics as tie-break. Report gain and SHAP importances for
the parent.

**R1.3 GBDT data-span preview** on the parent feature set: five-year-only
(2021-08→ fit end) vs + pretrain window uniform vs + pretrain window with
756-session time-decay weights. Informational input to R2.1; no decision.

## Round 2 — the starter network, the data-span decision, network vs GBDT (GPU)

**R2.1 Data-span arms.** The §5 starter network on the R1 parent feature
set (sidecar features enter the slow stream; the intraday store feeds the
fast stream), λ_pers = 0, 20 epochs, patience 3, Patience/EMA states
archived, three seeds, F1–F3:
- Arm A — five-year-only: stage F from scratch on 2021-08-16 → fit end.
- Arm B — pretrain→fine-tune: stage P on 2010-01 → 2021-07-31 once per
  seed (20 epochs, patience on the internal holdout), then stage F with
  the 0.3 learning-rate multiplier on pretrained parameters.
- Arm C — joint: stage J on both windows with 756-session decay.
Paired deltas B−A and C−A on every readout. Preference: the arm with the
highest mean pooled IC across folds becomes the standard data regime
unless its headline economics are worse than Arm A's; if neither long-
history arm beats A by more than 0.002 pooled with intervals spanning
zero, prefer A (simpler) and record the long-history question as
"not harmful, not yet helpful." Stage-P holdout values are reported for
the era-difficulty record but carry no decision weight.

**R2.2 Network vs GBDT and the ensemble.** On the chosen regime: the
network (three-seed rank average) vs the GBDT parent (five-seed) vs their
equal-weight rank-average ensemble, paired on all readouts. Preference:
the best of the three by pooled IC with economics as tie-break becomes the
**v2 parent** carried into Round 3 (encoder ablations and the
IC/persistence frontier), registered separately.

## Outputs and hygiene

One immutable root per round: manifests and hashes for every run, all
readouts with intervals, paired-delta tables, importances, the rung/arm/
parent designations quoting the preference rules verbatim, access flags
false throughout, dated log entries. No sealed-window access, no reads,
no deployment change, no post-score additions of rungs, arms, or
readouts. Surprises go to the Round-3 registration.

## Expectations, recorded in advance

The integration run's F1 pooled IC of 0.046 (one seed, three epochs, no
sidecars) is the prior for Arm A on F1, with ±0.01 window uncertainty and a
known regime caveat; F2/F3 are expected to differ materially. The
horizon-rising IC profile and ~0.96 persistence are expected to persist —
that shape, not the level, is what makes v2 economically different from
v1. The honest open questions this registration answers: whether the
network beats naive characteristics and GBDT on the same features, and
whether sixteen years of history beats five.

## Frozen implementation resolutions

These resolutions remove implementation ambiguity without changing the supplied
candidate, split, seed, training, or readout grids.

- A candidate's pooled daily series concatenates the three chronological fold
  series. Bootstrap blocks are sampled independently within each fold and never
  cross a fold boundary. Point estimates average all finite daily observations.
- The four decision families are residual IC, raw-return Rank-IC, persistence,
  and economics. Decile spread remains a mandatory reported diagnostic.
  Persistence is reported separately at lags 1 and 5.
- GBDT rung A contains the 32 slow features only. Rung B adds the 20 intraday
  fields and the two registered availability flags. Rung C adds lending. Rung D
  adds, in this order, oddlot, options, rebalance, events, and fundamentals.
  Invalid normalized cells remain zero under the sealed store masks; no validity
  mask is added as an unregistered predictor.
- GBDT pretraining rows use their date-t slow/sidecar fields, zero intraday
  fields, `fast_present=0`, and `days_since_last_slow_row=0`. Fine-tuning rows
  use date-(t-1) slow/sidecar fields and date-t intraday fields and flags.
- A GBDT rung is dropped only when both its pooled residual-IC point delta and
  headline net-excess point delta are negative. All other later rungs are kept
  and an interval crossing zero is flagged as ambiguous. The GBDT parent is the
  kept rung with the largest pooled residual IC; an exact IC tie is broken by
  headline economics, then by the earlier rung.
- For Round 2, each seed/parity is a trajectory. Every seed is stitched by
  opposite five-session block parity, then the three stitched seed panels are
  tie-aware rank-averaged. Stage P runs once per seed and supplies its selected
  raw-Patience checkpoint to all that seed's Stage-F folds/parities.
- Long-history arms are eligible only when their headline economics are not
  below Arm A. If both B−A and C−A residual-IC estimates are at most 0.002 and
  both 95% intervals include zero, Arm A is selected. Otherwise the eligible
  arm with the largest pooled residual IC is selected; exact ties prefer A,
  then B, then C. The v2 parent uses the same IC/economics/exact-order tie rule
  in the order network, GBDT, ensemble.
- The sealed store's underlying v1 minute arrays are the fast-stream inputs.
  A v1 fast initialization is used only when one exact compatible checkpoint
  can be hash-bound before the Round-2 root is frozen; otherwise all arms use
  the same scratch fast initialization and the absence is recorded.
