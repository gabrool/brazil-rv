# Experiment 46 — Cross-equity structure program

Status: frozen before any Experiment-46 score access.

The following specification is preserved verbatim from
`C:\Users\gabri\Downloads\cross_equity_spec.md`.

# Cross-equity structure program — N0 neutralization + F-peer features

Preregistration-style instructions for a coding model in the Brazil-RV
repository. Log under the next free experiment number and freeze this section
in `EXPERIMENT_LOG.md` (full text under `research/preregistrations/`) before
any score exists. Discovery-only: official validation and the held-out test
are NOT accessed anywhere. Independent of the Experiment-45 read; nothing
here alters that event or the deployed recipe. Context on record: D1
(backward extension) and D2 (broker-flow) are closed; this is the remaining
data-side program, testing the one untested cell of the cross-equity matrix —
given-graph structure — after three learned-graph failures.

## Part 0 — the peer graph (shared infrastructure, PIT-causal by construction)

Built once as an immutable sidecar from the existing 1-minute archive; no
external data, no classification labels.

1. **Inputs.** Daily session close-to-close log returns per name,
   vol-normalized by the store's causal daily vol, then cross-sectionally
   demeaned per date (median-removed) so the graph reflects residual
   co-movement, not shared market beta.
2. **Estimation.** Trailing 126-session Spearman correlation between names,
   recomputed on the first session of each month, using only data through the
   prior session. A name is eligible in a month if it has ≥ 80% observed
   sessions in the window.
3. **Graph forms.** (a) *Peer sets* for F-peer: each name's top-8 correlation
   neighbors, excluding self, subject to ρ ≥ 0.15; if fewer than 3 qualify,
   the name's peer features are masked that month. (b) *Discrete clusters*
   for N0: average-linkage agglomerative clustering on correlation distance
   (1 − ρ), cut to G = 12 clusters; clusters smaller than 3 merged into the
   nearest cluster by average linkage. All parameters fixed here; no
   alternatives may be evaluated.
4. **Audit (report, no gate).** Monthly snapshots hashed; sanity checks:
   share-class pairs (e.g., PETR3/PETR4, ELET3/ELET6) mutually top-1;
   month-over-month cluster stability (adjusted Rand index); cluster size
   distribution.

## Part 1 — N0: exogenous group neutralization and score decomposition (zero GPU)

Runs on the retained store-v2 fold comparator predictions (Experiment-41
Stage-C cross-fit Raw Patience-3, folds C/A/B, both parities, seeds
rank-averaged). The canonical 58-field parent arrays may be reported as an
informational column.

**Groupings (frozen):** (1) correlation clusters (Part 0, primary);
(2) liquidity terciles from the store's causal median-daily-dollar-volume
field, recomputed monthly; (3) continuous beta exposure via the stored causal
`beta_to_WIN` (analysis use only — its input-mask policy is unaffected);
(4) current B3 sector labels — **diagnostic only, non-evidential**, with the
PIT violation (present-day labels on historical windows) recorded; no
adoption decision may cite grouping 4.

**Transforms (frozen):** per (date, decision, horizon) group instance —
discrete: s′ = s − λ·(cluster mean of s), λ ∈ {0.25, 0.5, 0.75, 1.0};
continuous: s′ = s − λ·(OLS-fitted exposure component). No other transform.

**Readouts and decision rule (frozen):**

1. **Decomposition diagnostic (always reported; the program's main product).**
   For each grouping, per fold and horizon: the cross-group variance share of
   scores; the IC of the cross-group component alone (group-mean scores vs
   the official target); the IC of the within-group component alone
   (group-demeaned scores); and for liquidity, the correlation of scores with
   ADV rank. This measures whether the model's group-level bets carry signal
   or noise, and whether measured IC leans on a liquidity factor.
2. **Adoption screen for a neutralized deployment transform.** λ and grouping
   are selected on two folds and evaluated only on the held-out third,
   rotated all three ways (the Experiment-44 honesty pattern). A candidate
   (grouping, λ) is *supported* only if, on the held-out evaluations, BOTH:
   (a) IC non-inferiority — mean paired delta ≥ −0.0005 with block-10 95%
   lower bound ≥ −0.001; and (b) stability superiority — reduction in daily-IC
   standard deviation (equivalently, improvement in mean/std of the daily IC
   series) with a paired block-10 bootstrap 95% interval excluding zero.
   The stability claim is the hypothesis; IC is the guardrail. Report
   per-cluster IC alongside global IC so the λ→1 stitching artifact (global
   Spearman degrading mechanically while within-group information is intact)
   cannot be misread.
3. A supported candidate earns registration as a deployment-transform arm in
   a future official read (applied to the deployed recipe's scores). Nothing
   is deployed from fold evidence. If nothing is supported, N0's diagnostic
   output stands as its result.

## Part 2 — F-peer: input-side peer-relative features

**F2 stage (no GPU).** Compute the predeclared candidates over the training
window from the 1-minute archive plus the Part-0 peer sets, strictly causal,
masked when peers < 3:

1. `peer_mean_return_15m` — mean of peers' trailing 15m vol-normalized
   returns, excluding self
2. `peer_mean_return_60m`
3. `peer_relative_return_60m` — own 60m return minus peer mean
4. `peer_dispersion_60m` — std of peers' 60m returns
5. `peer_breadth_15m` — fraction of peers positive over trailing 15m
6. `peer_relative_volume_surprise` — own volume surprise minus peer mean
7. `peer_mean_return_1d` — peers' prior-session close-to-close mean (lead-lag)
8. `peer_relative_return_1d` — own prior-session return minus peer mean

Screen with the Experiment-39 F2 rules: per-feature daily cross-sectional IC
on dates ≤ 2023-03-31, split-half sign consistency, |IC| ≥ 0.001, plus
redundancy exclusion at |ρ| ≥ 0.80 against any surviving store-v2 field or
retained candidate. Survivors (expect 2–5) form ONE combined candidate. If
fewer than 2 survive, the program ends here with the F2 table as its result —
no GPU is spent.

**F3 stage (GPU; 9 trajectories ≈ 2 h).** The combined candidate as a
bias-free zero-init sidecar adapter on the store-v2 parent (the Experiment-39
F3 mechanism; no store rebuild), folds C/A/B × seeds 11/29/47, canonical
contract, standard readout: bidirectional cross-fitted Raw Patience-3 primary,
final EMA-0.995 secondary, paired deltas vs the store-v2 fold comparator,
block-5/10 bootstraps, horizon/TOD guardrails. Gate (complexity-adding →
superiority): three-fold mean ≥ +0.001 AND every fold ≥ 0, intervals
supporting. On a pass, the candidate earns an arm in the next read
registration. Whatever the gate outcome, the candidate's EMA-state members
are archived and declared eligible for the next ensemble-pool registration
(they are NOT scored as pool members here — Experiment 44 is closed).

## Part 3 — T-peer (registered as conditional; NOT executed here)

If N0's decomposition shows the cross-group score component carries ≤ 0 IC
on at least two folds, a peer-demeaned auxiliary-target member family
(residual-aux mechanism with the Part-0 clusters) is declared a registered
future candidate. No training, scoring, or sidecar construction for T-peer
occurs in this program.

## Reporting and hygiene

One immutable program root: graph sidecar + audits; the full N0 decomposition
tables and neutralization sweep with rotation structure; the F2 screen table
with rule attribution per candidate; F3 analyses and gate verdicts quoting
rules verbatim; any arm registrations; SHA-256 for every artifact;
`official_validation_accessed=false`, `test_accessed=false`; deletion-first
cleanup after decisions are recorded (retain the graph sidecar, all
prediction archives, F3 member predictions and manifests; F3 checkpoints
follow the standard retention rule for prospective read-arm members).
Explicit non-goals: no learned-graph mechanisms, no sector-label adoption
decisions, no alternative graph parameters, no target changes, no
Experiment-44 reopening, no official access, no post-score edits.

## Budget and honest priors

Part 0 + Part 1 + F2: CPU only. F3: ≈ 2 GPU-h. Priors: N0 adoption is
unlikely (IC-flat/stability-up is a demanding joint claim) but its
decomposition is guaranteed information — it prices the model's group-level
bets and the liquidity leaning for the first time. F2 survival is a genuine
coin-flip given the P1 library's fate; an F3 pass at these effect sizes would
be the program's first successful feature addition, so expect shrinkage and
treat any pass as an arm, not a victory. The program's floor outcome — all
gates fail — still delivers the graph sidecar, the decomposition, and the
closure of the given-graph cell, completing the cross-equity question at
every tier.

## Frozen implementation resolutions

- Experiment number: 46. This remains discovery-only and does not change the
  Experiment-45 deployment.
- The monthly graph return for store date `d` is slow field
  `previous_close_to_close_return_normalized` at `d`; it represents the prior
  completed session. The 126 rows ending at the first session of the month
  therefore contain no return from that session.
- Intraday peer returns use incumbent dynamic fields `return_15m_normalized`
  and `return_60m_normalized` at `DECISION_EQUITY_INDICES[q] - 1`, the last
  predecision minute. Volume surprise uses incumbent `volume_surprise` at the
  same minute. Prior-day features use the same slow return field as the graph.
- A peer measurement is valid only when the focal name and at least three of
  its frozen monthly peers are active and have the applicable incumbent
  measurement. An intraday return is applicable when at least 80% of its
  trailing window's minute bars are observed; volume surprise requires the
  last predecision bar observed; the daily field requires store readiness.
- F2 follows Experiment 39's chronological-half IC and incremental ordering,
  with this specification's overrides: eight candidates only, no family cap,
  no top-k cap below eight, redundancy threshold `abs(rho) >= 0.80`, and F3
  requires at least two survivors. Store-v2 redundancy comparisons exclude
  precisely the dynamic and slow fields zeroed by the immutable Stage-C
  store-v2 feature specification.
- F3 training applies those same immutable store-v2 equity-input zero masks
  from epoch zero. Context/global inputs and all history masks are unchanged.
- Current B3 sector labels are diagnostic only if an existing present-day
  mapping covers the canonical security axis. If no such mapping exists in
  canonical project inputs, the sector diagnostic is recorded as unavailable;
  it is never synthesized and never enters selection.
- Rotation selection maximizes training-fold mean stability improvement among
  candidates satisfying training-fold IC non-inferiority, with ties broken by
  larger mean IC delta, then grouping order (clusters, liquidity, beta), then
  smaller lambda. This rule is applied independently for each held-out fold.
- Bootstrap uses 10-session moving blocks, 10,000 deterministic draws with
  seed 46. F3 analysis retains the existing deterministic block-5/block-10
  implementation and adds the interval-support requirement to the final gate:
  after concatenating the three folds' daily paired deltas in fold C/A/B
  order, both the block-5 and block-10 95% lower bounds must be strictly above
  zero.
- The active paid GH200 instance is explicitly retained after Experiment 46
  at the user's request.

## Operational repair after frozen F3 training

- All nine F3 trajectories completed under commit
  `0976c4004194abe5b8a982b4bc6bdbf346cf4926` before the analysis repair.
  The first post-training comparison stopped before producing a fold verdict
  because the standing designated-challenger helper asserted that the supplied
  retention parent must equal the challenger artifact's legacy embedded parent.
- Experiment 46's frozen retention parent is instead the accepted store-v2
  parent. The repair therefore leaves every trained prediction, replay, fold,
  comparator, readout, interval, and gate unchanged; it only permits the
  supplied hash-verified store-v2 parent to remain the retention comparator
  while the standing challenger remains informational. An analysis-only resume
  path verifies all nine completed run manifests and reuses their immutable
  prediction archives without training another trajectory.
