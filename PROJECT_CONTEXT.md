# Brazil-RV project context

Last verified: 2026-08-23.

## Purpose and current research state

Brazil-RV is an offline research system for predicting 30-, 60-, and 120-minute
cross-sectional equity ranks from B3 M1 data. A sample is one eligible trading
date and one of 55 five-minute decisions over a fixed 158-slot point-in-time equity
axis.

The accepted incumbent is the peer-free, full causal time-of-day normalized,
width-64 causal TCN trained uniformly with soft Spearman and SAM-AdamW. The best
recorded exact validation result is seed-11 IC **0.041972**. The rejected
gap-pairwise loss, continuous-target sidecar, and residual equity-attention branch
are absent from the current tree; their commits, manifests, and immutable artifacts
remain the historical reproduction contract.

The historical parent was reproduced on 2026-08-19 at commit `4067962` with
matched seeds 11/29/47. Best-IC deltas versus the immutable records were
`+0.0000053`, `-0.0000065`, and `+0.0000009`; every best epoch and stop epoch
matched. A bidirectional odd/even-date cross-fit of the internal selection windows
subsequently froze raw Patience-3 as the trajectory rule, with its uncertainty and
Fold-B non-confirmation retained in the research record.
A no-retraining follow-up averaged five raw checkpoints around the parity-selected
Patience peak. It lost to raw Patience on both folds and was rejected; raw
Patience-3 remains frozen and checkpoint-rule investigation is closed.

Read [RESEARCH_HANDOFF.md](RESEARCH_HANDOFF.md) for architecture and campaign
history, exact results, artifact identities, and interpretations.

## Source and write boundaries

- Python 3.12; use `uv` and the `research/` project.
- Raw data under `quant-data/b3/raw/**`, canonical source archives, and
  `Trading/**` are immutable.
- Derived stores belong under `quant-data/b3/interim/**` or
  `quant-data/b3/processed/**`.
- Resolve canonical pointer files at runtime and record resolved identities in
  output manifests. Never hard-code a timestamped source when a valid pointer is
  available.
- Equity identity is permanent `security_id`/ISIN plus bounded source-assignment
  dates. Ticker is only a dated attribute.
- Monthly point-in-time membership is the eligibility contract.

## Current feature-store contract

The peer-free store contract is `M1_FEATURES_PIT_CAUSAL_TOD`: 1,248 dates, 1,228
eligible dates, 55 decisions, 158 equities, 26 dynamic channels, 32 slow channels,
three horizons, seven local contexts, and eight global contexts. It has no
human-prior or peer arrays.

Equity normalization uses an equity-wide 30-minute causal relative-variance TOD
profile with a 20-session-equivalent prior and `[0.25, 4.0]` bounds. Each training
date emits before updating; the profile freezes after 2024-06-28. Context series
retain their semantic causal transforms rather than receiving the equity overlay.

The full store built on persistent Lambda NFS is recorded in
`RESEARCH_HANDOFF.md`. The Windows local canonical pointer may still identify the
old V4 store; verify the pointer and schema in the execution environment before
training.

Core causal rules:

- History ends strictly before the decision; the entry bar is excluded.
- Label entry is `open[T]`; exit is exact `close[T+h-1]` within the session.
- Missing OHLC is never interpolated and stale prices are not label endpoints.
- Fitted scalers, volatility/TOD state, and other stateful transforms use only the
  information available at their historical timestamp.
- Training, validation, and test identities are immutable and audited.

External data experiments use the immutable `PIT_EXTERNAL_FEATURE_SIDECAR`
contract. A sidecar is bound to the exact canonical feature-store identity and
date/equity axis hashes. Daily arrays have shape `[date, 158, feature]`; intraday
arrays add the canonical 55-decision axis. Every feature has an explicit mask,
invalid values are exactly zero, and source-specific availability is materialized
as an exact no-fill join before training. The loader additionally gates values and
masks by point-in-time equity membership. A single per-equity bias-free linear
residual injects concatenated values and masks into the incumbent state; its
weight is zero-initialized, and candidate construction restores the parent's RNG
state after adding it. Thus every external-data candidate begins as the exact
parent without changing base weights or dropout randomness. An equity with no
valid external observation has an all-zero input and therefore receives exactly
zero direct adapter residual throughout training; learned mask weights still
represent observedness where data is present.

## Splits, discovery folds, and test policy

- Training: 2021-08-16 through 2024-06-28, 716 dates.
- Validation: 2024-07-08 through 2025-06-30, 244 dates.
- Held-out test: 2025-07-07 through 2026-07-17, 259 dates.
- Fold A: first 512 training dates fit, next 102 select.
- Fold B: first 614 training dates fit, final 102 select.

The two internal selection periods do not overlap. Both fit windows preserve an
effective batch of 512 distinct dates. Stored features are causal, but the TOD
profile adapted inside the historical training dates, so these are screening
folds rather than exact replicas of the officially frozen preprocessing regime.

The official validation split has already been consumed and is reserved for
sparse confirmation of stage winners. Embargo dates are not selection data. The
held-out test is the final lockbox and may be opened only through the explicit
standalone evaluator for an official-window run carrying a rule frozen on the
internal folds. Campaign drivers cannot request validation or test rows.

## Accepted model and trajectory contract

- One shared per-instrument width-64 causal TCN.
- Five-minute patches, 69 patches, kernel 3, dilations `(1, 2, 4, 8, 16, 32)`.
- Six residual LayerNorm/SwiGLU blocks and final-state readout.
- Projected 32-field slow state.
- Fixed context-plus-masked-equity-mean/dispersion gated fusion.
- `WIN$` and equity `beta_to_WIN` masked; WDO, five DI contexts, ZT, and ZN active.
- All three horizons trained jointly.
- Sole objective: soft Spearman, temperature 0.50.
- Uniform training dates; SAM-AdamW rho 0.125; effective batch 512.
- One fixed 20-epoch trajectory; no training-time early stopping.
- Raw checkpoint and raw/EMA validation predictions every epoch.
- EMA decays 0.98, 0.99, and 0.995.
- Frozen rule: raw Patience-3 with minimum IC improvement `0.0001`, patience three,
  maximum 20 epochs, and restoration of the best raw checkpoint. This entire rule
  is fixed before the sparse official-validation stage; do not retune it there.
- Last-3/last-5 weight averages and raw-score prediction averages are constructed
  without retraining.
- Retrospective best epoch remains diagnostic only.
- No peer/classification inputs and no cross-equity attention.

Hard Spearman is the primary selection metric. It is averaged across decisions
within each date and horizon, then equally across dates and horizons. Seed
ensembles uniformly average tie-aware within-sample/horizon ranks and never fit
ensemble weights.

The completed internal campaign at
`trajectory_discovery_e22dd67_20260819T134332Z` initially selected final EMA-0.995
from fixed rules, with fold-A/fold-B ensemble ICs `0.045309`/`0.050625` and mean
`0.047967`, versus final-raw `0.043416`/`0.049602` and mean `0.046509`. Paired
EMA-minus-raw deltas were positive on both folds (`+0.001892`, `+0.001024`) and at
every horizon, but their block-bootstrap intervals mostly included zero and
time-of-day deltas were mixed. Treat the rule as a deterministic variance-reduction
choice, not a claim of uniform statistical dominance.

The same-window Patience-3 IC `0.051860` was selection-biased. The corrective
artifact `trajectory_crossfit_3054228_20260819T161200Z` selected checkpoints on
one odd/even date parity and reported them only on the other, in both directions.
Raw Patience-3 scored `0.048416`/`0.050673` and mean `0.049545`, versus final
EMA-0.995 mean `0.047967`. Its paired advantage was `+0.003108` on Fold A but only
`+0.000048` on Fold B, with both block-bootstrap intervals including zero. EMA-0.995
Patience scored `0.048518`; last-10 raw weight averaging scored `0.048060`, while
last-7 scored `0.047352`. The outer rule replay chose raw Patience in three of four
directions and EMA Patience once, with mean out-of-half IC `0.048897`. Raw
Patience-3 is frozen as the numerical winner, but it is not treated as established
dominance over final EMA-0.995.

The one-candidate centered-average follow-up at evaluator commit `381dcb7`
scored `0.046655`/`0.050385`, mean `0.048520`, versus raw Patience mean
`0.049545`. Centered-minus-raw-Patience was `-0.001761` on Fold A and
`-0.000288` on Fold B and was negative in all four out-of-half directions. The
candidate was rejected without another sweep. Its code was removed from current
HEAD; exact reproduction uses the recorded evaluator commit and immutable
`trajectory_centered_crossfit_381dcb7_20260819T170100Z` artifact. Neither
official validation nor test was accessed.

## Current source-tree status

`modeling.train` is the canonical soft-Spearman trajectory entry point.
`modeling.run_discovery_campaign` runs exactly the two internal folds and seeds
11/29/47. Sidecar campaigns do not select a fresh checkpoint rule: they record
the frozen bidirectional odd/even Raw Patience-3 primary and final EMA-0.995
secondary readouts. `modeling.crossfit` selects
validation-adaptive checkpoints on one odd/even date parity and reports only on the
other, replays rule selection in both directions, and can materialize last-7/last-10
weight-average predictions without mutating source runs. Its frozen selection file
is the authority for the next official-window run. `modeling.analyze` strictly
aligns observations and reports member/ensemble IC, seed diversity, paired date
deltas, moving-block intervals, and horizon/time-of-day guardrails.

`modeling.external_data_screen` applies that frozen readout contract to one
completed sidecar campaign. It reports candidate deltas against both the canonical
parent and the standing designated challenger, plus a predeclared uniform
parent-plus-candidate diversity readout. The standalone candidate and that fixed
six-member diversity recipe are separate predeclared retention paths, and both
are keyed only to their canonical-parent deltas. The challenger and EMA columns
are informational and cannot create a "beats either" selection rule.
Either primary path requires at least `+0.001` mean fold IC gain and non-negative
gain on both folds. The diversity path additionally requires the standalone
candidate to lose no more than `0.001` on either fold.

The completed historical external-data program
`external_data_7e535ac_20260821T161800Z` tested B3 lending, SHFE ferrous/pulp,
COTAHIST options activity, CVM RAD events, B3 odd-lot activity, B3 index
rebalances, CCEE PLD, CVM fundamentals, regular trade activity, and ADR overnight
under this contract. No standalone or parent-plus-candidate recipe passed the
frozen gate, so none of these exact feature families is part of the accepted
recipe. Official validation and test were not accessed. Options activity was
positive on both folds but its mean gains were only `+0.000257` standalone and
`+0.000261` in the diversity recipe, below the threshold. Positive fixed-final-
EMA observations for several candidates remain informational and do not change
retention or reopen checkpoint selection. Source components that were explicitly
unavailable and excluded from the screens remain untested rather than rejected;
`EXPERIMENT_LOG.md` records the exact boundaries and results.

`modeling.evaluate` restores held-out evaluation without exposing it to campaign
drivers. It accepts only a completed official-window run with an internal-fold
selection file recorded in its manifest.

The old human-prior/peer, alternate model-family, routing, multiscale, attribution,
probe, overlay, V-numbered, recency-weight, hybrid-loss, target-sidecar, and
residual-attention experiment systems are not compatibility APIs. Historical
reproduction uses their recorded commits and immutable artifacts.

## Environment and operations

From the repository root:

    uv sync --project research --group dev
    uv run --project research python -m brazil_rv.preprocessing.build
    uv run --project research python -m brazil_rv.modeling.run_discovery_campaign --output-dir <new-campaign-directory>
    uv run --project research python -m brazil_rv.modeling.train --selection-window official --selection-rule-file <trajectory-selection.json> --seed 11
    uv run --project research python -m brazil_rv.modeling.evaluate --run-dir <completed-official-run-directory> --split test

`ops/lambda-gh200.ps1` is the only Lambda watcher/launcher. Launch requires
explicit billing acknowledgement, transfers a verified Git bundle, and leaves the
instance running. It never starts training or terminates a paid host. Confirm
provider state and exact instance identity before launch or termination.

## Limitations and authority

The system does not model order-book state, bid/ask spread, queue position,
slippage, costs, or live execution. Historical MT5 spread and tick volume are not
market microstructure. Raw absolute prices, tickers, identifiers, news, and
unapproved technical indicators are not model features.

When statements conflict, prefer immutable sources and canonical pointers, then
executable code and tests, then this document, then the detailed handoff.

## Phase A representation decision (2026-08-20)

Six zero-start residual representation candidates were screened on the two
internal discovery folds with seeds 11/29/47 and the frozen odd/even cross-fitted
raw Patience-3 readout. Final EMA-0.995 was recorded from the same trajectories as
a secondary readout. The campaign produced 720 checkpoints from commit `732b1b0`.

| Candidate | Primary mean candidate-minus-parent IC | EMA-0.995 secondary mean delta |
|---|---:|---:|
| Decision-time embedding | -0.000009522 | -0.000133778 |
| Temporal mean/std adapter | -0.000104324 | +0.000009399 |
| Block-2/4/6 multi-depth stats | -0.000147827 | -0.000975001 |
| Cross-sectional max/min | -0.000198130 | +0.000772140 |
| Learned set pool, width 16 | -0.000006211 | +0.000005459 |
| Conditional beta/volatility bucket means | -0.000198862 | +0.000424682 |

Every primary mean was non-positive. The secondary max/min and conditional-bucket
gains were confined to Fold A and reversed on Fold B. No candidate had a coherent
positive horizon/TOD profile, so none qualified for sparse official validation.
Raw Patience-3 remains the canonical parent and the held-out test remains sealed.

The completed immutable campaign is:

    /lambda/nfs/brazil-rv-east3/quant-data/b3/processed/model_runs/phase_a_732b1b0_20260819T180348Z

Its manifest records `official_validation_accessed=false` and
`test_accessed=false`. Rejected candidate code and its campaign driver were
removed from current HEAD. Reproduction is through commit `732b1b0` and the
immutable artifacts, not compatibility branches. The general strict
observation-level measurement layer remains canonical.

## Phase A autopsy and diversity follow-up (2026-08-20)

Checkpoint autopsy disproved the hypothesis that the historical decision-time and
learned-set paths were dead. Decision-time final-projection norms reached
`0.319-0.355`; learned-set final-projection norms reached `0.490-1.107`, and its
standard-initialized `phi` weights moved materially. Learned set already entered
the incumbent nonlinear shared fusion with only its final projection zeroed.
Their prediction ranks nevertheless remained above 0.9991 correlated with the
matched parent, so both were active but rank-ineffective.

A no-training uniform-rank reanalysis pooled the three parent members with the
three decorrelated multi-depth members. Under cross-fitted raw Patience-3 it added
`+0.001237` on Fold A and `+0.000284` on Fold B, mean `+0.000761`. The same
six-member pool added mean `+0.001651` under final EMA-0.995. Adding the three
temporal-statistics members diluted both readouts. All block-bootstrap intervals
included zero, so parent+multi-depth is retained only as the one Phase A
diversity-ensemble candidate eligible for sparse official-validation confirmation;
it is not yet a canonical lockbox recipe and no ensemble weights may be learned.
The immutable reanalysis is:

    /lambda/nfs/brazil-rv-east3/quant-data/b3/processed/model_runs/phase_a_autopsy_d237998_20260820T111500Z

The remaining decision-time routing objection was then tested directly. A
standard-initialized width-16 decision embedding fed a zero-only final projection
into the existing shared nonlinear fusion, and candidate construction preserved
the parent's RNG stream exactly. A 10-step soft-Spearman assertion confirmed that
both the final projection and upstream embedding moved. Across two folds and
three seeds, cross-fitted Patience candidate-minus-parent IC was
`-0.000001`/`-0.000009`, mean `-0.000005`; EMA-0.995 mean delta was
`-0.000001`. Final adapter norms ranged `0.299-0.992`. This is a conclusive active
null, so decision-time embedding is closed and must not consume official
validation. Exact reproduction uses commits `9828f72`/`b8d955a` and:

    /lambda/nfs/brazil-rv-east3/quant-data/b3/processed/model_runs/decision_time_fusion_b8d955a_20260820T113924Z

Raw Patience-3 on the parent architecture remains the canonical base for future
experiments. The rejected corrected-adapter code was deleted from current HEAD.
The canonical analyzer now permits candidate and parent ensembles with different
member counts while preserving strict observation alignment and uniform ranks.
Neither follow-up accessed official validation or the held-out test.

## Phase B target-decomposition decision (2026-08-20)

An immutable auxiliary-target sidecar was audited before training. It used stored
causal pre-neutralization `beta_to_WIN`, exact WIN decision-open-to-label-close
returns with observed endpoint masks, and no stale prices. Residual returns were
factor-neutralized, median-centered, normalized with the existing causal
volatility contract, and cross-sectionally midranked. Beta and exact WIN endpoint
coverage both exceeded 0.9985 across all horizons; mutation-based causality tests
passed. The immutable sidecar is:

    /lambda/nfs/brazil-rv-east3/quant-data/b3/processed/auxiliary_targets/phase_b_aux_15471e8_20260820T141500Z

Residual-rank, sign, magnitude, and combined auxiliary supervision were screened
on both internal folds with seeds 11/29/47 and 20-epoch trajectories. Primary
mean candidate-minus-parent IC was `-0.000625`, `-0.000482`, `-0.002599`, and
`-0.000351`, respectively. No primary candidate improved both folds, and no
Phase B member improved the existing Phase-A diversity stack on both folds. The
EMA-positive residual/combined secondary readouts did not override the frozen
Raw Patience-3 primary. Therefore the conditional common-component head was not
run. The completed 480-checkpoint campaign is:

    /lambda/nfs/brazil-rv-east3/quant-data/b3/processed/model_runs/phase_b_6b7b121_20260820T145500Z

The parent then received one three-epoch, latest-120-date, learning-rate-divided-
by-ten recency trajectory per fold/seed/direction. The best average candidate was
the epoch-3 50/50 full/fine rank ensemble at `+0.000457`, but it was
`-0.000930` on Fold A and `+0.001843` on Fold B. The both-fold guardrail retained
full history.

After Phase B, the only stage finalist was the prior parent-3 plus Phase-A
multi-depth-3 diversity pool. Its one sparse official-validation confirmation
scored `0.040495819` versus parent-3 `0.041639843`, delta `-0.001144024`;
30/60/120-minute deltas were all negative. Reject the six-member pool. The sole
canonical recipe remains the three-seed parent with Raw Patience-3. Official
validation is closed again, and the held-out test has never been accessed.
Official artifacts are:

    /lambda/nfs/brazil-rv-east3/quant-data/b3/processed/model_runs/phase_a_official_732b1b0_20260820T201500Z

Rejected Phase B sidecar/training/recency plumbing and the one-use confirmation
driver were deleted from current HEAD. Reproduction is through commits
`a04d63e`, `15471e8`, `6b7b121`, `e33a122`, and immutable artifacts, not
compatibility code.


## Designated challenger and comparator policy (2026-08-21)

The standing designated challenger is fixed as the uniform six-member rank
ensemble of:

- the three canonical parent members at seeds 11/29/47, with Raw Patience-3
  selected bidirectionally on the opposite odd/even discovery-date parity; and
- the three Experiment-18 residual-auxiliary members at the same seeds, read at
  fixed final EMA-0.995.

The auxiliary configuration is frozen to commit `3b60ac9` and its immutable run
manifests: WIN + WDO + ready-DI-level residual rank, soft-Spearman auxiliary
weight 0.5, separate zero-weight/zero-bias auxiliary head, parent initialization
and RNG stream preserved, width 64, 20 epochs, SAM rho 0.125, learning rate
0.0003, and EMA decay 0.995. All six predictions are tie-aware ranked within each
sample/horizon and averaged uniformly; weights are never learned.

This is not a second retention baseline. Every future discovery-fold candidate
must report paired deltas against both the canonical parent and this challenger,
but candidate retention remains keyed exclusively to the canonical parent. "Beats either" selection is prohibited. The challenger column is informational
evidence accumulated passively. The canonical entry point for future fold screens
is `modeling.designated_challenger.compare_discovery_screen`, whose summary
records this selection contract.

The challenger receives one official-validation comparison only when bundled into
the next official read already justified for a future stage winner. It does not
independently authorize an official read. The saved official residual final
EMA-0.995 payloads remain unopened for that purpose. The held-out test remains
sealed.

## Persistent Lambda retention cleanup (2026-08-21)

The `brazil-rv-east3` object store was reduced from 159,464,940,112 bytes
(148.513 GiB) to 20,202,855,773 bytes (18.815 GiB). The exact manifest deleted
5,928 objects totaling 139,262,802,568 bytes (129.699 GiB):

- an obsolete noncanonical human-prior V4 feature store;
- 120 unreferenced historical model-run prefixes;
- raw checkpoints, tail states, and redundant per-epoch predictions from closed
  campaigns; and
- all Phase-C/official binary intermediates except the designated challenger's
  required final EMA-0.995 predictions and observation references.

Raw and interim data were untouched. The canonical causal-TOD feature store and
pointer remain complete. The parent retains all 120 per-epoch discovery prediction
files and six references needed to reconstruct honest Patience cross-fits. The
challenger retains six discovery and three official residual epoch-20 prediction
files with their matching references. Manifests, histories, metrics, analyses,
and every run prefix named in the durable research record remain.

The immutable cleanup record is:

    /lambda/nfs/brazil-rv-east3/quant-data/b3/processed/model_runs/_retention/storage_cleanup_20260821

Its delete-list SHA-256 is
`b45f591cd4c77640ce2c924506f3040b1cdefe6b65679de7bae618217dd75f7b`.
Deleted binary intermediates are not recoverable in place; exact reruns use their
recorded commits, retained manifests/results, and canonical derived data. Lambda's
provider accounting endpoint still reported the pre-cleanup byte count immediately
after deletion, while a complete object-store listing returned the verified
post-cleanup total. No paid instance was active.

## Persistent Lambda retention cleanup, round 2 (2026-08-23)

After Experiments 27--40, a new complete object-store inventory found
109,637,262,343 bytes (102.108 GiB) in 18,970 objects. The principal growth was
the completed Experiment-27 external-data program: its 1,200 per-epoch
prediction files and 60 redundant tail bundles occupied 49.928 GiB even though
the selection rule and all ten rejection decisions were frozen.

A manifest-bound cleanup removed exactly 6,323 objects / 48,739,734,061 bytes
(45.392 GiB): 1,022 unselected external-data prediction epochs, 60 tail bundles,
5,211 ephemeral model-cache objects, and 30 rejected-preflight sidecar objects.
Before deleting the sidecar duplicate, both copies of every array were streamed
and SHA-256 checked against semantically identical manifests; the only manifest
difference was the creation timestamp. The accepted bias-free sidecar tree was
retained intact.

The external-data program retains all 178 epochs selected by at least one of
the two honest parity replays, whole-fold Patience-3, or the final epoch, plus
all 178 matching raw/EMA checkpoints, 60 observation-alignment references,
histories, manifests, diagnostics, analyses, and screen summaries. Its footprint
fell from 52.032 GiB to 9.158 GiB. Raw/interim sources, the canonical feature
store, canonical parent and challenger artifacts, Experiments 39--40, and every
other retained object were unchanged.

The fresh post-cleanup inventory contains 12,649 objects / 60,899,434,233 bytes
(56.717 GiB), including the two new audit records. Its SHA-256 is
`20f8fa4258e914f2a7731bfd0cee42d809fc34488c5f0a2716a28c6e7d9ecce6`.
The immutable object-store audit is under:

    quant-data/b3/processed/model_runs/_retention/storage_cleanup_20260823_round2

The applied plan SHA-256 is
`eda1cb77d361ac0a8fa8b5e00460aff71477812a29e63f9f46ded7535937b64b`;
the postcheck SHA-256 is
`cd68b22bc23a891a73eab6c1b75cc61e82e524dae02439153fb37cf965b23ce6`.
An independent before/after comparison found zero unexpected removals, zero
planned survivors, zero retained-object metadata changes, and exactly the two
expected audit additions. Recreating deleted prediction trajectories would
require an exact rerun from the recorded commit and retained canonical inputs.
No paid Lambda instance was active.

## Kronos-small zero-shot K0 decision (2026-08-22)

Kronos-small was evaluated zero-shot on the two 102-date discovery selection
windows using permanent point-in-time equity identity, the fixed six-decision
grid, 512 causal five-minute bars, exact per-context sampling seeds, and the
canonical 30/60/120-minute rank-IC machinery. The user narrowed the experiment
to Kronos-small before any small score or metric was inspected; an in-progress
Kronos-base pass was stopped, excluded, and its unmerged partial arrays deleted.

Kronos-small scored `0.008843` on Fold A and `0.018551` on Fold B, for mean IC
`0.013697`. This is below the preregistered `0.015` kill floor. Its matched
momentum control was `-0.016037`, parent correlation was only `0.133768`, and an
informational parent-plus-Kronos rank stack added just `+0.000128` mean IC with
opposite fold signs and block-bootstrap intervals spanning zero. The zero-shot
Kronos-small family is therefore rejected for the current program. Do not run
K1 or use official validation on the basis of K0. The canonical parent remains
unchanged, and the held-out test remains sealed.

The immutable run and reusable score artifact are:

    /lambda/nfs/brazil-rv-east3/quant-data/b3/processed/model_runs/kronos_k0_3f93b26_20260822T134800Z

Its manifest records `official_validation_accessed=false`,
`test_accessed=false`, and `k1_started=false`. The dedicated immutable K0 bar
sidecar is `kronos_k0_bars_3f93b26_20260822T134400Z` in the same model-runs
root. Exact settings, leakage caveats, operational corrections, model-scope
override, metric breakdowns, hashes, and cleanup provenance are in
`EXPERIMENT_LOG.md` and the run manifests.

Transient base partials, model caches, upstream clones, and MPS state were
deleted. The retained run and sidecar are read-only. Paid GH200 instance
`c0aef7522bf64fe0899e8703027668db` was terminated and confirmed absent from
Lambda's active inventory.

## P0/P1 feature-program decision (2026-08-22/23)

P0.2/Kronos closure was explicitly excluded and never run. P0.1 rejected both
predeclared mixed-state ensembles: the 24-member all-family stack added
`-0.001242/+0.001709` on Fold A/B (mean `+0.000233`), while the 12-member
residual/options/ADR stack added `-0.000272/+0.002278` (mean `+0.001003`). Both
failed the non-negative-every-fold gate.

Cross-fitted inference attribution classified 12 of 58 incumbent equity fields
dead, but they were removed only inside the subsequent joint candidate. F2
selected eight causal features on a disjoint first-407-date window. On Fold
C/A/B, that bias-free sidecar plus the 12-field ablation added
`-0.000568/+0.000576/+0.001054` standalone (mean `+0.000354`) and
`-0.000015/+0.000711/+0.000920` when pooled with parent-3 (mean `+0.000539`).
Neither path passed the preregistered three-fold gate. Final EMA and the P0
mixed-state secondary reads were also null-to-negative. F4 therefore recorded
`not_run` without ablation or retraining.

Reject the P0.1 stacks and the joint P1 feature/pruning recipe. P0.3 and F2 remain
diagnostic evidence only; do not delete fields from the canonical parent or
promote individual F2 features based on the joint screen. Raw Patience-3 on the
unchanged three-seed parent remains canonical, the designated challenger is
unchanged, official validation was not spent, and the held-out test remains
sealed. Exact results and source-reproduction validation are under:

    /lambda/nfs/brazil-rv-east3/quant-data/b3/processed/model_runs/p0_p1_27aa0d0_20260822T194900Z

Experiment implementations are reproducible through commits `5b6b5d4`,
`1b63661`, `27aa0d0`, and `8f46124`; rejected campaign-only code is absent from
current HEAD. Full settings, selected features, intervals, operational repair,
and artifact hashes are recorded in `EXPERIMENT_LOG.md` Experiment 39.

The final retention cleanup removed 26.607 GiB of redundant checkpoints and
per-epoch predictions while preserving all selected Patience epochs, epoch-20
EMA states, observation references, sidecars, manifests, analyses, and result
summaries. Its exact plan and postchecks are under the program root's
`_cleanup/20260823T024900Z` directory. Paid GH200 instance
`b3eac682796a4e1ea7912422a81f0e85` was terminated after the results were pushed
and was confirmed absent from Lambda's inventory twice.

## Final feature closure and P2 strong-source decision (2026-08-23)

Experiment 40 completed the feature program's final features-only test and the
three independently preregistered P2 screens. The unchanged eight-feature F2
sidecar, now tested without P0.3 pruning, added only `+0.000179` mean IC
standalone and `+0.000264` in the fixed parent stack. The fixed
late-market-momentum/HKS pair added `-0.000474` standalone and `-0.000213` in
the stack, with a materially negative Fold-B result. The feature program is
closed: retain neither candidate and do not search further feature subsets from
these readouts.

B3 lending rates/flows added `-0.000250` standalone and `-0.000051` stacked;
DCE iron ore added `-0.000499` and `-0.000122`. B3 listed-equity option open
interest was positive on all three folds but added only `+0.000586` standalone
and `+0.000400` stacked. None met the frozen `+0.001` mean and non-negative
every-fold gate, so no P2 source is promoted and no post-readout P2 combination
is authorized. Raw Patience-3 on the unchanged parent remains canonical, and
the designated challenger remains informational only.

The previously missing option history was acquired as exact official
BVBG.086/BVBG.028 final reports: 1,154 complete daily pairs from 2019-11-01
through 2024-06-28, preserved as an immutable 15 GB raw archive. Exact
underlying-instrument and dated cash-ISIN mapping yielded a normalized source of
95,045 rows and 142 permanent IDs. The free historical source does not expose
covered/uncovered positions, so those fields were not fabricated. DCE remains
the disclosed contract-specific Sina mirror because the official endpoint was
not freely accessible.

The completed program, all summaries, sidecars, retained training evidence, and
source provenance are under:

    /lambda/nfs/brazil-rv-east3/quant-data/b3/processed/model_runs/experiment40_final_feature_p2_0b6ff68_20260823T075000Z

The program manifest/summary SHA-256 values are
`495e69adf98b190a697c17d620a4ccb04dc02f4f05c13e4ce039ce0caee871ae`
and `db7ba8028f6955b2e753a4ed5272b792fd9f72fdc7325e5970a779d77814331c`.
All 45 trajectories and five three-fold analyses completed at repository commit
`0b6ff68a64276fff53c770b49b1ab9db64120e4b`; official validation and the held-out
test remained sealed. Exact fold deltas, uncertainty, availability contracts,
source hashes, and audit details are in `EXPERIMENT_LOG.md` Experiment 40.

The reviewed retention cleanup removed 36.078 GiB of redundant checkpoints and
per-epoch predictions while preserving all opposite-parity-selected epochs,
epoch-20 raw/EMA states, observation references, sidecars, manifests, histories,
analyses, and summaries. Its exact hash inventory and passing postchecks are
under the program root's `_cleanup/20260823T170000Z` directory; the final program
root is 6.7 GiB.

Paid GH200 instance `95098103c2da4ffcb8e9d10a4ac7704c` was terminated only
after results, cleanup evidence, and GitHub state were secured, then confirmed
absent in two consecutive Lambda inventory reads.

## Incumbent feature-removal decision (2026-08-24)

Experiment 41 produced a non-inferior retrained 34-field specification from the
58 incumbent equity inputs. The selected prune-R2 candidate added
`+0.000898/-0.000216/+0.002978` Raw Patience-3 IC on Fold C/A/B, mean
`+0.001220`; every fold stayed above the fixed `-0.0005` floor. Prune-R1 failed
that floor on Fold A. This is a store-v2 specification for the next rebuilt
parent, not an in-place mutation of the current canonical store/model, and it
does not authorize or alter the official-read lineup.

Remove six dynamic fields: `return_60m_normalized`,
`realized_vol_30m_log_ratio`, `session_range_position`,
`cross_section_return_rank_15m`, `cross_section_volume_rank`, and
`cross_section_volatility_rank_30m`. Remove 18 slow fields:
`overnight_gap_normalized`, `previous_close_to_close_return_normalized`,
`previous_open_to_close_return_normalized`,
`median_daily_real_volume_20d_log_scale`,
`median_daily_dollar_volume_20d_log_scale`,
`daily_dollar_volume_regime_20d`, `observed_fraction_5d`,
`observed_fraction_20d`, `dollar_volume_cross_section_rank`, `beta_to_WIN`,
`beta_to_DI1F27`, `beta_to_DI1F28`, `beta_to_DI1F29`, `beta_to_DI1F31`,
`weekday_sin`, `weekday_cos`, `month_end_proximity`, and
`quarter_end_proximity`. Retain the other 34 incumbent inputs; specifically,
the preview-proposed `volume_surprise`, 15/60-minute market-median returns,
15-minute market breadth, and 60-minute cross-sectional return rank remain
KEEP.

The frozen Stage-A/B root is
`feature_removal_d5b5e1f_20260823T224100Z`; the completed isolated Stage-C root
is `feature_removal_stage_c_repair_d5b5e1f_20260823T232938Z`.
Summary/specification SHA-256 values are
`1070ecfadb99eef42d224b8eacc0ef31fc8e0e08ecc6a7e39aa5153e57fb18b8`
and `08c04de3396fdc31d67b6baeabab1fea80cfd137d55bf2a1aef4ee69d1a34b72`.
The final audit passed with official validation and held-out test sealed. Exact
fold intervals, all 58 rule-attributed verdicts, the operational Stage-C repair,
and retention-cleanup hashes are recorded in `EXPERIMENT_LOG.md` Experiment 41.

A later explicitly authorized object-store cleanup removed 315 unselected
per-epoch prediction archives and 18 redundant tail bundles from the completed
Stage-C repair, totaling 14,309,636,868 bytes (13.327 GiB). The repair retains
the same frozen union of 45 epoch-20/cross-fit/whole-fold Patience predictions
and matching checkpoints, all 18 observation references, every history,
analysis, manifest, summary, verdict, and audit artifact. A fresh full-bucket
inventory found exactly the 333 planned removals, zero unexpected removals,
zero retained-object metadata changes, and the two expected immutable audit
additions. The object store now contains 12,941 objects / 63,474,753,532 bytes
(59.115 GiB). The exact plan and postcheck are under
`model_runs/_retention/storage_cleanup_20260824_round3`.

Paid GH200 instance `e975b774f5834e0fa265d11bbbef680f` was terminated only
after results and audit evidence were pushed, then confirmed absent in two
consecutive provider inventory reads. No paid Lambda instance remains active.

## R3 and full-options decision (2026-08-24)

Experiment 42 rejected all three attempted advances from the selected
Experiment-41 prune-R2 specification. Correlation-conditioned Stage B-prime
froze three further removal candidates: `realized_vol_60m_log_ratio`,
`realized_vol_20d_log_ratio`, and `vol_of_vol_20d`. Their retrained R3 candidate
added `+0.000496/-0.001669/-0.000260` Raw Patience-3 IC on Fold A/B/C (mean
`-0.000478`) and failed non-inferiority. Retain prune-R2's 34 fields as the
store-v2 specification; do not remove these three fields and do not run R4.

The full 14-field options program added
`-0.000410/+0.001526/-0.000073` standalone (mean `+0.000348`). The five-field IV
subset added `-0.001134/-0.002320/-0.000563` (mean `-0.001339`). Neither passed
the frozen fold/mean/uncertainty gate. Their predeclared mixed states also
failed, so the options family is parked and no candidate is registered for the
future official-read cycle. Do not search a third options subset from these
readouts.

The causal full-options source is now durable: 95,045 rows, 142 permanent IDs,
2021-07-19 through 2024-06-28, output SHA-256
`6a0cff033fb48a3b190ba49389e173c385ee1df0211a335e81665e9ec2af5686`.
The manifest-declared unpublished 2023-12-08 instrument master remains invalid;
no adjacent-date master was substituted. The completed program is:

    /lambda/nfs/brazil-rv-east3/quant-data/b3/processed/model_runs/r3_options_9a05b1d_20260824T053052Z

Program/R3/options/source-manifest SHA-256 values are
`888a1cef6c1488365db3870aa434127767c36ee544a82b8deed58dab7382f91d`,
`a058efc42b66be5117b13c044c774cf531ba57df8294823121bfd3877d6df452`,
`118746188fe2cc5d61c9f7dfad7a7a68173c433b6ff11979c957907d4cf4dafb`,
and `d03afc6c75fff445bac4d57df09b72d373f459e1edc08d4e15af679c20711509`.
All 27 trajectories completed; official validation and held-out test remained
sealed. Exact gates, fold analyses, source diagnostics, the pre-score
unpublished-master repair, and cleanup evidence are in `EXPERIMENT_LOG.md`
Experiment 42.

The reviewed cleanup removed 467 redundant checkpoints / 2,113,294,685 bytes
while preserving all 540 prediction archives, 73 selected/final checkpoints,
every analysis, manifest, source artifact, and sealed-data record. Its passing
postcheck is under the program root's `_cleanup/20260824T121000Z` directory.

Paid GH200 instance `e2cf2e517d9541ac93cac3906fc5c0e4` was terminated only
after the results, cleanup evidence, and documentation commit were secured. It
was absent from two consecutive provider inventory reads, and the Lambda
account then had zero active instances.

## Official-read deployment decision (2026-08-24)

Experiment 43 consumed the third and sole authorized official-validation read
for this lineup. Canonical parent-3 Raw Patience-3 scored `0.041639843`. The
stored residual challenger added `+0.000453978`, and the retrained Experiment-41
store-v2 mask added `+0.001595530`, but their paired block-10 95% intervals had
lower bounds of `-0.000684798` and `-0.000294105`. Neither arm passed the frozen
support rule. Canonical parent-3 therefore remains deployed; no 10-seed
expansion was run, and neither official retuning nor another read is authorized.

The completed program is:

    /lambda/nfs/brazil-rv-east3/quant-data/b3/processed/model_runs/official_read_c04ea91_20260824T140900Z

All 63 newly generated official prediction archives are retained. The reviewed
cleanup removed only 60 non-deployed store-v2 checkpoints / 270,571,140 bytes;
its plan and passing postcheck are under the program root's
`_cleanup/20260824T153000Z` directory. The validation ledger records event 3,
and every artifact records `test_accessed=false`. The held-out test remains the
final sealed lockbox.
