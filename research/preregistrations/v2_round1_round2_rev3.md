# v2 research registration rev 2 — development-grade Round 1 and Round 2

This registration replaces the voided first registration. It must be committed and
hash-bound before any candidate score is computed. It authorizes development-window
research only. It does not authorize the official-validation or permanently spent test
windows, deployment, or claims based on verified corporate actions, auction marks, or
executable borrow.

## Bound source tier and common protocol

- Every input store, checkpoint, score, evaluation, ledger, result, and registration
  record must carry `action_terms_source=inferred_cotahist_dismes_v1` and
  `schedule_source=reconstructed_v1`. These are development-grade labels, never
  synonyms for verified inputs.
- The exact real-store manifest and the completed Section-C acceptance report are
  SHA-256-bound in the Round-1 frozen design. The store must have been built by the
  exact clean implementation used to freeze this registration.
- Folds F1/F2/F3 are exactly those in `brazil_rv.v2.splits`: chronological fit,
  10 purge sessions, 55 selection sessions, 10 purge sessions, then one continuous
  evaluation window. There is no cross-fit: each fold/seed has one selected model.
  The sealed 2025/2026 windows remain untouched.
- Daily row e is available only at decision e+1. Current-session intraday inputs retain
  the same-session decision clock and use only the open-gap boundary diagnostic.
- Readouts for every candidate, per fold and pooled: median-adjusted,
  volatility-scaled primary IC over D in {1,2,3,5}, per-horizon IC including D10,
  raw-return Rank-IC, persistence
  at 1 and 5 sessions, decile spread in bps per holding session, and the swing-economics
  grid with 4 bps per side / 2% annual borrow as the headline cell. Daily series use a
  20-session moving-block bootstrap with 10,000 replications; blocks are sampled within
  folds and paired comparisons use the exact common population.
- Economic outputs are development-grade under inferred action terms and a close-price
  execution proxy. A held name with ten consecutive sessions without a print is settled
  at its last mark under the labelled `last_mark_after_10_sessions` convention; a
  parallel 30% adverse-price settlement scenario is reported but never headline.
  Unresolved inventory and settlement incidence are mandatory reporting fields, not
  silently discarded observations.
- The planned absolute-net cap is 20%. With 30 independently filled slots on each
  side, the former 10% cap treated ordinary asynchronous entry/exit imbalance as a
  liquidation event and prevented the light side from refilling. The wider cap remains
  a binding directional-risk limit; gross 2.25 and per-name 5% caps are unchanged, and
  breaches are corrected with partial lowest-conviction risk trims rather than a
  whole-book liquidation.
- Seeds are 11/29/47 for networks and the five fixed GBDT seeds in the implementation.
  CPU is used for Round 1. Round 2 uses at most one paid GH200 and 4–6 concurrent
  registered trajectories after its one-job smoke succeeds.

## Pre-research acceptance and sanity bounds

Before Round 1 can freeze, a completed `BRAZIL_RV_V2_PIPELINE_VALIDATION_V8` report
must bind the exact implementation and store and have status
`development_grade_inferred_actions`, no failed reasons, and both sealed-window access
flags false. It includes the survivorship gates, provider-invariance evidence, finite
and target-coverage audits, and a 20-name by 20-session independent native-fast audit.

The following engineering bounds are fixed in advance and stop the program if violated;
they are not strategy conclusions:

1. every naive signal has absolute pooled daily IC below 0.10;
2. reversal-5 uses its registered negative structural sign (the raw five-session return
   is multiplied by -1; its realized IC is not forced to have either sign);
3. deployed gross remains targeted at 2.0 and the 1.8--2.2 band is reported.
   A book outside that band is labelled `gross_underdeployed` or
   `gross_overdeployed`, rather than failed, when all entry-defect signatures
   are zero and mean gross is within the hard interval 1.5--2.25. The label,
   decomposition, and signatures accompany economics at achieved gross. A
   signature breach or mean gross outside the hard interval stops the program;
4. each evaluation's mean daily unresolved-or-stale inventory notional is below
   2% of NAV. Terminal unresolved notional and counts remain mandatory diagnostics
   but are not averaged into this window-utilization gate.

This fourth engineering bound was refined at `2026-09-06T20:11:20Z`, before a
post-refactor acceptance result. The sealed pass-4b replay showed that 19 of 22
terminal positions and 88.90% of terminal unresolved notional had no print on
the mechanically chosen last evaluation session. All 22 were already under
long-lived pending exits tied to unresolved inferred-action cells. A single
terminal wealth-index observation therefore measured calendar-end liquidation
luck, not ordinary ledger deployability. The replacement keeps the 2% bound
and applies it to every day's unresolved-or-stale marked inventory, averaged
within each evaluation; terminal count/notional and the nonexclusive reason
breakdown remain reported. No target, score, fold, portfolio cap, action term,
or accounting rule changes with this engineering-only refinement.

The ledger interpretation was further fixed at `2026-09-06T22:28:39Z`, before
the pass-4e ledger replay and before any Round-1 score. The sealed diagnostic at
`D:\quant-data\b3\processed\model_runs\v2_pass4e_unresolved_diagnostic_24abe56_20260906T222757Z`
(diagnostic-manifest SHA-256
`2801ea826df263525aec0247c9eeefcdd5975237e73dcab4807d20c68e623310`,
log-inclusive inventory SHA-256
`0f93b5e660c7701cfec392b3a92ff25fdb7eaccd212c270deffa91c0d07ed709`)
found 2,502 held unresolved-or-stale name-days across the 16 registered
development evaluations. Every one carried the ledger's latched unresolved flag;
84 (3.3573%) had originated on a session followed by a later print. There were no
held name-days on which the retrospective action mask was false despite a current
observed print, and the stored retrospective mask differed from the decision-known
feature mask as expected. Consequently no store rebuild or coverage reinterpretation
is authorized.

For the replay, action uncertainty is a current-session accounting state and never a
position-life latch. It blocks only new entries. Any observed positive close can fill
an exit, risk reduction, or terminal liquidation; a later resolved cell clears the
uncertainty state without requiring a fill. Stale-mark notional and current unresolved-
claim inventory notional are reported separately as well as through their unchanged
union gate. The evaluator accepts only the store's explicitly labelled retrospective
outcome/accounting arrays; decision-known action alignment remains feature-only. The
sealed baseline and GBDT score panels are reused without model or score recomputation,
and every non-ledger evaluation field must remain bit-identical. The advance expectation
is that every unresolved-or-stale mean falls below 2% and the three prior borderline
gross books return to 1.8--2.2. If a low-turnover book remains just below 1.8 without
frozen inventory, the program stops and reports daily slot occupancy; the bound is not
changed.

For continuity only, momentum's pooled IC moved from 0.056 on the prior 34-name liquid
subset to 0.040 on the complete population. Neither value is a research result.

The terminal-settlement convention was registered at `2026-09-06T23:30Z`, before
the pass-4f ledger replay and before any Round-1 score. The sealed diagnosis at
`D:\quant-data\b3\processed\model_runs\v2_pass4f_stale_diagnostic_64c5b76_20260906T232352Z`
(diagnostic-manifest SHA-256
`d6522ec1c20cf294fd8914e951c3d51f10bd662f7cf02d8d76b8a9e5f3ac391c`,
log-inclusive inventory SHA-256
`7d28de475e529cbd7aba7dccbac1dc14f1a05941d663a25b178570fe9cf794ee`)
reconstructed 37 stale holding episodes across 14 ISINs and 2,490 stale name-days.
No affected position printed again inside its evaluation holding window, so the
predeclared fill-defect stop did not fire. Twenty-seven positions never printed again
anywhere in the store; ten printed only outside the relevant holding window. None was
an ISIN-succession candidate. The deterministic flat/premium path diagnostic classified
zero as tender-like, so the data support terminal disappearance but do not support the
stronger proposed tender-offer attribution.

The headline convention settles long or short inventory at its last mark, with the
ordinary per-side cost, on the tenth consecutive no-print session and permanently
releases its slot. The prior position is never reopened if a later print appears; that
event is counted. The parallel scenario settles longs 30% below and shorts 30% above
the same mark. Cumulative settlement notional above 15% of contemporaneous NAV across
an evaluation labels that book `economics_unresolved`. The unchanged 2% daily gate
counts stale inventory during the grace period plus inventory exposed to an explicit
unresolved action term; an unobserved session alone is not an unresolved claim. The
advance expectation is that all 16 books pass the unchanged gross and stale-inventory
bounds. On failure, the program stops without Round 1 or Round 2.

Two score-free replay invocations at implementation commit `dab7c53` stopped before
an output root was created and before a score panel was loaded. The first supplied the
log-inclusive artifact-inventory hash where the CLI requires the sealed runner
inventory hash (stderr SHA-256
`c1f1e7e9fea4327e32090deb2c6a404ecebb767ba61a13852eaf88751e275baf`).
The corrected binding then exposed a replay-chain provenance check that compared the
prior replay's code commit with the store-build commit, although the manifest already
records those as distinct identities (stderr SHA-256
`ddc717b6e15081499c2de042a132986849fb19a4c72741fedbce8f870bf0fea4`).
The bounded repair verifies `store_build_implementation_commit` against the sealed
store and continues to bind the prior replay itself by its manifest and inventory
hashes. It changes no score, panel, ledger rule, threshold, or gate.

That repair then reached the first score-panel path but stopped before loading its
arrays or evaluating it: a chained replay's panel remains inside the original sealed
acceptance root, not the immediately prior replay root. The empty failed root
`D:\quant-data\b3\processed\model_runs\v2_development_acceptance_pass4f_b4715da_20260906T234950Z`
is retained. The follow-up provenance repair accepts that path only after verifying
the ancestor root named in the already hash-bound prior manifest, including the
ancestor manifest hash, complete inventory hash and every inventory row, store hash,
implementation identity, access flags, and score-manifest hash. Arbitrary external
score paths remain forbidden. This changes no score, panel, ledger rule, threshold,
or gate.

## Pass-4g gross-band disposition registered before diagnosis

This disposition was fixed at `2026-09-07T00:55:50Z`, before the pass-4g
occupancy diagnostic and before any Round-1 score. It replaces the gross band
as a standalone defect detector without relaxing the portfolio target or any
leakage, identity, clock, accounting, stale-inventory, or access rule.

For each evaluation session, with `K = 30`, end-of-day holdings, and
`gross_t = gross_fraction_nav[t]`, the diagnostic must report the exact identity

```text
gross_target - gross_t
  = 2 * (K - K_eff,t) / K
  + (2 * K_eff,t - n_held,t) / K
  + sum_held(1 / K - abs(shares * mark_t) / NAV_t).
```

The terms are labelled `small_universe`, `occupancy`, and `sizing`. Occupancy
is split into pending-entry, band-exhausted, blocked, same-day exit-gap, and
other terms. Sizing is split using each current holding episode's entry fill
cost basis and first submission NAV:

```text
1 / K - abs(shares * mark_t) / NAV_t
  = (submission_nav / K - fill_value) / NAV_t
  + (fill_value - abs(shares * mark_t)) / NAV_t
  + (1 / K) * (NAV_t - submission_nav) / NAV_t.
```

The session means of all terms must sum to `gross_target - mean_gross` within
`1e-9`. The registered entry-defect signatures and limits are:

- D1, a printed and unblocked pending entry left unfilled: exactly zero;
- D2, an entry fill shorter than its pending quantity when fill fractions are
  one: exactly zero;
- D3, an open slot left with unconsumed candidates: exactly zero;
- D4, the sum of gross-cap blocks below target, fresh-entry name-cap blocks,
  and net-cap blocks while the book was balanced: exactly zero;
- D5, ineligible exit instructions whose name becomes eligible again within
  three sessions inside its side's retention band, divided by all non-terminal
  exit instructions: at most `0.10`.

The disposition is read from the F2 inverse-volatility-20 decomposition after
all 16 sealed books reproduce every existing pass-4f headline field exactly:

- P0: any D1--D4 breach or D5 above `0.10` in any book stops the program with
  no rule change, merge, or Round-1 freeze.
- P1: with no signature breach and F2 inverse-volatility occupancy share at
  least `0.30`, add only the registered causal entry-liquidity screen and
  15-rank per-side entry reach-down, then replay all 16 sealed score panels.
- P2: with no signature breach and occupancy share below `0.30`, change no
  ledger rule; apply the labelled gross-deployment acceptance rule and replay
  the sealed panels only to carry the new report fields.

Under P1, an entry candidate on session t must have printed on immediately
preceding reconstructed session t-1, and the entry band may reach down by at
most 15 ranks per side while retaining exactly K_eff slots and the unchanged
retention exit edge. All other sizing, expiry, caps, trims, settlement, costs,
borrow, stale-inventory, same-close reuse, and K_eff rules remain unchanged.
The advance replay expectations are: all 16 books in the 1.8--2.2 band; mean
turnover and the holding-session approximation change by less than 10% for
each book; settlement counts do not rise; headline net excess may move in
either direction and has no decision weight. Any residual underdeployment is
labelled rather than prompting another rule change.

Under either P1 or P2, deployed gross remains targeted at 2.0 and the 1.8--2.2
band remains reported. A book outside it is labelled `gross_underdeployed` or
`gross_overdeployed`, rather than failed, only when D1--D5 remain within their
limits and mean gross is in the hard interval 1.5--2.25. The label,
decomposition, and signatures accompany its economics at achieved gross.
Any signature breach or mean gross outside 1.5--2.25 stops the program. IC,
persistence, and spread readouts remain valid for labelled books. Existing F3
`economics_unresolved` labels remain unchanged.

## Pass-4h transient-eligibility hold registered before replay

This rule and its advance expectations were fixed at `2026-09-07T01:53:41Z`,
before the pass-4h ledger replay and before any Round-1 score. Pass 4g established
that D1--D4 were zero in all 16 books and that F2 inverse-volatility's gross
shortfall was sizing-dominated: occupancy explained `0.059435`, while mark drift
was the largest sizing term at `0.107124`. Its low gross is therefore carried as
`gross_underdeployed`, not repaired. The pass-4g P0 signature arose only because
held names were exited on a transient loss of eligibility and often returned to
their side's retention band.

The headline ledger now sets `ineligible_hold_sessions=5`. A held name that is
ineligible remains held while its retention width is positive, its prior
no-print streak remains below the ten-session settlement grace, and its
ineligible streak is at most five sessions. Eligibility resets the streak to
zero. The position also resets it on close or settlement. On session six of
continuous ineligibility the ordinary exit is instructed with cause
`ineligible_hold_exhausted`; the next print remains the only possible market
fill. An ineligible name with no print still follows the unchanged Pass-4f
settlement path. Entries remain eligible-only. K, buffer, caps, expiry, sizing,
trims, same-close reuse, settlement, costs, and borrow do not change. Setting
`ineligible_hold_sessions=0` reproduces the pre-Pass-4h ledger path.

The current D5 signature is `D5_ineligible_exit_within_hold_window`: an
`ineligible_hold_exhausted` instruction issued with streak at most five. It must
be exactly zero, as must D1--D4. The former data diagnostic is retained only as
the threshold-free report
`ineligible_exit_reeligible_within_10_sessions_share`, applied to
`ineligible_hold_exhausted` exits with a ten-session look-ahead. Counts of held
ineligible sessions attributable to `score_valid=false`, `membership=false`,
and a non-finite score are also reported without thresholds.

The registered gross-deployment acceptance from Pass 4g now applies: the
1.8--2.2 band produces `within_band`, `gross_underdeployed`, or
`gross_overdeployed`; only a D1--D5 signature breach or mean gross outside
1.5--2.25 is a gross stop. IC, persistence, and spread readouts remain intact,
and economics remain reported at achieved gross. The independent per-evaluation
2% stale/unresolved bound and 15% settlement-incidence label remain unchanged.

Advance replay expectations, recorded without decision weight, are: D1--D4 and
the new D5 invariant are zero in all 16 books; `ineligible_hold_exhausted` exits
fall well below the Pass-4g `ineligible` counts in every book; mean turnover falls
for F3 inverse-volatility and F3 momentum; mean gross is unchanged or slightly
higher in every book and moves by no more than 0.05; settlement counts and stale
means are unchanged or lower; and net excess moves modestly in either direction
for the highest-flicker books. Missing an expectation is reported but does not
stop. Any signature breach, hard-gross breach, stale-bound breach, non-ledger
replay change, protected-window need, or paid-instance appearance stops before
merge or Round 1.

## Round 1 — baseline floor and GBDT parent (CPU)

**R1.1 baseline table.** Evaluate exactly reversal 5, reversal 21, momentum 12-1,
the equal rank-gauss reversal-5/momentum-12-1 blend, and inverse-volatility-20 on
F1–F3 with the complete registered readouts and ledger.

**R1.2 cumulative GBDT ladder.** Fit five-seed ensembles with early stopping on the
complete selection window:

1. `a_slow`: the slow feature library;
2. `b_intraday`: A plus intraday-derived daily features and availability flags;
3. `c_lending`: B plus lending;
4. `d_all_sidecars`: C plus oddlot, options, rebalance, events, and fundamentals.

Report paired rung-minus-previous deltas and gain/SHAP importance. A rung is dropped
only if both its pooled primary-IC point delta and headline-net-excess point delta are
negative. Undefined economics cannot establish improvement or worsening. Other rungs
are kept; intervals crossing zero are marked ambiguous. The parent is the kept rung
with the largest pooled primary IC, then defined headline economics, then the earlier
rung on an exact tie.

**R1.3 data-span preview.** On the selected parent feature set, compare fine-only,
pretrain-plus-fine with uniform weights, and pretrain-plus-fine with a 756-session
half-life. This preview is informational input to R2.1 and has no decision weight.

Every GBDT row is the canonical decision row. Pretraining rows have no intraday fields
and set `fast_present=0`; fine rows use the same canonical decision-row contract with
current-decision intraday fields. No additional consumer-side lag is applied.

## Round 2 — network data-span arms and parent comparison (GPU)

Before any registered Round-2 work, run exactly one disposable engineering smoke:
Arm-A shape, F1, seed 11, one epoch, no score artifact. Stop at its first failure. The
smoke checkpoint and history are never reused by a registered trajectory. Only a fully
finite completed smoke permits the Stage-P plan to be written.

**R2.1 data-span arms.** Use the starter network on the Round-1 parent feature set,
fresh native fast weights for every arm, lambda-persistence 0, maximum 20 epochs,
patience 3, lookback 60, archived raw-Patience and final-EMA states, seeds 11/29/47,
and F1/F2/F3:

- Arm A: Stage F from scratch on the fine window.
- Arm B: Stage P once per seed on the pretrain window, then Stage F using the selected
  raw-Patience checkpoint and the 0.3 pretrained-parameter learning-rate multiplier.
- Arm C: Stage J over both windows with 756-session time decay.

The three seed panels on each continuous evaluation window are tie-aware rank-averaged.
Report B-A and C-A paired deltas. A long-history
arm is eligible only when its headline economics are not below Arm A. If both long-arm
IC improvements are at most 0.002 and both intervals include zero, select A. Otherwise
select the eligible arm with the largest pooled IC; exact ties prefer A, B, then C.

**R2.2 network, GBDT, ensemble.** Compare the selected network arm, the Round-1 GBDT
parent, and their fixed equal-weight rank-average ensemble on every readout. Select the
largest pooled-IC result, use defined economics as tie-break, then prefer network, GBDT,
ensemble on an exact tie. The designation is only the development parent for a later
separately registered round.

## Immutability and stop rules

Round 1 and Round 2 each use a fresh immutable root with frozen design, exact paths and
hashes, code identity, source-tier labels, access flags, operational logs, and a complete
inventory. No candidate, seed, fold, feature group, readout, execution parameter, or
preference rule may be added after a score. A completed candidate is never retried.
Score-free operational failures may be repaired only in a new clean commit and fresh
root when doing so does not change this contract. Any gate or first-smoke failure stops
with evidence. The paid instance is terminated and its exact ID verified absent twice
after artifacts and logs are secured. No deployment changes occur.

## Machine-readable protocol

## Rev-3 registered amendment: neutral evaluation and executable borrow

This revision preserves the rev-2 protocol above except where this amendment explicitly
supersedes its measurement target, borrow sensitivity, rung-D evaluation support, and
batch-shape implementation. The sealed rev-2 Round-1 root remains engineering evidence
but is void for research decisions and its IC values are not research results.

The rev-2 readout showed that the learned score was primarily a low-volatility,
low-beta, momentum characteristic tilt; inverse volatility exceeded every learned rung.
The nominally dollar-neutral books had realized market betas near -0.7 to -1.0, and the
uniform 2% borrow assumption did not describe the high-volatility short leg. Oddlot
coverage was absent in F3, so rung D cannot be interpreted there.

For every date and horizon in {1,2,3,5,10}, the registered target population requires
the legacy scaled target and all three decision-row characteristics to be finite.
Let y be the clipped median-adjusted simple shareholder return divided by lagged
Yang-Zhang volatility and sqrt(H). Regress y with an intercept on the already
rank-Gaussianized decision-row values of yang_zhang_vol_20, beta_60, and
log_volume_mean_20. The primary target is the tie-aware cross-sectional midrank of the
OLS residual. Minimum support is 20. This deterministic virtual view is
target_primary_neutral/target_primary_neutral_valid; the physical legacy target remains
reported only as legacy_scaled_target_ic.

Every headline book must report realized daily beta from an OLS regression of ledger
daily return on the equal-weight active-universe daily shareholder return. Absolute
slope above 0.30 is labelled directional.

The registered uniform 4 bps/2% borrow cell remains the headline. A separate
cost_4_borrow_lending_v1 cell uses the D+1-safe lending sidecar: a name is shortable when
a rate or balance was observed in the prior 20 sessions, new short entries in other
names are blocked, and held shorts pay their observed annual rate floored at 2%.
Exits are never blocked. Coverage is reported for all active name-days and the
highest-volatility quartile.

Rung D is evaluated only on F1/F2. F3 is source_unsupported because the oddlot archive
ends before that fold, and D-minus-C uses only the common F1/F2 population. Inverse
volatility is a null control and must have absolute neutral IC below 0.02 in each fold.

Round-2 batching pads the compact name axis to the maximum active count over the stage,
rounded to a multiple of 16, with all padded masks false. This is an engineering-only
shape invariant; it does not change real-name eager outputs or losses. Round 2 is not
authorized by this registration pass.

## Machine-readable protocol

<!-- BRAZIL_RV_V2_PROTOCOL_JSON_BEGIN -->
```json
{
  "schema": "BRAZIL_RV_V2_REGISTRATION_PROTOCOL_V1",
  "purge_sessions": {
    "fit_to_selection": 10,
    "selection_to_evaluation": 10
  },
  "selection_sessions": 55,
  "evaluation_window_per_fold": {
    "F1": {
      "start": "2023-07-03",
      "end": "2023-12-29"
    },
    "F2": {
      "start": "2024-01-02",
      "end": "2024-06-28"
    },
    "F3": {
      "start": "2024-07-01",
      "end": "2024-12-30"
    }
  },
  "cross_fit": "none",
  "models_per_fold_seed": 1,
  "primary_population_rule": {
    "target": "target_primary_neutral",
    "horizons_sessions": [
      1,
      2,
      3,
      5
    ],
    "requirements": [
      "active_at_entry",
      "finite_target_scale_sigma_greater_than_1e-8",
      "valid_and_finite_neutral_target_on_every_primary_horizon",
      "valid_and_finite_neutralization_characteristics",
      "valid_and_finite_score_on_every_primary_horizon"
    ],
    "minimum_cross_section_names": 20,
    "per_horizon_metric": "tie_aware_spearman",
    "daily_aggregation": "equal_mean_of_all_primary_horizons_when_all_defined"
  },
  "target_neutralization": {
    "target_value_array": "target_primary_neutral",
    "target_validity_array": "target_primary_neutral_valid",
    "characteristics": [
      "yang_zhang_vol_20",
      "beta_60",
      "log_volume_mean_20"
    ],
    "method": "rank_gauss_ols_with_intercept",
    "clip": 5,
    "minimum_names": 20
  },
  "headline_cell": {
    "signal": "tie_aware_rank_average_D1_D2_D3_D5",
    "signal_horizons_sessions": [
      1,
      2,
      3,
      5
    ],
    "ledger": {
      "k_per_side": 30,
      "buffer_per_side": 30,
      "gross_target": 2,
      "planned_gross_cap": 2.25,
      "planned_absolute_net_cap": 0.2,
      "planned_name_weight_cap": 0.05,
      "cost_bps_per_side": 4,
      "annual_borrow_rate": 0.02,
      "borrow_source": "uniform",
      "annual_debit_spread": 0,
      "short_proceeds_remuneration": 0,
      "initial_capital_brl": 1,
      "lot_size": null,
      "entry_expiry_sessions": 3,
      "ineligible_hold_sessions": 5,
      "settlement_grace_sessions": 10,
      "settlement_haircut": 0.3,
      "settlement_economics_unresolved_fraction_nav": 0.15,
      "annual_sessions": 252
    }
  },
  "borrow_cells": [
    "uniform",
    "lending_sidecar_v1"
  ],
  "realized_beta_label_threshold": 0.3,
  "rung_d_evaluation_folds": [
    "F1",
    "F2"
  ],
  "inverse_volatility_neutral_ic_absolute_bound": 0.02,
  "source_tier_labels": {
    "action_terms_source": "inferred_cotahist_dismes_v1",
    "schedule_source": "reconstructed_v1"
  }
}
```
<!-- BRAZIL_RV_V2_PROTOCOL_JSON_END -->
