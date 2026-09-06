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
  execution proxy. Terminal inventory uses last marking in the headline ledger and the
  registered haircut sensitivity beside it. Unresolved inventory notional and counts
  are mandatory reporting fields, not silently discarded observations.
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

Before Round 1 can freeze, a completed `BRAZIL_RV_V2_PIPELINE_VALIDATION_V4` report
must bind the exact implementation and store and have status
`development_grade_inferred_actions`, no failed reasons, and both sealed-window access
flags false. It includes the survivorship gates, provider-invariance evidence, finite
and target-coverage audits, and a 20-name by 20-session independent native-fast audit.

The following engineering bounds are fixed in advance and stop the program if violated;
they are not strategy conclusions:

1. every naive signal has absolute pooled daily IC below 0.10;
2. reversal-5 uses its registered negative structural sign (the raw five-session return
   is multiplied by -1; its realized IC is not forced to have either sign);
3. deployed gross is within 10% of the 2.0 target in every fold/evaluation;
4. mean absolute unresolved terminal inventory notional is below 2% of NAV.

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
    "F1": {"start": "2023-07-03", "end": "2023-12-29"},
    "F2": {"start": "2024-01-02", "end": "2024-06-28"},
    "F3": {"start": "2024-07-01", "end": "2024-12-30"}
  },
  "cross_fit": "none",
  "models_per_fold_seed": 1,
  "primary_population_rule": {
    "target": "median_adjusted_volatility_scaled_midrank",
    "horizons_sessions": [1, 2, 3, 5],
    "requirements": [
      "active_at_entry",
      "finite_target_scale_sigma_greater_than_1e-8",
      "valid_and_finite_scaled_target_on_every_primary_horizon",
      "valid_and_finite_score_on_every_primary_horizon"
    ],
    "minimum_cross_section_names": 20,
    "per_horizon_metric": "tie_aware_spearman",
    "daily_aggregation": "equal_mean_of_all_primary_horizons_when_all_defined"
  },
  "headline_cell": {
    "signal": "tie_aware_rank_average_D1_D2_D3_D5",
    "signal_horizons_sessions": [1, 2, 3, 5],
    "ledger": {
      "k_per_side": 30,
      "buffer_per_side": 30,
      "gross_target": 2.0,
      "planned_gross_cap": 2.25,
      "planned_absolute_net_cap": 0.2,
      "planned_name_weight_cap": 0.05,
      "cost_bps_per_side": 4.0,
      "annual_borrow_rate": 0.02,
      "annual_debit_spread": 0.0,
      "short_proceeds_remuneration": 0.0,
      "initial_capital_brl": 1.0,
      "lot_size": null,
      "entry_expiry_sessions": 3,
      "forced_liquidation_haircut": 0.0,
      "max_missing_sessions": 10,
      "annual_sessions": 252
    }
  },
  "source_tier_labels": {
    "action_terms_source": "inferred_cotahist_dismes_v1",
    "schedule_source": "reconstructed_v1"
  }
}
```
<!-- BRAZIL_RV_V2_PROTOCOL_JSON_END -->
