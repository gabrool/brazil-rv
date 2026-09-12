# Round 6 experiment review

This document is intended to be read independently by an LLM reviewing the
Brazil-RV research program. **Research decision, 2026-09-12: retain S0 as the
working comparator; the new parent choice is inconclusive.** Fundamentals leads
the full three-seed panel and two omissions, but C6 leads the remaining omission.
Neither a candidate gain over S0 nor a seed-stable replacement is established.
The continuous-book readout is complete. All required research artifacts are
recovered and verified locally and on persistent storage.

## Research question and frozen contract

Round 6 asks whether additional information improves the existing multi-day
network, and whether its temporal architecture and fine-tuning recipe are useful.
The model consumes a slow historical sequence and predicts five daily horizons
from the 15:45 decision snapshot. This is offline research, not a live execution
system. There is no forward capture and no 2025/2026 evaluation in this round.

The comparison uses fourteen chronological development folds and seeds 11, 29,
47. New families enter through validity-gated, zero-initialized linear residual
projections before the shared trunk. Invalid families leave the parent forward
pass unchanged. The residual trunk can learn nonlinear interactions; this round
does not introduce attention across securities. Most arms reuse compatible
sealed parent pretraining (Stage P); each fold is subsequently fine-tuned (Stage F).

The concrete parent is a one-layer, width-64 GRU on 60 sessions of 32 slow fields,
with a width-128 fusion representation, two residual SwiGLU trunk blocks (inner
width 48), dropout 0.1 and five equally weighted horizon losses. Fast/intraday
input is disabled, persistence-loss weight is zero, soft-rank temperature is
0.5, and checkpoint selection uses D3/D5/D10. The MLP control replaces the GRU
with the registered latest-row residual MLP. Fundamentals adds 12 decision-row
fields; magnitudes adds four. Architecture contracts are recorded in every fit.

Session 1 tests magnitudes, odd-lot, options, events, fundamentals, lending,
fine-tuning multiplier 1.0, time decay 756, and a residual MLP comparator against
S0. Session 2 tests cross-market, sector, microstructure, rebalance, the registered
positive-family combination C6, and fresh-pretraining variants of C6 and the best
single family. Following the compute amendment, S0 was refitted alongside all
candidates: 17 arms × 14 folds × 3 seeds = 714 F fits, plus six fresh P fits.
The MLP's three compatible architecture-specific P fits were retained separately.

## Data changes before neural outcomes

The accepted store contains 3,717 sessions through 2024-12-30 and 933 historical
permanent security identities. All 94 protected arrays, both axes and 36 tables
match the accepted Round-5 input. Only cross-market, lending and rebalance were
replaced. Source archives remain immutable.

* Cross-market admits Brent at the next B3 decision, same-endpoint BRL ADR/EWZ
  gaps with dated identity, and published foreign-flow total differences and
  interactions. There were 496 recovered participation tables from 687 bounded
  archive requests. Date-only bulletins enter the next decision without a second
  lag; missing observations remain missing.
* Index releases comprise 44 events, including 25 recovered 2018–2022 releases.
  Missing effective cycles are not silently bridged.
* Lending adds 6,483 positive latest-vintage observations with 1,535 exact
  overlaps. The vintage limitation is explicit.
* Nineteen fundamental/event fields received pooled and annual sign diagnostics.
  Thirteen targeted TTM, unit, receipt and valuation checks passed. Adverse-era
  profitability/size signs were retained rather than changed to improve results.
* The historical borrow-rate search found four isolated 2016–2017 reference
  days, not a usable continuous first-vintage archive. These were not admitted
  to training.

Informative folds are determined by input support, not outcomes: lending F6–F14,
sector F2–F14, rebalance F3–F10/F12–F14, and all fourteen for other families.
See [preflight evidence](v2_ROUND6_PREFLIGHT.md) and
[input manifest](v2_round6_inputs.json) for source clocks and acceptance details.

## Mid-experiment efficiency amendment

The user authorized correcting three sources of wasted computation and examining
the epoch ceiling. The revised comparison uses a fresh root so the old and new
training recipes are not mixed in candidate-versus-S0 claims.

The 933 identities are a historical union, not a simultaneous universe. The
maximum point-in-time active count is 243. Stage-specific axes now pack every
active security and round the required width to a multiple of 16. They retain
full 60-session history and scatter scoring back to the permanent identity axis.
No eligibility requirement was tightened to obtain the speedup. Eligibility uses
prior-session history: at least 15 traded sessions out of 20, median traded value
at least R$2m, prior close at least R$1, and at least 60 sessions of history.
The data audit found no target-valid inactive security cells.

Unique-date batches visit each eligible date once per epoch, including remainder
batches. The prior adjacent-date construction duplicated visits even when the
objective did not require them. Positive persistence objectives retain required
adjacent pairs. Time-decay weights retain global fit-population normalization.

BF16 autocast is used for suitable network operations; losses, moments and
optimizer arithmetic retain FP32. Independent head ranking losses are vectorized
and both SAM passes share transferred batch inputs. The epoch cap increases to
60, validation occurs every two epochs, and patience is three validation checks.
Approximately 40 new epochs match the old 20 paired-visit epochs in update count.

In 102 earlier completed F fits, 20 hit the old cap and 15 selected epoch 20.
A nine-fit engineering bridge passed: mean IC change +0.002518345, with no fit
hitting the revised 60-epoch cap. One F12/seed-11 fit selected epoch 44 and improved
IC by 0.001267 over its best result through epoch 40. This supports providing
enough training budget; it is not an independent model-selection experiment.
The measured GPU update speedup was 2.33× over dense FP32; this is not an end-to-end
wall-clock claim. CPU packing improved approximately 2× with bit-exact outputs.
See [compute validation](v2_ROUND6_COMPUTE_VALIDATION.md).

The 102 old F fits were recovered and retained as historical evidence, but
excluded from the revised comparison. Compatible S0 and MLP P weights were
reused. After a PyTorch 2.13 partitioned CUDA-graph failure, three exact failed
configurations passed score-free recovery smokes with
`TORCHINDUCTOR_GRAPH_PARTITION=0`. The 49 already-completed revised F fits were
retained. A later planner correction sorted sidecar field metadata to match the
trainer's canonical ordering; all nine Session-2 smokes were retained and the
frozen neural implementation was unchanged. Failure evidence was preserved.
See [runtime recovery](v2_ROUND6_RUNTIME_RECOVERY.md).

## Available results and decision limits

All 714 F fits and six fresh P fits completed at 16:49:14 UTC on September 12.
All training trajectories have been recovered and hash-verified locally. Both
session summaries, all required inference ablations, and three borrow-cost
scenarios are complete and mirrored to persistent storage. Original-rate cost
replay headlines match exactly; no cost-sensitivity gates failed.

Session-1 informative-fold paired primary-IC point changes against S0 were:

| Family | Paired IC change |
| --- | ---: |
| Fundamentals | +0.002571351 |
| Magnitudes | +0.000779266 |
| Odd-lot | -0.000015207 |
| Events | -0.000176754 |
| Lending | -0.000190117 |
| Options | -0.002294917 |

The fixed rule therefore formed C6 from fundamentals and magnitudes, and selected
fundamentals for the best-single fresh-P arm. The full-panel screening review
provisionally designates fundamentals, without an economics override or triggered
confirmation seeds. Its paired IC change versus S0 is +0.002571351 with interval
[-0.002937179, +0.008393834]; paired net change is +1.321974 bps/day with interval
[-1.742990, +3.672872]. Both intervals span zero. A provisional leader is not
evidence of statistically established improvement.

The final designation must pass all three fixed leave-one-seed-out decision
panels. These are overlapping sensitivity checks, not independent replications.
C6 is not refitted or rerostered in these panels. The registered bootstrap uses
20-session blocks, preserves folds and draws 10,000 replications. Placeholder-v2
and latest-vintage borrow sensitivities have zero promotion weight; the former
uses retrospective calibration and is never a model feature.

The completed audit produced the following choices, with no rejected books or
economics overrides in any panel:

| Panel | Remaining seeds | Primary IC leader and provisional choice |
| --- | --- | --- |
| Full | 11, 29, 47 | fundamentals |
| Omit 11 | 29, 47 | fundamentals |
| Omit 29 | 11, 47 | fundamentals |
| Omit 47 | 11, 29 | C6 |

The required unanimous designation fails. S0 is economically eligible in every
panel and remains the working comparator for a future registered round.
`research_designation` is null and `parent_inconclusive` is true. The close-decision
confirmation trigger is absent in all four panels, so no 61/79/97 runs are required.
The fixed C6 family roster was never changed: omitting seed 47 would have added
events under the roster rule, but this is reported only as sensitivity evidence.
The audit SHA-256 is
`5fa40a7c056f4b24802755a76eeba461501dba9b4be4dc4374949e5aa9108176`.

## Known limitations requiring separate work

The main comparisons do not establish a significant candidate improvement over
S0: every all-fold paired IC interval and every paired net interval spans zero.
Fundamentals leads on the primary metric, while C6 has the higher net point
estimate. The registered economics override does not fire. Fresh pretraining
does not improve the point estimates over the corresponding reused-P arm.
These are screening observations, not evidence that fresh pretraining is always
unhelpful. The MLP remains close in IC, with a negative but uncertain net delta;
this experiment does not establish a large temporal-modelling premium.

Fundamentals' original-rate mean turnover is 0.2045 NAV/day versus S0's 0.2940,
roughly 30% lower. This is a descriptive point comparison. The registered
non-circular moving-block bootstrap underweights fold endpoints relative to
interior observations in finite samples. Its turnover intervals even lie below
their sample means (fundamentals 0.1674–0.1834 versus mean 0.2045; S0
0.2559–0.2724 versus mean 0.2940). Fold-boundary trading makes endpoint sensitivity
particularly relevant. Do not interpret those turnover intervals as centered
uncertainty around the displayed means. The frozen bootstrap and selection rule
remain unchanged; boundary-weight sensitivity belongs in a future registered
statistical audit, alongside the continuous-book comparison.

Magnitudes has positive within-model attribution despite weaker economic results
than S0. C6's joint inference ablation also has a positive interval, although its
training comparison against S0 does not. Removing an input after training is a
different intervention from retraining without it; these intervals must not be
substituted for the registered candidate comparisons.

The larger epoch budget was used: 34 of 714 fits selected after epoch 40, and
15 reached the cap of 60. Seven of those capped fits are MLP fits; the leading
fundamentals arm has no capped fits and a maximum selected epoch of 36.
Thus the original blanket 20-epoch ceiling was a credible constraint, but the
revised cap is no longer a widespread constraint on the leading family arm.
The residual MLP cap concentration merits a separately controlled convergence
check if that architecture is pursued; this round does not extend its budget
after observing its outcomes.

Inherited inferred corporate actions include false adjustments for PETR3 on
2020-03-09 and EMBR3 on 2020-03-12. New ADR/oil inputs exclude 6,200 inferred-action
return cells, but old targets and economic contracts remain frozen. Their repair
requires a separate registered rebaseline. This round does not certify absolute
wealth estimates.

All 22 stale-name settlements were investigated. Three apparent endings arise
from B3 cash-category parser exclusions: AMER3 (2023-01-20, BDI08), OIBR4
(2023-03-03, BDI08), SEQL3 (2024-10-15, BDI07). Claim continuity and these source
categories require a separate repair, without retroactively changing Round-6
comparisons. Limited early CVM identity coverage and sparse historical borrow
vintages remain substantive information limits.

## Continuous working-comparator book

Because the final parent choice is inconclusive, this descriptive readout uses
S0. The accepted three-seed score/mask panels are stitched over 1,738 sessions,
with thirteen model switches, one portfolio initialization, and positions,
pending orders and smoothing state carried across every fold boundary. Saved
missing-score masks are unchanged. The existing economic evaluator and original
rate contract are reused; this is not an additional candidate or selection test.

Continuous mean net excess is **4.3586 bps/day**, versus fold-reset S0's
4.5082 bps/day, a difference of -0.1496 bps/day. Mean turnover is 0.2743 NAV/day,
mean equity gross is 1.9717 NAV, and the descriptive annualized net-excess Sharpe
is 0.7545. All thirteen boundary continuity checks pass, D1–D5 are zero and
engineering gate failures are empty.

Economic uncertainty remains explicitly true. There are 21 settlements under
the inherited last-mark convention, and the terminal NAV of 3.3608 falls to
2.8230 in the registered 30% settlement-haircut scenario. Approximately 67.5% of
short notional uses placeholder rates. These are development assumptions, not
verified liquidation wealth or a claim that the strategy can earn these returns.
No 2025 readiness or access rule is relaxed, and that evaluation remains unspent
by Round 6. Result SHA-256:
`44a939b9fd25d432575606f4283b0238cb34fc26cd431779e105e8206d7fbb35`.

## Implications for the next registered work

Keep S0 as the comparison anchor. Fundamentals and C6 are useful hypotheses,
but the present selection is too seed-sensitive to promote either under the
accepted rule. Preserve the compact-axis/BF16/unique-date training recipe for
future matched comparisons; it materially reduces wasted work without removing
eligible examples. Prioritize the separately identified contractual-action and
cash-category continuity repairs before strong implementability claims.
If the MLP remains a serious alternative, resolve its concentrated cap hits in
a bounded convergence comparison. Any new architecture, attention, overlay or
feature experiment requires the next registration; none was started here.

## Reproduction anchors

* Frozen neural implementation: `0bf8130b73fe6b207c6b6f62603bf383b8f82e1f`.
* Planner metadata-order correction: `447e680`.
* Accepted store: `v2_round6_store_b72950b_20260911T145000Z`, manifest
  `968df3947b68ec68d62638ee53402c908b9ea2e24dfd27291e156e17d31a31c1`.
* Run root: `v2_round6_0bf8130_20260911T233404Z`, frozen design
  `954b48eb3c8c3ddf730b9af1853c789a3762fb3a06c048cb46483b624a28ca0f`.
* Session-1 result: `a67c1e891e1ef5fbfe0c8b7863babe418e21f67890ed28a7ba35b5d9fc508b7b`.
* Session-2 result: `734f7a5c62c9c7aad5fa878117f01066cc3d893659edcc2c333991a52af49500`.
* Screening review: `002b7c83eb89979cffeb2925107451b8467456953352852e8b9e3d1d5d1eb146`.

The complete research implementation and report were merged into GitHub main at
`cf391e414fe7c89531590f9208dcb73190f6f7c4`, including the prior cleanup. The
identical commit is retained in a verified persistent Git bundle, SHA-256
`d9c7fa9a4f1903f49fe525ef5506cfea515fcda69df81a5743ce1163bfa12a69`.
Subsequent closure commits add documentation only. The final analysis mirror
contains 12,968 files; its inventory SHA-256 is
`5c305468e8c552ecaa64260e6a49b9e394f38f8ae6374d56d8019bb6965c8f9b`.
This is in addition to the previously verified training and per-arm readouts.
The final SSH caller timed out after the remote verifier had successfully
written its proof; that existing proof was recovered and matched to the local
inventory without rerunning experiments or repeating the transfer.

The healthy instance ran approximately 27.1 hours through its termination request
at 2026-09-12 20:14 UTC. At its provider-reported US$2.29/hour, that is approximately
US$62 for this host, including engineering, evaluation and recovery time. This
is not an invoice and excludes earlier failed hosts, storage, taxes and billing
rounding. Main training ended at 16:49 UTC; the remaining host time covered the
decision-dependent readouts and verified recovery. Exact provider shutdown and
two absence checks are recorded in
[v2_round6_operations.json](v2_round6_operations.json).

Local run artifacts reside under `D:/quant-data/b3/processed/model_runs/` and
their persistent copies under `/lambda/nfs/brazil-rv-east3/quant-data/b3/processed/model_runs/`.
The tables below cover all arms, including losers. The companion JSON preserves
full numeric readouts and hashes, including the seed decision and continuous
book. The separate operations evidence records artifact recovery and closure.

<!-- GENERATED READOUT TABLES -->

## Complete main-panel results

All IC deltas below use exact common populations. Brackets are registered
95% temporal bootstrap intervals, conditional on the fitted ensembles.
They are not adjusted for screening multiple candidates.

| Arm | Primary IC | Net excess bps/day | All-fold paired IC delta vs S0 |
| --- | ---: | ---: | --- |
| S0 | 0.025616 | 4.508 | Reference |
| C6 | 0.027807 | 6.071 | 0.002191 [-0.003369, 0.008098] |
| C6_fresh_p | 0.027440 | 5.875 | 0.001825 [-0.003136, 0.006508] |
| best_single_fresh_p | 0.027046 | 4.667 | 0.001430 [-0.003477, 0.005422] |
| cross_market | 0.024152 | 4.538 | -0.001464 [-0.004765, 0.001381] |
| events | 0.025439 | 5.073 | -0.000177 [-0.002951, 0.002665] |
| finetune_lr_1 | 0.025517 | 5.385 | -0.000099 [-0.001523, 0.002194] |
| fundamentals | 0.028187 | 5.830 | 0.002571 [-0.002937, 0.008394] |
| lending | 0.025424 | 4.881 | -0.000191 [-0.001976, 0.000634] |
| magnitudes | 0.026395 | 3.151 | 0.000779 [-0.000415, 0.002691] |
| microstructure | 0.025127 | 3.065 | -0.000489 [-0.002757, 0.002799] |
| mlp | 0.025267 | 3.462 | -0.000348 [-0.003534, 0.001377] |
| oddlot | 0.025600 | 4.058 | -0.000015 [-0.000241, 0.000155] |
| options | 0.023321 | 4.607 | -0.002295 [-0.004764, 0.000526] |
| rebalance | 0.025458 | 3.800 | -0.000158 [-0.000483, 0.000141] |
| sector | 0.025094 | 4.041 | -0.000522 [-0.004102, 0.001997] |
| time_decay_756 | 0.024312 | 4.573 | -0.001304 [-0.002804, 0.000281] |

| Arm | Informative paired IC delta | All-fold paired net delta bps/day |
| --- | --- | --- |
| C6 | 0.002191 [-0.003369, 0.008098] | 1.563 [-1.345, 4.960] |
| C6_fresh_p | 0.001825 [-0.003136, 0.006508] | 1.366 [-0.833, 4.246] |
| best_single_fresh_p | 0.001430 [-0.003477, 0.005422] | 0.159 [-2.477, 2.597] |
| cross_market | -0.001464 [-0.004765, 0.001381] | 0.030 [-1.635, 1.792] |
| events | -0.000177 [-0.002951, 0.002665] | 0.565 [-1.581, 2.067] |
| finetune_lr_1 | -0.000099 [-0.001523, 0.002194] | 0.877 [-1.255, 2.473] |
| fundamentals | 0.002571 [-0.002937, 0.008394] | 1.322 [-1.743, 3.673] |
| lending | -0.000190 [-0.002910, 0.001079] | 0.373 [-1.455, 1.421] |
| magnitudes | 0.000779 [-0.000415, 0.002691] | -1.358 [-2.865, 0.578] |
| microstructure | -0.000489 [-0.002757, 0.002799] | -1.443 [-3.049, 0.177] |
| mlp | -0.000348 [-0.003534, 0.001377] | -1.046 [-3.421, 0.615] |
| oddlot | -0.000015 [-0.000241, 0.000155] | -0.450 [-1.630, 0.356] |
| options | -0.002295 [-0.004764, 0.000526] | 0.099 [-2.159, 1.913] |
| rebalance | -0.000104 [-0.000459, 0.000235] | -0.708 [-2.077, 0.228] |
| sector | -0.000561 [-0.004414, 0.002318] | -0.467 [-3.156, 1.424] |
| time_decay_756 | -0.001304 [-0.002804, 0.000281] | 0.064 [-1.408, 1.415] |

## Inference attribution

Trained model minus the same model with the named family forced invalid.
This measures reliance within a fitted model, not the causal gain from
training that family. It has zero selection weight.

| Model | Family invalidated | All-fold paired primary IC |
| --- | --- | --- |
| events | events | -0.000947 [-0.003250, 0.001249] |
| fundamentals | fundamentals | 0.002999 [-0.001461, 0.008471] |
| lending | lending | -0.000255 [-0.001864, 0.000635] |
| magnitudes | magnitudes | 0.002628 [0.000983, 0.004855] |
| oddlot | oddlot | 0.000045 [-0.000008, 0.000112] |
| options | options | -0.001713 [-0.003441, 0.001103] |
| C6 | all | 0.005976 [0.000744, 0.012227] |
| C6_fresh_p | all | 0.004294 [0.000668, 0.008168] |
| cross_market | cross_market | 0.000194 [-0.001704, 0.003028] |
| C6 | fundamentals | 0.001916 [-0.002902, 0.007130] |
| C6_fresh_p | fundamentals | 0.002463 [-0.001065, 0.005887] |
| best_single_fresh_p | fundamentals | 0.001521 [-0.002054, 0.004945] |
| C6 | magnitudes | 0.003323 [0.001523, 0.004914] |
| C6_fresh_p | magnitudes | 0.002447 [0.001307, 0.003719] |
| microstructure | microstructure | 0.001575 [-0.000433, 0.003807] |
| rebalance | rebalance | -0.000045 [-0.000194, 0.000071] |
| sector | sector | -0.001461 [-0.004275, 0.000811] |

## Borrow sensitivity and portfolio diagnostics

Net columns are bps/day. Turnover is the original-rate daily traded
notional as a fraction of NAV. Persistence is the one-session diagnostic.
Only original-rate economics enter the promotion rule.

| Arm | Original net | Placeholder-v2 net | Latest-vintage net | Turnover | Persistence |
| --- | ---: | ---: | ---: | ---: | ---: |
| S0 | 4.508 | 4.875 | 4.509 | 0.2940 | 0.7943 |
| C6 | 6.071 | 5.779 | 6.072 | 0.2124 | 0.8238 |
| C6_fresh_p | 5.875 | 6.032 | 5.873 | 0.2054 | 0.8363 |
| best_single_fresh_p | 4.667 | 4.820 | 4.666 | 0.2140 | 0.8455 |
| cross_market | 4.538 | 5.000 | 4.538 | 0.2674 | 0.8105 |
| events | 5.073 | 4.995 | 5.073 | 0.2955 | 0.8389 |
| finetune_lr_1 | 5.385 | 5.384 | 5.384 | 0.2929 | 0.7958 |
| fundamentals | 5.830 | 5.898 | 5.830 | 0.2045 | 0.8294 |
| lending | 4.881 | 5.074 | 4.882 | 0.2930 | 0.7980 |
| magnitudes | 3.151 | 2.970 | 3.151 | 0.2700 | 0.7860 |
| microstructure | 3.065 | 2.866 | 3.066 | 0.2732 | 0.8052 |
| mlp | 3.462 | 3.318 | 3.462 | 0.3126 | 0.7907 |
| oddlot | 4.058 | 4.018 | 4.058 | 0.2904 | 0.7950 |
| options | 4.607 | 4.470 | 4.608 | 0.2802 | 0.8121 |
| rebalance | 3.800 | 3.970 | 3.800 | 0.2936 | 0.7945 |
| sector | 4.041 | 4.177 | 4.042 | 0.2722 | 0.8082 |
| time_decay_756 | 4.573 | 4.438 | 4.573 | 0.2677 | 0.8177 |

## Momentum and era readouts

These are descriptive diagnostics, not additional selection criteria.

| Arm | Primary IC after momentum residualization | Composite momentum rank correlation |
| --- | ---: | ---: |
| S0 | 0.014574 | 0.6983 |
| C6 | 0.019850 | 0.6023 |
| C6_fresh_p | 0.018324 | 0.6417 |
| best_single_fresh_p | 0.018520 | 0.6465 |
| cross_market | 0.012281 | 0.7035 |
| events | 0.016158 | 0.6713 |
| finetune_lr_1 | 0.015084 | 0.6386 |
| fundamentals | 0.020242 | 0.6213 |
| lending | 0.014696 | 0.6950 |
| magnitudes | 0.016201 | 0.6821 |
| microstructure | 0.013621 | 0.6749 |
| mlp | 0.013899 | 0.7192 |
| oddlot | 0.014401 | 0.6992 |
| options | 0.011298 | 0.6621 |
| rebalance | 0.014283 | 0.6974 |
| sector | 0.013383 | 0.6944 |
| time_decay_756 | 0.012670 | 0.7100 |

| Control | Era | Paired IC delta | Paired net delta bps/day |
| --- | --- | --- | --- |
| finetune_lr_1 | 2018_2019 | -0.000026 [-0.002492, 0.002300] | 0.189 [-2.418, 3.721] |
| finetune_lr_1 | 2020_2021 | 0.000580 [-0.002650, 0.003693] | 0.164 [-4.190, 2.901] |
| finetune_lr_1 | 2022_2024 | -0.000597 [-0.002953, 0.003972] | 1.802 [-1.822, 4.497] |
| time_decay_756 | 2018_2019 | -0.000818 [-0.001683, 0.000014] | -0.838 [-2.253, 1.897] |
| time_decay_756 | 2020_2021 | -0.000150 [-0.001799, 0.000854] | 1.476 [-1.622, 4.292] |
| time_decay_756 | 2022_2024 | -0.002386 [-0.005449, 0.001507] | -0.277 [-2.919, 1.556] |

## Revised training-budget utilization

| Arm | Median selected epoch | Median completed epochs | Maximum selected epoch | Fits at cap 60 | Selected after 40 |
| --- | ---: | ---: | ---: | ---: | ---: |
| S0 | 13.0 | 19.0 | 60 | 2 | 3 |
| C6 | 8.0 | 14.0 | 36 | 0 | 0 |
| C6_fresh_p | 18.0 | 24.0 | 54 | 1 | 3 |
| best_single_fresh_p | 14.0 | 20.0 | 56 | 1 | 2 |
| cross_market | 10.0 | 16.0 | 52 | 0 | 1 |
| events | 18.0 | 24.0 | 60 | 2 | 3 |
| finetune_lr_1 | 8.0 | 14.0 | 38 | 0 | 0 |
| fundamentals | 8.0 | 14.0 | 36 | 0 | 0 |
| lending | 12.0 | 18.0 | 38 | 0 | 0 |
| magnitudes | 13.0 | 19.0 | 44 | 0 | 1 |
| microstructure | 10.0 | 16.0 | 50 | 0 | 2 |
| mlp | 12.0 | 18.0 | 60 | 7 | 10 |
| oddlot | 13.0 | 19.0 | 60 | 1 | 3 |
| options | 9.0 | 15.0 | 36 | 0 | 0 |
| rebalance | 13.0 | 19.0 | 54 | 1 | 4 |
| sector | 9.0 | 15.0 | 32 | 0 | 0 |
| time_decay_756 | 13.0 | 19.0 | 50 | 0 | 2 |

Per-fold results, full intervals, era comparisons, momentum diagnostics,
cost readouts and source hashes accompany this document in
[v2_round6_results.json](v2_round6_results.json). No new fit or bootstrap
was run to construct these tables.
