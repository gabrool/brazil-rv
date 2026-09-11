# Round 5 data round — complete

The historical dataset, source admissions, BOVA correction, continuous-book
evaluator and all registered CPU information screens are complete. Magnitudes,
oddlot and options have positive paired nominal 95% IC intervals in the tree
screens; these prioritize neural experiments without establishing an S0 gain.
This report supplies the Round-6 order and ends Round 5. No neural fit, paid
compute or held-out evaluation has run. Forward capture remains stopped; its
existing snapshot is quarantined.

## Accepted store

The accepted root is
`D:/quant-data/b3/processed/v2_round5_store_9020bde_20260911T023227Z`,
manifest SHA-256
`44a19df181f4896d12e510d1f49cc815762a057c0a74ca7bad879ea02a86cf7b`.
It has 3,717 sessions from 2010-01-04 through 2024-12-30 and the unchanged
933-ISIN axis. It was produced from clean commit `9020bde` in 46.13 seconds;
measured peak RSS was 4.034 GiB, below the 8-GiB limit.

The registered base is `v2_daily_store_3d67624_20260909T100323Z`, manifest
`db4f751d47133a6611733739ad9bfc15721452218365e338fec61a6b608bea64`.
Of its 79 arrays, the three lending arrays are authorized replacements.
**All 76 protected arrays, both axes and all 36 tables are exact.** The accepted
store has 103 arrays. Existing tables retain their original meaning; new source
coverage and composition audits live in the extension metadata.

S0 uses the unchanged slow inputs and can reuse its sealed scores without a
neural refit. The acceptance record binds slow values, masks, ages, timestep
validity and activity by hash. The neural sidecar contract is implemented:
separate zero-initialized, validity-gated residual projections; invalid families
preserve parent forwards, initialization/RNG and parent gradients exactly. These
are engineering tests, not evidence of a new neural improvement.

[Machine-readable acceptance and complete feature/year coverage](v2_round5_store_acceptance.json)
bind the store, sources, proofs, calendar audit, S0 inputs and screen design.

## What is available to the model

Counts below use at least one valid **transformed** family field at an active
decision/name, not merely a downloaded filing or identity match. First informative
fold requires observations in both the fit and evaluation periods. The JSON
also reports each field's exact first/last usable date, all yearly counts and
all informative folds. Late or sparse fields keep separate masks and ages.

| Family | Fields | First model date | Covered ISINs | Model name-days | First informative fold; count |
|---|---:|---|---:|---:|---|
| cross_market | 57 | 2010-01-06 | 439 | 568,567 | F1; 14/14 |
| events | 7 | 2010-03-12 | 323 | 310,858 | F1; 14/14 |
| fundamentals | 12 | 2010-03-12 | 323 | 310,858 | F1; 14/14 |
| lending | 7 | 2019-10-02 | 308 | 182,649 | F6; 9/14 |
| magnitudes | 4 | 2010-02-02 | 437 | 566,292 | F1; 14/14 |
| microstructure | 2 | 2010-02-02 | 437 | 565,557 | F1; 14/14 |
| oddlot | 2 | 2010-01-04 | 439 | 568,218 | F1; 14/14 |
| options | 7 | 2011-04-04 | 251 | 244,455 | F1; 14/14 |
| rebalance | 2 | 2023-08-02 | 237 | 20,438 | F13; 2/14 |
| sector | 3 | 2018-02-15 | 267 | 195,146 | F2; 13/14 |

| Family | Data and processing | Decision availability / remaining source limitation |
|---|---|---|
| Events | RAD/ENET timestamps, financial filings and material facts; causal ages/counts and an expected filing date based on the issuer's prior-year lag | Minute timestamp upper bound plus one minute, then first 15:45 decision. Date-only receipts use the next session. Expected dates are forecasts, not announced schedules. |
| Fundamentals | Original CVM ITR/DFP versions, dated FCA CNPJ/CVM-to-ISIN identity, consolidated-first accounts; causal standalone quarters, TTM and share-class valuation | Every version enters at its own receipt. Original quantity and currency scales are audited separately. All positive share classes need contemporaneous identity and observed prior closes; unresolved capital actions invalidate valuation, not unrelated accounting. |
| Lending | Balances backfilled to 2019-10-02; rates, registered flow, true circulating-share utilization | Next-session publication. All 81,324 prior balances and 41,353 accepted rate rows remain exact; no additional older rate vintage passed admission. Rates still begin 2023-07-11. Flow surprise uses at least 15 actual observations within the preceding fixed 20 sessions. |
| Cross-market | PTAX, six B3 PRE/DI tenors, Treasury curve, VIX, EWZ and Asian individual-expiry settlements; causal exposures, sector shrinkage and shock interactions | Actual fixing/market close, historical DST and US early closes. PTAX and Asian day settlements can enter the same decision. Prior-session OI selects the futures expiry; no continuous-contract level jump. Vendor revisions remain a disclosed limitation. |
| Options | COTAHIST option volume plus dated instruments and observed OI | Next session. Complete aggregate OI requires actual source support; separately named observed-subset fields preserve incomplete evidence. Missing series OI is not assumed zero. No IV/pricing model or covered/uncovered split is invented. |
| Sector | Own-receipt historical FCA classifications and causal code/label evidence; issuer-equal peer returns and momentum | Historical groups only. Recovery can add or remove valid groups. Own classes do not count as independent peer issuers. Sector labels themselves are not model features. |
| Rebalance | Nineteen official releases for IBOV/IBXX/SMLL in 2023–2024 | Twelve opening proofs permit same-day entry; seven date-only releases enter the next decision. Pressure is preview-minus-last-effective weight and contains price drift; no fund AUM or pure-flow interpretation is claimed. |
| Magnitudes | Four raw scale channels alongside ranked features | Causal returns, volatility, traded value and economic beta. Clipping is estimated on each actual fit window only; selection/evaluation cannot refit it. |
| Oddlot | Existing two fractional-market fields | Retained exactly. Buy/sell direction is absent, so net imbalance cannot be derived. |
| Microstructure | Average trade size and nonregular/after-hours volume share | Completed-source information at the next decision, with independent feature coverage. Earlier COTAHIST trade counts support average size before the later after-hours archive. |

Entirely unsupported fields stay masked in the accepted store:

- **cross_market**: `shock_oil_1`, `shock_oil_5`, `exposure_oil`, `exposure_oil_times_shock_1`, `exposure_oil_times_shock_5`, `adr_premium_close`, `foreign_flow_5`, `foreign_flow_5_times_log_volume_mean_20`.
- **options**: `put_call_oi_log_ratio`, `delta_oi_to_volume_1`, `uncovered_call_share`.

The [cross-market source audit](v2_round5_cross_market_source_audit.md) documents
failed or unsuitable routes, the official B3 replacement for the proposed SGS
DI route, Asian history gaps, vendor-adjusted US prices and their revision limits.
Brent's assessed/latest-vintage archive does not establish the requested immutable
market-price interpretation; it was not assigned a guessed clock. ADR premiums
lack dated cash-price/conversion proof. Retrieved foreign-flow MTD summaries do
not establish the requested causal daily flow. These omissions are explicit
results. The [B3 source audit](v2_round5_b3_source_audit.md) and
[index audit](v2_round5_index_source_audit.md) retain the source evidence.

## Identity, quality and leakage protection

The accepted CVM family has 880,858 dated identity rows across 581 ISINs; every
row reproduces the completed source preflight after sorting by date/ISIN.
Identity coverage includes inactive periods and is not model coverage. Own
receipt, legal CNPJ root, CVM registration, listing dates and share class govern
the join. Modern viewer labels, current constituents and later classification
translations cannot backfill history.

The original account-version recovery, revised-cohort gaps and original FCA
supplement are complete. All 47 identified account-currency discrepancies have
own-document reconciliations. The capital file contains 157 dispositions:
150 reconciled and seven explicitly unavailable. The focused 50-filing review
closed with 11 corrections and 39 retained originals. Adjacent ratios locate
reviews but never set quantities, masks or availability. Published rounding is
retained where sufficient; no blanket rescaling or exact legal-count claim.

Terra Santa's original 2020Q2 ITR was recovered from a complete CRC-verified
member of a preserved partial download. Multiplus's own contemporaneous report
corroborates its literal count at published precision. Moura's disclosed
post-reference capital approval prevents obsolete counts from being paired
with later post-consolidation prices.

Copasa's gross/net narrative ambiguity remains disclosed. Its specific current
capital/treasury tables remain canonical. A bounded alternative-input readout
changes 30 transformed cells for 2021Q3 (15 Copasa and 15 peer-rank effects),
maximum absolute difference 0.05416; it changes none for the one selected 2022Q1
session. Masks and ages remain exact. No predictive fits were used to choose
the source interpretation. See [CVM acceptance](v2_round5_cvm_final_acceptance.json),
[capital evidence](v2_round5_capital_note_audit.json),
[currency evidence](v2_round5_currency_unit_audit.json) and
[dated identity review](v2_round5_fca_identity_review.json).

Every admitted family has a bound source/availability proof. Actual producer,
parquet join and FeatureSpec fixtures mutate source values and verify unchanged
pre-publication outputs and change at the first eligible decision. Receipt,
price-scale, class, action-boundary and fit-scaler isolation checks protect the
relevant contracts. Fixture causality evidence is distinguished from full-data
identity/hash checks; this does not claim a full-panel mutation experiment.
Unknown observations remain masked; prices are not interpolated or silently
forward-filled as observed. Known market fixings are not delayed for an archive's
later upload, and consumers apply no second arbitrary lag.

### Coverage audit correction

The first store attempt stopped after 25 seconds: lending's pooled lowest-ADV
quartile bootstrap lower bound was 5.643 percentage points. Its balance history
starts years before rates and registered flow, while low-liquidity delisted
names contribute almost no observations in the later source era. The original
pooled warnings remain in the final store, including quartiles 1–3.

The registered correction compares the same decision sessions in each liquidity
stratum, weighting each group/session equally and resampling continuation names
with fixed calendar weights. It keeps the 5-point threshold, 1,000 draws, 95%
interval and 20-name/2,000-name-day support thresholds. It does not alter a
feature, source mask, age, universe or training weight. Fifty-two targeted tests
include a true contemporaneous-bias case that still stops admission.

In lending's lowest quartile, the matched gap is +0.907 percentage points,
95% interval [-0.485, +2.238], on 758 shared sessions. Another 351 sessions have
only one survival group and are reported outside that comparison. Quartiles 1–3
pass; quartile 4 lacks required support and is reported without a passing claim.
Missing common support does not establish equal coverage. The original
[stop evidence](v2_round5_store_coverage_stop.json) remains immutable.

The dependency checker was also corrected to allow sector coverage to change
with the accepted historical identity. Its 17,048 added and 38,410 removed keys
are reported; every final row has a valid contemporaneous sector. Magnitudes,
common shocks, regression support and all six non-utilization lending fields
remain exact. The completed derived artifacts were reused.

## Economic fixes and continuous book

The BOVA11 filter now accepts BDI 02 and 14 for the exact ETF ISIN, market and
specification. It recovers 92 closes in 2019H2 and preserves all 3,871 prior
observations. The defect left an existing hedge stale; it did not necessarily
make every affected book unhedged.

All 465 registered final Round-4 books have been replayed with protected scores,
targets, populations and non-ledger readouts exact. Only F4/F5 have numerical
BOVA/beta changes. The originally rejected panel remains rejected. The replay
took 564.96 seconds, peak RSS 1.36 GiB. S0 net excess changes from 4.935761 to
4.822005 bps/day; paired delta -0.113755, 95% interval [-0.240366, +0.024140].
This uses 1,738 matched sessions and the original folded 20-session block
bootstrap with 10,000 draws. [Full economic readout](v2_round5_bova_replay_readout.json).

The continuous S0 book carries positions, orders, cash and claims across all
13 model switches. Net excess is 4.549761 bps/day versus 4.822005 for repaired
fold resets. Engineering gates pass. Twenty-two stale-name settlements retain
the registered convention, and 66.96% of short notional still uses placeholder
borrow pricing. No unsupported historical rate backfill is presented as observed
borrow. Peak RSS was 3.438 GiB. [Continuous-book readout](v2_round5_continuous_book.md).

## Completed CPU information screens

The run at
`D:/quant-data/b3/processed/model_runs/v2_round5_screens_9020bde_20260911T023500Z`
completed at 2026-09-11 04:28:47 UTC, exit code 0 and empty stderr. All 770
registered cells are accounted for: 680 trained and 90 explicitly reused their
matched parent because no family observation existed in that fit window.
Fourteen chronological folds, seeds 11/29/47/61/79 and five horizon models per
trained cell produced 3,400 saved models, whose individual hashes were verified.
Two local workers with four LightGBM threads each took 112.68 minutes, including
the paired summary. Maximum fitted-cell RSS was 4.103 GiB; cache preparation
peaked at 5.732 GiB. Model size, stopping rules, seeds and chronology were retained.

Primary IC is equal-head D3/D5/D10 neutral-target Spearman for the five-seed rank
ensemble. Every comparison uses the same supported names, dates and outcomes.
There are 1,598 finite daily IC observations across 1,738 roster sessions: the
last ten outcome dates per fold are masked by the registered target-window rule,
while scores remain available. Intervals use 20-session blocks and 10,000 draws,
preserving fold boundaries. Input coverage determined informative folds before
fitting. The `a_slow` parent IC is **0.017440 [0.009502, 0.023782]**.

| Added family | All-fold IC | All-fold paired delta | Informative-fold paired delta [95% interval] | Informative folds | Folds with positive delta |
|---|---:|---:|---|---:|---:|
| magnitudes | 0.020662 | +0.003223 | +0.003223 [+0.001033, +0.005823] | 14 | 11 |
| oddlot | 0.020542 | +0.003103 | +0.003103 [+0.000858, +0.005128] | 14 | 11 |
| options | 0.020202 | +0.002762 | +0.002762 [+0.000329, +0.004708] | 14 | 10 |
| events | 0.019788 | +0.002348 | +0.002348 [-0.001593, +0.005382] | 14 | 9 |
| fundamentals | 0.014896 | -0.002543 | -0.002543 [-0.009227, +0.003572] | 14 | 6 |
| lending | 0.019522 | +0.002082 | +0.003224 [-0.000126, +0.005053] | 9 | 6 |
| cross_market | 0.017343 | -0.000097 | -0.000097 [-0.003897, +0.003331] | 14 | 9 |
| sector | 0.017138 | -0.000302 | -0.000325 [-0.003431, +0.002157] | 13 | 7 |
| microstructure | 0.018197 | +0.000757 | +0.000757 [-0.001371, +0.003139] | 14 | 6 |
| rebalance | 0.016524 | -0.000915 | -0.006333 [-0.014179, +0.000018] | 2 | 0 |

The positive intervals for magnitudes, oddlot and options are promising
**exploratory** evidence. These are ten comparisons with nominal intervals,
without a multiplicity adjustment. Lending has a similar informative-period
point estimate, but its interval includes zero. Events are also suggestive;
fundamentals and cross-market do not demonstrate improvement in this learner.
Rebalance has only 231 finite dates in F13/F14 and cannot support a broad
historical conclusion. An informative fold requires some family observations,
not full coverage of every field or name. The JSON retains exact coverage.

The [full readout](v2_round5_cpu_screen_readout.json) contains every fold, seed,
daily ensemble IC, both summary populations and field-level attribution. It is
a byte-identical copy of the immutable completed readout, SHA-256
`41d31ff88021fb244c0fd216ce1b60a1bcf33c32a2f4a9e1b0d2c6131405a6ef`.
The [screen acceptance](v2_round5_cpu_screen_acceptance.json) binds the producing
code, store, design, cell inventory, serialized models, timing and memory.
These screens have **zero network-selection weight**. They do not measure
strategy P&L, promote a neural parent or reject a neural family after a null result.

### What the tree model used

Exact TreeSHAP uses at most 16 mask-selected eligible ISINs per evaluation day,
without outcome-based sampling. The following ranks sum each field's value and
age contributions, averaging equally across the five heads, seeds and
informative folds. Age share is relative to the appended family's absolute
contribution, not the whole prediction or P&L. It includes only the separate
age channels; an explicitly named event-age feature is still a value channel.

| Family | Leading fields, in attribution order | Separate age-channel share |
|---|---|---:|
| magnitudes | `log_traded_value_20`, `economic_beta_60`, `daily_vol_20_raw` | 0.0% |
| oddlot | `oddlot_volume_share`, `oddlot_volume_share_change_5` | 0.0% |
| options | `put_call_volume_ratio_5`, `option_to_stock_volume_20`, `observed_series_put_call_oi_log_ratio` | 2.3% |
| events | `sessions_until_expected_filing`, `dividend_announcement_age`, `sessions_since_material_fact` | 28.7% |
| fundamentals | `gross_profitability`, `liabilities_to_assets`, `sue` | 24.6% |
| lending | `loan_balance_to_volume_20`, `loan_balance_change_5`, `loan_balance_change_1` | 5.5% |
| cross_market | `exposure_vix`, `exposure_rates_br`, `exposure_rates_us` | 3.8% |
| sector | `sector_momentum_12_1`, `name_minus_sector_return_21`, `name_minus_sector_return_5` | 1.6% |
| microstructure | `avg_trade_size_20`, `after_hours_volume_share_5` | 3.1% |
| rebalance | `index_pressure`, `index_event_age` | 10.3% |

Attribution describes fitted usage, not the causal source of incremental IC.
For example, the options result does not isolate volume from observed-subset OI,
and fundamentals can receive substantial attribution without improving the
paired readout. Sparse source eras and age channels remain visible for review.
No field is removed or selected from these descriptive rankings.

## Round-6 experiment order and end state

The recommended execution order below combines the registered source priorities
with the three positive screens. It advances promising screens; null screens do
not remove any family from the research program. Each arm starts from the same
S0 and adds one family through the tested validity-gated sidecar contract.

1. **Magnitude channels (E6).** Broad history, four fields and the strongest
   full-period paired screen; test whether retaining scale helps the network.
2. **Oddlot.** Broad coverage and a positive screen across 11 of 14 folds.
3. **Options.** Positive screen, while preserving distinct volume and observed-OI
   masks; the full family is tested before any field attribution ablation.
4. **Events**, then **fundamentals** as separate arms. Retain the registration's
   high prior for information absent from prices. The weaker fundamental tree
   result does not settle how the neural model will use it.
5. **Backfilled lending.** Retain its prior neural evidence and test the longer
   balance history; do not describe unavailable early rates as new information.
6. **Cross-market exposures**, then **sector-relative**, **microstructure** and
   **rebalance** arms. Preserve all candidates; identify the short rebalance
   evaluation history explicitly when interpreting its result.

Alongside these single-family arms, retain the registered cheap checks:
fine-tuning multiplier 1.0 versus 0.3, time decay 756, and the point-in-time MLP
comparator. Fix their protocol before fitting and use matched seeds and folds.
Only after the individual neural results should a combination of helpful
families be registered. The existing close-call confirmation-seed rule remains;
no extra confirmation run is launched by this report. Attention and the broader
hyperparameter pass remain Round 7. The continuous evaluator is ready for a
separately registered portfolio round once meaningful-capital costs are fixed.

The delivered artifact is a processed, model-ready development store with
source provenance, masks, ages, feature definitions, leakage/identity proofs,
economic repairs and completed information screens. No further retrieval,
processing, feature build or CPU screen is pending within Round 5. Unsupported
fields and source limitations are explicit research constraints, not promises
of a later capture job. **Stop here; Round 6 has not begun.**

## Group B and forward capture

[Funds](v2_round5_group_b_funds.md), [energy](v2_round5_group_b_energy.md) and
[Focus](v2_round5_group_b_focus.md) timing tables are complete and supply no
Round-5 model arrays. The sole pre-stop 36-response snapshot is immutable and
quarantined. No forward capture process or capture schedule is authorized.
The historical five-minute continuation ends with publication of this complete
report. Its automation is to be paused after the final push; daily capture must
not be restored.
