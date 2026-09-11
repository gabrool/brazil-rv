# Round 5 data round — dataset accepted; CPU screens running

The historical dataset, source admissions, BOVA correction and continuous-book
evaluator are complete. The registered CPU information screens are running.
Round 5 remains open until their readout and the resulting Round-6 ordering are
published. No neural fit, paid compute or held-out evaluation has run. Forward
capture remains stopped; its existing snapshot is quarantined.

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

## CPU screens and remaining work

The immutable screen cache is ready and the run is active at
`D:/quant-data/b3/processed/model_runs/v2_round5_screens_9020bde_20260911T023500Z`.
It uses fourteen chronological folds and seeds 11/29/47/61/79, one `a_slow`
parent per fold/seed reused across ten family comparisons. Cells without any
family fit observations reuse their matched parent explicitly. Two local
workers use four LightGBM threads each. Cache preparation peaked at 5.732 GiB.

Each fitted cell includes the five registered horizon models and exact TreeSHAP
on at most 16 eligible names per evaluation day. Primary IC is the matched
D3/D5/D10 readout; five-seed rank ensembles and paired folded block-bootstrap
intervals follow. The original model size, stopping rule, seeds and chronology
remain unchanged. A null GBDT screen cannot reject a neural family, and screens
carry zero network-selection weight. No arm is promoted from an unfinished run.

At 2026-09-11 02:48 UTC, 81 fitted cells were complete. The 70 parent cells
averaged 15.59 seconds and the first 11 cross-market cells averaged 17.81 seconds,
including all five horizons and TreeSHAP. There are 680 actual fitted cells;
90 of the 770 roster cells reuse the parent because the family is unlearnable
in that fit window. These are runtime observations, not completed scientific
readouts. The provisional remaining budget is 2–3 hours including summary/report;
later families may differ. All completed models and cells are resumable.

Only completion/summary of these screens and the final Round-6 ordering remain.
Events/fundamentals, lending, cross-market and options remain research candidates,
with the registered magnitude and other non-data arms available for Round 6.
No neural improvement is claimed from data coverage alone.

## Group B and forward capture

[Funds](v2_round5_group_b_funds.md), [energy](v2_round5_group_b_energy.md) and
[Focus](v2_round5_group_b_focus.md) timing tables are complete and supply no
Round-5 model arrays. The sole pre-stop 36-response snapshot is immutable and
quarantined. No forward capture process or capture schedule is authorized.
The five-minute continuation advances this historical CPU task only and will
pause when the complete report and results have been pushed.
