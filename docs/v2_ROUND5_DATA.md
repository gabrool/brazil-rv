# Round 5 data round — work in progress

Status: acquisition and family validation are running. **The final store,
GBDT screens and continuous-book evaluator are not yet complete.** This document
will be replaced with the completed acceptance tables and readouts before the
round is closed. CPU only; no paid instance or neural fit has run this round.

The development boundary is 2024-12-30. Historical consumers are bounded to that
date. The explicitly requested Group-B forward archives are separate, quarantined
source capture and confer no permission for held-out model evaluation.

## Completed economic correction

The BOVA11 filter now accepts BDI02 and14 for the exact ETF ISIN, market and
security specification. It recovers92 closes from2019-08-19 through2019-12-30;
all3,871 old observations remain unchanged. The defect left an existing hedge
stale; the supplied description that every affected book was completely unhedged
was not accurate.

All465 registered Round-4 final books have been replayed. Scores, targets,
populations and protected non-ledger readouts remain exact. OnlyF4/F5 contain
numerical BOVA/beta changes; unrelated accounting is reused after exact input
verification. The originally rejected panel remains rejected. The full replay
took564.96seconds and peaked at1.36GiB. Pooled before/after economics are being
compiled from its immutable comparisons. The completed
[economic readout](v2_round5_bova_replay_readout.json) gives S0 pooled net excess
of 4.935761 before and 4.822005 bps/day after. The paired difference is
-0.113755 bps/day [95% interval -0.240366, +0.024140], using 1,738 matched
development sessions and the original 20-session, 10,000-replication folded
block bootstrap. Different arms and seed subsets are reported separately.

Replay root: `D:/quant-data/b3/processed/model_runs/v2_round5_bova_replay_02c1166_20260910`.
The [registration byte-normalization evidence](v2_round5_registration_byte_normalization.json)
explains the original versus normalized document hashes without changing the
frozen replay. Original registration bytes are retained.

## Source findings so far

| Family | Verified finding | Remaining admission work |
|---|---|---|
| Events and fundamentals | Refreshed RAD history; original filing recovery is necessary because account CSVs often retain only the latest version. Annual/YTD TTM bridges preserve usable original information. | Finish original recovery, version reconciliation and joined coverage tables. |
| Identity and sectors | Dated FCA plus COTAHIST, including exact historical legal-name matches before ticker fields improve. Current sector files are not backprojected. | Final field coverage and sector-relative features. |
| Lending | Legacy PDFs recover balances before2022. Two numeric formats required parser repairs. Every41,353 accepted rate row has exact next-session availability. | Complete corrected balance extraction, seam/level audit and true free-float join. Pre2023 observed-rate recovery remains unresolved. |
| Options | Dated official PR/IN archives retrieved2,562 of2,564 files; COTAHIST identifies underlying ISIN directly. Opening OI has a different position date from traded volume. | Complete parsing and completeness proof; unverified omission-as-zero is not admitted. |
| Cross-market | PTAX historical clocks, all six DI swap vertices, Treasury tenors, VIX, Asian contract archives, EWZ/ADR vendor histories retrieved. | Final calendars, exposure joins; ADR cash-price/conversion-ratio and Brent measurement timing remain unverified. |
| Foreign flow | Correct BDI02 chapter recovered; figures are month-to-date accumulated flows with a documentedD+2 reference lag. | Establish exact publication/vintage semantics and valid daily increments before admitting a five-session signal. |
| Index | Dated BDI opening tables can corroborate preview disclosure, avoiding attachment-upload delays. | Reconcile recoverable snapshots and build only proven cycles. |
| Magnitudes | Raw causal channels and fit-only clipping implementation tested. | Consumer binding and immutable family build. |
| Oddlot | Existing two fields are retained exactly. Trade summaries do not provide buy/sell direction. | Final store hash comparison. |
| Microstructure | COTAHIST cash trades and official nonregular quantities support the requested formulas. | Finish source extraction and coverage. |

The [CVM source audit](v2_round5_cvm_source_audit.md),
[B3 source audit](v2_round5_b3_source_audit.md), and
[fund-report timing tables](v2_round5_group_b_funds.md) retain detailed findings.
The [formula amendment](../research/preregistrations/v2_round5_data_formulas.md)
records exact choices under the user's standing authorization. No unavailable
source is replaced by a guessed lag or a different economic quantity.

## Store and model contract

The registered base is `v2_daily_store_3d67624_20260909T100323Z`, manifestSHA256
`db4f751d47133a6611733739ad9bfc15721452218365e338fec61a6b608bea64`.
One new root will copy every protected array, axis and table byte-for-byte and
extend only the authorized sidecar families. Measured peak RSS must remain below8GiB.

Each enabled neural sidecar now has its own masked, zero-initialized residual
projection. Tests show exact parent forward pass, random initialization and
parent gradients with invalid sidecars; invalid names contribute no family
gradient. These are unit tests, not neural research fits. S0 reuse on the eventual
store requires the recorded exact input-array hashes.

## Outstanding before closure

Finish source archives and family proofs; build and accept the new store;
run the registered14-fold×5-seed CPU information screens and TreeSHAP; finish
Group-B capture/timing tables; implement and compare the continuous-book evaluator
after auditing saved score coverage. Then publish the final family coverage,
revision shares, unavailable-source reasons, economic deltas, screen results,
store identities and resulting Round-6 ordering here.
