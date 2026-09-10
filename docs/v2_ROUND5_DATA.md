# Round 5 data round — work in progress

Status: acquisition and family validation are running. **The final store and
GBDT screens are not yet complete. The continuous-book replay is complete.** This document
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
took564.96seconds and peaked at1.36GiB. Pooled before/after economics are compiled from its immutable comparisons. The
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
| Events and fundamentals | Refreshed RAD history; original filing recovery is necessary because account CSVs often retain only the latest version. Annual/YTD TTM bridges preserve usable original information. | Finish capital-count recovery, version reconciliation and joined coverage tables; account-page recovery alone does not supply share counts. |
| Identity and sectors | Dated FCA plus COTAHIST, including historical legal-name matches before ticker fields improve. Three sector-relative fields are sealed; current sector files are not backprojected. | Final store admission. |
| Lending | Sealed seven-field archive adds 160,908 balance rows from 2019-10-02, including the recovered October legacy archive. All 81,324 old balances and 41,353 accepted rates remain exact. True free-float utilization and observed contract-flow surprise are built. | Final store admission. No additional historical rates passed source-vintage admission. |
| Options | Sealed seven-field archive has 274,580 decision/name rows across 268 ISINs. Opening OI has a different position date from traded volume. Two separately named observed-subset fields preserve useful incomplete coverage. | Final store admission; unverified omission-as-zero is not admitted. |
| Cross-market | Sealed 57-field archive includes public shocks, causal exposures and interactions, with historical DST and early closes. Unavailable oil, ADR-premium and foreign-flow fields are explicitly masked. | Final store admission; no guessed measurement or publication lag. |
| Foreign flow | BDI02 recovered; published figures are month-to-date totals with a documented D+2 reference lag and historical methodology revisions. Differencing totals can mix revisions with daily flows. | Source-semantics unavailable for this round; no invented daily increments. |
| Index | Sealed 20,438 decision/name rows across 107 sessions in 2023–2024. Twelve releases have opening proof; seven date-only releases enter the next decision. Weight-change signal is explicitly a proxy containing price drift. | Final store admission. |
| Magnitudes | Four raw causal fields sealed. Fit-only clipping is bound through training, checkpoints and scoring; selection/evaluation cannot refit it. | Final store admission. |
| Oddlot | Existing two fields are retained exactly. Trade summaries do not provide buy/sell direction. | Final store hash comparison. |
| Microstructure | Sealed two-field archive has 966,714 decision/name rows across 712 ISINs. COTAHIST cash trades and official nonregular quantities support the requested formulas. | Final store admission. |

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

## Continuous book and forward collection

The [continuous S0 replay](v2_round5_continuous_book.md) carries positions, pending
orders, cash and claims across all 13 model switches over 1,738 development days.
Net excess is 4.549761 bps/day, versus 4.822005 for repaired fold resets. All
registered engineering gates pass. Economic uncertainty remains explicit:
22 lifetime stale-name settlements use the registered convention, and 66.96%
of short notional still uses placeholder borrow pricing. Peak worker RSS was
3.438 GiB; the completed output was hash-verified and reused after the PC restart.

[Forward capture](v2_round5_forward_capture.md) is implemented and has a verified
36-response initial snapshot. A quiet daily 12:10 São Paulo automation captures
current index views and exact-contract Asian minute data into a separate future
archive. Group-B [funds](v2_round5_group_b_funds.md),
[energy](v2_round5_group_b_energy.md), and [Focus](v2_round5_group_b_focus.md)
timing tables are complete; none is a Round-5 model input.

## Outstanding before closure

Finish source archives and family proofs; build and accept the new store;
run the registered 14-fold × 5-seed CPU information screens and TreeSHAP. Then publish the final family coverage,
revision shares, unavailable-source reasons, economic deltas, screen results,
store identities and resulting Round-6 ordering here.
