# Round 5 data round — work in progress

Status: original capital and FCA registration recovery and family validation are running. **The final store and
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
| Events and fundamentals | The initial 4,344 missing account versions are recovered. The first capital batch recovered 15,586 of 15,596 filings; exceptions are being audited. A further 6,728 exact capital pages are being recovered to establish quantity units absent from annual CSVs. | Finish source exceptions, expanded-cohort coverage and final joined admission tables. |
| Identity and sectors | Dated FCA plus COTAHIST, including historical legal-name matches before ticker fields improve. A further 6,765 missing original FCA versions are recoverable and are being acquired. | Accept the improved dated bridge, then rebuild sector, cross-market exposures and lending utilization. Existing archives remain immutable comparators. |
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

The post-restart audit also corrected two forms of avoidable error. Financial
account numbers change meaning across bank reporting charts, and missing
minority interests cannot be silently treated as zero. Supported ordinary and
preferred shares now contribute their own counts and separate prices to issuer
valuation, instead of excluding every issuer with multiple classes. Seasonal
filing expectations require the actual original version's receipt; a later
revision first observed inside a truncated archive is not an original filing.
The corresponding amendments and targeted tests are committed before the final
family build. A quiet five-minute continuation monitor advances the remaining
acquisition, build and screen stages, then returns to daily source capture.

The final source freeze also waits for original FCA recovery. Among 401 issuers,
6,765 versions (3,386 originals) are absent from annual detail tables. The median
gap to the retained same-reference version is 124 calendar days; this measures
source opportunity, not additional model coverage, because earlier-year metadata
can already be known. Original security descriptions can be generic shares even
when today's viewer renders a typed label. Historical source semantics and dated
B3 cash identity govern the join. Original sector codes and separately sourced
labels must remain distinct. The old identity-dependent admissions will be
replaced only after the revised bridge and dependent families pass their proofs.

Exact original-versus-annual comparisons now confirm that older annual exports
can relabel generic shares as ordinary shares. Original descriptions therefore
govern each version, and genuine modern ticker and preferred-class fields are
retained. Listing dates are also distinct from entry into a governance segment;
the latter must not erase an earlier valid listing. These source corrections
are registered in the [FCA identity amendment](../research/preregistrations/v2_round5_fca_identity_amendment.md).

The capital source audit also found a material unit defect. Annual composition
CSVs retain printed quantities without their unit scale: Bradesco DFP 2022
requires a factor of 1,000, while the checked Ourofino and Petrobras filings
already report individual shares. The model path now uses exact-version XML or
HTML with explicit quantity units; the unsafe CSV fallback has been removed.
Negative treasury quantities cannot inflate outstanding shares. Any repair must
be reconciled to the same filing's notes and retain its evidence; otherwise the
count remains unavailable without erasing unrelated accounting fields. The
[valuation amendment](../research/preregistrations/v2_round5_valuation_amendment.md)
records these corrections; 106 targeted tests passed before the new collection.

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

The extension reruns the existing survivor/liquidity composition audits on each
new family's raw masks and reconstructs the baseline survival flag. Updated
coverage and audit results live in the extension metadata; copied baseline
tables remain byte-identical and describe the original store. Nine targeted
tests cover extension admission, survival/ADV gates, and the existing provider
invariance and causal-price fixture. Actual full-data audit results remain
pending the final store build.

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
