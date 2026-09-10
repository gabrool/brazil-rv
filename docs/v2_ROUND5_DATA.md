# Round 5 data round â€” work in progress

Status: original capital and FCA registration recovery and family validation are running. **The final store and
GBDT screens are not yet complete. The continuous-book replay is complete.** This document
will be replaced with the completed acceptance tables and readouts before the
round is closed. CPU only; no paid instance or neural fit has run this round.

The development boundary is 2024-12-30. Historical consumers are bounded to that
date. The user stopped all forward capture on September 10, 2026. The existing
Group-B snapshot remains quarantined evidence, outside historical consumers.

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
| Events and fundamentals | The initial 4,344 missing account versions are recovered. Both capital queues are complete, including all 6,728 scale-supplement documents and its nine audited exceptions. The 1,352-document note collection has finished: 1,052 readable originals, nine separately recovered identity cases and 291 transport failures undergoing one bounded retry. The capital disposition file now has 31 reconciled and seven audited unavailable records. | Finish the identified unit discrepancies and actual admitted-cohort coverage, then produce final joined admission tables. Source retrieval is not admission. Candidate ratios do not determine corrections or masks. Unsupported quantities do not erase usable accounts. |
| Identity and sectors | Dated FCA plus COTAHIST, including historical legal-name matches before ticker fields improve. The 2,196-document supplement is complete and both failures are recovered: 1,766 original XML and 430 exact HTML metadata sources. The main 6,768-document queue continues. Sector translation uses only evidence received by each decision. | Audit remaining primary failures and actual newly admitted account/capital gaps; accept the revised dated bridge, then rebuild sector, cross-market exposures and lending utilization. Candidate retrieval is not automatic identity admission. Existing archives remain immutable comparators. |
| Lending | Sealed seven-field archive adds 160,908 balance rows from 2019-10-02, including the recovered October legacy archive. All 81,324 old balances and 41,353 accepted rates remain exact. True free-float utilization and observed contract-flow surprise are built. | Refresh utilization against the accepted new identity/free float, then admit the final store. The other six fields remain exact. No additional historical rates passed source-vintage admission. |
| Options | Sealed seven-field archive has 274,580 decision/name rows across 268 ISINs. Opening OI has a different position date from traded volume. Two separately named observed-subset fields preserve useful incomplete coverage. | Final store admission; unverified omission-as-zero is not admitted. |
| Cross-market | Sealed 57-field archive includes public shocks, causal exposures and interactions, with historical DST and early closes. Unavailable oil, ADR-premium and foreign-flow fields are explicitly masked. | Rebuild identity-dependent exposures and interactions against the accepted sector bridge, preserving common shocks and unavailable fields, then admit the final store. No guessed measurement or publication lag. |
| Foreign flow | BDI02 recovered; published figures are month-to-date totals with a documented D+2 reference lag and historical methodology revisions. Differencing totals can mix revisions with daily flows. | Source-semantics unavailable for this round; no invented daily increments. |
| Index | Sealed 20,438 decision/name rows across 107 sessions in 2023â€“2024. Twelve releases have opening proof; seven date-only releases enter the next decision. Weight-change signal is explicitly a proxy containing price drift. | Final store admission. |
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
historical acquisition, build and screen stages, then pauses. It cannot launch
forward capture or restore a capture schedule.

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

The completed scale supplement has 6,719 HTML sources, six own-note
reconciliations, one usable original ZIP and two audited unavailable quantities.
The broader unit audit found that explicit source headings can also be wrong:
Petrobras's own notes contradict a thousands heading in two inspected versions.
The builder now applies a sealed exact-filing disposition even when the wrong
number parses successfully; 33 targeted capital/valuation tests pass, including
the actual builder's source precedence. Statement-parent attribution was also
corrected after an expanded-cohort Inter filing exposed an obsolete zero NCI
row beneath liabilities; 87 targeted tests and the actual account audit pass.
These corrections are committed before final family construction. The broader
unit audit remains open; it has not produced a model-input admission yet.

Five additional exact filings now have visually reviewed capital corrections:
MPX/Eneva 5553, TOTVS 6714, Autometal 10338, Magnesita 14794 and Triunfo 25144.
There are now 31 reconciled and seven audited unavailable source dispositions.
These are source decisions, not a completed family or store. Afluente 5872 is
an explicit counterexample: its own thousand-share lot label confirms the
existing large count, so no correction applies. CEG 6022/6029 are now resolved
using each filing's explicit ordinary-share count and treasury ownership0.0047%.
The derived quantity retains that percentage's precision; a half-step rounding
bound contributes approximately0.005bps of the net valuation denominator.
Six further corrections cover Triunfo42119, CCX45620, Celesc56368, Fleury58321,
Eneva63712 and Biomm78942. Literal text extraction produces
review proposals only; it neither admits values nor creates missing masks.
Two misleading date contexts found in proposals were rejected and the helper
was corrected: a prior-year sentence cannot supply current classes, and a
current page header cannot redate an explicitly earlier quarter. Fifteen
targeted note-helper tests pass; no accepted quantity used those bad proposals.

Six more source corrections cover Pague Menos 54638/63573, SmartFit 73028/93475,
Guararapes 91021 and B2W 92819. Each Pague Menos version independently reports
300,000,000 paid-in shares and a further 42,726,580 subscribed shares awaiting
payment; the latter are not added to the paid-in quantity. SmartFit's original
notes establish class quantities, payment status and cancellation of the
described prior treasury holdings. Guararapes's current ordinary-share count is
499,200,000; its prior-year ON/PN composition does not supply current classes.
Its explicit year-end zero treasury is retained as reported. The note also
describes a February purchase of 320 shares without a year-end holding count;
no subsequent disposal is invented. If all remained after the disclosed split,
the denominator sensitivity would be 0.0513 bps. This is a disclosed source
limitation, not an independently reconstructed treasury balance.

Of the completed note collection's 231 non-ZIP responses, 228 contain the same
CVM service-unavailable error, two report a closed connection and one is an
incomplete ZIP. They are transport failures, not evidence that
the historical filings do not exist. One two-worker retry preserves the original
failures and skips all successful originals and nine cached identity recoveries.
The retry also covers 38 timeouts, 15 truncated responses and seven disconnected
responses. Further automatic retry loops are not scheduled.

The review-priority check finds that1,318 of1,352 flagged documents could enter
the earlier identity bridge; four lack a link and30 arrived after its final
mapped date. These34 are lower priority, not final exclusions: the bridge is
being repaired. Restricting review to relevant observations therefore saves
only a small part of this batch. Own-source rounding and bounded recovery are
used to resolve exceptions without demanding nonexistent numerical precision.

Nine additional cached Cremer/Nadir originals were recovered without network
requests after resolving a blank public CNPJ against the exact original inner
document. The public ID/version and inner CNPJ/CVM/reference remain exact;
58 targeted XML, capital and note-review tests pass, and all nine actual
originals reparse successfully. The same review identified a separate possible
currency-unit problem in early statements: some structured sources label units
while their notes state thousands of reais. Matching own-period account rows
must establish any correction; the presence of a note heading alone does not.
This investigation remains a source-freeze hold. See the
[source-note checkpoint](v2_round5_capital_note_audit.json).

Tracing the 47 currency candidates through the actual builder confirms that
all use recovered HTML accounts, with monetary values matching original XML.
Fourteen own-document reviews establish thousands of reais despite the source
unit heading. These include MPX/Eneva, Magnesita, CEG, Excelsior, OSX, Unipar,
Cemig, Telefonica Brasil, Triunfo, Karsten, Tegma, Telebras and Banese. An explicit correction now binds each filing's complete input
rows and source evidence before any temporal arithmetic. It preserves receipts,
periods and independent share counts. Sixty-nine targeted tests pass, including
first-eligible-decision behavior and changed-version/row/evidence failures.
The [currency-unit audit](v2_round5_currency_unit_audit.json) records the reviewed
cases, including eight further accepted filings, and the 33 remaining currency candidates.
The Triunfo and Banese unit proofs reconcile precise current-period tables to
explicit million-real amounts in their own management reports; rounded prose does
not replace the precise table values. No blanket currency rescaling
or model-input admission is implied by a matching number or a note heading.

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

## Continuous book and stopped forward collection

The [continuous S0 replay](v2_round5_continuous_book.md) carries positions, pending
orders, cash and claims across all 13 model switches over 1,738 development days.
Net excess is 4.549761 bps/day, versus 4.822005 for repaired fold resets. All
registered engineering gates pass. Economic uncertainty remains explicit:
22 lifetime stale-name settlements use the registered convention, and 66.96%
of short notional still uses placeholder borrow pricing. Peak worker RSS was
3.438 GiB; the completed output was hash-verified and reused after the PC restart.

[Forward capture](v2_round5_forward_capture.md) produced one verified 36-response
snapshot before the user's stop instruction. No capture process is running and
the recurring task now performs historical Round-5 continuation only. The
existing snapshot remains immutable and quarantined. Group-B [funds](v2_round5_group_b_funds.md),
[energy](v2_round5_group_b_energy.md), and [Focus](v2_round5_group_b_focus.md)
timing tables are complete; none is a Round-5 model input.

## Outstanding before closure

Finish source archives and family proofs; build and accept the new store;
run the registered 14-fold Ã— 5-seed CPU information screens and TreeSHAP. Then publish the final family coverage,
revision shares, unavailable-source reasons, economic deltas, screen results,
store identities and resulting Round-6 ordering here.
