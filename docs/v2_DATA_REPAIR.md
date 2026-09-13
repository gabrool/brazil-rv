# Post-Round-7 data repair

Prepared for independent LLM review, 13 September 2026. This implements the data pass requested after [the pipeline audit](v2_POST_ROUND7_PIPELINE_AUDIT.md). The supplied LLM assessment was supplementary; its training and architectural proposals were not treated as requirements for this pass.

## Outcome and interpretation

The repaired development dataset preserves the historical universe, outcomes and execution inputs while correcting source accounting errors and changing how auxiliary information reaches model consumers. The new canonical pointer is [v2_data_inputs.json](v2_data_inputs.json). The sealed Round-7 inputs and results remain historical artifacts under their original code and contracts.

This is a data-correctness result, **not evidence of additional alpha**. The saved Round-7 curves still establish severe overfitting in the richer models. Conditioning and source corrections are necessary work, but do not establish that they explain the whole performance gap, or that any architecture now wins. A0 remains the completed experiment's comparator. There were no optimizer steps, new fits, GPU instances, forward capture or 2025/2026 consumer reads in this pass.

“All data issues” cannot honestly mean that every cell supplied by free vendors is now certified economically correct. All demonstrated defects in this pass have been addressed; residual source-verification gaps and inherited assumptions are listed below. No blanket data deletion was used to make the diagnostics look better.

## 1. Exact-filing source corrections

The rebuild loads the full accepted development ledger: **21,783 financial documents across 398 issuers**, including public revisions. The source investigation used cross-period scale discontinuities and extreme derived ratios to choose records for review. Neither test automatically changes or rejects a record.

The accepted correction ledger contains **43 filing currency corrections and 27 filing capital/treasury corrections**. These counts overlap in some documents and are not counts of distinct issuers. [The source review](v2_data_source_review.json) records each document's identity, reference, version, corrected quantities or multiplier, own-filing reconciliation, source paths and SHA-256 hashes. Currency adjustments affect monetary accounts, not share quantities; capital adjustments separately reconcile paid-in and treasury shares. Originals remain immutable.

Examples:

| Source problem | Repair and evidence |
| --- | --- |
| Auren's capital table understated shares by 1,000 | Own ITR 125769's shareholder note establishes one billion shares. The original extreme earnings yield around 184.835 becomes approximately 0.184835. |
| BK Brasil's TTM/growth bridge combined incompatible currency scales | Own DFP 70549's revenue is reported in thousands of reais. Rebuilding the dependent growth gives approximately 0.280312 instead of 1,279.311839. |
| Três Tentos had inconsistent capital and treasury units | Own DFP 134143 reconciles 498,298,000 issued shares and 135,000 treasury shares. Both counts are corrected, rather than guessing a net denominator. |
| Odontoprev's mixed-unit financial bridge produced gross profitability near 250.842 | Exact-period monetary corrections propagate through assets and TTM components; the traced value becomes approximately 0.523322. |
| Several issuer capital tables reported already-net shares as paid-in | Own capital and treasury notes replace those rows, preventing a second subtraction of treasury shares. |
| JBS's front table misstated both paid-in and treasury shares | Own Q2-2022 note PDF page 16 reconciles 2,373,866,570 minus 155,750,200 cancelled shares to 2,218,116,370 issued shares. Its separate treasury rollforward ends at zero. Inferring treasury from the erroneous front-table difference would have been wrong. |
| Helbor's front table used rounded quantities after a reverse split | Its own year-end shareholder table explicitly reports 133,851,072 issued and 1,157,460 treasury shares in units. Earlier pre-split treasury counts are not reused. |
| Orizon's Q1-2021 prose/front total disagreed with the shareholder table | The same filing's shareholder components and issuance arithmetic independently reconcile to 71,500,000 shares. The conflicting 71,550,000 total is not accepted merely because it parses; later filings are not used to repair an earlier information set. |

When CVM's complete-ZIP route failed, the new bounded note retriever recovered the **same filing's** explanatory PDF through its official public viewer. It binds the original document ID, issuer, kind, reference and version; preserves viewer/postback/PDF evidence; and does not substitute a later revision. Image-based Omega and Braskem statement tables were visually inspected with their current-period columns and unit headings.

Corrections take effect at the **original filing's historical availability**, because the evidence is in that same original filing. They are not backdated from a later statement. Receipt precision is preserved: 21,767 documents have exact receipts and 16 have the existing date-only availability bound. Filing source quantities retain their reported precision where exact quantities are unavailable.

The final extreme-ratio review retains **five earnings-yield observations**: four Contax days and one Oi day. The relevant original notes confirm the loss/share-count information; these are distressed valuations rather than demonstrated unit errors. The other three review screens have no remaining cases. The [financial comparison](v2_data_financial_comparison.json) contains exact before/after counts and remaining identities. These thresholds are diagnostic, not model-input caps.

## 2. Financial construction and point-in-time dependencies

Corrections are propagated by rebuilding from the document ledger, not by patching final ratios. This updates all dependent TTM earnings, gross profitability, accruals, growth and SUE observations, including later values that no longer look extreme.

The financial builder now:

- Retains public historical versions and chooses the latest version available at each decision, separately from fiscal reference dates. Prior-assets selection also respects version and receipt ordering.
- Falls back **per field** to the latest coherent calculation when a newer statement lacks the necessary components. It does not mix a newer balance-sheet denominator with an older flow calculation merely to produce a value.
- Requires only each feature's actual dependencies. Earnings/revenue flows and SUE do not require unrelated assets or equity. Gross profitability requires positive assets; it does not require positive equity. Signed book equity and negative earnings remain usable.
- Records the newest contributing receipt as update age. The oldest contributing receipt and the economic reference age remain separate in the derived audit data. A newly updated TTM signal is no longer presented as years old just because it uses historical components.
- Keeps an explicit negative-earnings flag, independent of the availability of a price-based valuation denominator, and an incomplete-latest-statement flag for real missing/fallback contents. SUE warm-up alone does not label a filing incomplete.

Fallback uses only observations already public at that decision. There is no stale-age expiry, no forward-filled observation relabeled as newly observed, no borrowing from later revisions, and no account-wide rejection when an unrelated field is absent. The original identity map and accepted capital-change observations remain source-bound.

## 3. What is fed into the models

Auxiliary continuous values are stored in their physical, signed representation and conditioned at the consumer with:

`asinh((value - fit_median) / fit_IQR)`

The statistics use **fit rows only**. If IQR is zero but the field varies, the fallback is the median nonzero absolute deviation; a genuinely constant field uses unit scale. The smooth transform limits numerical domination without hard clipping distinct tails to an identical endpoint. It is monotone for each field. Binary flags, already bounded quantities and bounded age encodings pass through according to the feature specification.

This applies consistently to fundamentals, magnitudes, per-name cross-market exposures/interactions, event distances, rebalance pressure and the other continuous sidecars. Common market variables receive one fit observation per date rather than being weighted by the number of active securities on that date. All 44 common fields are retained; there is no PCA reduction in this pass.

Pretraining-to-fine-tuning coordinates are explicit. A field that varied during P retains its parent's center and scale during F. A field absent or constant in P can acquire coordinates from **F's fit rows**, with support and inheritance recorded in the checkpoint. Selection/evaluation rows never fit a scaler. Features are bound by their names and order. Scoring reads the saved preprocessing and never estimates it again.

The same contract reaches the generic S0 training/scoring path and the Round-7 characteristic path. Tree screens use the physical continuous inputs with explicit missing values rather than a neural clipping adapter. Tests cover serialization, inference reuse and train/score consistency.

The fewer-than-20-names rank-support rule is removed from the affected **auxiliary** fields by using the continuous representation throughout. It is not replaced by an unstable rule that switches a field between ranks and physical units as coverage changes. The established ranked slow features remain ranked where that was their contract; this pass does not add handcrafted features or duplicate rank channels speculatively.

The duplicated `fundamentals_native` family is removed from active code and configurations. The canonical `fundamentals` family carries the native financial information once. All four magnitude features remain available: being related to neutralization controls does not prove their levels or interactions are useless. The scalar census changes from **152 to 145 fields**: nine duplicated native-fundamental channels removed and two meaningful financial flags added. The separate three common diagnostics remain available.

## 4. Missingness, freshness, tails and history

Value validity and known age are now independent throughout source alignment, collators and consumers. An unavailable value can still have a known information age. An ablated or entirely unknown family has both its value/mask and age cleared; otherwise it would not be a complete intervention. This repairs the existing three-channel contract rather than introducing a new architecture.

Slow realized skew and kurtosis use a smooth signed `asinh` transform instead of saturating at ±5. Observed-history age uses a smooth bounded logarithmic encoding that continues to distinguish histories beyond 252 sessions. The decoder used by the pipeline's history requirement now understands the stored feature-specification version; it does not interpret the new age scale with the old formula.

Reconstruction preserves the source warm-up. In particular, slow wealth/moment reconstruction retains the original 2009 observations, and odd-lot changes are derived on the full source calendar **before** cropping to development output dates. A first intermediate build exposed lost early odd-lot observations when warm-up was cropped too soon. That intermediate was not accepted; the final builder fixes the order and the acceptance explicitly checks retention.

No OHLC interpolation, invented timestamps, shortened lookback, fixed top-N universe truncation, or relaxed security identity rule is introduced. The full 60-session histories and all point-in-time active names are retained.

## 5. Label construction defect found during acceptance

The characteristic-neutralization buckets used ordinal tie-breaking, allowing the order of security identities to split equal characteristic values into different groups. They now use **zero-based average ranks** so equal values stay together. Permuting securities in a tied synthetic cross-section leaves the result unchanged.

This is a label-construction invariance fix, not a new economic target or neutralization experiment. The acceptance compares bucket membership across every actual development date/horizon and finds **zero changed assignments**. All stored outcome arrays remain byte-identical. The tie policy is recorded in the data/checkpoint target contract.

## 6. Acceptance and reproducibility

The final store has 3,717 dates ending 2024-12-30, 933 historical security identities, **568,815 active stock-days**, and at most **243 active names**. Acceptance binds the store and source hashes rather than trusting a directory name.

It checks every scalar field for finite valid values and known valid ages; source-to-store support; raw financial support before/after by field and year; unchanged slow masks and unaffected slow coordinates; and exact preservation of **73 protected arrays**. These include eligibility, prices, shareholder wealth, corporate-action accounting, target masks/values, fast inputs and execution-related inputs. The final comparison reports zero lost valid observations in the retained fields. Gains are reported separately from the two new flags, so adding channels is not counted as recovery of old observations.

| Family | Recovered observations in existing fields | Lost |
| --- | ---: | ---: |
| Fundamentals | 115,793 | 0 |
| Options | 42,538 | 0 |
| Sector | 34,417 | 0 |
| Lending | 9,747 | 0 |
| Events | 492 | 0 |
| Other retained families | 0 | 0 |
| **Total** | **202,987** | **0** |

These are **feature–stock-day observations**, not distinct stock-days or independent samples. The two new flags add another 607,607 observations, separately. Source-level financial recovery also differs from transformed-channel recovery: for example, physical revenue-growth support rises from 240,758 to 264,520, while the old rank channel had only 240,056. The native duplicate already carried some information excluded from the ranked channel; these coverage gains should not all be interpreted as information absent from every old model input.

The actual P, F2 and F14 loaders are exercised with their full history and stage-compacted axes, followed by S0 and C1 CPU forward passes in FP32 and BF16. The checks verify finite model inputs/outputs and exact retention of active names. These are representative consumer checks, not a full historical rescore and not a GPU performance benchmark. P-to-F inherited coordinates are checked exactly.

Test evidence includes the 362-test data/model suite, the later 177-test source/transform/store/score/training suite, and separate history-decoding and target-contract checks. These suites overlap; their counts must not be added as though they were distinct tests. The later source suite passed in 50.94 seconds with 14 PyTorch deprecation warnings. [CPU acceptance](v2_data_acceptance.json) and [the store comparison](v2_data_store_comparison.json) contain the final machine-readable evidence.

The implementation commits and artifact identities are recorded in [the canonical pointer](v2_data_inputs.json). Reproduction from the repository root uses the research environment:

```powershell
uv run --project research python research/scripts/materialize_data_source_repairs.py --root <repair-root>
uv run --project research python -m brazil_rv.v2.data_repair --repair-root <repair-root>
uv run --project research python -m brazil_rv.v2.data_repair --repair-root <repair-root> --output <fresh-derived-store>
uv run --project research python research/scripts/accept_data_repair.py --repair-root <repair-root>
```

The bound archives/evidence must be present. A changed source/disposition set requires a fresh financial extraction; the explicit cache-reuse option is only for a downstream feature-only change with identical source/disposition hashes. Store building requires a clean implementation commit and a fresh output directory. The final store was placed under the existing C: derived-data root because D: lacked room; no canonical raw archive was deleted or overwritten.

## 7. Source contracts reviewed and preserved

| Family or layer | Contract retained; material limit |
| --- | --- |
| COTAHIST prices, corporate actions, universe and labels | Dated ISIN identity, historical eligibility, entry-bar exclusion, observed endpoints and decision-known input terms remain. Reconstructed calendars and inferred action/terminal terms retain their previously disclosed limits; protected accounting is not silently rewritten. |
| M1 and slow history | Original observed masks, source-assignment boundaries, warm-up and lag convention remain. No new intraday predictors or capture. |
| CVM fundamentals/events/sector | Receipt-aware revisions, historical issuer mapping, same-basis fiscal bridges and the corrected dependency rules above. Missing original evidence and class-price coverage remain visible. |
| Cross-market prices, rates, commodities, ADRs and flows | Existing historical local-clock/release rules remain, with no blanket additional delay. Revised vendor histories are not perfect publication vintages; DI has its documented decision-equivalent bound; foreign flow uses receipt bounds where earlier publication was not established. |
| Lending | Sparse utilization values are recovered; rate/balance data are not represented as guaranteed executable stock-specific borrow. |
| Options | Sparse ratios are recovered without relabeling partial observed-series OI as full-exchange OI. The uncovered-call field has no source observations; removing support gates cannot create them. |
| Odd-lot/microstructure | Original observed source coverage is retained; rolling changes keep warm-up. No missing sessions are fabricated. |
| Rebalance | Historical source availability and identity remain; index-pressure price-drift contamination is a disclosed feature interpretation, not proof of mechanical demand. |

The source unit screen initially identified 567 review candidates and extracted 479 PDFs. Follow-up recovery leaves **86 candidates without a recovered exact original PDF** in that inventory. This is a verification limitation, not 86 proven bad filings and not grounds to drop their accepted observations. Additional exact-note recoveries and residual-case reviews are separately recorded. Some candidates are false positives caused by legitimate units, subsequent splits or actual balance-sheet changes.

No source-independent code can guarantee “perfect” vendor history. The result is a reproducible ledger of corrections, causal construction rules, coverage-preserving conditioning and explicit limits, rather than an unsupported blanket quality certificate.

## 8. Decisions on the supplemental assessment and next boundary

Accepted: comprehensive scaling, coherent P-to-F coordinates, removing auxiliary rank-support exclusions, using one canonical fundamentals family, independent age/mask semantics, better tail/history encoding and real consumer acceptance.

Not adopted as data fixes: deleting magnitude inputs, compressing the market vector to eight PCs, hard ±5 clipping, training-time family dropout/age jitter, altered SAM, learning rates, early stopping, fresh-parent recipes, attention initialization or an architectural ladder. Some are valid experiments; others discard information or confound a data pass. In particular, a lagged-IC peak is **not proof** of leakage or delay: serial dependence and genuine predictive timing can move that peak. Historical source availability and future-mutation tests are the relevant causality evidence.

The next step should address the already demonstrated overfitting/checkpoint-policy problem on these repaired inputs, with matched controls and a small declared experiment. Old checkpoints must be analyzed with their original bound preprocessing and code; they cannot be silently reinterpreted as models trained on this store. No claim that the data repairs restore rich-model performance has been made.
