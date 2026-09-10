# Round 5 B3 source audit

This audit records source recovery and information availability through 2024, not experimental outcomes. No paid compute or later development payload was used. The registered base has 3,717 sessions and 933 accepted ISINs. Raw inputs and earlier archives remain unchanged. The acquisition root is `D:/quant-data/b3/interim/round5_data_20260910T140442Z`.

## Admitted lending archive

The final source family is `lending_feature_archive_october/manifest.json`, SHA-256 `6f3924b041174df8fef0fffadc7f1d355e3c9f2e391263dc52dd2cdea054e57f`. Its `features.parquet` has 242,232 unique decision-date/ISIN rows, seven nullable features and a separate age for each feature. Its `date` is already the first eligible 15:45 decision; the store must not shift it again.

The archive adds **160,908** balance observations for **481** dated ISINs, from decision **2019-10-02** through **2022-03-21**. All **81,324** previously accepted balance rows and **41,353** previously accepted rate rows remain exactly unchanged. This extends the balance inputs; it does not establish the requested earlier economic borrow rates.

| Feature | Valid observations | Definition |
| --- | ---: | --- |
| `loan_balance_to_volume_20` | 240,169 | Outstanding BRL balance / mean BRL turnover over its source session and preceding 19 sessions |
| `loan_balance_change_1` | 228,751 | Exact one-session change in that balance/turnover ratio, using only the known source vintage |
| `loan_balance_change_5` | 223,039 | Corresponding exact five-session change |
| `loan_rate` | 41,353 | Published taker annual rate in decimal units |
| `loan_rate_change_5` | 36,491 | Exact five-session change in the known annual-decimal rate |
| `new_loan_volume_surprise` | 32,781 | Current registered share quantity relative to the mean and sample standard deviation of observed quantities in the preceding fixed 20 sessions, with at least 15 observations |
| `utilization_proxy` | 29,266 | Outstanding share quantity / genuinely circulating shares of the same dated share class |

The original legacy site also supplies all 23 October-2019 reports, despite the current BDI service returning empty HTTP 500 responses for the sampled dates. Their contemporaneous PDF creation/modification times establish original vintage. Recovered 7,171 printed rows yield 5,895 exact dated cash-identity matches for 295 ISINs. The other 1,276 rows remain identity-unresolved. This adds October context and fills 242 previously missing one-session and 1,205 five-session changes in early November. Every old nonnull value/age stays exact, and all features from 2019-11-11 onward are identical. The preceding strict and support-amended archives remain immutable. Source evidence is `b3_lending_october2019/manifest.json`.

The first five formulas retain the existing contract. No missing daily lending observation is carried forward. Their older availability history and the narrower original later-period source population are explicit in the coverage file.

The flow support rule was amended using coverage alone, before any outcome inspection. All accepted quantities are positive and nonmissing, but 26 entire source sessions are missing during July–November 2024. A 20-of-20 requirement masked every later window after 2024-07-02. The registered 15-of-20 rule restores 9,429 observations while still rejecting August 2024, whose maximum support is 14. It keeps the fixed exchange-calendar window, excludes the current value from normalization, requires a present numerator and finite positive sample standard deviation, and never substitutes zeros or the last 20 observed rows. All 23,352 strict-support values and every other feature remain exact. The strict archive stays sealed separately. [Support amendment](../research/preregistrations/v2_round5_lending_support_amendment.md)

## Recovery, publication and source defects

The repaired PDF parsers handle spacing, compact printed dates, ticker prefixes and a wrapped final year digit. A second repair recognizes seven-decimal monetary values such as `7816985.3500000` as BRL 7,816,985.35. Before that repair, a lone decimal point could be stripped and inflate values by ten million. The corrected extraction processed 625 already downloaded PDFs in 415 seconds with two CPU workers. `b3_lending_tables_corrected` contains 274,336 raw balance rows and 31,810 raw registered-loan rows. The rate extraction hash is unchanged by the monetary parser repair.

Of the 587 legacy reports, 514 supply an admitted, interpretable original-vintage balance table. The other reports are individually accounted for:

- **67 conflicting tables**, 2020-10-27 through 2021-02-05, contain repeated ticker records in unlabeled restarts. The original 2020-11-30 page prints three different ABEV3 quantities and implausible monetary values. Visual inspection and alternative text extraction confirm source ambiguity; the records are not silently summed.
- **Three later-modified PDFs**, 2021-03-01, 2021-12-30 and 2022-01-03, lack the original content vintage by the proposed first decision. They remain excluded. Eighteen other late HTTP modification times are merely server-copy metadata: contemporaneous PDF creation/modification and the official schedule establish their original vintage, so those reports receive no extra delay.
- **Three files without the lending table**, 2020-05-12, 2020-07-03 and 2021-02-02, instead contain other bulletin sections. Missing rows are not invented.

Two precisely identified field defects are handled separately. The 2020-07-15 table's monetary column omits decimal separators with inconsistent implied scales; its 350 BRL fields are null while its unambiguous integer quantities remain available. One 2020-10-15 GMAT3 record prints quantity 10 and BRL 9.07 million, incompatible with the dated cash unit scale; both fields are excluded. No guessed rescaling or general outlier clipping is used. Source-level price comparisons are diagnostics, not outcome-based filters.

Legacy PDFs print a position date that often precedes their report date, and original publication is after the report-day decision. They therefore enter on report D+1 while retaining the true, often two-session, position age. A glossary for a different CSV channel does not prove that the same PDF values were already available at the earlier market open. All 41,353 old rate rows were separately audited and have actual availability exactly at trade D+1; the new flow calculation does not add another session. [BDI balance glossary](https://www.b3.com.br/data/files/BF/F1/58/15/391EA810E9C1AAA8AC094EA8/Glossario%20_%20Posicao_Em_Aberto.pdf), [era-1 balance glossary](https://www.b3.com.br/data/files/BB/D1/67/58/C841B810E9C1AAA8AC094EA8/LendingOpenPositionFile%202023.pdf)

## Lending data that remain unavailable

The public frontend specifies a two-step request: `api/download/requestname?fileName=LoanBalance&date=YYYY-MM-DD&recaptchaToken=`, then `api/download/?token=...`, on `arquivos.b3.com.br`. Base names and aliases ending in `File` can resolve filenames even when no historical body remains. Fresh immediate requests returned HTTP 200 with zero bytes for sampled historical lending, derivatives, instrument and after-hours CSVs. Later token reuse returned 401; some later-era lending names returned 400. These are access failures, not empty economic observations. Requests and frontend evidence are retained in `C:/quant-data/b3/interim/round5_b3_source_probe_20260910`. B3's original November 2019 announcement promised only a ten-day downloadable history. [B3 announcement](https://www.b3.com.br/pt_br/noticias/dados-para-download.htm)

`LendingOpenPosition` is outstanding stock: asset quantity and BRL balance are different fields. `LoanBalance` is registered contract/asset flow. Its donor/taker minimum, average and maximum rates are annual percentages, divided by 100 once for annual decimals. Registered flow includes manual renewals and is not strictly new-loan initiation. A printed rate on a zero-flow day is not a fresh observed quote. [B3 taxonomy](https://www.b3.com.br/data/files/1E/F0/54/58/FADF371045F0BD37AC094EA8/Catalogo_de_Taxonomia_UP2DATA_-_Portugues.pdf), [registered-loan glossary](https://www.b3.com.br/data/files/32/02/C0/25/391EA810E9C1AAA8AC094EA8/Glossario%20_%20Emprestimos_Registrados.pdf)

A separately frozen latest-vintage candidate recovers 6,483 additional positive-flow observations, preserves all old rates and matches 1,535 overlap quantities/rates. It retains all six donor/taker statistics. However, the newly recovered late-2024 PDFs were regenerated in April 2025. This does not prove values changed, but original first-publication values cannot be verified. The candidate is **not admitted** to model inputs or headline economic replays. Its manifest is `lending_rates_latest_vintage_candidate/manifest.json`, SHA-256 `755e9d524e7521a46c6183ce2e6b48945059731d435b2f314ace160f17d5b901`. The unknown revision share is reported as unknown, not estimated without evidence.

Consequently, verified observed rates still begin **2023-07-11**, and their last accepted decision is **2024-11-27**. No claimed October-2019 rate backfill or shortening of the economic placeholder era is justified. There is also a real methodology seam: B3 removed its outlier filter from weighted registered-loan rates on 2024-11-18. [B3 068/2024 circular](https://www.b3.com.br/data/files/83/F5/96/95/522239106EEC8429AC094EA8/CE%20068-2024-VNC%20Metodologia_taxaBTB_PT.pdf)

The corrected balance overlap is exact in both quantity and BRL: 715 observations across five March-2022 reports, 1,269 across nine March/April-2024 reports, and 3,243 across 23 late-2024 reports have zero differences from the frozen archive. Matching the existing overlap does not establish the first vintage of additional unmatched records. Full evidence is `b3_lending_tables_corrected/seam_and_rate_distributions.json`.

Accepted rate distributions below are grouped by **source trade year**, so 140 December-2023 observations become available in January 2024. Figures are annual percent, unweighted across name-days; these are source distributions, not portfolio borrow costs.

| Source year | Rows | Median | Mean | 95th percentile | 99th percentile | Maximum |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2019–2022 | 0 | Unavailable | Unavailable | Unavailable | Unavailable | Unavailable |
| 2023 | 13,090 | 0.4358% | 2.8654% | 14.7889% | 39.3000% | 422.6163% |
| 2024 | 28,263 | 0.3198% | 2.6868% | 11.1642% | 34.0302% | 360.5470% |

No source rate is winsorized at acquisition. Training-only transforms or portfolio-weighted cost diagnostics belong to their respective consumers, and the unrecovered earlier years do not receive fabricated observed rates.

## Genuine free float

Utilization uses CVM FRE circulating ON/PN share quantities, not total issued shares. The dated bridge requires matching legal CNPJ root **and CVM code**, and exactly one contemporaneous security of the relevant class. Total PN float is never allocated across PNA/PNB, and units are not substituted for shares.

The FRE filing-year date is not the measurement date. The actual `Data_Ultima_Assembleia` identifies the distribution snapshot; its own-version receipt establishes availability. Intervening known capital events or already observed share-unit uncertainty invalidate an older denominator until a suitable later snapshot exists. No retrospective action-resolution array adjusts the old share count. Feature age follows the older of the loan-position age and float-publication age; the separate measurement date governs unit continuity. This distinction avoids treating Jan 1 as a false annual measurement date. [CVM's 2017 guidance](https://conteudo.cvm.gov.br/export/sites/cvm/legislacao/oficios-circulares/sep/anexos/oc-sep-0117.pdf)

The archive documents 47,024 identity/class exclusions, 4,226 missing suitable float observations, 36,070 known capital boundaries and 125,646 observed unit boundaries. These are reported denominator limitations; they do not mask the independent lending balance/rate fields.

## Options and microstructure sources

The official nested-ZIP endpoint remains live: `https://www.b3.com.br/pesquisapregao/download?filelist=PR241227.zip` and the equivalent `IN241227.zip`. Acquisition retrieved 2,562 of 2,564 requested files over 2019-11-01 through 2024-12-27. The two missing names returned HTTP 200 with a 22-byte empty ZIP on an independent retry, rather than a network error. This proves current archive absence, not historical nonpublication. The full archive is roughly 18 GB and extraction uses four bounded CPU workers.

The store axis contains 933 ISINs; its development security master contains 911 historical ticker segments for 902 identities. The 31 additional axis identities have zero active cells and zero observations in every canonical 2010–2024 daily archive. They receive no fabricated historical data. Eight identities have multiple accepted intervals; the parser now preserves every interval without bridging gaps. This correction changes no BVBG eligibility during its 2019–2024 source range. The cash source already used all intervals. Re-extracting the earlier COTAHIST intervals across 2010–2014 and 2019 finds zero previously omitted option trades, so the existing 261,904 option name-days remain exact. `b3_identity_axis_audit.json` and `cotahist_interval_source_audit.json` bind this evidence.

Each original XML version's creation timestamp is preserved. A later HTTP archive timestamp is not used as publication lag; versions later than the historical decision are excluded. Full PR-header inspection found only one file with before-15:45 versions: 2021-06-11 includes delayed June-10 reports and an early June-11 opening report. The delayed reports are identified by their printed `TradDt`, not their ZIP filename, and their actual June-11 morning availability is retained. The early June-11 OI snapshot has no complete underlying aggregate, but its separately labeled observed-subset fields are available at the June-11 decision.

The malformed 2021-01-04 final PR has a preceding intact 19:19 original publication. Its 367 overlapping cash quantities match COTAHIST exactly; original XML bytes and hashes are retained without source repair. The moved June-10 report has internally exact total=regular+nonregular quantities, but only 58/374 cash quantities equal the current COTAHIST archive. It is preserved as the originally published PR statistic with that discrepancy disclosed; it does not replace COTAHIST trading data. The missing IN2023-12-08 cannot supply complete option listings, while cash PR rows can be identified independently by exact dated COTAHIST ticker/ISIN.

COTAHIST itself explicitly stores the underlying cash ISIN for equity-option trades. This recovers **261,904 traded option name-days from 2010–2024** without a ticker-prefix reconstruction. Cross-checks against BVBG's explicit underlying ID join match all 2,123 traded series on 2019-11-01 and all 8,060 on 2024-12-27, with no unmatched or conflicting identities. `cotahist_option_volume/manifest.json` binds all 15 annual source and output hashes. The cash denominator axis independently contains 1,173,583 exact dated market-10 rows for 854 accepted ISINs. The remaining accepted ISINs are not assigned invented cash observations.

The volume formulas use reported units, not option premium BRL divided by stock turnover and not a US-style assumed 100-share multiplier. `option_to_stock_volume_20` is trailing option-unit volume / stock-unit volume; `put_call_volume_ratio_5` uses trailing put/call units. Complete COTAHIST with verified listed options permits a no-trade zero. Before instrument-listing coverage, a missing name-day remains unknown rather than an invented zero. [B3 equity-option contract](https://www.b3.com.br/pt_br/produtos-e-servicos/negociacao/renda-variavel/opcoes-sobre-acoes.htm)

## OI and activity timing

B3's own BVBG.086-to-UP2DATA mapping explicitly says `OpnIntrst` is the **opening position of D**, equivalent to the previous close. An after-hours D publication first used at D+1 therefore has position age two. Its `delta_oi_to_volume_1` denominator is stock **share-quantity** ADV20 ending on the actual OI position date D−1; dividing contracts/shares by BRL turnover would be dimensionally wrong. Trading volume and nonregular activity are D measurements, used at D+1 with age one. [Official BVBG mapping workbook](https://www.b3.com.br/data/files/57/85/8C/A3/5C11881036DB3088AC094EA8/BVBG.086%20para%20UP2DATA.xlsx)

The PR format does not establish that every omitted OI field equals zero. A 2024 sample has 24,762 positive OI fields, 12,170 omissions and no explicit zero OI; omissions include both untraded and traded option series. This strongly suggests zero suppression, but the optional-field catalogue is not an authoritative completeness guarantee. Until independent totals or an explicit rule establishes that guarantee, aggregate OI is admitted only when every relevant listed series has an actual OI field. Volume features are not masked by that restriction. The separate covered/uncovered split is absent from PR; the failed derivatives CSV route cannot supply `uncovered_call_share`. [Price catalogue](https://www.b3.com.br/data/files/16/70/29/9C/6219D710C8F297D7AC094EA8/Catalogo_precos_v1.3.pdf), [derivatives glossary](https://www.b3.com.br/data/files/1E/D1/BA/58/C841B810E9C1AAA8AC094EA8/DerivativesOpenPositionFile%202023.pdf)

A pre-result amendment retains the information actually present in incomplete reports. `observed_series_put_call_oi_log_ratio` is the natural log of reported put OI divided by reported call OI, requiring both sums to be positive. `observed_series_oi_coverage` separately reports explicit OI fields divided by known listed series. Unknown listings stay unknown; support zero does not claim zero economic OI. Normal after-hours publications have age two. The authentic 2021-06-11 01:53:59 PR and 10:31:28 IN supply 133 both-positive ratios and 156 coverage observations at the June-11 decision, with age one; the 18:13 IN version is excluded. The newest known source position governs when an older position report arrives at the same decision. Cash volumes are not backdated. This preserves useful reported positions without making a complete-market claim. Series composition can change, so no unmatched subset OI difference is added. The requested complete-aggregate fields keep their strict masks. [Observed-series amendment](../research/preregistrations/v2_round5_observed_oi_amendment.md)

`avg_trade_size_20` uses trailing cash BRL turnover divided by actual trade count. `after_hours_volume_share_5` uses explicit PR nonregular quantity; when only total and regular quantities exist, their exact difference is usable. A missing component is not automatically zero. Daily unsigned quantities do not identify buyer/seller imbalance, so no odd-lot net-flow direction is fabricated.

The targeted fixtures protect future XML version exclusion, exact dated identity, first eligible decision, separate position/publication dates, no compressed rolling windows, missing-field semantics, source-unit continuity and support thresholds. Source mutations also pass through the actual family alignment function: raw values, masks and ages remain exact before publication and first change at the eligible decision.

After the PC restart, all 890 completed daily outputs matched their recorded hashes and row counts. Another 304 failed attempts were preserved as operational failures, not mislabeled source gaps; all 780 remaining input files matched their original acquisition hashes. Investigation found that the first lxml optimization retained previously parsed B3 message headers. The corrected streaming parser frees completed prior sibling subtrees, preserving exact output while using only 0.122 GiB peak RSS on a large 2024 daily pair. The benchmark took 10.32 seconds versus 15.35 seconds for the original parser. Four workers resumed from verified daily checkpoints. The manifests distinguish these operational attempts from real source limitations.

## Final activity archives and review bindings

Extraction finished all 1,280 available IN/PR pairs. The one malformed final PR
is replaced by its intact earlier original publication; the two separately
recovered source gaps remain documented above. Normalized source artifacts contain
210,071 published OI snapshot rows, including the authenticated earlier release,
and 555,511 cash PR rows. No operational failure remains a consumer source gap.

`options_feature_archive/manifest.json` has SHA-256
`36e5b13ecfeff2cc9d78ae9481ad6d00c1d582d4faad8f11c90631a87b9cb7ca`.
Its 274,580 decision/ISIN rows cover 268 identities, beginning 2010-01-11 and
ending 2024-12-30. `microstructure_feature_archive/manifest.json` has SHA-256
`ab20364d4f8e02a619e4fec25b61a4a1f0fdf77b78399799575089776c5cd7a7`.
Its 966,714 rows cover 712 identities, beginning 2010-02-02 and ending 2024-12-30.

| Feature | Nonmissing source observations |
| --- | ---: |
| `option_to_stock_volume_20` | 259,201 |
| `put_call_volume_ratio_5` | 241,261 |
| `put_call_oi_log_ratio` — complete aggregate | 462 |
| `delta_oi_to_volume_1` — complete adjacent aggregates | 409 |
| `uncovered_call_share` | 0 — separate source unavailable |
| `observed_series_put_call_oi_log_ratio` | 183,373 |
| `observed_series_oi_coverage` | 209,915 |
| `avg_trade_size_20` | 933,969 |
| `after_hours_volume_share_5` | 420,321 |

These counts precede the active-universe and feature-transform support masks;
the final store report measures effective model coverage. In particular, the
small complete-aggregate OI population cannot be presented as broad market OI.
Every archive includes annual feature coverage, individual source/audit hashes,
the amended formula and publication rules, and original source limitations.

The three ready family-admission objects are in
`b3_family_admissions/admissions.json`, SHA-256
`4c7ec9fc2d8c61bee81135a1ce83f68103d7e3bf7aa9a8fbb7de914773588474`.
Each binds the exact feature parquet, immutable source manifest and a separate
availability proof. All eight B3 fixtures passed again when these proofs were
written. The proofs include 20 active names, 19 frozen peers and the actual final
feature transforms, demonstrating unchanged values/masks/ages before publication
and the first transformed change or mask change at the eligible decision. Lending's
source functions are AST-identical to those used for its earlier sealed archive.
Neither admission wrappers nor this report rewrite the preserved source manifests.
