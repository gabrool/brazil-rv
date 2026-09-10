# Round 5 CVM source audit

This is source feasibility evidence, not a claim that the completed Round-5 store has passed acceptance. Only development source years were acquired. Source probes are under `C:/quant-data/b3/interim/v2_round5_source_audit_cvm_20260910`; registered acquisitions are under `D:/quant-data/b3/interim/round5_data_20260910T140442Z/cvm`.

## Headers preserve versions; account tables do not

Annual ITR/DFP headers contain `DT_RECEB`, `VERSAO` and the original `ID_DOC`. All header versions in the ten-issuer sample match the RAD receipt history. Account CSVs retain only the latest version for revised periods. Every header surviving does **not** establish that first-publication financial values survive.

Reference periods are 2019–2023; receipts are bounded through 2024. Exact keys, hashes and representative revisions are in `v2_round5_cvm_ten_issuer_versions.json`.

| Issuer | Headers / RAD versions | First / later versions | Headers without account contents |
| --- | ---: | ---: | ---: |
| Petrobras | 22 / 22 | 20 / 2 | 2 |
| Vale | 28 / 28 | 20 / 8 | 8 |
| Itaú | 23 / 23 | 20 / 3 | 3 |
| Bradesco | 23 / 23 | 20 / 3 | 3 |
| WEG | 20 / 20 | 20 / 0 | 0 |
| Suzano | 22 / 22 | 20 / 2 | 2 |
| Magazine Luiza | 23 / 23 | 20 / 3 | 3 |
| B3 | 21 / 21 | 20 / 1 | 1 |
| Porto Seguro | 28 / 28 | 20 / 8 | 8 |
| Americanas | 23 / 23 | 20 / 3 | 3 |

There are no header-version gaps in these ten issuers. This is not a universal completeness claim: all structured metadata across those reference years contains 16,168 eligible distinct archive keys versus 16,148 RAD keys, including 10 RAD-only and 30 archive-only keys. Exceptions require individual classification; another version's timestamp is never substituted.

Petrobras DFP 2023 v1 was received on 2024-03-08 at 02:30, and v2 on 2024-03-25 at 19:21. Both headers survive, but the annual account table contains v2 only. Vale ITR 2022 Q1 and Itaú ITR 2019 Q2 have the same distinction. If v1 cannot be recovered, v2 values may enter only at v2 receipt.

## Original-version recovery is possible

The original download successfully returned Petrobras ID 134555 (12,225,509-byte ZIP) and Itaú ID 85681 (8,960,609-byte ZIP). The newer package contains full `XmlDemonstracoesFinanceiras` XML, PDF and XLSX. The older package contains a nested `.itr` ZIP with relational XML tables. A Vale probe returned HTTP 200 with a backend-error body, so status alone cannot establish successful acquisition.

A smaller route opens the public ENET viewer for the exact document ID, follows its returned child URLs and retains the public session cookie. Petrobras v1's four account pages total roughly 235 KB. The HTML, original XML and original PDF reconcile: consolidated revenue 511,994,000; gross profit 269,933,000; parent-attributable earnings 124,606,000, each in BRL thousands. These appear on PDF physical page 14, printed page 13. The PDF's generated filename is a retrieval artifact; availability remains its original RAD receipt.

Public endpoints: [version-specific viewer](https://www.rad.cvm.gov.br/ENET/frmGerenciaPaginaFRE.aspx?NumeroSequencialDocumento=134555&CodigoTipoInstituicao=1), [original Petrobras v1](https://www.rad.cvm.gov.br/ENETCONSULTA/frmDownloadDocumento.aspx?CodigoInstituicao=1&NumeroSequencialDocumento=134555). No CAPTCHA bypass is used. Failed original versions remain explicitly unavailable.

## Identity, sectors and capital

- FCA starts in 2010 and has receipt/version headers, CNPJ, activity sector, security types, ticker fields and negotiation bounds. It has no direct ISIN column; dated COTAHIST identity is also required.
- FCA 2010–2017 cash-security rows have **zero ticker values**. FCA 2018 has 417 populated ticker rows out of 456 cash rows; the 2019–2024 rows examined have populated tickers. A current company-name match does not establish historical identity.
- FCA details are latest-only too. FCA 2023 has 1,246 eligible headers but 731 general detail documents / 731 issuer-periods, with no multiversion detail period. Recover original contents or respect the actual surviving version's receipt.
- Capital-composition CSVs are absent in all ITR/DFP annual files before 2020. From 2020 they provide ON/PN totals and treasury shares, but not every preferred subclass or unit decomposition.
- FRE has issued capital, preferred-class, capital changes and circulating-share fields. Its `distribuicao_capital` can supply actual public float, unlike total issued shares. FRE details are latest-only: 5,201 eligible 2023 headers versus 713 capital detail documents / issuer-periods.
- FRE `Data_Referencia` identifies a filing year, while `Data_Ultima_Assembleia` dates the distribution snapshot. For example BB's 2010-v11 has reference 2010-01-01 but distribution date 2010-08-05. The [CVM's 2017 issuer circular](https://conteudo.cvm.gov.br/export/sites/cvm/legislacao/oficios-circulares/sep/anexos/oc-sep-0117.pdf) describes section 15.3 as the last-meeting capital distribution and also requires circulating shares to update when ownership section 15.2 changes. The consumer therefore preserves the source snapshot date separately from receipt and reference year; share-unit continuity begins at that snapshot. The literal `float_source_fields_20260910` companion contains 9,459 document records with CNPJ, CVM code and snapshot date (14 missing dates), hash-bound to the annual manifest. Known capital evidence in `capital_source_events_20260910` contains 30,914 observations with both identity keys and separate effective/availability dates. These source projections do not substitute issued shares for free float.
- Issuer market value must sum appropriately valued classes without double counting deposited shares and units. Share counts need capital-event and price-unit reconciliation. Parent equity/earnings exclude explicitly reported minority interests. Unestablished denominators remain masked.

Primary catalogues: [ITR](https://dados.cvm.gov.br/dataset/cia_aberta-doc-itr), [FCA](https://dados.cvm.gov.br/dataset/cia_aberta-doc-fca), [FRE](https://dados.cvm.gov.br/dataset/cia_aberta-doc-fre). Verified directories begin [DFP in 2010](https://dados.cvm.gov.br/dados/CIA_ABERTA/DOC/DFP/DADOS/), [FCA in 2010](https://dados.cvm.gov.br/dados/CIA_ABERTA/DOC/FCA/DADOS/), and [IPE in 2003](https://dados.cvm.gov.br/dados/CIA_ABERTA/DOC/IPE/DADOS/); ITR begins 2011. Complete events since 1998 are not established. Pre-2010 observations are outside this round's consumer scope.

## Availability and accounting

RAD receipts are local `America/Sao_Paulo`. The minute upper bound makes 15:44 usable at 15:45; a 15:45 receipt enters next session. Date-only financial receipts enter next session. Archive upload, reference date, original receipt and version are separate fields.

Quarter flows use already received versions. Q4 requires annual DFP and corresponding 9M ITR. A received twelve-month annual report establishes annual TTM directly, even if a historical ITR is unavailable. Interim TTM uses current YTD + prior annual − prior-year same YTD, with exact fiscal-period and accounting-basis agreement and each source's latest individually available version. Standalone quarters and SUE retain the stricter Q4 subtraction requirement. SUE is an accounting seasonal change, not consensus surprise. Its denominator uses the previous eight seasonal changes. Bank and insurer metrics require appropriate account semantics, with masks for incomparable fields.

The code now exposes a sparse valuation subset: exactly one issued non-treasury class, exactly one dated ISIN, own-version capital count and an observed t−1 close. Any intervening completed DISMES/split/ambiguity boundary or already disclosed FRE share-capital change invalidates an older count. It never adjusts a count using a diagnostic split ratio. The store's `action_session_resolved.npy` is retrospective and is not used for admission. This conservative subset is development-grade price-unit evidence, not independently verified corporate-action completeness; cash DISMES changes can also temporarily suppress valuation until another capital snapshot.

All raw features have a matching `<feature>_age_sessions` field. Carried filing states retain actual publication age. For multiple contributing quarters the age is the oldest contributing receipt, including the earlier YTD report needed to isolate a quarter. Daily revaluation does not reset capital/accounting age. Event-window counts and absence flags use the current completed query information set and have age zero; filing-type and elapsed-event states retain their source-event ages.

Six annual account keys share issuer, reference and version but have two document IDs. One is Banco BMG DFP 2021, IDs 111848 and 112039. Their annual account rows do not identify which ID supplied a value, so both originals require recovery; the later same-number filing changes the ledger only at its own receipt. The other cases are Proman, Vert-Ume, Dibens Leasing, Inter & Co (also two CVM registrations), and TRV XLIII. No arbitrary dictionary ordering determines historical values.

## Early exact-name bridge

Historical FCA `Nome_Empresarial` is a dated field distinct from the current name sometimes returned in financial archive headers or viewer URLs. For example, FCA 2010 retains CENTRAIS ELETRICAS BRASILEIRAS SA; a retrieved 2011 financial viewer URL can display today's AXIA name. The latter is not used for historical identity.

The bounded source audit compared historical FCA legal spelling with contemporaneous COTAHIST `issuer_short_name`, removing only accents, punctuation and terminal corporate S.A. No truncation, abbreviation expansion, brand dictionary or fuzzy matching is applied. Results are before decision-receipt/class/lifetime filters:

| Year | Observed ISINs | Exact unique-CNPJ matches | Later independent identity agreement |
| --- | ---: | ---: | ---: |
| 2010 | 546 | 105 | 74 |
| 2011 | 535 | 110 | 80 |
| 2012 | 504 | 106 | 81 |
| 2013 | 470 | 109 | 86 |
| 2014 | 458 | 103 | 88 |
| 2015 | 461 | 99 | 86 |
| 2016 | 447 | 96 | 85 |
| 2017 | 443 | 93 | 88 |

There were zero within-year ambiguous legal spellings in this probe. Later agreement is an audit, never a survivor requirement. Full evidence is `cvm/identity_name_probe.json`; the source probe includes all COTAHIST source instruments, while the consumer remains restricted to the fixed store ISIN axis.

The consumer requires the FCA detail version already received, a unique contemporaneous legal CNPJ, matching cash-share class and effective negotiation bounds, and a prior-session COTAHIST observation. For name-only evidence, the instrument must already have been observed by that FCA receipt; an old issuer spelling cannot admit a newly issued security. The ticker route applies the same protection against later ticker reuse, with one explicit exception: an FCA already disclosing a future listing date may admit a new ISIN whose first observed session is that stated listing's first B3 session. No generic stale-age cutoff is imposed. The first FCA publication in 2010 remains the earliest sector availability, without retrojection of a later sector.

Two apparent cross-era disagreements were classified:

- Paranapanema retains CNPJ legal root 60398369 and CVM 009393, while the establishment suffix changes from 0001-26 to 0004-79. Issuer joining requires **both** the same legal CNPJ root and CVM registration; full raw CNPJs remain recorded. The legal-root interpretation follows [DREI's official definition](https://www.gov.br/empresas-e-negocios/pt-br/drei/legislacao/instrucoes-normativas/inatrucoes-normativas-em-vigor-html/in_81).
- Smiles is a genuine issuer succession. The old SMLE3 instrument ended on 2017-10-20; Smiles Fidelidade's SMLS3 started on 2017-10-23 under a different CNPJ. The new ISIN BRSMLSACNOR1 cannot inherit old Smiles' May 2017 FCA. The [issuer's 2017 report](https://api.mziq.com/mzfilemanager/v2/d/5e992a5e-252e-44bd-acfa-11cbee904064/4d05306a-ebfc-4982-8154-bee3580231af?origin=2) and [B3-hosted incorporation protocol](https://siteempresas.bovespa.com.br/DWL/FormDetalheDownload.asp?prot=566990&site=C) document this succession.

## Acquisition and output contract

The annual manifest binds 74 named archives: FCA, DFP, IPE and FRE 2010–2024, ITR 2011–2024. Existing immutable 2019–2024 sources are reused. The complete RAD manifest binds 150 bounded requests and 582,586 rows: structured 50,059; material facts 32,433; market communications 378,153; shareholder notices 52,705; cadastre 47,419; offerings 6,282; proventos 15,535. Receipts range from 2010-01-04 07:59 to 2024-12-30 23:30; post-cutoff rows cannot enter decisions.

The first source inventory contained 4,230 financial versions missing account contents, including 3,356 first versions. A three-worker public-viewer recovery is ongoing, with every successful document written into `cvm/originals/{id}` and bound by its own manifest. The first 100 recovered documents parsed into positive assets: 87 consolidated and 13 individual-only. Source download success is not equivalent to a complete final family. The expanded exact-name issuer cohort contains 415 issuers and 22,561 financial documents; 4,344 originals require recovery, or 114 additional IDs beyond the original batch. This includes delisted issuers and the same-version ambiguity correction. Progress/failures remain in `recovery_progress.json` / `recovery_result.json`; the supplemental source inventory is `recovery_supplement_inventory.json`.

After code commit, `python -m brazil_rv.v2.round5_cvm build --root <registered-root>/cvm --store <base-store> --output <fresh-family-root>` writes `identity.parquet`, `events.parquet`, `fundamentals.parquet`, raw `free_float_observations.parquet`, known FRE capital-change evidence, annual coverage and a source-bound manifest. `date` is already the 15:45 decision's B3 session, `isin` is the permanent security key, and feature floats are nullable. The store builder must not apply another lag. The identity table supplies dated `sector` for family 2.8. This source program does not fit models or mutate the base store.

Targeted tests cover first eligible minute, source-version mutation, actual joined family receipt mutation, current-close exclusion, future capital-boundary exclusion, unit/multiclass masking, capital-event knowledge timing, early legal-name identity, new-security succession, CNPJ/CVM identity, missing-original values, Q4/TTM accounting and insurer semantics. The final acceptance still requires the complete source snapshot and joined store checks.

## Group B: fund timing

| Source | Catalogue history | Actual information availability | Unresolved issue | Disposition |
| --- | --- | --- | --- | --- |
| Daily fund reports | Since 2000 | Recent M/M−1 files refresh Monday–Saturday at 08:00 from reports received through previous-day 23:59. | Current schema has competence date but no original receipt/version; archives contain revisions and delays. Competence+1 is not proven. | Timing audit only; no model input. |
| Monthly CDA portfolios | Since 2005 | Individual holdings become public after publication and requested confidentiality expiry. Recent three months refresh Tuesday–Saturday at 08:00 from prior-day receipts. | Block 4 has ISIN and confidentiality expiry, but no complete first-publication/vintage history. Final archives expose formerly confidential positions. | Timing audit only; no model input. |

The [daily-report catalogue](https://dados.cvm.gov.br/dataset/fi-doc-inf_diario) and [CDA catalogue](https://dados.cvm.gov.br/dataset/fi-doc-cda) provide schedules and history links. Metadata dictionaries were acquired; no 2025/2026 fund payload was requested. The CDA catalogue states that FIIM holdings with reference months from June 2018 entered this archive in March 2024: reference month must not be mistaken for disclosure. Any future flow-pressure feature also needs a portfolio-staleness analysis.
