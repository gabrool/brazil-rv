# Round 5 Group B: fund-flow and portfolio timing acceptance

This is the requested timing/coverage audit. Neither source enters model arrays in Round 5. Three historical schema probes retrieved only **14,645 bytes** using HTTP ranges; no full fund archive or 2025/2026 payload was acquired. Exact URLs, original byte ranges, hashes, headers and three representative rows per file are recorded under `D:/quant-data/b3/interim/round5_data_20260910T140442Z/cvm/group_b_schema_probe/manifest.json`.

## Daily fund reports — one-page acceptance sheet

| Item | Evidence and decision |
| --- | --- |
| Historical reach | Official catalogue links history from 2000. This establishes archive availability, not point-in-time completeness. |
| Useful fields | `DT_COMPTC`, fund CNPJ, NAV, quota, total assets, daily subscriptions, paid redemptions, investor count. A January 2021 ZIP header and rows dated January 4–6 were inspected. |
| Source revision policy | Current and previous months refresh Monday–Saturday at 08:00 from CVMWeb receipts through prior-day 23:59; the older ten months refresh weekly with resubmissions. Platform delays are explicitly possible. |
| Statutory delivery | ICVM 555 art. 59 required daily delivery within one business day and provided a three-business-day correction window. RCVM 175 Annex I art. 24 likewise distinguishes delivery and correction. A deadline does not prove actual receipt. |
| Original receipt/version | Absent from both the inspected 2021 header and current dictionary. There is no `DT_RECEB` or per-observation filing/version ID. HTTP Last-Modified describes the archive artifact. |
| Identity/schema break | The 2021 schema uses `CNPJ_FUNDO`/`TP_FUNDO`; the published 2024 transition introduces fund/class naming and subclass identity. Fund, class and subclass must not be blindly collapsed or double counted. |
| Measured coverage | Three sample rows and one historical schema verified; full fund counts, complete development-day coverage, late-filing share and revision magnitude **not measured**. |
| Acceptance | **Timing audit complete; verified historical predictive input unavailable from this archive alone. No first usable model fold admitted.** |

Catalogue history and refresh policy come from the [CVM daily-report catalogue](https://dados.cvm.gov.br/dataset/fi-doc-inf_diario). Delivery and correction rules are in [ICVM 555, art. 59](https://conteudo.cvm.gov.br/export/sites/cvm/legislacao/instrucoes/anexos/500/inst555consolid.pdf) and [original RCVM 175, Annex I art. 24](https://conteudo.cvm.gov.br/export/sites/cvm/legislacao/resolucoes/anexos/100/resol175.pdf). The latter's per-class/subclass structure requires dated adaptation, not retrospective use of today's class hierarchy. The [CVM change log](https://dados.cvm.gov.br/pages/novidades) dates the daily-file schema change to 2024-10-28. Earlier ICVM 409-era delivery details were not separately reconstructed; no universal 2000–2024 deadline assumption is made.

**Correct future information set.** Use the first public payload containing that exact fund/date/version. A Monday observation filed Tuesday night can first appear Wednesday at 08:00; Monday's competence date plus one B3 session is not proven public availability. An observation actually published Tuesday morning should be usable Tuesday at 15:45, without an invented additional lag. A fixed two-day or five-day delay still cannot repair a later corrected value. Prospective immutable daily snapshots or recovered historical publication versions could solve this; no such complete history was found here.

**Economic meaning.** `RESG_DIA` is paid redemptions, not the investor's order timestamp. A flow-pressure experiment must distinguish payment dates, conversion dates and advance redemption requests, and avoid treating fund-of-fund reallocations as independent external investor flows. Use dated classifications including closed funds. These are requirements for a later registered experiment, not new features in this round.

## Monthly CDA holdings — one-page acceptance sheet

| Item | Evidence and decision |
| --- | --- |
| Historical reach | Official catalogue links archives from 2005. Equity-relevant coded assets reside in block 4; derivatives and other assets share that block and must be distinguished by instrument type. |
| Historical sample | Block-4 headers and three rows each from the 2010 annual ZIP and December 2023 monthly ZIP. Both contain asset code, ISIN, month-end quantity/value, confidentiality deadline and instrument-validity dates. |
| Delivery | Under ICVM 555, monthly CDA was due within ten days after month-end. Original RCVM 175 Annex I changes this to ten **business** days; the applicable regime must be dated by fund/class adaptation. |
| Confidentiality | Individual positions can be concealed. The applicable 555/175 rules allow specified fixed-income categories up to 30 days; other categories normally up to 90 days after month-end, exceptionally extended with CVM approval up to 180 days. |
| Public archive | Recent three months refresh Tuesday–Saturday at 08:00 from previous-day receipts; older nine months refresh weekly. Confidential positions appear aggregated until their restriction expires. |
| Vintage availability | `DT_CONFID_APLIC` is a confidentiality boundary, not actual publication. No original receipt or historical content-version ID appears in either inspected header. Expiry alone cannot establish when a delayed/revised row became public. |
| Validity dates | `DT_INI_VIGENCIA`/`DT_FIM_VIGENCIA` describe instrument validity, **not report publication intervals**. The January 2010 PETR4 row starts 1973-09-18; its IND future expires 2010-02-17. They cannot be used as a filing-vintage index. |
| Measured coverage | Six representative rows/two historical schemas. ISIN coverage by portfolio value, confidential-value share, stale-month distribution, first-release delays and revision size remain **unmeasured**. |
| Acceptance | **Timing audit complete; no verified flow-to-stock consumer or first usable fold admitted.** |

The [CVM CDA catalogue](https://dados.cvm.gov.br/dataset/fi-doc-cda) documents history, refreshes and confidential aggregates. It also explicitly states that FIIM holdings from June 2018 onward entered this archive only on **2024-03-04**; these cannot be made publicly available in 2018 merely because their reference month is 2018. Confidentiality rules are in [ICVM 555, art. 56 §§2–3](https://conteudo.cvm.gov.br/export/sites/cvm/legislacao/instrucoes/anexos/500/inst555consolid.pdf) and [original RCVM 175, Annex I art. 22 §§3–4](https://conteudo.cvm.gov.br/export/sites/cvm/legislacao/resolucoes/anexos/100/resol175.pdf). The [2024 CDA schema notice](https://dados.cvm.gov.br/pages/novidades) dates fund/class column renaming to December 16; old reference-period downloads may already use the renamed schema.

**Correct future information set.** Join a daily flow only to the most recent portfolio actually public before that decision, at its then-visible version. Retain portfolio reference age and publication age separately. Do not spread a final confidential aggregate over stocks using its later-revealed composition. Asset ISINs require dated security identity; fund CNPJ is the reporting fund, not the portfolio issuer. Portfolio turnover, fund-of-fund look-through and the distinction between cash equities, units and derivatives require explicit treatment. Evaluate stale portfolio weights and unknown confidential exposure before interpreting a fund-flow signal as stock-specific demand.
