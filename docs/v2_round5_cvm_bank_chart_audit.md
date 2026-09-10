# Round 5: independent bank and insurer account-chart audit

This source-only audit confirms that account numbers are not universal semantic
identifiers. In particular, `2.03`, `2.07.02`, `3.09`, and `3.11` change meaning
across CVM bank charts and individual/consolidated statements. The parser must
match a candidate code **and its published description**, then preserve the
statement basis. No source values or historical filing receipts were changed.

The bounded sample uses the existing official DFP annual archives for 2010,
2019, 2021 and 2024, with Banco do Brasil, Bradesco, Itaú, Santander, Inter,
BB Seguridade, Porto Seguro and Sul América. The extracted 1,815 rows also
include two leasing subsidiaries matched by their names. These are chart-audit
observations, not a new point-in-time feature archive. Annual detail rows often
contain a later version; their numbers must never inherit an earlier filing's
receipt. The 2024 DFP sample is likewise unavailable to a model ending in 2024
until its actual later filing receipt.

## Equity and income

The following amounts are the source's **thousands of BRL**, before the single
`MIL` to BRL conversion. `con` and `ind` are the explicit source statement basis.

| Sample / source version | Basis | Total equity | Noncontrolling equity | Net income | Parent-attributed income |
|---|---|---|---|---|---|
| Bradesco 2010 v3 | con | 2.08: 51,158,565 | 2.08.09: 107,331 | 3.09: 10,052,193 | 3.09.01: 9,939,575 |
| Bradesco 2010 v3 | ind | 2.05: 48,042,850 | Not a consolidated NCI account | 3.13: 10,021,673 | Individual net income |
| Banco do Brasil 2021 v2 | con | 2.07: 146,110,233 | 2.07.02: 3,358,751 | 3.11: 19,722,871 | 3.11.01: 18,344,326 |
| Banco do Brasil 2021 v2 | ind | 2.07: 134,225,898 | 2.07.02 is **capital reserves**, 1,399,561 | 3.11: 19,574,419 | Individual net income |
| Itaú 2021 v1 | con | 2.08: 164,476,000 | 2.08.09: 11,612,000 | 3.09: 28,384,000 | 3.09.01: 26,760,000 |
| Inter 2021 annual v3 | con | 2.08: 8,462,483 | 2.08.09: 74,165 | 3.09: -38,350 | 3.09.01: -72,663 |
| Inter 2021 annual v3 | ind | 2.05: 8,488,639 | Not a consolidated NCI account | 3.13: 34,625 | Individual net income |
| BB Seguridade 2019 v1 | con | 2.03: 5,248,754 | 2.03.09: 0 | 3.13: 6,658,781 | 3.13.01: 6,658,781 |
| Sul América 2019 v1 | con | 2.03: 7,147,705 | 2.03.09: 1,879 | 3.11: 1,181,627 | 3.11.01: 1,182,585 |

The source descriptions distinguish these roles:

- Equity: `Patrimônio Líquido` or `Patrimônio Líquido Consolidado` at
  2.03, 2.05, 2.07 or 2.08. Older bank 2.03 is `Resultados de Exercícios
  Futuros`; other charts use it for provisions or financial liabilities.
- NCI: `Participação dos Acionistas Não Controladores` at 2.03.09/2.08.09,
  or `Patrimônio Líquido Atribuído aos Não Controladores` at 2.07.02.
  Older consolidated 2.07.02 means `Passivos sobre Ativos de Operações
  Descontinuadas`; newer individual 2.07.02 means `Reservas de Capital`.
- Net income: `Lucro/Prejuízo [Consolidado] do Período` or
  `Lucro ou Prejuízo Líquido [Consolidado] do Período` at 3.09/3.11/3.13.
  Older individual bank 3.11 is JCP reversal. Newer bank 3.09 is income
  **before profit sharing and statutory contributions**. Ordinary-company
  3.09 and insurer 3.11 can instead be continuing-operations subtotals.
- Parent income: the published `Atribuído a/aos Sócios da Empresa
  Controladora` child of the correctly identified total. Parent income can
  exceed total income when noncontrolling interests have losses; Sul América
  above is a concrete counterexample to a parent-less-than-total check.

Use total consolidated equity when deriving liabilities/assets. Use equity
attributable to the parent for a parent market-cap denominator, with explicit
NCI evidence or an independently reported parent total. Do not subtract a
same-number capital-reserve account or assume an absent NCI observation is an
observed zero. Keep individual and consolidated flow histories separate.

## Assets, revenue, gross result and operating cash flow

The remaining requested account numbers were stable in the bounded sample:

| Role | Source code | Description variants that matter |
|---|---|---|
| Total assets | BPA 1 | `Ativo Total`; all-issuer inventory also has `Ativo` and `Ativo total:` |
| Reported operating revenue | DRE 3.01 | Sales/services; `Receitas da/de Intermediação Financeira`; `Receitas das Operações`; `Receitas das Atividades Seguradoras/Resseguradoras` |
| Reported gross result | DRE 3.03 | `Resultado Bruto`; `Resultado Bruto [de] Intermediação Financeira` |
| Operating cash flow | DFC_MI or DFC_MD 6.01 | `Caixa Líquido [das] Atividades Operacionais`; insurer `Caixa Líquido Atividades Seguradora/Resseguradora` |

Direct versus indirect cash-flow presentation does not change the 6.01 role.
Negative cash flow is a valid signed source observation: Bradesco 2019 con is
-19,453,969 thousand BRL. All 3.01, 3.03 and 6.01 flow samples have explicit
January 1–December 31 fiscal intervals; assets have the statement endpoint.
Year-to-date/annual construction must preserve those intervals and the basis.

The matching code does **not** establish cross-sector economic comparability:

- Porto Seguro 2019 con reports revenue and gross result both 18,080,692,
  with cost-of-sales 3.02 explicitly zero. Its 3.04.05 other operating expenses
  are -13,620,756. This reported gross subtotal is not comparable to an
  insurer deducting claims above gross result. Sul América 2019 instead puts
  claims and selling costs in 3.02, giving gross result 2,787,632 on revenue
  21,725,615. The existing insurer gross-profitability exclusion is justified.
- BB Seguridade 2019 con explicitly reports zero at 3.01 and 3.03, while
  commission revenue 3.05.01 is 3,066,231 and equity-method income 3.06 is
  2,323,759. These zeros must not be described as zero economic revenue or
  evidence of missing statements. Do not invent a common revenue/gross total
  by summing selected lower accounts.
- Bank 3.03 is a genuine intermediation gross result, and can follow the
  already specified bank-specific admission. Bank operating cash flow remains
  a real source value, but does not thereby become an industrial-company
  accrual measure. The financial-sector exclusions should remain explicit.

If dated sector is unknown, a financial-intermediation account description is
itself contemporaneous evidence of financial accounting. A missing sector
label should not silently classify such a statement as an ordinary industrial
company. Conversely, the code alone cannot identify an insurer using the
ordinary-company chart.

Two all-issuer total-asset descriptions in these four annual files are `A`
and `0`. Those labels do not establish semantics without additional source
evidence; `Ativo` and punctuation/case variants of `Ativo Total` should not be
lost merely because of a narrow string equality.

## Confirmation tests for the owning implementation

1. Feed old-bank individual and consolidated, new-bank individual and
   consolidated, and insurer account rows through the shared semantic mapper.
   Verify the numeric cases above and rejection of same-number wrong roles.
   In particular, a reported wrong-role zero must never replace valid equity
   or income elsewhere in the chart.
2. Verify every source adapter—annual CSV, viewer HTML, nested original XML,
   flat original XML—uses that same mapping and preserves basis, own version,
   period, description and monetary scale. Inter document 111916 is the
   existing first-version flat-XML counterexample; its v1 original must not
   borrow the annual archive's v3 consolidated values.
3. Check parent versus group arithmetic without imposing nonnegative NCI
   profit or parent-income bounds. Keep observed zero separate from missing.
4. Retain exact annual/YTD arithmetic within one basis and fiscal calendar.
   Do not switch to the individual bank because a generic consolidated
   equity code was absent.
5. Assert that insurer gross and bank/insurer industrial accrual/growth
   exclusions remain unavailable while admitted equity/income/assets and
   explicitly supported bank gross measures stay usable.

## Reproducible evidence

All files are under
`D:/quant-data/b3/interim/round5_data_20260910T140442Z/cvm/`:

- `bank_chart_audit.py`: read-only extraction; `bank_chart_sample.json` binds
  source paths, URLs and hashes, plus the 1,815 original selected rows.
- `bank_chart_role_inventory.json`: 122 distinct top-level
  statement/code/description combinations across all issuers in four years.
- `bank_chart_detail_audit.py` and `bank_insurer_dre_details.json`: bounded
  deeper insurer/Inter DRE evidence, reusing those ZIPs.
- `original_zips/111916/manifest.json`: exact Inter DFP2021 v1 original,
  receipt 2022-02-21; source ZIP SHA-256
  `d2a336222101997be22736114062286cdbce13e384ca24834c9d138c762e3d68`.

The official annual sources are
[CVM DFP 2010](https://dados.cvm.gov.br/dados/CIA_ABERTA/DOC/DFP/DADOS/dfp_cia_aberta_2010.zip),
[2019](https://dados.cvm.gov.br/dados/CIA_ABERTA/DOC/DFP/DADOS/dfp_cia_aberta_2019.zip),
[2021](https://dados.cvm.gov.br/dados/CIA_ABERTA/DOC/DFP/DADOS/dfp_cia_aberta_2021.zip),
and [2024](https://dados.cvm.gov.br/dados/CIA_ABERTA/DOC/DFP/DADOS/dfp_cia_aberta_2024.zip).
Their cached versions were reused; this audit made no downloads, feature
builds or model fits and edited no production code.
