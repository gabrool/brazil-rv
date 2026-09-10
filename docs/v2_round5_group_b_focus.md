# Round 5 — Focus timing audit

Status: timing table only; no Focus feature or historical panel is built in this
round. The weekly public report and daily expectations database are different
publication products.

| Item | Verified semantics | Future causal use |
|---|---|---|
| Weekly Focus report | BCB describes publication on Mondays between 08:25 and08:30, summarizing expectations collected over the preceding30 calendar days through the reference date. | Admit at the actual publication date/time. A normal Monday release is available for Monday15:45; an extra Tuesday lag discards useful information. |
| Reference date | The report commonly refers to the preceding Friday. BCB's August25,2023 report explicitly records publication on August28. | Friday is the survey cutoff, not proof of Friday public availability. |
| Publication exceptions | The actual publication calendar must govern holidays, delays, suspensions and exceptional releases. | Never synthesize one report per Monday or backdate a delayed report to its usual slot. Keep the previous genuinely known report with its age where that is the chosen state contract. |
| Forecast horizon | Calendar-year, monthly and rolling-horizon forecasts refer to future economic quantities. | Keep observation/publication time separate from forecast target year. A forecast of2025 published in2024 is not an observation of a2025 outcome; Round5 nevertheless builds no such input. |
| Daily expectations API | BCB provides historical daily calculated statistics separately from the weekly summary. | Establish that product's first-publication policy before using daily records; the weekly clock cannot automatically authorize daily historical values. |
| Corrections | A current historical API snapshot alone cannot identify first versions or the extent of later corrections. | Retain original PDFs or explicitly versioned responses. Revision share is unmeasured until compared, not assumed zero. |

Primary evidence: [BCB description of the expectations system and release
schedule](https://www.bcb.gov.br/controleinflacao/expectativasmercado),
[BCB's dated August2023 report and publication date](https://www.bcb.gov.br/en/publications/focusmarketreadout/25082023),
and the [historical report description](https://www.bcb.gov.br/publicacoes/focus/01112018).
Some BCB pages return only a JavaScript shell to direct readers; indexed official
text supplies the evidence above. No inferred download timestamp is used as a
historical release time.

Research decision: retain the registered deferral. Candidate information is the
revision in inflation, rates, growth or exchange-rate expectations and forecast
dispersion, not a mechanical copy of already supplied market prices. Common
state should enter without cross-sectional ranking, with dated company exposures
if used for relative predictions. A future preregistration should freeze a small
set of horizons and changes, prove actual release alignment and compare it with
the existing DI/PTAX inputs before increasing model complexity.
