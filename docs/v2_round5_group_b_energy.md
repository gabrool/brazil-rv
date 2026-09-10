# Round 5 — ONS / CCEE timing audit

Status: source and timing research only. No energy observation enters the
development store, no energy feature is fitted, and no paid source is required.

| Source | History and contents verified | Publication and revision treatment | First possible 15:45 use |
|---|---|---|---|
| ONS daily stored energy (EAR), subsystem | Annual resources from 2000; four subsystem reservoir-energy aggregates and capacity. CSV, XLSX and Parquet catalogues exist. | Current metadata says updates at 12:00 and 19:00, but does not establish the historical first-publication clock or timezone for each row. ONS explicitly permits later consistency corrections. Treat the present files as `latest_vintage`. | Use an observed original release/first-seen timestamp. A proven midday release may enter the same day; a proven evening release enters the next eligible decision. Do not equate the reservoir reference day with publication. |
| CCEE official hourly PLD | The public catalogue has weekly 2001–2020 history and separate hourly files from 2021 onward. Units are R$/MWh, by hour and submarket. | The 2021 procedure specifies daily publication by 20:00 Brasília time for the following day's hours. This is a publication deadline, not every file's actual timestamp. Preserve corrections and contingency announcements. | The published curve for day D is already known by the preceding evening under that procedure. Do not wait until D closes. Earlier availability on D−1 requires its actual release timestamp. |
| CCEE shadow hourly PLD | A separate shadow-price catalogue exists. | Do not relabel the pre-2021 experimental series as the official settlement regime. | Requires its own publication evidence and an explicit regime flag before a future arm. |

The [ONS catalogue](https://dados.ons.org.br/dataset/ear-diario-por-subsistema)
and its CKAN metadata were retrieved. The 83-resource response is preserved at
`group_b_metadata/ons_ear.json` under the Round-5 acquisition root. It illustrates
why file timestamps cannot substitute for publication: the 2000 resource was
modified in 2024, and some 2024 resource objects were created in 2025. Those are
archive operations, not historical information dates. This round cannot estimate
the fraction of corrected historical values from a single vintage; it is
**unknown**, not zero. The [ONS dictionary](https://ons-aws-prod-opendata.s3.amazonaws.com/dataset/ear_subsistema_di/DicionarioDados_EarPorSubsistema.pdf)
describes the variables and missing-value conventions.

The [CCEE catalogue](https://dadosabertos.ccee.org.br/dataset/pld_horario) is
readable. Both direct CKAN API probes returned HTTP403; this is an endpoint
limitation, not evidence that the published CSV files are unavailable.
[Procedure 1.4, revision5, §§3.26 and3.64–3.65](https://www.ccee.org.br/documents/80415/29314541/1.4%20-%20Atendimento_v5.0.pdf/e2d57f4a-86fb-c5db-25e2-0be61f5f37a3)
establishes Brasília time, the day-ahead publication rule, and announced fallback
calculations when the normal model fails. A future archive should identify those
fallbacks instead of treating a repeated price as an ordinary fresh estimate.

Research decision: defer construction as registered. If commissioned later,
first collect original/first-seen releases and compare vintages. Then map company
exposure to generation mix, regulated distribution and merchant sales using
dated disclosures. A uniform “electricity sector × PLD” interaction could combine
opposing business exposures. Test any such feature against the already available
rates, currency and sector information before claiming new information.
