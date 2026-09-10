# Exact-version CVM capital recovery

The public ENET viewer exposes the original filing's **Composição do Capital**
under **Dados da Empresa**. Its own-period paid-in ON/PN counts less its own-period
treasury counts fill the missing capital tables without downloading the entire
original ZIP. This is neither public float nor a preferred-subclass or unit map.
All normal RAD receipt admission and dated security identity rules still apply.

The route is the viewer GET, its returned normal ASP.NET group-selection POST,
then the actual returned `frmDadosComposicaoCapitalITR.aspx` child. The helper
preserves the cookie opener, hidden form fields, public Hash, exact document ID,
reference date, version and CVM code. Group IDs are discovered by their returned
label; they are 1 for the observed ITR forms and 201 for DFP forms. The viewer's
current display name is never used for historical identity: the 2011 example
now displays AXIA, despite being an Eletrobras filing.

The legacy ASP.NET site omits its postback handler for the existing research
User-Agent, even on a healthy page. A standard `Mozilla/5.0` User-Agent obtains
the site's ordinary handler. Both versions report CAPTCHA disabled. Any enabled
or unknown CAPTCHA state stops the helper; no challenge is bypassed. Some old
cached viewers also contain a WCF backend error. That error is not the only
reason the handler was absent.

## Source checks

| Filing | Printed quantity unit | ON after treasury | PN after treasury | Independent check |
| --- | --- | ---: | ---: | --- |
| Itaú ITR 85681, 2019-06-30 v1 | Thousands | 4,958,290,000 | 4,784,963,000 | Exact original nested XML agrees |
| Petrobras DFP 134555, 2023-12-31 v1 | Units | 7,442,231,382 | 5,497,905,879 | Exact original flat XML agrees |
| Eletrobras ITR 11365, 2011-06-30 v1 | Units | 1,087,050,297 | 265,583,803 | HTML-only; two bounded original-ZIP transfers truncated |

Petrobras's surviving annual CSV has version 2 and all six identical capital and
treasury counts. This is supplementary consistency evidence, not grounds to
assign version 2 an earlier receipt. The two truncated 2011 ZIP responses were
not admitted. Telebras 45890 still has no available HTML viewer; its retained
exact original ZIP is needed.

The three successful HTML probes transferred 65,747–66,313 response bytes each
and took 0.85, 5.74 and 7.93 seconds. They do not establish sustained throughput.
11,299 missing filings at this byte rate are about 0.75 GB. At four workers the
small sample's mean implies roughly four hours, with substantial uncertainty;
the acquisition progress should replace this estimate after its first tranche.

Evidence is in
`D:/quant-data/b3/interim/round5_data_20260910T140442Z/cvm/capital_html_route_probe/reconciliation.json`,
SHA-256 `7055274a6ee37574ad1790f6ca2078c4f77227ccb4035de2136caa4aaa75527b`.
A separate successful live helper capture is under `cvm/capital_helper_probe/85681`.

## Integration contract

`round5_cvm_capital.capital_page(document, destination)` acquires one exact
document and returns its sealed manifest. `document` supplies `id`, `cnpj`,
`cvm_code`, `reference`, `version`, `kind`, and `receipt`. Date objects or ISO
date strings are accepted. Reference/receipt after 2024-12-30 are rejected.

`load_capital(document, destination)` verifies every archived source hash and
the viewer/child identity chain, then reparses the source. It returns
`reference`, `quantity_scale`, `paid_in`, `treasury`, and `shares` (physical ON/PN
counts). Attach only `shares` to the same filing; retain the manifest hash in the
family provenance. Missing treasury is not zero. Units come from the capital
table heading, independently of currency units. Independently rounded thousands
can make the printed total differ from ON+PN; printed class counts remain usable.

Completed manifests are immutable and verified on resume. A failed attempt
retains its raw bytes in `attempts/000N`; retrying starts a separate attempt.
Transient network/backend failures may retry the whole three-request sequence.
No successful source is overwritten. The missing-only acquisition skips only
capital supported by exact own-version HTML or original ZIP with an explicit
quantity unit. Annual capital CSV alone is insufficient: its printed values omit
the independent quantity scale. Exact Bradesco 123448 reports thousands while
Ourofino 129981 and Petrobras 135086 report units, and all six annual CSV fields
match those unscaled printed values. The unsafe annual fallback was removed in
`0ddbe3d`; a separate 6,728-document acquisition recovers the missing own scales.

Targeted validation:

```powershell
uv run --project research --group dev pytest research/tests/test_v2_round5_cvm_capital.py -q
```

The 18 tests cover printed quantity units, own-period treasury, missing values,
wrong versions/dates/issuers, CAPTCHA and missing handlers, actual form fields,
all four saved source hashes, immutable successful resume, and preserved failed
attempts. This helper does not change valuation's corporate-action uncertainty
policy or claim that share counts remain valid after subsequent capital events.

## Completed primary collection and bounded exception review

The primary collection finished 15,596 documents in 2,744.437 seconds with
15,586 successful HTML captures and ten exceptions. Successful captures contain
1,111,088,953 archived bytes, including saved POST bodies; actual successful
response bytes total 1,028,423,397. These totals exclude failed attempts and the
separate original-ZIP downloads.

| Exact public filing ID | Bounded review result | Admitted ON / PN quantity |
| --- | --- | ---: |
| 10064 | Blank HTML viewer; exact original ZIP verifies own identity and capital | 5,389,000 / 10,778,000 |
| 76263 | Blank HTML viewer; exact original ZIP | 15,701,103,000 / 0 |
| 76265 | Blank HTML viewer; exact original ZIP | 15,710,221,000 / 0 |
| 124710 | Blank HTML viewer; retained exact flat-XML original | 8,407,000 / 134,000 |
| 82925 | Negative structured treasury; own note17.2, physical PDF page60, states positive 52,119 ordinary treasury shares at2019-03-31 | 90,250,881 / 0 |
| 88901 | Negative structured treasury; own note17.2, physical PDF page58, states positive 5,207 ordinary treasury shares at2019-09-30 | 90,948,793 / 0 |
| 86096 | Negative printed treasury; exact-original endpoint returned preserved non-ZIP backend error | Unavailable |
| 93466 | Negative printed treasury; exact-original endpoint returned preserved non-ZIP backend error | Unavailable |
| 136853 | Negative printed treasury; exact-original endpoint returned preserved non-ZIP backend error | Unavailable |
| 134538 | Own HTML/XML both negative; the retained original note confirms paid-in capital but does not establish positive end-period treasury holdings | Unavailable |

The two note reconciliations are document-specific. Filing82925's numeric
52,119 conflicts with a parenthetical spelled-out 45 thousand; the numeric value
is supported by the reported R$2.332million balance and R$44.75 average cost
(both rounded), and the structured magnitude52 thousand. That conflict is
retained explicitly. Filing88901's numeric and spelled-out counts agree. Paid-in
capital remains at its original thousand-share reporting precision; the more
precise treasury note is retained without rounding it back to thousands.

No global absolute-value conversion is used. The shared HTML/XML quantity rule
requires finite, nonnegative paid-in and treasury counts, with treasury no larger
than paid-in. Unknown treasury for a positive issued class remains missing.
Explicit zero paid-in implies zero outstanding even if its treasury cell is
blank. Invalid capital never removes unrelated accounting observations.

The four unavailable entries are source-quality outcomes, not invented zeros.
Three preserve a297-byte service-unavailable response; this does not prove
permanent unretrievability. A later bounded exact-source retry can revisit them.
For134538, cash treasury balances and weighted-average EPS share denominators are
not substituted for an end-period holding count. Own receipt timing is unchanged
for both supported note counts and unrelated accounting fields.

Immutable evidence under the Round5 root is:

- `cvm/capital_source_dispositions.json`, SHA-256 `7bc18a5a580a8ebf371bdf2ebfb8b9af4495825b640aecb13ea7014899235e6c`: exact document identities, source hashes, reasons, positive note quantities and unavailable dispositions, including the completed annual review below. The preceding primary-only version is preserved byte-exact under `capital_annual_failure_audit_20260910/capital_source_dispositions_before_annual.json`.
- `cvm/capital_failure_final_audit_20260910/`: all ten saved HTML chains, final original-source outcomes and exact PDF note pages. Earlier diagnostic parses are superseded by this corrected-parser audit.
- `cvm/capital_pending_source_audit.json`, SHA-256 `0b3161ded6822866d0740bc3824a4c7298b13ee77c908ac31bd7e707cde90d4a`: records the completed six-document annual review and the remaining broader source-freeze requirements. Known-scale probes123448/129981 were excluded from the scale queue only because their units were already established; both now have explicit own-count outcomes.

The primary failure gate accepts only verified own-version HTML/XML, a
hash-bound positive own-note reconciliation, or an explicit audited-unavailable
disposition. Unreviewed errors still stop it. Passing that primary gate does not
clear the overall source freeze: remaining scale/FCA collections,
any new batch failures and newly admitted identity-cohort gaps still
require their source audits. Only afterward can the new CVM family be sealed and
its dependent identity-based families rebuilt.

The106 targeted tests passed for the shared count rule, XML account preservation,
own-document disposition identity/source mutation, FCA identity and actual
transformed first-decision causality. No model was fitted for these checks.

## Completed six-document annual review

All six own-version originals were retrieved in one bounded pass. The two
Bradesco cases were reviewed independently, including the rendered capital and
treasury-note pages. Four filings establish positive holdings; two remain
unsupported after the exact-source review.

| Filing and own reference | Own-note evidence | Net ON / PN shares |
| --- | --- | ---: |
| Bradesco123448, 2022-12-31 v1 | Physical PDFpages108/110: exact paid-in5,338,393,881 ON +5,320,094,147 PN; positive treasury8,089,200 ON +8,228,600 PN | 5,330,304,681 / 5,311,865,547 |
| Bradesco136104, 2024-03-31 v1 | Physical PDFpages68/69: exact paid-in5,330,304,681 ON +5,311,865,547 PN; positive treasury11,970,600 ON +10,589,200 PN | 5,318,334,081 / 5,301,276,347 |
| Azzas96021, 2020-06-30 v1 | Own note18.2, physical PDFpage62: positive65,207 ON treasury; paid-in90,954thousand ON | 90,888,793 / 0 |
| Azzas98404, 2020-09-30 v1 | Own note18.2, physical PDFpage62: positive3,679 ON treasury; paid-in90,954thousand ON | 90,950,321 / 0 |
| Ourofino129981, 2023-06-30 v1 | Exact original capital still prints−181,400 ON treasury; no positive end-period holding established by the reviewed own notes | Unavailable |
| Ourofino132020, 2023-09-30 v1 | Same signed-quantity issue in its own exact original; no later filing borrowed | Unavailable |

Bradesco's notes also provide exact physical paid-in counts, removing the
structured table's thousand-share rounding. The Azzas reconciliations retain
the paid-in table's thousand-share precision and use the exact note treasury
holdings. All four use their own filing receipts; none uses an absolute value
of the negative structured cell or a later period.

Both Ourofino originals use the flat `XmlInformacoesTrimestraisFinanceiras`
payload, which is outside the current original-account parser's nested-ITR and
flat-DFP scope. A bounded independent XML audit verified their public envelopes,
issuer, reference, version and quantity unit before reviewing the capital
and PDF notes. This parser limitation does not remove accounting data: all six
annual CSV keys map to exactly one matching public header, with24 requested
own-version account rows per Bradesco filing and27 per other filing. Those
account observations remain available when capital is unusable.

Evidence and outcomes are retained under
`cvm/capital_annual_failure_audit_20260910/`, including
`annual_account_attachment_evidence.json`, `bradesco_evidence.json`, exact
original bytes, note pages, flat-ITR identity/scale proofs and
`annual_capital_outcomes.json`. All12 dispositions (six reconciled and six
unavailable across the primary and annual reviews) passed source-hash and
exact-document checks. The primary failure gate was rerun successfully under
`5dfdaa4`. No final CVM family or store was built, and source freeze remains held
for completed-batch exception and expanded-cohort audits.
