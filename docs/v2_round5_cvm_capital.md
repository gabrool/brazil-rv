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
No successful source is overwritten. The missing-only bulk orchestrator belongs
to `round5_cvm.py`; it should skip capital already supported by exact annual or
original-ZIP data, use modest concurrency, and report unretrievable filings.

Targeted validation:

```powershell
uv run --project research --group dev pytest research/tests/test_v2_round5_cvm_capital.py -q
```

The 18 tests cover printed quantity units, own-period treasury, missing values,
wrong versions/dates/issuers, CAPTCHA and missing handlers, actual form fields,
all four saved source hashes, immutable successful resume, and preserved failed
attempts. This helper does not change valuation's corporate-action uncertainty
policy or claim that share counts remain valid after subsequent capital events.
