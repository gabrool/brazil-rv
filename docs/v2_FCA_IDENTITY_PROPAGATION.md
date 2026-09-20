# FCA identity and dependent observations — 2026-09-20

The broader repair now reconstructs issuer information on **26,271 additional
already eligible stock-days**, with no eligible identity rows lost. It has been
propagated into separate financial and event parquets. This is an intermediate
derived contract for Stage B, not a new accepted model store or a profitability
result. Resolve `fca_identity_admission` and `fca_financial_propagation` through
`v2_economic_data_scaling_run.json`; old inputs and fits remain unchanged.

## Reconstruction and source exceptions

The original pure identity function at `43250b0` reproduced all 880,858 sealed
rows exactly. The new reconstruction consumes the completed 2,820-document FCA
fallback audit, its 1,068 changed documents, and the newer C&A source records.
It retains all 3,717 development sessions and 933 permanent security axes.

The first candidate exposed 708 lost eligible mappings. Direct source review
found CSN Mineração's CVM number `25585` in its ticker field and Eletromidia's
misspelling `ELEMD3`. The archived original offering documents explicitly bind
CMIN3/BRCMINACNOR2 and ELMD3/BRELMDACNOR3, their CNPJ/CVM identities and ordinary
shares. Their timelines establish publication February17/12 2021 and first
trading February18/17 respectively. Source-bound amendments correct these
specific FCA records and two omitted CSN listings. A prior own-ID Recrusul FCA
separately establishes RCSL3 ON/RCSL4 PN; two later records repeat RCSL3 in both
blocks. Only their separate PN block is corrected. Thirteen exact records are
listed in `v2_fca_source_amendments.json`; raw files are unmodified.

An external correction now carries its own per-security known-session bound.
Eletromidia's February12 filing can contribute its already-known metadata, but
the conservatively February17-known offering evidence cannot repair its ticker
in a February12 decision. The source clock is separate from the filing's receipt
and listing date. No fuzzy name/ticker matching or modern HTML class labels are
admitted. CMIN's earlier pre-IPO FCAs are not backfilled with the later offer.

The reviewed result contains 943,419 issuer rows: 64,333 additions across 350
security axes, of which 26,271 observations are already eligible. The remaining
1,772 removals concern nine inactive old share/receipt identities, such as AMBV,
VALE5, BIOM4 and old GRND11/JSLG11 receipts. Their prior original quotes and the
later FCA's explicit current ticker establish the difference; no security axis,
earlier history or eligibility is removed. No shared row changes issuer, CVM,
share class or sector. The 15,257 shared metadata changes concern identity method
and sector-mapping provenance. There are no cross-issuer conflicts. Deleting
future source documents and observations preserves the full pre-2021 prefix.

The initial full baseline was reused after its hash-bound agreement, rather than
run again. The revised identity reconstruction and prefix check took 58.37s.

## Financial and event propagation

Six newly linked issuer identities include Magazine Luiza and Banco ABC Brasil,
which were absent from the old financial extraction cache. The incremental
loader requested eight missing issuer roots in total, including two previously
mapped roots with no cached financial records, and recovered 427 own-version
filings from existing archives. Their unit dispositions, account and capital
sources remain explicitly bound. The prior 21,783-document cache is reused;
the previously admitted C&A capital tables retain their own receipt dates.

Holding financial formulas fixed, the eight financial fields gain **154,656
usable values on eligible dates** relative to the sealed store. These totals
include the earlier C&A recovery, not an additional gain on top of it:

| Field | Additional usable values |
| --- | ---: |
| Accruals/assets | 22,553 |
| Book/market | 12,162 |
| Earnings yield | 12,162 |
| Gross profitability | 24,986 |
| Liabilities/assets | 26,271 |
| Log market capitalization | 12,162 |
| Revenue growth | 22,413 |
| Standardized earnings surprise | 21,947 |

Statement age gains 26,271 values, with dependency ages and flags counted
separately. The seven event fields gain **181,629 usable values**. Neither family
loses a previously valid eligible value. Shared valid financial numbers and
event values remain exact. The valuation-availability flag changes from zero to
one on 213 eligible dates after resolving duplicate/incorrect share-class
identities; the full-calendar 358 changed flags concern Recrusul, Ambev and
Grendene. Newly available valuation numbers are included in the gains above.
All financial/event values and ages remain bit-exact for 203 unaffected names.
The combined extraction and propagation took 131.36s, not a neural-fit estimate.

## Remaining boundaries

This does not complete the deep audit. New parquets need propagation into the
other identity-dependent families, full continuation/universe/warmup/wealth/label
reconstruction and actual neural tensors under a new store/refit contract.
Independent financial numerator/TTM/denominator and remaining publication/revision
checks are still required; loader success is not that independent audit.

Source anomalies remain visible. Some annual MGLU records label MGLU3 as PNA,
while its archived April29 2011 offering explicitly states ordinary shares and
the same ISIN. ABCB4 is PNA in some FCA records but PN in COTAHIST; other early
records lack literal tickers. These must receive source-specific resolution,
not a global relaxation of class checks. The original MGLU offering is archived
and visually checked but has not yet amended those extra records. The indexed
2007 ABC offering URL returned non-PDF content during retrieval; its bytes were
not admitted. Current FAQs do not establish historical identity.

Ten affected identity tests passed, including a new future-source mutation case
and independent clocks for another security and sector metadata. Ruff passed.
The initial source-clock guard and a diagnostic JSON date-encoding failure are
preserved as harness/implementation attempts; neither changed source data or
scored a model. The registration was reread. A/B remain incomplete, C/D unstarted;
no GPU job, forecast scoring or corrected profitability was produced.
