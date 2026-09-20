# FCA identity and dependent observations — 2026-09-20

The broader repair now reconstructs issuer information on **31,858 additional
already eligible stock-days**, with no eligible identity rows lost. It has been
propagated into separate financial, event, sector, cross-market and lending parquets. This is an intermediate
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

The initial reviewed result contained 943,419 issuer rows: 64,333 additions across 350
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

Holding financial formulas fixed, the eight financial fields gain **179,685
usable values on eligible dates** relative to the sealed store. These totals
include the earlier C&A recovery, not an additional gain on top of it:

| Field | Additional usable values |
| --- | ---: |
| Accruals/assets | 25,085 |
| Book/market | 13,786 |
| Earnings yield | 13,786 |
| Gross profitability | 30,445 |
| Liabilities/assets | 31,730 |
| Log market capitalization | 13,786 |
| Revenue growth | 24,945 |
| Standardized earnings surprise | 26,122 |

Statement age gains 31,730 values, with dependency ages and flags counted
separately. The seven event fields gain **219,291 usable values**. Neither family
loses a previously valid eligible value. Shared valid financial numbers and
event values remain exact. The valuation-availability flag changes from zero to
one on 213 eligible dates after resolving duplicate/incorrect share-class
identities; the full-calendar 358 changed flags concern Recrusul, Ambev and
Grendene. Newly available valuation numbers are included in the gains above.
All financial/event values and ages remain bit-exact for 203 unaffected names.
The combined extraction and propagation took 131.36s, not a neural-fit estimate.

## MGLU and ABC source resolution and dependent peers

The final overlay has **949,539 rows**, adding **70,453 identity rows** relative
to the sealed baseline, including **31,858 already eligible observations**. The
1,772 inactive removals and 15,257 provenance differences are unchanged. Every
row of the earlier 943,419-row overlay remains exact. The additional 6,120 full
calendar rows concern only MGLU and ABC; 5,587 are already eligible. No universe,
quote or historical-security axis changes here.

MGLU's original [April29 2011 offering announcement](https://ri.magazineluiza.com.br/List/Download.aspx?Arquivo=1nnQI3g28rWgPlec29FzWA%3D%3D)
explicitly identifies MGLU3, BRMGLUACNOR2 and ordinary shares; its second page
starts trading on the next business day, May2. Twenty-six exact FCA amendments
restore omitted tickers/listings or correct the ON/PN conflict. Original source
bytes and the before-fields remain preserved; there is no general class override.

ABC's [original fourth-quarter 2007 release](https://api.mziq.com/mzfilemanager/d/6298ef6f-2b75-43f8-b2ab-99e3fe33e809/80dc0629-c461-4a77-8677-4c914b7313fd?origin=2)
was recovered from its own 2007-only archive and visually verified. The document
is dated January30 **2008**, explicitly names ABCB4, and is conservatively known
January31. The archive's December30 quarter metadata is not its release time.
Thirty-two exact FCA amendments combine that issuer ticker, original dated
CNPJ/CVM records and prior COTAHIST BRABCBACNPR4/PN observations. The FCA's PNA
wording remains in the source evidence; its exchange-class join is normalized
only for this permanent security. This does not assert a legal class conversion,
use today's FAQ, or admit the unavailable 2007 prospectus bytes.

Externally corrected securities carry an exact ISIN constraint as well as the
already implemented evidence clock. Another security cannot inherit a corrected
ticker. Each original filing's own receipt and sector clock remain unchanged.
Seventy-one exact source records are now listed, including the original thirteen.
The new global reconstruction and future-source-deletion prefix took 70.09s;
its recorded manifest supplies the exact timing. The prior sealed baseline proof
was reused, not recomputed. Financial/event regeneration reused all 427 newly
extracted own-version filings and took 119.11s. Their prior valid values are exact.
The table and totals above include all identity repairs, not just this increment.

Identity-only dependent propagation holds each family's original market inputs
fixed, including the different original oil/action-return contract. It reuses
archived market shocks rather than downloading or reparsing market sources.

| Dependent input | Added usable eligible values | Changed existing eligible values | Lost |
| --- | ---: | ---: | ---: |
| Sector-relative returns and sector momentum | 91,468 | 283,551 | 0 |
| Six cross-market exposures and their shock interactions | 0 | 2,060,611 | 0 |
| Loan utilization proxy | 2,652 | 0 | 0 |

Peer changes can affect stocks whose own identity is unchanged: issuer-equal
sector means and leave-one-issuer-out exposure shrinkage use the corrected group.
Every coordinate outside the affected old/new sector groups agrees exactly.
All non-exposure cross-market fields, including shocks, ADR gaps and foreign
flow, agree exactly across the entire family. All six other lending fields and
ages remain exact. Borrow rates, balances, publication timing and float formulas
were held fixed for this attribution; this is not the recovered-lending-data
feature amendment. No source observations or eligible names are removed.

Propagation took 124.36s. An output-type oversight initially compared Float64
utilization calculations with the old Float32 storage, creating tiny apparent
differences (maximum 1.39e-8). Restoring the existing storage type makes every
shared valid utilization value exact. The initial output/reproducer is retained;
only the typed utilization output and its comparison were regenerated. Passed
sector/cross-market work was not repeated. This was a readout-harness issue,
not an economic finding or a change to the old store.

Eleven affected identity tests and Ruff passed, including exact-ISIN binding,
future-source isolation and separate metadata clocks. The new global prefix and
prior-overlay preservation checks pass. The registration was reread. Timings
are CPU data work, not fit estimates. Source receipts, original and corrected
reproducers, manifests and parquets are bound in the canonical run pointer.

## Remaining boundaries

A/B remain incomplete and C/D unstarted; no new GPU fit or corrected profitability
result exists. These parquets have not replaced the accepted model store or been
attached to old checkpoints. Financial numerator/TTM/denominator and free-float
source audits, other original publication/revision checks, ALLOS/ISA continuation,
full universe/warmup/history/wealth/labels and actual changed-store neural tensors
remain. The peer propagation isolates identity with old market inputs; final
accepted wealth/history changes require explicit further propagation. Historical
single-vintage revision fidelity remains unknown where no earlier vintage exists.
