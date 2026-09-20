# CVM original receipts, capital tables and C&A identity recovery

This is Stage B of the economic/data/scaling program. Resolve the source and
overlay receipts through `v2_economic_data_scaling_run.json`. Accepted stores,
old fit roots and static policy coordinates remain unchanged.

## What reconciles

The independent original-source audit verifies all 21,783 cached ITR/DFP document
identities against annual headers: public ID, issuer, reference, version and own
receipt. Structured RAD receipts agree for 21,767 documents; 16 retain date-only
availability. Minute precision uses its upper bound before the B3 decision. These
receipts independently reproduce all 880,858 financial statement-age observations.

All 21,749 original HTML capital tables reconcile, including the distinct share
quantity units (11,653 in shares; 10,096 in thousands). Paid and treasury counts,
reference periods, viewer/group IDs and source hashes agree. The 184 previously
reviewed own-note dispositions retain their exact filing identities and arithmetic;
this is not a second interpretation of every note. The audit verifies 109,777
source files and reports zero mismatches in 96.30 seconds.

Thirteen remaining positive capital counts reconcile independently against their
original ZIP envelopes and relational/flat capital records in 0.83 seconds. A
filename selector in this audit initially included a cadastral envelope as a
financial payload; fixing the audit selector revealed no model defect.

Two missing C&A tables were recovered from their exact CVM filing pages. ITR
96365, reference March2019/version2, supplies 1,035,720 ON shares and zero treasury;
its August2020 receipt first permits use on August21, never in March2019. ITR
114352, reference March2022/version1, supplies 308,245,068 ON less 214,500 treasury,
or 308,030,568 outstanding; first decision is May6 2022. Their incremental feature
effect is zero under both the original and repaired issuer joins. They are useful
source completion, not an attributed performance gain.

## Actual defect and implemented repair

C&A (CNPJ 45.242.914/0001-05, CVM024848, CEAB3/BRCEABACNOR1) had only 140 dated
issuer rows ending May26 2020, although it remained model-eligible. Subsequent
cached FCA fallbacks omitted the literal ticker. Exact-name recovery could not
match the legal C&A spelling to the B3 short name. Explicit generic-share tickers
were also unnecessarily forced through that name fallback.

The current parser now reads the literal ticker within its exact HTML security
block. It also supports recovered flat FCA XML packages, validating the outer
public filing and the nested issuer/year/version. These packages can contain
Windows-1252 bytes despite declaring UTF-8; strict decoding records the choice and
never edits the source. Numeric type1 stays generic shares. Neither a contemporary
HTML/PDF rendering nor the numeric enum alone assigns ON/PN. The dated original
ticker joins only prior B3 observations; those establish class, preferred suffix
and permanent identity. Conflicting issuer assignments still fail. Later ticker
births need the separately disclosed listing boundary, and generic shares cannot
become units. No name-similarity heuristic was added.

Ten bounded retries recovered useful additional C&A originals. Requests returning
the CVM download-service error remain preserved. Where a ZIP is unavailable,
the exact-version HTML supplies its literal ticker without promoting modern class
labels. The source loader reparses hash-verified bytes without overwriting source
manifests. One initial admission-report write failed on the Windows junction's
path spelling after all data checks; its partial outputs remain preserved and the
successful receipt resolves the canonical path explicitly.

The explicit C&A overlay now contains 1,286 issuer rows, adding 1,146 eligible
sessions from May27 2020 through December30 2024. All 140 existing identity and
financial rows agree exactly; no existing row is lost. Deleting future FCA
documents leaves earlier output unchanged. Original receipt indices remain fixed.
Previously available sector codes establish the mapping before each recovered
record; no new earlier global sector translation is introduced.

| Financial field | Additional usable values |
| --- | ---: |
| Log market capitalization | 1,110 |
| Book to market | 1,110 |
| TTM earnings yield | 1,110 |
| Gross profitability | 1,146 |
| Liabilities to assets | 1,146 |
| Accruals to assets | 1,146 |
| Year-over-year revenue growth | 1,146 |
| Standardized earnings surprise | 664 |
| **Total across these eight fields** | **8,578** |

The overlay also recovers 1,146 statement-age observations and the separately
reported dependency ages/flags. It changes financial coverage for already eligible
stock-days; it does not add 1,146 universe members or new eligibility days. Paired
identity/financial generation took 15.50 seconds. Financial formulas are held fixed
for this attribution, not newly certified by the paired calculation.

## Admission and remaining work

The canonical run pointer binds the C&A identity, FCA-document and financial
overlays for the next derived-store contract. It also binds a bounded reread of
the 2,820 previously fallback-only FCA records to measure the repair's wider scope;
the final reread parses all 2,820 with zero failures and identifies changed metadata
in 1,068 documents across 327 issuers. It recovers 283 original XML packages from
existing caches. Seventy-one pages initially exposed multiple securities within
one rendered class tab; the parser now associates each literal ticker with its own
preceding market/date row. The final affected reread took 9.97 seconds. Those wider
candidates need global identity/ambiguity and causality checks before admission.
No source download or completed quote/lending/tensor census was repeated.

Twenty-four focused parser/identity checks pass, including encoding, exact nested
version, unsupported source types, literal HTML ticker, conflicting issuers and
future ticker births and multiple-security row association. The historical overlay
adds exact baseline and future-FCA deletion evidence. Its ten reloaded source
metadata records remain exact after the row-association repair; the data replay was
not repeated. The new source and derived artifacts are separately hash-bound.

Still required: global identity and auxiliary-family propagation, financial account
numerators/TTM/denominators, other original-source clocks and revisions, complete
universe/warmup/wealth/labels, then a separately accepted full store and compatible
refits. A single retrieved CVM vintage cannot certify the converter's historical
revision behavior. Accounting Stage A's remaining execution bounds also remain.
These source gains are not model bps/day, and C/D have not begun. Foundation results
stay sealed. No GPU fit, old-checkpoint rescore or held-out consumer read occurred.
