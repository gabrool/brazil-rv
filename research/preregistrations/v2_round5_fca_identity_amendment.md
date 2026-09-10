# Round 5: recover original cadastre versions and preserve source semantics

This source-quality amendment is authorized by the user's instruction to recover
usable data completely and choose the recommended implementation autonomously.
It changes source coverage and identity metadata before the final Round-5 store;
it does not select features or change a fitted model using outcomes.

The annual FCA files retain every header but generally only a later version's
detail. Missing original details are recovered from the exact public document
ID, preserving reference, version, CNPJ, CVM registration, original bytes and
receipt. Later detail is never applied at an earlier header's receipt. Failed
original retrievals remain recorded and cannot silently become verified data.

Original XML security descriptions govern historical classes. Legacy `Ações`
is generic: a modern HTML rendering such as `Ações Ordinárias` must not turn it
into historical ON evidence. A generic share row may map only to cash ON/PN
classes independently observed in prior COTAHIST rows, through a unique exact
contemporaneous legal-name/CNPJ association. Units are excluded from this route.
Observed preferred class suffixes remain explicit, so an aggregate PN capital
count cannot be relabelled PNA. Explicit original ON/PN constraints remain in
force. Existing security birth, legal succession, dated ticker and preannounced
listing rules continue to apply; no current survivor filter is introduced.

Matched original/annual source IDs establish the converter problem: Vale's
2010-v4 (2190) and WEG's 2015-v2 (44425) contain generic original `Ações`, while
the exact annual rows render ON. Their segment labels also differ. Vale's
2018-v5 (83236) instead explicitly contains ON and its ticker, agreeing with
the annual row. Thus there is no blanket calendar cutoff or blank-ticker rule
that loosens modern explicit classes. Surviving blank-ticker cash documents in
the relevant cohort enter missing-only original retrieval; original class
verification gaps remain reported individually.

Exchange listing and cancellation bound the association. The original
`DataInicioRelc`/`DataFimRelc` and annual CSV `Data_Inicio_Listagem`/
`Data_Fim_Listagem` are these dates. `DataInicioNeg`/`DataFimNeg` and CSV
`Data_Inicio_Negociacao`/`Data_Fim_Negociacao` describe negotiation beside the
segment field, and do not restart an already listed issuer's identity. Both
source date pairs are retained in provenance. WEG's 1982 listing versus 2007
Novo Mercado entry and Vale's 1968 listing versus 2017 segment entry establish
the distinction in the retrieved source tables.

Sector grouping uses the original numeric FCA sector code immediately at that
document's receipt. Exact-ID HTML labels are separately bound annotations. No
independently fixed historical taxonomy has been established: empirical
code/label evidence is therefore part of the historical information set. An
annual label lacking a code may use only unambiguous exact-ID code/label
observations already received by the current decision. All-history unambiguity
is insufficient: a later observation could otherwise fill or erase an earlier
sector. Unsupported or conflicting translations remain missing. Numeric codes
and display labels are never mixed as grouping keys.

The existing decision sweep adds mapping evidence at its first eligible receipt
session and re-resolves active annual records each session. A newly available
mapping helps immediately, without waiting for the issuer to file again. A
later conflict may remove a translation only from that decision onward; it
cannot change earlier sectors or override a document's explicit numeric code.
`sector_mapping_id` binds the earliest known exact source document for the
translation; `sector_known_date` is the later of that source's availability and
the issuer classification's own availability. Direct codes use their own
document and receipt. Missing sectors have neither provenance field populated.
The complete source inventory also retains mapping-provider documents without
a cash-security row. Source acceptance verifies these bindings and the absence
of already-known contradictory evidence.

The issuer's original display label remains available for financial-accounting
classification at its own receipt, together with the contemporaneous statement
chart. Numeric translation does not delay that known label. Sector-relative
feature ages remain the completed-return endpoint's one-session age; shrinkage
ages remain the oldest latest observed fit-pair age among contributing names.
Classification and translation availability are recorded separately and gate
both consumers; a new translation never makes an old market measurement appear
new. Actual producer/parquet/FeatureSpec fixtures must keep earlier values,
masks and ages exact under future-only and future-conflicting mapping mutations,
with the first change at the mapping's first eligible decision.

Final admission records source manifests and the complete dated identity
document provenance. It reports new/removed/changed identity name-days relative
to the earlier 9120d7d archive, rather than requiring equality to that incomplete
bridge. Receipt mutation, exact-name ambiguity, share-class, listing expiration,
new-security birth and successor tests remain required. A changed bridge requires
rebuilding every identity-dependent sector, cross-market exposure and lending
utilization archive before assembling one final store. The prior bridge and all
derived artifacts remain immutable.

The source freeze remains held until original retrieval failures, surviving
annual converter semantics, sector translations and any newly exposed missing
financial/capital source versions are audited. Acquisition may run independently;
no final family or model fit may bypass this source-coverage decision.

The original FCA's full CNPJ is retained as `source_cnpj`. Its establishment
suffix may differ from the current public envelope while the legal eight-digit
root and exact CVM registration remain identical, as in Paranapanema FCA359:
original60398369000126 versus public-header60398369000479. The exact public
ID/version/reference/envelope identity remains required. The nested original
and exact-ID HTML use the same legal-root-plus-CVM contract already used by
the dated identity bridge; a changed legal root, missing CNPJ or changed CVM
registration still fails. No legal-name alias or issuer-succession merge is added.
