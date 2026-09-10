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

Sector grouping uses the original numeric FCA sector code. Exact-ID HTML
labels are separately bound taxonomy annotations. An annual label lacking a
code may be translated only when matched original code/label observations give
one unambiguous code. This translates vocabulary, not a later issuer-sector
assignment. Ambiguous or unsupported translations remain missing and enter the
source recovery audit. Numeric codes and display labels are never mixed as
grouping keys. Financial-accounting classification may use verified sector
semantics together with the contemporaneous statement chart.

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
