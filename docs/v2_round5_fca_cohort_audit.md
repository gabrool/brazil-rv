# Round 5: original FCA acquisition cohort audit

The earlier dated identity archive covers 403 root-CNPJ/CVM registrations.
Restricting all source recovery to that archive would miss 12 plausible source
registrations. The bounded audit uses the fixed 933-ISIN store axis, its dated
2009–2024 COTAHIST cash observations, and already downloaded annual FCA
headers/general/security rows. It reads no later market payload and performs
no fuzzy identity matching or model fit.

The existing exact legal-spelling normalization finds 414 source registrations
through FCA names or observed historical tickers. Reconstructing the earlier
accounting acquisition rule reproduces 22,561 financial headers, 415 exact
CNPJs and 412 root-CNPJ/CVM registrations. The two CNPJ counts use different
units; 415 was not the old identity's registration count.

Their union exposes these 12 source-retrieval candidates outside the old bridge:

| CVM registration | Source name/evidence | FCA versions | Missing annual detail | Prior observed exact name |
|---|---|---:|---:|---|
| 020249 | Klabin Segall | 1 | 0 | Yes |
| 017434 | Longdis | 35 | 20 | Yes |
| 017779 | Maori | 17 | 10 | Yes |
| 017493 | Capitalpart, historical ticker | 27 | 12 | No |
| 025585 | CSN Mineração | 8 | 3 | Yes |
| 022110 | Prior accounting acquisition CNPJ evidence | 10 | 6 | No |
| 008818 | Minasmaquinas, historical ticker | 25 | 10 | No |
| 027677 | Atom Educação e Editora, historical ticker | 2 | 1 | No |
| 027413 | Automob, later observed exact name | 3 | 1 | No |
| 024848 | C&A Modas, historical ticker | 11 | 5 | No |
| 004774 | Construtora Lix da Cunha, historical ticker | 19 | 10 | No |
| 014931 | Dixie Toga | 2 | 0 | Yes |

All 160 FCA versions were uncached at audit time; 78 lack their own annual
detail. They have been added to the queued missing-only supplement, alongside
2,036 surviving blank-ticker versions. The 2,196-document union remains
disjoint from the healthy primary collection.

This is acquisition candidacy, not admission of a historical identity. A later
exact name or an old ticker does not prove the earlier issuer association.
Original receipt, unique contemporaneous legal name/CNPJ, exact security class,
listing dates, instrument birth and legal succession still govern every row.
The audit leaves 828 FCA registrations without exact axis-name, ticker or prior
accounting-cohort support explicitly non-admitted. It does not claim all-name
source completeness; aliases requiring additional independent evidence remain
outside this bounded check.

Eleven candidate registrations have 344 financial headers in total. These are
a prospective source checklist, **not 344 missing documents**. Most candidates
already lie in the broader previous accounting acquisition cohort. After the
new identity actually admits registrations, compare their own annual account
and capital contents with verified original caches and retrieve only remaining
gaps. Do not acquire or backdate financial features solely from this candidate
list.

Local evidence is under
`D:/quant-data/b3/interim/round5_data_20260910T140442Z/cvm/`:

- `fca_cohort_audit.json`: exact source bindings, candidate evidence and the160
  retrieval documents; SHA-256
  `5ca423579e3ab3ab0fc75e299f80eef351acf37d4733d849f5acd4a01b70b4bd`.
- `fca_cohort_financial_candidates.json`: prospective344-header checklist,
  explicitly conditional on actual identity admission.
- `fca_retrieval_supplement_inventory.json`: queued2,196-document union;
  SHA-256 `72d01134b2c280126f33e6d210cb64a113f232ac33ffdc782603373f0e3b03fb`.

Capital failures also require a finite source-quality outcome. For documents
82925, 88901 and 86096, the preserved HTML attempts contain missing/non-numeric
quantities. After bounded exact HTML/original-ZIP investigation, a genuinely
unavailable count may receive an explicit audited-unavailable disposition with
exact document identity, reason and source hashes. Unaudited failures still
block admission. No blank becomes zero; a document without capital contributes
no count, and valuation remains missing where no other independently valid
known count satisfies the existing class and unit-boundary rules. Unrelated
accounting features remain usable. The concrete implementation plan is
`capital_unavailable_disposition_plan.json`; it approves no disposition itself.

The primary 6,768-document acquisition and 2,196-document supplement are now
complete. The primary batch had 6,761 successes, six unavailable results and one
interrupted read. A bounded pass reusing original responses recovered Klabin476,
Globex4929, Fica7099, Bradesco34107 and Tupy48539. Tupy uses the already tested
historical CNPJ establishment rule; all original failures remain preserved.

Two remaining packages have explicit source dispositions. Light125223 contains
a single FCA XML encoded as CP1252 despite its UTF-8 declaration. Its two issuer
blocks, year and version match the public envelope. Literal XML supplies LIGT3,
the 12December2005 listing date and sector3120. Same-package PDF pages2/4 confirm
the equity/stock-exchange category. The admitted class remains generic SHARES;
the export's modern ordinary-share label cannot replace dated B3 class proof.
No broad numeric-enum decoder was introduced for this single source.

Soma91233's nested registration091243 is a registration applicant. Its own
securities XML and same-package PDF contain only an expired debenture listing.
The disposition preserves that source code and emits no equity listing. It does
not establish an alias to equity registration025011 or invent SOMA3. The exact
public envelope, original source bytes, reviewed pages and derivation are bound
in [the completion evidence](v2_round5_fca_completion_audit.json).

The first full source preflight completed against clean51fc446:821,266 dated
identity rows,398 registrations,22,349 financial documents and no remaining FCA
source failures. Its twelve capital gaps are recovered. The single account gap,
Santander133758, has usable own-version individual statements while the original
consolidated pages explicitly defer publication. The separate group is now
recovered without altering the old cache or borrowing a later release; both
manifests enter provenance. See [source-gap evidence](v2_round5_source_gap_audit.json).
Recompute capital selection with these sources before the remaining review.
Source support does not itself clear the final source/store hold.
