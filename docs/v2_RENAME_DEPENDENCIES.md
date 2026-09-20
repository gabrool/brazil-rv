# Rename dependencies and independently valid daily activity

This Stage B milestone continues the three admitted ALLOS/ISA unit/no-cash
links through issuer identity, valuation, financial/event observations, sector
peers, physical magnitudes and cross-market fields. It also admits printed
activity from GOLL's inconsistent 2011-02-16 OHLC row. These are intermediate
derived contracts. The sealed store, earlier fits and profitability results
remain unchanged; A/B are incomplete and C/D have not started.

Resolve `rename_issuer_propagation`, `goll_activity_admission`,
`rename_dependents`, `activity_magnitude_propagation` and
`dependency_update_audit` through `v2_economic_data_scaling_run.json`.
The preceding daily/history/input acceptance and all earlier source audits
remain valid for their recorded inputs and commits.

## Exact issuer continuity and valuation

The FCA bridge already identified ISA's preferred predecessor and ALLOS's
predecessor. Dated unit-renaming evidence now carries their exact CNPJ, CVM,
class and preferred subclass across the boundary. Effect and knowledge clocks
both apply; a ticker resemblance cannot establish this mapping. Later sector
metadata retains its separate receipt clock. Existing successor mappings must
agree on issuer/class and remain unchanged; conflicting mappings stop.

The overlay adds31 rows:30 eligible issuer-days and one ineligible ISA ordinary
day. It retires two stale inactive ALSO rows and loses no eligible mapping.
All shared identity rows remain exact. Deleting later input rows preserves two
complete earlier prefixes. The resulting identity overlay has949,568 rows.
This specifically supplies ISAE4's28 sessions and the first twoALOS dates;
ISAE3 keeps its information without overriding the existing liquidity rule.

Valuation also needs the predecessor's prior price on the rename decision.
The valuation consumer now follows only admitted unit-history links, preserving
both the prior class price and predecessor capital-uncertainty barriers. It
cannot use a future-known link or erase an earlier distribution/split barrier.
This does not infer an ex-date loan reference or BDI loan alias.

Only the two affected issuers' financial/event paths were recomputed, reusing
the previously extracted and audited filing cache. The31-row identity repair
adds240 usable values across eight financial fields,30 statement-age values
separately, and210 values across seven event fields. Existing valid numbers and
all unaffected issuer histories remain exact. No original-source census or
full FCA reconstruction was repeated. This step took21.00seconds.

## Printed activity without an invented price

The immutable GOLL4/BRGOLLACNPR4 row for2011-02-16 prints open23.20,
high23.38, low23.38 and close23.46, which fail the existing OHLC bounds.
Its separately printed turnover isR$21,129,704, quantity908,400 and2,743 trades.
The panel now retains these activity fields while keeping all rejected prices
unobserved. Activity and units are checked independently because the first
price rejection can hide a second activity/currency/quote-factor defect.

The sealed store represented this complete-source row as zero activity. The
new amendment keeps its valid-activity mask, changes the printed activity and
observed-trade flag, and leaves eligibility unchanged. No20-session exclusion
is introduced. The four affected slow-field controls reproduce74,640 old
cells exactly before admission. The change affects471 ranked values across
124 names over the following20 decisions:65 log-volume,117 volume-z,
192 trade-count-z and97 turnover values. All slow masks/ages remain unchanged.
The invalid return endpoint still prevents a spurious Amihud observation.

The physical log-volume magnitude changes on20 GOLL decisions. Its2011 source
period has no supported foreign-flow interactions, so this activity change
adds no invented flow data. Quantity-dependent remaining families must consume
the explicit activity amendment when the complete store is assembled.

## Dependent families and preserved boundaries

The already verified daily continuation basis supplies historical observations;
public successor coordinates still do not exist before birth. Sector and
cross-market calculations use all933 permanent axes, the restored eligible
population and their unchanged support requirements. Only the changed tail
and sufficient preceding history are calculated. The original oil return
exclusion contract remains distinct from the original non-oil contract.
Historical CDI in these frozen feature contracts is unchanged; the explicit
accounting cash-calendar amendment remains separate.

Versus the preceding FCA family overlays (and the sealed magnitude family),
using the new eligible calendar:

| Family | Added valid values | Lost valid values | Changed shared valid values |
| --- | ---: | ---: | ---: |
| Eight financial fields | 240 | 0 | 0 |
| Seven event fields | 210 | 0 | 0 |
| Three sector fields | 457 | 0 | 3,648 |
| Four physical magnitude fields, including GOLL | 352 | 0 | 662 |
| Cross-market fields | 6,508 | 0 | 19,111 |

These are increments over the earlier overlays, not additional totals to add
to their sealed-baseline comparisons. Changed shared magnitudes include the
new continuation basis, beta history and20 GOLL activity observations.
Common shocks/flows retain their original publication clocks. Their exact
equality across previously active names is verified before routing them to
newly eligible names. Source ADR pair declarations establish the absence of
an admitted pair for these exact successor ISINs; no current classification
or inferred overseas listing is used. Public loan aliases are not changed.

The audit proves exact values/masks/ages outside the affected old/new sector,
issuer and activity scopes. It also checks the actual dated-family loader on
12 dates/all933 names:3,090,096 value/mask/age cells across92 aligned fields
match an independent date/ISIN/field reconstruction. This is the intermediate
family-alignment boundary, not a new neural forward or final-store acceptance.
The dependency build took91.31seconds; final scope/alignment audit3.18seconds.

## Qualifications and remaining work

The first activity audit used the replicated source-completeness matrix where
the universe function requires a session vector. Correcting that audit-axis
selection produced the unchanged-universe proof; source values were unchanged.
The initial rename magnitude calculation promoted volume toFloat64, introducing
four differences on already valid ALOS observations. A narrow log-volume/flow
qualification restored the originalFloat32 arithmetic without repeating passed
sector, volatility or regression work. Its first one-column reduction exposed
NumPy's different pairwise-sum ordering; the final strided reduction matches the
original full-population producer exactly after the inherited window ends.
Initial executed source and output bytes are retained with the final qualification.

Thirty-six affected source/identity/valuation tests pass, including independent
activity defects, late-known/conflicting renames, sector clocks and retained
capital barriers. Ruff passes. No passed accounting or source-census suite was
repeated. Accounting V31 and the CNPJ assumptions remain unchanged.

Next: propagate recovered lending and remaining M1/other auxiliary families,
finish upstream clock/revision qualifications and contractual wealth/label
terms, then accept the complete derived store with final tensors and an explicit
refit contract. Do not score old checkpoints on these changed coordinates.
Stage A still requires the registered execution/lifecycle sensitivities; no
corrected profitability, lead replication or capacity result is claimed here.
