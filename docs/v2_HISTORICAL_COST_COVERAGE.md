# Historical cost coverage and corporate custody integration

2026-09-20. This closes the earlier tariff-coverage and corporate custody
implementation gaps within the Stage A closeout plan. Spot invoice treatment,
two calendar ambiguities, event/data composition and integrated admission still
remain. Stage A is not accepted; C/D have not started. Both accepted stores and
all original fits remain immutable.

## Dated exchange costs

Both actual accounts now use `b3_spot_dated` for July18 2016–December30 2024.
It replaces the bundled execution cost. Ordinary-CNPJ clearing is 2.75bp before
February2 2021 and 2.5bp thereafter; trading uses the old previous-month global
market schedule, then 0.5bp. Brokerage remains zero and shortfall1bp. Auction
trading0.7bp is the existing separate hypothesis. No fund discount, client-capital
ADTV substitution or extra B3 charge on bundled4bp is admitted.

Original018/2013 and061/2013 establish the old progressive global-market tariff
and second-business-day implementation. Fifty original monthly TX archives supply
the published rates; independent XML traversal qualifies all50. Across2099
development sessions,1014 use recovered monthly rates,976 use the fixed2021
regime, and109 have an explicit source-supported range because the required
monthly record is absent or not yet conservatively available. The primary upper
endpoint0.5bp and limiting lower endpoint0.2bp apply only to those109 cells.
The latter is an asymptotic tariff bound, not an observed monthly rate.

Availability preserves each XML creation/date/validity boundary. In particular,
the recovered November2016, December2016 and January2017 files are not backdated
to their nominal monthly dates. XML creation does not prove first internet
publication or revision completeness. Mapping of the old message's group/type
codes is corroborated by later technical documentation and remains an explicit
cross-version interpretation: the own-vintage external code list was not
recovered. The2025 technical presentation supplies schema semantics only;
its rates and market observations are not consumed.

Original082/2009,144/2015,120/2016,011/2017 and101/2018 establish earlier custody
brackets and annual maintenance updates. The old annual progressive value rates
are0.013%,0.0072%,0.0032%,0.0025%,0.0015%,0.0005%, divided by12 at month end.
The literal R$300000 exemption is inclusive under082/2009 and strict under
101/2018; this does not establish an intentional policy change at the exact
boundary. Active resident maintenance below/above R$5000 is7.59/8.02 in2016,
8.18/8.65 in2017,8.40/8.88 in2018 and8.78/9.28 from2019 toFebruary1 2021.
One open, active account/custodian is an explicit hypothesis, including a zero
stock balance. Existing177/2020 and later threshold proofs are reused.

Nine new tariff/technical PDFs have33 visually qualified pages. Fifty recovered
monthly files coexist with saved empty responses/timeouts; no missing monthly
rate is invented. The original monthly archive request selection, all retrieval
attempts, receipts, parsers and independent qualification are recoverable.

## Corporate custody

Both accounts preserve ordinary physical settlement and now distinguish delivered
stock from positive undelivered corporate rights. Delayed bonus receipts retain
separate dates from original shares and pending purchases. Conversion routes
pending quantities through each sourced successor; loan cash extinction removes
the loan obligation without inventing physical stock. Actual pending cover stock
still arrives on its original spot date. Negative fractional obligations are not
negative physical custody. Unknown ALSC credit retains the entire locked claim.

The charged primary base includes separately valued positive rights; exclusion
is a one-factor hypothesis. Neither is described as an observed B3/broker
invoice rule. Own closing/last available prices and the already-qualified local
opening-claim valuation remain explicit valuation hypotheses. Physical/rights
bases, maintenance, total fees, payments and unpaid liabilities are separately
reported. Assessment-close client payment retains the prior funding clock;
earlier payment3/10 bounds are reused. A general cash cancellation with live,
unconverted share loans still requires explicit loan terms.

## Verification and economic limits

Eighteen new historical-cost books cover October24 2016–February7 2017 and
January26–February8 2021, all933 names and R$10m/R$1m/R$5m. They compare the
old bundled bridge, dated costs/custody, and the missing-rate lower endpoint.
All completed on their first invocation. Independent747NAVs/3513141saved cells,
9986fills and30assessments qualify. NAV error is at mostR$3.73e-9; current-code
qualification reuses498 saved fee-enabled states, with no account/book replay.

Thirty-four new corporate cost books cover the combined ALSC/Natura2019,
BRML/Dommo2023, Copel2023 and SOMA/ENAT2024 interactions. Twenty-four primary
books are followed by ten rights-exclusion books; fourteen variants are skipped
because the primary has no assessed positive rights. All34 completed first-pass.
Independent1888NAVs/8883040saved cells,40267fills,111assessments and1212 sourced
entitlement checks pass. Long successor arrivals plus actual fills are exact;
maximum NAV error isR$5.59e-9. The independent physical-flow oracle covers
1755483 noncorporate cells. Corporate component reconciliation additionally uses
the previously qualified loan subledger; it is not mislabeled as an independent
loan reconstruction. Seventeen focused corporate tests cover pending flows,
bonus, multiple legs, fractions, offsets, cash closeout and independent training
copies/gradients. Four new historical-cost tests and72 affected existing tests
pass; overlapping batches are not added as distinct tests.

All books use frozen synthetic preferences and fixed engineering risks through
the actual constrained allocator, shallow-copied OLD PolicyData, correctedCDI,
qualified loans and strict prior references. There is no neural scoring or fit.
Same-intention money agrees withinR$1.31e-8. Independently adaptive corporate paths
differ by up to0.0387828161/0.2010558935/0.0209277910bp atR$10m/R$1m/R$5m;
retain larger earlier uncertainty as well. Several fee contrasts are smaller
than this uncertainty, so their signs do not establish an economic improvement.

AtR$10m, the2016 missing-rate lower endpoint changes final synthetic NAV by
+0.4616858885bp. Corporate rights exclusion changes the2019 long-focus final
path by+0.0016627681bp, withR$1.7651817971 less total custody expense;2023/2024
R$10m effects are below0.000001bp. These are total synthetic path contrasts,
not model alpha, pure direct charges or all-interior adaptive extrema. Every
capital/path result is saved. None of the34books has an actual Cielo loan cash
redemption. Its held-loan cent sensitivity remains conditional and unqualified;
no rate, preference, availability or quote was changed to force exposure.

CPU runtimes: historical books20.9745s; corporate books72.9160s; independent saved
qualifications0.7587s and2.6213s. They are not GPU-fit estimates. The time-consuming
work was source retrieval/interpretation and integration. No earlier completed
engineering matrix, source census or accepted-store assembly was repeated.

Initial attempts are retained. A maintenance scalar probe caught Float32 creation
of fixed amounts before any historical book; the implementation now preserves
Float64. Corporate test failures were fixture precision/expected D1-loan receipt
issues, corrected without production changes or historical reruns. The first
corporate saved qualifier encountered an empty current-price comparison on a
zero-stock date; the corrected reduction reused all books. An older report's
copied limitation prose was corrected separately; all numerical fields remain.
Read-only path/wildcard errors and an overwritten early download-failure receipt
are described as reconstructed records, not exact original shell artifacts.

## Calendar and remaining admission

One additional original,047/2017-DP page1, proves that the former equity clearing
house ended operations August25 and equities migrated to the named multiasset
clearing on August28 2017. Combined with the already-qualified111/2016 holiday
table, this resolves November20 2017 as a closure of the equity clearing then in
operation. This is an inference from the two dated notices, not a newly observed
settlement receipt. No calendar array changes. December30 2016 andJanuary25 2017
remain unresolved; the targeted2016 index request timed out and is preserved.

The next work is applicable spot invoice rounding/daytrade disposition, those
two calendar boundaries, composition of already-qualified primary event terms
and separately attributed succession-data implications, followed by integrated
StageA admission. Unknown client terms can remain bounded hypotheses. Unknown
physical receipts cannot become invented custody dates. Exact broker invoices
and unknown historical web revisions are not new prerequisites. C/D remain
unstarted, and this document makes no corrected-model profitability claim.
