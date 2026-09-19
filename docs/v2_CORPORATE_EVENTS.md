# Corporate event admission

Source receipts are in `v2_corporate_loan_sources.json` and the sealed foundation
primary-source manifest. This document separates a verified contractual rule from
an event admitted to a historical account. No new historical profitability result
has been accepted. Model stores and serialized feature coordinates remain unchanged.

## Cielo: loan closeout and shareholder redemption are separate

B3 circular 011/2024-VNC, dated 2024-07-25, specifies on pages 2–3 that outstanding
CIEL3 loans close in cash four business days after the final trading date. Returns
due on or after that date are superseded. Original contract references determine
rent and fees through closeout, paid at closeout. New borrowing is then prohibited.
This is a loan obligation; it does not redeem a long shareholder's stock.

The archived issuer auction result (CVM protocol 1272035, 2024-08-14) establishes
R$5.82 per share and auction settlement on August 16. The original offer document,
obtained through its embedded issuer link in CVM protocol 1258856, distinguishes
CDI adjustment before the auction from SELIC correction afterward (sections 8.1.1
and 9.1.1). The August 26 conversion announcement and the accepted COTAHIST-backed
close series agree on the final trading date. Thus the B3 loan date is August 30.

The bounded SGS 11 retrieval contains 30 daily SELIC observations from August 16
through September 26, 2024. Applying the published daily percentage rates to R$5.82
from August 16 inclusive to August 30 exclusive gives **R$5.842895570784521**.
This is a source-derived continuous calculation, not an obtained B3 invoice price.
Retain the continuous convention and a one-cent per-share payment bound until
invoice-level rounding is established. Substituting CDI or a future redemption
price is not justified. Extending the same calculation to September 26 yields
R$5.886909802213979, which rounds to the issuer's stated R$5.89 payment.

The September 23 redemption announcement (CVM protocol 1284473) establishes that
shareholders then holding stock receive R$5.89 on September 26. That amount must
not appear in an August valuation or decision. Shareholders could separately elect
the offer's documented post-auction sale procedure, with custody transfer and a
payment deadline; do not invent an automatic immediate exchange fill or choose
the favorable route after seeing outcomes. The default unrequested shareholder
path retains inventory until the announced compulsory redemption.

Both accounts now implement the distinct compulsory loan cash event. The loan
principal payment is separate from borrowing expense. Interest and exchange fees
retain their original accrual basis; settled restricted proceeds are released and
pending external cash value dates are preserved. A covering purchase whose planned
physical return is superseded remains an asset, with its prior observed valuation.
The final cash price enters only the settlement realization, after that session's
intention. The event's effective prohibition prevents new shorts, including in a
flat-start replay begun after the event. Policy inputs keep their serialized static
coordinates; event terms are hashed as an accounting input in Evaluation V25.

The run pointer now binds a separate `corporate_replay` manifest. Its sparse loader
applies CIEL (axis 235, ISIN BRCIELACNOR3) loan settlement on August 30, resolves
known closed-register coverage from August 27, and recognizes the shareholder
cash cancellation on September 24 for September 26 payment. Recognition is after
the September 23 announcement, avoiding any earlier use of the later cash term.
The accepted store remains immutable. Only 86 coverage cells and one cell each
of q/cash/action/payment change; six of the coverage cells were eligible stock-days.
Quotes, membership, prediction coordinates and labels are unchanged. Both accounts
exactly reproduce the sourced cash flows on the actual August–September calendar
for predetermined long/short positions with zero fees/CDI, against closed-form
oracles. This is historical event arithmetic, not model profitability. See
`v2_corporate_replay_acceptance.json`. A saved `PolicyData` object must be shallow
copied and have only its inputs replaced; reconstructing it regenerates static
features and is forbidden for an accounting-only comparison.

The model-data consequences belong to Stage B. The general physical-custody
and holding-basis treatment of superseded purchases remains a separate boundary;
under Cielo's regular T+2 convention, final-day purchases settle before the D+4 loan
closeout. The synthetic superseded-return test establishes cash/NAV conservation,
not a claim to model every failed delivery or unusual loan instruction.

## BR Malls: separate clocks and a preserved loan principal

B3 circular 001/2023-VNC, dated January 3, 2023, binds the conversion to
0.398551577675763 ALSO3 plus R$1.62899410177968 per BRML3. The reference date is
January 6; BRML ceases trading from January 9. Loan conversion occurs at the end
of January 10, preserving the financial contract principal and rate; resulting
shares are available at the beginning of January 11. The cash leg settles January
20, including borrower-to-lender compensation through the B3 window. The document
explicitly retains fractional entitlements in loan quantities. Shareholder fraction
auction/payment rules require their own treatment rather than being inferred from
the loan convention. These terms are now admitted on BRBRMLACNOR9 (axis 169) and
BRALSOACNOR5 (axis 34). January 10 closing loan conversion is represented at January
11 opening, before decisions and that day's accrual. Principal/rate do not change
across that zero-session boundary. The issuer's final cash already includes its
projected CDI correction through January 13; January 20 is its operational payment
date. Do not add an invented correction to the announced final amount.

The newly archived January 25 issuer auction result establishes R$17.694416 per
fractional share, net of auction fees, available by February 2. Long shareholders
receive whole shares January 11; the remainder stays a priced, non-tradable successor
claim. January 26 recognizes the announcement conservatively at the next session;
February 2 pays at the announced deadline. Earlier receipt between January 26 and
February 2 is a bounded cash-timing sensitivity, not an invented exact sweep.
Until recognition, only available successor marks value the claim. Loan quantities
retain all fractions. Ordinary research fills still use continuous quantities;
auction rounding applies to the resulting beneficial entitlement. Integer order
sizing and unusual gross-versus-net custody registers remain distinct assumptions.

A recently purchased ALSO long can offset a converted short economically while
its loan and proceeds remain outstanding until purchase value date. The actual
January 10 purchase/January 11 conversion oracle pays one original-principal loan
portion January 12 and the remaining cover portion January 13. Both accounts and
closed-form rent agree exactly. Admission adds 494 BRML coverage cells (six eligible),
leaving the accepted store, neural arrays and serialized static policy coordinates
untouched. Long/short R$1m/R$5m/R$10m historical oracles also agree exactly. These
predetermined positions are not forecast-profitability results.

The admitted disposal rule waits for January 11 custody. The issuer also states
that the newly issued shares start trading January 9. That does not alone establish
this account's ability to dispose of its undelivered entitlement. Before final A/C
acceptance, resolve or bound a prearranged January 9-10 sale settling after credit;
do not present custody-first disposal as the only legally feasible route. This
affects execution opportunity, unlike the requirement for available shares when
physically returning borrowed stock.

## Remaining source admission

Complete the DMMO lender-election/default distinction, bound the admitted Copel
loan allocation in actual model books, propagate ALLOS/ISA dated identity transitions, then
remaining exposed events. The archived Dommo B3 circular specifies that the lender
chooses PNB; the borrower cannot pick the cheapest consideration afterward. Preserve
every eligible security and signed obligation. Unknown terms require a documented
bound or explicit retained claim, never guessed successor marks, double cash payment,
or a permanent blanket exclusion. Neither these sources nor the mechanics tests
complete Stages A/B or authorize interpreting new model profitability yet.

## Verified ticker renames and the data admission defect

The original continuation loader required a same-ticker heuristic proposal. That
made it impossible to admit an explicitly sourced rename that also changes ISIN.
The loader now validates explicit source-backed pairs directly against original
dated observations: predecessor last date, successor first date on the next
session, successor ticker, non-overlap and one-to-one identity. Heuristic candidates
remain diagnostics and never automatically authorize links.

The now populated allowlist binds ALSO3/BRALSOACNOR5 to ALOS3/BRALOSACNOR5 on
2023-10-25, and TRPL4/BRTRPLACNPR1 to ISAE4/BRISAEACNPR9 plus
TRPL3/BRTRPLACNOR4 to ISAE3/BRISAEACNOR2 on 2024-11-18. These are the same issued
share classes under new codes, with q=1 and no cash. The October 17 ALLOS notice,
newly archived November 7 ISA notice (CVM protocol 1299634), November 18 issuer
confirmation and dated COTAHIST rows establish the mapping. The November 7 notice
contains the typo `TRLP4`; the other issuer notice and original quotes establish
`TRPL4`. Raw documents retain the typo. No price-ratio heuristic supplies the terms.
Availability is conservatively the next local day after each advance notice,
comfortably before the effective sessions, without inventing an exact release time.

Running the existing causal history routing and unchanged universe rules on the
six relevant axes recovers 60 ALOS3 days and 28 ISAE4 days; ISAE3 retains its history
but gains no eligible days because it still fails the original liquidity tests.
Twelve stale predecessor-active cells retire; no eligible successor cell is lost
and no successor becomes eligible before its dated identity boundary. This is a
bounded audit, not an accepted full derived store. Actual tensors, wealth, labels
and every auxiliary join still need propagation and verification in Stage B.
`v2_identity_source_admission.json` binds the exact evidence and allowlist.

Stock-loan aliases require separate dated handling: the BDI can continue publishing
predecessor loan codes after spot trading has renamed. The equity continuation does
not silently relabel lending sources or assert a loan conversion/renewal date.


## Copel units: sourced shares and bounded loan allocation

The December 18 issuer notice binds CPLE11 / BRCPLECDAM13 (axis 262) to one
CPLE3 / BRCPLEACNOR8 (259) and four CPLE6 / BRCPLEACNPB9 (261). December 22
is the last unit trading day; December 26, 2023 is the economic split and
December 28 is custody credit. The accounting amendment preserves those dates,
original market observations and all model coordinates. No new fractional shares
arise from an integer unit. The normal research order quantities remain continuous.
Coverage gains 254 cells, including six eligible unit-days, on the existing axes.

The B3 manual specifies issuer factors for allocating original loan principal
across multiple assets. We have not recovered Copel's issuer K. The account now
requires explicit fractions summing to one and no longer substitutes constituent
market values for that contractual allocation. The reference is an openly labelled
20% ON / 80% PN research assumption, with the entire 0%-100% ON allocation range
as a one-factor sensitivity. Original total principal, rate and accrued charges
are preserved. Inventory cost basis/proceeds retain their separate causal
market-value allocation. Conversion at the night preceding December 28 custody
is an inference from the general manual and the issuer credit date; a one-session
timing range remains to be assessed along with prearranged presettlement disposal.
This is a bounded accounting amendment, not a claim to have recovered exact K.

`ops/verify_copel_replay.py` checks 36 actual-calendar cases: R$1m/R$5m/R$10m,
long/short 4% predetermined positions, same-day versus one-session-later ON disposal,
and 0%/20%/100% ON principal. Both accounts and independent original-reference
cash-flow formulas agree. At R$10m the original short principal is
R$395,131.69237921346, using the causal published-average R$49.51. With both
returns on January 3, rent at 4% annual is R$307.606711497966 for every K.
Delaying the ON return to January 4 gives a full allocation-range rent span of
R$61.5500726458255, or 0.0615501 bp of initial NAV over that prescribed path;
the reference rent is R$319.91672602713106. This is not a daily model improvement
or a bound for an adaptive policy. Model replays still need those sensitivities.
The arithmetic oracle uses zero execution/B3 fees and CDI to isolate the new
contract; the registered corporate account cost assumptions are unchanged.

## Additional Dommo receipts and remaining distinct obligations

`v2_dommo_source_terms.json` binds two newly recovered original CVM notices:
March 17 auction scheduling (protocol 1074297) and March 30 result (1080446).
They confirm that the PRIO shareholder fractions remain until the March 29 auction;
its amount becomes available for our decisions March 31. The printed result is
approximately R$31.94031 per share (R$565,024 / 17,690). The March 30 notice gives
five business days from that notice, an April 6 deadline, superseding the earlier
tentative April 5 deadline. Do not backdate the auction value to January or mistake
option/warrant fractions for ordinary PRIO share fractions.

The default PNA case is .0375 PRIO plus R$.4625 per DMMO: January 9 economic
conversion, January 11 custody, January 17 cash. A qualifying lender's PNB election
was due December 26 at 16:00 and applied to the whole loan. That loan's quantity
was extinguished December 26, with rent through that day paid December 28, and a
separate redemption liability paid January 13. The January 6 final R$1.90432468607
must not be backdated into December valuation. This path needs causal accrued CDI
and separate liability/payment handling. The borrower cannot choose it afterward.
The B3 circular also provisions PNA loan fractions; BRML's special retention of
fractional loan shares cannot automatically be applied to Dommo. These recovered
terms are now admitted through the separate PNB endpoint and default PNA manifest
described below.

The CVM 2023 delivery index currently names this CNPJ PRIO FORTE. Source retrieval
therefore used CNPJ 08.926.302/0001-05 / CVM 23493 and the dated issuer document,
not an assumed historical company name. This is a useful identity-audit boundary;
the index's current name must not become a point-in-time model attribute.


### Dommo lender-elected PNB endpoint admitted (2026-09-19)

The run pointer now binds `dommo_pnb_scenario`, separate from the primary
Cielo/BRML/Copel/Dommo-default manifest. This freezes all legally eligible pre-existing lender
contracts electing PNB on Dec26; it does not assert actual participation or let the
borrower select a favorable future outcome. The event runs after actual Dec26
fills, excludes whole roots with pending returns and same-day D+1 registrations,
and permits later non-elected DMMO borrowing. The current endpoint also includes
default PNA for contracts that remain after the election.

The Oct24 approval notice and 58 original SGS12 observations are archived. R$1.85
compounded Oct24 inclusive/Jan13 exclusive equals R$1.9043246860694236, matching
the Jan6 issuer figure R$1.90432468607. Dec26 closing liability uses only accrued
CDI (R$1.891796083009226); later daily marks use only then-elapsed observations,
including money-only Dec30. The fixed issuer amount first enters Jan9. Quantity
extinguishes Dec26, original rent stops there and pays Dec28; redemption pays Jan13.
Proceeds remain remunerated/restricted until that payment. Its exact date is used
as a future realization, never to discount earlier marks or release cash early.
Known principal payment and proceeds release occur together before that day's
policy decision; interest continues to use prior-close settled balances.

Six prescribed R$1m/R$5m/R$10m cases, with zero funding and with sourced 100% CDI,
reconcile exactly across both accounts and independent cash/rent formulas. The
R$10m 4% Dec23 short has original principal R$395721.92412462615, rent R$61.59406319349257
and redemption R$407342.1777998081. These are accounting amounts, not model alpha.
Actual lender election, finite maturity, partial-return exception handling and
custodian restrictions remain assumptions to bound in actual books.

### Default Dommo PNA admitted with provisioned loan fractions

The primary `corporate_replay` now binds `dommo_manifest.json`: .0375 PRIO and
R$.4625 per DMMO, January9 economic succession, January11 custody and January17
cash. Payment terms from the January6 notice first enter January9. No learned
feature, quote, label or eligibility array changes. This adds 494 resolved coverage
cells, six eligible, for a total 1,328/24 with Cielo, BRML and Copel.

The B3 manual (pp135–137) truncates the resulting quantity contract by contract,
preserves original financial principal and provisions fractions for issuer auction
cash. Unlike BRML's special rule, Dommo fractions cannot be covered by trading a
fractional PRIO loan share. Both books retain a signed, non-tradable fraction
marked from contemporaneously available PRIO prices. The March30 result first
enters March31; R$31.94031 is paid April6, with original precision and earlier
custodian-sweep sensitivities retained. Remaining restricted proceeds earn CDI
until that payment, including after the whole successor shares have been covered.

Rounding is per original loan, not per net position. Last DMMO purchases and covers
on January6 settle January10 under the registered T+2 convention, before January11
credit. Pending source purchases/returns or already split loan roots stop this
specific path explicitly; they need their own gross-custody allocation rather
than an invented net rounding. No security is dropped. A tiny original loan that
converts into less than one whole PRIO share has a documented research convention:
stop rent at the preceding conversion close and pay accrued rent on January11.
The alternative retains original-principal rent through April6; neither is claimed
as a recovered invoice. Ordinary research quantities remain continuous.

Six full-933-name actual-calendar 4% long/short oracles at R$1m/R$5m/R$10m agree
exactly across the two accounts and independent original-reference cash formulas.
At R$10m the original reference is R$1.78 and principal R$404,545.4567375262;
8,522 whole shares deliver, .7273189085572085 share remains provisioned, and 4%
annual rent through the January13 whole-share return is R$314.93524816795866.
The oracle has zero execution/B3/CDI to isolate arithmetic. It does not revise
registered economic assumptions or establish model alpha. Presettlement disposal,
actual lender election, fractional payment timing and clearing/lifecycle bounds
still belong in the corrected model replays.
