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

Complete the DMMO lender-election/default distinction, CPLE multi-asset issuer
principal allocation and delayed credit, ALLOS/ISA dated identity transitions, then
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
