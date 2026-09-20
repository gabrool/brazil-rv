# Loan invoice cents and historical minimum allocation

2026-09-20. V39 qualifies explicitly bounded accounting engineering; final A/C/D remain open. Resolve loan_invoice_audit, loan_invoice_qualification, loan_invoice_acceptance and loan_invoice_recovery from the economic run pointer. Both accepted Stage B stores, original fits, canonical corporate replay and all prior event books remain immutable.

## Contract

The primary remains unrounded loan payments and residual R10 historical minimum on final return. New one-factor hypotheses round rent and total B3 loan fees separately on their actual physical payment date, per original contract: nearest half-up, down or up to cents. A separate nearest-original-security/day scenario changes invoice grouping only, compared with contract-nearest. It never combines loan registrations, changes rates/principal, reduces the number of historical minimum obligations, groups across payment dates or merges rent with B3 fees. These are analyst invoice hypotheses, not sourced broker/B3 instructions. Original-security grouping remains explicit through corporate successor conversion.

Both accounts recognize the paid-minus-unrounded difference as expense at payment, separately exposed in loan_invoice_adjustment (rent/B3 columns). No prior intention or prior-close interest changes. Old loan_charges retain accrued rent/fees; invoice adjustments must be added separately when reconciling them to paid expense. Exact floor/ceil arithmetic retains its actual local derivative, zero away from cent boundaries; no straight-through gradient approximation. The metadata states each selected hypothesis.

Another one-factor hypothesis allocates the currently unpaid minimum residual pro rata to original principal actually returned. Early amounts become minimum_credit, offsetting subsequent B3 invoices. The credit reduces remaining loan liabilities and survives partials, corporate leg allocation and independent SAM/TBPTT copies; final return or renewal settles the old root exactly once. No refund or second minimum is invented. Same-intention closed-form tests preserve total max(accrued B3 fees,R10), including a path crossing the minimum threshold. Adaptive scenario paths can trade differently and therefore create different later contracts; they are not a pure interest-only arithmetic difference.

## Frozen books and checks

36 new64-session books start February1 2019 (through May7) and February1 2024 (through May6), six variants and R10m/R1m/R5m. All933 names and original histories remain. The actual allocator receives sin(axis*.31+localday*.07+head*.2), fixed beta1/idio.0004/market.0001 and original calibration. OLD PolicyData is shallow-copied for corrected CDI/qualified lending/strict-prior references. No neural forward or model score is used. The old bundled4bp bridge has no additional B3 spot and is not final corporate pricing. All36 completed on the first invocation; none was repeated.

Independent saved cash/proceeds/unsettled/claims/stock/hedge/loan arithmetic checks 2304 NAVs, max R5.587935447692871e-09; 8660736 account cells. Prior funding max R7.916241884231567e-09; liability/expense rollforward max R2.9103830456733704e-11. 9738 root-payment records produce 9503 invoice groups; independent Decimal minimum allocation max R2.32856523044e-15, grouping max R1.1e-14, actual-coordinate invoice error zero. Before the first differing payment/credit, all NAV prefixes and current intentions are exact.

The2019 pro-rata variants have81/235/97 actual partial minimum allocations and maximum fee credits R34.0620348024/R63.0762494277/R38.9071349453 atR10m/R1m/R5m. The2024 books have no old minimum and their pro-rata/primary paths are exact. Security/day grouping actually merges roots in every tested capital/window. Final-boundary unsettled loans/charges remain accounted for.

Independent components match within1.1e-14BRL; Decimal rounding of saved Float64 invoice coordinates matches all9503 groups exactly.72 recomposition/cent straddles are operand-printing precision boundaries around an actualR10 coordinate, not observed invoice uncertainty or evidence for9.99/10.01 exact fees. All are enumerated; actual lower/upper scenario paths retain their frozen coordinate contract.

Identical-intention independent accounts agree within R1.6763806343078613e-08. Independentadaptive maxpath .010282502409070731/.00000816549058072269/.000022326227277517318bp atR10m/R1m/R5m. Preserve larger V32-V38 fixed-fee uncertainty. Many2019R10m path signs are below this bound. Total synthetic paths, not dailyalpha/modelprofit or an all-interior adaptive extremum proof. Maximum target distance 2.006825300226306e-07. AtR10m in2019, nearest rent/B3 rounding totals R.0182052/.0751801 while the path contrast is+.00995962bp; this includes adaptive trading/minimum effects and is below measured implementation uncertainty. Do not call it a beneficial rounding effect. In2024, down/up total final contrasts are+.002207685/-.002196860bp. These are bounded synthetic paths, not corrected model economics.

| Start / variant minus base | Capital | Final bp | Max path bp |
| --- | ---: | ---: | ---: |
| 2019-02-01 / contract_nearest minus primary | 10,000,000 | 0.009959622 | 0.010241852 |
| 2019-02-01 / contract_nearest minus primary | 1,000,000 | -0.000823796 | 0.000870670 |
| 2019-02-01 / contract_nearest minus primary | 5,000,000 | 0.000094581 | 0.000109655 |
| 2019-02-01 / contract_down minus primary | 10,000,000 | 0.003033416 | 0.003033416 |
| 2019-02-01 / contract_down minus primary | 1,000,000 | 0.030105232 | 0.030105232 |
| 2019-02-01 / contract_down minus primary | 5,000,000 | 0.005878199 | 0.005878199 |
| 2019-02-01 / contract_up minus primary | 10,000,000 | 0.006939977 | 0.009199515 |
| 2019-02-01 / contract_up minus primary | 1,000,000 | -0.031435177 | 0.031435177 |
| 2019-02-01 / contract_up minus primary | 5,000,000 | -0.006302811 | 0.006302811 |
| 2019-02-01 / security_day_nearest minus primary | 10,000,000 | 0.009989571 | 0.010241852 |
| 2019-02-01 / security_day_nearest minus contract_nearest | 10,000,000 | 0.000029949 | 0.000029949 |
| 2019-02-01 / security_day_nearest minus primary | 1,000,000 | -0.000424156 | 0.000794125 |
| 2019-02-01 / security_day_nearest minus contract_nearest | 1,000,000 | 0.000399640 | 0.000399640 |
| 2019-02-01 / security_day_nearest minus primary | 5,000,000 | 0.000154203 | 0.000161964 |
| 2019-02-01 / security_day_nearest minus contract_nearest | 5,000,000 | 0.000059623 | 0.000059946 |
| 2019-02-01 / minimum_pro_rata minus primary | 10,000,000 | 0.009829037 | 0.010143452 |
| 2019-02-01 / minimum_pro_rata minus primary | 1,000,000 | -0.005822562 | 0.005822562 |
| 2019-02-01 / minimum_pro_rata minus primary | 5,000,000 | -0.000698573 | 0.000698573 |
| 2024-02-01 / contract_nearest minus primary | 10,000,000 | -0.000012849 | 0.000040419 |
| 2024-02-01 / contract_nearest minus primary | 1,000,000 | 0.000563759 | 0.000565151 |
| 2024-02-01 / contract_nearest minus primary | 5,000,000 | 0.000226755 | 0.000226755 |
| 2024-02-01 / contract_down minus primary | 10,000,000 | 0.002207685 | 0.002207685 |
| 2024-02-01 / contract_down minus primary | 1,000,000 | 0.020139085 | 0.020139085 |
| 2024-02-01 / contract_down minus primary | 5,000,000 | 0.004188492 | 0.004188492 |
| 2024-02-01 / contract_up minus primary | 10,000,000 | -0.002196860 | 0.002196860 |
| 2024-02-01 / contract_up minus primary | 1,000,000 | -0.023882651 | 0.023882651 |
| 2024-02-01 / contract_up minus primary | 5,000,000 | -0.004608960 | 0.004608960 |
| 2024-02-01 / security_day_nearest minus primary | 10,000,000 | -0.000015899 | 0.000038066 |
| 2024-02-01 / security_day_nearest minus contract_nearest | 10,000,000 | -0.000003050 | 0.000016776 |
| 2024-02-01 / security_day_nearest minus primary | 1,000,000 | -0.000350268 | 0.000350268 |
| 2024-02-01 / security_day_nearest minus contract_nearest | 1,000,000 | -0.000914027 | 0.000914027 |
| 2024-02-01 / security_day_nearest minus primary | 5,000,000 | 0.000127914 | 0.000128301 |
| 2024-02-01 / security_day_nearest minus contract_nearest | 5,000,000 | -0.000098841 | 0.000098841 |
| 2024-02-01 / minimum_pro_rata minus primary | 10,000,000 | 0.000000000 | 0.000000000 |
| 2024-02-01 / minimum_pro_rata minus primary | 1,000,000 | 0.000000000 | 0.000000000 |
| 2024-02-01 / minimum_pro_rata minus primary | 5,000,000 | 0.000000000 | 0.000000000 |

## Execution and limitations

8passed/3failed: cash roundoff1.82e-12 atR10000 against an absolute1e-12 helper; T+3 credit probe incorrectly day4, correctly day5. Narrow fixtures corrected: comparison atR1000, probe day5 and mutate one credit cell. Production unchanged. Initial test and implementation saved. Qualified11 passed, added threshold-crossing case gives12 plus9loan/6payment existing=27pass; two additional delivery/renewal credit cases produce14distinct new tests. Overlapping batches not additive.

Initial Decimal recomposition of independently printed accrued fees and minimum residual straddled R10 by ulps, causing floor/ceil cent mismatches. Verify component arithmetic then round actual Float64 invoice coordinate in independent Decimal.72 enumerated boundary cases retained; all coordinate-based invoice amounts exact. No production change or book rerun.

All36 books completed first invocation, no numerical failure or repeated book.

Post-book evaluation report now states chosen invoice/minimum hypothesis instead of always describing final-return allocation. Exactly two metadata text changes verified; all other executed production bytes match current.

14distinctnewtests,9existingloancontracttests,6existingfractionpaymenttests pass; overlapping batches not summed. Ruff passes. Engineering 74.986028s (sumcases 73.683735s); saved qualification 1.087650s. These are CPU audit runtimes, not fit ETAs.

Actual-held-Cielo cent bound; final historical B3 spot/custody/rent-intermediation/brokerage/shortfall/cash/debit separation and negotiated sensitivities; three older clearing ambiguities; held source/succession data admission; then registeredC/D. ALSC credit/exactnet/actualreceipt and ENATfractionauction remainunknown. Both acceptedstores/oldfits and all prior books are sealed; no newfit/modelprofit.
