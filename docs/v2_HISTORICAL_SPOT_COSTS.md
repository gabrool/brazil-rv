# Dated ordinary-CNPJ spot costs and cash sensitivities

2026-09-20. V40 is a bounded Stage A engineering acceptance, not final corporate pricing. Resolve historical_spot_audit, historical_spot_qualification, historical_tariff_sources, historical_spot_runtime_identity, cielo_saved_exposure and historical_spot_acceptance from the run pointer. Both accepted stores, original fits and all V31-V39 books remain immutable.

## Source and implementation contract

Three new original B3 circulars and15 selected visual pages are bound. [177/2020](https://www.b3.com.br/data/files/68/50/2D/3E/99E4671059300467AC094EA8/OC%20177-2020%20PRE%20_Modelo_Intermediario_%28PT%29.pdf), dated December10 2020, activates the intermediate model on February2 2021: ordinary spot trading0.5bp plus clearing2.5bp; auction trading0.7bp. [017/2023](https://www.b3.com.br/data/files/BB/35/6C/6C/E810B810E9C1AAA8AC094EA8/OC%20017-2023-VPC%20Consolida%C3%A7%C3%A3o%20de%20Regras%20da%20Pol%C3%ADtica%20Tarifa%C3%A7%C3%A3o%20Produtos%20Mercado%20Renda%20Vari%C3%A1vel%20PT.pdf) expressly consolidates existing rules without changing them. Ordinary CNPJ is not a local investment fund. The later0.5/2.24/0.26 split and client-ADTV discounts are prospective, not these historical components.

Both actual accounts now expose execution_charges in BRL, columns bundled/B3trading/B3clearing/brokerage/shortfall. The explicit b3_spot_2021 mode requires BOTH old bundled costs zero, rejects dates outside February2 2021-December30 2024, and replaces the bundle. Regular phase is the primary hypothesis; auction is a separate one-factor execution hypothesis, since a close-price proxy does not establish actual auction fills. Zero brokerage and1bp provisional shortfall reproduce4bp exactly. The actual allocator retains its frozen4bp cost estimate; changed realized NAV influences later adaptive decisions. No new estimator, preference calibration, source rate, locate or price is fabricated.

The implemented contract uses unrounded charges and existing fractional research units. Same-security/day opposing fills are explicitly rejected pending daytrade matching/rate admission, rather than silently applying regular fees. All9888 saved actual fills in these books satisfy the one-direction condition. Corporate loan conversions and cash settlements are not spot trades. The source also establishes spot invoice grouping/6decimal fee rounding/final2decimal truncation; those physical-invoice mechanics remain separate work, distinct from completed V39 loan invoices.

177/2020 sources monthly custody calculation on the last-business-day portfolio, annual progressive tiers prorated by month, active resident maintenance exemption and a belowR20000 value exemption. [014/2022](https://www.b3.com.br/data/files/21/E4/03/B3/AAF55810F534EB48AC094EA8/OC%20014-2022-VPC%20Depositaria_Compliance%20e%20Jur%C3%ADdico_REVISADO%201.pdf), effective January1 2023, keeps the printed percentage tiers and changes the threshold to belowR23084.39. Custody physical base/payment/later dated fixed-value updates are not yet implemented. The January2020 proposal deferred launch; its daily custody and cash-event processing proposals cannot be treated as activated fees. Earlier market-wide ADTV progressive trading tiers require dated monthly rates; the old TMRV download returned403. No account-capital substitution or flat3.25bp blanket is admitted. These are retrieved copies with later revocation annotations, not proof of historical revision completeness or publication minute.

## Frozen adaptive books and checks

Thirty new32-session/all933 books cover June3-July16 2024, R10m/R1m/R5m, bundledcontrol/primary/auction/shortfall0or2/brokerage0.5or1/proceeds95%/debit+50or100annualbp. Frozen sin(axis*.31+localday*.07+head*.2), fixedbeta1/idio.0004/market.0001 and original calibration use the actual constrained allocator and arbitrary-policy book_summary. OLD PolicyData is shallow-copied for corrected CDI, qualified lending and strict-prior references. Both complete accepted stores and old fits are unchanged; no neural scoring or fit.

All30 first invocations and independent qualification passed. Saved960NAVs/3613440accountcells reconcile withinR3.725290298461914e-9. Independent Decimal charges from9888 actual fills differ by at mostR9.094947017729282e-13; both-account component differenceR1.8189894035458565e-12, cost sumR4.547473508864641e-12. Prior-close funding maxR8.381903171539307e-9 and income formulasR3.183231456205249e-12. The primary and bundled control have exact every-array equality except the intentionally separated execution columns. NAV prefixes and current intentions before first different realization are exact; 95% proceeds first changes day3 after settlement.

Identical-intention account maxNAVR1.862645149230957e-8. Independent adaptive max total-path differences .000003898082301020623/.000004510261351242661/.000003898082301020623bp atR10m/R1m/R5m; target1.0332988412068564e-8. Preserve larger V32-V39 fixed-fee uncertainty. These are synthetic total-path contrasts, not daily alpha, model profits or proven interior/monotone extrema.

R10m primary direct tradingR1441.4738585434425, clearingR7207.369292717213, shortfallR2882.947717086885, brokerage0. Free incomeR57378.03669867306 and settled proceeds incomeR75465.01321975412; loans remain separately reported inclusive taker proxies with B3 loan charges, no extra invented intermediation markup. No debit occurs in these adaptive books, and their debit variants are exact unexposed controls. A separate deterministic both-account funded-long fixture verifies settled-debit financing on actual funded debit at0/+50/+100annualbp; it is not an adaptive exposed-capacity bound.

| Scenario minus primary | Capital | Final bp | Max path bp |
| --- | ---: | ---: | ---: |
| auction | 10,000,000 | -0.574886580 | 0.574886580 |
| broker_half | 10,000,000 | -1.437197232 | 1.437197232 |
| broker_one | 10,000,000 | -2.874346747 | 2.874346747 |
| debit_100 | 10,000,000 | 0.000000000 | 0.000000000 |
| debit_50 | 10,000,000 | 0.000000000 | 0.000000000 |
| proceeds_95 | 10,000,000 | -3.745547478 | 3.745547478 |
| shortfall_two | 10,000,000 | -2.874346747 | 2.874346747 |
| shortfall_zero | 10,000,000 | 2.872895740 | 2.872895740 |
| auction | 1,000,000 | -0.574881317 | 0.574881317 |
| broker_half | 1,000,000 | -1.437198492 | 1.437198492 |
| broker_one | 1,000,000 | -2.874346419 | 2.874346419 |
| debit_100 | 1,000,000 | 0.000000000 | 0.000000000 |
| debit_50 | 1,000,000 | 0.000000000 | 0.000000000 |
| proceeds_95 | 1,000,000 | -3.745545473 | 3.745545473 |
| shortfall_two | 1,000,000 | -2.874346419 | 2.874346419 |
| shortfall_zero | 1,000,000 | 2.872895765 | 2.872895765 |
| auction | 5,000,000 | -0.574886580 | 0.574886580 |
| broker_half | 5,000,000 | -1.437197232 | 1.437197232 |
| broker_one | 5,000,000 | -2.874346747 | 2.874346747 |
| debit_100 | 5,000,000 | 0.000000000 | 0.000000000 |
| debit_50 | 5,000,000 | 0.000000000 | 0.000000000 |
| proceeds_95 | 5,000,000 | -3.745547478 | 3.745547478 |
| shortfall_two | 5,000,000 | -2.874346747 | 2.874346747 |
| shortfall_zero | 5,000,000 | 2.872895740 | 2.872895740 |

Seven distinct new tests and32 affected existing account/ledger tests pass; first6 overlap final7. Gradient/hedge/settlement/copy probes pass. All executed production bytes match current. Engineering30.9497025s/sumcases29.7562155s/independent qualification.5765957s are CPU audit runtimes, not fit ETAs. Source retrieval times exclude search/render; the failed generation and read-only inspection attempts are described in attempts.md, not falsely presented as exact archived shell bytes.

## Cielo and remaining admission

Twelve saved V32 2024 books have last negative Cielo stock positions August6 and zero shares August29/30. Only the three base books saved source-specific loan_charges; their charge tails are inspected in the bound receipt. The other nine variants lack those files, so flat inventory is not proof of no pending old loans. No new held-redemption case is established and the actual-held-Cielo cent bound stays open. Earlier18 unexposed Cielo controls were not repeated. This is not an exhaustive old-model exposure search.

Final A remains open: custody/earlier tariffs, spot invoices/daytrade if exposed, clearing ambiguities, Cielo held-loan precision and remaining source/data admission. Both stores remain sealed; C/D unstarted. No final profitability conclusion.
