# Copel original-principal and loan conversion bounds

2026-09-20. V37 is bounded engineering acceptance, with final Stage A and C/D still open. Resolve `copel_loan_bounds_audit`, `copel_loan_bounds_qualification`, `copel_loan_bounds_acceptance` and `copel_loan_bounds_recovery` from the economic run pointer. Canonical `corporate_replay`, both accepted stores and all old fits remain unchanged.

## Contract and implementation

Copel units BRCPLECDAM13 split economically on December26 2023 into one BRCPLEACNOR8 and four BRCPLEACNPB9. The recovered issuer receipt fixes positive shareholder custody on December28. The general B3 manual requires issuer factors for original loan-principal allocation; the event-specific K and loan-conversion instruction remain unrecovered. The earlier source receipts and prescribed36 cases are retained; none was retrieved or replayed again.

Use K=.2 ON/.8 PN primary and K0/K1 as separate endpoint sensitivities. The existing loan subledger already preserves total original principal/rate and allocates accrued rent/fees. K is unrelated to the1:4 quantities or the market-value allocation of inventory basis and proceeds. Tested endpoints do not prove an extremum over every interior K for an adaptive policy.

Both actual accounts now admit an optional `loan_conversion_session`/`loan_conversion_date`. A negative source claim uses this date to become successor loan obligations; positive shareholder inventory uses the original physical credit. The one-factor timing variants use December27/28/29, with December28 primary. A conversion retains original rate/reference principal, open date, fees and pending-return dates. It does not open a new loan or infer a locate, reference price, loan alias or shareholder delivery. Existing positive successor inventory can offset the converted debt only with its actual settled/dated owned receipt; physical return and restricted-proceeds release follow that receipt. A cover after conversion retains normal spot T+2.

This is explicitly a **net borrowed claim** timing hypothesis. Flat or positive source inventory with pending old loan returns keeps the prior convention; no new claim is made about that separate gross-offset case. Separate timing with a fraction auction is rejected because that requires additional event-specific terms. The current qualified Copel basket has no auction leg. Both economic accounts, claim-date readouts and sector exposure scheduling use the same sign-dependent clock. The leg's shareholder credit remains separately stored. Loader/slicing preserve both dates, and existing independent SAM/TBPTT copies preserve the immutable event and mutable account state.

## Frozen adaptive books

Thirty new30-session/all933 books cover December15 2023–January30 2024, six pre-effect plus24 effect/following sessions. Five separate variants (primary, K0, K1, early loan, late loan), both source sides and R10m/R1m/R5m use the actual constrained allocator. Synthetic source preferences are +/-4 before effect and opposite afterward; ON reverses at effect while PN keeps the original source sign until credit+2, then reverses. All other sinusoidal preferences, beta1/idio.0004/market.0001 risks and original calibration are frozen. This creates staggered disposal pressure without model scores or outcome-selected parameters.

Accounting inputs shallow-copy frozen OLD PolicyData, using corrected CDI, qualified lending and exact prior references. The old bundled4bp cost bridge has no additional B3 spot component; it is not final corporate pricing. Original Copel loan roots here use the recovered .0002 annual rate (0.02%), unlike the earlier prescribed4% rent example. No old book or source/store audit was repeated.

## Independent saved results

All30 identical-intention accounts agree within R$1.1175870895385742e-08. Independent saved cash, restricted proceeds, unsettled money, claims, marked inventory/hedge and loan liability reconcile900 daily NAVs within R$3.725290298461914e-09; 3,380,400 account cells are retained. Decimal signed entitlements plus actual fills reproduce both legs with maximum 1.13e-11 share discrepancy. Source post-effect fills are absent. All long-focus arrays are exact across variants; NAV prefixes before first differing realization and all effect-day targets are exact. This branch equality is distinct from parity of independently adaptive implementations.

For each converted original root, saved before/prepared states independently reconcile quantities1:4, principal K:(1-K), accrued rent/fees, original rate/opening/reference principal and unchanged fee-growth/minimum state. Maximum Decimal allocation discrepancy is R$5.7e-11. Independent Decimal daily compounded rent and both B3 fee components match660 source/day checks (including zero controls; 102 nonzero rent checks), maximum errors R$5.289731300870007e-16 and R$5.359740182121749e-16. These are repeated scenario checks, not distinct source observations. Post-2020 minimums here are zero; the older R$10 minimum bound remains separate.

Independent adaptive maximum total-path differences at R10m/R1m/R5m are 6.877599284052849e-06/2.4427147582173346e-06/6.877599284052849e-06bp, target distance 2.6492040085951407e-08. Preserve larger V32–V36 measured fixed-fee uncertainty for future actual-model comparisons. These are neither daily alpha nor bitwise adaptive equality.

The table gives total synthetic-path NAV contrasts in bp of initial capital, with original Copel-root rent shown separately. Timing changes adaptive disposal/exposure, so its NAV difference must not be described as just one day's rent. Every long-focus contrast is zero. K's small effect at the recovered low rate does not establish small effects for different rates, timings or actual model exposures.

| Short-focus variant minus primary | Capital | Final path bp | Maximum absolute path bp | Copel original-root rent difference R$ |
| --- | ---: | ---: | ---: | ---: |
| k0 | 10,000,000 | -0.000435061 | 0.000435061 | 0.193551947 |
| k1 | 10,000,000 | 0.001740257 | 0.001740257 | -0.774208197 |
| loan_early | 10,000,000 | -3.999501232 | 7.449071841 | -0.156607439 |
| loan_late | 10,000,000 | -3.949972862 | 4.839939996 | 0.152763622 |
| k0 | 1,000,000 | -0.000435076 | 0.000435076 | 0.019355195 |
| k1 | 1,000,000 | 0.001740251 | 0.001740251 | -0.077420820 |
| loan_early | 1,000,000 | -3.999502372 | 7.449071662 | -0.015660744 |
| loan_late | 1,000,000 | -3.949973702 | 4.839940277 | 0.015276362 |
| k0 | 5,000,000 | -0.000435061 | 0.000435061 | 0.096775974 |
| k1 | 5,000,000 | 0.001740257 | 0.001740257 | -0.387104098 |
| loan_early | 5,000,000 | -3.999501232 | 7.449071841 | -0.078303719 |
| loan_late | 5,000,000 | -3.949972862 | 4.839939996 | 0.076381811 |

Eleven distinct new tests and seven affected existing precredit tests passed in focused batches; Ruff passes. The added pending-owned fixture initially bought fewer units by applying40% to NAV already reduced by rent. Using the actual original units qualified its Decimal NAV/gradient oracle; production code and all historical books were unchanged. Initial/failed test bytes are retained. A read-only Polars pretty-print failed under Windows cp1252 after successful sample/schema reads and changed no data. All30 books and saved qualification completed on their first invocation. Engineering 27.535423s and saved qualification 0.495864s are CPU audit timings, not fit estimates. Executed research runtime hashes match current implementation.

## Remaining admission

Apply these explicit variants later to matched actual-model books. Cash/fraction payment sweeps, precision/invoice/security-day grouping/older minimum, dated B3 spot/custody and other corporate cost components, three older equity-clearing ambiguities and remaining held-event source/data admission are still open. ALSC physical credit/exact net fees/payment and ENAT auction remain unknown; their completed source/engineering records stay sealed. No corrected model profitability, new neural forward, GPU fit or final economic acceptance is claimed.


Verified recovery: implementatione423e462a185d240a026c38bfdecb07b46f30c63; archive copel_loan_bounds_e423e46.zip, SHA24739f9b1089618a017104cbe41f26a8badc489555cc2cc61977bc9a3416fd2a, 5598321bytes. 724unique/1024logical restored/hashchecked members, 300aliases;30completebooks/3380400accountcells. Recovery1.142679s. Current research/patch/frozen plan/five term variants/executed recipes/failed-qualified tests/newbooks recover without immutable inputs/stores/oldfit duplication. Prior precredit/ENAT/Natura/lifecycle/economic-store dependencies remain required. AcceptanceSHA049f9b06a502b38de1e9a1ebe68d9c16f6db827fc27ce9c090d3964fdc8fb263; final A/C/D remain open.
