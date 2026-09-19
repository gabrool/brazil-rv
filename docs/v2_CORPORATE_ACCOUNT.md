# Corporate low-touch account: economic contract and evidence

As of 2026-09-19. This amends the account assumptions in the
[economic/data/scaling registration](../research/preregistrations/v2_economic_data_scaling.md).
The user now specifies a well-capitalized CNPJ low-touch account with interest on
short-sale proceeds. Earlier sealed books remain unchanged. Current operating
documents are evidence about prospective terms, not new model inputs or permission
to consume held-out 2025/2026 market data.

## Account decision

Use a domestic company trading its own capital through electronic execution,
separately negotiated securities lending and cash management. No fund structure,
tax exemption or expensive retail advisory package is presumed.

The primary case earns **100% of historical CDI on eligible settled short-sale
proceeds**, with **zero execution brokerage**. These are authorized package targets
and research assumptions. Public sources establish relevant capabilities and cash
instruments; no broker has quoted this complete package to the project. Do not
present the best individual terms at different providers as one confirmed offer.

Planning capital is explicitly R$10m, with R$1m/R$5m cost and capacity sensitivities.
These are not claims about actual capital or broker admission minima. Capital changes
order size, liquidity participation, fixed-cost incidence and margin, not forecasts.
Preserve all model-eligible names and quantify execution capacity separately.

## Component contract for the corrected account

| Component | Primary treatment | Status / bounded sensitivity |
| --- | --- | --- |
| Free cash and settled short proceeds | 100% of historical daily CDI on eligible balances | Authorized negotiated-account hypothesis; 95% short-proceeds CDI sensitivity |
| Execution brokerage | Zero | Authorized target; +0.5 and +1 bp per traded side sensitivity, not broker quotes |
| B3 stock/ETF trading, clearing and transfer | Applicable dated exchange components | Actual investor, turnover and execution category; no ordinary-CNPJ fund discount |
| Loan rent | Causal security-specific historical rate with contractual reference/rate and duration | Published rates are proxies, not executable locates; remove unsupported universal rate floors |
| Loan intermediation | Observed component where identifiable; no invented additional markup on an inclusive taker rate | If decomposition is impossible, report a combined proxy explicitly rather than claim zero embedded commission |
| B3 loan charges | Dated modality-specific components | Separate from rent; annual rate is not a per-turnover toll |
| Custody and fixed charges | Separate B3 pass-through, broker custody, platform and operating costs | Establish waivers rather than assume all are zero; no invented management/performance fee |
| Execution shortfall | Preserve the old all-in cost bridge, then attribute spread/slippage/impact | Provisional 1 bp per side where the exchange component is 3 bps; not a measured fill-cost guarantee |
| Debit financing | CDI plus zero spread as an explicit favorable target, on actual funded debit only | No secured CNPJ quote obtained; +50/+100 annual bp spread sensitivities on debit, not NAV/gross exposure |

The existing independent and differentiable accounts already use
`short_proceeds_remuneration=1.0` and 4 bps/side bundled stock/hedge execution cost.
Therefore, changing the intended account **does not itself add 3 bps/day to the old
headline books**. The earlier no-interest sensitivity is archived, not a required
account or admission case. For a 3 bp exchange charge, 3 bp B3 + zero brokerage +
1 bp provisional shortfall reproduces the old 4 bp total. Adding B3 on top of 4 bp
would double-count. A lower shortfall requires evidence and separate attribution.

Remunerated restricted cash cannot simultaneously be spent on longs while still
earning a second income stream. Reconcile actual settled cash, credit, collateral
and short liabilities. Haircuts on collateral constrain funding capacity; they are
not automatically expenses. Do not invent margin-cover penalties for a properly
funded account. Preserve dated settlement conventions and do not backdate income
on unsettled proceeds. The ledger implementation remains Stage A work.

## Exchange evidence

The [current regular spot schedule](https://www.b3.com.br/en_us/products-and-services/fee-schedules/listed-equities-and-derivatives/equities/equities-and-investment-funds-fees/spot/)
lists 0.50 bp trading + 2.24 bp CCP + 0.26 bp transfer = **3.00 bps per side**
up to R$3m monthly ADTV. Above this, the progressive tier is 2.25 bps with a R$225
adjustment: at R$5m ADTV, the implied regular rate is 2.70 bps. The lower tier is
not immediately applicable to all turnover. Opening/closing auctions use a 0.70 bp
trading component; classify actual fills. These fees apply to traded notional.

The [B3 transition record](https://clientes.b3.com.br/w/nte-nova-tarifacao-de-equities)
dates launch to 2025-08-01 and distinguishes earlier ordinary-investor 3 bp from
local-fund 2.3 bp treatment. Historical attribution must use historical schedules;
newer discounts belong to a separately labelled prospective counterfactual. The
[large non-day-trader program](https://www.b3.com.br/pt_br/produtos-e-servicos/tarifas/listados-a-vista-e-derivativos/programas-de-incentivo/programa-grandes-nao-day-traders/)
requires R$150m ADTV and approval, not R$150m assets. A separately linked quantitative
incentive page returned an error; no unverified discount from it is included.

The [current loan tariff](https://www.b3.com.br/pt_br/produtos-e-servicos/tarifas/tarifas-de-emprestimo-de-ativos/)
distinguishes normal electronic, direct electronic, OTC and compulsory borrowing.
At 1% annual contract rent, ordinary fee totals are approximately 20, 20.5 and
30 annual bps for the first three modalities; at 2%, they are 40, 41 and 60 bps.
Apply the individual components' floors/caps. Do not add taxes already included in
the published tariff. Compulsory rates are not the normal prearranged-loan case.

Prefer electronic sourcing where executable, comparing rent plus all fees. Do not
attach the cheapest modality's fee to an OTC-only quote, pick ex-post donor minima,
or apply current caps to earlier years. Historical circulars and effective dates
are bound in `v2_economic_data_loan_sources.json`. The
[B3 contract description](https://www.b3.com.br/pt_br/produtos-e-servicos/emprestimo-de-ativos/informacoes.htm)
has separate compounded annual rent and intermediation fields. Establish the precise
bulletin semantics before decomposing its averages. A difference between differently
weighted donor/taker averages is not automatically our executable commission.

[B3 accepts government securities and other eligible collateral](https://www.b3.com.br/pt_br/produtos-e-servicos/compensacao-e-liquidacao/clearing/administracao-de-riscos/garantias/garantias-aceitas/).
This supports remunerated collateral instead of assuming all margin earns zero;
it does not itself quote a broker short-cash rebate. The
[current custody schedule](https://www.b3.com.br/en_us/products-and-services/fee-schedules/central-depositary-services/custody-services-fees/)
is progressive on eligible balances. Do not replace it with a large flat custody
charge on total gross long/short exposure. Establish broker absorption separately,
and distinguish own holdings, stock loans and government-security collateral.

## Providers and a coherent negotiated package

These are quote candidates, not a verified price ranking:

| Candidate | Public evidence | Unquoted for this project |
| --- | --- | --- |
| Ideal / Itaú group | [Ideal section 15](https://api-site.idealctvm.com.br/uploads/Regras_e_Parametros_de_Atuacao_v_9_0_fev_2026_v_final_MARCAS_3fffecc6d5.pdf) explicitly negotiates brokerage at engagement | CNPJ eligibility at this scale, cash/borrow terms, electronic platform fees and minimum revenue |
| BTG Pactual | [DMA section 14](https://static.btgpactual.com/media/regras-e-parametros-ctvm.pdf) describes electronic routing through a specific agreement | Complete institutional cash and securities-financing terms, rather than retail tariffs |
| XP Institutional | [Institutional service](https://bancodeatacado.xpi.com.br/corretora-institucional/) offers electronic execution, algorithms and BTC | Negotiated CNPJ pricing and remuneration |

The [XP public cost page](https://www.xpi.com.br/custos-operacionais/) excludes
CNPJ from certain retail discounts and lists **0.25% of loan financial volume on
liquidation/renewal**, in addition to stock rent and B3. It also lists an operational
surcharge of 5.9% on specified brokerage/exchange charges. These public standard
terms are not our primary corporate assumptions. A 25 bp turnover charge on a
R$100k loan is R$250 per triggering event; a 25 bp annual charge for five sessions
is roughly R$5. Negotiate annual accrual or a genuinely low explicit fee, partial
returns and renewals without that repeated notional toll. No generic retail penalty
or uncovered-margin service is imported into the intended funded account.

[Inter's corporate cash description](https://blog.inter.co/investimentos-para-empresas)
advertises a daily-liquidity CDB at 100% CDI. It supports the cash-yield target but
does not prove collateral acceptance or availability for short proceeds. The chosen
broker's rebate, a government-security investment or a repo are alternative ways
to earn cash income, not additional layers to add together. CDI is a benchmark.

[Santander's corporate repo terms](https://cms.santander.com.br/sites/WPS/documentos/arq-investimentos-pj-compromissadas-condicoes/21-07-21_204408_lamina_compromissada_port.pdf)
describe daily-liquidity debenture-backed repos and their IOF exemption. The 40%-CDI
worked example is illustrative, not a competitive quote to impose on this account.
The chosen instrument, yield, credit exposure, liquidity, collateral acceptance and
transactional taxes must agree. Excluding entity income tax from the research
metric does not exclude applicable transactional IOF. Avoid assuming that constantly
redeeming short-aged CDBs is equivalent to an efficient corporate cash arrangement.

Obtain one coherent package per candidate: electronic equity/ETF execution; lender
rent and broker loan commission; B3 pass-through by modality; partial returns,
renewals and recalls; 100%-CDI proceeds remuneration or equivalent cash-management
mechanism; remunerated collateral; debit line; and all fixed charges. Compare total
cost at actual turnover and short balances. No provider has been contacted.

## Sensitivity and reporting contract

Use one-factor replays around the favorable primary account: 95% proceeds CDI,
+0.5/+1 bp brokerage, and +50/+100 annual bp debit spread. These are analyst-chosen
robustness distances, not inferred quotes. Do not stack all adverse values into the
headline or select assumptions after seeing an architecture's outcomes. Assess an
unresolved loan commission once source decomposition establishes its proper basis.

Before compounding, +1 bp execution fee at 0.20 NAV daily traded notional costs
0.20 bps/day. A +50 annual bp debit spread on 0.10 NAV debit costs about 0.020
bps/day. Charging the same spread to all NAV exaggerates that expense tenfold.
Reducing proceeds remuneration from 100% to 95% reduces that income component by
5%, not five annual percentage points. Compute actual impacts from actual balances.

Report securities P&L; loan rent and identifiable intermediation; B3 spot/loan/custody
charges; brokerage; execution shortfall; free-cash income; proceeds income; debit
interest; fixed operating costs; and net above CDI. Label entity income taxes
excluded. Keep historical results and current-terms counterfactuals separate, using
matched calendars/accounts and the same three currency-consistent Sharpes.

This is a completed account-assumption amendment and source assessment, **not a
completed accounting repair or new economic result**. Stage A implementation and
Stages B-D remain open. Snapshots and hashes are in `v2_corporate_account_sources.json`.
Some sites allowed web-indexed reads but blocked direct snapshot retrieval; their
receipts retain that distinction. The active continuation uses this amendment.
