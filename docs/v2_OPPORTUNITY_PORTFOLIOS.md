# Conditional opportunity and portfolio arrangements

This is the user-authorized follow-up to the completed decision Phases 1–3.
Read with the [registration](../research/preregistrations/v2_opportunity_portfolios.md)
and [canonical run pointer](v2_opportunity_run.json). Prior results remain sealed.

## Question and controls

Separate three questions: do neutrality constraints unnecessarily cost money; can
we forecast when the stock-selection signal works; and can a coherent market
forecast improve optional directional allocation? Negative half-years alone do
not establish any of these. A negative hedge contribution can be offsetting a
positive common-market component in the stocks rather than destroying stock alpha.

The fixed portfolio comparisons use completed Phase-3 neutral three-seed ensemble
forecasts, unchanged dated eligibility and economic inputs, all 14 development
folds and one continuous 2018–2024 account. Baseline economic mappings are the
sealed prior mappings. Newly supervised calibration instead uses only earlier
new-ensemble OOS dates; 2018–2019 supplies its initial history and comparisons
begin in 2020. An independently restarted neutral control isolates the different
account boundary. No GPU fitting or new pretraining is involved.

## Portfolio choices

All portfolios retain optional CDI-earning cash, transaction and borrowing costs,
five-session planning, a 5% individual stock cap, 60% BOVA cap, and 225% joint gross
ceiling. The neutral control limits absolute net and beta to 5% NAV. Flexible-net
allows net 45%, retaining beta 5%; direction-permitted allows both 45%.
These are exposure bands, not a rule that forces 60/40 at every gross level.
Sector variants retain the neutral caps and impose either a 5% sector net band
or zero known-sector net. Missing classifications remain eligible.

The classifier is the existing receipt-dated **CVM FCA sector code**, not B3's
present taxonomy. It covers 77.12% of eligible forecast stock-days in 2018–2024.
B3 itself describes periodic sector reclassification; assigning today's labels
to past books would not be a point-in-time experiment. A verified historical B3
hierarchy was not established in this pass. Source:
[B3 classification criteria](https://www.b3.com.br/pt_br/produtos-e-servicos/negociacao/renda-variavel/acoes/consultas/criterio-de-classificacao/).

Known-sector neutrality means stock dollar exposure, not sector-factor neutrality.
BOVA is not assigned an invented sector, and unknown stocks cannot make the
entire portfolio sector-neutral. Strict neutrality can suppress a singleton
sector; the looser sector-band control measures that tradeoff.

## Conditional learning and causal discipline

One supervised observation per date predicts the five-session equal-rank shadow
payoff and BOVA excess return, in daily-return units. Inputs reuse verified common
market context but recompute shadow outcomes and seed/horizon disagreement from
the new ensemble. At decision t, origin t−6 is the latest admitted five-session
label. Common stock residual intercept stays at the same historical average.
Median/IQR and smooth asinh are fitted on preceding observations only. A purged
126-session prior selection period chooses ridge strength from 1/10/100; refits
then use all matured prior dates. A simple historical-average forecast is the
matched comparison. This is a bounded probe, not an exhaustive test of timing.

The replay includes conditional alpha under both neutral and relaxed constraints,
then adds the same predicted market premium to BOVA and stock beta exposure.
This separates changing the constraints from adding a directional forecast.
All studied years are development history, including the later 2022–2024 check.

## Reporting conventions

Three annualized daily Sharpes use sample standard deviation and 252 sessions:
BRL returns minus CDI; BRL returns minus zero; and USD returns minus US cash.
USD valuation converts consecutive NAV endpoints with historical BCB PTAX
BRL-per-USD fixes. Holiday carries use only prior fixes and are disclosed.
EFFR is explicitly an overnight US cash **proxy**, with ACT/360 calendar accrual;
it is not a Treasury instrument or an executable deposit guarantee. These are
ex-post benchmark inputs, never predictive features. Sources:
[BCB exchange-rate history](https://www.bcb.gov.br/estabilidadefinanceira/historicocotacoes),
[New York Fed EFFR](https://www.newyorkfed.org/markets/reference-rates/effr).

Win/loss/flat fractions distinguish absolute BRL days from days above CDI.
Maximum drawdown includes initial NAV, and currency-specific return/drawdown
labels are explicit. Undefined Sharpe is null. Currency conversion can materially
change results: this is the unhedged USD investor's experience, not an FX-hedged book.

## Engineering and verification

Financial implementation is frozen at `056aad2`; the source worktree is isolated
from subsequent reporting edits. Thirty-eight targeted tests passed, including
future-mutation/maturity, currency/calendar, allocation gradients, sector bounds,
and coherent stock/hedge market preferences. Runtime uses two independent CPU
arms and one numerical thread each. All names, histories and FP64 accounting
remain; each first neutral replay must reproduce its sealed daily account.

The completion audit verifies input hashes, prior-only fitting, artifact hashes,
consecutive dates, account reconciliation and currency-statistic reproduction.
Sector/issuer P&L contributions are labelled shareholder-return-based proxies;
unattributed differences from exact ledger stock P&L remain visible. Historical
settlement uncertainty and the terminal haircut sensitivity are never hidden by
the additional statistics.

## Results and disposition

Pending the registered CPU replays and completion audit. No candidate has been
promoted by this program at this checkpoint.
