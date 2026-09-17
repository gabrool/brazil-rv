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
CVM also distinguishes industry holding companies from operating-company sectors;
this finer taxonomy is not a definitive test of broad B3 industry neutrality.
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

All twenty continuous books completed, and both full-history neutral controls
reproduce every sealed daily field exactly (maximum difference **zero**). The
two-arm CPU campaign took about **5m39s**, excluding implementation and reporting;
individual books took 26–42 seconds. The OS process wrapper reported unavailable
child exit codes and therefore skipped automatic summarization. All twenty
completion artifacts and the independent audit passed; summarization was then
run explicitly. The wrapper receipt is preserved rather than relabelled success.

**The dollar-neutrality constraint is the most useful finding.** Keeping the 5%
beta cap while allowing net up to 45% raises C6 from 4.000 to 6.707 bps/day above
CDI and TE from 4.497 to 5.937. C6's paired 40-session-block interval is
+2.707 [+.418, +4.974] bps/day; TE's is +1.440 [−.642, +3.543]. These are nominal
development intervals, not multiplicity-adjusted or untouched-test evidence.
Relaxing beta as well does less well: 5.488 C6 and 5.645 TE. The result supports
separating dollar neutrality from estimated market neutrality, not removing hedges.

The flexible books retain near-neutral estimated beta but are persistently net
long: mean net/beta is .442/.044 for C6 and .390/.047 for TE. This is not a learned
confidence switch between 60/40 and 40/60. Gross and realized volatility also rise.
C6's annualized BRL volatility moves from 10.35% to 13.05%, TE's from 11.03% to
12.46%; CDI Sharpes improve. Cash remuneration is accounted for: interest falls
as more capital funds net-long holdings. The improvement is not free extra CDI.

Neither sector constraint improves full-period net returns. Some drawdowns are
smaller, so neutrality is a risk tradeoff rather than a universal mistake.
The classifier's coverage and granularity limit conclusions about an alternative
full-coverage historical B3 industry taxonomy. No stocks were removed to obtain
these comparisons.

### Account statistics

Net is average daily return above CDI. Sharpes are ordered **BRL−CDI / USD−EFFR / BRL−zero**. Win/loss percentages refer to absolute BRL days; all books have zero flat days. Drawdowns include the initial account value.

**2018–2024, identical continuous start**

| Architecture / rule | Net bps/day | Three Sharpes | Winning / losing days | Max drawdown BRL |
| --- | ---: | --- | --- | ---: |
| TE_all / neutral | 4.497 | 1.028 / 0.472 / 1.722 | 55.8% / 44.2% | -12.92% |
| TE_all / flexible_net | 5.937 | 1.201 / 0.638 / 1.815 | 56.2% / 43.8% | -12.01% |
| TE_all / direction_permitted | 5.645 | 1.150 / 0.587 / 1.768 | 56.0% / 44.0% | -12.41% |
| TE_all / sector_band | 4.272 | 1.040 / 0.452 / 1.780 | 56.4% / 43.6% | -12.17% |
| TE_all / sector_neutral | 3.436 | 0.856 / 0.331 / 1.613 | 54.7% / 45.3% | -11.73% |
| C6 / neutral | 4.000 | 0.975 / 0.413 / 1.714 | 56.4% / 43.6% | -11.17% |
| C6 / flexible_net | 6.708 | 1.295 / 0.734 / 1.881 | 56.8% / 43.2% | -11.69% |
| C6 / direction_permitted | 5.488 | 1.098 / 0.570 / 1.705 | 56.6% / 43.4% | -15.49% |
| C6 / sector_band | 3.161 | 0.821 / 0.298 / 1.608 | 55.8% / 44.2% | -10.82% |
| C6 / sector_neutral | 3.506 | 0.873 / 0.342 / 1.630 | 54.9% / 45.1% | -10.38% |

**2020–2024, matched fresh start**

| Architecture / rule | Net bps/day | Three Sharpes | Winning / losing days | Max drawdown BRL |
| --- | ---: | --- | --- | ---: |
| TE_all / neutral_2020 | 4.952 | 1.048 / 0.563 / 1.744 | 56.7% / 43.3% | -12.92% |
| TE_all / supervised_unconditional | 5.084 | 0.831 / 0.524 / 1.370 | 54.5% / 45.5% | -23.86% |
| TE_all / supervised_conditional | -5.066 | -0.684 / -0.573 / -0.239 | 50.2% / 49.8% | -52.10% |
| TE_all / supervised_flexible | -4.390 | -0.557 / -0.454 / -0.139 | 51.4% / 48.6% | -51.87% |
| TE_all / supervised_directional | -5.315 | -0.572 / -0.523 / -0.217 | 52.0% / 48.0% | -61.48% |
| C6 / neutral_2020 | 4.461 | 0.965 / 0.507 / 1.676 | 56.5% / 43.5% | -11.17% |
| C6 / supervised_unconditional | 5.490 | 0.903 / 0.589 / 1.444 | 55.6% / 44.4% | -18.11% |
| C6 / supervised_conditional | 1.509 | 0.187 / 0.084 / 0.594 | 53.7% / 46.3% | -26.61% |
| C6 / supervised_flexible | 0.230 | 0.026 / -0.027 / 0.403 | 52.3% / 47.7% | -36.02% |
| C6 / supervised_directional | 1.720 | 0.174 / 0.089 / 0.507 | 51.3% / 48.7% | -49.91% |

### Why the bad halves occur

For TE's 2020H2, stocks make +6.330 bps/day and the hedge loses −11.125. However,
about +10.180 of stock P&L is the lagged-beta market component: residual stock
selection is **−3.850**, market-plus-hedge is −.945, and costs/borrow cost another
1.289. Calling the full hedge loss avoidable alpha destruction is misleading.
2018H1 has residual stock selection −2.957 and net −4.266. In 2023H2, residual
selection is −.981; hedge/market mismatch contributes +.714, while costs and borrow
consume 1.658. Different losses have different causes.

Relaxed net substantially reduces 2020H2 losses (TE −6.124 to −.688; C6 −5.741
to −.944) but worsens 2018H1 (TE −4.266 to −7.823; C6 −2.617 to −6.052). It does
not solve regime timing or eliminate bad halves. Issuer/sector contribution proxies
and their difference from exact ledger P&L are retained in the source analyses;
unknown classifications account for meaningful exposure and positive contributions,
which is another reason not to exclude those names.

### Conditional learning: failure and the diagnostic follow-up

The original 178-field common-context regression is insufficiently regularized
for this task: all twenty prior selections hit the strongest tested penalty,
and its conditional forecasts worsen squared error against a historical mean.
For TE, conditional alpha loses −5.066 bps/day above CDI versus +5.084 for its
matched unconditional calibration; C6 gets +1.509 versus +5.490. These are
2020–2024 fresh-start books, not directly comparable to the 2018-start table.
The forecast can reverse stock ranks when estimated alpha is negative. No
minimum gross is imposed, but noisy large forecasts keep gross near the cap;
the mere availability of cash does not make uncertain predictions calibrated.

The explicitly registered **post-hoc** follow-up expands shrinkage through
10,000 and includes the constant forecast as a selectable causal fallback. It
also tests seven compact context fields plus validity flags, selects penalties
separately by prediction head, and preserves purged prior-only fitting.
Common-context shadow models choose the constant in 7/10 TE and 5/10 C6 folds;
market models do so in 7/10 and 8/10. This sharply reduces the earlier excess
prediction error, but no head/feature-set combination passes the gate of positive
paired error improvement in both the early and later periods. No additional
financial replay is triggered. The compact models also fail that gate.

Thus we found a real regularization/design weakness in the first timing probe,
corrected its diagnostic range, and still found no reliable timing improvement
from these inputs and models. This does **not** establish that ML/RL is inherently
unsuitable or that no other state representation can predict conditional returns.
Five-session labels overlap; hundreds of daily observations are not hundreds of
independent regime examples. A supervised shadow-payoff target is also a probe
of gross stock opportunity, not the entire realized account objective.

### Decisions and remaining work

Keep flexible dollar exposure with tight beta as a **research candidate and
comparison policy**, not an automatically accepted trading policy. Do not promote
the timing models or force sector neutrality on the basis of these results.
Neither attention's architecture nor the completed forecast training was changed.

Before replacing the reference policy, prioritize settlement/event sensitivity,
financing/cost stress and exposure robustness for the flexible candidate. All books
retain unresolved-economics flags. Terminal haircut sensitivity, measured as NAV
difference divided by initial capital, is −.374/−.531 for TE neutral/flexible and
−.371/−.614 for C6; these are **not daily returns or final-NAV percentages**. The
larger sensitivity prevents treating the reported improvement as validated live P&L.

An economic checkpoint-selection experiment remains distinct from regularization:
the completed forecasters still select on the existing neutral criterion. It
requires checkpoint-specific earlier-selection forecasts and a frozen selection
rule, not choosing checkpoints from these evaluation returns. It was not silently
performed by this CPU portfolio pass. Existing completed fits should be reused
when that separately registered experiment is undertaken.

Multi-period retention remains a targeted follow-up if forecast decay/turnover
evidence supports it. Costs matter in 2023H2, but the larger 2018/2020 failures
also contain negative stock residual opportunity; a more elaborate trade planner
cannot by itself supply a missing timing forecast. This pass therefore does not
expand into a speculative receding-horizon controller. Parent/fresh transfer,
new market/event histories and alternative fusion from original D/F also remain
conditional research ideas, not completed work. Intraday intervention stays deferred.

The [compact numerical review](v2_opportunity_results.json) binds full daily
records and attribution. The [reliability follow-up](v2_opportunity_reliability.json),
[completion audit](v2_opportunity_audit.json) and
[verified D-drive recovery](v2_opportunity_recovery.json) accompany it. Recovery
contains all 180 run files in a 222.6 MB archive, with every member read back and
hashed. Original large economic caches and the accepted store remain source-bound
references. All requested statistics use actual account returns.
No forward capture, held-out consumer access, new GPU training or paid instance
was needed.
