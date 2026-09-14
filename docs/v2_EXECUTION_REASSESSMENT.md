# Economic reconciliation and learned portfolio policy

14 September 2026. Prepared for independent LLM review.

The user's objective is a model that learns profitable long/short or market-neutral decisions with realistic costs, without unnecessary RL machinery or conservative restrictions that destroy useful opportunities. The existing execution policy is a historical comparator, not a design requirement. Intraday intervention is a future direction, not part of this implementation pass.

This pass is **analysis**, with source-bound attribution and fixed-signal CPU replays. It introduces no new alpha fits, portfolio-policy fits, GPU instance, live infrastructure, forward capture or 2025/2026 consumer read. Recommendations below are not implemented experiments or proven improvements.

## What the analysis establishes

1. The apparent baseline IC deterioration was mainly a comparison of different historical windows. On the same four folds, Round-7 A0 IC is .018725 and current S0 is .018849.
2. The economic gap to Round-6 C6 is real in the recorded backtests. It comes predominantly from equity positions, not unusually punitive costs or hedging imposed on current attention.
3. The known accounting repairs do **not** remove C6's advantage. Replaying its unchanged signals increases its four-fold net result from 6.593565 to **6.789236 bps/day**. Its 2018 result is unchanged.
4. The current system does not learn a portfolio policy from P&L. Its learned forecaster is followed by a hand-built allocation/retention/hedging policy. This is an important difference from the user's intended system.
5. A small stateful policy trained through differentiable portfolio accounting is a credible next design. A critic or learned value function is not required under an exogenous-price, adequately modeled fill/cost assumption. Removing the critic does not remove financial estimation error or simulator risk.

## 1. Evidence and comparison boundaries

The new [attribution evidence](v2_execution_reassessment_evidence.json) reads **36 saved books**: five current candidates, three Round-6 candidates, and Round-7 A0, each on F2/F6/F10/F14. Every daily P&L decomposition reconciles with its recorded net return within 1e-7 bps. Current archive members match the recovery inventory's SHA-256. The four detailed F2 name attributions reconcile within 1.3e-11 bps per day.

The separate [C6 accounting bridge](v2_c6_accounting_bridge.json) runs eight headline ledgers: original and repaired accounting for each of four folds. The original replay matches **every saved daily record exactly**, including accounting and order-state fields. All repaired books pass the existing engineering gates. The eight replays completed in approximately 26 seconds locally; no training was needed.

Reproduction scripts: [attribution](../ops/analyze_post_data_economics.py), [accounting bridge](../ops/reconcile_c6_accounting.py). They resolve historical roots from their published recovery/input records and do not alter sealed experiments. The bridge uses the accepted Round-7 economic store; the subsequent data repair preserves execution inputs and all economic-beta arrays, so it addresses the accounting transition relevant here. It does not feed historical checkpoints through new preprocessing or claim that C6 was retrained on repaired data.

All comparisons below use the second halves of 2018, 2020, 2022 and 2024: 501 book sessions, 461 common three-head IC dates, and seeds 11/29/47. IC is the mean of D3/D5/D10 Spearman correlations, not percentage prediction accuracy.

| Model | Four-fold IC | Original-accounting net bps/day |
| --- | ---: | ---: |
| Round-6 S0 | .016916 | .997919 |
| Round-7 A0 | .018725 | 1.142334 |
| Current S0 | .018849 | -.027859 |
| Current S0, common training recipe | .022241 | 2.860661 |
| Current TE_all, ASAM .2 | .027103 | 1.440401 |
| Round-6 C6 | .026470 | 6.593565 |
| Round-6 C6, fresh P | .023691 | 3.081775 |

“Original-accounting” means each row's own saved version, not one uniform version across rounds. The bridge below isolates that difference for C6. Its roster was fundamentals plus magnitudes; it was not an all-dataset attention model.

Round-6 full-14-fold headlines were S0 .025616 IC / 4.508188 net and C6 .027807 / 6.071385. Round-7 A1 illustrates the danger of projecting a screen result: it reached .028245 IC on these four folds but lost its advantage over A0 in the full development comparison.

C6 was selected from a development search, with unstable designation across seed omissions. Its all-fold paired IC and net intervals against S0 include zero. Its attractive historical result must remain a benchmark to explain, not a guaranteed 6-bps entitlement. The current TE_all-minus-S0 IC interval excludes zero nominally, but its net interval does not, and neither accounts for every research choice made across rounds.

## 2. Where the money went

### Recorded P&L decomposition

Entries below are mean bps/day over the same 501 sessions. Cost rows are positive expenses. Financing is cash interest plus short-proceeds interest minus the all-cash CDI benchmark.

| Component | Round-6 C6 | Current TE_all .2 | Current S0 |
| --- | ---: | ---: | ---: |
| Equity gross P&L | 9.418 | 3.183 | 2.813 |
| Hedge gross P&L | -.861 | .177 | -.534 |
| Financing after cash benchmark | -.002 | -.148 | -.273 |
| Equity trading expense | .858 | .711 | .954 |
| Hedge trading expense | .067 | .073 | .077 |
| Equity borrow expense | .828 | .808 | .820 |
| Borrow registration expense | .136 | .135 | .138 |
| Hedge borrow expense | .072 | .044 | .044 |
| **Net excess P&L** | **6.594** | **1.440** | **-.028** |

C6's equity-gross advantage is **6.235 bps/day**. Current attention's hedge gross is **1.038 bps/day better**, and its equity trading expense is lower. Consequently, removing the hedge or making the cost model more generous cannot be the main explanation of the gap. Those changes might alter performance in a new replay, but subtracting recorded hedge P&L is only attribution: it does not reconstruct changed NAV, funding, orders, risk or positions.

The .008254 IC gain of current attention over current S0 translates into only **.369496 bps/day additional equity gross**. Of its 1.468260 net advantage, approximately .711484 comes from a better recorded hedge gross and .243282 from lower equity trading expense. Thus the forecaster's IC improvement is not showing up proportionately in the equity book even before costs.

The Round-7-A0 to current-S0 net decline of 1.170193 bps/day is also primarily stock P&L: equity gross falls 1.354158, while hedge gross improves .206401 and equity trading expense is almost unchanged. This is compatible with different selected holdings after the three slow-encoding changes and fresh fits; it is not evidence that a new execution cost assumption caused the decline.

### Accounting bridge

| Fold | C6 before repair | Same C6 scores after repair | Change |
| --- | ---: | ---: | ---: |
| F2 / 2018 H2 | 7.614945 | 7.614945 | 0 |
| F6 / 2020 H2 | 2.286655 | 2.286655 | 0 |
| F10 / 2022 H2 | 3.736453 | 3.736453 | 0 |
| F14 / 2024 H2 | 12.720010 | 13.491908 | +.771898 |
| **Pooled** | **6.593565** | **6.789236** | **+.195671** |

The bridge isolates accepted retrospective action/continued-print accounting, not fundamental-feature repairs or model training. Unknown source errors can still exist, but the demonstrated repairs cannot be invoked to dismiss C6's advantage.

### 2018 is the principal failure window

Original C6-minus-current-attention net is 5.153164 bps/day pooled. F2 contributes 3.776144, approximately **73%**. F2 equity gross is +12.117527 for C6 and -4.664422 for attention. Their ex-ante residual beta after the hedge is similar in magnitude (mean absolute .01753 and .01745), and neither F2 book reports an entry candidate blocked by a gross/net/name cap. This points away from those caps as the explanation in this window; it does not prove the risk model is perfect.

The five largest security-level differences account for **9.520186 of 16.781949 gross bps/day**, approximately 57% of the F2 equity gap:

| Permanent security ID | C6 gross contribution | TE_all gross contribution | Observed difference in positions |
| --- | ---: | ---: | --- |
| BRCMIGACNOR6 | +.471 | -1.797 | C6 positive mean held weight; attention negative, held on 66 days |
| BRGOLLACNPR4 | +2.211 | +.097 | Both positive mean held weight; C6 held on 50 days versus 17 |
| BRMOVIACNOR0 | 0 | -1.959 | Absent from C6; attention negative mean held weight, held on 97 days |
| BRCYREACNOR7 | 0 | -1.779 | Absent from C6; attention negative mean held weight, held on 79 days |
| BRMULTACNOR5 | +.736 | -.664 | Opposite mean held signs, both held on most days |

These are contributions to each original book's gross P&L using its own daily NAV, not standalone security returns. “Held days” can include separate spells and is not a claim of uninterrupted holding. Identity is preserved; no present-day ticker mapping is used. Cash claims and actual fill flows are included, and no cross-identity transition occurred in these four audited F2 books.

The end-of-session held-name-day median age is 32 sessions for C6 and 18 for attention in F2; their corresponding means are 41.0 and 27.5. These distributions overweight long-lived holdings and include censored positions. They do not establish that holding longer is beneficial. They show that a model trained on 3/5/10-session labels is being evaluated through holdings that often persist much longer, including persistent wrong-side exposure.

No hindsight stop, name exclusion or 2018-specific rule follows from this table. A policy must distinguish such situations using information available before its action; simply memorizing these five losers would be another backtest optimization.

The existing saved D5-only diagnostic also fails to provide a universal shortcut. Attention's pooled net falls from 1.4404 to .8991 when only its D5 head is used under the same book settings, while individual folds change in both directions. Learned horizon combination remains a research question.

## 3. What is questionable in the current design

| Current choice | Why it can matter | Recommended treatment |
| --- | --- | --- |
| Equal average of separately ranked D3/D5/D10 scores | Discards score magnitude/calibration and gives correlated, overlapping horizons equal influence regardless of regime, cost or position | Learn an economic mapping; keep the equal-rank rule as a control |
| Thirty positions per side and roughly 200% equity gross | Forces exposure and equal initial size even when incremental opportunities are weak | Permit variable positions/exposure within a fixed risk budget; retain cash as an economic option |
| Six entries per volatility quintile plus fixed retention buffer | Hard quotas can displace attractive names; retention depends on rank rather than expected value of keeping versus replacing a position | Joint risk/cost allocation; do not impose quintile occupancy by default |
| No voluntary resizing to changing conviction except exits/risk trims | Cannot gradually reduce a deteriorating bet or distinguish a marginal retained name from a strong one | Predict target trades/positions conditional on current inventory |
| Fixed hedge applied after constructing the equity book | The two decisions do not jointly weigh alpha, hedge cost, funding and risk | Optimize equity weights and any hedge jointly |
| Clipped, volatility-scaled, characteristic-neutral ranking target | Optimizes something different from return in reais; removes variation related to volatility groups, beta groups and liquidity before portfolio construction | Preserve it as an auxiliary/control objective; test economic training with risk imposed on the portfolio |
| IC-based checkpoint/model selection | A stronger all-name ranking can have worse held-tail outcomes or different horizon decay | Keep forecast diagnostics, but select a learned portfolio by declared net utility and risk criteria |
| Fold-end forced liquidation/settlement | Boundary costs and unresolved marks can dominate a short evaluation and distort learned exits | Use continuous model-switch evaluation as the economic primary for the new policy; retain fold diagnostics and explicit valuation assumptions |

The exact current target first median-centers shareholder return, divides by lagged volatility times sqrt(horizon), clips to ±5, and residualizes against ten volatility-group indicators, five beta-group indicators and linear rank-Gaussianized log ADV, with a smaller-cross-section fallback. This is much stronger than the user's requirement to remain long/short or approximately market-neutral. It is causal label engineering, not direct feature leakage, but its economic desirability is open.

Likewise, the existing minimum mean-gross and volatility-occupancy acceptance gates belong to the old policy. They must not automatically reject a new risk-aware strategy for intentionally holding less exposure or allocating differently. Essential accounting, causality, feasible-trade and risk-limit checks remain.

Equal sizing, buffering and hedging are not established defects merely because they are hand-built. The known result proves a mismatch of objectives and a rigid action map; replacing them must earn its improvement empirically.

## 4. A methodology that fits this problem

### Direct, stateful policy learning

My preferred first candidate is a **small shared portfolio policy on top of frozen forecasts**, trained on realized net portfolio outcomes through a differentiable sequential accounting model. Use a constrained optimization layer to express risk and cost trade-offs, with a direct constrained-weight network as a subsequent comparator if needed. This avoids retraining a large attention encoder for every policy experiment.

At decision t, the policy receives the separate horizon scores, permitted causal market/stock state, current drifted inventory, cash, pending commitments, known borrow/cost information and risk estimates. It emits trades or target exposures. After realizations arrive, the simulator updates actual shares, cash, costs and NAV; that state becomes input at the next decision. Holdings provide the essential recurrence even if the policy's trainable network is a small MLP.

Train on the sum of **non-overlapping one-period accounting increments**. A schematic objective is:

`maximize mean(realized net excess P&L) - fixed risk penalty`

subject to gross, individual-name, liquidity/borrow and explicit neutrality constraints. The actual implementation must reconcile self-financing cash, short proceeds, collateral and actions, rather than use this shorthand as an accounting formula. Sharpe and drawdown are important evaluation statistics; maximizing a noisy minibatch Sharpe alone is not my preferred first loss. Pure P&L maximization without a fixed risk/leverage contract can reward risk concentration instead of improved decisions.

For this candidate, backpropagation differentiates through the policy and inventory transitions while treating the realized market path as exogenous. It needs neither a Bellman value function nor an actor-critic system. Differentiability may be piecewise; non-smooth fills and real-world constraints require an explicit surrogate and exact-ledger verification. Market impact that materially changes future prices, queue-dependent fills, or unobserved borrow decisions would weaken the counterfactual assumptions. They cannot be solved by calling the model “RL.”

This mechanism can learn to reduce, exit, retain or reverse a losing position when its *conditional future value* changes. Past P&L alone is not evidence that a trade should be closed: an entry price is a sunk reference unless the path since entry contains incremental information or triggers a real constraint. Do not impose a loss threshold as the first purportedly optimal solution. Compare a few causal state variables such as forecast changes and volatility-scaled adverse movement only where they add information absent from the current forecast/state.

### A transparent constrained allocation layer

A useful initial layer chooses weights by trading off:

`mu_t' w - risk_aversion * w' Sigma_t w / 2 - trading_cost(w - w_drifted) - holding_cost(w)`

The network can learn the mapping from horizon scores/state into the economic preference `mu_t`; risk-budget limits and economic cost schedules remain externally defined, not freely learnable knobs that the model can turn off. If `mu_t` is presented as expected return rather than merely a decision parameter, it must be calibrated and evaluated in return units on causal held-out predictions. A rank of .8 is not an expected gain of .8 bps.

Include the hedge as an optional asset in the same allocation problem. Prefer a small causal factor-plus-diagonal covariance estimate to a noisy unrestricted covariance inverse. Constrain planned signed net and estimated market beta explicitly, with an operational tolerance for unfilled orders. Dollar neutrality and beta neutrality are different; neither implies neutrality to every sector or style. Beta estimation error requires reporting realized residual exposure as well as the ex-ante constraint.

I would initially use the existing gross ceiling as an upper risk budget, with **no compulsory minimum deployment**. Compare policies at a common declared ex-ante risk budget and report cash time and gross exposure; do not rescale each result using its own evaluation-period volatility to manufacture comparability. Include a cash comparator and ensure low exposure is not mistaken for alpha.

Linear transaction costs naturally permit a no-trade region; continuous quadratic adjustment costs induce gradual movement. These offer a principled starting point for holding behavior, instead of choosing one rank buffer for every name and regime. A multi-period extension can plan future desired allocations and execute only the first action. Future *forecasts* may enter the plan, not realized future prices.

A network may learn horizon combination directly without treating the heads as explicit expected returns. If a multi-period optimizer uses cumulative 3/5/10-session return forecasts, their overlap must be handled: they are not independent payoffs that can be added. Calibrate a coherent forecast path or incremental intervals before using them that way.

### Why these methods, and what the literature actually supports

| Primary source | Relevant support | Important limit for Brazil-RV |
| --- | --- | --- |
| [Boyd et al., Multi-Period Trading via Convex Optimization](https://web.stanford.edu/~boyd/papers/cvx_portfolio.html) | Explicit return, risk, trading/holding costs; multi-period planning with only the first trade executed | An allocation/control framework, not proof that our forecasts contain sufficient alpha |
| [Gârleanu and Pedersen, Dynamic Trading with Predictable Returns and Transaction Costs](https://www.nber.org/papers/w15205) | Current holdings, signal decay and future desired portfolios jointly determine trading | Closed-form theory under restrictive assumptions; empirical application is commodity futures, not this B3 panel |
| [Agrawal et al., Differentiable Convex Optimization Layers](https://arxiv.org/abs/1910.12430) | A solver can be part of a trainable network, with gradients passing through its solution | General optimization machinery; numerical feasibility and throughput need our own acceptance |
| [Buehler et al., Deep Hedging](https://arxiv.org/abs/1802.03042) | Direct neural control under costs/constraints using pathwise training | Derivative hedging, including simulated Heston examples; no demonstration of Brazilian equity alpha or unlimited independent real-market paths |
| [Lim, Zohren and Roberts, Deep Momentum Networks](https://arxiv.org/abs/1904.04912) | Direct performance training can learn signals and sizing, including turnover considerations | 88 continuous futures contracts and time-series momentum, not a changing cross-sectional equity universe |
| [Zhang, Zohren and Roberts, Deep Learning for Portfolio Optimization](https://arxiv.org/abs/2005.13665) | Direct portfolio-weight learning without a separate expected-return forecast | Four market-index series and allocation, a much narrower action space than our stock book |
| [Butler and Kwon, Efficient Differentiable Quadratic Programming Layers](https://arxiv.org/abs/2112.07464) | ADMM/implicit-differentiation methods address solver cost inside learning | Paper timings cannot be used as our throughput estimate; benchmark the full sequential workload |

The proposed design is my synthesis for this research contract, not a claim that the literature identifies one universally optimal architecture. There is no reason to add PPO, a critic, simulated market generation or a separate value network as the default first implementation.

## 5. Realism without gratuitous conservatism

Separate the **policy**, which may change substantially, from the **accounting and fill model**, which establishes what a proposed action could have earned.

- Keep permanent identities, causal eligibility, real action/cash-claim accounting, short-side availability, transaction/borrow costs and self-financing reconciliation. These protect economic meaning, not an arbitrary old strategy.
- Maintain the 15:45 information cutoff. The current fill is a later close proxy, not a guaranteed 15:45 execution. A new policy must not choose quantities using the later close or NAV as though known at decision time. Intended notional can become a realized quantity at the later fill; that realized quantity may affect subsequent state.
- Use an explicit decision-to-fill convention and plausible cost sensitivities. A fixed 4-bps-per-side scenario is a research assumption, not measured per-name executable cost. Neither reduce it to restore a favorite result nor add several overlapping conservatism penalties.
- Results at a normalized one-real NAV do not establish capacity. Costs/participation must eventually be assessed at a stated capital size with point-in-time volume. Do not introduce an arbitrary liquidity exclusion instead of modeling feasible size, and do not claim calibrated market impact from the present close-only simulation.
- Keep finite missing-price/action uncertainty visible. A forced stale-mark settlement is an accounting scenario, not a witnessed fill. A learned policy must not profit by choosing names whose missing returns are set to zero or by anticipating artificial fold-end settlement.
- Current short-proceeds remuneration and debit/borrow conventions include favorable assumptions as well as frictions. Audit actual intended funding arrangements before calling net returns executable; applying an extra conservative haircut everywhere is not a substitute.
- Evaluate neutrality alternatives by net utility **and risk**: joint stock-only beta control, or stock-plus-hedge control. An unhedged comparator can diagnose hedge value, but additional market exposure is not automatically extra alpha. A futures hedge may merit a later comparison only with proper contract/roll/margin/basis data.

There is no need to build live order management, queue simulation or intraday data collection now. The essential next realism check is a small differentiable learner whose output can be replayed under an independently verified economic contract.

## 6. Leakage, overfitting and computational design

The largest new learning risk is **training a second model on forecasts that were themselves fitted to those same outcomes**. Frozen weights do not make their training-period predictions out of sample. The previous .8 fit IC pathology would make that shortcut especially misleading.

Two valid routes are available:

1. Fit a frozen-forecast portfolio policy on strictly chronological out-of-fit predictions, with a later, separate policy-selection window and evaluation. Purge overlapping forecast/calibration labels at boundaries. The four current TE evaluation windows alone do not provide a rich independent meta-training history; additional causal inner prediction blocks are required for a serious controller comparison.
2. Train forecaster and policy jointly on an allowed fit period, then select/evaluate the entire combined system on later periods. This is end-to-end training, not an out-of-fit meta-model; it is valid but more expensive and more vulnerable to representation overfitting.

Start with the first route and cached predictions. Never let F14 outcomes select a controller used to claim historically deployable F2 performance. Global development choices should retain their retrospective label, even if all feature timestamps are causal.

For a sequential policy, preserve actual inventory and costs across adjacent decisions. Training chunks need causal state initialization/burn-in and no fictitious free liquidation at each chunk edge. Truncated gradients are an approximation to test, not justification to erase held positions. Future mutation should leave earlier **intentions** unchanged while allowing later fills and returns to change legitimately.

Efficiency priorities:

- Precompute frozen-forecaster scores and permitted features once; policy experiments should not repeatedly execute a 60-session encoder.
- Use all eligible names and the existing compact axis; preserve the forecaster's full lookback and capacity.
- Share the policy across securities with mask-aware operations; avoid one separate RL agent per stock.
- Begin with a small action head and a structured covariance/cost layer. Warm-start successive solves; benchmark dense solves against structured/ADMM alternatives before committing to one implementation.
- Keep financial accounting and constraint residuals in FP32 or FP64 where needed. AMP/compile are appropriate for neural work after parity checks, not reasons to use low-precision cash accounting.
- Verify gradients, feasibility, and learned no-trade behavior on small controlled paths; then measure complete training and exact replay time. Do not report a layer microbenchmark as a full-run speedup.
- Diagnostics should track net/gross P&L, costs, risk/exposure, active constraints, forecast-to-position sensitivity, holding-time/exit behavior, long/short attribution and surrogate-versus-ledger discrepancy. Avoid a large generic monitoring framework.

The number of independent portfolio dates is much smaller than the stock-day count. A direct economic loss does not create more data. Start with modest policy capacity and paired chronological comparisons; retain the incumbent and cash controls even if a neural policy looks attractive in sample.

## 7. Proposed next experiment order

This is a recommended sequence, not a silently launched extension of A–C.

| Step | Work | What it can establish |
| --- | --- | --- |
| **Completed: economic reconciliation** | Saved-book decomposition, F2 name/holding investigation, fixed-C6 accounting bridge | Costs/hedging and known accounting repairs do not explain away C6; identifies concrete stock-position differences |
| **1. C6 representation bridge** | Reproduce the C6 input roster and inherited-parent recipe on accepted repaired inputs, with compatible newly trained parents and the old policy as control | Whether the useful old family combination survives data/coordinate repairs; not an attention or policy verdict |
| **2. Small policy engineering and causal prediction cache** | Constrained optimizer/control, differentiable accounting parity, chronological forecast blocks | Whether a controller can be trained validly and quickly; engineering success is not financial success |
| **3. Bounded learned-policy comparison** | Old policy; simple cost/risk optimizer with calibrated frozen forecasts; small stateful decision-trained policy. Initially use S0 and TE_all .2; include repaired C6 when available | Whether learning sizing, horizon use and retention improves net utility under the same risk and fill contract |
| **4. Remaining-fold confirmation** | Freeze survivors and run matching remaining development folds, reusing compatible A–C fits; continuous model-switch book | Whether gains persist outside the four screen windows and survive continuous accounting |
| **5. Conditional architecture/objective extension** | Rich early-versus-late attention control, raw-return/neutral auxiliary objective, and only then joint encoder-policy fine-tuning if justified | Which representation and learning objective help the portfolio rather than merely the proxy IC |

Freeze the small roster, budgets, seeds, risk limits, cost basis, selection objective and advancement rule before fitting. Use three matched seeds for financial comparisons; a deterministic convex solver does not need three redundant fits. Do not expand simultaneously across architectures, loss functions, hedge instruments and dozens of risk settings.

The C6 bridge should preserve the historical family roster, not its demonstrated source mistakes. If parent-input coordinates change, regenerating compatible parents is necessary; feeding old weights new transforms would not be an isolated ablation. Distinguish inherited slow-P transfer from fresh all-family P, since the historical experiment already shows material sensitivity to that choice.

A short CPU screen can establish accounting parity and controller learnability. It cannot settle financial superiority if it uses in-sample forecast inputs, too few independent dates or an unrealistic differentiable fill model. Conversely, changing the portfolio policy does not automatically require another full attention pretraining program.

## 8. Future intraday intervention

Separate **forecast horizon**, **decision frequency** and **order execution**. A multi-day thesis can remain multi-day while its exposure is reconsidered during the day. More frequent decisions do not require training an intraday alpha model or closing the entire book every evening.

The future extension would keep a slow alpha state, update observable prices/risk/liquidity and permitted event information intraday, and call the same inventory-aware controller when state changes warrant a decision. A fast risk overlay can initially reduce exposure or enforce constraints; expanding it to new discretionary alpha trades requires a separate experiment. The system should be able to react to changes in a thesis, not mechanically to every noisy price move.

That work needs reliable historical intraday decision/fill alignment and realistic latency/costs. Daily bars cannot validate an intraday stop, nor can we assume a stop fills at its trigger through a gap. Keep the state/action interface extensible, but do not build this layer or restart forward capture now.

The immediate priority is to reconnect predictions with economic decisions while preserving the historical evidence. Attention has become a credible candidate; the current economic mapping has not yet established that it extracts the candidate's best attainable net return.
