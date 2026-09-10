# Round 5 continuous S0 book

The registered continuous replay is complete. It uses one chronological portfolio over all 1,738 saved-score dates from 2018-01-02 through 2024-12-30, with 13 model switches. Mean daily net excess is **4.550 bps**, versus **4.822 bps** for repaired fold resets and **4.936 bps** for sealed historical fold resets. Carrying the actual book therefore changes the repaired comparison by **−0.272 bps/day**. This is descriptive development evidence; the S0 architecture and execution policy were already selected on these development data.

| Portfolio calculation | Mean net excess, bps/day | Difference from continuous |
| --- | ---: | ---: |
| Continuous, repaired BOVA/beta | 4.549761 | — |
| Repaired fold resets | 4.822005 | +0.272244 |
| Sealed historical fold resets | 4.935761 | +0.385999 |

## Contract and continuity

Code commit `c1de021` ran from a clean detached worktree. The frozen design binds every sealed score manifest and its literal score/mask arrays, all economic inputs, the completed repaired comparator files, and ledger/evaluator/policy/input-builder source code. Every head has some saved coverage on all 1,738 dates, including fold tails. Missing name-level masks stay missing. The new model supplies scores at its first decision; the canonical execution lag applies. No missing tail predictions, model fits, or 2025/2026 observations were invented.

Policy is the accepted equal-weight 3/5/10-session signal, θ=1, nine buffered candidates per volatility quintile, and existing position/risk/borrow rules. Repaired BOVA prices and economic beta are used; accepted lending shortability and rates remain the sealed archive. No newly sourced pre-2023 lending rates were admitted.

There is one ledger initialization and no internal terminal liquidation or end-of-evaluation cancellation. Prior closing NAV equals next opening NAV exactly at every switch; opening long/short counts match carried positions. Cash, hedge collateral, receivables/payables and pending instructions remain in the same ledger state. The machine-readable report preserves the actual cash and claim values around every boundary. Claims happen to be zero at these 13 real switches; a targeted fixture proves unpaid corporate claims survive a switch until their actual payment date. A partial-fill fixture proves the same pending order ID fills on both sides of a switch.

| Switch | New model decision | Carried long / short names | Pending exits before switch | Old order IDs resolved later |
| --- | --- | ---: | ---: | --- |
| F1 → F2 | 2018-07-02 | 30 / 30 | 0 | None |
| F2 → F3 | 2019-01-02 | 30 / 30 | 0 | None |
| F3 → F4 | 2019-07-01 | 30 / 31 | 0 | None |
| F4 → F5 | 2020-01-02 | 30 / 30 | 0 | None |
| F5 → F6 | 2020-07-01 | 30 / 30 | 0 | None |
| F6 → F7 | 2021-01-04 | 30 / 30 | 0 | None |
| F7 → F8 | 2021-07-01 | 30 / 30 | 0 | None |
| F8 → F9 | 2022-01-03 | 30 / 30 | 1 | order-00008046 |
| F9 → F10 | 2022-07-01 | 30 / 30 | 2 | order-00009345, order-00009359 |
| F10 → F11 | 2023-01-02 | 30 / 30 | 0 | None |
| F11 → F12 | 2023-07-03 | 30 / 30 | 0 | None |
| F12 → F13 | 2024-01-02 | 30 / 30 | 1 | order-00011901 |
| F13 → F14 | 2024-07-01 | 30 / 30 | 0 | None |

All D1–D5/registered engineering checks pass for the headline, the other 12 sensitivity/comparator books, and the D5-only diagnostic. The final 2024-12-30 boundary leaves equity and hedge inventory flat, with no unpriced final inventory or unpaid claims.

## Economic uncertainty remains

**`economics_unresolved=true` remains explicit.** During the seven-year path, 22 stale-name exits use the registered last-mark settlement convention after ten sessions. These are lifetime settlements, not 22 unpriced positions at the final date. The registered haircut scenario changes terminal NAV from 3.4674 to 2.9132 and compounded net excess versus all-cash from 1.0450 to 0.7181. The inferred COTAHIST/DISMES action terms and reconstructed schedule retain their source labels; they are not independently verified corporate-action economics. The repaired reset comparators also retain unresolved F8/F9 labels; sealed resets retain F4/F8/F9.

Pre-2023 borrow pricing remains a material limitation: placeholder rates price 66.96% of short notional across this path. The annualized descriptive excess-return Sharpe is 0.7774, mean equity gross exposure is 1.9742× NAV, and mean daily turnover is 0.2799× NAV. These figures do not authorize deployment or supply fresh selection evidence.

## Reproducibility and runtime

Runtime was 121.4 seconds with **3.438 GiB peak working set** for the actual Python worker. Windows launches that worker through a tiny virtual-environment launcher; the separate worker-memory record is the relevant memory measurement. All saved JSON/file hashes were reverified after the PC reboot; the completed simulation was reused. Five targeted tests cover score-mask preservation, calendar gaps, future model-score mutation, pending orders, claims, and internal terminal resets. Ruff passed.

The compact [machine-readable review](v2_round5_continuous_book.json) includes all 13 boundaries, 14 fold means, source hashes, policy, uncertainty flags and actual-worker memory evidence. The full 160 MB economic ledger is retained locally, not copied into GitHub:

`D:\quant-data\b3\interim\round5_data_20260910T140442Z\continuous_s0_c1de021`

Frozen continuous design SHA-256: `8d5e8e1a012d7b3b4df1d9ab07c0378d67b673646414d262e20bbf92b2fb7027`. Parent repair design SHA-256: `75b8c314fd8a586ef9c82cde6b59db08f22ed12189f38529046812bd87b178b0`.
