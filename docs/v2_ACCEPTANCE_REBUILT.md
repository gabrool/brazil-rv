# Fresh 16-book acceptance on the repaired store

All 16 evaluations pass the registered development-grade engineering acceptance: 15 controls across F1-F3 and the five-seed, five-head GBDT on F1. The 75 legacy per-horizon control ICs reproduce the immutable reference exactly. The independent native-fast audit passes with zero reconstruction error. All registered execution bounds pass, and D1-D5 counts are zero.

The book uses the adopted in-sample R3.1 policy: theta 1, D3/D5/D10, equal notional, buffer 9 per quintile. Economic beta is unchanged in value and bound to the repaired store. No support threshold or execution bound was relaxed.

F3 inverse-volatility and the reversal/momentum blend retain unresolved economics under the terminal-settlement rule. Their terminal-settlement notional is 19.93% and 16.55% of NAV, respectively. They are excluded from supported economics comparisons under the frozen rule; passing engineering acceptance does not relabel those outcomes as resolved.

| Evaluation | Mean equity gross | Mean stale/unresolved inventory | Terminal settlement/NAV | Economics unresolved | D1-D5 total |
| --- | --- | --- | --- | --- | --- |
| baseline:F1:inverse_volatility_20 | 1.9643 | 0.587% | 8.102% | False | 0 |
| baseline:F1:momentum_12_1 | 1.8561 | 0.466% | 6.417% | False | 0 |
| baseline:F1:reversal_21 | 1.9781 | 0.511% | 6.995% | False | 0 |
| baseline:F1:reversal_5 | 1.9433 | 0.240% | 3.298% | False | 0 |
| baseline:F1:reversal_5_momentum_12_1_blend | 1.9788 | 0.478% | 6.594% | False | 0 |
| baseline:F2:inverse_volatility_20 | 1.9596 | 0.220% | 3.001% | False | 0 |
| baseline:F2:momentum_12_1 | 1.8496 | 0.000% | 0.000% | False | 0 |
| baseline:F2:reversal_21 | 1.9722 | 0.117% | 1.628% | False | 0 |
| baseline:F2:reversal_5 | 1.9615 | 0.313% | 4.287% | False | 0 |
| baseline:F2:reversal_5_momentum_12_1_blend | 1.9662 | 0.000% | 0.000% | False | 0 |
| baseline:F3:inverse_volatility_20 | 2.0169 | 1.413% | 19.932% | True | 0 |
| baseline:F3:momentum_12_1 | 1.9836 | 0.405% | 5.732% | False | 0 |
| baseline:F3:reversal_21 | 1.9895 | 0.675% | 9.550% | False | 0 |
| baseline:F3:reversal_5 | 1.9810 | 0.902% | 12.730% | False | 0 |
| baseline:F3:reversal_5_momentum_12_1_blend | 1.9816 | 1.178% | 16.549% | True | 0 |
| gbdt:F1:ensemble | 1.9144 | 0.138% | 1.899% | False | 0 |

Root: `D:\quant-data\b3\processed\model_runs\v2_acceptance_rebuilt_615ae41_20260909T104639Z`

Manifest SHA-256: `c5f7058d1fb77859313d0875a27c8bdd672cbb39635441561d9f02cc52504a15`

Inventory SHA-256: `f4b6194673fad27486c3be8dc8096166dd1031b6450cc6f5d92ad86c458e4c9c` (125 inventory entries, plus inventory self-files).

[Full acceptance evidence](v2_acceptance_rebuilt_evidence.json) includes beta diagnostics, occupancy, all bound values and reasons, source identities and the date contract. This is development-grade evidence under inferred corporate actions and a reconstructed calendar. The corrected reader consumes only the through-2024 store; historical integrity-only later-row scans are disclosed in [the reader correction](v2_store_comparison_stop_evidence.json).
