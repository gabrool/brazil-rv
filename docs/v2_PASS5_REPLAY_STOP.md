# Rev4f first replay: registered diagnostic stop

The first replay stopped on `reversal_5 / F1`. This is **not a completed Round-1
rebaseline** and no Round-2 replay or downstream sweep/build/fit was started.
Implementation: `1ad391ee816e06883b07e91fbf6091128bd1e947` (already pushed).

The slow-valid-masked Yang-Zhang feature also defines the high-volatility stratum
inside lending coverage. The six fields below were missing from the declared
recomputation list. The verifier rejected these differences as required by pass 5
section 6. The proposed amendment adds exactly these paths under
`diagnostics.lending_coverage`; the user subsequently approved this exact amendment, now applied before a fresh replay.

| Field | Rev4e | Stopped rev4f attempt |
| --- | ---: | ---: |
| `high_volatility_quartile_name_days` | 6192 | 6181 |
| `high_volatility_quartile_rate_imputed_fraction` | 0.429909560724 | 0.429865717521 |
| `high_volatility_quartile_rate_observed_prior_60_fraction` | 0.519864341085 | 0.520142371785 |
| `high_volatility_quartile_rate_placeholder_fraction` | 0.0502260981912 | 0.0499919106941 |
| `high_volatility_quartile_shortable_fraction_by_cell.borrow_balance` | 0.570897932817 | 0.570943213072 |
| `high_volatility_quartile_shortable_fraction_by_cell.borrow_strict` | 0.545381136951 | 0.54538100631 |

Every other field outside the already registered exemptions is bit-identical.
This includes scores, score masks, targets, target masks, populations, IC,
persistence and spreads. Raw lending rates and availability inputs are unchanged.
For this one evaluated book, D1-D5 counts are all zero, mean gross is 1.94187 NAV,
and mean unresolved/stale inventory is 0.19879% NAV. These satisfy the corresponding
hard bounds. Same-day actions affected held/pending names on 56 sessions.

The complete first-cell before/after diagnostics are retained in the stopped root.
Its sealed inventory also includes the original copied panels that were never
replayed; the absent `round1_result.json` and the failure record identify the root
as incomplete. Do not use this root as a canonical result or resume it in place.

Root: `D:\quant-data\b3\processed\model_runs\v2_round1_rev4f_1ad391e_20260908T232905Z`

| Artifact | SHA-256 |
| --- | --- |
| Frozen design | `976b2a2c2e21fb8b7767e981b5bdaf665bb6593e9d08b5316615a44594607b28` |
| Source evaluation | `b2f7772073f736115fb9616b480ebfe900503a546602c49747be1d77aba8af0f` |
| Attempted evaluation | `0b13e3194919f846ad32dda0936b6bb2efd97ed5bae33277d58430bfb2955481` |
| Failure record | `acb036d1ae284c216bd42a6be50bea0deaffa2d750ab0bc91b8a3b1a54ce1756` |
| Access audit | `d5acdac4555d5c14fe75c4d9891c194d9aa8dd31b845e3246581358630cdbf7c` |
| Inventory | `25e4994bb199153e94febc6c335d1f42cb947fff6ac2387b2b6257329e34f832` |

[Machine-readable evidence and exact proposed paths](v2_pass5_replay_stop_evidence.json).
Official-validation/test access remains false/false. No instance was launched.
The copied sealed Round-2 root remains verified and unchanged. Previous cleanup,
causal-action repairs and both replay commands are committed on GitHub.
