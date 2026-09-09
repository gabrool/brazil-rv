# R3.1 baseline stop: missing-score carry changes the current policy

The first R3.1 baseline panel stopped before any alternative grid cell. The
supplied registration calls theta=1 the current policy, but its five-session
missing-score carry changes risk-trim priorities for temporarily inactive
holdings. No policy was selected. All source model scores, targets, masks,
quality fields and reports remain unchanged. The user explicitly chose to preserve current theta=1 behavior and apply carry
only at theta<1. A fresh freeze will use that clarified rule.

The implementation and pre-result registration are committed as `8e592df`.
Validation: 139 existing targeted tests plus 11 policy/attribution tests pass.
The new regression reproduces the trim-priority difference on a small fixture.

## Completed baseline evidence

Arm B / F1, theta=1, D1/D2/D3/D5, equal sizing, buffer 6:

| Scenario | Sealed net excess, bps/day | With carry | Changed days |
|---|---:|---:|---:|
| borrow_balance | 0.016444582 | 0.189576706 | 22 |
| borrow_open | -0.707541834 | -0.707541834 | 0 |
| borrow_strict | 1.042631057 | 0.992589374 | 22 |
| comparator_sterile_proceeds | -4.733967493 | -4.643273137 | 22 |

All D1-D5 counters are zero in these four scenarios. The headline gross,
stale-inventory and occupancy bounds pass. The failure is the baseline
identity check, not a protected model-input difference or a hard book bound.

A baseline-only isolation test found:

| Case | Exact sealed daily table | Net excess, bps/day | Orders |
|---|---|---:|---:|
| registered_carry | False | 0.189576706 | 2695 |
| rerank_without_carry | True | 0.016444582 | 2697 |
| unchanged_current | True | 0.016444582 | 2697 |

The first difference is 2023-11-28. The existing book treats the missing
current score of held `BRPRNRACNOR4` as lowest conviction and exits its whole
position during the risk trim. The carry rule gives it a positive retained
score, keeps it, fully trims `BRYDUQACNOR3` instead of only 26.45%, and trims
16.99% of `BRVLIDACNOR5`. Hedge intentions and subsequent inventory then differ.
Removing carry, while retaining daily reranking, reproduces the entire sealed
headline daily table exactly. No smoothing/horizon/sizing alternative was run.

## Clarification approved

The user approved: theta=1 preserves current missing-score behavior; five-session
carry applies only when theta<1. This keeps the registered 18+2 grid and its
current-policy cell meaningful. The concrete change is to use a zero-session
carry at theta=1, retain five sessions for theta=0.5/0.25, and amend the
registration before freezing a fresh root. The sealed stopped root is retained.

Alternative: retain carry at every theta, describe it as an additional common
execution change, and compare all 18+2 cells to a separately retained sealed
current-policy reference. The nominal theta=1 cell would no longer be called
the unchanged policy. This also requires clarifying the registration.

The approved interpretation is implemented and registered before the fresh freeze. Intraday coverage, the store
rebuild, acceptance and Round 1' have not run; they follow the completed sweep.
No 2025/2026 payload or paid instance was accessed.

## Sealed roots and hashes

### stopped_sweep

`D:\quant-data\b3\processed\model_runs\v2_execution_sweep_8e592df_20260909T002609Z`

Exact 8-file inventory verified.

- access_audit.json: `5c8b62a61b7d81aaf0b81c0f85fbabc7e3dca8b987b045ad4b307d66e9ed2acd`
- artifact_inventory.json: `09d8d32a00fae7ec2a9e69ff43850e59d6a2d0b9cd30d1c08c615dd709bf9db5`
- failure.json: `00725d1315208cdb54ad9250b9baf41c437cf68cc5b435f49b5b606c696661aa`
- frozen_design.json: `2021cfb54afdb33446f4b5b6ce9c2e436458b7c3ceb5e32eef77f79ee321ee94`

### baseline_diagnosis

`D:\quant-data\b3\processed\model_runs\v2_execution_baseline_diagnosis_8e592df_20260909`

Exact 4-file inventory verified.

- access_audit.json: `6a12280b6ac8dc63ba2ea931f7963aedd486034fddf9d3abd2ab65fae22c0bba`
- artifact_inventory.json: `60305ec4f86d02ddccaf5e1b70250c79f2b077a4bcd1ac2f9a33fa355446d5a9`
- diagnosis.json: `de28427e733b19a499581470496b60463f479064ef508a281f6647f86c6e7262`

### order_trace

`D:\quant-data\b3\processed\model_runs\v2_execution_baseline_trace_8e592df_20260909`

Exact 4-file inventory verified.

- access_audit.json: `63dd8cc2743176e0be440a4c5921026c3cf0525005ebe39ed9d8077751f206ea`
- artifact_inventory.json: `84a6732f6ac76e37c37ee1f2dc73cdb998ffe0b1c45ac11d8648d82af6c5bea7`
- diagnosis.json: `79393c87a014edc9293bce14bd53e562f067bd391ca3401f439c6b3acfe2859b`

The initial diagnosis used an incorrect filter label for its order trace;
its empty trace lists have no evidentiary meaning. The separately sealed
`order_trace` root uses the actual `risk_exit` purpose and contains the full
orders. Both roots retain the same exact baseline-isolation results.

[Machine-readable evidence](v2_r31_baseline_stop_evidence.json) includes the
four summaries, exact order trace, source identity and inventory hashes. The
original-trade liquidity attribution reconciles; its descriptive contribution
is retained in the stopped panel and does not inform selection.
