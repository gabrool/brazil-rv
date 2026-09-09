# Rev4f Round-2 stop: decoded history-age provenance

The approved six-field lending amendment is implemented at `a25ae45`. Round 1
completed and is sealed; [its complete before/after report](v2_PASS5_REBASELINE.md)
covers all 18 panels. Round 2 stopped **before its first new ledger evaluation**
at `aggregates/arm_A/F1`. The stopped root contains copied source artifacts,
not a completed Round-2 result. No sweep, rebuild, fit or paid instance ran.

## Exact failure and investigation

`input_hashes.history_age_sessions` differs between the sealed GH200 Round-2
evaluations and host reconstruction. It is not in `recomputed_input_hashes`.
Pass-5 section 6 therefore requires the stop. The exception also prints the
already authorized provenance differences; those are not additional blockers.

The history-age decoding formula is unchanged from source commit `2b40b24`:

```python
np.where(history_valid,
         np.expm1(np.clip(transformed_history_age, 0.0, 1.0) * np.log1p(252.0)),
         np.nan)
```

Both runs bind the same canonical store manifest and slow-feature arrays. The
host reproduces the sealed host-run Round-1 age hashes. Different floating-point
transcendental results across platforms are the working explanation, not a
proven element-level diagnosis: the original decoded age arrays were not saved.
The investigation did not change or round this formula.

A bounded input-only investigation reconstructed all 12 Round-2 panels:

- `history_age_sessions` is the only unregistered input-hash difference in all 12.
- Every other protected input hash matches: scores, score masks, targets, target
  masks, populations, raw closes, corporate-action terms, funding and borrow inputs.
- **All quality-stratification fields are bit-identical in all 12 panels**, including
  the history-age group populations, support counts and ICs. No quality diagnostic
  needs an exemption on the evidence observed so far.
- No score or model was recomputed, and no new ledger evaluation was performed.
  Full replay identity for the remaining report is still a required future check.

The same fold-specific hashes apply to A, B, GBDT and ensemble:

| Fold | Sealed decoded-age SHA-256 | Host decoded-age SHA-256 |
| --- | --- | --- |
| F1 | `1fd83a52eaae80a03c30716edac17cdc907f7cb79efdaae5c9a06d9563356411` | `ee3db7b9ea2188912418b96f406107ec31fb736862642c6d70dda85a955878b1` |
| F2 | `b39666f41506291758daa83ab17f4fd481220f6c8dd8c60d354ab04b19ac514e` | `85e549f58305093cd2fb9c9dbc2cf5d3e3b631eb670b248d3fa634f624d5deab` |
| F3 | `c2540a7a12cc2ccf489fea75ad70bc6a92e2be7c8c4b2f9424e9d8b9fc05b53f` | `4bfbbf10901dd83887f74590ac6a311ac27a5780fe27c24ddb5a25dfd5cc85dd` |

## Amendment subsequently approved

Add exactly **`history_age_sessions`** to the rev4f `recomputed_input_hashes` list.
It is a decoded diagnostic provenance array, not a model input or outcome. Keep
the source-store identity, formula and every quality/score-derived field unchanged.
Do not add anything to `recomputed_diagnostics`, round the decoded ages, or weaken
the protected-field comparison. Report the old/new hash explicitly.

The user explicitly approved this exact amendment in
`v2_pass5_round2_stop_approval.md`. Commit the registration/code-list amendment, freeze a fresh
Round-2 root, and rerun its 12 sealed panels. The completed Round-1 root remains
valid under its own frozen registration and does not need another replay. Preserve
this stopped root unchanged. Any further protected-field difference still stops.

This was a separate amendment from the six lending fields already approved; it
was not inferred from that earlier approval. The user-supplied pass-5 section 6
required the stop. The explicit new approval now authorizes continuation.
Nearest-integer history-age canonicalisation is registered for Round 1' on the
rebuilt store only; it is not applied to this replay.

## Immutable evidence

Root: `D:\quant-data\b3\processed\model_runs\v2_round2_rev4f_a25ae45_20260908T233834Z`

| Artifact | SHA-256 |
| --- | --- |
| Frozen design | `c90cf848b93aa168951eb3e036f69ca9327901397bf970958156a69f1c301798` |
| Failure record | `38767a46136669d6f7fa7b892842ec36469f6dab14ffcb3bb2d6ee7ccfdc5b30` |
| Input investigation | `9b5ba90504efdaf68911937e75e1f8774938ae6bde59ecb1eb1aa1ede74ca6b6` |
| Access audit | `87f7474f81c55f8d2733d7bae4b26021ce527ac2a8dfdf0768fcfdf6dde421a2` |
| Inventory | `9d90d85129368f571eb6daf43185eb35aad945e90e1e46f63df36248898d7ef6` |

[Full per-panel investigation and proposed amendment](v2_pass5_round2_stop_evidence.json).
The root is sealed with `research_claim=false`; official-validation/test access is
false/false. No new Round-2 result or revised Round-2 designation is claimed.
