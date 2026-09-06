# Brazil-RV multi-day (v2) experiment log

This file is the durable research log for the multi-day model. The original
intraday program remains intact in `EXPERIMENT_LOG.md`; references here do not
change its deployment or its sealed access ledger.

## Prior intraday research program

The v1 program built a causal, point-in-time M1 research store and a peer-free
causal TCN with 30/60/120-minute heads. It established odd/even chronological
cross-fit for validation-adaptive Raw Patience-3 checkpoint selection, tested
architecture and external-data families, and adopted a store-v2 input mask.
Experiment 45 became the measured ten-seed store-v2 ensemble at official IC
`0.043718770472`; Experiment 50 did not replace it. Experiment 51 consumed the
single registered held-out test read and measured IC `0.040345936073`. The
subsequent execution studies (Experiments 52–58) retained their own immutable
artifacts and did not alter that v1 deployment. The v1 history, checkpoints,
access records, and conclusions remain governed by `EXPERIMENT_LOG.md`.

## V2 foundation and Section C acceptance

The v2 program changes the research horizon from intraday to multi-day while
preserving point-in-time identity and causality. Its accepted daily store is
`v2_daily_store_98e9386_20260904T165924Z`, manifest SHA-256
`6a7e13195c6cde92fbdc756a585e4cb65d73998faa94e237595c7be7cdfb6919`, with
4,102 sessions, 933 linked security identities, and all feature/target arrays
stored as float32. Section C acceptance completed at commit `23586d4`: both
survivorship gates passed under the registered internal/external-family rules,
the full-F1 CPU and GPU pipeline legs passed, artifact relocation remained
hash-bound, and `official_validation_accessed=false` / `test_accessed=false`.
Those runs were integration evidence, not v2 research claims.

## Round 1 / Round 2 registration — frozen before scoring

The first v2 research registration is
`research/preregistrations/v2_round1_round2.md`. Its exact SHA-256 and the clean
implementation commit are written into the immutable Round-1 root before the
first Round-1 score. It authorizes development folds F1/F2/F3 only; no official
validation or held-out test token exists. Round 1 establishes the naive floor,
GBDT feature ladder, and GBDT data-span preview on CPU. Round 2 compares the
registered neural data-span arms, the selected network, the GBDT parent, and
their fixed equal-weight rank ensemble on one GH200 session. No deployment
change is authorized.

## Round 1 / Round 2 registration voided (2026-09-05)

Round 1/2 registration `v2_round1_round2.md` was voided before any result was
read for a research decision. The interrupted run is engineering evidence
only: no result is admissible or cited as a research claim. External review
confirmed five measurement defects in code at `29ffb88`: (F1) the specified
corporate-action heuristic treated ordinary price jumps as actions and used
future observations, (F2) the specified stateless simulator discarded whole
portfolio days, (F3) the specified target removed the cross-sectional median
after volatility scaling and used same-day volatility, (F4) block-parity
selection and evaluation shared label increments and economic dates, and (F5)
the specified swing simulator did not maintain inventory. F1–F5 arose from
the specification supplied to the implementation; the implementation followed
that contract. No v1 artifact, sealed window, or deployment was changed.

The exact paid GH200 `46ee1e1c16f14d7c8fd737919ed66400` was stopped before
queued work continued, terminated, and confirmed absent in two independent
provider inventories. The partial Round-2 root was superseded before any
trajectory wrote a history, manifest, checkpoint, or score. The completed
Round-1 headline-cell interval support audit was:

| candidate | F1 valid/total | F2 valid/total | F3 valid/total |
|---|---:|---:|---:|
| momentum | 123/123 | 123/123 | 126/126 |
| reversal21 | 2/123 | 6/123 | 9/126 |
| reversal5 | 0/123 | 0/123 | 2/126 |
| blend | 123/123 | 123/123 | 126/126 |
| a_slow | 1/123 | 3/123 | 2/126 |
| b_intraday | 0/123 | 1/123 | 3/126 |
| c_lending | 0/123 | 2/123 | 5/126 |
| d_all_sidecars | 1/123 | 2/123 | 2/126 |
| uniform | 0/123 | 1/123 | 1/126 |
| decay | 1/123 | 0/123 | 2/126 |

There were 61 zero-support readouts serialized as JSON `null`. This quantifies
the F1/F2 missing-outcome defect and is not a performance result.

The preregistered standalone action audit on the immutable accepted store is
`v2_fix_pass_3_action_audit_20260905T230000Z`, audit SHA-256
`29188dbe8a07236e8abd40d7cdc5ed1e6ff5ea2c67c4e8fa5f93e09a17128e73`.
The audit hash-verifies its accepted source manifest and every array/table it
reads; source-manifest SHA-256 is
`6a7e13195c6cde92fbdc756a585e4cb65d73998faa94e237595c7be7cdfb6919`.
DISMES-only split recall/precision were `0.192469/0.464646`; adding the strict
fallback changed them to `0.192469/0.253444`, a recall gain of zero and a
precision loss of `0.211203`. The frozen decision rule therefore rejects the
fallback. The old classifier labelled 114,604 cash events, of which 102,372
(`89.3267%`) had no DISMES change. Old target validity fell from `51.9151%` at
D1 to `30.1509%` at D10. The replacement is consequently DISMES-only, causal,
and fixed before the rebuilt store.

### Fix-pass-3 local rebuild stop (2026-09-05)

The clean implementation was frozen and pushed at
`555ef2e7d33cd289cd0030fad674c129f4490b5f`; Ruff, compile, all 634 research
tests, and the production-axis 8-GiB memory guard passed before the real-data
rebuild. The no-fallback rebuild then exhausted the 16-GiB Windows host while
the `events` sidecar adapter executed
`list(source.sort("isin", "available_date").iter_rows(named=True))` in
`_raw_events_features`. This is an unbounded Python-row materialization of the
real event archive, outside the family-output memmap invariant.

The process stopped before atomic promotion: final root
`v2_daily_store_555ef2e_20260905T232000Z` does not exist, no survivorship gate,
acceptance score, or sealed-window read occurred, and no retry was made. The
unpromoted staging directory
`.v2_daily_store_555ef2e_20260905T232000Z.arrays-9sr0bjjz` is retained for
inspection. Operational-log SHA-256 is
`e0335ffec7049b50dda59072bfa34c49d100830f25bd1a727ab2f0193ffce97c`.
Per the bounded-memory acceptance instruction, work stops here for an explicit
decision before changing the sidecar adapter or attempting another rebuild.
