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
