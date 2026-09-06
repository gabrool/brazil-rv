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

## Canonical multi-day refactor closeout (2026-09-06)

The canonical refactor implementation is frozen at
`f0cf568303715e8783e539a683568535e2232c7f`. Phases A–F passed their
engineering and deterministic-fixture acceptance: one 15:45 decision row,
validated raw observations and masks, contractual action/identity interfaces,
typed and revision-safe features, bounded native consumers, intended orders
before fills, signed-share/cash/claim accounting, unified common-population
evaluation, and explicit rejection of stale semantic artifacts. This is an
incompatible current-model contract; earlier stores and Round-1/Round-2
results remain immutable historical evidence and cannot be resumed.

Phase G real-data acceptance is **unsupported**, not passed or failed. Its
present hard blockers are the missing verified historical B3 session schedule
and complete contractual action/conversion/terminal-settlement evidence. The
available raw point-in-time inputs also cannot attest every proposed options,
fundamentals, or rebalance field. No substitute was inferred. Verified auction
marks/capacity and executable borrow would strengthen an economics claim, but
are not universal blockers: the plan permits explicitly labelled
`close_proxy`, conditional-borrow, or long-only screening. The expanded
immutable-source action audit is
`D:\quant-data\b3\processed\v2_refactor_action_audit_c316330_20260906T013921Z`:
audit SHA-256
`d717899d7174cfb4645d06db63957c88ab3fea608ad3f33ebf49aac96c340d43`,
detailed-breakdown SHA-256
`98bd1b9133c30cc30ca1337c8219a7692c903f3a183a2e714b82e0576692e871`,
and source-store manifest SHA-256
`6a7e13195c6cde92fbdc756a585e4cb65d73998faa94e237595c7be7cdfb6919`.
At audit commit `c316330`, both the legacy classifier and strict fallback
failed the declared coverage/precision gates; consequently
`canonical_price_ratio_adjustment_authorized=false` and uncovered action
intervals remain unresolved. This supersedes the earlier provisional
fix-pass statement that DISMES-only could become accepted economics.

Phase H and the user's experimental Section D were **not run / not
authorized**. There was no revised preregistration, new model score,
official-validation access, permanently spent test access, or deployment
change. Final local verification recorded Ruff pass, Python compile pass, 810
passing research tests, five passing dedicated T23 resource/memory tests, and
one passing authentic synthetic T24 raw-to-store-to-fit/score/ledger/report
test. The T23 tests exercise populated production axes and actual archive row
counts; T24 independently exercises semantic integration from deterministic
raw fixtures. No paid Lambda instance was launched for this refactor.

## Pass-4 development-grade acceptance stop (2026-09-06)

After the user explicitly authorized bypassing only the 10-GiB admission
check, the canonical source-tier store rebuilt at commit
`12e6ae08eb67014c40160809d47212453a4f3f90`. The immutable root is
`D:\quant-data\b3\processed\v2_daily_store_12e6ae0_20260906T173300Z` and its
manifest SHA-256 is
`2f537944ba857675265031da4f2f4327fdffd6d452b76d4de818d752f1272764`.
Peak RSS was 7.802536 GiB; the unchanged 8-GiB measured-build invariant,
calendar/identity audits, internal and target survivorship gates, external
contemporaneity/composition checks, and access flags all passed. The matching
raw-M1 native-fast audit also passed with exact equality for all seven channels
and masks; audit SHA-256 is
`f1c23ef1884b5bcf3f0224ee4db22533e0bfc6e67d62cae0b0cf701f9ebd7174`.

The full local classical acceptance completed all 15 naive baseline cells and
the 25-member F1 GBDT ensemble, then returned `unsupported` at the first
unchanged research gate. Naive pooled primary IC magnitudes were below 0.10 and
reversal-5 had the required negative sign. Mean absolute terminal unresolved
inventory was `0.0179667` NAV, below 0.02. Deployed gross, however, ranged only
from `0` to `0.947743` NAV across all 16 books, so every book missed the
registered 1.8--2.2 band around gross target 2; the F1 GBDT value was
`0.477454`.

The immutable acceptance root is
`D:\quant-data\b3\processed\model_runs\v2_development_acceptance_12e6ae0_20260906T183511Z`.
Its pipeline-manifest and log-inclusive inventory SHA-256 values are
`34ac98f8646934cda82290826a1ad0fd12921393876d726a829cb53825669c38`
and `8b011067471067c3ceafb2b78b2153b6331b18a66db4bd89ba6490da04207445`.
Official-validation and test access were false, transfer chronology was clean,
and no deployment changed. The frozen stop rule therefore prevented Round 1,
Round 2, a merge to `main`, or a paid-instance launch. No scored candidate was
retried and the gross gate was not relaxed after seeing the result.

## Pass-4b ledger-policy repair and second acceptance stop (2026-09-06)

Pass 4b repaired the evaluator-side entry starvation without rebuilding the
store or changing targets, masks, folds, scores, costs, the 5% name cap, or any
registered acceptance bound. The stateful ledger now accounts for occupied
slots while exits are pending, refills the long and short sides independently,
skips blocked candidates instead of abandoning the day's queue, trims only the
excess exposure on risk breaches, and uses the registered 0.20 absolute planned
net cap. The code-bearing commits are `2bfa986f4404ae7dd724fdcaa3f98e73d6c00686`
and the reporting-boundary repair
`48d55622f094535039ee7ab613ecbeb51b0e8b2e`.

The mandated replay reused the exact 16 sealed score arrays from the prior
acceptance root; it performed no model fit or score recomputation. All IC,
persistence, coverage, and other non-ledger report fields were bit-identical.
The immutable replay root is
`D:\quant-data\b3\processed\model_runs\v2_development_acceptance_ledger_replay_48d5562_20260906T192609Z`.
Its pipeline manifest, gate-diagnostic report, and log-inclusive artifact
inventory SHA-256 values are
`b62d53d31986d304fabedd6fbfde6dff32167198e634f5b9fb2dad9aab49bdeb`,
`1bdeab6d58da51fec251fec7643f974cb7a617ea676c68deaf806cd4868c8953`,
and `dfa783d6500cc72c0cd124ea75cf435f935494a259f9f9a43075265ff7898bcd`.

The policy repair materially increased utilization, but the unchanged gate
still rejected 12 of 16 books. Only F1 reversal-21, F2 reversal-21, F3 inverse
volatility, and F3 reversal-21 entered the 1.8--2.2 mean-gross band, at
`1.855342`, `1.810454`, `1.901746`, and `1.848958`. The F1 GBDT reached
`1.773067` and still failed. All momentum and momentum/reversal-blend books
remained at zero because their exact primary-head score masks are too small:
F1 contains at most 34 eligible names per day and F3 at most 24. At a binding
5% name cap, even holding every eligible name cannot reach 1.8 gross. Mean
absolute terminal unresolved inventory was `0.0361789` NAV, also above its
unchanged 0.02 ceiling.

The access audit passed with official-validation and test access both false,
transfer chronology clean, `research_claim=false`, and no deployment change.
Per the explicit stop rule, the branch was not merged to `main`, Round 1 and
Round 2 were not started, and no paid instance was launched. A continuation
would require a new, pre-result contract resolving the incompatibility between
the small exact score populations, the 5% name cap, and the 1.8 gross floor;
none of those research rules was changed after observing this replay.
Ruff and Python compilation passed on the final branch state, and the complete
research suite passed all 824 tests in 380.76 seconds.
