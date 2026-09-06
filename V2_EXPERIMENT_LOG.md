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
Two provider inventory reads at 2026-09-06T19:40:23Z and 19:40:25Z each
returned zero instances. There was therefore no paid instance to terminate.

## Pass-4c pre-rebuild diagnostic stop (2026-09-06)

The mandatory no-rebuild `momentum_12_1` waterfall contradicted pass 4c's
causal premise. On the sealed `12e6ae0` store, slow-feature momentum covers
98.97%/98.65%/98.22% of active F1/F2/F3 name-days, and a restart-after-missing
scan removes zero additional evaluation name-days in all three folds. The
required missing-print-restart dominance condition therefore failed.

The audit found the source of the 15--46-name naive baseline panels instead:
the baseline interval guard currently excludes every resolved corporate
action. It retains only 14.35%/14.43%/11.15% of active name-days, whereas the
specified unresolved-action-only guard would retain
99.09%/98.68%/98.37%. The score-free diagnostic root is
`D:\quant-data\b3\processed\model_runs\v2_pass4c_prechange_waterfall_19a9dbe_20260906T195326Z`;
manifest SHA-256 is
`8343c4c98619dae5b3cbbd53d75ec28834f9435ace70c6570d202c8d5f34d4de`.
Official-validation and test access remained false, and no score was written.

Per pass 4c section 1.1, work stopped before the proposed recurrence rebuild.
No ledger or acceptance change, merge, experiment, paid-instance launch, or
deployment change occurred.

## Pass-4d pre-result interval and ledger repair (2026-09-06)

Pass 4d accepted the pass-4c diagnosis and withdrew wealth-chain bridging.
The implementation now gives slow wealth features and naive baselines one
shared validity function: both exact endpoints must exist, the interval may
not cross a raw-series restart, and only an unresolved corporate action breaks
the shareholder-wealth return. Resolved dividends, splits, and conversions are
already represented by the wealth index and therefore do not mask it. The next
store records `wealth_chain_restarts_active_name_days` by calendar year; no
restart is bridged.

The complete old-mask consumer inventory was: `build_store.py` constructed and
stored `decision_action_boundary_mask`, `baselines.py` used it as its interval
guard, and `validate_pipeline.py` required its presence. No other model,
target, or ledger calculation consumed it. The two old intraday boundary masks
were constructed in `build_store.py`, supplied to the intraday validity-clock
helpers, stored for diagnostics, and inspected by store tests. The replacement
guards are deliberately consumer-specific:

- wealth features and baselines use unresolved actions plus exact-endpoint and
  restart validity;
- raw-price cross-session fields `overnight_return`,
  `overnight_return_sum_5`, `overnight_return_sum_20`,
  `overnight_minus_intraday`, `overnight_minus_intraday_mean_20`,
  `corwin_schultz_spread_20`, and all three `_lag1` full-session summaries use
  unresolved actions or unit changes, including successor conversions;
- same-session scale-free fields have no action guard; and
- targets and ledger action handling are unchanged.

The immutable current-store coverage audit is
`D:\quant-data\b3\processed\model_runs\v2_pass4d_intraday_guard_audit_5e500a4_20260906T200144Z`.
Its manifest, diagnostic JSON, and parquet table SHA-256 values are
`ffb537f0cf05515a355e42f9cb6833baae46f208aadbda5f005b89dec87564da`,
`0822c3ac5629d893edcb5dade4d6a37a9e948d0e71e69045b53a2c1ad42fab9b`,
and `3ed00f7712a41f9479c1c694fc823ee37bde1fd4f3de27cf1562ab1dd851c389`.
Among active, fast-present evaluation name-days the corrected guard recovers
as much as 24.471996 percentage points, and 24 feature/fold cells exceed the
registered 10-point trigger. A clean-commit store rebuild is therefore
mandatory; this decision was recorded before the new acceptance result.

The prior sealed ledger replay's terminal unresolved inventory was also
classified before changing its gate. Nineteen of 22 terminal positions had no
terminal print and they represented 88.8999% of terminal unresolved notional;
all 22 also had unresolved actions and prior pending exits. Terminal non-print
therefore dominates. At `2026-09-06T20:11:20Z`, before rerunning, the
engineering registration was updated exactly as pass 4d permits: mean daily
unresolved-or-stale marked notional must be below 2% of NAV in every
evaluation, while terminal count/notional and a non-exclusive reason breakdown
remain reported diagnostics.

The ledger now lets a fully submitted same-close exit release that side's slot
for a same-auction entry, while an older pending exit continues to occupy it.
If the exit fails and the replacement fills, the existing next-day risk trim
handles the temporary over-allocation. Each side uses
`K_eff = min(K, floor(N/2))` at the unchanged configured slot notional, and
reports mean daily exits per side and same-close replacements. Gross target,
gross/name/net caps, targets, folds, seeds, roster, costs, and protected-data
rules remain unchanged.

## Pass-4d rebuilt store and binding acceptance stop (2026-09-06)

The score-bearing implementation was frozen at clean commit
`8021e42ae658d3eb7bba19e39b58e2f9c8d65331`. Ruff and Python compilation
passed, and the complete research suite passed all 830 tests in 361.31
seconds before the fresh acceptance scores were produced.

The mandatory rebuild completed at
`D:\quant-data\b3\processed\v2_daily_store_8021e42_20260906T202315Z`.
Its V3 manifest SHA-256 is
`deb9ca8449c5b9a83bf25ac19218069c006e183361b6f6ba836717ade63b491b`.
Peak RSS was 7.138950 GiB, below the unchanged measured-build limit of 8 GiB.
The identity/calendar audits, internal-feature and target survivorship gates,
external contemporaneity and name-clustered composition checks, and protected
access audit all passed. The store records raw wealth-chain restarts without
bridging: 3 active name-days in 2010, 4 in 2011, 1 in 2016, 1 in 2022, and 0
in every other year. None falls in the active F1/F2/F3 evaluation populations.
Naive baseline coverage is now 98.22%--99.94% across all fold/book cells,
confirming that the resolved-action over-mask was removed without inventing
history. The independent native-fast parity audit is sealed at
`D:\quant-data\b3\processed\model_runs\v2_native_fast_audit_8021e42_20260906T184200Z`;
its audit SHA-256 is
`98d2e5346a0e2555b77fdb0cb034c63e9d8fa7860eb30f6e749680883ba29218`,
with exact equality for the registered 20-name by 20-session sample.

The full acceptance then ran from scratch at
`D:\quant-data\b3\processed\model_runs\v2_development_acceptance_8021e42_20260906T184700Z`.
The pipeline-manifest and log-inclusive inventory SHA-256 values are
`f7ce5a47a5be4bc8d8f4e11b5adde83a97c6e334b87038068f0f56a7ba7beaa6`
and `2caf60d4d3202fa3228b295aa3245d04bf6bf92dc197a00264dbdab9d3672bf2`.
The access-audit SHA-256 is
`88caeaec04ab0e3434781299fd2dee0f34dce6257d608cc90dd6a99300834cb3`;
official-validation and test access are false, transfer chronology is clean,
`research_claim=false`, and no deployment changed.

All naive signal checks passed: pooled primary IC was `0.03985972` for
momentum, `-0.01816006` for reversal-21, `-0.00703711` for reversal-5, and
`0.02366325` for their registered blend. The unchanged 1.8--2.2 gross gate
passed 13 of 16 books. The registered per-evaluation unresolved/stale mean
below 0.02 passed 7 of 16. Exact ledger diagnostics are:

| Book | Mean gross | Mean unresolved/stale | Terminal unresolved | Mean exits/side/day | Same-close replacements | Binding failure |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| F1 inverse-volatility | 1.788554 | 0.035150 | 0.068998 | 0.596774 | 68 | gross, unresolved |
| F1 momentum | 1.798077 | 0.019070 | 0.058789 | 0.572581 | 68 | gross |
| F1 reversal-21 | 1.968796 | 0.010528 | 0.028309 | 2.326613 | 475 | none |
| F1 reversal-5 | 1.927972 | 0.013646 | 0.057694 | 6.532258 | 1,370 | none |
| F1 blend | 1.982435 | 0.014318 | 0.061482 | 4.250000 | 941 | none |
| F2 inverse-volatility | 1.739684 | 0.021198 | 0.029680 | 0.524194 | 57 | gross, unresolved |
| F2 momentum | 1.828920 | 0.000000 | 0.000000 | 0.584677 | 74 | none |
| F2 reversal-21 | 1.934637 | 0.012840 | 0.015506 | 2.197581 | 433 | none |
| F2 reversal-5 | 1.915457 | 0.030632 | 0.050162 | 6.366935 | 1,344 | unresolved |
| F2 blend | 1.944545 | 0.018658 | 0.023014 | 3.862903 | 839 | none |
| F3 inverse-volatility | 1.889690 | 0.038970 | 0.078370 | 0.913386 | 90 | unresolved |
| F3 momentum | 1.913107 | 0.107829 | 0.125821 | 0.724409 | 62 | unresolved |
| F3 reversal-21 | 1.936944 | 0.040265 | 0.098445 | 2.271654 | 437 | unresolved |
| F3 reversal-5 | 1.940560 | 0.055983 | 0.094767 | 6.413386 | 1,383 | unresolved |
| F3 blend | 1.965007 | 0.109153 | 0.161402 | 3.881890 | 888 | unresolved |
| F1 GBDT ensemble | 1.856018 | 0.024496 | 0.032407 | 1.145161 | 217 | unresolved |

Because at least one binding gate failed, engineering acceptance is
`unsupported`. The registered stop prevented a merge to `main`, Round 1,
Round 2, or any result-changing retry. No acceptance bound, name/gross/net
cap, score population, or execution rule was changed after observing the
result. No paid Lambda instance was used for this pass.
Two provider inventory reads at 2026-09-06T21:55:41Z each returned zero
instances, confirming there was nothing to terminate and no adjacent instance
was touched.

## Pass-4e per-session action state and final acceptance stop (2026-09-06)

The mandatory pre-change diagnostic used the sealed `8021e42` store and
acceptance without rebuilding or changing the ledger. Across the 16 books it
found 2,502 held unresolved/stale name-days, all carrying the old latched
unresolved-action flag. Eighty-four name-days (3.3573%) traced to a flag origin
followed by a later print. No held cell had a false retrospective action mask
on an observed session; inferred-term, null-DISMES, and observed-OHLC-invalid
causes were all absent. The evaluator had received the retrospective mask, not
the decision-known feature mask. Coverage semantics therefore did not qualify
for a rebuild under the frozen conditional rule.

The canonical score-free diagnostic root is
`D:\quant-data\b3\processed\model_runs\v2_pass4e_unresolved_diagnostic_24abe56_20260906T222757Z`.
Its diagnostic-manifest, log-inclusive artifact-inventory, and access-audit
SHA-256 values are
`2801ea826df263525aec0247c9eeefcdd5975237e73dcab4807d20c68e623310`,
`0f93b5e660c7701cfec392b3a92ff25fdb7eaccd212c270deffa91c0d07ed709`,
and `9e02b999a1aa34e076c6940fcf3e87ed0845b01e62c83b1c455a124cc56d46ff`.
An earlier pre-seal diagnostic directory at timestamp `20260906T222336Z` is
retained as noncanonical evidence; no root was deleted.

Commit `eca09d6851074e79d7ffb7f1d4e9a11c022c3a02` implements the bounded
ledger correction. Action uncertainty is recomputed per session and cannot
latch to a position. It gates new entries only; any observed positive close
can fill ordinary exits, risk reductions, and terminal liquidations. The
evaluator now asserts that accounting receives the store's retrospective
outcome/action arrays. Daily output reports unresolved-claim inventory and
stale-mark inventory separately while retaining their unchanged union gate.
The required tests cover one-session uncertainty followed by an exit, a
dividend receivable followed by an exit, terminal liquidation despite current
uncertainty, and rejection of decision-known accounting alignment. Ruff and
Python compilation passed; all 861 research tests passed in 392.66 seconds.

The final replay hash-verified and reused all 16 sealed score panels. It fitted
and scored no model, and all non-ledger fields were bit-identical. The immutable
root is
`D:\quant-data\b3\processed\model_runs\v2_development_acceptance_pass4e_eca09d6_20260906T225238Z`.
Pipeline-manifest, replay inventory, log-inclusive artifact-inventory, and
access-audit SHA-256 values are
`846d278d1a9bf9a038c7687406e570d77c968ccb7f7c5b13cfeba616a1c1de78`,
`e90e2dbb6e83a078446d3403ef59fc1c0e411640afd940d99e73afdb23e164e5`,
`49d6b2e0d7bbcbbdd4402b676a9e5891c81004f325928d440cd985c28cc1b534`,
and `4edf9ca35b08e4c461d7bac77cb703ee2d8a308a73b013ae7e2ad7fefef5d3f9`.

The predicted acceptance recovery did not occur. F1 momentum entered the
gross band at `1.806721`, but F1/F2 inverse-volatility remained below it at
`1.786278` and `1.741195`. The unresolved/stale gate still passed only 7 of 16
books. Exact mean gross / mean unresolved-stale values were:

| Book | Mean gross | Mean unresolved/stale | Gate failure |
| --- | ---: | ---: | --- |
| F1 inverse-volatility | 1.786278 | 0.035131 | gross, unresolved |
| F1 momentum | 1.806721 | 0.019116 | none |
| F1 reversal-21 | 1.970187 | 0.010521 | none |
| F1 reversal-5 | 1.928655 | 0.013640 | none |
| F1 blend | 1.971609 | 0.014209 | none |
| F2 inverse-volatility | 1.741195 | 0.020533 | gross, unresolved |
| F2 momentum | 1.828920 | 0.000000 | none |
| F2 reversal-21 | 1.933180 | 0.012789 | none |
| F2 reversal-5 | 1.916526 | 0.029282 | unresolved |
| F2 blend | 1.943261 | 0.018603 | none |
| F3 inverse-volatility | 1.888997 | 0.039003 | unresolved |
| F3 momentum | 1.848432 | 0.104513 | unresolved |
| F3 reversal-21 | 1.946001 | 0.040237 | unresolved |
| F3 reversal-5 | 1.939774 | 0.056152 | unresolved |
| F3 blend | 1.967017 | 0.109910 | unresolved |
| F1 GBDT ensemble | 1.856964 | 0.024467 | unresolved |

For every book the separately reported unresolved-claim and stale-mark means
were equal to the union mean: the current inferred-tier false action cells on
held inventory coincide with no-print sessions. The latch was real, but after
removing it the binding exposure is still current stale/no-print inventory,
not an exit blocked on a later observed print. The two low-turnover gross
failures also retain stale inventory, so the contract's special no-frozen-
inventory slot-occupancy branch does not apply; their mean pending-exit ages
were 32.52 and 41.47 sessions.

Engineering acceptance remains `unsupported`. The frozen stop rule prevented
the conditional fast-forward to `main`, Round 1, Round 2, or any result-changing
retry. Momentum's continuity-only pooled IC is recorded as `0.03985972` versus
`0.056` on the former 34-name liquid subset; neither is a research result.
Official-validation and test access remained false, transfer chronology was
clean, `research_claim=false`, and no deployment changed.
Provider inventories at `2026-09-06T22:57:42.9989658Z` and
`2026-09-06T22:57:45.8155995Z` each returned zero instances. There was no paid
instance to terminate and no adjacent instance was touched.
