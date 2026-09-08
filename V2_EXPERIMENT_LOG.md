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

## Pass-4f labelled terminal settlement and acceptance stop (2026-09-07)

The required pre-change diagnosis was sealed before the ledger convention was
implemented. It reconstructed every stale holding in the Pass-4e replay: 37
holding episodes, 14 ISINs, and 2,490 stale name-days across 16 books. Zero
stale name-days came from a position that printed again inside its evaluation
holding window, so the predeclared fill-defect stop did not fire. Twenty-seven
episodes never printed again anywhere in the store; ten printed only outside
the relevant evaluation window. There were no ISIN-succession candidates. The
predeclared deterministic flat/premium heuristic classified zero episodes as
tender-like, so the evidence supports terminal disappearance but not the
stronger proposed tender-offer attribution.

The immutable diagnostic root is
`D:\quant-data\b3\processed\model_runs\v2_pass4f_stale_diagnostic_64c5b76_20260906T232352Z`.
Its diagnostic-manifest, log-inclusive artifact-inventory, and access-audit
SHA-256 values are
`d6522ec1c20cf294fd8914e951c3d51f10bd662f7cf02d8d76b8a9e5f3ac391c`,
`7d28de475e529cbd7aba7dccbac1dc14f1a05941d663a25b178570fe9cf794ee`,
and `ae8d0edfa9ea15adfc3571770459ba4e4e8baa0ea4c66cb02c90f1ea5b51ba55`.

Final pre-score implementation commit
`36a868c6455fd1b58f128b94d03112588b9899c7` adds the explicitly labelled
`last_mark_after_10_sessions` development convention. A held long or short
with ten consecutive no-print sessions settles at its last mark with the
ordinary cost, releases its slot, and can never reopen. A parallel 30% adverse
settlement-price path is reported but is never headline. Later prints are
counted. Cumulative settlement notional above 15% of contemporaneous NAV makes
economics unresolved. The unchanged 2% utilization gate now counts stale held
inventory inside the grace period plus explicit unresolved action exposure;
an unobserved session alone is not an unresolved claim. Ledger identities,
long/short settlement, costs, the scenario, grace reset, permanent no-reopen,
later-print counting, and the 15% consequence have regression coverage.
Repository-wide Ruff and Python compilation passed, and all 869 tests passed
in 377.29 seconds before the accepted replay.

Three score-free operational starts are preserved. Two stopped before creating
a root (an incorrect inventory-hash CLI binding, then a chained replay/store-
build identity check). The third retained empty root at
`D:\quant-data\b3\processed\model_runs\v2_development_acceptance_pass4f_b4715da_20260906T234950Z`
stopped after selecting, but before loading, the first ancestor score panel.
The bounded provenance repairs verify the prior replay and original ancestor
through their manifests, complete inventories and every inventory row, store
hash, code identity, access flags, and score-manifest hash. No score or model
was recomputed during those failures.

The final replay hash-verified and reused all 16 sealed score panels. It
recomputed ledger economics only; every non-ledger field was bit-identical.
The immutable root is
`D:\quant-data\b3\processed\model_runs\v2_development_acceptance_pass4f_36a868c_20260906T235841Z`.
Pipeline-manifest, replay-inventory, log-inclusive artifact-inventory,
acceptance-table, and access-audit SHA-256 values are respectively
`6a137bb00b447a04d52462b28f20e76c1465a981deb2cfa07485387abb064409`,
`2b1cf5a0d84986c42890ccedeb6316d238dfb46238f0540ed626a188fdf7ed8f`,
`1139ef305f94ecb941a50b795fb158203990513615f3262f5ef8a7d187d1d414`,
`9f1f523005a37e624b4b86a332f75846dbeb2701188ce5393707c2279a1f4094`,
and `5ab13dc1c7ce9f3b1b9f565b12abea8669345023d0bab25d6fe62ff274657580`.
Official-validation/test access is false, transfer chronology is clean,
`research_claim=false`, and no deployment changed.

The 2% stale/unresolved bound passed all 16 books, and 15 of 16 passed the
unchanged 1.8--2.2 mean-gross band. Exact per-book incidence is:

| Book | Mean gross | Mean stale/unresolved | Settlements | Settlement notional/NAV | Later prints | Econ unresolved | Gate failure |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| F1 inverse-volatility | 1.800173 | 0.005877 | 3 | 0.069501 | 0 | no | none |
| F1 momentum | 1.804345 | 0.004074 | 2 | 0.057348 | 0 | no | none |
| F1 reversal-21 | 1.973163 | 0.002003 | 1 | 0.028131 | 0 | no | none |
| F1 reversal-5 | 1.931453 | 0.004142 | 2 | 0.057507 | 0 | no | none |
| F1 blend | 1.972213 | 0.004259 | 2 | 0.059517 | 0 | no | none |
| F2 inverse-volatility | 1.755766 | 0.002713 | 2 | 0.036755 | 1 | no | gross |
| F2 momentum | 1.828920 | 0.000000 | 0 | 0.000000 | 0 | no | none |
| F2 reversal-21 | 1.952331 | 0.001106 | 1 | 0.015400 | 0 | no | none |
| F2 reversal-5 | 1.937269 | 0.004297 | 2 | 0.058664 | 1 | no | none |
| F2 blend | 1.958096 | 0.001585 | 1 | 0.021677 | 0 | no | none |
| F3 inverse-volatility | 1.888777 | 0.006381 | 3 | 0.087638 | 0 | no | none |
| F3 momentum | 1.848185 | 0.011440 | 5 | 0.160227 | 0 | yes | none |
| F3 reversal-21 | 1.964110 | 0.006373 | 3 | 0.089967 | 0 | no | none |
| F3 reversal-5 | 1.947440 | 0.006853 | 3 | 0.095979 | 0 | no | none |
| F3 blend | 1.973285 | 0.013590 | 6 | 0.191541 | 0 | yes | none |
| F1 GBDT ensemble | 1.840379 | 0.002448 | 1 | 0.033729 | 0 | no | none |

Engineering acceptance is still `unsupported` solely because F2 inverse-
volatility's mean gross is `1.755766`, below the immutable `1.8` floor. The
predeclared stop therefore prevented fast-forwarding `main`, Round 1, Round 2,
or any result-changing retry. The two F3 books whose settlement incidence
exceeds 15% are also transparently labelled `economics_unresolved`, with the
existing downstream consequence that their economics cannot support a paired
research decision. No paid Lambda instance was used or launched for Pass 4f.
Direct provider inventory reads at `2026-09-07T00:03:31.2895930Z` and
`2026-09-07T00:03:36.9256438Z` each returned zero instances, confirming that
the stale local state file did not represent a billable host and no adjacent
instance was touched.

## Pass-4g gross-occupancy disposition and P0 stop (2026-09-07)

The disposition was registered before diagnostics in commit
`a3f1059a42b0ead8cb25a8a1bca77d923fd91583`. The append-only diagnostic
instrumentation in commit `fb027809c6199990c2673dfc95d550f16bed733c`
records exact gross-shortfall decomposition, per-side occupancy and entry
state, entry/exit defect signatures, and per-name band/holding counts without
changing any order, fill, position, cost, settlement, or accounting result.
Two bounded reporting repairs followed: commit
`6991abb7c4290848bb74362610b26c79f3369cb8` fixed a missing per-name slice in
one band diagnostic, and commit
`fa5c8afb37bb20b0baac70de93928e32a58fc902` converted NumPy scalar counts to
portable JSON integers. Repository-wide Ruff, compilation, and all 844 tests
passed after each repair; the final suite completed in 348.60 seconds.

Both score-free failed roots are retained. Root
`D:\quant-data\b3\processed\model_runs\v2_pass4g_occupancy_diagnostic_fb02780_20260907T011842Z`
stopped on the diagnostic indexing error after one book's sealed headline
identity passed; its corrected failure-record and inventory SHA-256 values are
`f153c6b576224531f9f4fc2be8e62d00cb1da997e3848665f545732703a66e55`
and `31f5b2e297bd181b7f7674c2406662c848fc2b357b8a7eb584fc3a397596150f`.
Root
`D:\quant-data\b3\processed\model_runs\v2_pass4g_occupancy_diagnostic_6991abb_20260907T012721Z`
reproduced all 16 headlines and stopped only while serializing the final
manifest; its failure-record and inventory SHA-256 values are
`b0af887501e8124e317bee169628da783238e56f4c7ecb8a280fb699b930c587`
and `5197668eed9e063c5fef31ab9e9611a3df12b3ef17c1705b3b903f4bbcd6f844`.
Neither failure wrote a disposition or recomputed a model/score.

The canonical immutable diagnostic root is
`D:\quant-data\b3\processed\model_runs\v2_pass4g_occupancy_diagnostic_fa5c8af_20260907T013518Z`.
Its diagnostic-manifest, log-inclusive artifact-inventory, and access-audit
SHA-256 values are respectively
`2085ed71008821b983f61a8f2ad2667de8a4a74977f7b6079e5def98f9a377d9`,
`1663e488cdad3ca0187cfe31491d91ff3067aa0079c3277f4cdeffddbb010feb`,
and `ae47dcc237755949476f76344e9bd0b8938cc97899e146a34f310fc304f68974`.
It hash-verified and replayed all 16 sealed Pass-4f score panels. Every
pre-existing headline field was bit-identical, official-validation/test access
was false, transfer chronology was clean, `research_claim=false`, and no
deployment changed.

The complete disposition table is below. `D1--D4` is zero in every row.
Occupancy share is the registered occupancy-component sum divided by total
gross shortfall; it can exceed one when signed sizing components offset it.

| Fold/book | Mean gross | Shortfall | Occupancy share | Largest occupancy term | Largest sizing term | D1--D4 | D5 |
| --- | ---: | ---: | ---: | --- | --- | ---: | ---: |
| F1 inverse-volatility | 1.800173 | 0.199827 | 0.076679 | exit gap 0.016667 | NAV drift 0.155072 | 0 | 0.054054 |
| F1 momentum | 1.804345 | 0.195655 | 0.071445 | exit gap 0.016398 | NAV drift 0.159350 | 0 | 0.056338 |
| F1 reversal-21 | 1.973163 | 0.026837 | 0.801337 | exit gap 0.016398 | mark drift 0.028642 | 0 | 0.018557 |
| F1 reversal-5 | 1.931453 | 0.068547 | 0.929426 | band exhausted 0.047849 | fill 0.004120 | 0 | 0.003327 |
| F1 blend | 1.972213 | 0.027787 | 1.170576 | band exhausted 0.016935 | mark drift -0.015916 | 0 | 0.007307 |
| F2 inverse-volatility | 1.755766 | 0.244234 | 0.059435 | exit gap 0.016667 | mark drift 0.107124 | 0 | 0.048387 |
| F2 momentum | 1.828920 | 0.171080 | 0.094278 | exit gap 0.016129 | mark drift 0.071675 | 0 | 0.013514 |
| F2 reversal-21 | 1.952331 | 0.047669 | 0.428584 | exit gap 0.016398 | mark drift 0.028647 | 0 | 0.004367 |
| F2 reversal-5 | 1.937269 | 0.062731 | 0.827052 | band exhausted 0.036290 | fill 0.010426 | 0 | 0.000000 |
| F2 blend | 1.958096 | 0.041904 | 0.545282 | exit gap 0.016398 | fill 0.013762 | 0 | 0.002247 |
| F3 inverse-volatility | 1.888777 | 0.111223 | 0.120351 | exit gap 0.016273 | fill 0.145001 | 0 | **0.103448** |
| F3 momentum | 1.848185 | 0.151815 | 0.207463 | band exhausted 0.019685 | fill 0.112098 | 0 | **0.148936** |
| F3 reversal-21 | 1.964110 | 0.035890 | 0.511920 | exit gap 0.016535 | mark drift 0.020057 | 0 | 0.008032 |
| F3 reversal-5 | 1.947440 | 0.052560 | 0.853919 | band exhausted 0.029921 | fill 0.010717 | 0 | 0.001281 |
| F3 blend | 1.973285 | 0.026715 | 0.707380 | exit gap 0.017323 | NAV drift 0.010383 | 0 | 0.003052 |
| F1 GBDT ensemble | 1.840379 | 0.159621 | 0.114518 | exit gap 0.016398 | NAV drift 0.085670 | 0 | 0.013453 |

For the binding F2 inverse-volatility gross failure, occupancy explains only
`0.059435` of shortfall. The largest occupancy term is exit gap (`0.016667`),
whereas mark drift is the largest sizing term (`0.107124`). Its ten most
frequent raw-band names that were never held were `BRCOGNACNOR2` (68 band
sessions), `BRSUZBACNOR0` (57), `BRPTBLACNOR8` (55), `BREQTLACNOR0` (39),
`BRARMLACNOR1` (38), `BRCSEDACNOR9` (36), `BRYDUQACNOR3` (31),
`BRCMIGACNPR3` (25), `BRGRNDACNOR3` (24), and `BRMDIAACNOR7` (24). Every one
printed in all 124 evaluation sessions, had no missing-prior-print or
unresolved-action band session, and received zero entry submissions: these
are raw-band appearances while capacity was occupied, not unfillable orders.

The predeclared P0 stop fired because D5 exceeded `0.10` for F3
inverse-volatility (`0.103448`) and F3 momentum (`0.148936`). D1 through D4
were zero in all books. Accordingly, the conditional Section 5 gate rewrite
was not implemented, `main` was not fast-forwarded, no Round-1 root was frozen
or run, and no paid instance was launched. The result is a labelled defect
signature requiring a new pre-result contract; it is not an acceptance or
research claim. Direct provider inventory reads at
`2026-09-07T01:39:39.9369263Z` and `2026-09-07T01:39:45.5250149Z` each
returned zero instances. There was no paid host to terminate, and no adjacent
instance was touched.

## Pass-4h eligibility hysteresis, accepted replay, and Round 1 (2026-09-07)

Commit `f8986a4e1bccd672a692ac7a184bf2a4dad4bf53` implements the registered
five-session hold-through rule for a held name whose eligibility is temporarily
absent. Eligibility or closure resets the streak; session six submits an
`ineligible_hold_exhausted` exit, while the existing ten-session no-print
settlement path retains precedence. Entries remain eligible-only. Setting the
new field to zero is bit-identical to the pre-4h ledger. Commit
`384fd8142d6fd4c3cbd63ce1656b998854cfbf53` completes chained ancestor
verification for the acceptance replay. Ruff, compilation, the protocol guard,
and the full test suite passed before the replay.

The immutable accepted replay is
`D:\quant-data\b3\processed\model_runs\v2_development_acceptance_pass4h_384fd81_20260907T021757Z`.
Its pipeline manifest, sealed access audit, log-inclusive artifact inventory,
and 16-book CSV SHA-256 values are respectively
`97a1ed757e87e1e19f2c46999437b31f436470f2dae49d2ce0c5782635607f8f`,
`422df9edf2c6138e761cf14e907414ad1e60f8c15b9c2f9d06ad9755c73f8f0e`,
`94ac863f997f53769434dfa73a503bb89074973f2bea8ffc4d405bfc2d69c4a4`,
and `54e293cd10c6752c083d6d32e3f0566564670b62f6f5f6ac9d218e76fe734fbf`.
The result is `development_grade_inferred_actions` with no reasons. Every
non-ledger field is bit-identical to the sealed ancestor, D1--D5 are zero in
all books, all mean gross values are within 1.5--2.25, every mean stale or
unresolved fraction is below 2%, and protected access is false/false. The
single gross label is F2 inverse-volatility, `gross_underdeployed` at 1.725810.

The full before/after headline table is:

| Book | Gross 4g→4h | Label | Ineligible exits 4g→4h | Turnover 4g→4h | Net bps/day 4g→4h | Settlements 4g→4h | Stale mean 4h |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| F1 inverse-volatility | 1.800173→1.849769 | within | 35→28 | .071012→.067294 | 6.671→4.291 | 3→4 | .006682 |
| F1 momentum | 1.804345→1.802461 | within | 29→21 | .069464→.064802 | 6.766→5.725 | 2→3 | .005003 |
| F1 reversal-21 | 1.973163→1.964108 | within | 37→22 | .303311→.287610 | -13.325→-11.807 | 1→2 | .003055 |
| F1 reversal-5 | 1.931453→1.940747 | within | 24→12 | .868498→.859234 | -6.317→-6.878 | 2→3 | .005293 |
| F1 blend | 1.972213→1.983837 | within | 33→20 | .561452→.541686 | -6.099→-6.890 | 2→2 | .004179 |
| F2 inverse-volatility | 1.755766→1.725810 | under | 19→14 | .060188→.057059 | 9.852→10.962 | 2→2 | .002682 |
| F2 momentum | 1.828920→1.826791 | within | 15→11 | .067997→.064582 | 1.570→2.647 | 0→0 | .000000 |
| F2 reversal-21 | 1.952331→1.949948 | within | 22→16 | .282340→.272580 | -8.646→-7.873 | 1→1 | .001116 |
| F2 reversal-5 | 1.937269→1.945585 | within | 13→9 | .852630→.847550 | -11.961→-13.862 | 2→2 | .004287 |
| F2 blend | 1.958096→1.958106 | within | 22→16 | .512286→.505945 | -5.366→-4.244 | 1→1 | .001585 |
| F3 inverse-volatility | 1.888777→1.948083 | within | 22→13 | .077747→.089709 | 4.241→-2.091 | 3→3 | .006670 |
| F3 momentum | 1.848185→1.900454 | within | 28→13 | .066913→.066915 | 15.153→12.612 | 5→5 | .011474 |
| F3 reversal-21 | 1.964110→1.966923 | within | 24→19 | .297032→.293829 | -9.295→-7.572 | 3→3 | .006362 |
| F3 reversal-5 | 1.947440→1.962103 | within | 28→18 | .864234→.851757 | -5.589→-7.290 | 3→4 | .009013 |
| F3 blend | 1.973285→1.972861 | within | 24→15 | .546419→.533259 | 7.388→9.075 | 6→6 | .013596 |
| F1 GBDT integration ensemble | 1.840379→1.858859 | within | 11→5 | .158602→.151960 | 4.977→3.817 | 1→1 | .002475 |

Several preregistered expectations missed without triggering a stop: the largest
gross move was 0.059306, F3 inverse-volatility turnover rose 0.011962, and five
books gained a terminal settlement. These are reported outcomes, not defects;
the exhaustive engineering gates all passed.

Round 1 was frozen only after that acceptance. Commit
`81fe0cb824f79b7605920068d01442f5fa48fae3` separates and binds the store-build,
acceptance, and freeze implementations. The frozen root is
`D:\quant-data\b3\processed\model_runs\v2_round1_81fe0cb_20260907T023339Z`,
with frozen-design SHA-256
`b7d5fcccd2b7109f2a0a8892942023d387e1d33d4c8d08c04bf3f6c76320dc84`.
It binds store build `8021e42ae658d3eb7bba19e39b58e2f9c8d65331`, store-manifest
`deb9ca8449c5b9a83bf25ac19218069c006e183361b6f6ba836717ade63b491b`,
and the accepted replay above.

The first run completed all baselines and rungs A/B/C, then stopped before
writing a rung-D artifact because the adapter requested an options column from
a store that explicitly records options, rebalance, events, and fundamentals
as source-missing. Its logs and `failure_record.json` are retained. Commit
`cb6a0a5c4de202fe046dba48d9fd6c5af168dbfd` consumes only materialized
sidecar families (lending and oddlot here) while requiring every omitted family
to be explicitly source-missing. It does not invent values or change the
registered rung. All 852 tests passed in 376.53 seconds. The recovery hash-
verified and reused every completed candidate and scored only rung D and the
two absent span candidates; no completed candidate was retried.

The completed Round-1 result SHA-256 is
`ca39f11340f13956dca11da7fb6b3a8fa0a38f8a79cb558387d81aff26bf57a0`.
The sealed access audit and complete inventory SHA-256 values are
`7ea5772e253785264d4b724277d0d8376918cde674e09bcceb2920546a09f180`
and `b5084c817fc478c2759d962af12e282c292cade6f8e715aebed290ba5500ce08`.
The audit covers 696 files / 505,196,221 bytes and all 88 JSON artifacts carrying
access flags; official-validation/test access is false/false, transfer
chronology is clean, and deployment is unchanged. Independent verification
also passed for all 86 hash sidecars, all 33 score manifests / 66 array
payloads, and all 18 model manifests / 450 LightGBM model files.

Pooled Round-1 readouts are:

| Kind | Candidate | Primary IC [95%] | P1 / P5 | Spread bps/holding session | Net excess bps/day | Finite net days |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| naive | inverse-volatility-20 | .062256 [.029096,.098948] | .9940 / .9654 | 14.306 | 4.357 | 251 |
| naive | momentum-12-1 | .039860 [.022785,.060148] | .9935 / .9708 | 20.082 | 4.186 | 248 |
| naive | reversal-21 | -.018160 [-.039885,-.010381] | .9375 / .7313 | -2.491 | -9.072 | 375 |
| naive | reversal-5 | -.007037 [-.021975,.001454] | .7507 / .0109 | -3.299 | -9.327 | 375 |
| naive | blend | .023663 [.010381,.033518] | .8576 / .4352 | 5.249 | -5.567 | 248 |
| rung | A slow | .052871 [.026554,.084610] | .8884 / .8003 | 18.046 | 6.435 | 375 |
| rung | B intraday | .055264 [.027897,.087070] | .8785 / .7974 | 15.558 | 5.430 | 375 |
| rung | C lending | .053795 [.026543,.086484] | .8914 / .8199 | 14.321 | 5.364 | 375 |
| rung | D all available sidecars | .032725 [.005171,.059129] | .8484 / .7589 | 9.850 | 1.866 | 375 |
| span | fine only | .055264 [.027897,.087070] | .8785 / .7974 | 15.558 | 5.430 | 375 |
| span | pretrain decay 756 | .053459 [.032198,.080393] | .8516 / .7641 | 15.424 | 5.341 | 375 |
| span | pretrain uniform | .053680 [.029956,.082135] | .8314 / .7626 | 15.624 | 7.590 | 375 |

Per-fold GBDT ladder readouts (headline 4-bps/2%-borrow ledger) are:

| Rung/fold | IC [95%] | P1/P5 | Spread | Net | Gross (label) | Settlements / notional NAV | Econ unresolved | Stale mean |
| --- | ---: | ---: | ---: | ---: | --- | ---: | --- | ---: |
| A/F1 | .060022 [-.004792,.128546] | .9133/.8334 | 14.936 | 6.080 | 1.8504 (within) | 2 / .0442 | no | .0032 |
| A/F2 | .040432 [-.003467,.079738] | .8391/.7241 | 19.724 | 5.486 | 1.8641 (within) | 1 / .0202 | no | .0015 |
| A/F3 | .058028 [.025935,.105279] | .9121/.8422 | 19.442 | 7.709 | 1.8760 (within) | 4 / .1189 | no | .0086 |
| B/F1 | .057424 [-.007850,.126421] | .9027/.8304 | 13.892 | 6.512 | 1.8444 (within) | 2 / .0449 | no | .0033 |
| B/F2 | .045056 [-.000020,.081549] | .8294/.7251 | 13.115 | 6.911 | 1.8732 (within) | 1 / .0201 | no | .0015 |
| B/F3 | .063113 [.030185,.114925] | .9029/.8357 | 19.567 | 2.927 | 1.9001 (within) | 4 / .1222 | no | .0088 |
| C/F1 | .052741 [-.010798,.123072] | .9027/.8351 | 15.707 | 6.206 | 1.8076 (within) | 2 / .0528 | no | .0039 |
| C/F2 | .048702 [.000871,.088222] | .8755/.8006 | 13.580 | 7.870 | 1.7759 (under) | 1 / .0194 | no | .0015 |
| C/F3 | .059792 [.026978,.109782] | .8959/.8239 | 13.691 | 2.095 | 1.9012 (within) | 4 / .1134 | no | .0082 |
| D/F1 | .052268 [-.011935,.119748] | .8946/.8241 | 15.611 | 1.873 | 1.9078 (within) | 1 / .0342 | no | .0025 |
| D/F2 | .050332 [.000598,.090665] | .8849/.8175 | 14.380 | 9.311 | 1.7511 (under) | 1 / .0191 | no | .0014 |
| D/F3 | -.003512 [-.022342,.014170] | .7678/.6382 | -.187 | -5.410 | 1.9751 (within) | 3 / .0900 | no | .0064 |

Every row above has D1--D5 equal to zero. The naive floor has the same clean
signature and hard-bound/stale passes. Its only gross label is F2 inverse-
volatility; economics are unresolved for F1 inverse-volatility, F3 momentum,
and the F3 blend. The span preview has one gross label, uniform-pretrain F1 at
1.7569; all other span cells are within-band and resolved.

Registered paired pooled deltas are:

| Comparison | Δ primary IC [95%] | Δ P1 / P5 | Δ spread | Δ net [95%] bps/day |
| --- | ---: | ---: | ---: | ---: |
| B−A | .002393 [-.001660,.005711] | -.009869 / -.002832 | -2.488 | -1.006 [-7.101,3.310] |
| C−B | -.001468 [-.003293,.000971] | .012880 / .022480 | -1.238 | -.066 [-2.649,3.362] |
| D−C | -.021071 [-.040851,-.007148] | -.042946 / -.060974 | -4.471 | -3.498 [-12.680,2.483] |
| decay−fine | -.001804 [-.009564,.007663] | -.026872 / -.033301 | -.134 | -.089 [-3.281,3.597] |
| uniform−fine | -.001584 [-.008156,.005319] | -.047138 / -.034835 | .066 | 2.160 [-1.077,7.401] |

The rung rule keeps A and B, drops C and D because each step's primary-IC and
headline-net point deltas are both negative, and designates `b_intraday` as
the Round-2 GBDT parent. The naive inverse-volatility control has the highest
pooled IC, but it is a registered floor/control rather than a ladder parent.
The data-span preview remains informational: neither long-history arm improves
pooled IC over fine-only, while uniform pretraining has the better point
economics with an interval spanning zero.

Round 2 was not frozen or run. No paid instance was launched for Pass 4h or
Round 1. The next authorized operation, after Gabriel's explicit go-ahead, is
the disposable one-epoch Arm-A/F1/seed-11 smoke with no score directory,
followed only on success by the three registered Stage-P seeds. Direct Lambda
provider inventories at `2026-09-07T04:51:39.2629896Z` and
`2026-09-07T04:51:42.8034166Z` both returned zero instances; there was no paid
host to terminate and no adjacent instance was touched.

## Round-1 readout detail

Sealed result SHA-256: `ca39f11340f13956dca11da7fb6b3a8fa0a38f8a79cb558387d81aff26bf57a0`. Sealed inventory SHA-256: `b5084c817fc478c2759d962af12e282c292cade6f8e715aebed290ba5500ce08`. Machine-readable artifact: `round1_readout_detail.json` (SHA-256 `e2e02dc8f99b4f0f33748e9f9b7e9e6b9c6733491b44e6a194fe34c248b89de2`). No score or model was recomputed; only 2023–2024 development sessions were read; official-validation/test access remained false/false.

### Persisted pooled headline fields

| Candidate | Primary IC | 95% interval | Shareholder IC | Price IC | Spread bps/holding session | Net excess bps/day |
|---|---:|---:|---:|---:|---:|---:|
| rung_a | 0.052871 | [0.026554, 0.084610] | 0.055257 | 0.050284 | 18.045883 | 6.435485 |
| rung_b | 0.055264 | [0.027897, 0.087070] | 0.057656 | 0.052482 | 15.558374 | 5.429892 |
| inverse_volatility_control | 0.062256 | [0.029096, 0.098948] | 0.063783 | 0.058843 | 14.305564 | 4.357358 |
| momentum_control | 0.039860 | [0.022785, 0.060148] | 0.044495 | 0.041714 | 20.082323 | 4.186160 |

### Per-horizon readouts

| Candidate | Fold | H | Scaled target IC | Shareholder rank IC | Price rank IC | Shareholder spread bps/holding session |
|---|---|---:|---:|---:|---:|---:|
| rung_a | F1 | D1 | 0.042591 | 0.042720 | 0.040102 | 13.930751 |
| rung_a | F1 | D2 | 0.053889 | 0.053446 | 0.048731 | 11.906216 |
| rung_a | F1 | D3 | 0.071109 | 0.070389 | 0.064819 | 16.775000 |
| rung_a | F1 | D5 | 0.072497 | 0.069699 | 0.063467 | 14.158697 |
| rung_a | F1 | D10 | 0.091637 | 0.081455 | 0.071144 | 7.275807 |
| rung_a | F2 | D1 | 0.036557 | 0.039884 | 0.038463 | 22.176343 |
| rung_a | F2 | D2 | 0.040914 | 0.041661 | 0.037772 | 15.454825 |
| rung_a | F2 | D3 | 0.037330 | 0.039717 | 0.034088 | 18.421271 |
| rung_a | F2 | D5 | 0.046929 | 0.047917 | 0.040492 | 19.618995 |
| rung_a | F2 | D10 | 0.098055 | 0.097915 | 0.085328 | 25.000011 |
| rung_a | F3 | D1 | 0.047166 | 0.053602 | 0.050729 | 22.835314 |
| rung_a | F3 | D2 | 0.049071 | 0.056492 | 0.051664 | 19.575545 |
| rung_a | F3 | D3 | 0.056778 | 0.065058 | 0.059086 | 16.524651 |
| rung_a | F3 | D5 | 0.079098 | 0.085750 | 0.077180 | 22.124281 |
| rung_a | F3 | D10 | 0.119690 | 0.126599 | 0.115782 | 14.018617 |
| rung_b | F1 | D1 | 0.042681 | 0.042707 | 0.039839 | 10.319986 |
| rung_b | F1 | D2 | 0.049057 | 0.048088 | 0.045098 | 12.021883 |
| rung_b | F1 | D3 | 0.061983 | 0.061841 | 0.056953 | 10.812444 |
| rung_b | F1 | D5 | 0.075974 | 0.074725 | 0.067361 | 19.420623 |
| rung_b | F1 | D10 | 0.092037 | 0.082342 | 0.071187 | 11.262685 |
| rung_b | F2 | D1 | 0.041894 | 0.044106 | 0.041685 | 11.765706 |
| rung_b | F2 | D2 | 0.042302 | 0.042149 | 0.038130 | 13.927313 |
| rung_b | F2 | D3 | 0.043830 | 0.043967 | 0.037369 | 15.538734 |
| rung_b | F2 | D5 | 0.052199 | 0.055534 | 0.049119 | 6.137282 |
| rung_b | F2 | D10 | 0.096545 | 0.096904 | 0.084004 | 15.432067 |
| rung_b | F3 | D1 | 0.042191 | 0.049095 | 0.045851 | 18.389361 |
| rung_b | F3 | D2 | 0.054886 | 0.061915 | 0.056837 | 22.319286 |
| rung_b | F3 | D3 | 0.068570 | 0.075964 | 0.068336 | 16.943519 |
| rung_b | F3 | D5 | 0.086806 | 0.092495 | 0.083512 | 24.405735 |
| rung_b | F3 | D10 | 0.136057 | 0.140172 | 0.128070 | 18.574803 |
| inverse_volatility_control | F1 | D1 | 0.040141 | 0.041269 | 0.038901 | 19.149994 |
| inverse_volatility_control | F1 | D2 | 0.053785 | 0.053504 | 0.050202 | 18.642269 |
| inverse_volatility_control | F1 | D3 | 0.065127 | 0.064167 | 0.060271 | 19.995783 |
| inverse_volatility_control | F1 | D5 | 0.081530 | 0.077383 | 0.071820 | 18.903757 |
| inverse_volatility_control | F1 | D10 | 0.112514 | 0.103042 | 0.094324 | 13.280097 |
| inverse_volatility_control | F2 | D1 | 0.046689 | 0.049214 | 0.046582 | 18.246667 |
| inverse_volatility_control | F2 | D2 | 0.057633 | 0.056966 | 0.051710 | 15.185155 |
| inverse_volatility_control | F2 | D3 | 0.064751 | 0.068430 | 0.061783 | 15.330469 |
| inverse_volatility_control | F2 | D5 | 0.079570 | 0.080888 | 0.072356 | 16.591549 |
| inverse_volatility_control | F2 | D10 | 0.115216 | 0.114242 | 0.101295 | 21.672737 |
| inverse_volatility_control | F3 | D1 | 0.041248 | 0.048680 | 0.045901 | 5.372896 |
| inverse_volatility_control | F3 | D2 | 0.051925 | 0.058925 | 0.054122 | 3.993646 |
| inverse_volatility_control | F3 | D3 | 0.069758 | 0.075382 | 0.069299 | 6.019446 |
| inverse_volatility_control | F3 | D5 | 0.094699 | 0.096789 | 0.088221 | 10.193072 |
| inverse_volatility_control | F3 | D10 | 0.140788 | 0.140889 | 0.128987 | 12.286752 |
| momentum_control | F1 | D1 | 0.027003 | 0.026869 | 0.025598 | 8.105349 |
| momentum_control | F1 | D2 | 0.032803 | 0.034686 | 0.032900 | 11.116940 |
| momentum_control | F1 | D3 | 0.037389 | 0.042967 | 0.040375 | 12.300189 |
| momentum_control | F1 | D5 | 0.041534 | 0.047451 | 0.043790 | 12.683741 |
| momentum_control | F1 | D10 | 0.050721 | 0.053968 | 0.047782 | 12.530458 |
| momentum_control | F2 | D1 | 0.011621 | 0.017291 | 0.016082 | 11.994850 |
| momentum_control | F2 | D2 | 0.016236 | 0.019907 | 0.017632 | 12.359435 |
| momentum_control | F2 | D3 | 0.022548 | 0.027237 | 0.025005 | 13.816713 |
| momentum_control | F2 | D5 | 0.029823 | 0.035071 | 0.032000 | 13.932958 |
| momentum_control | F2 | D10 | 0.029303 | 0.040782 | 0.035452 | 15.309216 |
| momentum_control | F3 | D1 | 0.044251 | 0.048781 | 0.047204 | 33.588448 |
| momentum_control | F3 | D2 | 0.055081 | 0.062364 | 0.059161 | 33.449386 |
| momentum_control | F3 | D3 | 0.068907 | 0.074709 | 0.070402 | 32.813681 |
| momentum_control | F3 | D5 | 0.088663 | 0.092594 | 0.086385 | 34.686590 |
| momentum_control | F3 | D10 | 0.138000 | 0.144445 | 0.135471 | 35.344350 |

### Incremental-horizon ICs

| Candidate | Fold | Head | Increment | Mean IC |
|---|---|---:|---|---:|
| rung_a | F1 | D1 | 0→1 | 0.041145 |
| rung_a | F1 | D1 | 1→3 | 0.051974 |
| rung_a | F1 | D1 | 3→5 | 0.048405 |
| rung_a | F1 | D1 | 5→10 | 0.056586 |
| rung_a | F1 | D2 | 0→1 | 0.041996 |
| rung_a | F1 | D2 | 1→3 | 0.052772 |
| rung_a | F1 | D2 | 3→5 | 0.054597 |
| rung_a | F1 | D2 | 5→10 | 0.063807 |
| rung_a | F1 | D3 | 0→1 | 0.047674 |
| rung_a | F1 | D3 | 1→3 | 0.056142 |
| rung_a | F1 | D3 | 3→5 | 0.051127 |
| rung_a | F1 | D3 | 5→10 | 0.057080 |
| rung_a | F1 | D5 | 0→1 | 0.039064 |
| rung_a | F1 | D5 | 1→3 | 0.049409 |
| rung_a | F1 | D5 | 3→5 | 0.044820 |
| rung_a | F1 | D5 | 5→10 | 0.050856 |
| rung_a | F1 | D10 | 0→1 | 0.039000 |
| rung_a | F1 | D10 | 1→3 | 0.052975 |
| rung_a | F1 | D10 | 3→5 | 0.048978 |
| rung_a | F1 | D10 | 5→10 | 0.053770 |
| rung_a | F2 | D1 | 0→1 | 0.037853 |
| rung_a | F2 | D1 | 1→3 | 0.028286 |
| rung_a | F2 | D1 | 3→5 | 0.038603 |
| rung_a | F2 | D1 | 5→10 | 0.055375 |
| rung_a | F2 | D2 | 0→1 | 0.036818 |
| rung_a | F2 | D2 | 1→3 | 0.032749 |
| rung_a | F2 | D2 | 3→5 | 0.037217 |
| rung_a | F2 | D2 | 5→10 | 0.063035 |
| rung_a | F2 | D3 | 0→1 | 0.034547 |
| rung_a | F2 | D3 | 1→3 | 0.020575 |
| rung_a | F2 | D3 | 3→5 | 0.027782 |
| rung_a | F2 | D3 | 5→10 | 0.052008 |
| rung_a | F2 | D5 | 0→1 | 0.030292 |
| rung_a | F2 | D5 | 1→3 | 0.022764 |
| rung_a | F2 | D5 | 3→5 | 0.029229 |
| rung_a | F2 | D5 | 5→10 | 0.055068 |
| rung_a | F2 | D10 | 0→1 | 0.041034 |
| rung_a | F2 | D10 | 1→3 | 0.039563 |
| rung_a | F2 | D10 | 3→5 | 0.040288 |
| rung_a | F2 | D10 | 5→10 | 0.073753 |
| rung_a | F3 | D1 | 0→1 | 0.049421 |
| rung_a | F3 | D1 | 1→3 | 0.048655 |
| rung_a | F3 | D1 | 3→5 | 0.064388 |
| rung_a | F3 | D1 | 5→10 | 0.104047 |
| rung_a | F3 | D2 | 0→1 | 0.043481 |
| rung_a | F3 | D2 | 1→3 | 0.047928 |
| rung_a | F3 | D2 | 3→5 | 0.055666 |
| rung_a | F3 | D2 | 5→10 | 0.099482 |
| rung_a | F3 | D3 | 0→1 | 0.038382 |
| rung_a | F3 | D3 | 1→3 | 0.045807 |
| rung_a | F3 | D3 | 3→5 | 0.052260 |
| rung_a | F3 | D3 | 5→10 | 0.095450 |
| rung_a | F3 | D5 | 0→1 | 0.040429 |
| rung_a | F3 | D5 | 1→3 | 0.045760 |
| rung_a | F3 | D5 | 3→5 | 0.056937 |
| rung_a | F3 | D5 | 5→10 | 0.095240 |
| rung_a | F3 | D10 | 0→1 | 0.042673 |
| rung_a | F3 | D10 | 1→3 | 0.047877 |
| rung_a | F3 | D10 | 3→5 | 0.052605 |
| rung_a | F3 | D10 | 5→10 | 0.093072 |
| rung_b | F1 | D1 | 0→1 | 0.040603 |
| rung_b | F1 | D1 | 1→3 | 0.048727 |
| rung_b | F1 | D1 | 3→5 | 0.047542 |
| rung_b | F1 | D1 | 5→10 | 0.054406 |
| rung_b | F1 | D2 | 0→1 | 0.035366 |
| rung_b | F1 | D2 | 1→3 | 0.047896 |
| rung_b | F1 | D2 | 3→5 | 0.042865 |
| rung_b | F1 | D2 | 5→10 | 0.057036 |
| rung_b | F1 | D3 | 0→1 | 0.038439 |
| rung_b | F1 | D3 | 1→3 | 0.051957 |
| rung_b | F1 | D3 | 3→5 | 0.045629 |
| rung_b | F1 | D3 | 5→10 | 0.059820 |
| rung_b | F1 | D5 | 0→1 | 0.043180 |
| rung_b | F1 | D5 | 1→3 | 0.049930 |
| rung_b | F1 | D5 | 3→5 | 0.044673 |
| rung_b | F1 | D5 | 5→10 | 0.049888 |
| rung_b | F1 | D10 | 0→1 | 0.035756 |
| rung_b | F1 | D10 | 1→3 | 0.047086 |
| rung_b | F1 | D10 | 3→5 | 0.043658 |
| rung_b | F1 | D10 | 5→10 | 0.057341 |
| rung_b | F2 | D1 | 0→1 | 0.042455 |
| rung_b | F2 | D1 | 1→3 | 0.024620 |
| rung_b | F2 | D1 | 3→5 | 0.029651 |
| rung_b | F2 | D1 | 5→10 | 0.046609 |
| rung_b | F2 | D2 | 0→1 | 0.038257 |
| rung_b | F2 | D2 | 1→3 | 0.032404 |
| rung_b | F2 | D2 | 3→5 | 0.037843 |
| rung_b | F2 | D2 | 5→10 | 0.062966 |
| rung_b | F2 | D3 | 0→1 | 0.036473 |
| rung_b | F2 | D3 | 1→3 | 0.027805 |
| rung_b | F2 | D3 | 3→5 | 0.030830 |
| rung_b | F2 | D3 | 5→10 | 0.051754 |
| rung_b | F2 | D5 | 0→1 | 0.032974 |
| rung_b | F2 | D5 | 1→3 | 0.030484 |
| rung_b | F2 | D5 | 3→5 | 0.030671 |
| rung_b | F2 | D5 | 5→10 | 0.056701 |
| rung_b | F2 | D10 | 0→1 | 0.043415 |
| rung_b | F2 | D10 | 1→3 | 0.043542 |
| rung_b | F2 | D10 | 3→5 | 0.042620 |
| rung_b | F2 | D10 | 5→10 | 0.068248 |
| rung_b | F3 | D1 | 0→1 | 0.044715 |
| rung_b | F3 | D1 | 1→3 | 0.051858 |
| rung_b | F3 | D1 | 3→5 | 0.065252 |
| rung_b | F3 | D1 | 5→10 | 0.103904 |
| rung_b | F3 | D2 | 0→1 | 0.046403 |
| rung_b | F3 | D2 | 1→3 | 0.055849 |
| rung_b | F3 | D2 | 3→5 | 0.060995 |
| rung_b | F3 | D2 | 5→10 | 0.105160 |
| rung_b | F3 | D3 | 0→1 | 0.045331 |
| rung_b | F3 | D3 | 1→3 | 0.054911 |
| rung_b | F3 | D3 | 3→5 | 0.060933 |
| rung_b | F3 | D3 | 5→10 | 0.104771 |
| rung_b | F3 | D5 | 0→1 | 0.043421 |
| rung_b | F3 | D5 | 1→3 | 0.051878 |
| rung_b | F3 | D5 | 3→5 | 0.060580 |
| rung_b | F3 | D5 | 5→10 | 0.101669 |
| rung_b | F3 | D10 | 0→1 | 0.044477 |
| rung_b | F3 | D10 | 1→3 | 0.052271 |
| rung_b | F3 | D10 | 3→5 | 0.061361 |
| rung_b | F3 | D10 | 5→10 | 0.105645 |
| inverse_volatility_control | F1 | D1 | 0→1 | 0.039124 |
| inverse_volatility_control | F1 | D1 | 1→3 | 0.056273 |
| inverse_volatility_control | F1 | D1 | 3→5 | 0.055861 |
| inverse_volatility_control | F1 | D1 | 5→10 | 0.074318 |
| inverse_volatility_control | F1 | D2 | 0→1 | 0.039124 |
| inverse_volatility_control | F1 | D2 | 1→3 | 0.056273 |
| inverse_volatility_control | F1 | D2 | 3→5 | 0.055861 |
| inverse_volatility_control | F1 | D2 | 5→10 | 0.074318 |
| inverse_volatility_control | F1 | D3 | 0→1 | 0.039124 |
| inverse_volatility_control | F1 | D3 | 1→3 | 0.056273 |
| inverse_volatility_control | F1 | D3 | 3→5 | 0.055861 |
| inverse_volatility_control | F1 | D3 | 5→10 | 0.074318 |
| inverse_volatility_control | F1 | D5 | 0→1 | 0.039124 |
| inverse_volatility_control | F1 | D5 | 1→3 | 0.056273 |
| inverse_volatility_control | F1 | D5 | 3→5 | 0.055861 |
| inverse_volatility_control | F1 | D5 | 5→10 | 0.074318 |
| inverse_volatility_control | F1 | D10 | 0→1 | 0.039124 |
| inverse_volatility_control | F1 | D10 | 1→3 | 0.056273 |
| inverse_volatility_control | F1 | D10 | 3→5 | 0.055861 |
| inverse_volatility_control | F1 | D10 | 5→10 | 0.074318 |
| inverse_volatility_control | F2 | D1 | 0→1 | 0.047911 |
| inverse_volatility_control | F2 | D1 | 1→3 | 0.049775 |
| inverse_volatility_control | F2 | D1 | 3→5 | 0.050304 |
| inverse_volatility_control | F2 | D1 | 5→10 | 0.081522 |
| inverse_volatility_control | F2 | D2 | 0→1 | 0.047911 |
| inverse_volatility_control | F2 | D2 | 1→3 | 0.049775 |
| inverse_volatility_control | F2 | D2 | 3→5 | 0.050304 |
| inverse_volatility_control | F2 | D2 | 5→10 | 0.081522 |
| inverse_volatility_control | F2 | D3 | 0→1 | 0.047911 |
| inverse_volatility_control | F2 | D3 | 1→3 | 0.049775 |
| inverse_volatility_control | F2 | D3 | 3→5 | 0.050304 |
| inverse_volatility_control | F2 | D3 | 5→10 | 0.081522 |
| inverse_volatility_control | F2 | D5 | 0→1 | 0.047911 |
| inverse_volatility_control | F2 | D5 | 1→3 | 0.049775 |
| inverse_volatility_control | F2 | D5 | 3→5 | 0.050304 |
| inverse_volatility_control | F2 | D5 | 5→10 | 0.081522 |
| inverse_volatility_control | F2 | D10 | 0→1 | 0.047911 |
| inverse_volatility_control | F2 | D10 | 1→3 | 0.049775 |
| inverse_volatility_control | F2 | D10 | 3→5 | 0.050304 |
| inverse_volatility_control | F2 | D10 | 5→10 | 0.081522 |
| inverse_volatility_control | F3 | D1 | 0→1 | 0.044295 |
| inverse_volatility_control | F3 | D1 | 1→3 | 0.055754 |
| inverse_volatility_control | F3 | D1 | 3→5 | 0.064523 |
| inverse_volatility_control | F3 | D1 | 5→10 | 0.102899 |
| inverse_volatility_control | F3 | D2 | 0→1 | 0.044295 |
| inverse_volatility_control | F3 | D2 | 1→3 | 0.055754 |
| inverse_volatility_control | F3 | D2 | 3→5 | 0.064523 |
| inverse_volatility_control | F3 | D2 | 5→10 | 0.102899 |
| inverse_volatility_control | F3 | D3 | 0→1 | 0.044295 |
| inverse_volatility_control | F3 | D3 | 1→3 | 0.055754 |
| inverse_volatility_control | F3 | D3 | 3→5 | 0.064523 |
| inverse_volatility_control | F3 | D3 | 5→10 | 0.102899 |
| inverse_volatility_control | F3 | D5 | 0→1 | 0.044295 |
| inverse_volatility_control | F3 | D5 | 1→3 | 0.055754 |
| inverse_volatility_control | F3 | D5 | 3→5 | 0.064523 |
| inverse_volatility_control | F3 | D5 | 5→10 | 0.102899 |
| inverse_volatility_control | F3 | D10 | 0→1 | 0.044295 |
| inverse_volatility_control | F3 | D10 | 1→3 | 0.055754 |
| inverse_volatility_control | F3 | D10 | 3→5 | 0.064523 |
| inverse_volatility_control | F3 | D10 | 5→10 | 0.102899 |
| momentum_control | F1 | D1 | 0→1 | 0.022992 |
| momentum_control | F1 | D1 | 1→3 | 0.030723 |
| momentum_control | F1 | D1 | 3→5 | 0.029349 |
| momentum_control | F1 | D1 | 5→10 | 0.031052 |
| momentum_control | F1 | D2 | 0→1 | 0.022992 |
| momentum_control | F1 | D2 | 1→3 | 0.030723 |
| momentum_control | F1 | D2 | 3→5 | 0.029349 |
| momentum_control | F1 | D2 | 5→10 | 0.031052 |
| momentum_control | F1 | D3 | 0→1 | 0.022992 |
| momentum_control | F1 | D3 | 1→3 | 0.030723 |
| momentum_control | F1 | D3 | 3→5 | 0.029349 |
| momentum_control | F1 | D3 | 5→10 | 0.031052 |
| momentum_control | F1 | D5 | 0→1 | 0.022992 |
| momentum_control | F1 | D5 | 1→3 | 0.030723 |
| momentum_control | F1 | D5 | 3→5 | 0.029349 |
| momentum_control | F1 | D5 | 5→10 | 0.031052 |
| momentum_control | F1 | D10 | 0→1 | 0.022992 |
| momentum_control | F1 | D10 | 1→3 | 0.030723 |
| momentum_control | F1 | D10 | 3→5 | 0.029349 |
| momentum_control | F1 | D10 | 5→10 | 0.031052 |
| momentum_control | F2 | D1 | 0→1 | 0.014589 |
| momentum_control | F2 | D1 | 1→3 | 0.015868 |
| momentum_control | F2 | D1 | 3→5 | 0.018298 |
| momentum_control | F2 | D1 | 5→10 | 0.020100 |
| momentum_control | F2 | D2 | 0→1 | 0.014589 |
| momentum_control | F2 | D2 | 1→3 | 0.015868 |
| momentum_control | F2 | D2 | 3→5 | 0.018298 |
| momentum_control | F2 | D2 | 5→10 | 0.020100 |
| momentum_control | F2 | D3 | 0→1 | 0.014589 |
| momentum_control | F2 | D3 | 1→3 | 0.015868 |
| momentum_control | F2 | D3 | 3→5 | 0.018298 |
| momentum_control | F2 | D3 | 5→10 | 0.020100 |
| momentum_control | F2 | D5 | 0→1 | 0.014589 |
| momentum_control | F2 | D5 | 1→3 | 0.015868 |
| momentum_control | F2 | D5 | 3→5 | 0.018298 |
| momentum_control | F2 | D5 | 5→10 | 0.020100 |
| momentum_control | F2 | D10 | 0→1 | 0.014589 |
| momentum_control | F2 | D10 | 1→3 | 0.015868 |
| momentum_control | F2 | D10 | 3→5 | 0.018298 |
| momentum_control | F2 | D10 | 5→10 | 0.020100 |
| momentum_control | F3 | D1 | 0→1 | 0.045551 |
| momentum_control | F3 | D1 | 1→3 | 0.058744 |
| momentum_control | F3 | D1 | 3→5 | 0.057476 |
| momentum_control | F3 | D1 | 5→10 | 0.093288 |
| momentum_control | F3 | D2 | 0→1 | 0.045551 |
| momentum_control | F3 | D2 | 1→3 | 0.058744 |
| momentum_control | F3 | D2 | 3→5 | 0.057476 |
| momentum_control | F3 | D2 | 5→10 | 0.093288 |
| momentum_control | F3 | D3 | 0→1 | 0.045551 |
| momentum_control | F3 | D3 | 1→3 | 0.058744 |
| momentum_control | F3 | D3 | 3→5 | 0.057476 |
| momentum_control | F3 | D3 | 5→10 | 0.093288 |
| momentum_control | F3 | D5 | 0→1 | 0.045551 |
| momentum_control | F3 | D5 | 1→3 | 0.058744 |
| momentum_control | F3 | D5 | 3→5 | 0.057476 |
| momentum_control | F3 | D5 | 5→10 | 0.093288 |
| momentum_control | F3 | D10 | 0→1 | 0.045551 |
| momentum_control | F3 | D10 | 1→3 | 0.058744 |
| momentum_control | F3 | D10 | 3→5 | 0.057476 |
| momentum_control | F3 | D10 | 5→10 | 0.093288 |

### Quality stratification

| Candidate | Fold | Dimension | Stratum | Primary IC | Used name-days | Status |
|---|---|---|---|---:|---:|---|
| rung_a | F1 | calendar_year | 2023 | 0.060022 | 23551 | supported |
| rung_a | F1 | causal_liquidity_quartile | Q1 | 0.064100 | 5877 | supported |
| rung_a | F1 | causal_liquidity_quartile | Q2 | 0.041557 | 5861 | supported |
| rung_a | F1 | causal_liquidity_quartile | Q3 | 0.061075 | 5901 | supported |
| rung_a | F1 | causal_liquidity_quartile | Q4 | 0.079570 | 5912 | supported |
| rung_a | F1 | history_age_sessions | 0_to_59 | NA | 2 | unsupported |
| rung_a | F1 | history_age_sessions | 60_to_251 | NA | 204 | unsupported |
| rung_a | F1 | history_age_sessions | 252_plus | 0.061052 | 23345 | supported |
| rung_a | F1 | eventual_survival_audit_label | survives_to_final_year | 0.060897 | 19955 | supported |
| rung_a | F1 | eventual_survival_audit_label | delisted_within_panel | 0.048998 | 3596 | supported |
| rung_a | F2 | calendar_year | 2024 | 0.040432 | 22610 | supported |
| rung_a | F2 | causal_liquidity_quartile | Q1 | 0.051623 | 5640 | supported |
| rung_a | F2 | causal_liquidity_quartile | Q2 | 0.035606 | 5638 | supported |
| rung_a | F2 | causal_liquidity_quartile | Q3 | 0.039009 | 5662 | supported |
| rung_a | F2 | causal_liquidity_quartile | Q4 | 0.051133 | 5670 | supported |
| rung_a | F2 | history_age_sessions | 0_to_59 | NA | 4 | unsupported |
| rung_a | F2 | history_age_sessions | 60_to_251 | NA | 286 | unsupported |
| rung_a | F2 | history_age_sessions | 252_plus | 0.038721 | 22320 | supported |
| rung_a | F2 | eventual_survival_audit_label | survives_to_final_year | 0.047120 | 19713 | supported |
| rung_a | F2 | eventual_survival_audit_label | delisted_within_panel | 0.020990 | 2897 | supported |
| rung_a | F3 | calendar_year | 2024 | 0.058028 | 22041 | supported |
| rung_a | F3 | causal_liquidity_quartile | Q1 | 0.074152 | 5505 | supported |
| rung_a | F3 | causal_liquidity_quartile | Q2 | 0.058145 | 5513 | supported |
| rung_a | F3 | causal_liquidity_quartile | Q3 | 0.044271 | 5497 | supported |
| rung_a | F3 | causal_liquidity_quartile | Q4 | 0.045439 | 5526 | supported |
| rung_a | F3 | history_age_sessions | 0_to_59 | NA | 4 | unsupported |
| rung_a | F3 | history_age_sessions | 60_to_251 | NA | 290 | unsupported |
| rung_a | F3 | history_age_sessions | 252_plus | 0.056449 | 21747 | supported |
| rung_a | F3 | eventual_survival_audit_label | survives_to_final_year | 0.061538 | 19691 | supported |
| rung_a | F3 | eventual_survival_audit_label | delisted_within_panel | 0.017056 | 2350 | supported |
| rung_b | F1 | calendar_year | 2023 | 0.057424 | 23551 | supported |
| rung_b | F1 | causal_liquidity_quartile | Q1 | 0.059795 | 5877 | supported |
| rung_b | F1 | causal_liquidity_quartile | Q2 | 0.041491 | 5861 | supported |
| rung_b | F1 | causal_liquidity_quartile | Q3 | 0.058409 | 5901 | supported |
| rung_b | F1 | causal_liquidity_quartile | Q4 | 0.077853 | 5912 | supported |
| rung_b | F1 | history_age_sessions | 0_to_59 | NA | 2 | unsupported |
| rung_b | F1 | history_age_sessions | 60_to_251 | NA | 204 | unsupported |
| rung_b | F1 | history_age_sessions | 252_plus | 0.058744 | 23345 | supported |
| rung_b | F1 | eventual_survival_audit_label | survives_to_final_year | 0.059980 | 19955 | supported |
| rung_b | F1 | eventual_survival_audit_label | delisted_within_panel | 0.046476 | 3596 | supported |
| rung_b | F2 | calendar_year | 2024 | 0.045056 | 22610 | supported |
| rung_b | F2 | causal_liquidity_quartile | Q1 | 0.050518 | 5640 | supported |
| rung_b | F2 | causal_liquidity_quartile | Q2 | 0.027028 | 5638 | supported |
| rung_b | F2 | causal_liquidity_quartile | Q3 | 0.045023 | 5662 | supported |
| rung_b | F2 | causal_liquidity_quartile | Q4 | 0.065092 | 5670 | supported |
| rung_b | F2 | history_age_sessions | 0_to_59 | NA | 4 | unsupported |
| rung_b | F2 | history_age_sessions | 60_to_251 | NA | 286 | unsupported |
| rung_b | F2 | history_age_sessions | 252_plus | 0.043626 | 22320 | supported |
| rung_b | F2 | eventual_survival_audit_label | survives_to_final_year | 0.050836 | 19713 | supported |
| rung_b | F2 | eventual_survival_audit_label | delisted_within_panel | 0.026061 | 2897 | supported |
| rung_b | F3 | calendar_year | 2024 | 0.063113 | 22041 | supported |
| rung_b | F3 | causal_liquidity_quartile | Q1 | 0.074781 | 5505 | supported |
| rung_b | F3 | causal_liquidity_quartile | Q2 | 0.067619 | 5513 | supported |
| rung_b | F3 | causal_liquidity_quartile | Q3 | 0.045467 | 5497 | supported |
| rung_b | F3 | causal_liquidity_quartile | Q4 | 0.054216 | 5526 | supported |
| rung_b | F3 | history_age_sessions | 0_to_59 | NA | 4 | unsupported |
| rung_b | F3 | history_age_sessions | 60_to_251 | NA | 290 | unsupported |
| rung_b | F3 | history_age_sessions | 252_plus | 0.061799 | 21747 | supported |
| rung_b | F3 | eventual_survival_audit_label | survives_to_final_year | 0.067659 | 19691 | supported |
| rung_b | F3 | eventual_survival_audit_label | delisted_within_panel | -0.000883 | 2350 | supported |
| inverse_volatility_control | F1 | calendar_year | 2023 | 0.060146 | 23551 | supported |
| inverse_volatility_control | F1 | causal_liquidity_quartile | Q1 | 0.057361 | 5877 | supported |
| inverse_volatility_control | F1 | causal_liquidity_quartile | Q2 | 0.050374 | 5861 | supported |
| inverse_volatility_control | F1 | causal_liquidity_quartile | Q3 | 0.052044 | 5901 | supported |
| inverse_volatility_control | F1 | causal_liquidity_quartile | Q4 | 0.089776 | 5912 | supported |
| inverse_volatility_control | F1 | history_age_sessions | 0_to_59 | NA | 2 | unsupported |
| inverse_volatility_control | F1 | history_age_sessions | 60_to_251 | NA | 204 | unsupported |
| inverse_volatility_control | F1 | history_age_sessions | 252_plus | 0.060864 | 23345 | supported |
| inverse_volatility_control | F1 | eventual_survival_audit_label | survives_to_final_year | 0.055823 | 19955 | supported |
| inverse_volatility_control | F1 | eventual_survival_audit_label | delisted_within_panel | 0.082960 | 3596 | supported |
| inverse_volatility_control | F2 | calendar_year | 2024 | 0.062161 | 22610 | supported |
| inverse_volatility_control | F2 | causal_liquidity_quartile | Q1 | 0.070483 | 5640 | supported |
| inverse_volatility_control | F2 | causal_liquidity_quartile | Q2 | 0.054285 | 5638 | supported |
| inverse_volatility_control | F2 | causal_liquidity_quartile | Q3 | 0.033796 | 5662 | supported |
| inverse_volatility_control | F2 | causal_liquidity_quartile | Q4 | 0.073085 | 5670 | supported |
| inverse_volatility_control | F2 | history_age_sessions | 0_to_59 | NA | 4 | unsupported |
| inverse_volatility_control | F2 | history_age_sessions | 60_to_251 | NA | 286 | unsupported |
| inverse_volatility_control | F2 | history_age_sessions | 252_plus | 0.061807 | 22320 | supported |
| inverse_volatility_control | F2 | eventual_survival_audit_label | survives_to_final_year | 0.069245 | 19713 | supported |
| inverse_volatility_control | F2 | eventual_survival_audit_label | delisted_within_panel | 0.032928 | 2897 | supported |
| inverse_volatility_control | F3 | calendar_year | 2024 | 0.064407 | 22041 | supported |
| inverse_volatility_control | F3 | causal_liquidity_quartile | Q1 | 0.072832 | 5505 | supported |
| inverse_volatility_control | F3 | causal_liquidity_quartile | Q2 | 0.056606 | 5513 | supported |
| inverse_volatility_control | F3 | causal_liquidity_quartile | Q3 | 0.051051 | 5497 | supported |
| inverse_volatility_control | F3 | causal_liquidity_quartile | Q4 | 0.063266 | 5526 | supported |
| inverse_volatility_control | F3 | history_age_sessions | 0_to_59 | NA | 4 | unsupported |
| inverse_volatility_control | F3 | history_age_sessions | 60_to_251 | NA | 290 | unsupported |
| inverse_volatility_control | F3 | history_age_sessions | 252_plus | 0.064073 | 21747 | supported |
| inverse_volatility_control | F3 | eventual_survival_audit_label | survives_to_final_year | 0.068938 | 19691 | supported |
| inverse_volatility_control | F3 | eventual_survival_audit_label | delisted_within_panel | -0.050084 | 2350 | supported |
| momentum_control | F1 | calendar_year | 2023 | 0.034682 | 23342 | supported |
| momentum_control | F1 | causal_liquidity_quartile | Q1 | 0.024489 | 5721 | supported |
| momentum_control | F1 | causal_liquidity_quartile | Q2 | 0.030653 | 5816 | supported |
| momentum_control | F1 | causal_liquidity_quartile | Q3 | 0.033745 | 5893 | supported |
| momentum_control | F1 | causal_liquidity_quartile | Q4 | 0.056459 | 5912 | supported |
| momentum_control | F1 | history_age_sessions | 0_to_59 | NA | 0 | unsupported |
| momentum_control | F1 | history_age_sessions | 60_to_251 | NA | 0 | unsupported |
| momentum_control | F1 | history_age_sessions | 252_plus | 0.034682 | 23342 | supported |
| momentum_control | F1 | eventual_survival_audit_label | survives_to_final_year | 0.040404 | 19830 | supported |
| momentum_control | F1 | eventual_survival_audit_label | delisted_within_panel | 0.002311 | 3512 | supported |
| momentum_control | F2 | calendar_year | 2024 | 0.020057 | 22320 | supported |
| momentum_control | F2 | causal_liquidity_quartile | Q1 | 0.013058 | 5607 | supported |
| momentum_control | F2 | causal_liquidity_quartile | Q2 | 0.010715 | 5555 | supported |
| momentum_control | F2 | causal_liquidity_quartile | Q3 | -0.009360 | 5521 | supported |
| momentum_control | F2 | causal_liquidity_quartile | Q4 | 0.064534 | 5637 | supported |
| momentum_control | F2 | history_age_sessions | 0_to_59 | NA | 0 | unsupported |
| momentum_control | F2 | history_age_sessions | 60_to_251 | NA | 0 | unsupported |
| momentum_control | F2 | history_age_sessions | 252_plus | 0.020057 | 22320 | supported |
| momentum_control | F2 | eventual_survival_audit_label | survives_to_final_year | 0.016255 | 19500 | supported |
| momentum_control | F2 | eventual_survival_audit_label | delisted_within_panel | 0.041161 | 2820 | supported |
| momentum_control | F3 | calendar_year | 2024 | 0.064225 | 21686 | supported |
| momentum_control | F3 | causal_liquidity_quartile | Q1 | 0.068068 | 5493 | supported |
| momentum_control | F3 | causal_liquidity_quartile | Q2 | 0.038138 | 5316 | supported |
| momentum_control | F3 | causal_liquidity_quartile | Q3 | 0.073302 | 5389 | supported |
| momentum_control | F3 | causal_liquidity_quartile | Q4 | 0.069583 | 5488 | supported |
| momentum_control | F3 | history_age_sessions | 0_to_59 | NA | 0 | unsupported |
| momentum_control | F3 | history_age_sessions | 60_to_251 | NA | 0 | unsupported |
| momentum_control | F3 | history_age_sessions | 252_plus | 0.064225 | 21686 | supported |
| momentum_control | F3 | eventual_survival_audit_label | survives_to_final_year | 0.056789 | 19447 | supported |
| momentum_control | F3 | eventual_survival_audit_label | delisted_within_panel | -0.019334 | 2239 | supported |

### Exposure summary

| Candidate | Fold | Feature | Spearman | 95% interval | Defined dates |
|---|---|---|---:|---:|---:|
| rung_a | F1 | beta_60 | -0.665035 | [-0.684215, -0.642258] | 124 |
| rung_a | F1 | log_return_5 | 0.018342 | [-0.061816, 0.123442] | 124 |
| rung_a | F1 | log_volume_mean_20 | 0.128396 | [0.088376, 0.169578] | 124 |
| rung_a | F1 | momentum_12_1 | 0.403159 | [0.381298, 0.445303] | 124 |
| rung_a | F1 | yang_zhang_vol_20 | -0.810151 | [-0.827689, -0.801176] | 124 |
| rung_a | F2 | beta_60 | -0.487451 | [-0.527964, -0.452240] | 124 |
| rung_a | F2 | log_return_5 | -0.086593 | [-0.128679, -0.053686] | 124 |
| rung_a | F2 | log_volume_mean_20 | -0.109795 | [-0.139270, -0.081556] | 124 |
| rung_a | F2 | momentum_12_1 | 0.288566 | [0.227521, 0.322340] | 124 |
| rung_a | F2 | yang_zhang_vol_20 | -0.627951 | [-0.676062, -0.587409] | 124 |
| rung_a | F3 | beta_60 | -0.597697 | [-0.622968, -0.570342] | 127 |
| rung_a | F3 | log_return_5 | 0.004699 | [-0.042802, 0.064982] | 127 |
| rung_a | F3 | log_volume_mean_20 | 0.092312 | [0.052123, 0.133394] | 127 |
| rung_a | F3 | momentum_12_1 | 0.462551 | [0.425728, 0.485782] | 127 |
| rung_a | F3 | yang_zhang_vol_20 | -0.788980 | [-0.825344, -0.755069] | 127 |
| rung_b | F1 | beta_60 | -0.674199 | [-0.703049, -0.648551] | 124 |
| rung_b | F1 | log_return_5 | 0.019186 | [-0.061255, 0.123558] | 124 |
| rung_b | F1 | log_volume_mean_20 | 0.126699 | [0.095435, 0.159725] | 124 |
| rung_b | F1 | momentum_12_1 | 0.407355 | [0.386539, 0.448484] | 124 |
| rung_b | F1 | yang_zhang_vol_20 | -0.799393 | [-0.809041, -0.793566] | 124 |
| rung_b | F2 | beta_60 | -0.491631 | [-0.536701, -0.463459] | 124 |
| rung_b | F2 | log_return_5 | -0.082865 | [-0.129243, -0.048529] | 124 |
| rung_b | F2 | log_volume_mean_20 | -0.059195 | [-0.085133, -0.037099] | 124 |
| rung_b | F2 | momentum_12_1 | 0.271328 | [0.213932, 0.303963] | 124 |
| rung_b | F2 | yang_zhang_vol_20 | -0.631831 | [-0.671945, -0.603021] | 124 |
| rung_b | F3 | beta_60 | -0.598495 | [-0.620906, -0.573366] | 127 |
| rung_b | F3 | log_return_5 | 0.012867 | [-0.037953, 0.078563] | 127 |
| rung_b | F3 | log_volume_mean_20 | 0.105282 | [0.064291, 0.144206] | 127 |
| rung_b | F3 | momentum_12_1 | 0.466311 | [0.437219, 0.486679] | 127 |
| rung_b | F3 | yang_zhang_vol_20 | -0.792170 | [-0.827194, -0.755859] | 127 |
| inverse_volatility_control | F1 | beta_60 | -0.702494 | [-0.721042, -0.673981] | 124 |
| inverse_volatility_control | F1 | log_return_5 | 0.049760 | [-0.023623, 0.149984] | 124 |
| inverse_volatility_control | F1 | log_volume_mean_20 | 0.282766 | [0.249059, 0.331104] | 124 |
| inverse_volatility_control | F1 | momentum_12_1 | 0.224426 | [0.201383, 0.273411] | 124 |
| inverse_volatility_control | F1 | yang_zhang_vol_20 | -1.000000 | [-1.000000, -1.000000] | 124 |
| inverse_volatility_control | F2 | beta_60 | -0.692458 | [-0.717484, -0.662055] | 124 |
| inverse_volatility_control | F2 | log_return_5 | 0.084167 | [-0.006831, 0.142977] | 124 |
| inverse_volatility_control | F2 | log_volume_mean_20 | 0.242124 | [0.214205, 0.253873] | 124 |
| inverse_volatility_control | F2 | momentum_12_1 | 0.087469 | [0.011313, 0.125294] | 124 |
| inverse_volatility_control | F2 | yang_zhang_vol_20 | -1.000000 | [-1.000000, -1.000000] | 124 |
| inverse_volatility_control | F3 | beta_60 | -0.705853 | [-0.709321, -0.698000] | 127 |
| inverse_volatility_control | F3 | log_return_5 | 0.067113 | [0.004844, 0.142413] | 127 |
| inverse_volatility_control | F3 | log_volume_mean_20 | 0.239775 | [0.220566, 0.257374] | 127 |
| inverse_volatility_control | F3 | momentum_12_1 | 0.364823 | [0.315279, 0.396642] | 127 |
| inverse_volatility_control | F3 | yang_zhang_vol_20 | -1.000000 | [-1.000000, -1.000000] | 127 |
| momentum_control | F1 | beta_60 | -0.272813 | [-0.310982, -0.258980] | 124 |
| momentum_control | F1 | log_return_5 | 0.049122 | [-0.000862, 0.097800] | 124 |
| momentum_control | F1 | log_volume_mean_20 | -0.078077 | [-0.091283, -0.065614] | 124 |
| momentum_control | F1 | momentum_12_1 | 0.999501 | [0.998973, 0.999833] | 124 |
| momentum_control | F1 | yang_zhang_vol_20 | -0.225489 | [-0.273987, -0.202745] | 124 |
| momentum_control | F2 | beta_60 | -0.157840 | [-0.199117, -0.090349] | 124 |
| momentum_control | F2 | log_return_5 | 0.041822 | [0.011967, 0.073468] | 124 |
| momentum_control | F2 | log_volume_mean_20 | -0.048840 | [-0.086053, -0.016251] | 124 |
| momentum_control | F2 | momentum_12_1 | 0.999945 | [0.999883, 1.000000] | 124 |
| momentum_control | F2 | yang_zhang_vol_20 | -0.088495 | [-0.126372, -0.012130] | 124 |
| momentum_control | F3 | beta_60 | -0.437687 | [-0.502421, -0.365844] | 127 |
| momentum_control | F3 | log_return_5 | 0.091022 | [0.049222, 0.150645] | 127 |
| momentum_control | F3 | log_volume_mean_20 | 0.134036 | [0.101918, 0.185006] | 127 |
| momentum_control | F3 | momentum_12_1 | 0.998861 | [0.997115, 0.999850] | 127 |
| momentum_control | F3 | yang_zhang_vol_20 | -0.366950 | [-0.398550, -0.316786] | 127 |

### GBDT importance: top 15 by mean absolute TreeSHAP

| Candidate | Fold | Rank | Feature | Mean absolute TreeSHAP |
|---|---|---:|---|---:|
| rung_a | F1 | 1 | yang_zhang_vol_60 | 0.002641 |
| rung_a | F1 | 2 | idiosyncratic_vol_60 | 0.000964 |
| rung_a | F1 | 3 | max_return_21 | 0.000950 |
| rung_a | F1 | 4 | distance_52_week_high | 0.000800 |
| rung_a | F1 | 5 | cluster_dispersion | 0.000762 |
| rung_a | F1 | 6 | momentum_12_1 | 0.000702 |
| rung_a | F1 | 7 | amihud_20 | 0.000531 |
| rung_a | F1 | 8 | cluster_mean_return_5 | 0.000505 |
| rung_a | F1 | 9 | log_return_252 | 0.000428 |
| rung_a | F1 | 10 | log_return_21 | 0.000425 |
| rung_a | F1 | 11 | beta_60 | 0.000413 |
| rung_a | F1 | 12 | cluster_mean_return_21 | 0.000383 |
| rung_a | F1 | 13 | vol_of_vol_60 | 0.000356 |
| rung_a | F1 | 14 | yang_zhang_vol_20 | 0.000306 |
| rung_a | F1 | 15 | realized_skew_60 | 0.000296 |
| rung_a | F2 | 1 | max_return_21 | 0.017765 |
| rung_a | F2 | 2 | amihud_20 | 0.014649 |
| rung_a | F2 | 3 | log_return_21 | 0.013366 |
| rung_a | F2 | 4 | yang_zhang_vol_60 | 0.012005 |
| rung_a | F2 | 5 | momentum_12_1 | 0.010614 |
| rung_a | F2 | 6 | beta_60 | 0.009869 |
| rung_a | F2 | 7 | distance_52_week_high | 0.008698 |
| rung_a | F2 | 8 | log_return_63 | 0.007199 |
| rung_a | F2 | 9 | realized_kurtosis_60 | 0.006967 |
| rung_a | F2 | 10 | vol_of_vol_60 | 0.006583 |
| rung_a | F2 | 11 | realized_skew_60 | 0.005643 |
| rung_a | F2 | 12 | idiosyncratic_vol_60 | 0.005017 |
| rung_a | F2 | 13 | log_return_252 | 0.004397 |
| rung_a | F2 | 14 | high_low_range_5 | 0.004045 |
| rung_a | F2 | 15 | cluster_mean_return_21 | 0.003925 |
| rung_a | F3 | 1 | yang_zhang_vol_60 | 0.011589 |
| rung_a | F3 | 2 | distance_52_week_high | 0.007459 |
| rung_a | F3 | 3 | momentum_12_1 | 0.004683 |
| rung_a | F3 | 4 | beta_60 | 0.004027 |
| rung_a | F3 | 5 | log_return_252 | 0.003965 |
| rung_a | F3 | 6 | idiosyncratic_vol_60 | 0.003506 |
| rung_a | F3 | 7 | amihud_20 | 0.003099 |
| rung_a | F3 | 8 | log_return_21 | 0.002471 |
| rung_a | F3 | 9 | max_return_21 | 0.002404 |
| rung_a | F3 | 10 | vol_of_vol_60 | 0.002277 |
| rung_a | F3 | 11 | yang_zhang_vol_20 | 0.002220 |
| rung_a | F3 | 12 | log_return_63 | 0.001953 |
| rung_a | F3 | 13 | log_return_126 | 0.001601 |
| rung_a | F3 | 14 | cluster_mean_return_21 | 0.001577 |
| rung_a | F3 | 15 | cluster_dispersion | 0.001443 |
| rung_b | F1 | 1 | log_return_21 | 0.007466 |
| rung_b | F1 | 2 | yang_zhang_vol_60 | 0.006373 |
| rung_b | F1 | 3 | max_return_21 | 0.005863 |
| rung_b | F1 | 4 | idiosyncratic_vol_60 | 0.005542 |
| rung_b | F1 | 5 | amihud_20 | 0.005035 |
| rung_b | F1 | 6 | observed_history_age_sessions | 0.004491 |
| rung_b | F1 | 7 | beta_60 | 0.004360 |
| rung_b | F1 | 8 | vol_of_vol_60 | 0.003001 |
| rung_b | F1 | 9 | distance_52_week_high | 0.002831 |
| rung_b | F1 | 10 | log_return_63 | 0.002666 |
| rung_b | F1 | 11 | cluster_mean_return_5 | 0.002240 |
| rung_b | F1 | 12 | momentum_12_1 | 0.002198 |
| rung_b | F1 | 13 | log_return_252 | 0.002071 |
| rung_b | F1 | 14 | realized_skew_5m_20 | 0.001792 |
| rung_b | F1 | 15 | observed_history_left_censored | 0.001646 |
| rung_b | F2 | 1 | max_return_21 | 0.016833 |
| rung_b | F2 | 2 | amihud_20 | 0.014691 |
| rung_b | F2 | 3 | log_return_21 | 0.014589 |
| rung_b | F2 | 4 | beta_60 | 0.010935 |
| rung_b | F2 | 5 | distance_52_week_high | 0.010270 |
| rung_b | F2 | 6 | yang_zhang_vol_60 | 0.009526 |
| rung_b | F2 | 7 | momentum_12_1 | 0.007488 |
| rung_b | F2 | 8 | vol_of_vol_60 | 0.005493 |
| rung_b | F2 | 9 | idiosyncratic_vol_60 | 0.005456 |
| rung_b | F2 | 10 | log_return_252 | 0.005395 |
| rung_b | F2 | 11 | observed_history_age_sessions | 0.005354 |
| rung_b | F2 | 12 | log_return_63 | 0.005183 |
| rung_b | F2 | 13 | realized_kurtosis_60 | 0.004934 |
| rung_b | F2 | 14 | realized_skew_60 | 0.004221 |
| rung_b | F2 | 15 | high_low_range_5 | 0.004133 |
| rung_b | F3 | 1 | yang_zhang_vol_60 | 0.014091 |
| rung_b | F3 | 2 | distance_52_week_high | 0.008186 |
| rung_b | F3 | 3 | momentum_12_1 | 0.007544 |
| rung_b | F3 | 4 | beta_60 | 0.005427 |
| rung_b | F3 | 5 | idiosyncratic_vol_60 | 0.005093 |
| rung_b | F3 | 6 | max_return_21 | 0.004956 |
| rung_b | F3 | 7 | amihud_20 | 0.004864 |
| rung_b | F3 | 8 | log_return_252 | 0.004263 |
| rung_b | F3 | 9 | log_return_21 | 0.003840 |
| rung_b | F3 | 10 | yang_zhang_vol_20 | 0.002869 |
| rung_b | F3 | 11 | vol_of_vol_60 | 0.002593 |
| rung_b | F3 | 12 | cluster_mean_return_21 | 0.002287 |
| rung_b | F3 | 13 | cluster_dispersion | 0.002174 |
| rung_b | F3 | 14 | log_return_63 | 0.002109 |
| rung_b | F3 | 15 | log_return_126 | 0.002096 |

### Rung-B intraday-derived feature ranks

These are exactly the persisted rung-B features absent from rung A, including their age channels and `fast_present`.

| Fold | Feature | TreeSHAP rank | TreeSHAP | Gain rank | Gain |
|---|---|---:|---:|---:|---:|
| F1 | overnight_return | 92 | 0.000000 | 92 | 0.000000 |
| F1 | intraday_return_1545 | 27 | 0.000834 | 34 | 323.260187 |
| F1 | overnight_return_sum_5 | 95 | 0.000000 | 95 | 0.000000 |
| F1 | overnight_return_sum_20 | 93 | 0.000000 | 93 | 0.000000 |
| F1 | intraday_return_sum_5 | 51 | 0.000268 | 45 | 155.155162 |
| F1 | intraday_return_sum_20 | 79 | 0.000000 | 79 | 0.000000 |
| F1 | overnight_minus_intraday | 89 | 0.000000 | 89 | 0.000000 |
| F1 | overnight_minus_intraday_mean_20 | 90 | 0.000000 | 90 | 0.000000 |
| F1 | last_30_minute_return_share_lag1 | 80 | 0.000000 | 80 | 0.000000 |
| F1 | last_hour_volume_share_lag1 | 46 | 0.000313 | 46 | 152.364149 |
| F1 | close_vwap_deviation_lag1 | 73 | 0.000000 | 73 | 0.000000 |
| F1 | vwap_deviation_1545 | 55 | 0.000212 | 50 | 128.682860 |
| F1 | realized_vol_5m_1 | 48 | 0.000312 | 39 | 285.529018 |
| F1 | realized_vol_5m_5 | 42 | 0.000409 | 30 | 391.262213 |
| F1 | realized_vol_5m_20 | 31 | 0.000686 | 24 | 607.932551 |
| F1 | realized_skew_5m_20 | 14 | 0.001792 | 21 | 643.587445 |
| F1 | roll_spread_20 | 39 | 0.000501 | 25 | 582.398489 |
| F1 | corwin_schultz_spread_20 | 23 | 0.001027 | 22 | 628.692727 |
| F1 | intraday_range_1545 | 40 | 0.000479 | 36 | 291.926573 |
| F1 | volume_1545_relative_median_20 | 101 | 0.000000 | 101 | 0.000000 |
| F1 | overnight_return__age_sessions | 62 | 0.000081 | 61 | 18.235996 |
| F1 | intraday_return_1545__age_sessions | 56 | 0.000208 | 56 | 44.809429 |
| F1 | overnight_return_sum_5__age_sessions | 61 | 0.000115 | 58 | 31.895329 |
| F1 | overnight_return_sum_20__age_sessions | 94 | 0.000000 | 94 | 0.000000 |
| F1 | intraday_return_sum_5__age_sessions | 30 | 0.000757 | 26 | 579.280282 |
| F1 | intraday_return_sum_20__age_sessions | 33 | 0.000639 | 23 | 625.299477 |
| F1 | overnight_minus_intraday__age_sessions | 66 | 0.000028 | 65 | 4.724472 |
| F1 | overnight_minus_intraday_mean_20__age_sessions | 91 | 0.000000 | 91 | 0.000000 |
| F1 | last_30_minute_return_share_lag1__age_sessions | 65 | 0.000029 | 62 | 13.750499 |
| F1 | last_hour_volume_share_lag1__age_sessions | 38 | 0.000510 | 40 | 280.382067 |
| F1 | close_vwap_deviation_lag1__age_sessions | 63 | 0.000072 | 57 | 36.866114 |
| F1 | vwap_deviation_1545__age_sessions | 37 | 0.000515 | 44 | 218.647336 |
| F1 | realized_vol_5m_1__age_sessions | 54 | 0.000221 | 55 | 60.518030 |
| F1 | realized_vol_5m_5__age_sessions | 44 | 0.000370 | 47 | 141.930133 |
| F1 | realized_vol_5m_20__age_sessions | 52 | 0.000266 | 42 | 276.740539 |
| F1 | realized_skew_5m_20__age_sessions | 53 | 0.000249 | 48 | 131.783112 |
| F1 | roll_spread_20__age_sessions | 34 | 0.000613 | 35 | 314.794123 |
| F1 | corwin_schultz_spread_20__age_sessions | 57 | 0.000187 | 51 | 115.997125 |
| F1 | intraday_range_1545__age_sessions | 58 | 0.000152 | 60 | 23.977750 |
| F1 | volume_1545_relative_median_20__age_sessions | 47 | 0.000313 | 52 | 104.890745 |
| F1 | fast_present | 75 | 0.000000 | 75 | 0.000000 |
| F2 | overnight_return | 92 | 0.000000 | 92 | 0.000000 |
| F2 | intraday_return_1545 | 35 | 0.001018 | 35 | 783.673868 |
| F2 | overnight_return_sum_5 | 95 | 0.000000 | 95 | 0.000000 |
| F2 | overnight_return_sum_20 | 93 | 0.000000 | 93 | 0.000000 |
| F2 | intraday_return_sum_5 | 51 | 0.000584 | 43 | 517.649620 |
| F2 | intraday_return_sum_20 | 78 | 0.000000 | 78 | 0.000000 |
| F2 | overnight_minus_intraday | 89 | 0.000000 | 89 | 0.000000 |
| F2 | overnight_minus_intraday_mean_20 | 90 | 0.000000 | 90 | 0.000000 |
| F2 | last_30_minute_return_share_lag1 | 79 | 0.000000 | 79 | 0.000000 |
| F2 | last_hour_volume_share_lag1 | 47 | 0.000753 | 46 | 438.607228 |
| F2 | close_vwap_deviation_lag1 | 74 | 0.000000 | 74 | 0.000000 |
| F2 | vwap_deviation_1545 | 53 | 0.000468 | 47 | 412.914291 |
| F2 | realized_vol_5m_1 | 43 | 0.000786 | 37 | 735.170988 |
| F2 | realized_vol_5m_5 | 32 | 0.001146 | 32 | 888.922597 |
| F2 | realized_vol_5m_20 | 40 | 0.000843 | 30 | 975.535750 |
| F2 | realized_skew_5m_20 | 27 | 0.001508 | 22 | 1281.589605 |
| F2 | roll_spread_20 | 37 | 0.000945 | 28 | 1041.320465 |
| F2 | corwin_schultz_spread_20 | 26 | 0.001798 | 26 | 1130.876174 |
| F2 | intraday_range_1545 | 36 | 0.000994 | 36 | 766.684469 |
| F2 | volume_1545_relative_median_20 | 101 | 0.000000 | 101 | 0.000000 |
| F2 | overnight_return__age_sessions | 62 | 0.000112 | 61 | 37.171221 |
| F2 | intraday_return_1545__age_sessions | 58 | 0.000233 | 58 | 86.248540 |
| F2 | overnight_return_sum_5__age_sessions | 56 | 0.000279 | 57 | 86.505595 |
| F2 | overnight_return_sum_20__age_sessions | 94 | 0.000000 | 94 | 0.000000 |
| F2 | intraday_return_sum_5__age_sessions | 41 | 0.000818 | 33 | 832.988881 |
| F2 | intraday_return_sum_20__age_sessions | 30 | 0.001213 | 24 | 1201.100247 |
| F2 | overnight_minus_intraday__age_sessions | 67 | 0.000019 | 65 | 6.450860 |
| F2 | overnight_minus_intraday_mean_20__age_sessions | 91 | 0.000000 | 91 | 0.000000 |
| F2 | last_30_minute_return_share_lag1__age_sessions | 63 | 0.000077 | 62 | 21.376173 |
| F2 | last_hour_volume_share_lag1__age_sessions | 45 | 0.000768 | 42 | 548.904472 |
| F2 | close_vwap_deviation_lag1__age_sessions | 55 | 0.000300 | 56 | 112.950284 |
| F2 | vwap_deviation_1545__age_sessions | 48 | 0.000698 | 45 | 454.737828 |
| F2 | realized_vol_5m_1__age_sessions | 57 | 0.000263 | 55 | 116.103565 |
| F2 | realized_vol_5m_5__age_sessions | 54 | 0.000336 | 49 | 204.826775 |
| F2 | realized_vol_5m_20__age_sessions | 31 | 0.001194 | 41 | 650.906801 |
| F2 | realized_skew_5m_20__age_sessions | 46 | 0.000754 | 48 | 217.803761 |
| F2 | roll_spread_20__age_sessions | 42 | 0.000801 | 44 | 461.471886 |
| F2 | corwin_schultz_spread_20__age_sessions | 50 | 0.000616 | 50 | 204.667645 |
| F2 | intraday_range_1545__age_sessions | 61 | 0.000134 | 60 | 42.156233 |
| F2 | volume_1545_relative_median_20__age_sessions | 52 | 0.000553 | 52 | 178.932768 |
| F2 | fast_present | 68 | 0.000006 | 69 | 1.674831 |
| F3 | overnight_return | 92 | 0.000000 | 92 | 0.000000 |
| F3 | intraday_return_1545 | 55 | 0.000148 | 44 | 163.050481 |
| F3 | overnight_return_sum_5 | 95 | 0.000000 | 95 | 0.000000 |
| F3 | overnight_return_sum_20 | 93 | 0.000000 | 93 | 0.000000 |
| F3 | intraday_return_sum_5 | 54 | 0.000159 | 48 | 146.486611 |
| F3 | intraday_return_sum_20 | 79 | 0.000000 | 79 | 0.000000 |
| F3 | overnight_minus_intraday | 89 | 0.000000 | 89 | 0.000000 |
| F3 | overnight_minus_intraday_mean_20 | 90 | 0.000000 | 90 | 0.000000 |
| F3 | last_30_minute_return_share_lag1 | 80 | 0.000000 | 80 | 0.000000 |
| F3 | last_hour_volume_share_lag1 | 56 | 0.000121 | 55 | 65.207506 |
| F3 | close_vwap_deviation_lag1 | 74 | 0.000000 | 74 | 0.000000 |
| F3 | vwap_deviation_1545 | 57 | 0.000090 | 51 | 97.437217 |
| F3 | realized_vol_5m_1 | 41 | 0.000382 | 40 | 203.599348 |
| F3 | realized_vol_5m_5 | 44 | 0.000355 | 34 | 282.009528 |
| F3 | realized_vol_5m_20 | 32 | 0.000577 | 21 | 649.128705 |
| F3 | realized_skew_5m_20 | 21 | 0.001110 | 23 | 602.046398 |
| F3 | roll_spread_20 | 33 | 0.000550 | 26 | 507.152824 |
| F3 | corwin_schultz_spread_20 | 28 | 0.000729 | 25 | 514.254204 |
| F3 | intraday_range_1545 | 30 | 0.000657 | 39 | 205.741899 |
| F3 | volume_1545_relative_median_20 | 101 | 0.000000 | 101 | 0.000000 |
| F3 | overnight_return__age_sessions | 62 | 0.000016 | 61 | 11.174937 |
| F3 | intraday_return_1545__age_sessions | 59 | 0.000057 | 58 | 18.306473 |
| F3 | overnight_return_sum_5__age_sessions | 50 | 0.000210 | 53 | 85.407936 |
| F3 | overnight_return_sum_20__age_sessions | 94 | 0.000000 | 94 | 0.000000 |
| F3 | intraday_return_sum_5__age_sessions | 42 | 0.000356 | 31 | 345.834808 |
| F3 | intraday_return_sum_20__age_sessions | 20 | 0.001148 | 8 | 1302.770292 |
| F3 | overnight_minus_intraday__age_sessions | 65 | 0.000009 | 64 | 4.655178 |
| F3 | overnight_minus_intraday_mean_20__age_sessions | 91 | 0.000000 | 91 | 0.000000 |
| F3 | last_30_minute_return_share_lag1__age_sessions | 63 | 0.000011 | 63 | 5.378600 |
| F3 | last_hour_volume_share_lag1__age_sessions | 47 | 0.000255 | 33 | 312.970187 |
| F3 | close_vwap_deviation_lag1__age_sessions | 58 | 0.000082 | 56 | 38.421316 |
| F3 | vwap_deviation_1545__age_sessions | 40 | 0.000388 | 37 | 238.731776 |
| F3 | realized_vol_5m_1__age_sessions | 51 | 0.000198 | 43 | 177.875468 |
| F3 | realized_vol_5m_5__age_sessions | 48 | 0.000245 | 38 | 220.175403 |
| F3 | realized_vol_5m_20__age_sessions | 35 | 0.000495 | 27 | 440.503097 |
| F3 | realized_skew_5m_20__age_sessions | 39 | 0.000394 | 45 | 156.074366 |
| F3 | roll_spread_20__age_sessions | 31 | 0.000589 | 28 | 414.036576 |
| F3 | corwin_schultz_spread_20__age_sessions | 49 | 0.000226 | 49 | 144.677801 |
| F3 | intraday_range_1545__age_sessions | 61 | 0.000021 | 60 | 11.914786 |
| F3 | volume_1545_relative_median_20__age_sessions | 38 | 0.000397 | 36 | 263.225595 |
| F3 | fast_present | 75 | 0.000000 | 75 | 0.000000 |

### Diagnostic-only realized market-beta proxy

OLS includes an intercept. The market proxy is the equal-weight D1 shareholder return of names active at t−1, aligned to the book ledger return at t. It was not used for selection.

| Candidate | Fold | Slope | t-stat | R² | Observations |
|---|---|---:|---:|---:|---:|
| rung_b | F1 | -0.737266 | -11.194547 | 0.506708 | 124 |
| rung_b | F2 | -0.694404 | -11.607279 | 0.524791 | 124 |
| rung_b | F3 | -0.912572 | -12.503744 | 0.555703 | 127 |
| inverse_volatility_control | F1 | -0.921385 | -11.958934 | 0.539651 | 124 |
| inverse_volatility_control | F2 | -0.913068 | -13.122840 | 0.585329 | 124 |
| inverse_volatility_control | F3 | -1.025159 | -13.093836 | 0.578342 | 127 |
| momentum_control | F1 | -0.368295 | -5.901614 | 0.222083 | 124 |
| momentum_control | F2 | -0.111374 | -1.642384 | 0.021632 | 124 |
| momentum_control | F3 | -0.763192 | -10.738612 | 0.479855 | 127 |

### Oddlot source coverage in rung D

No monthly source-coverage rows were persisted, so the available fold summaries are reported.

| Fold | Active name-days | Present name-days | Present fraction | Valid fraction conditional on present | Status |
|---|---:|---:|---:|---:|---|
| F1 | 24598 | 24565 | 0.998658 | 1.000000 | supported |
| F2 | 23568 | 23555 | 0.999448 | 1.000000 | supported |
| F3 | 23059 | 0 | 0.000000 | NA | unsupported |

Interpretation: oddlot was essentially complete in F1/F2 but absent in F3, matching the registered rung-D collapse as a source-coverage regime break rather than evidence for a stable sidecar improvement.

## Round 2 required smoke — first-failure stop (2026-09-07)

Round 2 was authorized after the detailed Round-1 readout. The exact sealed
Round-1 root and V3 store were copied to `brazil-rv-east3` NFS and reverified:
all 696 Round-1 inventory entries (505,196,221 bytes), the complete V3 store,
and both CDI inputs matched their registered SHA-256 values. On Linux, Ruff,
compileall, and all 852 tests passed; the production-axis memory test also
passed its 8-GiB gate. The two test-only Linux portability corrections are on
main at `e868e927eecc70e2d4d9701ba1a1b19ad1569cf4`.

The initial freeze attempt correctly refused current-main commit identity
because sealed Round 1 records implementation `cb6a0a5c4de202fe046dba48d9fd6c5af168dbfd`.
The entire `research/src` tree is byte-identical between that commit and main,
so the canonical Round-2 root was frozen from a clean detached checkout of the
exact Round-1 implementation:

    /lambda/nfs/brazil-rv-east3/quant-data/b3/processed/model_runs/v2_round2_cb6a0a5_20260907T143327Z

Its frozen-design SHA-256 is
`30b75d1b5bd126222ed4d318b8edccb242f84734deeb9222299b3dee55e65d8e`.
The design binds parent `b_intraday`, no external sidecars, native fresh fast
weights, no legacy checkpoint, maximum parallelism four, and protected access
false/false.

The mandatory disposable Arm-A/F1/seed-11 one-epoch smoke failed before any
epoch history, checkpoint, completed training manifest, score directory, or
registered trajectory was written. `torch.compile(fullgraph=True)` exhausted
Dynamo's recompile limit of eight as the active-name dimension changed; the
last guard mismatch was `v1_equity_slow` dimension 1, expected 128 versus
actual 137. The failure is therefore a dynamic-shape compilation defect, not a
research result. Per the first-failure rule, there was no retry, Stage P and
the registered A/B/C grid were not started, and no R2.1 or R2.2 selection was
made.

The failure-record and complete failed-root inventory SHA-256 values are
`d830cbb6a139b2c7c18b64a9adef6593095a89954cf3ba7aef5c4f9b2aaacf23`
and `f0c61411b729b9f3aeaff1cab4f133e14e02928994a0b1725bc598c0d0deeb07`.
The traceback SHA-256 is
`a2e611bcb61d677709bc36da7706fb1a8b14f95ebc08d5f221526b0a81e9341d`.
All operational logs are hash-secured in the failed root. Official validation,
the permanently spent test, and deployment remain untouched. The exact paid
GH200 instance `c484c5fd446149f0a7994e9f6d4a0d32` was terminated after the
evidence was secured. Provider inventories at
`2026-09-07T14:57:53.2767470Z` and `2026-09-07T14:57:58.6356783Z` both
confirmed that exact ID absent and contained no adjacent instance.

## Registration rev3 neutral evaluation — acceptance stop (2026-09-07)

Registration rev3 was implemented without rebuilding or mutating the sealed
V3 store. The evaluation layer now constructs the characteristic-neutral
target as a deterministic row-wise float64 OLS virtual view, keeps the prior
scaled target as a named legacy diagnostic, reports realized market beta, and
uses fixed 16-name compact padding for compiled training paths. Executable
borrow accounting is bound to the store-manifest-hashed D+1 lending archive:
the recorded `tanh(log1p(rate_percent)/2)` values are inverted exactly for
rates, while store-materialized lending balance/age determines shortability.
Missing or stale rates retain the registered uniform-floor charge. These
changes are on main through
`924b4fc4c8ecf11c2a9c403c505c16e06e0d5418`; Ruff, compile, and the complete
865-test set pass (the last fixture-only correction was rechecked in its
affected suite).

The first acceptance freeze at `ef5b893` stopped before root creation because
an obsolete guard equated the immutable store-build commit with the evaluator
commit. Commit `fc953504abe84e885880546c4250dee5480eb865` replaced that with
the correct store-manifest hash binding. Its fresh attempt is retained at
`D:\quant-data\b3\processed\model_runs\v2_development_acceptance_rev3_fc95350_20260907T163429Z`;
it wrote only the first deterministic score panel and stopped before an
evaluation when the materialized lending feature roster did not contain a
loan-rate field. No result was inferred from that partial root.

The complete classical acceptance run is sealed at:

    D:\quant-data\b3\processed\model_runs\v2_development_acceptance_rev3_924b4fc_20260907T165511Z

Its pipeline-manifest and inventory SHA-256 values are
`4fd087f6b442a82ac7e1fed4f7274513e50c5d98987f870a79a85438847d20fa`
and `9e56b56128ade40fb3b19b197553b00efc7e0a86631e20586855951e3933c1c0`.
All 15 registered baseline evaluations and the F1 GBDT integration leg are
complete, transfer chronology is clean, and official-validation/test access
is false/false. The inverse-volatility neutral IC is 0.0041771, 0.0067263,
and 0.0046084 in F1/F2/F3, each strictly below the registered absolute 0.02
engineering bound. Pooled neutral ICs for momentum, reversal-21, reversal-5,
and their registered blend are 0.0227477, -0.0126254, -0.0030471, and
0.0156703. These are integration diagnostics only because acceptance is
unsupported.

The registered legacy-identity gate compared all 75 baseline/horizon values
against sealed Round 1 and found 15 baseline-fold summary mismatches. The
underlying scaled-target payload did not change: D5 and D10 are exactly
identical. The mismatch is a readout-population defect—rev3 used per-horizon
validity for D1/D2/D3, while sealed Round 1 used the common D1--D5 validity
population for those primary horizons. The manifest therefore records
`engineering_acceptance_status=unsupported` with the sole reason
`legacy_scaled_target_ic_differs_from_round1`. Per the registration's
first-failure rule, no rev3 Round 1 or Round 2 run was started and no result was
retried. Protected data and deployment remain untouched, and no paid instance
was launched.

## Rev3 legacy-readout repair, accepted replay, and Round 1 (2026-09-07)

Commit `f567e0f6def33bfd49e5f02b4dfab953a94fb589` restores the exact
rev2 legacy scoring population without changing the rev3 neutral target. The
primary D1/D2/D3/D5 legacy rows now share the original active, finite-score,
positive-sigma, all-four-primary-target-valid population; D10 remains
per-horizon. The neutral path starts from that same rev2 population and then
intersects characteristic validity. The horizon audit records possible and
used date/name-day counts for both populations. Ruff, compileall, and all 866
tests passed before scoring.

The fresh from-scratch acceptance root is:

    D:\quant-data\b3\processed\model_runs\v2_development_acceptance_rev3_f567e0f_20260907T173213Z

Its pipeline-manifest and inventory SHA-256 values are
`dacef39764492213599fb9fb696b716583432511d1d8b8b1bfa239f824f1d7fc`
and `22ebbfaeec8306f039a7aadc42b72fd8d1c08f992c3fc7ac334b22ceead8830d`.
All 75 sealed legacy baseline/horizon comparisons are exactly identical with
zero mismatches. Inverse-volatility neutral IC is 0.0041771, 0.0067263, and
0.0046084 in F1/F2/F3, below the registered absolute 0.02 engineering bound.
All 15 baselines and the F1 GBDT integration leg completed, every entry-defect
signature is zero, hard ledger/gross bounds passed, transfer chronology is
clean, and official-validation/test access is false/false. The accepted status
remains explicitly `development_grade_inferred_actions`; verified action terms,
auction execution marks, and historically executable borrow remain unsupported
research claims.

Rev3 Round 1 is completed and sealed at:

    D:\quant-data\b3\processed\model_runs\v2_round1_rev3_f567e0f_20260907T173815Z

Frozen-design, result, access-audit, and complete inventory SHA-256 values are
`6ad0d9b5de9fce20e3b5fa952b9fbfc48fffbf49a903f0bff3dc1bc073e37f16`,
`e51a46e1ddfed7730c178268c083da2f375ed92a660bbbeac4bb3f2619264daf`,
`a2336328dd0a7e85bce1c710724abf2e287806daad335e252bb2fafbde114708`,
and `32eb364d391319d833cd649e628c541e42c5953fc0b5d188029f377658cac35f`.
The audit covers 657 artifacts and passes with clean chronology, protected
access false/false, and no deployment change. The exact registered execution
contains 15 baseline evaluations; A/B/C on F1/F2/F3; rung D only on F1/F2;
and all six data-span preview evaluations. No D/F3 result exists.

The ladder keeps A, B, and D, drops C, and designates `b_intraday` as the
Round-2 parent. Pooled neutral ICs for A/B/C/D are
0.0200034/0.0216976/0.0184769/0.0136093; D is an F1/F2-only readout. B's
neutral IC interval is [0.0139469, 0.0327263], its exact legacy IC is 0.0319429
[0.0213486, 0.0484990], and headline net excess is 1.3382 bps/day
[-6.3396, 12.3077] over 375 finite observations. C versus B worsens neutral IC
by -0.0032207 [-0.0071478, -0.0009433] and economics by -2.2929 bps/day, so C
is dropped. D versus C improves neutral IC by 0.0045720
[0.0010613, 0.0089978] and economics by 2.0140 bps/day on its registered
F1/F2 support, so D is kept, but B remains the best kept pooled neutral-IC
parent. The inverse-volatility control has neutral IC 0.0051659, legacy IC
0.0622560, and net excess 4.3574 bps/day; it is not an eligible ladder parent.

For parent B, the neutral exposure diagnostics are dominated by positive
12-to-1 momentum rank correlation (0.4611/0.4928/0.4135 across F1/F2/F3),
with beta-60 exposure -0.1532/-0.1341/-0.1830. Realized market-beta slopes are
-0.3588 (`directional`), -0.2450 (`beta_neutral`), and -0.5659
(`directional`), so the diagnostic does not support realized beta neutrality
in two of three folds. Headline net excess by fold is
3.5125/-2.4082/2.8731 bps/day. Active-name prior-20 lending-rate coverage is
0.6446/0.6929/0.7198, while active shortability is
0.6118/0.6929/0.1127; F3 shortability is therefore coverage-limited.

The data-span preview remains informational. Fine-only, 756-session decay,
and uniform-pretrain neutral ICs are 0.0216976/0.0219637/0.0212408. Decay
minus fine-only is +0.0002661 [-0.0078523, 0.0065975] neutral IC and
+2.7928 bps/day [-4.0730, 7.4039], while uniform minus fine-only is
-0.0004568 [-0.0081245, 0.0067001] and -1.3548 bps/day. Per the registration,
execution stopped before Round 2. No paid instance was used for this CPU run.

## Rev3 Round 2 completion and data-currency audit (2026-09-08)

Round 2 is complete and sealed at:

    /lambda/nfs/brazil-rv-east3/quant-data/b3/processed/model_runs/v2_round2_rev3_916ac0b_20260907T215945Z

The score-bearing implementation is
`916ac0b7e6e3ab16dea72dbf480ebb086a8981b0`; frozen-design SHA-256 is
`8ca83757f48204e94db153c3ff1b9891f1447e255510963433d7c5994d8d0b16`.
The root contains the disposable smoke, all three Stage-P seeds, and exactly
27 registered A/B/C × F1/F2/F3 × 11/29/47 main trajectories. Stage P ran
8/7/8 epochs and selected epochs 5/4/5. The main selected-epoch counts were:

| Arm/fold | Seed 11 | Seed 29 | Seed 47 |
| --- | ---: | ---: | ---: |
| A/F1 | 6 | 6 | 6 |
| A/F2 | 7 | 18 | 8 |
| A/F3 | 4 | 4 | 4 |
| B/F1 | 4 | 5 | 4 |
| B/F2 | 9 | 5 | 9 |
| B/F3 | 4 | 4 | 4 |
| C/F1 | 8 | 5 | 13 |
| C/F2 | 4 | 4 | 5 |
| C/F3 | 11 | 4 | 6 |

Compiled-graph audits passed: the separate P and F paths each used two graphs
in total, while joint Stage J correctly used two stable training graphs plus
one selection graph. The original main launcher had assumed the P/F graph
count for J and stopped after 23 completed trajectories. The recovery plan
SHA-256 `b12ef7817e45c71ac6ae2392ed8db7e60f4987371bacab64dc2544639226b03c`
contained only the four never-started C jobs; it preserved all completed
scores and changed no run, seed, split, target, or model rule.

Pooled registered readouts are:

| Candidate | Neutral IC [95%] | Legacy IC | P1 / P5 | Spread bps/holding session | Net excess bps/day [95%] |
| --- | ---: | ---: | ---: | ---: | ---: |
| Arm A, fine only | .015342 [.006382,.025507] | .037740 | .9417 / .8820 | 12.184 | 5.305 [-1.122,17.684] |
| Arm B, uniform pretrain | .027168 [.017567,.039919] | .053489 | .8709 / .7803 | 21.461 | 11.861 [3.224,22.959] |
| Arm C, decay pretrain | .022510 [.013364,.036517] | .043230 | .8393 / .7997 | 13.249 | 3.532 [-3.125,15.172] |
| Round-1 GBDT parent | .021698 [.013947,.032726] | .031943 | .7990 / .6564 | 9.724 | 1.338 [-6.340,12.308] |
| Network/GBDT rank ensemble | .029127 [.020274,.041677] | .050420 | .8431 / .7430 | 15.050 | 5.205 [-2.219,17.844] |

Arm B beat Arm A by `+0.011826` neutral IC
`[+0.006217,+0.019384]` and `+6.5564` net bps/day
`[+0.5000,+8.8891]`, so it is the registered R2.1 choice. Arm C improved
neutral IC over A by `+0.007168` but worsened point economics by `-1.7729`
bps/day; it was not designated. The ensemble beat the GBDT parent by
`+0.007430` neutral IC `[+0.002645,+0.012852]` and is the R2.2 research
designation under the registered best-IC rule. Relative to the Arm-B network,
the ensemble's neutral-IC delta was only `+0.001959`
`[-0.002241,+0.006873]`, while net economics fell `-6.6557` bps/day
`[-10.8520,-0.1627]`; that tradeoff is retained rather than hidden.

All nine aggregate and six comparator evaluations had completed before two
reporting-only defects prevented final JSON serialization: old score manifests
lacked the newly required evaluation-index metadata, and canonical Stage-P
history files are JSON arrays. Commits
`9679f040de4efe33065806909f22c89050d28009` and
`6e0554c94214f82fa660ce439c3eb424bb5519e7` added a hash-bound reporting-only
recovery that rebuilt and verified full evaluation inputs, re-derived every
three-seed rank aggregate exactly, reused all 15 evaluations, and recomputed
zero scores and zero evaluations. Commit
`2ee5334aef9bebbd2aa9088d4156a7c6919d64e4` made the seal scanner inventory
JSON-array research artifacts. Linux Ruff/compile and all 901 tests passed at
the final seal commit.

Round-2 result, access-audit, and log-inclusive inventory SHA-256 values are
`e98792213e16615c6e44ba3036ed829100940eeef0b8a9a175a9a2b2f752f9a0`,
`18c4f83cac0411ef215cfc4bde867329b41bd254b39c999ec664c4dc4a0b1b5b`,
and `39ee66e731aa51bf3859a70c974abdf2ee69301dff01f05fa1dfb626d3d755ce`.
The inventory contains 579 files. The access audit inspected 134 JSON
artifacts, passed, and records official-validation/test access false/false,
clean transfer chronology, and no deployment change.

After the root and all operational logs were secured, Lambda accepted
termination of exact paid GH200 instance
`aed9c4a5d2e94ee8bd667ee52c104945` (`gpu_1x_gh200`, `us-east-3`, IP
`192.222.50.196`) at `2026-09-08T01:49:50Z`. Provider inventories at
`2026-09-08T01:51:53Z` and `2026-09-08T01:51:59Z` both confirmed that exact
ID absent and returned zero active instances. No adjacent instance was touched.

The required CPU-only source-currency audit is:

    D:\quant-data\b3\processed\model_runs\v2_data_currency_audit_2ee5334_20260908T015300Z\data_currency_audit.json

Its SHA-256 is
`eb8a1370d1a4929657c9595bcc2ac49b40c85070c36230d2fdf1fdbcb483977f`.
It hash-verified every sidecar archive against the sealed V3 store manifest and
read only manifest/capability metadata plus the sidecar archive indexes; no
2025/2026 model-session array was opened. Monthly counts from 2023-01 are in
the JSON. The durable source cutoffs are:

| Family | Store capability | Last available date | 2023+ archive name-days | 2023+ feature-valid name-days |
| --- | --- | --- | ---: | ---: |
| Options | source-semantics unavailable | 2024-07-01 | 49,033 | 49,033 |
| Lending | enabled | 2024-07-01 | 35,429 | 35,427 |
| Oddlot | enabled | 2024-06-28 | 130,445 | 130,445 |
| Rebalance | source-semantics unavailable | 2024-06-28 | 45,978 | 45,978 |
| Events | source-semantics unavailable | 2024-12-30 | 64,035 | 64,035 |
| Fundamentals | source-semantics unavailable | 2024-06-28 | 50,613 | 50,613 |

The indexed raw lending PDFs stop at report date 2024-06-28, so no raw lending
file for 2024-07 onward is present. The raw 2024 annual COTAHIST archive is
present and extends beyond July, so oddlot can be rebuilt beyond its current
2024-06-28 derived cutoff without obtaining a new raw price archive. This
audit changes no score, designation, protected access state, or deployment.

## Round-2 diagnostic readout before Round 3

The requested machine-readable readout is `round2_readout_detail.json`, SHA-256
`ec84b540fc5cf0b4d7bbdef5e47b3be2798a8d8133be14125b4350dbf3a8781a`.
It was extracted read-only from the sealed Round-2 result
`e98792213e16615c6e44ba3036ed829100940eeef0b8a9a175a9a2b2f752f9a0`
and inventory
`39ee66e731aa51bf3859a70c974abdf2ee69301dff01f05fa1dfb626d3d755ce`.
All 136 mirrored JSON artifacts matched their sealed hash sidecars. No model,
score, or evaluation was recomputed. The only session arrays opened were the
hash-verified `active` and `fast_present` slices for the registered 2023--2024
F1/F2/F3 windows; no 2025/2026 session was opened. Official-validation/test
access remained false/false and deployment remained unchanged.

The predeclared parent check fails. “Within a third” is applied literally as
`abs(lending net - headline net) / abs(headline net) <= 1/3`. Arm B misses the
absolute 0.25 volatility-exposure bound in all three folds, misses the absolute
0.30 realized-beta bound in all three folds, and its lending-cell net is not
within one third of headline in either binding fold. F3 is reported but is not
binding because lending coverage is limited there.

| Fold | Vol exposure [95%] | Within ±.25 | Realized beta | Within ±.30 | Headline net | Lending net | Relative distance | Lending pass |
| --- | ---: | :---: | ---: | :---: | ---: | ---: | ---: | :---: |
| F1 | -.429788 [-.483945,-.396760] | no | -.559660 | no | 11.2659 | 3.6549 | 67.56% | no |
| F2 | -.335214 [-.385725,-.254109] | no | -.330522 | no | 4.8538 | .1689 | 96.52% | no |
| F3 | -.506668 [-.527354,-.473761] | no | -.896435 | no | 19.2836 | -2.0394 | 110.58% | n/a |

Per the rule fixed before this extraction, Round 3 must therefore begin with
the rev-4 evaluation change: nonlinear neutralization using vol-decile and
beta-decile dummies plus a beta-hedged ledger cell, followed by reruns of Round
1 and the Round-2 arms. Arm B is not accepted as the Round-3 parent from the
rev-3 evidence, and no Round-3 registration was written by this extraction.

### Candidate fold diagnostics

The table reports point estimates; the full 95% exposure intervals, support
counts, and labels are in the JSON.

| Candidate | Fold | Vol-20 | Beta-60 | Log volume-20 | Momentum 12-1 | Return-5 | Realized beta | R² |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Arm A | F1 | -.642716 | -.512222 | .231809 | .615755 | .135826 | -.670671 | .5154 |
| Arm A | F2 | -.238712 | -.275360 | .025047 | .712898 | .119295 | -.195496 | .0659 |
| Arm A | F3 | -.656363 | -.622655 | .270619 | .709773 | .216177 | -.948758 | .5923 |
| Arm B | F1 | -.429788 | -.347891 | .134637 | .748851 | -.080819 | -.559660 | .3587 |
| Arm B | F2 | -.335214 | -.328942 | .202269 | .722763 | -.031392 | -.330522 | .1757 |
| Arm B | F3 | -.506668 | -.491428 | .330827 | .773772 | -.038663 | -.896435 | .5695 |
| Arm C | F1 | -.382457 | -.309385 | .126939 | .618553 | .069751 | -.630349 | .4405 |
| Arm C | F2 | -.231544 | -.311753 | -.043571 | .738630 | .092074 | -.236850 | .1132 |
| Arm C | F3 | -.441640 | -.438470 | .328095 | .725167 | .038161 | -.801821 | .5037 |
| GBDT | F1 | -.154918 | -.153236 | -.106627 | .461146 | .025962 | -.358759 | .3015 |
| GBDT | F2 | -.172366 | -.134110 | -.073871 | .492773 | -.021598 | -.244978 | .1507 |
| GBDT | F3 | -.240064 | -.183007 | .074384 | .413484 | -.029445 | -.565906 | .5070 |
| Ensemble | F1 | -.346833 | -.293583 | .023880 | .717014 | -.037396 | -.445017 | .3453 |
| Ensemble | F2 | -.287475 | -.261961 | .080138 | .697496 | -.031047 | -.331064 | .2036 |
| Ensemble | F3 | -.440387 | -.399565 | .241915 | .698382 | -.037594 | -.832715 | .5923 |

Headline and lending-sidecar economics are:

| Candidate | Fold | Headline net / gross label | Lending net / gross label | Settlement label |
| --- | --- | ---: | ---: | --- |
| Arm A | F1 | 4.1434 / within | 2.5989 / under | resolved |
| Arm A | F2 | .6933 / within | -3.5352 / within | resolved |
| Arm A | F3 | 10.9406 / within | .7556 / under | resolved |
| Arm B | F1 | 11.2659 / within | 3.6549 / under | resolved |
| Arm B | F2 | 4.8538 / within | .1689 / within | resolved |
| Arm B | F3 | 19.2836 / within | -2.0394 / under | economics unresolved |
| Arm C | F1 | -3.2494 / within | -7.0290 / within | resolved |
| Arm C | F2 | .6637 / within | -3.2765 / within | resolved |
| Arm C | F3 | 12.9529 / within | -2.8621 / under | economics unresolved |
| GBDT | F1 | 3.5125 / within | 4.7092 / within | resolved |
| GBDT | F2 | -2.4082 / within | -5.8482 / within | resolved |
| GBDT | F3 | 2.8731 / within | -10.1318 / under | economics unresolved |
| Ensemble | F1 | 6.0479 / within | 6.6048 / under | resolved |
| Ensemble | F2 | -2.3008 / within | -2.1009 / within | resolved |
| Ensemble | F3 | 11.7112 / within | -8.4018 / under | economics unresolved |

For F3 the active-name-day shortable fraction is `0.112668` and the high-vol
quartile shortable fraction is `0.104657`, versus `0.611838/0.495963` in F1
and `0.692931/0.589325` in F2. This is the registered F3 lending-coverage
caveat, not an omitted negative result.

### Arm-B horizon and quality shape

| Fold | Horizon | Neutral IC | Legacy IC | Price IC | Spread bps/holding session |
| --- | ---: | ---: | ---: | ---: | ---: |
| F1 | D1 | .017593 | .036744 | .035368 | 18.691 |
| F1 | D2 | .026861 | .050016 | .048040 | 16.700 |
| F1 | D3 | .026365 | .050782 | .051178 | 16.113 |
| F1 | D5 | .029227 | .054304 | .054870 | 12.458 |
| F1 | D10 | .020856 | .048000 | .045768 | 8.082 |
| F2 | D1 | .017406 | .036960 | .039170 | 24.552 |
| F2 | D2 | .020922 | .042409 | .041070 | 15.892 |
| F2 | D3 | .026243 | .050415 | .050716 | 16.113 |
| F2 | D5 | .027240 | .053612 | .053636 | 16.096 |
| F2 | D10 | .019807 | .056218 | .056988 | 16.748 |
| F3 | D1 | .019582 | .042114 | .046765 | 28.653 |
| F3 | D2 | .028493 | .053721 | .057985 | 28.843 |
| F3 | D3 | .037929 | .071819 | .073449 | 30.499 |
| F3 | D5 | .047530 | .097680 | .093900 | 31.863 |
| F3 | D10 | .073605 | .151904 | .146254 | 37.078 |

The Arm-B neutral IC by causal liquidity quartile and survival audit stratum is:

| Fold | Q1 | Q2 | Q3 | Q4 | Survives | Delisted |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| F1 | .004847 | .024709 | .028542 | .052685 | .030893 | -.000124 |
| F2 | .036189 | .013969 | -.008835 | .048679 | .018395 | .045081 |
| F3 | .049391 | .026172 | .019593 | .034862 | .026604 | .030391 |

The JSON contains the complete five-candidate per-horizon tables, all
incremental-horizon IC rows, all requested quality rows, and the exact support
counts. The network fast gate's mean activation was not logged in any sealed
artifact and is recorded as JSON `null`, not inferred. The hash-verified
`fast_present` coverage is:

| Fold | Fast-present active name-days | Active name-days | Share | Gate activation |
| --- | ---: | ---: | ---: | --- |
| F1 | 14,296 | 24,598 | 58.1185% | not logged |
| F2 | 13,873 | 23,568 | 58.8637% | not logged |
| F3 | 14,106 | 23,059 | 61.1735% | not logged |

### Exact-common-population parent comparisons

| Pair | Neutral-IC delta [95%] | P1 delta | P5 delta | Headline-net delta [95%] | Lending-net delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| B minus GBDT | .005471 [-.003678,.014582] | .071898 | .123890 | 10.5228 [2.2233,18.6515] | 4.3817 |
| Ensemble minus B | .001959 [-.002241,.006873] | -.027780 | -.037304 | -6.6557 [-10.8520,-.1627] | -1.9298 |

The registered IC, persistence, and headline-net deltas above are copied from
the sealed exact-common-population paired readouts. The lending-net point
deltas are same-date differences of sealed lending-cell daily rows; no new
interval was computed. Fold lending-net / realized-beta-slope deltas are:

| Pair | Fold | Lending-net delta | Realized-beta delta |
| --- | --- | ---: | ---: |
| B minus GBDT | F1 | -1.0543 | -.200901 |
| B minus GBDT | F2 | 6.0171 | -.085544 |
| B minus GBDT | F3 | 8.0924 | -.330529 |
| Ensemble minus B | F1 | 2.9498 | .114642 |
| Ensemble minus B | F2 | -2.2698 | -.000542 |
| Ensemble minus B | F3 | -6.3623 | .063720 |

No pooled realized-beta statistic existed in the sealed result, so it remains
JSON `null`; it was not synthesized from fold slopes.

## Rev4 constructed-book acceptance stop (2026-09-08)

The rev4 evaluation implementation and preregistration were frozen in commit
`1c62085bca4c0d5381aa6884aed57a3aef4c6f6a`. A score-free operational loader
repair, which permits a hash-verified BOVA11 history to extend outside the
consumer store calendar while continuing to reject noncanonical dates inside
that calendar, was committed as
`3242ad3910efcf748b9844ad0c5480727eacb886`. Ruff, compileall, and all 887
tests passed before the accepted replay.

Rev4 computes the nonlinear neutral target as a deterministic virtual view of
the immutable V3 store: ten volatility-decile dummies, five beta-quintile
dummies, and linear rank-Gaussian log ADV, with no explicit intercept. Rows
with 20--39 observations use and flag the registered rev3 linear fallback;
rows with fewer than 20 observations remain invalid. The headline ledger uses
executable lending shorts, volatility-quintile-balanced entry, and a BOVA11
beta hedge outside the equity name caps. The fixed-width COTAHIST source shows
BOVA11 continuously as `TIPREG=01`, `CODBDI=14` over the registered period;
the document's requested BDI02 binding would retain only 92 observations in
2019, so the preregistered canonical identity is BDI14. The sealed hedge
artifact is
`D:\quant-data\b3\interim\external\bova11_hedge_close_v1_2009_2024_20260908T115000Z`;
its manifest and parquet SHA-256 values are
`858c6fb07e234d8c28219a72efa29775d83ba334f9b457ed24317ab7d270544d`
and `4aa998afb26558c9c2d1cbf9374a6a3881e04fd3a322b769d0c2d6950371a7e9`.
The existing sealed lending archive has no BOVA11-ISIN rows, so the hedge
borrow charge honestly uses the registered 2% floor.

The 16-book CPU acceptance is sealed at
`D:\quant-data\b3\processed\model_runs\v2_development_acceptance_rev4_3242ad3_20260908T131920Z`.
Pipeline-manifest and inventory SHA-256 values are
`175e95ec91e38d546c2857f53c05274efb793e78accc7ae2c55125b375763102`
and `0f16f41c93a896c96e522bccc52f5252e71763e1156638dd7214fece78293d5b`.
It completed with `engineering_acceptance_status=unsupported` and protected
access false/false. The inverse-volatility neutral-IC gate itself passed at
`-0.00004450`, `0.01093123`, and `0.00229996` for F1/F2/F3. Acceptance stopped
on 15 volatility-occupancy bounds, three F3 gross bounds, and one legacy
identity check. No Round-1 root was created, no Round-2 or GPU work was run,
and no paid instance was launched.

The occupancy failures expose a construction error that must be resolved in a
new registration rather than repaired after this score: the implementation
forms the long and short candidate pools from the global top and bottom halves
before applying volatility-quintile quotas. A score monotone in volatility can
therefore never source both sides from every quintile. A future revision should
rank long and short candidates within each volatility quintile across the full
eligible population, while preserving global long/short disjointness and the
registered spillover rules. The F3 gross failures remain genuine consequences
of sparse executable-short coverage; they were not relaxed.

The legacy identity failure was a caller/reference-binding mistake, not a
calculation change. This acceptance was given the rev3 Round-1 root
`v2_round1_rev3_f567e0f_20260907T173815Z`, while the rev3 acceptance's sealed
legacy comparison root is
`D:\quant-data\b3\processed\model_runs\v2_round1_81fe0cb_20260907T023339Z`
(result SHA-256
`ca39f11340f13956dca11da7fb6b3a8fa0a38f8a79cb558387d81aff26bf57a0`).
The scored acceptance was not rerun because the independent occupancy and
gross bounds already require a new design.

For a later lending-sidecar rebuild, 127 official B3 BDI Chapter 05 PDFs for
2024-07-01 through 2024-12-30 were downloaded without modifying any immutable
source into
`C:\quant-data\b3\raw\b3\bdi_lending_open_balance\pdf_20240701_20241230`.
The download-manifest SHA-256 is
`f1754e98ad2cd890375a6038907670a4552e96f8c3c232c3bc3f136d7fcce03b`.
They have not been parsed and the lending sidecar/store has not been rebuilt;
doing so would change the immutable data contract and belongs in the next
explicitly registered round.
