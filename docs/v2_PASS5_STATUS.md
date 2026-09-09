# Pass-5 implementation status

Both rev4f replays completed and are sealed: **18 Round-1 panels and 12 Round-2
panels, all protected fields bit-identical, headline gates passed**. See the
[Round-1 report](v2_PASS5_REBASELINE.md) and
[Round-2 report](v2_PASS5_ROUND2_REBASELINE.md). The six lending fields and the
single history-age provenance hash were explicitly approved in separate amendments.
The unrounded history-age decoder and all quality-stratification fields remain
unchanged. Both earlier stopped roots remain immutable historical evidence.

Arm B remains the network choice and the ensemble remains the overall designation.
The CPU execution sweep is complete. The user clarified that the
liquidity readout attributes the original trades, without replacement names.
Nearest-integer history-age canonicalisation is registered for Round 1' on the
rebuilt store, not the sealed-panel replays.

The user subsequently authorized autonomous completion while absent, including
the recommended resolution of questions and the GH200 stage after successful CPU
checks. The effective Round-3 registration records that authority, the unchanged
holdout boundary, and the required artifact-copy/termination sequence.

The intraday repair is implemented for the archive coverage audit: return rather
than level consistency, separate daily/intraday action clocks, observed-support
rolling estimators and their metadata, M1-internal to-close endpoints, bounded
source decoding, and preserved canonical identity metadata. Ninety focused tests
pass, including joined builder causality and provider invariance. Both existing
production-axis memory tests passed. Real-data coverage and the new store build
are still pending; no repaired-model result has been observed.

The bounded odd-lot derivation reuses the validated annual COTAHIST parser,
retains exact ISIN identity and raw BRL volumes, and publishes on the next session.
The new lending archive's `annual_taker_rate` is consumed directly in decimal
units. The independent native-fast audit also filters its raw M1 date range before
decoding. Thirty-two targeted sidecar/archive/native-fast tests pass.

The prior cleanup is committed separately as `fb6fc92`; the initial rev-4f
registration was committed as `0302b27` before new derived beta output.
The rev-4e registration is retained verbatim inside
[the rev-4f amendment](../research/preregistrations/v2_round1_round2_rev4f.md).
Its rev-4f JSON block includes the clarified order, action and replay contracts.

## Implemented and exercised

- Economic BOVA11 beta: intercept OLS on 60 completed sessions ending t-1,
  minimum 40 adjacent valid return pairs, Blume 0.67/0.33, clipping to [-1, 3].
  Neither missing equity observations nor BOVA11 gaps are bridged.
- Hash-bound derived beta arrays, masks, date/security axes, source-store identity,
  BOVA11 identity, implementation hash and annual distributions. Requests for
  2025/2026 are refused before payload access.
- Ledger arguments `hedge_beta` and `hedge_beta_valid` replace the model-feature
  argument. Fallback uses the last valid beta for at most 20 sessions, then 1.0.
  The adapter supplies the preceding 20 sessions at a fold boundary. Missing beta
  no longer excludes a name from entry selection. Fallback sessions are reported.
- Hedge notionals are fixed before current-session prints. The target uses held
  prior marks, pending/planned entries and planned exits; threshold and cap use
  prior NAV. BOVA11 prior close records the reference. Hedge orders, fills and missed-print
  cancellations are recorded. Execution imbalance is carried to the next decision.
- Conditional last-mark settlement intentions are also fixed before a possible
  final print, then expire if a print makes settlement unnecessary. The ten-session
  convention and accounting fixtures are retained.
- Pending entries leaving their current-quintile side retention band cancel with
  reason `band_exit`; the switch-off fixture retains the previous slot behavior.
- Equity borrow, registration fee and hedge borrow use separate compounded daily
  accrual. The nine cost/availability cases all preserve headline construction:
  2/4/7 bps by balance/strict/open. The 4-bps rows retain the canonical
  `borrow_balance`, `borrow_strict`, `borrow_open` names without duplicate runs.
  The unused unhedged legacy-strategy grid and its annual-rate label are removed.
- Extracted diagnostic slow fields respect `slow_valid`. Realized beta against
  BOVA11 is reported alongside the unchanged equal-weight-universe diagnostic.
  Evaluation schema is V16; new freeze/validation commands require beta bindings.

The example in Â§1.1 contains an arithmetic typo: for notionals (-1,+1) and raw
betas (1,1.2), the hedge is -0.20 NAV; Blume betas (1,1.134) imply **-0.134**,
not -0.194. The specified formula is implemented and tested.

## Derived artifact

Root: `D:\quant-data\b3\processed\v2_hedge_beta_rev4f_20260908T225248Z`

Manifest SHA-256:
`f5fa41536740ca412550cb81d2b44c9541e338dc060d674ab8ea270edd7be712`

Source store:
`v2_daily_store_8021e42_20260906T202315Z`,
manifest `deb9ca8449c5b9a83bf25ac19218069c006e183361b6f6ba836717ade63b491b`.

Source BOVA11 manifest:
`858c6fb07e234d8c28219a72efa29775d83ba334f9b457ed24317ab7d270544d`.

The sidecar spans 3,717 sessions, 2010-01-04 through 2024-12-30, and 933 ISINs.
It has 986,957 valid name-days. During 2024, 46,601 of 46,627 active name-days
are valid (99.94%). Median beta is 1.05315; 80.08% of valid active observations
are in [0.5, 1.5]. The 5th/95th percentiles are 0.57235/1.93712.
[Full manifest and annual distributions](v2_pass5_hedge_beta_evidence.json)
are copied into Git for review. This is input-only evidence, not a measured
change in model edge. The canonical store and sealed results were not rewritten.

## Clarifications implemented; both replays completed

The joined inferred-action builder -> evaluator -> ledger causality test passes:
changing only the event-day close changes inferred cash terms but no event-day
intention. Entries and ordinary hedge trades carry notionals. Exits, trims and
terminal liquidations carry position fractions of converted opening inventory.
Partial fractions rebase after fills and survive conversions. Terminal liquidation
supersedes a still-partial trim. Same-day action cancellations and round-lot sizing
are removed; only prior-session uncertainty may block or cancel entries.

The adapter supplies the previous session's uncertainty at a fold boundary. The
registration declares recomputed diagnostics/provenance, requires their presence,
reports before/after and preserves all other identities. Round 1 and Round 2 have
CPU replay commands that reuse sealed panels. Replays stop on invariant differences,
D1-D5 defects, mean gross outside [1.5,2.25], or mean unresolved/stale inventory >=2%.
The explicit notional rule supersedes the impossible requirement for unchanged
fixed-share economics when the fill price differs from the reference price.

Round 2 is recovered locally and fully hash-verified; see
[v2_round2_host_copy_evidence.json](v2_round2_host_copy_evidence.json).
No paid instance was required. The first reversal_5/F1 replay stopped on six
unregistered high-volatility lending-coverage fields. The user approved that exact
amendment, and a fresh Round-1 replay completed under it. Its result SHA-256 is
`415aae05aed3df1a21f20d87fbe2b06827df0fec317af58b4bf0f8a84ae35839`;
inventory `9940b01076677f19ea8dbf1cca0af2991e2899494e491c92d666d0ceb0c7aafb`.
Round 2 was frozen under the same commit and stopped at its input-identity check,
before any new evaluation. The stopped inventory is
`9d90d85129368f571eb6daf43185eb35aad945e90e1e46f63df36248898d7ef6`.
The subsequent approved fresh replay completed under `564efc8`. Its result hash is
`808369c99b14eebcf3d6f55d67b888c8a2e4cb536965f21adae9c17d282d25a6`;
inventory `0f8b3bb6116c0fe285671fd77666fa50ea9690d8ca707a70d3ba3c5dd69767f6`.
The R3.1 runner is implemented. Its first baseline panel stopped because the
supplied missing-score carry changed theta=1 risk-trim decisions. A baseline-only
isolation exactly reproduced the sealed book when carry was disabled. The user
explicitly clarified that theta=1 retains current behavior and carry applies only
at theta<1. [Evidence and preserved roots](v2_R31_BASELINE_STOP.md) record this.
The fresh `29fb045` freeze reproduced all 12 baseline panels / 48 scenario daily
tables exactly, then stopped in the first inverse-sigma cell on 32 D4 flags in
GBDT/F1's open-borrow scenario. A stale partial exit in one overweight name was
incorrectly vetoing unrelated valid-size entries. The minimal candidate-specific
name-cap correction is prepared and passes 151 targeted tests. The stopped root,
trace and partial results are [reported here](v2_R31_D4_STOP.md). The user explicitly
approved the tested correction and a fresh R3.1 restart; the effective registration
records it. No policy was selected from the stopped partial grid.
The approved restart completed all 240 panels / 960 scenario ledgers. Every
D1-D5 and headline gate passed; all 48 sealed baseline daily tables are exact.
The registered joint-first rule selected unsmoothed D3/D5/D10, equal sizing and
buffer 9 per quintile. B's paired gain is 3.2533 [-0.4199, 6.5484] bps/day;
the horizon-only buffer-6 change has a larger gain but lower selection priority.
The label is `execution_parameter_selected_in_sample`. Full tables, methods and
sealed hashes are in [the completed sweep report](v2_R31_EXECUTION_SWEEP.md).
Next: intraday coverage repair, one development-only store build,
16-book acceptance, Round 1' CPU, then stop for the user's GPU go.

Validation before replay: Ruff and compilation pass; the full research suite passed
908 tests in 467.91 seconds. The final focused run passed 100 tests, including
the added invariant-exemption check and the joined corporate-action causality test.
