# Pass-5 implementation status

This is a **partial implementation, stopped before the first rev-4f score replay**.
It is not a re-baseline result or authorization to start Round 3.

The prior cleanup is committed separately as `fb6fc92`; the initial rev-4f
registration was committed as `0302b27` before new derived beta output.
The rev-4e registration is retained verbatim inside
[the rev-4f amendment](../research/preregistrations/v2_round1_round2_rev4f.md).
Its own rev-4f JSON block describes the implemented hedge/borrow/retention changes.

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
- Hedge quantities are fixed before current-session prints. The target uses held
  prior marks, pending/planned entries and planned exits; threshold and cap use
  prior NAV. BOVA11 prior close sets quantity. Hedge orders, fills and missed-print
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

The example in §1.1 contains an arithmetic typo: for notionals (-1,+1) and raw
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

## Required clarifications and remaining work

**§1.3 is not implemented.** The existing retrospective-action decision path
remains a known defect. The response asks both for a decision-known path and for
new-entry blocking on the inferred event day. The current inferred event flag
itself can depend on the later close. Blocking on that retrospective flag leaks
event existence even if q/d are hidden. Clarification requested: use only
evidence available at 15:45 for decision-time uncertainty, then account for
retrospective entitlements and unit changes after the close. The joined
builder/evaluator/ledger action mutation fixture remains required.

That repair must preserve opening-position entitlements when an event-day exit
fills. Applying q/d indiscriminately to end-of-day holdings would drop cash owed
to positions sold that day and could grant cash to new entries that were not
entitled. A split also changes the relationship between decision-unit quantity
and executed shares. Existing split, cash, successor, unresolved-entry,
any-print exit and terminal tests remain the baseline for the correction.

**§1.6 literal non-ledger equality conflicts with §1.1.** Fixing slow diagnostic
validity changes prior-feature hashes and can change exposure diagnostics. The
new economic sidecar also adds explicit input provenance. The current replay
identity checks remain strict and will refuse this difference; no broad
projection exemption was added. Clarification requested: retain exact score,
target, primary population and IC equality, and explicitly enumerate corrected
diagnostic/provenance fields in the before/after report. Do not claim a successful
literal bit-identical replay unless that is actually demonstrated.

**The Round-2 sealed panels are not locally available.** The recorded root is
`/lambda/nfs/brazil-rv-east3/quant-data/b3/processed/model_runs/v2_round2_rev4e_2b40b24_20260908T202100Z`.
A local mirror or accessible source is needed. No paid instance was launched
to obtain it. A rev-4f Round-2 replay entrypoint and its conclusion table remain
to be implemented after the replay contract is resolved.

**The referenced Round-3 document was not located.** The attachment delegates
its adoption rule, liquidity-screened readout and 2025 rule to that document.
Its path or contents are required before those rules can be implemented faithfully.

After resolving these items, preserve the specified order:
action repair and joined test → sealed Round-1 and Round-2 replay/report →
CPU execution sweep/report → M1 return-consistency coverage →
one bounded store rebuild and 16-book acceptance → Round 1' →
stop for Gabriel's GPU go.

The intraday return-consistency repair, rolling-support/activity changes,
lending/oddlot rebuild, execution sweep, Round 1', GPU gate/gradient logging,
fast ablation and Round-3 runs have **not** been performed. No Round-4 hypothesis
was started. No 2025/2026 feature or outcome payload was used and no deployment
changed. New economics and designation conclusions are unavailable.

## Verification

Targeted ledger/beta/evaluator checks passed (115 tests), as did the pipeline
fixtures (26 tests), research-round fixtures (26 tests), and the complete T24
store/build/score/relocation/retained-replay acceptance fixture (61.50 seconds).
The additional invalid-slow diagnostic adapter variants both passed.
The full suite passed: **904 tests in 372.80 seconds**. After the final reporting-label
cleanup and the additional invalid-slow fixture variant, six targeted evaluator,
adapter and registration checks passed. Ruff, compileall, registration/code parity,
and `git diff --check` passed. The real beta sidecar also passed the full reader
hash/source/axis verification. These checks do not substitute for the still-pending
joined action-causality test or a sealed-score re-baseline.
