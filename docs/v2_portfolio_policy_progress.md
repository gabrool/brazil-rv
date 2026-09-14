# Cash-aware portfolio program: continuation state

Status: initial local engineering passed; forecast implementation ready for
committed-host freeze. No new research fit or paid instance launched yet.

Read research/preregistrations/v2_portfolio_policy.md at each stage transition.
The user authorized the full execution reassessment program, cash and compute.
Accepted source: docs/v2_data_inputs.json. Historical A–C source and recovery:
docs/v2_post_data_recovery.json and docs/v2_post_data_final_inventory.json.

## Acceptance checklist

- [x] Initial local allocation/gradient and causal share/cash accounting acceptance.
- [ ] Frozen source-bound forecast/cache plan, compatible A–C reuse.
- [ ] C6 repaired representation bridge, full chronological forecast cache.
- [ ] Chronological optimizer and three-seed learned policy experiments.
- [ ] Paired screen/remaining-fold/continuous/cost sensitivity readouts.
- [ ] Result-dependent decision on conditional architecture/objective follow-ups.
- [ ] Full review, artifact recovery/hash verification, commit/push, paid shutdown.

Operational authority is C:/Users/gabri/Downloads/Work in CBrazil-RV..txt.
Use ops/lambda-gh200.ps1 -Mode Launch -IUnderstandBilling only, after committed
main is synchronized. No forward capture, no 2025/2026 consumer access.

See docs/v2_PORTFOLIO_POLICY_ENGINEERING.md for 121 targeted passing checks,
controlled independent accounting parity and the real two-date parent smoke.
Actual-data surrogate replay and the full sequential timing/gradient diagnostic
remain pending. Modules: execution/allocation.py, execution/portfolio_account.py,
stateful_ledger.py's portfolio callback, and v2/portfolio_program.py.

The six reusable S0/TE parent checkpoints and manifests plus frozen_design.json
have been restored and hash-verified locally under the source root derived from
docs/v2_post_data_recovery.json. The full original root remains on persistent
Lambda storage. No historical artifact was changed.

Host sequence: portfolio_program freeze; plan --phase parents and run_many;
plan --phase prelude and run_many; plan --phase forecasters and run_many.
Use max-parallel 6. There are 3 new P, 9 parent forecast panels, 102 new F and
24 reused F. The new run root and exact instance receipt must be recorded here
immediately after launch. Continue controller/cache implementation locally while
independent forecasters run; preserve the frozen host training revision.
