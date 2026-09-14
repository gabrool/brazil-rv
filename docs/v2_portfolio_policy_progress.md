# Cash-aware portfolio program: continuation state

Status: all 105 new forecast fits and nine preludes completed successfully;
policy cache preparation and subsequent CPU experiment chain started.
Exact paid instance: ee625bd03bdc41fd9ba676de6e171511, IP 192.222.51.211.
Frozen forecast checkout: 49d3c9a53b74277f269c8248c09d65e1cb481dd6.
Persistent run root:
/lambda/nfs/brazil-rv-east3/quant-data/b3/processed/model_runs/v2_portfolio_49d3c9a_20260914T143600Z
Completed forecast dispatcher PID 5119; log and PID receipt under model_runs/_ops/:
portfolio_20260914T143600Z.log and portfolio_20260914T143600Z.pid.
External dispatcher /home/ubuntu/run_portfolio_forecasts.py SHA-256:
a0352bfcad8328425b377bec9c0fd23eef2bd5d116129d4dc15f225dc81a526d.
Do not start another dispatcher or modify the frozen forecast checkout.
Bootstrap verified clean 49d3c9a and idle GH200 before dispatch.

Initial and real-market local engineering passed. Approved launcher
started at 2026-09-14 12:58 UTC from 17465f7, then exited at 13:18:58 UTC when
its local log could not be written during a temporary full-disk condition.
No instance had been obtained by that first watcher. Free space recovered
after the memory-heavy local check ended. The approved launcher was restarted at
13:28:26 UTC from clean pushed commit 49d3c9a; obtained the above instance,
which became active at 14:32:31 UTC. Both launcher processes have now exited.
Existing five-minute heartbeat brazil-rv-gh200-launch-monitor is active and bound
to this program. Do not start a second watcher. Local policy work continues.

Read research/preregistrations/v2_portfolio_policy.md at each stage transition.
The user authorized the full execution reassessment program, cash and compute.
Accepted source: docs/v2_data_inputs.json. Historical A–C source and recovery:
docs/v2_post_data_recovery.json and docs/v2_post_data_final_inventory.json.

## Acceptance checklist

- [x] Initial local allocation/gradient and causal share/cash accounting acceptance.
- [x] Frozen source-bound forecast/cache plan, compatible A–C reuse.
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
Actual-data surrogate replay and 32/64-session gradient acceptance now pass:
docs/v2_portfolio_policy_engineering_evidence.json and the implementation
resolutions in research/preregistrations/v2_portfolio_policy_implementation.md.
The cache/causal-input modules, resumable three-seed policy fitter, matched
fold/continuous readouts, cost/funding stresses and offline CPU batch are built
and locally tested. No new financial result is available yet.

The six reusable S0/TE parent checkpoints were restored and hash-verified locally.
Three extracted TE weight copies were subsequently removed to free disk; all
original verified recovery archives and manifests remain intact. The S0 copies
and frozen_design.json remain at the source root derived from the recovery
pointer. The full original root remains on persistent Lambda storage. Receipt:
D:/quant-data/b3/interim/portfolio_policy_duplicate_removal.json.

Host sequence: ops/run_portfolio_forecasts.py --root ROOT, running against the
frozen host checkout (49d3c9a from the current launcher). This script can be copied
outside that checkout and executed with its research Python/PYTHONPATH. It combines
the independent 3 C6 P and 102 F in one six-job queue, then scores 9 P preludes.
C6 F inherits existing S0 P, so it does not depend on the new C6 P. Reuse 24 F.
Record the exact instance ID, IP and new persistent run root when launch finishes.
Do not update the running forecast checkout while fits execute.

After the forecast source is complete, use the latest committed policy code in
a separate clean checkout with the existing research Python and PYTHONPATH:
portfolio_batch prepare, policies --workers 12, continuous --workers 3, summarize,
each with --root ROOT. Prepare families sequentially to share the economic/beta
build, then one CPU process per fold loads data once and fits all three seeds.
Read research/preregistrations/v2_portfolio_policy.md again at each transition.
Review the registered conditional architecture/objective follow-ups after results;
completion also requires that decision, full review, recovery and exact shutdown.

Launch correction: first dispatcher PID 4827 exited before any fit because the
operator precreated its run root for logging; freeze requires a new directory.
The failed 143500Z directory contains only the error log/PID. Restarted at the
fresh 143600Z root with logs outside it; frozen design and 105-fit plan exist.

## CPU policy transition

Forecast completion record: forecast_program_result.json (3 P, 102 F,
24 reused F, nine preludes). All 114 new run manifests completed.
Independent clean policy checkout, revision 74c4021:
/home/ubuntu/portfolio-workspace/quant/b3-quant
Set BRAZIL_RV_ROOT=/home/ubuntu/portfolio-workspace and PYTHONPATH to its
research/src; use original forecast research/.venv/bin/python and policy
checkout data_roots.lambda_us_east_3.json. Original frozen checkout preserved.
Shell dispatcher PID 61124 started at 2026-09-14T16:52:49Z, child prepare 61126.
Script: /home/ubuntu/run_portfolio_policies.sh. Log under model_runs/_ops/:
portfolio_policies_20260914T165400Z.log (filename differs from actual start time).
Sequential chain: prepare, policies (12 workers), continuous (3), summarize.
Check this existing chain before any restart. Initial PID 60817 exited during
import because the separate checkout lacked the required workspace layout;
fixed layout and explicit root before any policy fitting. Failure log retained
as portfolio_policies_20260914T165200Z.log. No model or financial contract change.
## Numerical repair before policy conclusions

The initial CPU batch failed; no aggregate policy result is accepted. Frozen
forecasts/cache/economics remain valid. All initial policy outputs are retained
under rejected_policy_numerics_74c4021 before fresh policy fitting. Resolutions
are registered in v2_portfolio_policy_implementation.md: binary validity units,
NAV-based marked P&L state (removes tiny-cost-basis Jacobian), cash exemption from
legacy minimum-gross unresolved flag, full-record audit schema inference.
Twenty-seven targeted policy tests and 77 legacy ledger tests pass. Real-path
acceptance again passes: NAV max error 1.23e-10, 32/64 gradient cosine .902,
norm ratio .802. See docs/v2_portfolio_policy_repair_acceptance.json.
Same S0/F2 fit-only Adam-updated 13 chunks now have gradient norms 13.71–59.14.
Original CPU chain exited on failure; do not resume its old checkpoints.

Restarted original CPU phase chain at 2026-09-14T17:07:27Z from clean 0b34978.
Current shell PID 63776; policies parent PID 63788. Preparation reused all three
verified caches and finished in three seconds. Current log under model_runs/_ops/:
portfolio_policies_repair_0b34978.log. Old failed chain is gone. Rejected artifacts
were moved intact into ROOT/rejected_policy_numerics_74c4021. The diagnostic
uncommitted host edits are stashed; active host checkout is clean committed code.
