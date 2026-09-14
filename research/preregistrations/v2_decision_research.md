# Decision research: authorized Phases 1–3

Registered 2026-09-14 before new financial results. User authorized all three
phases of docs/v2_PORTFOLIO_DECISION_POSTMORTEM.md, section 13. Read that section
again and record acceptance after EACH phase. Phase 4 is outside this program.

Use the accepted repaired store, full 60-session history and all PIT-eligible
names. No forward capture, held-out 2025/2026 reads or deployment. Retain old
sealed results. New results go to a new source-bound run root. Three decision
seeds remain 11/29/47. Primary economics: continuous daily net return above CDI
minus 2.5 times causal daily portfolio variance, same costs/accounting as the
preceding portfolio program. Report net return separately and keep settlement
uncertainty, seed differences and risk exposures visible.

## Phase 1: frozen forecasts and coherent economic calibration

Reuse all fourteen folds and causal prelude for S0, TE_all .2 and C6. Original
forecast/cache sources bind the completed portfolio recovery inventory. A hash
verified copy is reusable; do not recreate forecasts or substitute in-sample
predictions. Fit-only calibration labels must mature within fit. Selection and
evaluation retain existing 10-session purges and 55-session selection.

Six deterministic cells plus cash:

| Cell | Specification |
| --- | --- |
| raw | Exact original raw shareholder-minus-CDI ridge calibration, including intercept; reproduction control |
| no_intercept | Same raw slopes/coordinates, zero intercept; mechanism diagnostic, not the proposed final economic model |
| benchmark | Fit shareholder excess minus decision-time beta times BOVA excess; intercept retained as residual-universe premium |
| uncertainty | Same benchmark calibration, penalize absolute stock weights by one fitted standard error of their expected-return prediction |
| equal_rank | Benchmark calibration using the equal-average three-head rank as one regressor; same QP risk/cost retention and sizing |
| stock_only | Benchmark calibration with hedge cap zero; otherwise same constraints; hedge attribution diagnostic |

Original ridge 0.001 and equal-date loss weights remain fixed. Standardize only
on fitting observations. Use a 20-session block-cluster sandwich covariance of
the calibration parameters (including intercept), preserving same-date stock
dependence; inference has no outcome input. This is a declared one-standard-error
uncertainty scenario, not a calibrated coverage guarantee. Do not add a second
arbitrary alpha haircut or tune the penalty using evaluation outcomes. Missing
benchmark endpoints can mask calibration labels, never erase prediction inputs
or universe membership. Report any affected observations.

BOVA's assumed residual alpha remains zero. The benchmark's realized H-session
return is used ONLY for labels, with beta fixed at the prediction date. Compute
CDI over the same interval. Preserve shareholder action/claim labels. Net/gross/
beta/name limits and 4-bps-per-side base costs are unchanged; stock_only is the
explicit exception for hedge availability. All-cash earns the same historical CDI.

Report fitted intercept/slopes, condition number, calibration errors and tail
returns, joint gross/hedge saturation, turnover/costs, realized exposures, and
paired fold/continuous economics. The raw control must reproduce sealed books
within financial numeric tolerance before financial interpretation. Compare
continuous paired circular blocks of 20/40/60 sessions with 10,000 draws and seed
20260914; 40 is primary. These are nominal development intervals, not untouched
holdout tests. Inspect centering; do not use the prior biased finite-block
percentile result as a decision gate.

Phase 2's reference is the coherent benchmark model. The uncertainty cell is a
separate contrast, not an evaluation-selected change to that reference. A
demonstrated engineering defect can be corrected with an explicit amendment and
fresh affected outputs; preserve evidence of the failed attempt.

## Phase 2: learnability and a small reliability controller

First require actual allocation/account/training behavior on independent
synthetic dates: known useful/zero opportunity state, continuation versus
reversal after adverse moves, varying costs/borrow, real inventory and costs.
Compare known controls, profitable trading, cost-aware inactivity and distinct
exit responses. Include future-mutation and sequential/account parity. Gradient
existence or a single fitted path is insufficient. Record any recipe amendment
before inspecting new financial evaluation results.

Then compare coherent deterministic control, a low-dimensional reliability/
calibration model, and a corrected shared stateful residual model. Use cached
chronological out-of-fit forecasts. Reliability inputs can include causal
common state, cross-seed agreement, current alpha-distribution summaries and
strictly matured shadow-forecast outcomes. Keep economic meanings/scaling
explicit; raw rank-model scores are not calibrated confidence. Synthetic
engineering fixes the exact feature roster/training recipe in an implementation
amendment before financial fits. Retain 32-session carried-state truncated
backpropagation and exact-ledger selection; test longer gradients where changed.

Cash, calibrated model and learned checkpoint fallback must be selected using
only the fold's prior selection window, including actual liquidation costs in
continuous transitions. Report the candidate without fallback separately.
Initial financial screen is F2/F6/F10/F14 x three seeds; run remaining matching
folds only for a survivor with positive paired screen mean net/utility versus
coherent control and positive utility in at least three screen folds, without
material seed reversal. Missing that gate does not block Phase 3's separately
authorized objective test. Record failed controller evidence honestly.

## Phase 3: magnitude-aware forecaster objective

Use TE_all .2 and C6 as the strong comparison. Test neutral-only control versus
a neutral-rank plus magnitude-preserving economic auxiliary objective. Define
exact target, loss weight, head initialization/parent adaptation and selected
controller in a phase-specific amendment BEFORE evaluation. Preserve full
history/eligibility and AMP/compile/unique-date efficiencies. Use compatible
existing neutral parents where their representation/coordinates are unchanged;
new economic heads must be explicitly initialized and recorded, never silently
loaded from incompatible weights. Changes in recipe need a matched fresh
neutral-only control rather than misleading reuse.

Screen F2/F6/F10/F14 x seeds11/29/47; use the same fixed corrected portfolio
controller for both target variants, with calibration fitted only on each fold's
permitted past. Report rank IC, cardinal calibration, tail outcomes and net
utility. Include an inexpensive causal C6/TE blend or residual control with
weights selected only from permitted past data. Continue qualifying survivors
under the same screen gate as Phase 2; disclose global development choices and
all attempts. Do not run a full architecture/optimizer/objective factorial.

## Completion and operating discipline

After each phase: reread the postmortem, map every promised deliverable to
evidence, record decisions and next-phase scope in docs/v2_decision_progress.md.
Keep implementation lean, targeted tests and full-run throughput measurements.
Use local CPU where practical; use the approved GH200 launcher when paid compute
is useful and use its capacity for independent necessary jobs. Recover/hash
required artifacts, commit/push all changes and results, then terminate the exact
paid instance and verify absence twice. Do not leave a paid instance idle after
the required work is recovered. Final LLM-ready report covers all phases,
engineering amendments, results and limitations.

### Phase 1 numerical reproduction amendment

The initial local replay stopped because a cross-host original-control difference
of 0.000005416 bps exceeded an indiscriminate 0.000001 threshold. Exposures
agreed within 1.36e-9 and turnover within 1.87e-8 NAV. Use explicit daily
field tolerances: 0.0001 bps for return/cost fields and 0.000001 NAV for fraction
fields. This is numerical equivalence, not permission for material drift. Save
all errors and thresholds. Preserve the partial initial attempt and rerun the
same financial roster from a fresh Phase 1 directory; no data/model/allocation
calculation changes. Local loading uses one worker after two exceeded memory.

The complete cross-host numerical check subsequently exposed a 0.000223-bps
maximum daily difference in S0/F7, with gross within 3.13e-8 NAV. Its ridge slopes
differ from the sealed Linux fit by at most 2.2e-19, consistent with native
floating-point solver differences. Final reproduction acceptance uses TWO
conditions: every bps series' mean within 0.0001 bps, every daily bps observation
within 0.01 bps, and each daily fraction within 0.0001 NAV. This is less than
one four-hundredth of the per-side transaction-cost assumption even at the
largest permitted daily deviation; the aggregate check remains tighter.
Preserve both stopped partial attempts. Rerun the same unchanged financial
calculations in a fresh directory and report actual maxima, not just pass/fail.
No further tolerance widening is planned; a failure needs mechanism diagnosis.
