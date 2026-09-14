# Decision research continuation and phase acceptance

User authorization: Phases 1–3 of v2_PORTFOLIO_DECISION_POSTMORTEM.md section 13.
Reference registration: research/preregistrations/v2_decision_research.md.

## Current state

Phases 1 and 2 are complete. Phase 3 CPU engineering is prepared; GPU acceptance
and financial fits are deferred at the user's request while Lambda authentication
is unavailable. No instance was launched. The postmortem section 13 was reread
after Phase 1 and again after all eight Phase 2 fits and their gate completed.

Canonical pointer: docs/v2_decision_run.json. Run root:
C:/quant-data/b3/processed/model_runs/v2_decision_60dc9a2_20260914T212110Z.
Source freeze 60dc9a2; final Phase 1 financial implementation eed4a4e. All nine
copied cache/metadata files match the sealed recovery inventory's SHA-256.
Original portfolio results remain sealed. No paid instance launched for this
program. No forward capture or held-out consumer read.

## Phase 1 acceptance

Complete: [report](v2_DECISION_PHASE1.md) and
[numerical readouts](v2_decision_phase1_results.json). Seven cells, three arms,
fourteen folds and continuous books. All ninety raw/cash controls reproduce;
maximum daily-field error 0.000741 bps, maximum mean error 0.00000180 bps.
No stock-day/input removal; zero additional masked calibration labels.

Implemented benchmark residuals with decision-time beta, one-regressor equal
rank, block-cluster forecast uncertainty and absolute-exposure penalty,
stock-only attribution, and centered circular 20/40/60-session inference.
Thirty-eight targeted tests passed before numerical execution. Original
attempts with overly tight tolerances remain under phase1_numeric_probe and
phase1_numeric_probe_unit. Local memory failures were recovered without changing
financial calculations. NTFS compression preserves cache bytes while reducing
their physical storage by approximately 1.5 GB.

Main results: continuous equal-rank net 4.112/4.555/4.897 bps for S0/TE/C6;
three-regressor benchmark -0.363/2.535/4.972. Equal-rank improves S0/TE with lower
turnover, but does not dominate C6. Keep the registered benchmark reference and
report equal-rank as an additional strong control. No automatic promotion.

## Phase 2 acceptance

Complete under the recorded admission amendment. See
[the report](v2_DECISION_PHASE2.md), [financial results](v2_decision_phase2_results.json)
and [all engineering attempts](v2_decision_phase2_engineering.json).

The small conditional model passes independent-date behavioral acceptance. The
staged stateful MLP fails the inactivity prerequisite in one of three fixed seeds;
the 24 MLP financial fits were explicitly not admitted. All eight unique
conditional fits completed, using the same sealed three-seed forecast ensembles.
The conditional controller is deterministic; its repeated initialization seeds
are exact aliases and are not claimed as independent replications.

Two account state defects were fixed without deleting holdings or changing
eligibility. Trained synthetic NAV agreement improved to 9.882e-12. Twenty-four
new deterministic reference books reproduce Phase 1 within the already registered
bounds. The financial fitting/account/calibration implementations are unchanged
through the eight-fit screen; individual source commits and all failed launcher/
import attempts remain recorded. Final launcher verification skips all eight
completed manifests. Total completed training: 169 epochs, 3,913.742 summed
epoch-seconds (not wall-clock runtime).

No controller advances. Conditional-minus-benchmark net is −3.086 bps/day for TE
and −.067 for C6; utility differences are −3.366/−.530. All primary paired
intervals span zero. C6 improves two folds; TE improves none. The prior-selection
fallback does not rescue the screen. No additional Phase 2 confirmation or
continuous books are triggered. The historical broader ML question remains open.

Acceptance review against postmortem section 13: independent synthetic behavior,
actual optimizer/accounts, failure disclosure, corrected context information,
real chronological comparison, causal fallback and no automatic seed expansion
are covered. The MLP comparison stopped at its declared prerequisite rather than
being presented as a completed real-data experiment. Return to Phase 3 is accepted
even without a Phase 2 winner, as specified by the original plan.

## Phase 3 acceptance

Prepared: exact objective/parent amendment, three unchanged TE parents, three
explicit C6 adaptations, sealed economic labels, single-pass auxiliary heads,
fit-only scaling/masks, matched neutral training, score exports, fixed-policy
readouts, prior-only blend, gate/confirmation dispatch and continuous inventory
replay. Four screen mappings are frozen from old OOS fit/selection observations:
TE weights F2=1, F6=1, F10=0, F14=.25. No new evaluation outcome selected them.

Real fit-only CPU checks retain 60 sessions and all active names, preserve neutral
initialization exactly, and show auxiliary gradients entering both encoders.
The earlier 23 objective/model tests, three new readout tests, four inference/
reuse tests and the continuous fallback regression pass. These are engineering
checks, not a financial result or GPU performance acceptance.

Pending: compiled/BF16 GPU numerical and two-epoch throughput acceptance, 48 new
matched F fits, complete real-score readout, then only gate-triggered ten-fold
confirmation. Keep the same neutral-IC checkpoint selector within both objectives.
No Phase 3 financial fit has started. The user's CPU-only steering defers Lambda
attempts and monitoring; no new capacity automation is active for this program.

The [restart document](v2_DECISION_RESUME.md) records exact paths, commands, roster,
limitations and recovery/termination requirements. The
[CPU checkpoint](v2_decision_cpu_checkpoint.json) binds saved local artifacts.
After the later Phase 3 completion, reread postmortem section 13 again and write
the combined LLM-ready report. Phase 4, forward capture and held-out reads remain
outside this task.
