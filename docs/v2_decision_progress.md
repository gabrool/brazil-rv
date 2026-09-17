# Decision research continuation and phase acceptance

User authorization: Phases 1–3 of v2_PORTFOLIO_DECISION_POSTMORTEM.md section 13.
Reference registration: research/preregistrations/v2_decision_research.md.

## Current state

Phases 1-3 are complete. The local RTX 2060 finished all 168 Phase 3 financial
fits and 352 books. Both screen candidates received full confirmation; neither
passed. No candidate is promoted. The postmortem section 13 was reread at final
acceptance. See [the combined review](v2_DECISION_PHASES123.md),
[verified results](v2_decision_phase3_results.json), and
[recovery](v2_decision_complete_recovery.json).

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

Complete under the registered local RTX 2060 amendment. The original parents,
repaired store, labels, histories, full eligible populations and financial gates
were retained. Four disposable smoke fits passed before the 48-fit screen.
Both models passed screening, triggering all 120 confirmation fits. Sixteen
continuous books carry actual inventory across all fourteen new score blocks.

TE confirmation: +.086 net bps/day, 95% interval [-.046,+.220], only six positive
utility folds out of ten (eight required). C6: -.056, interval [-.227,+.117],
six positive utility folds and a seed reversal below the -.25-bps floor. Neither
advances. The diagnostic blend is +.106 [-.045,+.259] and remains non-promotional.

The completion audit passed all 168 fits, 352 books and 5,366 bound artifact
hashes, exact score axes/eligible masks and development-only access. The CUDA
allocation failure after 91 confirmation fits and the empty-directory recovery
stop are preserved; completed fits were verified/reused under unchanged b16732e.
The final supervisor exited zero at 2026-09-17T04:03:38Z.

The combined review covers every original phase, attempted/gated branches,
optimization, measured runtime, financial/forecast diagnostics, settlement
sensitivity, storage cleanup and interpretation limits. The recovery archive is
verified member by member. No additional experiment, forward capture, held-out
consumer read, paid host or deployment was started.
