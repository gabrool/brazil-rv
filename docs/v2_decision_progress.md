# Decision research continuation and phase acceptance

User authorization: Phases 1–3 of v2_PORTFOLIO_DECISION_POSTMORTEM.md section 13.
Reference registration: research/preregistrations/v2_decision_research.md.

## Current state

Phase 1 accepted; Phase 2 engineering is in progress. The postmortem's Phase 1 and
Phase 2 sections were reread after all 315 Phase 1 books completed.

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

Pending: independent-date behavioral learning acceptance; exact feature/training
amendment before financial fits; chronological four-fold/three-seed controller
screen and gate-dependent continuation. Always RETURN TO postmortem section13
Phase3 afterward, even if no controller wins.

Engineering prepared: small conditional calibration and stateful MLP;
fit-only smooth robust scaling; actual cash liquidation through the shared
ledger; epoch resume; separate daily market context; strictly matured shadow
outcomes; cross-seed rank disagreement. 675 source score/identity/mask/manifest
members verified against sealed sources; three agreement arrays are saved under
phase2/context, approximately 1.8 MB each. These inputs are not new financial
results. Three new targeted causality/scaling/conditional-account tests pass;
ten existing policy tests also pass. Synthetic behavioral acceptance is still
required before financial dispatch.

Subsequent engineering found and fixed an actual training/replay state mismatch:
sub-1e-10 NAV order intentions created microscopic holdings only in training,
and the replay retained the old holding age when an opposite entry crossed a
sub-threshold residual. Trained-path NAV reconciliation now agrees within
9.882e-12; the dedicated regression and 21 other account/allocation tests pass.
The synthetic null originally contained predictable adverse shocks and was
under-sampled. Preserved attempts and all corrections are documented in the
Phase 2 amendment. The balanced 1,920-session test accepts the conditional model;
standalone MLP initializations remain unreliable in zero-opportunity states.
A stateful residual initialized from the accepted frozen conditional map is
being tested across three seeds under phase2/behavioral_acceptance. No real
controller fit has yet started. Phase 3 compatible-parent and auxiliary-loss
engineering proceeds independently while those tests run.

## Phase 3 acceptance

Pending: precise economic auxiliary objective and parent adaptation amendment,
matched neutral control, engineering/throughput acceptance, financial screen,
causal blend control and gate-dependent confirmation. Preserve full history,
eligibility, unique-date batches, AMP/compile where beneficial. TE_all .2 and C6
are the matched architecture roster. Then combined LLM-ready report, verified
artifact recovery, GitHub synchronization and exact paid-instance closure if
one was launched. Phase 4 remains outside this task.
