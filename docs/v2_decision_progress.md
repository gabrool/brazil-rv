# Decision research continuation and phase acceptance

User authorization: Phases 1–3 of v2_PORTFOLIO_DECISION_POSTMORTEM.md section 13.
Reference registration: research/preregistrations/v2_decision_research.md.

## Current state

Phase 1 ready for numerical reproduction and financial readouts. Reviewed
PROJECT_CONTEXT and postmortem section13. Main started from d682962; registration
is committed at 60dc9a2. Existing portfolio results remain sealed. Run root:
`C:/quant-data/b3/processed/model_runs/v2_decision_60dc9a2_20260914T212110Z`.
All nine copied cache/metadata files match the sealed recovery inventory's
SHA-256, including three 699,471,732-byte PolicyData files. Temporary S3
credentials were process-only. Lambda ETags were not MD5 hashes; rclone's MD5
comparison was disabled and replaced by exact sealed SHA-256 verification.
No paid instance has been launched for this program. Initial local free space
C:4.85GB / D:1.32GB; avoid duplicate caches.

Implemented coherent benchmark residuals using decision-time beta, one-regressor
equal-rank control, block-cluster forecast uncertainty and QP absolute-exposure
penalty, plus stock-only attribution. Deterministic calibration has its own small
inference class; it avoids executing an identically zero MLP. Original MLP remains
the stateful control and inherits the same economic mapping. Circular 40-session
bootstrap replaces the underweighted-boundary implementation, with 20/60
sensitivities. Ruff passes; 38 targeted allocation/account/causality/readout/source
tests pass. Fit-only F2 calibration check reproduces original 8.0525483-bps
intercept and yields 2.4206585-bps benchmark-residual intercept; no financial
evaluation has yet been interpreted.

## Phase 1 acceptance

Pending: freeze source root, implement calibration/uncertainty and deterministic
roster, targeted causality/accounting tests, exact raw reproduction, financial
readouts and diagnostic report. Then RETURN TO postmortem section13 Phase2.

## Phase 2 acceptance

Pending: independent-date behavior-learning acceptance, exact feature/training
amendment, real chronological screen and gate-dependent continuation. Then
RETURN TO postmortem section13 Phase3 regardless of whether the controller wins.

## Phase 3 acceptance

Pending: precise economic auxiliary objective and parent adaptation amendment,
matched neutral control, engineering/throughput acceptance, financial screen,
blend control and gate-dependent confirmation. Then combined report, recovery,
GitHub synchronization and exact paid-instance closure if one was launched.
