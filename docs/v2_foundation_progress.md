# Foundation program progress

User authorization: proceed with all five workstreams; prioritize experiment
quality and speed. Registration: `research/preregistrations/v2_foundation.md`.

- Inventory/canonical contract: all 84 original Phase-3 neutral F trajectories have
  their hash-verified selected and requisite preceding epoch states. Averaging does
  not need retraining. `v2_foundation_inventory.json` records exact source bindings.
  The accepted store still has 568,815 active stock-days and at most 243 active names.
- Sparse option fields are distributed across hundreds of dates but typically only
  one observed security per date: put/call OI has 442 stock-days on 409 dates;
  delta OI/volume has 409 on 385. Retain them pending conditional evidence. The
  uncovered-call field has no values or known ages and is excluded by name from
  new compatible input contracts; immutable source data are unchanged.
- Engineering: named exclusions, matched graph settings and true update EMA are
  implemented. 32 targeted input/training tests pass, including exact interrupted
  resume, unchanged raw trajectories with EMA tracking, fit-only subset scalers and
  retained masks/ages. Two additional readout/gate tests pass. Removed the superseded
  pre-factorization cache test; its invariant is covered by the current exact-cache test.
- Local worker: clean source `d21a93a`, detached worktree
  `C:/quant/brazil-rv-foundation-d21a93a`, launched 2026-09-18 21:49 UTC.
  Resolve `v2_foundation_run.json`; inspect worker.json and actual process command
  lines before resuming. Full P/F2 GPU admission is in progress, then the same worker
  runs the nine fresh matched P parents and 36 screen F fits. Raw and EMA are
  selected on identical preceding selection dates; raw patience bounds each trajectory.
- New financial fits: not started.
- CPU readouts: seed-size analysis reuses sealed single-seed and three-seed books;
  only pair books need replay. Current account must reproduce a sealed ensemble
  before those new books. Do not run saved-average GPU scoring concurrently with
  the fit worker. Use original compatible inference source for old checkpoints.
- Remaining: original-checkpoint averaging and ensemble diagnostics; matched input wave; averaging/readout wave;
  conditional component/capacity waves; residual information probe; confirmation,
  accounting uncertainty review, consolidated report and recovery archive.

After each wave reread section 9 of `v2_NEXT_RESEARCH_DECISION.md`; record gates,
accepted/rejected candidates, and the exact next contrast before dispatch.

Training worktree is immutable. Readout/report changes on main must not change the
running fit code. The existing 15-minute continuation heartbeat has been updated
for this whole program; it stays quiet on healthy unchanged progress. It is not a
replacement for the explicit review and next-wave decision.
