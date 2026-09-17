# Phase 3 local GPU execution

**Completed 2026-09-17.** All 168 financial fits and registered readouts are done;
no candidate passed confirmation. Do not relaunch this campaign. Use the
[combined review](v2_DECISION_PHASES123.md), [results](v2_decision_phase3_results.json)
and [verified recovery](v2_decision_complete_recovery.json). The procedures and
intermediate status statements below are retained as the historical execution
checkpoint, not current pending work.

The user authorized local RTX 2060 optimization and the remaining Phases 1–3
experiments on 16 September 2026. Phases 1–2 are complete with no controller
survivors. Phase 3 requires 48 new F fits, their registered fixed-policy analysis,
and 60 additional F fits per surviving architecture only. There is no P retrain,
new architecture search, held-out consumption, forward capture or Lambda launch.

## Exact experiment and sources

Resolve `docs/v2_decision_run.json`. Current root is
`C:/quant-data/b3/processed/model_runs/v2_decision_60dc9a2_20260914T212110Z`.
Use the repaired store bound in `docs/v2_data_inputs.json`, manifest
`61149d43a23fc55fcd92b43aaa4e74d747fbdf62f852bedbbbdb2cf18281f3cc`.
The local input verification matches 126 saved file hashes: accepted store
arrays, economic caches/labels, original parents and prior-only frozen mappings.
The store ends on 2024-12-30. The workspace code path is a junction to C:/quant;
the environment's editable package resolves the same checked-out source.

Compare TE_all ASAM .2 and C6 SAM .125; neutral versus .25 economic auxiliary;
F2/F6/F10/F14; seeds 11/29/47. Retain full 60-session histories, all eligible
names, balanced unique-date batches of at most 16, 60-epoch schedule and
five-check patience, raw neutral-IC selection and complete diagnostics. Each
architecture/objective has a disposable two-epoch F2 seed11 smoke fit first.
The original Phase 3 registration defines labels, historical mappings, fixed
equal-rank QP, ensembles, causal blend, uncertainty and survivor gates.

## Optimization decisions and acceptance

- Supported native Windows compilation: PyTorch 2.6.0/cu126 with
  Triton-Windows 3.2.0.post21. Upstream newer Triton dropped Turing. Linux/GH200
  retains the existing 2.13.0 environment. These are fresh matched local controls,
  not claims of identical numerical trajectories to previously sealed fits.
- FP16 Tensor Core autocast; FP32 parameters, moments and losses. Both SAM passes
  are unscaled before perturbation/clipping. Overflow retries reuse the exact
  batch/RNG and do not skip updates; scaling state is saved in resume checkpoints.
- Replace expanded overlapping-window caches with session/security history
  storage and exact gathers. Labels/snapshots retain their decision-date axis.
  Full GPU caching becomes feasible without reducing batch/history/name count.
  Canonical collator equality is checked on both changing membership and real data.
- Compiled forward, backward and rank/economic losses; fused CUDA AdamW and
  foreach SAM arithmetic. Default Inductor mode preserves fusion without costly
  exhaustive reduction tuning. The discarded max-autotune probe spent several
  minutes tuning its first backward. A short cache path fixes Windows filename
  length failures. Failed engineering probes are excluded from financial results.
- One persistent GPU worker avoids repeated Python/CUDA startup and redundant
  immutable-file hashing. Fit parameters, optimizer, RNG and compiler references
  reset at every boundary. Sequential execution avoids competing for 6 GiB VRAM.
- Full validation remains every epoch. Scores reuse the selected live model.
  Existing compact stock batches, single auxiliary head/encoder pass, early
  stopping, compatible parent reuse and three-seed gate-driven confirmation stay.

No batch accumulation is needed in the full-size engineering test: 16 dates,
256 padded names and 60 sessions fit directly. FP16 attention prediction RMS
error is .000450 of FP32 prediction standard deviation in the F14 probe. Its
eager ASAM updates take about .31s; default-compiled updates about .18s after a
70s cold first update. These are representative fit-only steps, not total-run
benchmarks. Final real smoke timing, cache memory and other cases remain in
`phase3/local_engineering` and `phase3/gpu_acceptance.json`.

Targeted tests: 23 passed, covering canonical cache equality, changing identity
membership, scaled SAM/ASAM equivalence, exact same-batch overflow retry, resume,
objective causality/scaling and matched-account readouts. The full smoke acceptance
is required before financial dispatch; unit tests alone do not authorize it.

Storage: lossless NTFS compression of the three existing derived data stores
recovered roughly 18 GB without changing their logical bytes. Accepted store
arrays were rehashed after compression. Raw sources and prior research artifacts
remain intact. Only unused managed uv package-cache files were removed.

## Runtime and continuation

Use `powershell -NoProfile -ExecutionPolicy Bypass -File ops/run-decision-local.ps1 -Smoke`
after a clean committed checkout and `objective_program prepare` binds the final
implementation. Preserve the earlier phase3 design as engineering history first.
Then run the same launcher without `-Smoke`. It executes the screen, all readouts,
registered survivor-only confirmation and full continuous books when applicable.
The launcher stops on failure. It does not start a paid host.

Environment variables, short compiler caches and single-thread BLAS are set in
the launcher. Fit histories and `phase3/screen_local_progress.json` provide actual
progress. Do not edit code while fits are queued. After numerical completion,
reread postmortem section 13, verify/recover artifacts, write the combined LLM
review and commit/push code and reports. No architecture is automatically promoted.

All four representative architecture/fold checks passed. Compiled steady update
seconds: TE_all F2 .115, F14 .176; C6 F2 .137, F14 .179. Peak allocated memory
in these probes was at most 1.607 GiB (excluding the subsequently constructed
full fit caches and desktop use). All precision errors were below .000450 of
FP32 prediction standard deviation. No overflow retry was needed in these probes.
Source is now frozen for the four smoke fits. No financial Phase 3 fit yet.

The first actual smoke launch exposed a pre-existing Phase 3 descriptor omission:
parent records had paths/hashes but lacked the byte count required by the portable
file resolver. Preparation now records the verified parent file size as well.
No parent weights changed and the failed launch stopped before creating a fit.
