# Post-data stages A–C progress

Reference: [accepted plan](v2_POST_DATA_RESEARCH_PLAN.md) and
[implementation registration](../research/preregistrations/v2_post_data.md).

## Stage A — accepted

Implemented selection from epoch 1 with fixed-schedule patience, ASAM,
module-owned bias/norm routing, explicit transferred names/LR, and exact selected
checkpoint scoring. Unconditional late averaging is removed from the current
characteristic trainer; the incumbent's separate recipe is preserved. Historical
Round-7 orchestration is reproducible from its original Git commit, not through
the changed current trainer CLI.

Bounded diagnostics restore weights, optimizer moments and RNG, and record
per-module activation/gradient/update scale, FiLM gamma/beta, sampled attention
entropy/logits and peer bypass sensitivity. No hooks enter compiled training.
The new program binds current inputs and declares the B/C roster and parent
reuse before scoring. The old native-fundamentals construction command, builder,
specification and tests are removed; shared source-audit helpers remain. Current
magnitude formulas describe the uncapped smooth transform. Sealed manifests
retain their original bytes; the runtime preprocessing contract is unchanged.

Local targeted checks pass: 27 training/ASAM/resume/runner checks, 37
architecture/specification/source checks, and 16 program/preprocessing/runner
checks (these groups overlap; not a unique test count). Real-input CPU acceptance
passed P/F2/F14 for S0, C1, TE and TL with 60 sessions and all sampled active
names. BF16 forward discrepancy was 0.56–1.20% of the FP32 score standard
deviation; actual adaptive optimizer steps were finite. Evidence is at
the source-bound [CPU evidence](v2_post_data_cpu_engineering.json).
No financial trajectory or evaluation scoring has begun. Exact GH200 instance
`7eed8bef79944d619ab42818b4978022` launched at 23:11:04 UTC and is active at
`192.222.58.84`. The repaired-store delta was recovered with all file hashes
verified. CUDA acceptance runs in user unit `brazil-rv-post-data-a`; after the
first dispatch was stopped on SSH logout, user lingering was enabled and the
unit restarted. Require its result JSON and successful exit, not just an
inactive unit. Operational continuation is recorded at
`C:/quant-data/b3/interim/post_data_20260913/CONTINUE.md`.

Early-stop and score-attachment integration checks now also pass: an imposed
declining selection curve stops at epoch 3 and retains epoch 1; later scoring
reuses that fit. Explicit runner continuation preserves logs and requires the
trainer to validate the saved contract. Obsolete fixed-budget launcher call
sites are removed; historical result/path readers remain.

Final transfer review also separates wholly unexposed P family encoders from
learned transferred parameters. No valid value AND no known-age exposure in P
means full F LR; known-age-only and partially exposed encoders retain their
learned mappings. Shared fusion tensors remain transferred. This avoids
suppressing genuinely newly available families by the .3 transfer multiplier.
Twenty targeted adaptive-training/selection/resume tests pass after this change.

Acceptance closed after re-reading the reference plan's sections 4–6 and 9.
The [CUDA evidence](v2_post_data_cuda_engineering.json) passed on full F14
16-date, 256-slot, 60-session batches: BF16 discrepancies were 0.57–0.76%
of FP32 score standard deviation; complete ASAM updates stayed finite.
Compiled steady-state TE/TL steps were about 23.5/18.1 ms versus eager
49.8/38.7 ms. GRU C1 was about 88 ms eager and 91–107 ms compiled; it and
common-recipe S0 will use eager BF16. TE/TL retain compilation. First compile
costs (48–193 seconds) remain visible; these are batch timings, not fit ETAs.
The recorded CUDA memory is process high-water, not isolated per-model memory.
CUDA source hashes bind commit 983b176; the later cold-family transfer-contract
correction is separately covered by targeted tests and changes no model or
optimizer numerics. No financial trajectory or evaluation score was read.

## Stage B — in progress

Unseen-date full-model teachers, then fit/selection-only financial calibration.
Freeze screen settings/roster before evaluation-score access. Re-read the plan
and record the evidence for B's completion before C.

## Stage C — pending

Fresh incumbent/common-recipe and GRU/early/late comparisons, then repaired
family and FiLM contrasts on four folds and three seeds. Recover artifacts,
publish a combined report, and close the exact paid instance.
