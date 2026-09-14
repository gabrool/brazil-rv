# Post-data stages Aâ€“C progress

Reference: [accepted plan](v2_POST_DATA_RESEARCH_PLAN.md) and
[implementation registration](../research/preregistrations/v2_post_data.md).

## Stage A â€” accepted

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
names. BF16 forward discrepancy was 0.56â€“1.20% of the FP32 score standard
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

Acceptance closed after re-reading the reference plan's sections 4â€“6 and 9.
The [CUDA evidence](v2_post_data_cuda_engineering.json) passed on full F14
16-date, 256-slot, 60-session batches: BF16 discrepancies were 0.57â€“0.76%
of FP32 score standard deviation; complete ASAM updates stayed finite.
Compiled steady-state TE/TL steps were about 23.5/18.1 ms versus eager
49.8/38.7 ms. GRU C1 was about 88 ms eager and 91â€“107 ms compiled; it and
common-recipe S0 will use eager BF16. TE/TL retain compilation. First compile
costs (48â€“193 seconds) remain visible; these are batch timings, not fit ETAs.
The recorded CUDA memory is process high-water, not isolated per-model memory.
CUDA source hashes bind commit 983b176; the later cold-family transfer-contract
correction is separately covered by targeted tests and changes no model or
optimizer numerics. No financial trajectory or evaluation score was read.

A measured input bottleneck is addressed before the first financial fit:
canonical fit/selection tensors are materialized once on the GPU, retaining
every FP32 value, mask, age, date and name. Subsequent epochs gather those
exact dates instead of reconstructing overlapping history on the CPU.
Twenty-five cache/training/ASAM/selection checks pass. Real CUDA cache parity
also passes P/F2/F14, including all full-batch model tensors in changed order.
[Evidence](v2_post_data_cuda_cache_engineering.json) binds ba45f25 sources.
F14 canonical collation/transfer was 228 ms versus .69 ms per cached gather;
this is an input-stage comparison, not a complete-fit speedup. The plan was
re-read after this A efficiency amendment. Full-fit resources remain to measure.

## Stage B â€” accepted

Unseen-date full-model teachers, then fit/selection-only financial calibration.
Freeze screen settings/roster before evaluation-score access. Re-read the plan
and record the evidence for B's completion before C.

The initial 512-update SAM .125 teachers passed an own-current-state task (unseen-date IC
.9994/.9993, seeds11/29) but failed dynamic peer, lagged and context tasks
(roughly âˆ’.025 to .005). A peer-only 512-update bracket also failed with
ASAM .2/.5, SAM .05, SAM .125/LR3e-4 and AdamW. These are retained failures,
not evidence that real financial attention cannot work. Next diagnosis separates
stock routing from brief temporal-message retrieval with a persistent-message
control and tests whether a longer independent-date learning budget resolves
the original task. Neither replaces the original dynamic/lagged gate. Financial
calibration has not begun. No selection or evaluation alpha claim is available.

The original dynamic-peer task learns under ASAM .2 with 4096 fresh-date
updates (seed11 IC .7783). An independent rerun with restorative module probes
reproduces it exactly; seed29 is .7827. The lagged/context tasks still fail
at 4096 updates. Uniform peer control IC is .6860/.7819; own-only is .0204/.0107.
Thus there is cross-stock learning, but this does not establish superior learned
attention or historical routing. The original own teacher's query was constant
through time, so it did not test history. Its corrected definition uses observed
recipient channels at t-21/t-41, with no current-state shortcut. Next engineering
diagnosis compares BF16, an FP32 final head, and full FP32 without changing
history, features, masks, width, loss or optimizer. No financial model precision
change is made solely on that hypothesis. Larger-LR
SAM remains near zero; larger-LR AdamW encountered undefined flat predictions,
which are retained as a failed attempt. The harness now records null IC and
defined-date counts rather than failing JSON serialization and losing evidence.

The corrected own-history task passes under BF16, FP32 head and full FP32
(about .995). The lagged peer task fails under all three precisions (about
âˆ’.024), so low precision does not explain that failure. Next bounded diagnosis
isolates the duplicate current-state MLP and compares ASAM's LR bridge,
weaker SAM and one-pass optimization on the original lagged task. These are
engineering controls; no architecture or financial recipe is changed on an
unverified hypothesis. The first 33 completed/failed attempts and all 103 files
are [recorded and hash-verified locally](v2_post_data_engineering_attempts.json).

ASAM .2/LR3e-4, SAM .05, ordinary AdamW and a no-current-core control also
fail the original lagged task (IC âˆ’.008 to .006). Next is the explicitly
registered donor-supervision comparison: exactly unchanged features and
recipient targets, added own-signal labels only for donor stocks, and
recipient-only unseen-date evaluation with a control bypassing both stock
mixers. This is a teacher-design diagnosis, not a financial model amendment.

The current-store evaluation adapter is implemented while engineering runs:
unchanged economic-beta array proof, exact score provenance, common-population
head/composite/target-view diagnostics, seed omission IC and paired 20/60-session
intervals. Its targeted checks pass; full integration awaits actual scored panels.

The engineering gate is now accepted with an explicit limitation, after
re-reading reference section 6.3. Corrected own-history IC is .9951/.9968;
dynamic same-time peer IC is .7783/.7827. Jointly supervised lagged IC is
.9475/.9557 versus uniform -.0110/.3438 and own-only -.0278 (seed11).
Joint context IC is .8601/.8573 after the disclosed 8192-update diagnosis;
both current (.7587/.7541) and historical (.9616/.9513) regimes learn.
Uniform context is .0131/.0135; own-only at4096 is -.0002 (seed11).
Original unlabelled lagged/context failures remain failures. This evidence
permits financial calibration; it does not certify unlabelled market-history
learning, every optimizer recipe, universal attention superiority or alpha.
The [54-attempt record](v2_post_data_engineering_attempts.json) includes rosters,
source hashes, all failures, and 178 recovered files with exact hashes
(44,516,079 bytes). Confirmation terminal weights are retained for rescore.

Stage B financial work is still pending at this commit. Its fixed four-parent,
44-finetune roster will run with four isolated jobs; inspect actual cache memory,
complete-fit time and curves before closing B. C readouts now use four independent
fold processes, retaining chronological replay within each book. Targeted readout
tests pass; real score integration remains for Stage C. No financial evaluation
panel has been read.

Financial calibration is now running from clean training commit `368fbf7` in
`v2_post_data_368fbf7_20260914T004400Z`. All four P parents completed under
patience: TE seeds11/29 selected epochs34/11 and stopped39/16; C1-all
seeds11/29 selected4/32 and stopped9/37. Parent cache size was 4.27â€“4.43 GB
per process; peak allocated GPU memory was 5.29â€“5.37 GB. The parent archive
has been recovered locally and all 137 member files / 636,613,211 original
bytes verified exactly. The economic rebind also passed, proving all four
beta arrays unchanged. F calibration remains in progress; do not infer a
chosen recipe or financial screen winner from partial results. The combined
[review document](v2_POST_DATA_ABC.md) is explicitly marked work in progress.

Stage B closed after re-reading reference sections 6.3 and 9. All 4 P and
44 F fits completed without a ceiling hit. The registered rule selects
ASAM .2 for TE/TL (selection IC .030505; .5 is within the .001 tie band) and
ASAM .5 for C1 (IC .036144). The selected/terminal full-fit maxima across
48 runs are .1323/.1536; the previous .8-type trajectory is not reproduced
in this bounded program. F14 consistently prefers epoch1/2, and its low
selection IC remains an unresolved modeling/transfer question, not something
to conceal with longer training. Increasing the transfer LR does not improve
the matched F14 mean for either lane.

The temporal and peer modules have finite nonzero selected-state updates;
peer bypass changes scores, with sampled normalized peer entropy .760â€“.973
under the chosen attention recipe. FiLM is active, with selected gamma RMS
.187â€“.692. These are functional diagnostics, not evidence of useful alpha.
A selected-state precision check covers all eight chosen calibration fits
and all selection dates: BF16-versus-FP32 score-rank correlations exceed
.99984, centered RMS discrepancies are .246â€“.759% of cross-sectional score
spread, and the largest selection-IC difference is .000285. Keep the frozen
BF16 precision. The check initially hit a dataset-close typo; the original
failure log is retained, then the corrected diagnostic completed. No fit,
recipe or evaluation score changed. B's end-to-end 4P+44F runner took
29m20s (00:44:00â€“01:13:20 UTC); subsequent audit/recovery time is separate.

## Stage C â€” accepted, including the user-requested .5 extension

All 27 P and 188 distinct F fits completed across the combined program. C has
156 scored fits, 13 cells and 52 accepted books. The reference plan was revisited
after completion; no Dâ€“F work or held-out access was added. TE_all .2 has the
strongest observed IC (.027103 versus S0 .018849); economic superiority is not
established. Slow early attention does not beat matched late attention. The .5
extension reduces rich TE_all IC; both radii remain in the report. See
[the combined review](v2_POST_DATA_ABC.md) for uncertainty, seed/era sensitivity,
training curves, the S0_common ceiling, and settlement/source limitations.

Final audit corrected only policy-composite alignment diagnostics. The initial
helper used legacy horizons; the actual books already used the correct policy.
Five targeted tests passed, and the repair verified all 52 books and protected
score/readout artifacts exactly unchanged. Original diagnostic receipts remain.
All 6,796 final financial files (18,571,092,654 raw bytes) have been recovered in
locally verified lossless archives; their chronological union matches the final
remote inventory exactly. Published JSON copies also match. Exact-instance
shutdown is complete: the exact instance was absent in two provider checks at 03:34:16/03:34:52 UTC on 14 September 2026. The heartbeat is paused.

User-authorized extension: additionally test attention ASAM .5 alongside .2
for TE_slow, TL_slow, TE_family and TE_all. See
[the amendment](../research/preregistrations/v2_post_data_attention_asam50.md).
48 matched score jobs reuse four B fits; 44 new F fits, no new parents.
The original screen, calibration choice and training checkout stay intact.
Final readouts, recovery and shutdown must include the extension. Actual-plan
argument/recipe parity passed for all 48 jobs; four readout tests passed.

Fresh incumbent/common-recipe and GRU/early/late comparisons, then repaired
family and FiLM contrasts on four folds and three seeds. Recover artifacts,
publish a combined report, and close the exact paid instance.

The complete roster and chosen recipes are frozen before evaluation access.
Use six isolated jobs for C to fill preparation/compilation gaps: measured
F14 peak allocated memory is about10.44GB per candidate, so six fit within
the 97,871 MiB (95.6 GiB) GPU. This changes only concurrency, not seeds, data, batch size,
model width, precision or the selected training recipe. Monitor actual use.
The remote training checkout remains368fbf7 while local report commits advance.
