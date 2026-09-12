# Round 6 CUDA capture recovery

On 2026-09-12 the revised Session-1 dispatcher stopped after 49 complete fits
(42 S0 and seven magnitudes). Three other magnitudes processes failed during
PyTorch CUDA capture: F2/47, F3/29 and F4/11. The first reported error was
`cudaErrorStreamCaptureInvalidated`, at a generated `aten.randint.low_out`
operation in the second SAM pass. Earlier compiler warnings reported a missing
CUDA graph manager in `cudagraph_partition_post_compile`.

The installed PyTorch 2.13 runtime enables Inductor graph partitioning by default.
The training implementation already requests dynamic full-graph compilation and
`cudagraph_skip_dynamic_graphs=True`; the failures arise in capture of generated
partitions. The supported environment setting `TORCHINDUCTOR_GRAPH_PARTITION=0`
disables this partitioning and restores whole-graph handling. It does not disable
Inductor compilation or change the model, BF16 precision, loss, SAM/AdamW,
learning-rate schedule, dates, padding, epochs, checkpoint selection or seeds.

Under the user's standing authorization to choose and implement the recommended
recovery, test this setting on one-epoch, score-free versions of the three failed
configurations. Require the same one-training/one-selection compiled-graph gate
before resuming. Apply the setting to every remaining fit and record it in the
recovery plan and operational amendment. This is a runtime recovery, not a new
modeling experiment; keep the original frozen design and training commit
`0bf8130b73fe6b207c6b6f62603bf383b8f82e1f`. No outcome-based choice of compiler mode
or new evaluation-panel read is used. Bit-identical stochastic training across
compiler configurations is not claimed.

Preserve all 49 completed fits. Archive the failed attempts, then rerun only those
missing results and the undispatched jobs, with the original seed and recipe.
Do not overwrite failed evidence or retrain the complete baseline. All 164 files
from the magnitudes attempts and dispatcher failure have been recovered and
individually hash-verified. Recovery archive SHA-256:
`c35123942dc6bb24b3f900fa630504db94f5d5c6fbc9e02a6f8ad77ed8d90268`.

The S0 readouts completed independently: fourteen accepted aggregate books, all
three borrow scenarios without gate failures, and exact reproduction of the
original-cost headline. Cost summary SHA-256:
`22c7e4965ea8081719167021a08bfaccefe25ede42c632703772544c8b9488b8`.
These results do not select a candidate before the registered comparisons finish.
# Session 2 planner correction

All nine Session 2 smoke processes completed on 2026-09-12. Before any full
Session 2 pretraining or fine-tuning, the planner rejected `C6_fresh_p_P` because
the expected sidecar counts used roster order (magnitudes, fundamentals), while
training records them alphabetically (fundamentals, magnitudes). Every other
model-config field matched. The registered roster and the trained graph agree.

`arm_config` now sorts sidecar families to match training. P and F regression
tests cover the two-family roster and reversed input order. This is a planning
contract correction, with no data, objective, model, seed, budget or outcome
change. Existing smoke results are retained. The historical frozen training
checkout remains at `0bf8130`; its resume planner applies only this sorting
correction in memory. Child training processes continue to use the unchanged
frozen checkout and the previously documented graph-partition override.
