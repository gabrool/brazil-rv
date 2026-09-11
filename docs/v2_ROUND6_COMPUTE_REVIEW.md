# Round 6 training workload review

Inspected on 2026-09-11 against frozen training commit `6e9a411`. This is a
read-only implementation and artifact review, not a GPU profiler trace or an
amendment to the running experiment.

## Actual workload

The current daily model consumes 60 historical rows for each stock at each
decision. The dataset retains the entire 933-ISIN axis. The slow GRU runs on
that dense axis before the active-name mask is applied to pooling and outputs;
inactive slots are not removed from the encoder computation.

An effective batch contains eight adjacent-date pairs, or sixteen date rows.
The uniform sampler visits each adjacent pair once, so an interior date appears
twice per epoch before the incomplete final batch is dropped. This pairing
remains enabled with `lambda_persistence=0`. Changing it would change update
counts and sampling, and therefore needs a matched research comparison.

The optimizer is SAM-AdamW, with two forward/backward passes per update. Each
forward handles 16 x 933 x 60 = 895,680 stock-time positions and 28,661,760 base
feature values, before validity/age encoding and intermediate activations.

The soft-Spearman loss forms dense pairwise stock comparisons for each date and
horizon. It computes the sigmoid before masking invalid comparisons. Across
sixteen dates and five heads this represents 69,639,120 pairwise elements per
loss evaluation; SAM evaluates the loss twice. These are computational elements,
not independent statistical observations. Only the model forward is passed to
`torch.compile`; the loss and optimizer remain outside that compiled forward.

The frozen manifests have `use_bf16=false`. The F1 launcher also records the
PyTorch warning that TF32 matrix-multiplication acceleration is available but
not enabled. Mixed precision must not be described as active in these runs.

## Representative completed fit

`trajectories/magnitudes/F12_seed_11` contains 1,649 fit dates, a 55-date internal
selection window, 206 updates per epoch, and 20 completed epochs. This is 4,120
optimizer updates and 8,240 forward/backward passes. Across its fit dates, the
active-name count averages 168.3, ranges from 110 to 243, and is far below the
933-name dense axis. F1 averages 119.1 active names; F14 averages 171.9.

Original remote file timestamps give the following elapsed times while the
six-slot GPU workload was running. Local recovery rewrites file timestamps,
so local copied mtimes must not be used for these calculations.

| Magnitudes seed 11 | Setup, training and per-epoch selection | Final checkpoint/scoring tail | Total |
|---|---:|---:|---:|
| F1 | 412.0 s | 17.3 s | 429.3 s |
| F12 | 2,501.3 s | 19.7 s | 2,521.0 s |
| F14 | 680.7 s | 21.7 s | 702.4 s |

This separates final scoring from the preceding work, but does not separate
compilation, loading, network compute, rank loss, optimizer or epoch selection.
GPU utilization and these coarse timings do not establish which component is
the dominant wall-clock bottleneck. Inference-only costs and attribution run
locally while later GPU arms train.

## Optimization implications

There is concrete excess dense computation, especially from inactive slots in
the encoder and quadratic ranking loss. A compact active-name implementation
and an equivalent compiled/fused rank loss merit a bounded throughput profile
and output/gradient checks. Compaction must preserve dated identities, active
pooling, label masks and all needed historical inputs; stock order, floating
reductions and dropout RNG can affect training reproducibility.

BF16/TF32, independent-date sampling and replacing SAM also merit separate
measurement, but change numerical or statistical training behavior. They must
not be silently substituted into the frozen experiment. No speedup factor has
been measured for these changes, and no live training setting changed in this
review. The number of folds, seeds and arms explains the batch size of the
research program, but is not a sufficient explanation of per-fit efficiency.

Sources: `research/src/brazil_rv/v2/{data,model,train,losses}.py`,
`research/src/brazil_rv/modeling/engine.py`, frozen trajectory manifests/history,
the accepted store's date-bounded `active.npy`, and operational
`training_cost_timing_evidence.json` containing original remote timestamps.
