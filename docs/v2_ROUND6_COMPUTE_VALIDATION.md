# Round 6 compute amendment: validation and restart

The nine-fit engineering bridge passed the predeclared internal-selection
tolerances. Retain active-name packing, BF16, unique-date sampling, and the
60-epoch ceiling. This is engineering validation, not a candidate promotion or
an out-of-sample performance claim.

## Population and numerical equivalence

Every PIT-active name remains included, whether traded or labelled. The maximum
observed active count is 243; stage widths are rounded up to 16, without truncation.
Each selected name keeps all 60 historical input rows, irrespective of its earlier
eligibility. Targets and neutralization retain the canonical population, and
scoring restores the original 933-name ISIN axis.

Dense versus packed FP32 fit predictions differed by at most 2.3842e-7. BF16 versus
packed FP32 cross-sectional rank correlation was at least 0.999889, averaging
0.999967. Vectorized horizon losses passed separate-head loss and gradient
equivalence tests. Moving eligibility selection ahead of history copying produced
bit-identical tensors for every batch field on sixteen spread-out real F12 fit
dates. The fixture also verifies restored score identities. The earlier bridge
therefore remains valid after this CPU copy reduction.

## Training budget and quality bridge

Of 102 completed old F fits, 20 reached the old epoch-20 ceiling and 15 selected
epoch 20. The new sampler visits each fit date once per epoch instead of the old
overlapping pairs. Approximately 40 new epochs match the old maximum update
budget; 60 allow 50% more maximum updates. Validation every two epochs and
Patience-3 preserve approximately the old update-scale stopping patience.

The bridge used magnitudes, folds F1/F12/F14, seeds 11/29/47, and the same sealed
parent checkpoints as the old fits. Only internal-selection IC was inspected;
the bridge wrote no evaluation scores.

| Fold | Mean selected-IC change | New selected epochs | New completed epochs |
|---|---:|---|---|
| F1 | -0.000701 | 8 / 12 / 10 | 14 / 18 / 16 |
| F12 | +0.008904 | 44 / 36 / 38 | 50 / 42 / 44 |
| F14 | -0.000648 | 2 / 2 / 2 | 8 / 8 / 8 |

Overall mean change was +0.002518. Both investigation thresholds passed: no mean
loss above 0.001 overall or 0.003 within a fold. All nine runs passed the exact
one-training/one-selection compiled-graph gate. None reached 60 epochs. F12 seed
11 gained 0.001267 selected IC beyond the best through epoch 40. This supports
retaining additional headroom with early stopping; it does not attribute the
whole bridge gain to the ceiling because sampling and arithmetic also changed.

## Measured speed

Identical sixteen-date F12 inputs and initialization, after warmup:

| Implementation | Median SAM update | Peak allocated CUDA memory |
|---|---:|---:|
| Dense FP32, pre-vectorization | 104.17 ms | 5.506 GB |
| Packed FP32, vectorized loss | 49.51 ms | 1.562 GB |
| Packed BF16, vectorized loss | 44.65 ms | 1.219 GB |

The combined measured GPU-update improvement is 2.33x. These timings exclude
loading, validation and compilation. A separate local CPU check reduced median
sixteen-date preparation from 629 to 315 ms with early packing. This is a local
CPU measurement, not a GH200 end-to-end benchmark. The bridge's F12 fits took
roughly 34–39 minutes including epoch computation and compiler startup while
using 42–50 epochs. Do not multiply the GPU benchmark into an unsupported ETA.

## Immutable evidence and restart

Bridge implementation: `57f89aa1d8fc1426f98517a713be6e55507a91f5`.
Revised training implementation, including equivalent early packing:
`0bf8130b73fe6b207c6b6f62603bf383b8f82e1f`.

The 158 benchmark/bridge evidence files were recovered and individually hashed
locally against their persistent-source inventory. Archive SHA-256:
`8f04c92a31791bca13bcbbeebe256921002a5c56d29a6c899e35b2a408f7813c`.
The accompanying JSON records the bridge, numerical and timing evidence.

Fresh run: `v2_round6_0bf8130_20260911T233404Z`.
Frozen design SHA-256:
`954b48eb3c8c3ddf730b9af1853c789a3762fb3a06c048cb46483b624a28ca0f`.
Ten score-free configuration smokes precede the 420 Session-1 F fits: matched S0
plus the nine registered candidates, fourteen folds, three seeds. No old F
scores enter this revised comparison. All 102 old F fits and three MLP P fits
remain recovered; compatible S0 and MLP P checkpoints are reused.

The changed CPU path passed 62 targeted tests, including the full synthetic
raw-store/native-fit/score/ledger/relocation acceptance flow. Eleven decision and
confirmation tests passed. Conditional confirmation preserves the original
S0/MLP initialization recipe, uses the revised F recipe, and evaluates all six
fixed seed omissions without changing C6 membership or recursively extending
seeds. Final research readouts and designation remain pending. No forward capture
or 2025/2026 model consumer was introduced.
