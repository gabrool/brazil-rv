# Round 6 model implementation, fixed before neural fitting

E8 keeps S0's four-channel slow input encoding (values, feature validity, age,
history validity), 64-wide projection and normalization. It uses only the final
permitted row, t-1, including that row's masks and ages. Two 64-wide residual
SwiGLU blocks with inner width 48 and dropout 0.1 replace the GRU. S0's pooling,
fusion, two trunk blocks and target heads remain unchanged. Missing t-1 data
never triggers substitution of an older row. E8 receives architecture-matched
Stage-P training as specified in the original implementation clarification.

E4 changes only the learning-rate multiplier on parameters actually transferred
from S0 Stage P, from 0.3 to 1.0. Additive family projections retain the full
learning rate in all family arms. The time-decay arm changes Stage-F date-pair
sampling to the registered 756-session half-life; Stage P remains uniform.

The explicit parent-store transfer path proves exact non-sidecar arrays, tables,
axes, slow feature definitions, targets and remaining input semantics before
loading a sealed S0 checkpoint. It accepts missing zero-initialized sidecar
projection keys only; missing parent keys, changed parent dimensions and changed
architectures fail. Seed identity is checked. Every run records the proof,
missing keys and transferred parameter count.

Before fitting, all three actual S0 checkpoints (11/29/47) loaded successfully
against the accepted Round-5 store and an events sidecar: 34 parent parameter
tensors retained, 73 non-sidecar array records and all tables/axes exact, and
only the new events projection initialized to zero. The local evidence is
`round6_20260911T111805Z/s0_actual_transfer_preflight.json`. The 41 targeted
training/model/transfer tests passed. The difference from Round 5's count of 76
protected arrays is scope: this transfer proof excludes all six existing
sidecar arrays, while the Round-5 build replaced only three lending arrays.
