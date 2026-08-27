# Learned-policy OOF prediction manufacture

Frozen before the first OOF trajectory. This registration manufactures honest
deployed-quality ranks over all 716 TRAIN dates for future execution-policy
training. It is not a new alpha experiment, does not train the policy, and does
not access official validation or the permanently spent held-out test.

## Frozen inputs and splits

The canonical causal feature store and its exact identity are resolved at freeze.
The exact Experiment-54 frozen design supplies the three Experiment-41 Patience
comparison archives; their wrappers must load under the sealed discovery
contract before freeze. The Part-1 purged split is exactly five chronological
TRAIN blocks with five adjacent sessions embargoed on both sides. Held-out blocks
cover every TRAIN date exactly once. The emitted split manifest and every fold's
fit, held-out, and embargo date lists and hashes are immutable.

## Fifty monitor-free refits

Each of five folds is trained with seeds
`11/29/47/61/79/97/113/131/149/167`. Every run uses the deployed store-v2
zeroing, depth-six shared causal TCN, all three incumbent heads, temperature
`0.50`, uniform fit dates, SAM-AdamW, and exactly 20 epochs. No held-out score is
computed during training and no monitor, stopping, checkpoint choice, or retry
may depend on held-out outcomes. Only epoch-20 raw and final EMA-0.995 predictions
are archived; EMA-0.995 is the OOF member. At most two training processes run.

The known protocol mismatch is explicit: deployed members use official-monitor
Patience, while OOF members use the only monitor-free choice, fixed-20 EMA-0.995.
The final report compares OOF-EMA with the exact Experiment-41 Patience ranks on
the C/A/B windows using IC and rank correlation. The future policy consumes only
causal cross-sectional ranks.

## Canonical archive and proof

Within each held-out fold, the ten members are tie-aware rank averaged using only
point-in-time membership and data-readiness. The five fold archives are assembled
in canonical sample order. Every reference row records its source fold. The source
manifest binds all 50 run manifests, member predictions, references, fold hashes,
and fit-exclusion proofs. The OOF execution loader must independently reconstruct
the canonical purged folds and reject a changed fit list, an emitted fit/embargo
date, a missing run binding, a changed file hash, any official/test access, or
incomplete coverage. The accepted archive covers all 716 TRAIN dates and all 55
refreshes exactly once without reading a label mask at load time.

All prediction/reference archives, histories, manifests, calibration, results,
and audits are retained in an immutable root. Epoch-20 state checkpoints are
hash-inventoried and removed only after the final archive passes the loader.
`official_validation_accessed=false` and `test_accessed=false` throughout.
