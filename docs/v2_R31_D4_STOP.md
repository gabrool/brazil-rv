# R3.1 D4 stop: an unrelated overweight holding blocks valid entries

The approved theta=1 clarification worked: all 12 baseline panels reproduce
all 48 sealed scenario daily tables exactly. The fresh grid stopped in its
second cell, before policy selection, on 32 D4 name-cap flags in GBDT/F1
under open borrow. The stopped root and exact-failure diagnosis are sealed;
both source inventories remain hash-identical.

## Cause and proposed correction

All 32 flags occur on 2023-12-27. A held short in `BRCPLECDAM13` lacks a print
and has a partial risk exit pending from the previous session. Its projected
weight is -5.039847% of NAV after the existing pending exit. The entry gate
checks the largest weight anywhere in the book, so this old position blocks
unrelated new names whose own weights range from 1.806983% to exactly 5%.

For example, the proposed `BRMILSACNOR2` long is 2.839772% of NAV, with prior
projected gross 1.96736 and net 0.09814. Only the unrelated name-cap check
fails. These are entry attempts, not 32 missing filled positions. The original
unfilled position is real and remains in the ledger.

The prepared correction applies the unchanged name cap to the entry candidate.
Whole-book gross and net checks remain binding. Existing concentration breaches
still trigger risk exits and remain visible while no execution price exists.
No synthetic fill, pending-exit repricing, changed cap, or weakened D4 counter
is needed. A focused fixture fails before the correction and passes after it.
The fixture also checks that the old overweight inventory and its unfilled
partial exit remain in the book. The corrected sweep has not been run.
All 151 targeted ledger, evaluator, research-round and policy tests pass; Ruff is clean.

Pass-5 section 6 explicitly stops on any D1-D5 flag. The user explicitly approved
the tested correction in `1d19da2` and a fresh R3.1 restart. The effective Round-3
registration now records it; the stopped root remains immutable. No partial grid
may select a policy.

## Observed scope

Eighteen panels completed and one stopped part-way through its scenario set:
75 scenario ledgers in total. The stopped GBDT/F1 cell completed balance,
strict and open borrow; its sterile comparator was not run. Other candidates
and cells after this point were not run.

| Policy | Candidate/fold | Net excess bps/day | Mean gross | Liquid original-trade contribution bps/day | Status |
|---|---|---:|---:|---:|---|
| equal | arm_B/F1 | 0.0164 | 1.9357 | 1.6740 | completed |
| equal | arm_B/F2 | -1.8853 | 1.9291 | -1.4672 | completed |
| equal | arm_B/F3 | 10.7486 | 1.9097 | 10.1176 | completed |
| equal | ensemble/F1 | 5.8282 | 1.9280 | 4.6770 | completed |
| equal | ensemble/F2 | -1.5363 | 1.9518 | 1.4166 | completed |
| equal | ensemble/F3 | 3.5981 | 1.9586 | 2.1578 | completed |
| equal | gbdt/F1 | 1.9409 | 1.9485 | 3.4612 | completed |
| equal | gbdt/F2 | -0.9111 | 1.9451 | 0.5821 | completed |
| equal | gbdt/F3 | -0.7802 | 1.9802 | -1.8889 | completed |
| equal | momentum/F1 | 1.5924 | 1.9032 | 0.0041 | completed |
| equal | momentum/F2 | -1.2842 | 1.8729 | 2.4843 | completed |
| equal | momentum/F3 | 9.2041 | 1.9736 | 10.1947 | completed |
| inverse sigma | arm_B/F1 | -2.0582 | 1.7661 | 0.5827 | completed |
| inverse sigma | arm_B/F2 | -2.5355 | 1.8594 | -1.6370 | completed |
| inverse sigma | arm_B/F3 | 4.9924 | 1.8120 | 4.4927 | completed |
| inverse sigma | ensemble/F1 | 3.3169 | 1.7242 | 3.5600 | completed |
| inverse sigma | ensemble/F2 | -1.1272 | 1.7985 | 0.6817 | completed |
| inverse sigma | ensemble/F3 | 2.4680 | 1.7706 | 1.9801 | completed |
| inverse sigma | gbdt/F1 | 2.4678 | 1.7863 | 2.0832 | D4 stop in open borrow |

All completed headline books pass gross, stale-inventory and occupancy bounds.
The following paired results describe the already observed second cell only;
they do not select or qualify a policy from the incomplete grid.

| Candidate | Inverse-minus-current net bps/day [95% CI] | Traded neutral IC | Liquid contribution bps/day [95% CI] |
|---|---:|---:|---:|
| arm_B | -2.8504 [-6.8946, -0.9896] | 0.024758 | 1.1729 [-2.1791, 4.9043] |
| ensemble | -1.0778 [-4.4437, 1.6941] | 0.027183 | 2.0732 [0.2122, 6.7310] |

Liquidity figures attribute original entries with prior-20 median BRL volume
at least R$20m. They use full-book NAV, actual equity costs/borrow and the
registered explicit allocation of shared hedge/funding effects; there are
no replacement names or leverage rescaling. Readouts have no selection weight.

No source score or model was recomputed. Protected source fields, raw input
hashes and score provenance remain exact. No new store, fit, protected-date
payload or paid instance was accessed. Intraday coverage, store rebuild and
acceptance, and Round 1' remain after completion of this sweep.

## Roots and verification

### stopped_sweep

`D:\quant-data\b3\processed\model_runs\v2_execution_sweep_29fb045_20260909T003119Z`

Exact 46-file inventory verified.

- access_audit.json: `b52d5cf821ba44073d43a6ff7473a708c5429ad9353b32e44d18b037bad0091e`
- artifact_inventory.json: `a74798e30dddb0edb2aed456ae1c476ffdbbf686ac849e91e3e59c150961450d`
- failure.json: `43e7a52ef5b1ced321e1fa5def0902e64372388198c083572ab677ac4f5af155`
- frozen_design.json: `2ce44e477b456574cb7574fa8887d51f4514149b6d353096551ac0d607aa73ea`

### exact_failure_diagnosis

`D:\quant-data\b3\processed\model_runs\v2_execution_d4_diagnosis_29fb045_20260909`

Exact 4-file inventory verified.

- access_audit.json: `721c257c9b0be48a30022734018cbf2cb01e2870631e9cac6e5c08120614a644`
- artifact_inventory.json: `b5c64a268a304e47135be0d1ae05841f7dbfdbbeb9fe0d7322a97b6dadddcee4`
- diagnosis.json: `f98ab33e3a547fac700cc1a87559fa05b258f84535b34b74b584855f4c023fd4`

[Machine-readable evidence](v2_r31_d4_stop_evidence.json) retains the exact
gate counts, first full order trace, all 32 proposed candidate weights,
partial-grid readouts, panel hashes and root identities.
