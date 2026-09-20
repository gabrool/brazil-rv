# M1 scalar history, ages and target assembly

The ALLOS/ISA rename amendments now reach the final cross-sectional M1 scalar
features, source ages and the existing to-close target convention. These are
verified intermediate inputs for a new derived contract. Accepted stores and
old checkpoints remain unchanged. A/B are incomplete; C/D have not started.

## Sources and comparison

Resolve `m1_scalar_assembly`, `m1_scalar_input_audit`, `m1_scalar_attribution` and
`m1_scalar_target_oracle` through `docs/v2_economic_data_scaling_run.json`.
The original build deleted its temporary raw scalar workspace. The new bounded
reconstruction therefore recovers only the necessary dated cross-sections,
with original source assignments, decimal-cent official reference prices,
observed-minute masks and decision cutoffs. It reuses the previously admitted
ALLOS/ISA scalar intermediates. No native patch reconstruction, original-source
census, financial replay or forecast scoring is repeated.

Both qualified controls match all 3,358,800 transformed value/mask cells on
90 dates and 933 permanent security axes. The native scale-only ALOS tail was
initially reused for scalars, but it lacked the scalar consistency arguments.
Only that security was qualified in each affected window; all other recovered
raw cross-sections were reused. Initial outputs and executed reproducers remain.

## Actual changes versus the sealed store

| Change | ALOS | ISAE4 | Total |
|---|---:|---:|---:|
| Usable scalar feature gains | 716 | 400 | 1,116 |
| Of these, restored eligibility alone | 645 | 291 | 936 |
| Incremental inherited raw-history support | 71 | 109 | 180 |
| Shared valid transformed values changed | 70,115 | 34,333 | 104,448 |
| Source-age cells changed | 1,254 | 646 | 1,900 |
| Raw support-fraction cells changed | 176 | 174 | 350 |
| Valid to-close outcomes added | 54 | 24 | 78 |

No valid scalar feature is lost. The prior 196 raw scalar gains become 180
additional valid transformed observations after the unchanged rank-support rule;
raw validity is not itself model validity. Corrected cross-sectional ranks can
change other securities. Of 78 new target outcomes, 76 come from eligibility and
two from the dated predecessor reference bridge. Existing supported raw target
returns remain exact; adding names changes cross-sectional ranks and residual
centering. This stage does not change the primary five-horizon daily targets.

The age scanner now snapshots the predecessor's last observation strictly before
identity effect. It admits that snapshot only after effect and knowledge, preserves
newer successor observations, and does not refresh history with later predecessor
rows. Current raw observations retain their own source age. Public pre-birth
successor coordinates stay empty. The bounded age reconstruction is seeded from
already verified sealed ages immediately before effect. Its ALOS source clock
rejoins the original clock before the window ends, so no later age tail is omitted.

Sparse `amendments/deltas.npz` records all numeric and mask changes, including the
two return-consistency bridge cells. The initial and separate native scale-tail
deltas from the previous milestone remain required for complete final assembly.

## Verification and qualifications

The actual dataset/collator checked all 88 restored eligible dates plus two
pre-event controls: 90 samples, all 933 names, full 60-session history and
10,412,280 scalar/target cells, zero mismatches. Pre-event inputs are exact. No
neural forward or GPU fitting was performed. An independent endpoint-return,
median-centering, clipping and tie-rank oracle checks 8,786 supported outcomes
with zero differences. Future-deletion prefixes remain exact. Fourteen feature
specification/age tests, including the new delayed-knowledge mutation case, and
five affected store tests pass. Ruff passes.

Initial age harness failures came from seeding inside truncated reducer warmup,
comparing a newly active age against an old inactive sentinel, and omitting retired
predecessors from the control display. They did not change source data. The final
control uses sealed pre-effect clocks and separates source state from membership.
The successful ALOS assembly was reused when qualifying the remaining ISA readout.
The independent target oracle initially emitted NumPy empty-median warnings on
zero-outcome dates; all comparisons passed, and the reproducer now skips those
empty rows. Initial executed bytes and explicit source-format resolutions are kept.

A separate final-date AERI discrepancy is preserved, not silently repaired. On
2024-12-30, the sealed M1 diagnostic marks a completed corporate boundary and
return inconsistency. Its alignment-role table records an inferred bonus first
available Dec31; the stored action arrays/term table omit that last-date term.
The bounded reconstruction from those stored arrays instead reports consistency.
All four actual target-control arrays still match. This one non-rename diagnostic
cell stays false; the underlying corporate wealth/source boundary requires its
own audit and is not an issuer-confirmed bonus.

## Remaining target-clock defect

Entry and close use dated M1 schedule endpoints. However, the to-close helper
still normalizes with fixed 405 total minutes and cutoff 345. None of the 3,717
accepted session definitions has that pair. All 89,799 existing valid auxiliary
outcomes use that old scale; 65 normalized residuals are clipped. Those clipped
values cannot exactly recover the original pre-clipping normalization. The sigma
input is current pre-decision five-minute-return RSS; replacing it with a daily
volatility estimator would be a separate hypothesis. Correct the dated clock
explicitly and recover only the necessary original scalar inputs before admitting
this optional head. The historical foundation parent keeps this loss inactive;
this receipt does not claim a corrected full-calendar auxiliary target contract.

The raw cross-section reconstruction took 21.77 seconds; each narrow ALOS control
qualification took about two seconds. Consumer audit timing is in its manifest.
These are data/audit runtimes, not fit estimates. Remaining work includes the
auxiliary clock correction, other auxiliary histories/joins, corporate wealth and
labels, final derived-store/tensor acceptance, and Stage A execution/lifecycle bounds.
