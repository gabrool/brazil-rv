# Saved conditioning and actual input audit

2026-09-19. All 120 foundation fit manifests and checkpoint preprocessing payloads
agree with the accepted source contract. This is a Stage B correctness result; no
model forecast or profitability was recomputed.

`ops/audit_fit_conditioning.py` resolves the foundation and data-store pointers,
checks their source hashes and development date bounds, and independently rebuilds
all 15 distinct preprocessing coordinate systems. The reconstruction uses 34,852,386
unique scalar-fit observations, fit-only medians/IQRs, nonzero-MAD fallback, support
and variation flags, typed passthrough fields and exact P-to-F inheritance. Common
cross-market scalars use one observation per date. F-only support is admitted only
under the saved inheritance rule. Every saved statistic matches exactly.

`ops/verify_conditioned_tensors.py` separately checks all 120 selected-checkpoint
hashes and embedded preprocessing payloads against their manifests. Source feature
names and types independently determine common/per-name routing and transformation.
On three stratified fit dates per coordinate system, 45 full-population samples
retain all 60 sessions. All 1,856,628 tested packed float32/bool input cells match,
including validity, field ages, common diagnostics, padding and permanent security
indices. No training or neural inference is performed.

The scalar reconstruction took 25.38 seconds; the stricter checkpoint/tensor check
took 24.85 seconds, on CPU. These are not neural-fit estimates. Two harness mistakes
were corrected before completion: closing the store through the dataset, and using
compact positions instead of permanent security indices in the independent expected
index. Neither was a data or model defect. No accepted dataset or old fit was edited.

The run pointer binds `fit_conditioning_audit` and `conditioned_tensor_audit`, with
the full fit/checkpoint list, source hashes, exact field counts and reproduction
scripts. The separate completed auxiliary audit remains the producer-to-store
boundary. Upstream source publication/revisions, financial denominators, complete
identity propagation, shareholder wealth and labels still require Stage B work.
Passing these checks cannot make an uncertain historical source vintage causal.
