# Round 6 implementation decisions fixed before diagnostics or new scores

This document resolves mechanical ambiguities in the supplied registration under
Gabriel's standing authorization to choose the recommended course and his new
instruction to optimize speed while preserving experiment quality. The supplied
registration remains preserved in `v2_round6.md`.

## Initialization and experiment identity

The residual MLP replaces the GRU with a small encoder of the latest permitted
slow row. GRU recurrent weights cannot initialize this encoder. E8 therefore gets
three architecture-matched Stage-P runs on the same original P dates, targets
and selection rule. This adds three inexpensive P runs, preserves comparable
pretraining, and avoids calling a partially incompatible transfer an unchanged
S0 initialization. All recurrent family arms reuse the exact three S0 P
checkpoints, except the explicitly registered fresh-P variants. The learning-rate
multiplier and time-decay arms also reuse the S0 checkpoints.

An unchanged parent array identity is proved from the source and destination
manifests and consumed feature definitions, not from equal overall store hashes.
Only new residual projection parameters may be absent in a reused recurrent
checkpoint. Their zero initialization and missing-key list enter the transfer
audit. Missing or reshaped parent parameters remain errors. Training data,
targets, activity, chronology and parent feature definitions must remain exact.

C6 contains only the six session-1 data families with positive informative-fold
paired primary IC point estimates. The best single family for fresh P uses the
largest such point estimate; ties follow the registered session-1 order. These
are roster rules, distinct from final eligible all-fold promotion. If C6 is empty,
its identity with S0 is recorded instead of fabricating a novel combination.

## CPU sign readout

Report each fundamental and event field against D5 on dates through 2024-12-30.
The main pooled/by-year summary averages daily cross-sectional Spearman ICs with
at least 20 active, field-valid, target-valid names. Include supported name-days,
defined dates, and the literal pooled name-day Spearman as a secondary diagnostic.
Use the field's own validity within the family population. No imputation or
sign-dependent feature selection is permitted. Weak or contrary signs prompt a
bounded source/alignment check; only independently demonstrated implementation
or source errors justify a correction, never the desired sign itself. The stated
0.005–0.02 expectation is context, not an acceptance threshold.

## Source and economic interpretations

Brent uses the newly authorized next-B3-session decision convention. Preserve
its assessed-price/latest-vintage provenance; this convention is not evidence
that the archive retained every first publication. New source history remains
bounded to 2024-12-30. An ADR return gap requires dated B3/ADR identity and actual
adjacent return endpoints. A common multiplicative adjustment cancels in log
returns; changing adjustment factors, dividends, corrections and rounding do not
automatically cancel. Record the chosen return basis and mismatched action dates.

Foreign flow is explicitly `published_total_difference`, with each value timed
to the publication that reveals it. Within a month, subtract the previous
comparable published MTD total. At a verified new-month reset, the zero reference
is the start of that month, never the previous month's final total. A multi-day
gap cannot be labelled a one-session flow; retain its interval and mask any
unsupported exact one/five-session feature. Methodology changes break the
difference unless a comparable bridge is documented. No current snapshot is
collected or used to reconstruct historical publication vintages.

Late-2024 lending additions are a separately labelled latest-vintage input
extension, preserving all originally accepted observations. Their 1,535 exact
overlaps support the admission but do not prove an unrevised full history.

The requested `placeholder_v2` estimated using 2023–2024 observed rates is a
**hindsight cost sensitivity** on earlier books. It is not a decision-time
feature and cannot govern promotion. Keep flat-2%/original admitted observed
rates as the causal headline comparison, and report the requested sensitivity
beside it for every arm. Freeze shrinkage strength and liquidity grouping from
input/cost data before any Round-6 economic result; do not tune them to returns.
The observed latest-vintage rate replay is also labelled separately.

## Speed and operational scope

Reuse the sealed S0 predictions, original source archives, approved family
parquets, and the common CPU evidence. Rebuild only changed sidecar families;
carry every unrelated array/table and both axes exactly. Stage source acquisition
and local diagnostics independently where possible, without repeating completed
Round-5 audits. Failed archive routes receive a bounded probe and documented
outcome rather than an open-ended search.

Keep all fourteen folds, matched seeds, model size, stopping rules, masks,
selection and readouts. Measure score-free configuration smokes and throughput
at useful concurrency before estimating a paid session. One compiled training
graph plus a separate evaluation graph is permitted; additional training graphs
must be diagnosed. Reuse compilation artifacts only when configuration/compiler
identity matches. No throughput number is assumed from the document's ambiguous
per-fit versus concurrent-wall-time arithmetic.

The user's standing explicit paid-compute authorization applies after successful
CPU checks and a clean committed freeze. Use only the launcher in the `.txt`
handoff. Recover and verify all required artifacts before exact-ID termination
and two provider inventory reads. Forward capture and 2025/2026 consumer access
remain prohibited. Round 6 ends with its report; Round 7 is not launched.
