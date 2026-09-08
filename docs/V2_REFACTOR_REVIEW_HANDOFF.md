# Brazil-RV v2 refactor: implementation review handoff

This note reconciles the implementation choices made across the multi-day
refactor proposals, the pass-3 and pass-4 review rounds, rev3 neutral
evaluation, and the completed rev3 Round-1/Round-2 research program. It is
written for the LLMs that proposed the edits. The detailed chronological
evidence and artifact hashes remain in `V2_EXPERIMENT_LOG.md`.

## Outcome

The repository now has a deliberately incompatible multi-day pipeline rather
than an intraday model with longer labels grafted onto it. The current V3 store
uses one 15:45 decision row, causal daily and intraday information, permanent
security identity, explicit validity/age masks, a persistent position and cash
ledger, common-population evaluation, and native five-minute fast inputs. The
real-data development store is
`v2_daily_store_8021e42_20260906T202315Z`, manifest SHA-256
`deb9ca8449c5b9a83bf25ac19218069c006e183361b6f6ba836717ade63b491b`.

Rev3 Round 1 chose `b_intraday` as the neural parent. Rev3 Round 2 selected
uniform-pretrain Arm B over fine-only Arm A and decay-pretrain Arm C. Arm B's
pooled characteristic-neutral IC is `0.0271682` and its headline net excess is
`11.8610` bps/day. The fixed network/GBDT rank ensemble has the highest pooled
neutral IC, `0.0291273`, and is the registered Round-2 research designation.
Its economics are weaker than the network alone and its net interval spans
zero, so the designation is a signal-quality result, not a production claim.
Official validation and the permanently spent test were never opened; no
deployment changed.

## Design choices adopted

### 1. One decision clock, enforced once

The canonical row is the information available at 15:45
`America/Sao_Paulo`. Daily market state ends at the previous completed session.
Same-day intraday inputs contain only completed bars before the decision and
exclude the entry bar. Publication-lagged fields may enter only when their
source clock says they are available. Consumers do not add private extra lags.

This resolves the disagreement between stage-specific shifting and a single
canonical view in favor of the latter. A single clock is easier to audit and
prevents an apparently conservative second lag from silently changing the
research question.

### 2. Identity is permanent; ticker continuity is evidence, not authority

ISIN-backed `security_id` is the model identity and ticker is a dated
attribute. Same-ticker ISIN transitions are retained as auditable succession
candidates, but ticker continuity alone does not establish economic conversion
terms. Feature-history or shareholder-wealth continuity across an identity
change requires a point-in-time accepted transition with the appropriate
share/cash terms. Unknown terms remain unresolved.

This is stricter than the early suggestion to join every consecutive
same-ticker ISIN pair. That rule can be useful for finding candidates, but it
cannot prove that one predecessor share became one successor share or that a
cash component was absent.

### 3. Price, shareholder wealth, and model target are separate objects

Raw price return is not treated as shareholder return. The store keeps raw
price, contractual/inferred shareholder-wealth outcomes, validity, terminal
wealth, and normalized model targets as distinct families. Resolved actions
already represented in the wealth chain do not invalidate a wealth return;
unresolved actions and raw-series restarts do. The primary rev3 evaluation is a
deterministic characteristic-neutral virtual view constructed row-by-row in
float64. The prior volatility-scaled target remains an exact legacy diagnostic
on its original population, not the rev3 optimization target.

The earlier approach that removed a cross-sectional median after volatility
scaling and used same-day volatility was rejected. It mixed the desired
cross-sectional transformation with information unavailable at the decision
time.

### 4. Evaluation populations are explicit and comparable

D1/D2/D3/D5 primary legacy readouts use the original common four-horizon
population. The neutral view begins from that population and then intersects
the required characteristic validity. D10 remains per-horizon and diagnostic.
Fold and block boundaries are preserved in every interval and paired
comparison. Unsupported statistics serialize as `null`; they are never
coerced to zero.

This choice was made after the first rev3 acceptance correctly exposed that a
per-horizon rewrite changed legacy numbers even though the underlying payload
was identical. The repair restored exact legacy identity without weakening the
new neutral evaluation.

### 5. Economics use a persistent ledger

The simulator fixes intended orders before later fill observations and carries
signed shares, cash, funding, costs, failed exits, corporate claims, and
terminal settlement across sessions. It no longer treats each day as an
independent cross-section or discards the whole day when one name is
unavailable. Long and short capacity refill independently; submitted exits and
same-auction replacements follow explicit occupancy rules. Risk trims reduce
only excess exposure.

Held names may remain through up to five consecutive sessions of temporary
ineligibility. Entry still requires eligibility; session six submits the
registered exit. A separate ten-session no-print path produces a labelled
terminal settlement rather than silently marking or deleting the position.
Unresolved/stale notional, settlement incidence, gross shortfall components,
and entry/exit defects remain visible diagnostics.

Several pass-4 results missed predeclared gross or settlement gates. Those
results were stopped and recorded. Bounds were changed only under a new
pre-result contract after diagnosing the mechanism; no observed score was
retried under a friendlier rule.

### 6. Missingness is represented, not invented

Every model feature has an ordered `FeatureSpec`, payload, validity, and true
source age. Neural inputs zero invalid cells at the dataset boundary and assert
that every tensor reaching the model is finite. Masked reductions use
`where(mask, value, 0)` semantics rather than `value * mask`, because NaN times
zero is still NaN. Zero-lookback names and empty fast-patch sets have explicit
finite behavior. GBDT receives NaN only where the same canonical mask says the
field is invalid.

This structural boundary fix replaced the tempting evaluator-only workaround
of dropping non-finite names. Dropping them would change the evaluated
population and conceal upstream contamination.

### 7. The real store is bounded-memory and relocatable

The builder computes one feature or sidecar family at a time, uses float64 only
inside the bounded computation, writes float32 arrays directly to preallocated
memmaps, and releases the raw panel before the next family. Rank-gauss runs one
date cross-section at a time. Daily OHLC state is written once and reread from
memmap. The V3 real build peaked at `7.138950` GiB and passed the unchanged
measured 8-GiB invariant.

The user explicitly waived only a separate 10-GiB host-admission check after
freeing memory. The measured-build 8-GiB regression guard was not waived and
did pass. An earlier failure from materializing the event archive as a Python
list was removed rather than excused.

External artifact identity is by SHA-256, not absolute pathname. Immutable
manifests keep their original paths, while an environment-selected override
maps root prefixes on another OS. Every resolved file is hash-verified and the
resolution is recorded in the run manifest.

### 8. Fast inputs are native and freshly trained

The clean v2 path uses compact, separately masked five-minute patches and
fresh fast-branch weights. The v1 fast checkpoint is retained only as
contaminated historical diagnostic evidence and is not a dependency of the
current model. Fixed 16-name compact padding is used where compiled training
requires shape stability; masks ensure padding cannot become a sample.

The first Round-2 smoke exposed changing active-name dimensions as a Dynamo
recompile failure. The padding repair addressed compilation shape only and was
validated by graph-count audits; it did not change the logical batch or score.

### 9. Sidecars are capability-gated

Only lending and oddlot are materialized as enabled sidecars in the V3 store.
Options, rebalance, events, and fundamentals have archival data, but the exact
source semantics required by the current `FeatureSpec` roster are unavailable;
they are recorded as source-missing rather than filled, aliased, or silently
substituted. Round-1 rung D consumed only materialized families and required
every omission to be explicitly source-missing.

External-family leakage protection is publication-clock reconstruction plus a
name-clustered, ADV20-stratified composition audit. Internal features retain
the unconditional cell-level survival gap gate. This implements the refined
proposal: external coverage composition is not itself a leakage proof, while
publication-time reproducibility is.

### 10. Completed artifacts are immutable, but reporting can be repaired

Every run is commit-, design-, input-, and hash-bound. A completed candidate is
never rerun merely because later orchestration or reporting failed. Recovery
commands verify existing histories, checkpoints, scores, evaluation inputs,
and outputs, and run only genuinely absent registered jobs.

Round 2 exercised this rule twice. Four never-started Stage-J jobs were run
after a launcher graph-count assumption failed; 23 completed trajectories were
not touched. Later reporting fixes reused all nine aggregates and six comparator
evaluations, re-derived each seed-rank aggregate exactly, and recomputed zero
scores and zero evaluations. JSON arrays are now valid sealable research
artifacts, and sparse unsupported intervals remain JSON `null`.

## Suggestions intentionally not fully implemented

1. **No production-grade corporate-action claim.** The current real store uses
   a clearly labelled development tier inferred from COTAHIST `DISMES`.
   Independently verified historical share ratios, cash terms, conversion
   terms, payment dates, and terminal claims remain incomplete. Inferred terms
   were sufficient only because pass 4 explicitly authorized the shortest path
   to an honest development number without changing leakage, identity, clock,
   or accounting rules.

2. **No claim of an authoritative exchange calendar.** The current schedule is
   reconstructed and cross-checked with zero unexplained exceptions. A dated
   B3-authoritative session source is still required for a stronger operational
   or production claim.

3. **No auction-execution claim.** Close/auction behavior remains the labelled
   research proxy allowed by the development contract. Verified historical
   auction marks and capacity were not available and were not fabricated.

4. **No fully executable historical borrow claim.** D+1 lending rates are
   inverted from the sealed transform when present; balance/age controls
   shortability, and missing or stale rates receive the registered floor. Raw
   lending PDFs stop at 2024-06-28. This is sufficient for the registered
   conditional-borrow development analysis, not proof that every historical
   short could have been located at the reported rate.

5. **Four sidecar groups remain disabled.** Archives do not automatically imply
   contract-compatible model fields. Options, rebalance, events, and
   fundamentals will require new source adapters or a new registration before
   they can enter a candidate. Oddlot raw data can be rebuilt beyond its current
   June-2024 derived cutoff; lending requires newly acquired raw files.

6. **D10 is not promoted to a primary horizon.** It remains diagnostic because
   the registered common evaluation and overlap rules center on D1/D2/D3/D5.
   Promoting D10 would be a new research question, not an implementation fix.

7. **No backward-compatibility layer for old stores or checkpoints.** Earlier
   v2 artifacts are immutable evidence but are rejected by current schemas.
   Silent translation would make it impossible to know which decision clock,
   target, mask, or ledger produced a number.

8. **No resurrection of the first Round-1/Round-2 scores.** That registration
   was voided after the action, target, overlap, and stateless-ledger defects
   were identified. Those artifacts remain engineering evidence only.

9. **No official-validation, held-out-test, or deployment step.** The entire
   completed rev3 program uses development folds. The selected ensemble is a
   parent for further development research, not an authorized deployed model.

## Reproducibility anchors

- V3 store manifest:
  `deb9ca8449c5b9a83bf25ac19218069c006e183361b6f6ba836717ade63b491b`
- Rev3 Round-1 result:
  `e51a46e1ddfed7730c178268c083da2f375ed92a660bbbeac4bb3f2619264daf`
- Round-2 score implementation:
  `916ac0b7e6e3ab16dea72dbf480ebb086a8981b0`
- Round-2 frozen design:
  `8ca83757f48204e94db153c3ff1b9891f1447e255510963433d7c5994d8d0b16`
- Round-2 result:
  `e98792213e16615c6e44ba3036ed829100940eeef0b8a9a175a9a2b2f752f9a0`
- Round-2 log-inclusive inventory:
  `39ee66e731aa51bf3859a70c974abdf2ee69301dff01f05fa1dfb626d3d755ce`
- Final reporting/sealing implementation before documentation:
  `2ee5334aef9bebbd2aa9088d4156a7c6919d64e4`
- CPU data-currency audit:
  `eb8a1370d1a4929657c9595bcc2ac49b40c85070c36230d2fdf1fdbcb483977f`

The complete final implementation passed Ruff, Python compilation, and all 901
tests on Linux before sealing. The exact GH200 used for Round 2 was terminated
and confirmed absent twice after all artifacts and logs were secured.
