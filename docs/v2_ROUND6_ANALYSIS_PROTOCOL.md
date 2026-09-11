# Round 6 analysis implementation

This records mechanical analysis choices under the user's standing authority to
choose the recommended interpretation and preserve both speed and quality. It
does not change the frozen training commit, data, folds, seed roster, stopping
rules or execution policy. The implementation was prepared while Session 1 was
running, before inspecting any new candidate OOF result. Historical S0 results
and score-free startup diagnostics were already known.

## Full-panel decision

The all-fourteen-fold primary IC leader is selected among arms with defined IC
and nonnegative original-rate headline economics. Candidate order for exact ties
is S0, the registered Session 1 order, the four fixed Session 2 arms, C6 (if
nonempty), best-single fresh P and C6 fresh P (if nonempty). An economics override
requires both a paired IC interval including zero and a strictly positive paired
net-economics interval against that leader. Multiple qualifiers use the largest
absolute headline net point, with the same tie order. These are the existing
Round-4 leader/override rules applied to the Round-6 roster.

The informative-fold comparison to S0 determines the registered close-decision
confirmation trigger. All-fold and informative deltas are both retained. C6's
roster is the full three-seed Session 1 point-estimate rule; omitting a seed only
reports roster sensitivity and never changes or refits C6.

Persistence and turnover accompany the decision for explicit tradeoff review.
There is no preregistered numerical veto for either diagnostic; the analysis
does not manufacture a threshold after observing them. Placeholder and
latest-vintage cost sensitivities have zero promotion weight.

## Matched comparisons and seed sensitivity

Reuse accepted full-panel evaluations. Compute exact-common-population pairs to
S0, then any missing eligible-arm pairs to the IC leader. This supplies every
comparison required by the decision while avoiding a quadratic all-pairs sweep.
If seed-audit exclusions change the leader, compute the newly required pairs.
Keep the fixed 20-session, fold-preserving, 10,000-replication bootstrap.

Replay all three fixed two-seed panels (29/47, 11/47, 11/29) for every arm and
every fold. Report each excluded seed's primary IC and matched delta to S0 as
descriptive evidence, without additional fitted networks or portfolio gates for
single seeds. A single-seed rank ensemble preserves the per-head Spearman metric.

As in the accepted Round-4 audit, an isolated non-baseline quintile-occupancy
failure disqualifies that arm from all four panels' choices. Its failed report
is retained without an accepted marker. S0 failures, mixed failures, D1-D5,
other risk bounds, identity errors and implementation errors stop the audit.
No threshold is relaxed. The same non-null provisional designation must occur
in the full panel and all three omissions for a seed-stable designation.
Otherwise retain S0 as the research comparator and label the parent choice
inconclusive; separately disclose whether S0 remains economically eligible in
all four panels. This is sensitivity analysis, not independent replication.

## Conditional confirmation scope

The trigger remains an informative paired IC lower bound within +/-0.001 of
zero (inclusive) or an economics override. A retained S0 cannot trigger a
meaningless self-comparison refit. Confirm the designated candidate and S0 with
61/79/97. If an override selected a candidate other than the IC leader, also
confirm that leader: the override asserts a candidate-versus-leader claim, which
must have matched seeds on both sides. Three arms are needed only when those
identities are distinct. Do not refit every screened arm or invent archived
S0 confirmation seeds; the old archive contains only 11/29/47.

The analysis reports the trigger and required roster. The separate confirmation
execution and final acceptance readouts must finish before a triggered result
can be described as confirmed. No candidate outcome is implied by this document.

## Implementation and operations

`round6_decisions` builds the full-panel review after both sessions.
`round6_seed_audit panel --omitted 11|29|47` evaluates one fixed panel;
`round6_seed_audit finish` applies exclusions consistently and reports stability.
Run the three panel processes concurrently only when memory permits. Their
fresh output directories preserve partial failures for diagnosis.

Analysis code is isolated on the `round6-readouts` branch while the GPU checkout
remains at its frozen training commit. Recover and validate completed training
artifacts before evaluation. Publish both implementation and evidence; merge the
analysis branch after frozen training work is complete. No 2025/2026 consumer
access, forward capture, or Round 7 experiments are authorized.

`round6_confirmation.py` can be copied outside the training checkout and executed
by absolute path using that checkout's Python environment. Its imports must
resolve to the original frozen package (do not set the analysis `PYTHONPATH`).
The planner binds its own file hash separately and requires the completed seed
audit's matching design and triggered roster. Its `smoke`, `p` and `f` phases
are sequential. New S0 P checkpoints use the current accepted schema; other
compatible arms reuse those same-seed checkpoints, while MLP and fresh-P variants
use their own. The planner never sends these fresh checkpoints through the
older archived-store transfer path.
