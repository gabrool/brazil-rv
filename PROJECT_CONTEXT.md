# Brazil-RV project context

Last verified: 2026-09-10.

## Purpose and current research state

Brazil-RV has two offline research programs. The accepted v1 intraday system
and its deployed research recipe remain unchanged and reproducible. V2 is an
intentionally incompatible multi-day system that predicts five daily horizons
from one 15:45 decision snapshot. Its repaired development V3 store is
`v2_daily_store_3d67624_20260909T100323Z` (manifest SHA-256
`db4f751d47133a6611733739ad9bfc15721452218365e338fec61a6b608bea64`).
It ends on 2024-12-30, with repaired intraday fields, lending v2 and re-derived
oddlot. Full-build peak RSS is 6.7059 GiB. Untouched arrays match the historical
8021e42 store's authorized slice; native-fast reconstruction has zero value or
mask error. The new endpoint-dependent survival audit flag reconstructs exactly.
The economic-beta arrays are byte-identical to rev4f, with a new store binding.
See [the input audit](docs/v2_REBUILT_STORE.md). Fresh 16-book development-grade
acceptance passed, including 75 exact legacy-control IC comparisons. Two F3
control books retain unresolved economics under the frozen terminal-settlement
rule. [Acceptance evidence](docs/v2_ACCEPTANCE_REBUILT.md) preserves these limits.
Round 1' is complete and sealed on 21 panels. `a_slow` IC is 0.015301 and
`b_intraday` IC is 0.010465; paired b-minus-a is -0.004836 [-0.009128, 0.001347].
Paired net excess is -0.874 [-5.439, 7.458] bps/day. The repaired intraday inputs
have not demonstrated a GBDT improvement. The registered parent remains
`b_intraday`; native-TCN contribution is tested separately in Round 3.
See [the paired report](docs/v2_ROUND1_PRIME.md).
Round 3 is complete on that repaired store. Its registered rule retains six-seed
Arm B and designates the network over the GBDT and ensemble, with no economics
override. B6 neutral IC is 0.025078 [0.016108, 0.035333], and net excess is
4.436 [1.307, 11.653] bps/day. The seed extension's paired IC and net intervals
span zero. Native-fast B3 minus matched fast-off has neutral IC -0.000200
[-0.001068, 0.000718] and net 0.075 [-1.666, 2.292] bps/day; D1 also spans zero.
The fast stream has not demonstrated incremental value at these horizons.
All four registered development criteria for a future 2025 read are met, but
this pass does not spend that read. Round 4 requires a new registration.
All 24 aggregate/comparator books pass the registered gates; the complete sealed
732-file root is hash-verified locally and on the persistent Lambda filesystem.
See [the Round-3 report](docs/v2_ROUND3.md) and [operations evidence](docs/v2_round3_operations.json).
The historical 8021e42 store remains immutable and binds the sealed replays.
It is explicitly development-grade: the calendar is reconstructed and the
corporate-action terms are inferred from COTAHIST `DISMES`, not independently
verified contractual terms.

Round 1 is now re-baselined under rev4f on the 18 sealed score panels, with
every protected field bit-identical and headline gates passed. The `b_intraday`
GBDT retains neutral IC `0.0199906`; corrected net excess is `0.0763` bps/day.
Momentum retains IC `0.0214729`; corrected net excess is `3.2190` bps/day.
Both economics intervals span zero. See [the full before/after report](docs/v2_PASS5_REBASELINE.md).
Round 2 is also re-baselined under rev4f on its 12 sealed panels, preserving all
protected fields. Arm B remains the selected network: neutral IC `0.0247766`,
net excess `3.0222` bps/day [0.1445, 9.9769]. The ensemble remains the overall
research designation: IC `0.0265307`, net `2.6377` [-0.1505, 9.8608]. Paired
B−A and ensemble-versus-GBDT economics intervals still span zero; no economics
override fired. These are development research results. See
[the Round-2 before/after report](docs/v2_PASS5_ROUND2_REBASELINE.md).
R3.1 is complete on 240 panels / 960 ledgers, with zero D1-D5 flags and exact
reproduction of all 48 sealed baseline daily tables. Its registered joint-first
rule adopts unsmoothed D3/D5/D10, equal sizing and buffer 9 per quintile, labelled
`execution_parameter_selected_in_sample`. Arm B's paired gain is 3.2533 bps/day
[-0.4199, 6.5484]; the horizon-only buffer-6 cell has the higher gain but is an
individual change. This execution recipe applies to the remaining program;
sealed replays retain their original policy. See
[the sweep report](docs/v2_R31_EXECUTION_SWEEP.md). The user subsequently authorized
autonomous completion, including the GH200 work after successful CPU checks,
with verified artifact recovery and instance termination. The repaired intraday
archive coverage is reported in [the 2024 table](docs/v2_INTRADAY_COVERAGE.md).
Fresh acceptance and Round 1' explicitly bind the selected policy. Round 1' is
the paired `b_intraday` minus `a_slow` ablation; the registered parent stays
`b_intraday`, without reopening the rung ladder. Its diagnostic history age is
decoded to integer sessions with `np.rint`, with unchanged validity and an int32
hash. Five exact-60-session boundary cells move to their correct bucket; sealed
replay strata and model inputs remain unchanged.
Store readers validate sealed hashes and headers before date-bounded reads;
payload-value validation belongs to the immutable writer. A subsequent audit found
that the former duplicate reader validation scanned later ages/masks and
action/reference-price values in the old store during integrity checks. No
held-out fitting or performance evaluation ran, but earlier literal claims of no
later payload access require this qualification. Historical sealed flags record
granted model rows and are not rewritten. See
[the disclosure and correction](docs/v2_store_comparison_stop_evidence.json).
Official validation and the permanently spent test remain sealed; deployment
is unchanged. Earlier results remain immutable under their own registrations.

The [multiday pipeline audit](docs/v2_MULTIDAY_AUDIT.md) identified
economic-beta and execution-timing defects in the rev4e ledger: normalized `beta_60`
is used as an economic coefficient, hedge sizing consumes same-close outcomes,
and retrospective inferred actions can alter decision-time orders. Round-1 rev4f
and Round-2 rev4f economics now correct these findings. The decoded history-age
provenance exemption was explicitly approved; all quality strata remain exactly
unchanged. The original-store input audit found eight of twenty intraday scalar
features entirely invalid and unusable M1/COTAHIST level anchoring. Those contracts
are repaired in the new store; sparse archive gaps remain unknown. The audit's
code cleanup does not change the frozen historical results.

Pass 5 uses [the clarified rev4f registration](research/preregistrations/v2_round1_round2_rev4f.md).
The ledger consumes economic BOVA11 beta, notional entries and hedge rebalances,
and position-fraction exits/trims. Decisions use only action uncertainty known
through t-1. Retrospective opening-inventory conversion occurs after all intentions
and before fills. Replay exemptions enumerate only authorized diagnostic/provenance
fields; scores, masks, targets, populations and score-derived readouts remain fixed.
[Pass-5 status](docs/v2_PASS5_STATUS.md) records completed work. Sealed rev4f uses
`v2_hedge_beta_rev4f_20260908T225248Z`, manifest
`f5fa41536740ca412550cb81d2b44c9541e338dc060d674ab8ea270edd7be712`.
The repaired store uses `v2_hedge_beta_rebuilt_e02f934_20260909T104556Z`, manifest
`225adc6fe336d2380e5c94897ba5d6ece4fcc8e0b0aca6d631dca9063314edb4`;
all four array files are byte-identical to rev4f.

Sealed rev4e Round 2 is now mirrored in
`D:\quant-data\b3\processed\model_runs\v2_round2_rev4e_2b40b24_20260908T202100Z`.
All 400 inventory files (333,714,344 bytes) and the exact file set are hash-verified.
The only approved launcher remains `ops/lambda-gh200.ps1`. Every future GPU run
must be sealed, copied to the host and hash-verified BEFORE terminating its exact
recorded instance ID; verify that ID absent twice.

The accepted v1 incumbent is the peer-free, full causal time-of-day normalized,
width-64 causal TCN trained uniformly with soft Spearman and SAM-AdamW. The best
recorded exact validation result is seed-11 IC **0.041972**. The rejected
gap-pairwise loss, continuous-target sidecar, and residual equity-attention branch
are absent from the current tree; their commits, manifests, and immutable artifacts
remain the historical reproduction contract.

The historical parent was reproduced on 2026-08-19 at commit `4067962` with
matched seeds 11/29/47. Best-IC deltas versus the immutable records were
`+0.0000053`, `-0.0000065`, and `+0.0000009`; every best epoch and stop epoch
matched. A bidirectional odd/even-date cross-fit of the internal selection windows
subsequently froze raw Patience-3 as the trajectory rule, with its uncertainty and
Fold-B non-confirmation retained in the research record.
A no-retraining follow-up averaged five raw checkpoints around the parity-selected
Patience peak. It lost to raw Patience on both folds and was rejected; raw
Patience-3 remains frozen and checkpoint-rule investigation is closed.

Read [RESEARCH_HANDOFF.md](RESEARCH_HANDOFF.md) for architecture and campaign
history, exact results, artifact identities, and interpretations.

## Source and write boundaries

- Python 3.12; use `uv` and the `research/` project.
- Raw data under `quant-data/b3/raw/**`, canonical source archives, and
  `Trading/**` are immutable.
- Derived stores belong under `quant-data/b3/interim/**` or
  `quant-data/b3/processed/**`.
- Resolve canonical pointer files at runtime and record resolved identities in
  output manifests. Never hard-code a timestamped source when a valid pointer is
  available.
- Equity identity is permanent `security_id`/ISIN plus bounded source-assignment
  dates. Ticker is only a dated attribute.
- Monthly point-in-time membership remains the v1 intraday eligibility
  contract. V2 uses its separately documented causal daily 20-session
  activity, liquidity, price, and listing-history rules.

## Current canonical v2 refactor contract

The executable contract is documented in [docs/v2_README.md](docs/v2_README.md),
with the semantic break and required rebuild described in
[docs/v2_MIGRATION.md](docs/v2_MIGRATION.md). Canonical decision row `t` is the
information available at exactly 15:45 `America/Sao_Paulo`: daily market state
ends at `t-1`, decision-available publications may enter on `t`, completed
intraday bars exclude the 15:45 entry bar, and no consumer applies another
stage-specific lag. A versioned dated B3 schedule is required as calendar
authority. Source completeness, price observation, trade observation, valid
activity, entry eligibility, feature validity, and outcome validity remain
independent masks.

ISIN is permanent identity and ticker is only a dated attribute. A succession
can affect history or economics only through a verified point-in-time allowlist
with contractual share/cash terms and evidence; the current allowlist is empty.
The same verified action primitive must feed shareholder-wealth features,
targets, intended orders, and signed-share accounting. Price return, gross
shareholder holding return, and the median-adjusted volatility-scaled model
target are separate outcome families. Unknown actions or terminal wealth stay
unknown; known zero terminal wealth remains a valid total loss.

Every enabled model field has a mandatory ordered `FeatureSpec` and schema hash.
Neural, GBDT, baseline, scoring, and evaluation adapters share one canonical
date/security feature view with per-feature validity and true source age. Neural
inputs zero invalid payloads; GBDT uses NaN only where the same mask says the
value is invalid. The canonical fast branch uses native, compact, separately
masked five-minute inputs and fresh weights; v1 fast artifacts are isolated as
contaminated historical diagnostics and are not a clean-path dependency.

Round 5 is a CPU-only data round. Its neural input contract changes enabled
sidecars to separate decision-row, zero-initialized residual projections, gated
after cross-sectional pooling. The slow GRU inputs remain unchanged. Invalid
families preserve S0 forwards, parent gradients and RNG state exactly; no neural
fit runs in this round. Historical sidecar-concatenation recipes bind their
original code commits. See the registration and `docs/v2_README.md`.

The store builder streams family-by-family into disk-backed float32 arrays and
records peak RSS. Intended orders are fixed before later fill observations;
unfilled exposure, contractual claims, cash, funding, costs, and insolvency are
carried through one persistent ledger. Evaluation uses a common supported
population for D1/D2/D3/D5, keeps D10 diagnostic-only where required, preserves
chronological/fold blocks, and reports unsupported statistics rather than
coercing them to zero. Current schemas reject prior stores, checkpoints, scores,
and partial resumes rather than silently translating them.

Phases A-F have fixture-level engineering acceptance at commit
`f0cf568303715e8783e539a683568535e2232c7f`. Subsequent bounded passes produced
the V3 store and completed the registered development-fold research program.
Score-bearing Round-2 trajectories remain bound to commit
`916ac0b7e6e3ab16dea72dbf480ebb086a8981b0`; reporting and sealing fixes through
`2ee5334aef9bebbd2aa9088d4156a7c6919d64e4` reused and hash-verified completed
artifacts without recomputing a score or evaluation. Authoritative dated B3
schedule evidence, independently verified contractual action terms, auction
execution marks, and historically executable borrow remain unavailable. Any
stronger economics or production claim requires those sources and a new
pre-result registration; it cannot be obtained by relabelling the current
development-grade evidence.

## Superseded historical v2 foundation context

> **Historical only.** Everything in this section through the next `V1`
> heading describes the pre-refactor v2 schema, stores, Section-C integration,
> and their immutable evidence. Those paths and hashes remain useful for
> reproduction and audit, but none is a current-schema store, accepted current
> candidate, or authority for a new run. Where this history says a v2 store was
> “accepted” or “canonical,” read that strictly within its superseded schema.

The additive v2 implementation lives under `brazil_rv.v2`; its detailed
executable contract is documented in `docs/v2_README.md`. Corporate-action
candidates come from COTAHIST `DISMES` changes and independent price jumps.
Robust three-session medians and price/quantity continuity classify split/bonus,
cash-type, and ambiguous events. Only split/bonus events adjust prices; future
cash-type or ambiguous events invalidate targets; only ambiguous events shadow
return-feature lookbacks; only detected splits create M1 boundaries.

Provider acquisition, taxonomy, failures, and off-calendar rows remain available
for recall/precision and dividend-drop audits but cannot enter a panel array. A
mandatory small-store test verifies byte-identical arrays with full versus empty/
failed provider evidence. Store acceptance retains an unconditional five-point
survivorship-validity gap for every internally derived feature family and a
10-point gap for targets. External sidecars instead require exact independent
reproduction of their validity masks from publication-lagged raw availability;
daily archives must satisfy D+1. Their composition check is a five-point
one-sided threshold on a 1,000-replication, 95% name-clustered bootstrap of the
survivor-minus-delisted validity gap within each pooled causal prior-ADV20
quartile. A quartile binds only when both groups contribute at least 20
continuation names and 2,000 family-present name-days, and fails only when the
interval lower bound exceeds +5 points. Smaller strata remain reported. A
survivor-subset total-return target is registered but not implemented as a
future sensitivity experiment.

Exact same-ticker COTAHIST ISIN changes qualify as identity continuations only
when the predecessor's final observation is followed by the successor's first
observation on the next market session, with no same-date ambiguity, branch, or
cycle. Qualifying successors inherit strictly prior feature history and share a
survival identity; ticker reuse after a gap remains separate.

COTAHIST full-session M1 anchors additionally require same-name/day close-unit
agreement within 0.005 absolute log return. Events expose only causal RAD
sessions-since; announcement-dependent future flags and SUE remain unavailable.
Paired comparisons consume the selected preset's bootstrap settings, and preset
runs require a hash-bound fast pretrained checkpoint. Legacy action-cache schemas
are normalized in memory.

External artifacts recorded by a sealed manifest are identified by their byte
count and SHA-256, not by the host spelling of their path. A foreign absolute
path is relocated through a `BRAZIL_RV_DATA_ROOTS` JSON override kept outside
the immutable data root. Loaders require the mapped file to match the sealed
byte count and SHA-256 before decoding it, and every recorded-to-resolved path,
mapping prefix, override path, and override-file hash is written into the run
manifest. Relocation is excluded from model-input identity so the same sealed
artifact can move between Windows and Linux without changing the training
contract.

The accepted Section-C store is
`D:\quant-data\b3\processed\v2_daily_store_98e9386_20260904T165924Z`
(manifest SHA-256
`6a7e13195c6cde92fbdc756a585e4cb65d73998faa94e237595c7be7cdfb6919`).
Its 4,102-by-933 panel spans 2010-01-04 through 2026-07-17, peaked at 5.991
GiB RSS, and passed the 8-GiB memory ceiling. All six external publication-lag
masks reproduced exactly with zero D+1 violations. The four binding bootstrap
strata were the odd-lot quartiles and all passed; the largest binding 95% lower
bound was +0.0834 points. The maximum unconditional internal-family and target
gaps were 1.2899 and 1.8227 points. No qualifying consecutive-session ISIN
succession was present. The options audit retains its unstratified 5.4221-point
delisted-above-survivor composition signature, while lending's 14.2121-point
quartile-2 estimate remains reported rather than gated because it has only 5
delisted continuation names and 950 delisted present name-days.

The first spec-scale full-F1 validation started from that accepted store on
Linux and completed all 12 F1-F3 baseline evaluations plus both F1-F2 GBDT
evaluations and their 100 head/seed model files. It then stopped before neural
training because the sealed store recorded its v1 fast inputs under a Windows
absolute path. Commits `4d525f68c4fb5c38e5b2e7e84dad4943ec21994f` and
`b61b0d8d5bbc6ef35014762c88f3a04c9667955e` resolved that portability issue,
added hash-verifying foreign-path tests, and froze the exact completed-classical
source audit. The sealed store manifest was not changed and the completed
baseline/GBDT legs were not repeated.

The network-only continuation root is
`/lambda/nfs/brazil-rv-east3/quant-data/b3/processed/v2_pipeline_network_resume_b61b0d8_20260904T185200Z`.
Its first full-F1 scratch-F `select_even` leg completed one in-memory training
epoch, then the selection forward pass emitted at least one non-finite value;
the finite-vector rank guard stopped the run before history append or artifact
sealing. It contains zero checkpoints, histories, scores, evaluations, and run
manifests. Failure-record and log-inclusive inventory SHA-256 values are
`0fa8b46b23a34434f13240569fd89bbc64123cb387cddf49319b90fc1a0b005f`
and `e1ccd0824dcffd878dfbd6f1524b7c9e93c8c935b5b66cb97406a2e1fd0995bf`.
No retry was performed. Stage P, checkpoint hand-off, the opposite scratch
parity, and both persistence parities remain unrun. Full-F1 neural validation
is therefore blocked at this first numerical failure, not by artifact
portability, and remains an integration check rather than a research result.
Official-validation and test access remained false; Section D remains
unauthorized and unrun. The former `f048ea9`, `2cb204d`, and related validation
roots remain immutable historical engineering evidence only and must not be
used as canonical inputs.

The exact failed selection batch and post-epoch parameters were subsequently
sealed under
`/lambda/nfs/brazil-rv-east3/quant-data/b3/processed/v2_nonfinite_diagnosis_14f6648_20260904T192200Z`.
All 49 training steps, losses, parameters, and saved input arrays were finite;
invalid input cells were already zero. Six non-finite outputs affected active
names. Eager inference and every separately compiled model stage remained
finite, while only the whole compiled composition failed. Replaying the exact
batch with PyTorch's dynamic-graph CUDA-capture safeguard restored finite
outputs with a maximum absolute eager difference of `1.0489e-05`. The diagnosis
therefore identified stale-buffer contamination from whole-graph dynamic CUDA
graph capture, not training divergence or input contamination. Diagnosis audit
and inventory SHA-256 values are
`52391d502778dd588ccd821cb47e47b845ff39352f5f6e017c766f43ffcf121f`
and `ba2b0da9bf92514d0e88888c7e0c7f504972b7980f4597b4aee1c849d7ef7c74`.

Commit `bfba0d7b6c6743a6b246ceaff7039917b4ab8f2b` makes the data boundary
structural: every invalid floating input is zeroed under its mask, every
available value must be finite, masked reductions use `torch.where`, and both
SAM loss evaluations must be finite. Inductor skips CUDA graph capture for the
model's dynamic sparse-name graph. Regression coverage includes NaN-by-contract
cells in every array class, an active name with no valid lookback, a
fast-present name with no valid patches, marked-valid non-finite rejection, and
both non-finite SAM loss phases. Local and Linux Ruff/compile checks passed, as
did all 615 tests. An exact post-fix compiled replay is sealed at
`/lambda/nfs/brazil-rv-east3/quant-data/b3/processed/v2_nonfinite_postfix_replay_bfba0d7_20260904T194147Z`;
its audit and inventory SHA-256 values are
`67ec49a215efdb5dcbb7074a16d1f549fad607dbee1119f9054ffa84998ef7bc`
and `6afc2139f332c89b75a3f28d7a82da184ad45aed4856eb0ee4279917c8d4bd11`.

The completed network-only integration root is
`/lambda/nfs/brazil-rv-east3/quant-data/b3/processed/v2_pipeline_network_resume_bfba0d7_20260904T194500Z`.
It contains both three-epoch scratch-F parities, both one-epoch lambda-0.1
persistence parities, one Stage-P epoch, and its one-epoch Stage-F checkpoint
hand-off at full F1 scale, seed 11, lookback 60, CUDA, and compiled execution.
Scratch F produced pooled primary IC `0.04577236`; lambda 0.1 produced
`0.03274652` while increasing mean one-session prediction persistence from
`0.96028812` to `0.98053202` and five-session persistence from `0.85587469`
to `0.89919339`. These are integration diagnostics, not research results. The
log-inclusive completion audit and inventory SHA-256 values are
`985cdc3a3af81372cbd91550a345cfbbee7b1c3eba1213250a0ca2320c8c8333`
and `48d553e5ef1dc7848eb5c1404eb27bdf416234c26924d78e6fab33303832ac4d`;
the inventory covers 82 files totaling 31,635,333 bytes. All protected-access
flags are false. Section D remains unrun and no candidate or deployment changed.
After commit `a39c7e98166cb031a09832ecd8245fc70596c0bf` matched local,
GitHub, and the instance, exact paid GH200 instance
`86ee7e3fc35e4c8ba0daa1b7ff6d633f` was terminated. Provider inventory reads
at 2026-09-04T20:09:32.7623726Z and 2026-09-04T20:09:57.4535979Z confirmed
that exact ID absent and zero remaining instances.

The pass-4e development-tier continuation is frozen on branch
`fix/v2-development-grade-data` at implementation commit
`eca09d6851074e79d7ffb7f1d4e9a11c022c3a02`. It uses the V3 store built by
`8021e42ae658d3eb7bba19e39b58e2f9c8d65331` at
`D:\quant-data\b3\processed\v2_daily_store_8021e42_20260906T202315Z`, manifest
SHA-256
`deb9ca8449c5b9a83bf25ac19218069c006e183361b6f6ba836717ade63b491b`.
It passed the measured 8-GiB build ceiling at 7.138950 GiB RSS and all
identity, calendar, feature/target survivorship, external
contemporaneity/composition, and protected-access gates. Slow wealth features
and naive baselines share exact endpoint, raw-chain restart, and
unresolved-action validity. Raw-price cross-session intraday fields separately
guard unresolved actions and unit changes; same-session scale-free fields do
not. The exact native-fast audit SHA-256 is
`98d2e5346a0e2555b77fdb0cb034c63e9d8fa7860eb30f6e749680883ba29218`.

The full scratch acceptance root is
`D:\quant-data\b3\processed\model_runs\v2_development_acceptance_8021e42_20260906T184700Z`;
pipeline-manifest and log-inclusive artifact-inventory SHA-256 values are
`f7ce5a47a5be4bc8d8f4e11b5adde83a97c6e334b87038068f0f56a7ba7beaa6`
and `2caf60d4d3202fa3228b295aa3245d04bf6bf92dc197a00264dbdab9d3672bf2`.
Thirteen of 16 evaluated books passed the unchanged 1.8--2.2 mean-gross gate,
but F1 inverse-volatility, F1 momentum, and F2 inverse-volatility failed it.
Only 7 of 16 books met the registered per-evaluation requirement that mean
daily unresolved/stale inventory be below 0.02 NAV; every F3 naive book failed
that bound, as did F1 inverse-volatility, F2 inverse-volatility, F2 reversal-5,
and the F1 GBDT. The result is `unsupported`. The branch therefore remains
unmerged, Round 1 and Round 2 remain unrun, no GH200 was launched, official
validation/test remained unread, and no deployment changed. Any continuation
requires a new pre-result contract rather than reinterpretation of these failed
acceptance gates.

Pass 4e confirmed that the ledger incorrectly latched a one-day unresolved
action cell for the entire remaining position life. The bounded repair makes
uncertainty per-session, blocks only entries, permits every printed exit/risk
trim/terminal liquidation, separates current unresolved-claim notional from
stale-mark notional, and asserts retrospective accounting alignment at the
evaluator boundary. No store rebuild was warranted: the sealed diagnostic
found no false retrospective action cell on an observed held name-day. All 861
tests passed. The hash-reuse replay at
`D:\quant-data\b3\processed\model_runs\v2_development_acceptance_pass4e_eca09d6_20260906T225238Z`
has pipeline-manifest and log-inclusive artifact-inventory SHA-256 values
`846d278d1a9bf9a038c7687406e570d77c968ccb7f7c5b13cfeba616a1c1de78`
and `49d6b2e0d7bbcbbdd4402b676a9e5891c81004f325928d440cd985c28cc1b534`.
All non-ledger fields were bit-identical, but acceptance remains `unsupported`:
9 of 16 books still exceed 2% mean daily stale/unresolved notional, while F1
and F2 inverse-volatility remain below 1.8 gross. The separated measures are
equal in these data because current false inferred-action cells coincide with
no-print sessions. This is now a genuine stale-print/coverage limitation, not
a persistent action latch. `main` remains unchanged and Round 1/2 remain unrun.

Pass 4f closes that stale-holding policy gap with the labelled development
convention `last_mark_after_10_sessions`. The pre-change sealed diagnosis found
37 stale holding episodes / 2,490 name-days and no later print inside an
evaluation holding window, so it found no fill defect; it also found no
succession candidate and no tender-like path under the predeclared heuristic.
Commit `36a868c6455fd1b58f128b94d03112588b9899c7` settles a held long or short at
the last mark with ordinary costs after ten no-print sessions, permanently
releases the slot, reports a separate 30% adverse settlement scenario, counts
later prints, and marks economics unresolved above 15% cumulative settlement
notional/NAV. All 869 tests passed before the final replay.

The hash-reuse Pass-4f acceptance root is
`D:\quant-data\b3\processed\model_runs\v2_development_acceptance_pass4f_36a868c_20260906T235841Z`;
its pipeline-manifest and log-inclusive artifact-inventory SHA-256 values are
`6a137bb00b447a04d52462b28f20e76c1465a981deb2cfa07485387abb064409`
and `1139ef305f94ecb941a50b795fb158203990513615f3262f5ef8a7d187d1d414`.
All non-ledger fields were bit-identical and all 16 books passed the unchanged
2% stale/unresolved bound. Fifteen passed the unchanged gross band, but F2
inverse-volatility remained below it at mean gross `1.755766`; acceptance is
therefore still `unsupported`. F3 momentum and F3 blend also exceed the new
15% settlement-incidence resolution bound and are labelled accordingly. The
registered stop kept `main` unchanged and Round 1/2 unrun. Official validation
and test remained unread, no deployment changed, and no paid instance was used.
Two direct provider inventory reads after sealing returned zero instances; the
old local Lambda state file is stale and does not represent a billable host.

Pass 4g registered its gross-shortfall disposition before diagnostics and then
hash-verified/replayed all 16 sealed Pass-4f panels with append-only ledger
instrumentation. All prior headlines were bit-identical and direct
entry/fill/cap defect counters D1--D4 were zero in every book. The immutable
diagnostic root is
`D:\quant-data\b3\processed\model_runs\v2_pass4g_occupancy_diagnostic_fa5c8af_20260907T013518Z`;
diagnostic-manifest and log-inclusive artifact-inventory SHA-256 values are
`2085ed71008821b983f61a8f2ad2667de8a4a74977f7b6079e5def98f9a377d9`
and `1663e488cdad3ca0187cfe31491d91ff3067aa0079c3277f4cdeffddbb010feb`.
The preregistered P0 stop fired because eligibility-flicker exit share D5 was
`0.103448` for F3 inverse-volatility and `0.148936` for F3 momentum, both above
`0.10`. F2 inverse-volatility's occupancy share was only `0.059435`; its
largest shortfall source was sizing mark drift (`0.107124`), not occupancy.
Therefore no conditional gate rewrite was made, `main` remains unchanged, and
Round 1/2 remain unrun. Official validation/test stayed unread, no deployment
changed, and no paid instance was launched.
Direct provider inventory reads at `2026-09-07T01:39:39.9369263Z` and
`2026-09-07T01:39:45.5250149Z` each returned zero instances; there was no paid
host to terminate and no adjacent instance was touched.

V2 development folds end on 2024-12-30. Official validation (2025-01-02 through
2025-12-30) requires a hash-bound registration token, and test dates are refused
unconditionally by this code version. Target masks and corresponding numeric
values are clipped at the store capability boundary and again at each exact
evaluation window.

## V1 intraday feature-store contract

The peer-free store contract is `M1_FEATURES_PIT_CAUSAL_TOD`: 1,248 dates, 1,228
eligible dates, 55 decisions, 158 equities, 26 dynamic channels, 32 slow channels,
three horizons, seven local contexts, and eight global contexts. It has no
human-prior or peer arrays.

Equity normalization uses an equity-wide 30-minute causal relative-variance TOD
profile with a 20-session-equivalent prior and `[0.25, 4.0]` bounds. Each training
date emits before updating; the profile freezes after 2024-06-28. Context series
retain their semantic causal transforms rather than receiving the equity overlay.

The full store built on persistent Lambda NFS is recorded in
`RESEARCH_HANDOFF.md`. The Windows local canonical pointer may still identify the
old V4 store; verify the pointer and schema in the execution environment before
training.

Core causal rules:

- History ends strictly before the decision; the entry bar is excluded.
- Label entry is `open[T]`; exit is exact `close[T+h-1]` within the session.
- Missing OHLC is never interpolated and stale prices are not label endpoints.
- Fitted scalers, volatility/TOD state, and other stateful transforms use only the
  information available at their historical timestamp.
- Training, validation, and test identities are immutable and audited.

External data experiments use the immutable `PIT_EXTERNAL_FEATURE_SIDECAR`
contract. A sidecar is bound to the exact canonical feature-store identity and
date/equity axis hashes. Daily arrays have shape `[date, 158, feature]`; intraday
arrays add the canonical 55-decision axis. Every feature has an explicit mask,
invalid values are exactly zero, and source-specific availability is materialized
as an exact no-fill join before training. The loader additionally gates values and
masks by point-in-time equity membership. A single per-equity bias-free linear
residual injects concatenated values and masks into the incumbent state; its
weight is zero-initialized, and candidate construction restores the parent's RNG
state after adding it. Thus every external-data candidate begins as the exact
parent without changing base weights or dropout randomness. An equity with no
valid external observation has an all-zero input and therefore receives exactly
zero direct adapter residual throughout training; learned mask weights still
represent observedness where data is present.

## Splits, discovery folds, and test policy

- Training: 2021-08-16 through 2024-06-28, 716 dates.
- Validation: 2024-07-08 through 2025-06-30, 244 dates.
- Held-out test: 2025-07-07 through 2026-07-17, 259 dates.
- Fold A: first 512 training dates fit, next 102 select.
- Fold B: first 614 training dates fit, final 102 select.

The two internal selection periods do not overlap. Both fit windows preserve an
effective batch of 512 distinct dates. Stored features are causal, but the TOD
profile adapted inside the historical training dates, so these are screening
folds rather than exact replicas of the officially frozen preprocessing regime.

The official validation split has been consumed. Embargo dates are not selection
data. Experiment 51 opened the held-out test exactly once under its frozen
registration; the test is now permanently spent and no current evaluator or
campaign driver may request test rows. Hypotheses generated by that read require
new forward data.

## Accepted model and trajectory contract

- One shared per-instrument width-64 causal TCN.
- Five-minute patches, 69 patches, kernel 3, dilations `(1, 2, 4, 8, 16, 32)`.
- Six residual LayerNorm/SwiGLU blocks and final-state readout.
- Projected 32-field slow state.
- Fixed context-plus-masked-equity-mean/dispersion gated fusion.
- `WIN$` and equity `beta_to_WIN` masked; WDO, five DI contexts, ZT, and ZN active.
- All three horizons trained jointly.
- Sole objective: soft Spearman, temperature 0.50.
- Uniform training dates; SAM-AdamW rho 0.125; effective batch 512.
- One fixed 20-epoch trajectory; no training-time early stopping.
- Raw checkpoint and raw/EMA validation predictions every epoch.
- EMA decays 0.98, 0.99, and 0.995.
- Frozen rule: raw Patience-3 with minimum IC improvement `0.0001`, patience three,
  maximum 20 epochs, and restoration of the best raw checkpoint. This entire rule
  is fixed before the sparse official-validation stage; do not retune it there.
- Last-3/last-5 weight averages and raw-score prediction averages are constructed
  without retraining.
- Retrospective best epoch remains diagnostic only.
- No peer/classification inputs and no cross-equity attention.

Hard Spearman is the primary selection metric. It is averaged across decisions
within each date and horizon, then equally across dates and horizons. Seed
ensembles uniformly average tie-aware within-sample/horizon ranks and never fit
ensemble weights.

The completed internal campaign at
`trajectory_discovery_e22dd67_20260819T134332Z` initially selected final EMA-0.995
from fixed rules, with fold-A/fold-B ensemble ICs `0.045309`/`0.050625` and mean
`0.047967`, versus final-raw `0.043416`/`0.049602` and mean `0.046509`. Paired
EMA-minus-raw deltas were positive on both folds (`+0.001892`, `+0.001024`) and at
every horizon, but their block-bootstrap intervals mostly included zero and
time-of-day deltas were mixed. Treat the rule as a deterministic variance-reduction
choice, not a claim of uniform statistical dominance.

The same-window Patience-3 IC `0.051860` was selection-biased. The corrective
artifact `trajectory_crossfit_3054228_20260819T161200Z` selected checkpoints on
one odd/even date parity and reported them only on the other, in both directions.
Raw Patience-3 scored `0.048416`/`0.050673` and mean `0.049545`, versus final
EMA-0.995 mean `0.047967`. Its paired advantage was `+0.003108` on Fold A but only
`+0.000048` on Fold B, with both block-bootstrap intervals including zero. EMA-0.995
Patience scored `0.048518`; last-10 raw weight averaging scored `0.048060`, while
last-7 scored `0.047352`. The outer rule replay chose raw Patience in three of four
directions and EMA Patience once, with mean out-of-half IC `0.048897`. Raw
Patience-3 is frozen as the numerical winner, but it is not treated as established
dominance over final EMA-0.995.

The one-candidate centered-average follow-up at evaluator commit `381dcb7`
scored `0.046655`/`0.050385`, mean `0.048520`, versus raw Patience mean
`0.049545`. Centered-minus-raw-Patience was `-0.001761` on Fold A and
`-0.000288` on Fold B and was negative in all four out-of-half directions. The
candidate was rejected without another sweep. Its code was removed from current
HEAD; exact reproduction uses the recorded evaluator commit and immutable
`trajectory_centered_crossfit_381dcb7_20260819T170100Z` artifact. Neither
official validation nor test was accessed.

## Current source-tree status

`brazil_rv.v2` is the canonical additive daily foundation. It contains daily-panel
and corporate-action construction, causal point-in-time universe and feature
manufacture, lazy slow-history/fast-stream loading, multi-horizon targets, the
shared-weight GRU/TCN model, date-pair training with SAM/EMA/Patience, LightGBM
and naive baselines, sealing-aware scoring/evaluation, named triage/full presets,
bounded multi-run launch, and the development-only pipeline validator. A
legitimately active first-day entrant with no prior slow row receives the exact
zero GRU initial state; it remains active for targets, scoring, and pooled
statistics. The v1 packages remain unchanged.

`modeling.train` is the canonical soft-Spearman trajectory entry point.
`modeling.run_discovery_campaign` runs exactly the two internal folds and seeds
11/29/47. Sidecar campaigns do not select a fresh checkpoint rule: they record
the frozen bidirectional odd/even Raw Patience-3 primary and final EMA-0.995
secondary readouts. `modeling.crossfit` selects
validation-adaptive checkpoints on one odd/even date parity and reports only on the
other, replays rule selection in both directions, and can materialize last-7/last-10
weight-average predictions without mutating source runs. Its frozen selection file
is the authority for the next official-window run. `modeling.analyze` strictly
aligns observations and reports member/ensemble IC, seed diversity, paired date
deltas, moving-block intervals, and horizon/time-of-day guardrails.

`modeling.external_data_screen` applies that frozen readout contract to one
completed sidecar campaign. It reports candidate deltas against both the canonical
parent and the standing designated challenger, plus a predeclared uniform
parent-plus-candidate diversity readout. The standalone candidate and that fixed
six-member diversity recipe are separate predeclared retention paths, and both
are keyed only to their canonical-parent deltas. The challenger and EMA columns
are informational and cannot create a "beats either" selection rule.
Either primary path requires at least `+0.001` mean fold IC gain and non-negative
gain on both folds. The diversity path additionally requires the standalone
candidate to lose no more than `0.001` on either fold.

The completed historical external-data program
`external_data_7e535ac_20260821T161800Z` tested B3 lending, SHFE ferrous/pulp,
COTAHIST options activity, CVM RAD events, B3 odd-lot activity, B3 index
rebalances, CCEE PLD, CVM fundamentals, regular trade activity, and ADR overnight
under this contract. No standalone or parent-plus-candidate recipe passed the
frozen gate, so none of these exact feature families is part of the accepted
recipe. Official validation and test were not accessed. Options activity was
positive on both folds but its mean gains were only `+0.000257` standalone and
`+0.000261` in the diversity recipe, below the threshold. Positive fixed-final-
EMA observations for several candidates remain informational and do not change
retention or reopen checkpoint selection. Source components that were explicitly
unavailable and excluded from the screens remain untested rather than rejected;
`EXPERIMENT_LOG.md` records the exact boundaries and results.

`modeling.evaluate` is validation-only. Its former held-out mode was removed
after Experiment 51 permanently spent the test. Exact reproduction of the sole
test event uses the recorded Experiment-51 implementation commit and immutable
artifacts; it must not rerun inference.

The old human-prior/peer, alternate model-family, routing, multiscale, attribution,
probe, overlay, V-numbered, recency-weight, hybrid-loss, target-sidecar, and
residual-attention experiment systems are not compatibility APIs. Historical
reproduction uses their recorded commits and immutable artifacts.

## Environment and operations

From the repository root:

    uv sync --project research --group dev
    uv run --project research python -m brazil_rv.preprocessing.build
    uv run --project research python -m brazil_rv.modeling.run_discovery_campaign --output-dir <new-campaign-directory>
    uv run --project research python -m brazil_rv.modeling.train --selection-window official --selection-rule-file <trajectory-selection.json> --seed 11
    uv run --project research python -m brazil_rv.modeling.evaluate --run-dir <completed-official-run-directory>

`ops/lambda-gh200.ps1` is the only Lambda watcher/launcher. Launch requires
explicit billing acknowledgement, transfers a verified Git bundle, and leaves the
instance running. It never starts training or terminates a paid host. Confirm
provider state and exact instance identity before launch or termination.

## Limitations and authority

The system does not model order-book state, bid/ask spread, queue position,
slippage, costs, or live execution. Historical MT5 spread and tick volume are not
market microstructure. Raw absolute prices, tickers, identifiers, news, and
unapproved technical indicators are not model features.

When statements conflict, prefer immutable sources and canonical pointers, then
executable code and tests, then this document, then the detailed handoff.

## Phase A representation decision (2026-08-20)

Six zero-start residual representation candidates were screened on the two
internal discovery folds with seeds 11/29/47 and the frozen odd/even cross-fitted
raw Patience-3 readout. Final EMA-0.995 was recorded from the same trajectories as
a secondary readout. The campaign produced 720 checkpoints from commit `732b1b0`.

| Candidate | Primary mean candidate-minus-parent IC | EMA-0.995 secondary mean delta |
|---|---:|---:|
| Decision-time embedding | -0.000009522 | -0.000133778 |
| Temporal mean/std adapter | -0.000104324 | +0.000009399 |
| Block-2/4/6 multi-depth stats | -0.000147827 | -0.000975001 |
| Cross-sectional max/min | -0.000198130 | +0.000772140 |
| Learned set pool, width 16 | -0.000006211 | +0.000005459 |
| Conditional beta/volatility bucket means | -0.000198862 | +0.000424682 |

Every primary mean was non-positive. The secondary max/min and conditional-bucket
gains were confined to Fold A and reversed on Fold B. No candidate had a coherent
positive horizon/TOD profile, so none qualified for sparse official validation.
Raw Patience-3 remains the canonical parent and the held-out test remains sealed.

The completed immutable campaign is:

    /lambda/nfs/brazil-rv-east3/quant-data/b3/processed/model_runs/phase_a_732b1b0_20260819T180348Z

Its manifest records `official_validation_accessed=false` and
`test_accessed=false`. Rejected candidate code and its campaign driver were
removed from current HEAD. Reproduction is through commit `732b1b0` and the
immutable artifacts, not compatibility branches. The general strict
observation-level measurement layer remains canonical.

## Phase A autopsy and diversity follow-up (2026-08-20)

Checkpoint autopsy disproved the hypothesis that the historical decision-time and
learned-set paths were dead. Decision-time final-projection norms reached
`0.319-0.355`; learned-set final-projection norms reached `0.490-1.107`, and its
standard-initialized `phi` weights moved materially. Learned set already entered
the incumbent nonlinear shared fusion with only its final projection zeroed.
Their prediction ranks nevertheless remained above 0.9991 correlated with the
matched parent, so both were active but rank-ineffective.

A no-training uniform-rank reanalysis pooled the three parent members with the
three decorrelated multi-depth members. Under cross-fitted raw Patience-3 it added
`+0.001237` on Fold A and `+0.000284` on Fold B, mean `+0.000761`. The same
six-member pool added mean `+0.001651` under final EMA-0.995. Adding the three
temporal-statistics members diluted both readouts. All block-bootstrap intervals
included zero, so parent+multi-depth is retained only as the one Phase A
diversity-ensemble candidate eligible for sparse official-validation confirmation;
it is not yet a canonical lockbox recipe and no ensemble weights may be learned.
The immutable reanalysis is:

    /lambda/nfs/brazil-rv-east3/quant-data/b3/processed/model_runs/phase_a_autopsy_d237998_20260820T111500Z

The remaining decision-time routing objection was then tested directly. A
standard-initialized width-16 decision embedding fed a zero-only final projection
into the existing shared nonlinear fusion, and candidate construction preserved
the parent's RNG stream exactly. A 10-step soft-Spearman assertion confirmed that
both the final projection and upstream embedding moved. Across two folds and
three seeds, cross-fitted Patience candidate-minus-parent IC was
`-0.000001`/`-0.000009`, mean `-0.000005`; EMA-0.995 mean delta was
`-0.000001`. Final adapter norms ranged `0.299-0.992`. This is a conclusive active
null, so decision-time embedding is closed and must not consume official
validation. Exact reproduction uses commits `9828f72`/`b8d955a` and:

    /lambda/nfs/brazil-rv-east3/quant-data/b3/processed/model_runs/decision_time_fusion_b8d955a_20260820T113924Z

Raw Patience-3 on the parent architecture remains the canonical base for future
experiments. The rejected corrected-adapter code was deleted from current HEAD.
The canonical analyzer now permits candidate and parent ensembles with different
member counts while preserving strict observation alignment and uniform ranks.
Neither follow-up accessed official validation or the held-out test.

## Phase B target-decomposition decision (2026-08-20)

An immutable auxiliary-target sidecar was audited before training. It used stored
causal pre-neutralization `beta_to_WIN`, exact WIN decision-open-to-label-close
returns with observed endpoint masks, and no stale prices. Residual returns were
factor-neutralized, median-centered, normalized with the existing causal
volatility contract, and cross-sectionally midranked. Beta and exact WIN endpoint
coverage both exceeded 0.9985 across all horizons; mutation-based causality tests
passed. The immutable sidecar is:

    /lambda/nfs/brazil-rv-east3/quant-data/b3/processed/auxiliary_targets/phase_b_aux_15471e8_20260820T141500Z

Residual-rank, sign, magnitude, and combined auxiliary supervision were screened
on both internal folds with seeds 11/29/47 and 20-epoch trajectories. Primary
mean candidate-minus-parent IC was `-0.000625`, `-0.000482`, `-0.002599`, and
`-0.000351`, respectively. No primary candidate improved both folds, and no
Phase B member improved the existing Phase-A diversity stack on both folds. The
EMA-positive residual/combined secondary readouts did not override the frozen
Raw Patience-3 primary. Therefore the conditional common-component head was not
run. The completed 480-checkpoint campaign is:

    /lambda/nfs/brazil-rv-east3/quant-data/b3/processed/model_runs/phase_b_6b7b121_20260820T145500Z

The parent then received one three-epoch, latest-120-date, learning-rate-divided-
by-ten recency trajectory per fold/seed/direction. The best average candidate was
the epoch-3 50/50 full/fine rank ensemble at `+0.000457`, but it was
`-0.000930` on Fold A and `+0.001843` on Fold B. The both-fold guardrail retained
full history.

After Phase B, the only stage finalist was the prior parent-3 plus Phase-A
multi-depth-3 diversity pool. Its one sparse official-validation confirmation
scored `0.040495819` versus parent-3 `0.041639843`, delta `-0.001144024`;
30/60/120-minute deltas were all negative. Reject the six-member pool. The sole
canonical recipe remains the three-seed parent with Raw Patience-3. Official
validation is closed again, and the held-out test has never been accessed.
Official artifacts are:

    /lambda/nfs/brazil-rv-east3/quant-data/b3/processed/model_runs/phase_a_official_732b1b0_20260820T201500Z

Rejected Phase B sidecar/training/recency plumbing and the one-use confirmation
driver were deleted from current HEAD. Reproduction is through commits
`a04d63e`, `15471e8`, `6b7b121`, `e33a122`, and immutable artifacts, not
compatibility code.


## Designated challenger and comparator policy (2026-08-21)

The standing designated challenger is fixed as the uniform six-member rank
ensemble of:

- the three canonical parent members at seeds 11/29/47, with Raw Patience-3
  selected bidirectionally on the opposite odd/even discovery-date parity; and
- the three Experiment-18 residual-auxiliary members at the same seeds, read at
  fixed final EMA-0.995.

The auxiliary configuration is frozen to commit `3b60ac9` and its immutable run
manifests: WIN + WDO + ready-DI-level residual rank, soft-Spearman auxiliary
weight 0.5, separate zero-weight/zero-bias auxiliary head, parent initialization
and RNG stream preserved, width 64, 20 epochs, SAM rho 0.125, learning rate
0.0003, and EMA decay 0.995. All six predictions are tie-aware ranked within each
sample/horizon and averaged uniformly; weights are never learned.

This is not a second retention baseline. Every future discovery-fold candidate
must report paired deltas against both the canonical parent and this challenger,
but candidate retention remains keyed exclusively to the canonical parent. "Beats either" selection is prohibited. The challenger column is informational
evidence accumulated passively. The canonical entry point for future fold screens
is `modeling.designated_challenger.compare_discovery_screen`, whose summary
records this selection contract.

The challenger receives one official-validation comparison only when bundled into
the next official read already justified for a future stage winner. It does not
independently authorize an official read. The saved official residual final
EMA-0.995 payloads remain unopened for that purpose. The held-out test remains
sealed.

## Persistent Lambda retention cleanup (2026-08-21)

The `brazil-rv-east3` object store was reduced from 159,464,940,112 bytes
(148.513 GiB) to 20,202,855,773 bytes (18.815 GiB). The exact manifest deleted
5,928 objects totaling 139,262,802,568 bytes (129.699 GiB):

- an obsolete noncanonical human-prior V4 feature store;
- 120 unreferenced historical model-run prefixes;
- raw checkpoints, tail states, and redundant per-epoch predictions from closed
  campaigns; and
- all Phase-C/official binary intermediates except the designated challenger's
  required final EMA-0.995 predictions and observation references.

Raw and interim data were untouched. The canonical causal-TOD feature store and
pointer remain complete. The parent retains all 120 per-epoch discovery prediction
files and six references needed to reconstruct honest Patience cross-fits. The
challenger retains six discovery and three official residual epoch-20 prediction
files with their matching references. Manifests, histories, metrics, analyses,
and every run prefix named in the durable research record remain.

The immutable cleanup record is:

    /lambda/nfs/brazil-rv-east3/quant-data/b3/processed/model_runs/_retention/storage_cleanup_20260821

Its delete-list SHA-256 is
`b45f591cd4c77640ce2c924506f3040b1cdefe6b65679de7bae618217dd75f7b`.
Deleted binary intermediates are not recoverable in place; exact reruns use their
recorded commits, retained manifests/results, and canonical derived data. Lambda's
provider accounting endpoint still reported the pre-cleanup byte count immediately
after deletion, while a complete object-store listing returned the verified
post-cleanup total. No paid instance was active.

## Persistent Lambda retention cleanup, round 2 (2026-08-23)

After Experiments 27--40, a new complete object-store inventory found
109,637,262,343 bytes (102.108 GiB) in 18,970 objects. The principal growth was
the completed Experiment-27 external-data program: its 1,200 per-epoch
prediction files and 60 redundant tail bundles occupied 49.928 GiB even though
the selection rule and all ten rejection decisions were frozen.

A manifest-bound cleanup removed exactly 6,323 objects / 48,739,734,061 bytes
(45.392 GiB): 1,022 unselected external-data prediction epochs, 60 tail bundles,
5,211 ephemeral model-cache objects, and 30 rejected-preflight sidecar objects.
Before deleting the sidecar duplicate, both copies of every array were streamed
and SHA-256 checked against semantically identical manifests; the only manifest
difference was the creation timestamp. The accepted bias-free sidecar tree was
retained intact.

The external-data program retains all 178 epochs selected by at least one of
the two honest parity replays, whole-fold Patience-3, or the final epoch, plus
all 178 matching raw/EMA checkpoints, 60 observation-alignment references,
histories, manifests, diagnostics, analyses, and screen summaries. Its footprint
fell from 52.032 GiB to 9.158 GiB. Raw/interim sources, the canonical feature
store, canonical parent and challenger artifacts, Experiments 39--40, and every
other retained object were unchanged.

The fresh post-cleanup inventory contains 12,649 objects / 60,899,434,233 bytes
(56.717 GiB), including the two new audit records. Its SHA-256 is
`20f8fa4258e914f2a7731bfd0cee42d809fc34488c5f0a2716a28c6e7d9ecce6`.
The immutable object-store audit is under:

    quant-data/b3/processed/model_runs/_retention/storage_cleanup_20260823_round2

The applied plan SHA-256 is
`eda1cb77d361ac0a8fa8b5e00460aff71477812a29e63f9f46ded7535937b64b`;
the postcheck SHA-256 is
`cd68b22bc23a891a73eab6c1b75cc61e82e524dae02439153fb37cf965b23ce6`.
An independent before/after comparison found zero unexpected removals, zero
planned survivors, zero retained-object metadata changes, and exactly the two
expected audit additions. Recreating deleted prediction trajectories would
require an exact rerun from the recorded commit and retained canonical inputs.
No paid Lambda instance was active.

## Kronos-small zero-shot K0 decision (2026-08-22)

Kronos-small was evaluated zero-shot on the two 102-date discovery selection
windows using permanent point-in-time equity identity, the fixed six-decision
grid, 512 causal five-minute bars, exact per-context sampling seeds, and the
canonical 30/60/120-minute rank-IC machinery. The user narrowed the experiment
to Kronos-small before any small score or metric was inspected; an in-progress
Kronos-base pass was stopped, excluded, and its unmerged partial arrays deleted.

Kronos-small scored `0.008843` on Fold A and `0.018551` on Fold B, for mean IC
`0.013697`. This is below the preregistered `0.015` kill floor. Its matched
momentum control was `-0.016037`, parent correlation was only `0.133768`, and an
informational parent-plus-Kronos rank stack added just `+0.000128` mean IC with
opposite fold signs and block-bootstrap intervals spanning zero. The zero-shot
Kronos-small family is therefore rejected for the current program. Do not run
K1 or use official validation on the basis of K0. The canonical parent remains
unchanged, and the held-out test remains sealed.

The immutable run and reusable score artifact are:

    /lambda/nfs/brazil-rv-east3/quant-data/b3/processed/model_runs/kronos_k0_3f93b26_20260822T134800Z

Its manifest records `official_validation_accessed=false`,
`test_accessed=false`, and `k1_started=false`. The dedicated immutable K0 bar
sidecar is `kronos_k0_bars_3f93b26_20260822T134400Z` in the same model-runs
root. Exact settings, leakage caveats, operational corrections, model-scope
override, metric breakdowns, hashes, and cleanup provenance are in
`EXPERIMENT_LOG.md` and the run manifests.

Transient base partials, model caches, upstream clones, and MPS state were
deleted. The retained run and sidecar are read-only. Paid GH200 instance
`c0aef7522bf64fe0899e8703027668db` was terminated and confirmed absent from
Lambda's active inventory.

## P0/P1 feature-program decision (2026-08-22/23)

P0.2/Kronos closure was explicitly excluded and never run. P0.1 rejected both
predeclared mixed-state ensembles: the 24-member all-family stack added
`-0.001242/+0.001709` on Fold A/B (mean `+0.000233`), while the 12-member
residual/options/ADR stack added `-0.000272/+0.002278` (mean `+0.001003`). Both
failed the non-negative-every-fold gate.

Cross-fitted inference attribution classified 12 of 58 incumbent equity fields
dead, but they were removed only inside the subsequent joint candidate. F2
selected eight causal features on a disjoint first-407-date window. On Fold
C/A/B, that bias-free sidecar plus the 12-field ablation added
`-0.000568/+0.000576/+0.001054` standalone (mean `+0.000354`) and
`-0.000015/+0.000711/+0.000920` when pooled with parent-3 (mean `+0.000539`).
Neither path passed the preregistered three-fold gate. Final EMA and the P0
mixed-state secondary reads were also null-to-negative. F4 therefore recorded
`not_run` without ablation or retraining.

Reject the P0.1 stacks and the joint P1 feature/pruning recipe. P0.3 and F2 remain
diagnostic evidence only; do not delete fields from the canonical parent or
promote individual F2 features based on the joint screen. Raw Patience-3 on the
unchanged three-seed parent remains canonical, the designated challenger is
unchanged, official validation was not spent, and the held-out test remains
sealed. Exact results and source-reproduction validation are under:

    /lambda/nfs/brazil-rv-east3/quant-data/b3/processed/model_runs/p0_p1_27aa0d0_20260822T194900Z

Experiment implementations are reproducible through commits `5b6b5d4`,
`1b63661`, `27aa0d0`, and `8f46124`; rejected campaign-only code is absent from
current HEAD. Full settings, selected features, intervals, operational repair,
and artifact hashes are recorded in `EXPERIMENT_LOG.md` Experiment 39.

The final retention cleanup removed 26.607 GiB of redundant checkpoints and
per-epoch predictions while preserving all selected Patience epochs, epoch-20
EMA states, observation references, sidecars, manifests, analyses, and result
summaries. Its exact plan and postchecks are under the program root's
`_cleanup/20260823T024900Z` directory. Paid GH200 instance
`b3eac682796a4e1ea7912422a81f0e85` was terminated after the results were pushed
and was confirmed absent from Lambda's inventory twice.

## Final feature closure and P2 strong-source decision (2026-08-23)

Experiment 40 completed the feature program's final features-only test and the
three independently preregistered P2 screens. The unchanged eight-feature F2
sidecar, now tested without P0.3 pruning, added only `+0.000179` mean IC
standalone and `+0.000264` in the fixed parent stack. The fixed
late-market-momentum/HKS pair added `-0.000474` standalone and `-0.000213` in
the stack, with a materially negative Fold-B result. The feature program is
closed: retain neither candidate and do not search further feature subsets from
these readouts.

B3 lending rates/flows added `-0.000250` standalone and `-0.000051` stacked;
DCE iron ore added `-0.000499` and `-0.000122`. B3 listed-equity option open
interest was positive on all three folds but added only `+0.000586` standalone
and `+0.000400` stacked. None met the frozen `+0.001` mean and non-negative
every-fold gate, so no P2 source is promoted and no post-readout P2 combination
is authorized. Raw Patience-3 on the unchanged parent remains canonical, and
the designated challenger remains informational only.

The previously missing option history was acquired as exact official
BVBG.086/BVBG.028 final reports: 1,154 complete daily pairs from 2019-11-01
through 2024-06-28, preserved as an immutable 15 GB raw archive. Exact
underlying-instrument and dated cash-ISIN mapping yielded a normalized source of
95,045 rows and 142 permanent IDs. The free historical source does not expose
covered/uncovered positions, so those fields were not fabricated. DCE remains
the disclosed contract-specific Sina mirror because the official endpoint was
not freely accessible.

The completed program, all summaries, sidecars, retained training evidence, and
source provenance are under:

    /lambda/nfs/brazil-rv-east3/quant-data/b3/processed/model_runs/experiment40_final_feature_p2_0b6ff68_20260823T075000Z

The program manifest/summary SHA-256 values are
`495e69adf98b190a697c17d620a4ccb04dc02f4f05c13e4ce039ce0caee871ae`
and `db7ba8028f6955b2e753a4ed5272b792fd9f72fdc7325e5970a779d77814331c`.
All 45 trajectories and five three-fold analyses completed at repository commit
`0b6ff68a64276fff53c770b49b1ab9db64120e4b`; official validation and the held-out
test remained sealed. Exact fold deltas, uncertainty, availability contracts,
source hashes, and audit details are in `EXPERIMENT_LOG.md` Experiment 40.

The reviewed retention cleanup removed 36.078 GiB of redundant checkpoints and
per-epoch predictions while preserving all opposite-parity-selected epochs,
epoch-20 raw/EMA states, observation references, sidecars, manifests, histories,
analyses, and summaries. Its exact hash inventory and passing postchecks are
under the program root's `_cleanup/20260823T170000Z` directory; the final program
root is 6.7 GiB.

Paid GH200 instance `95098103c2da4ffcb8e9d10a4ac7704c` was terminated only
after results, cleanup evidence, and GitHub state were secured, then confirmed
absent in two consecutive Lambda inventory reads.

## Incumbent feature-removal decision (2026-08-24)

Experiment 41 produced a non-inferior retrained 34-field specification from the
58 incumbent equity inputs. The selected prune-R2 candidate added
`+0.000898/-0.000216/+0.002978` Raw Patience-3 IC on Fold C/A/B, mean
`+0.001220`; every fold stayed above the fixed `-0.0005` floor. Prune-R1 failed
that floor on Fold A. This is a store-v2 specification for the next rebuilt
parent, not an in-place mutation of the current canonical store/model, and it
does not authorize or alter the official-read lineup.

Remove six dynamic fields: `return_60m_normalized`,
`realized_vol_30m_log_ratio`, `session_range_position`,
`cross_section_return_rank_15m`, `cross_section_volume_rank`, and
`cross_section_volatility_rank_30m`. Remove 18 slow fields:
`overnight_gap_normalized`, `previous_close_to_close_return_normalized`,
`previous_open_to_close_return_normalized`,
`median_daily_real_volume_20d_log_scale`,
`median_daily_dollar_volume_20d_log_scale`,
`daily_dollar_volume_regime_20d`, `observed_fraction_5d`,
`observed_fraction_20d`, `dollar_volume_cross_section_rank`, `beta_to_WIN`,
`beta_to_DI1F27`, `beta_to_DI1F28`, `beta_to_DI1F29`, `beta_to_DI1F31`,
`weekday_sin`, `weekday_cos`, `month_end_proximity`, and
`quarter_end_proximity`. Retain the other 34 incumbent inputs; specifically,
the preview-proposed `volume_surprise`, 15/60-minute market-median returns,
15-minute market breadth, and 60-minute cross-sectional return rank remain
KEEP.

The frozen Stage-A/B root is
`feature_removal_d5b5e1f_20260823T224100Z`; the completed isolated Stage-C root
is `feature_removal_stage_c_repair_d5b5e1f_20260823T232938Z`.
Summary/specification SHA-256 values are
`1070ecfadb99eef42d224b8eacc0ef31fc8e0e08ecc6a7e39aa5153e57fb18b8`
and `08c04de3396fdc31d67b6baeabab1fea80cfd137d55bf2a1aef4ee69d1a34b72`.
The final audit passed with official validation and held-out test sealed. Exact
fold intervals, all 58 rule-attributed verdicts, the operational Stage-C repair,
and retention-cleanup hashes are recorded in `EXPERIMENT_LOG.md` Experiment 41.

A later explicitly authorized object-store cleanup removed 315 unselected
per-epoch prediction archives and 18 redundant tail bundles from the completed
Stage-C repair, totaling 14,309,636,868 bytes (13.327 GiB). The repair retains
the same frozen union of 45 epoch-20/cross-fit/whole-fold Patience predictions
and matching checkpoints, all 18 observation references, every history,
analysis, manifest, summary, verdict, and audit artifact. A fresh full-bucket
inventory found exactly the 333 planned removals, zero unexpected removals,
zero retained-object metadata changes, and the two expected immutable audit
additions. The object store now contains 12,941 objects / 63,474,753,532 bytes
(59.115 GiB). The exact plan and postcheck are under
`model_runs/_retention/storage_cleanup_20260824_round3`.

Paid GH200 instance `e975b774f5834e0fa265d11bbbef680f` was terminated only
after results and audit evidence were pushed, then confirmed absent in two
consecutive provider inventory reads. No paid Lambda instance remains active.

## R3 and full-options decision (2026-08-24)

Experiment 42 rejected all three attempted advances from the selected
Experiment-41 prune-R2 specification. Correlation-conditioned Stage B-prime
froze three further removal candidates: `realized_vol_60m_log_ratio`,
`realized_vol_20d_log_ratio`, and `vol_of_vol_20d`. Their retrained R3 candidate
added `+0.000496/-0.001669/-0.000260` Raw Patience-3 IC on Fold A/B/C (mean
`-0.000478`) and failed non-inferiority. Retain prune-R2's 34 fields as the
store-v2 specification; do not remove these three fields and do not run R4.

The full 14-field options program added
`-0.000410/+0.001526/-0.000073` standalone (mean `+0.000348`). The five-field IV
subset added `-0.001134/-0.002320/-0.000563` (mean `-0.001339`). Neither passed
the frozen fold/mean/uncertainty gate. Their predeclared mixed states also
failed, so the options family is parked and no candidate is registered for the
future official-read cycle. Do not search a third options subset from these
readouts.

The causal full-options source is now durable: 95,045 rows, 142 permanent IDs,
2021-07-19 through 2024-06-28, output SHA-256
`6a0cff033fb48a3b190ba49389e173c385ee1df0211a335e81665e9ec2af5686`.
The manifest-declared unpublished 2023-12-08 instrument master remains invalid;
no adjacent-date master was substituted. The completed program is:

    /lambda/nfs/brazil-rv-east3/quant-data/b3/processed/model_runs/r3_options_9a05b1d_20260824T053052Z

Program/R3/options/source-manifest SHA-256 values are
`888a1cef6c1488365db3870aa434127767c36ee544a82b8deed58dab7382f91d`,
`a058efc42b66be5117b13c044c774cf531ba57df8294823121bfd3877d6df452`,
`118746188fe2cc5d61c9f7dfad7a7a68173c433b6ff11979c957907d4cf4dafb`,
and `d03afc6c75fff445bac4d57df09b72d373f459e1edc08d4e15af679c20711509`.
All 27 trajectories completed; official validation and held-out test remained
sealed. Exact gates, fold analyses, source diagnostics, the pre-score
unpublished-master repair, and cleanup evidence are in `EXPERIMENT_LOG.md`
Experiment 42.

The reviewed cleanup removed 467 redundant checkpoints / 2,113,294,685 bytes
while preserving all 540 prediction archives, 73 selected/final checkpoints,
every analysis, manifest, source artifact, and sealed-data record. Its passing
postcheck is under the program root's `_cleanup/20260824T121000Z` directory.

Paid GH200 instance `e2cf2e517d9541ac93cac3906fc5c0e4` was terminated only
after the results, cleanup evidence, and documentation commit were secured. It
was absent from two consecutive provider inventory reads, and the Lambda
account then had zero active instances.

## Official-read deployment decision (2026-08-24)

Experiment 43 consumed the third and sole authorized official-validation read
for this lineup. Canonical parent-3 Raw Patience-3 scored `0.041639843`. The
stored residual challenger added `+0.000453978`, and the retrained Experiment-41
store-v2 mask added `+0.001595530`, but their paired block-10 95% intervals had
lower bounds of `-0.000684798` and `-0.000294105`. Neither arm passed the frozen
support rule. Canonical parent-3 therefore remains deployed; no 10-seed
expansion was run, and neither official retuning nor another read is authorized.

The completed program is:

    /lambda/nfs/brazil-rv-east3/quant-data/b3/processed/model_runs/official_read_c04ea91_20260824T140900Z

All 63 newly generated official prediction archives are retained. The reviewed
cleanup removed only 60 non-deployed store-v2 checkpoints / 270,571,140 bytes;
its plan and passing postcheck are under the program root's
`_cleanup/20260824T153000Z` directory. The validation ledger records event 3,
and every artifact records `test_accessed=false`. The held-out test remains the
final sealed lockbox.

Paid GH200 instance `c2da7efd0ab645178a847aad8fdf12c8` was terminated only
after the completed artifacts, cleanup audit, and result commit were secured.
Two consecutive provider reads confirmed the exact instance absent and the
Lambda account at zero active instances.

## Ensemble-science decision (2026-08-25)

Experiment 44 completed the frozen archive-only E1 analysis and all 45 E2
diversity trajectories without opening official validation or the held-out
test. E1 alone did not pass: its honest greedy mean delta was `+0.000966020`,
just below the complexity-adding `+0.001` gate. E2-only greedy also failed at
`+0.000667147`. The predeclared full-pool greedy path passed with Fold C/A/B
deltas of `+0.001739170/+0.000767959/+0.000721738`, mean `+0.001076289`.
Exactly one future-read arm, `e2_plus_archive`, is registered. Its exact
fold-specific member/repeat compositions live in the immutable Experiment-44
analysis; it is not deployed and does not itself authorize an official read.

Experiment-43 Amendment A1's separate exact reproduction failed because none
of 60 reproduced epoch prediction archives was byte-identical, despite matching
static input manifests and only small score differences. The original
Experiment-43 deployment decision therefore remains unchanged. Experiment 44
was explicitly independent of that reproduction verdict.

The completed program is:

    /lambda/nfs/brazil-rv-east3/quant-data/b3/processed/model_runs/ensemble_science_8dff0be_20260824T174700Z

The completed analysis SHA-256 is
`b933e740034cf6499d4569d98ebdb4242c697ee6664c44cafcbc9c606aa913b7`.
All 945 E2 epoch/tail prediction archives and required analyses/manifests are
retained.
The reviewed cleanup removed only 900 redundant E2 checkpoints /
4,060,129,980 bytes; its passing evidence is under
`_cleanup/20260825T045600Z`. Exact gate details, roster/design hashes, the A1
audit, and cleanup hashes are recorded in `EXPERIMENT_LOG.md` Experiment 44.

Paid GH200 instance `74ecb13e8b704ffcad890a5930ee74fd` was terminated only
after the result commit and retained artifacts were secured. Two consecutive
provider inventory reads confirmed it absent and the Lambda account at zero
active instances.

## Consolidation-read deployment decision (2026-08-25)

Experiment 45 consumed official-validation access event 4 and changed the
deployed measured recipe from canonical parent-3 to the fresh ten-seed
store-v2 ensemble. Fresh store-v2 seeds 11/29/47 reproduced the retained
Experiment-43 comparator within `+0.000004572` (`0.043239945` versus
`0.043235373`). The ten-seed form scored `0.043718770`, gaining
`+0.000478826` versus fresh three and passing its frozen deployment rule.

The frozen `e2_plus_archive` consensus scored `0.043916831`, but its paired
block-10 interval versus the comparator was
`[-0.000853732, +0.002373780]`. Its lower bound did not exceed zero, so the
complexity-adding arm was not supported and no hybrid was evaluated. The
deployed recipe is the uniform tie-aware rank average of store-v2 Raw
Patience-3 seeds `11/29/47/61/79/97/113/131/149/167`. Its measured selected
and final checkpoints are retained until a future deployment supersedes them.

The completed program is:

    /lambda/nfs/brazil-rv-east3/quant-data/b3/processed/model_runs/consolidation_read_e2eb713_20260825T105134Z

Its frozen design and completed manifest SHA-256 values are
`05e0eebfffe95fc870e3e8a1138689aebe5e0595a210333d03a60cd15f1ebe2b`
and `64a69be9345a0d9cc4a2a506f9f295fd50f9a2bcc210793458adfa9429cda34e`.
All 298 prediction archives and 18 references are retained. The event-4 ledger
and every member contract record `test_accessed=false`; the held-out test
remains sealed. Exact results, analyses, the score-neutral specialist JSON
repair, and the 730-output completion audit are recorded in
`EXPERIMENT_LOG.md` Experiment 45.

After the deployment declaration was pushed, the reviewed cleanup removed
only 84 checkpoints / 370,484,520 bytes belonging to the eight non-deployed
jobs. It retained all 298 prediction archives, all analyses/references, and all
200 checkpoints belonging to the ten deployed measured jobs. The cleanup plan
and passing retained-output postcheck are under
`_cleanup/20260825T164320Z` in the completed program root.

Per the user's explicit operational override, paid GH200 instance
`d0ebcd5f7dbb44dc99370080df7b47cc` remains active for the immediately
following experiment rather than being terminated at Experiment-45 closure.

## Experiment 46 cross-equity decision (2026-08-25)

The fixed monthly given-graph program completed on discovery folds only. N0
found improved stability but materially negative rotated IC delta, so no
neutralization transform was adopted; the T-peer registration condition also
did not trigger. F2 retained four peer fields: relative 60-minute return,
relative prior-session return, peer-mean 60-minute return, and peer-dispersion
60-minute return.

The nine-trajectory F3 candidate failed both frozen paths. Standalone Fold
C/A/B deltas were `+0.000054345/+0.000347304/-0.000881236`, mean
`-0.000159862`, and both pooled 95% intervals crossed zero. Parent-plus-candidate
mean delta was `-0.000007868`. No future official-read arm was registered and
the deployed Experiment-45 ten-seed store-v2 recipe is unchanged. The nine
final EMA-0.995 candidate members remain eligible only for a future
ensemble-pool registration, as preregistered.

The immutable program root is:

    /lambda/nfs/brazil-rv-east3/quant-data/b3/processed/model_runs/cross_equity_46_affea34_20260825T170000Z

Its 552-artifact pre-cleanup audit passed at SHA-256
`2f82faef622882768af07aeec9e3e750b195c5f49c5929f9344f5da52ab2b74b`.
The reviewed cleanup retained all 180 prediction archives and 24 exact
selected/final checkpoints while removing 156 redundant checkpoints; its
passing postcheck SHA-256 is
`68ab557488c49895821c62a38045581c5daa4d696b579e64848e30dc42a97427`.
Every access flag records `official_validation_accessed=false` and
`test_accessed=false`. Exact graph, N0, F2/F3, operational-repair, and artifact
hash details are in `EXPERIMENT_LOG.md` Experiment 46. Paid GH200 instance
`d0ebcd5f7dbb44dc99370080df7b47cc` was retained for Experiment 47 by explicit
user request and terminated after that program completed.

## Final HPO and next-generation recipe decision (2026-08-26)

Experiment 47 closed the bounded HPO and architecture axes. Its simplification
track adopted R1: four TCN residual blocks with dilations `(1, 4, 16, 32)` on
the 34-field store-v2 input specification. Experiment 48 then prospectively
combined R1 with soft-Spearman temperature `1.00`. The combination added
`+0.000198613/+0.000944777/+0.000125564` on discovery Folds A/B/C, mean
`+0.000422984`, with positive pooled block-5 and block-10 lower bounds. It
passed the frozen non-inferiority rule and is the accepted next-generation
training recipe: `R1_T1.0_three_head_30_60_120`.

The Experiment-48 early/late decomposition strongly replicated early
30-minute realization on all three folds, so a fixed equal-weight fourth
15-minute head was tested. Its incumbent three-horizon delta was
`+0.000819706/-0.000864794/-0.000089351` on Folds A/B/C, mean
`-0.000044813`; Fold B failed the `-0.0005` floor. Keep the three-head recipe.
The measured 15-minute head ICs were `0.054073317/0.051798396/0.038488832`
on A/B/C. A dedicated 15-minute model is only a future registered option and
was not built. No validation read, held-out test read, read registration,
pool scoring, or deployment change occurred. The deployed recipe remains the
Experiment-45 ten-seed store-v2 ensemble.

The immutable Experiment-48 program is:

    /lambda/nfs/brazil-rv-east3/quant-data/b3/processed/model_runs/nextgen_48_e5c1983_20260826T132551Z

Its frozen-design and final-audit SHA-256 values are
`def9ff3229e66825df4ef6856130edd2a12f599ed299ef30a9aa5b93de0192bd`
and `4b43baa7b1760ca22cc10c5f48d084e7d4976154d155d6a714d1539915db38d3`.
The reviewed cleanup retained every prediction/analysis and 47 exact
epoch-20/honest-selection checkpoints. Full results, hashes, and cleanup
evidence are recorded in `EXPERIMENT_LOG.md` Experiment 48.

Paid GH200 instance `f766c33a775344d394ec0bdc915fca6d` was terminated only
after the immutable artifacts, final audit, cleanup evidence, and result commit
were secured. Two consecutive provider reads confirmed it absent, and the
Lambda account then had zero active instances.

## Economics robustness and official-read event 5 decision (2026-08-27)

Experiment 49 rejected the shared 15-minute head for the next official read.
Although midpoint-label retention passed on all discovery folds, executable
15-minute net return was negative on every fold and the combined 15m/30m book
improved net Sharpe on zero folds. Keep the accepted next-generation recipe at
three heads (30/60/120 minutes); the dedicated 15-minute model remains only a
registered, unbuilt option.

Experiment 50 consumed official-validation access event 5. The ten-seed
three-head next-generation ensemble scored `0.043588809`, versus
`0.043718770` for the deployed Experiment-45 store-v2 ten-seed ensemble. Its
paired block-10 lower 95% bound was `-0.001054764`, failing the frozen
inclusive `-0.0005` non-inferiority rule. Do not deploy the next-generation
candidate. The deployed measured recipe remains the Experiment-45 store-v2
ten-seed uniform tie-aware rank ensemble.

The immutable programs are:

    /lambda/nfs/brazil-rv-east3/quant-data/b3/processed/model_runs/economics_49_77411e4_20260827T004334Z
    /lambda/nfs/brazil-rv-east3/quant-data/b3/processed/model_runs/nextgen_read_50_77411e4_20260827T004334Z

Experiment 50 retains all 210 official prediction archives and the 20 exact
selected/final candidate checkpoints after reviewed cleanup. Its final audit
SHA-256 is
`aca83d6d84c458b7e1db19645bc411942f6e8ef8530e86bda040a62ca2c116fa`.
Official validation was accessed exactly once for event 5; the held-out test
remains sealed. Full economics measurements, artifact hashes, operational
repair provenance, access ledger, and cleanup evidence are recorded in
`EXPERIMENT_LOG.md` Experiments 49 and 50.

Result/context commit `1683dfb` reached GitHub before shutdown. Exact paid
GH200 instance `2407fd931c3f47b7825bf6538617571d` was then terminated after
all retained evidence was secured. Two provider reads, at
`2026-08-27T03:54:38Z` and `2026-08-27T03:54:57Z`, confirmed the exact ID
absent and the account at zero active instances.
## First and only held-out test read (2026-08-27)

Experiment 51 measured the deployed Experiment-45 store-v2 ten-seed uniform
tie-aware rank ensemble on the 259-date held-out period exactly once. Before
access, the program rehashed the exact deployed-recipe declaration, all ten
member manifests and selected checkpoints, and the 20-file deployed checkpoint
inventory. No comparator, training, selection, new weight, execution metric,
or deployment change was permitted.

Test IC was `0.040345936`, Band A under the frozen interpretation bands. The
block-5 95% interval was `[0.033209566, 0.047739292]`; block-10 was
`[0.033313199, 0.047668906]`. Per-horizon ICs were `0.040721248` at 30 minutes,
`0.039138348` at 60 minutes, and `0.041178212` at 120 minutes. The frozen paired
staleness statistic, H2 minus H1, was `+0.012885851` with block-10 interval
`[-0.007851288, +0.023472244]`; it did not indicate deterioration or a
retrain-before-live policy. This read is calibration only and leaves the
deployed recipe unchanged.

The immutable program root is:

    /lambda/nfs/brazil-rv-east3/quant-data/b3/processed/model_runs/test_read_51_67f43b3_20260827T130349Z

The final program-manifest, artifact-inventory, analysis, result, access-ledger,
and completion-audit SHA-256 values are respectively
`19b87ed6d23a9d38b9defaa923a91e280e7af47d684625ae6c36cbd1a99c3cbf`,
`04a9c367089c6c1c5146cf8464521dc43200f4b26d07220ff3cabe5ef8d61f33`,
`013a490ae8c4a0b860edb1db4b84f4c47322785a9ae26726472bbb1f34a63e16`,
`23fa452636245170659a7889181f1285c3779d47cb3644236f88113359d88fa8`,
`5ad518caf94f02c4b8b9b02ed5f6afd4a78110002819ca942ab9bdb9d41bf517`,
and `99c06a406fd6cc3eed7f1d6adb76defbb5047f0bad1887b41f2c7a83c36626a8`.
The audit covers 27 artifacts / 358,294,145 bytes, including all ten member
prediction archives, the ensemble archive, test reference, predeclared
analyses, preregistration, interpretation statement, and operational logs.

The ledger records `test_accessed=true` and `test_spent_forever=true`. Current
code refuses further test evaluation. No second test read is authorized under
any registration; future validation requires new forward data.

Result/context commit `23a100750397e903aa8ef394af2c12b0906e30fd` reached
GitHub before shutdown. Exact paid GH200 instance
`a6c710df4bfa48c9a40f64ee4b7e85c4` was then terminated; provider inventory
confirmed the exact ID absent twice at `2026-08-27T13:17:35.6228739Z` and
`2026-08-27T13:17:57.0202321Z`, with zero active instances on both reads.

## Persistent Lambda retention cleanup, round 4 (2026-08-27)

Direct S3-compatible access was used for a reviewed cleanup; no paid Lambda
instance was launched. After the plan records were present, the bucket held
19,918 objects / 286,524,998,316 bytes (266.847 GiB). The exact delete list
removed 3,521 objects / 183,169,695,704 bytes (170.590 GiB), a 63.93% reduction.
The final bucket, including all six round-4 plan/result records, contains 16,399
objects / 103,355,305,165 bytes (96.257 GiB).

The deletion scope was limited to unselected `validation_predictions/epoch_*`
archives and redundant `tail_candidates.npz` bundles in eight closed programs:
Experiments 42, 43, 44, 45, 46, 47, 48, and 50. It retained 433 exact
selected/final or honest-selection prediction states across those programs,
their matching retained checkpoints, every manifest/history/reference/analysis,
the Experiment-44 frozen 90-member state catalogue, and all non-deployed
Experiment-45 arm archives. Experiment 45's deployed ten members retain their
exact selected and final predictions and checkpoints.

Raw data, interim data, the canonical feature store, external sidecars,
auxiliary targets, and the complete Experiment-51 test-read root were outside
the deletion scope. Their verified post-cleanup sizes were respectively
18,224,073,078; 287,164,841; 9,890,909,028; 1,952,163,984; 517,990,485; and
358,303,408 bytes. The historical unselected per-epoch prediction archives
removed by this cleanup are not recoverable in place; exact trajectory reruns
use the recorded commits, manifests, histories, retained selected states, and
canonical derived data.

The immutable cleanup record is:

    quant-data/b3/processed/model_runs/_retention/storage_cleanup_20260827_round4

Cleanup-plan, exact delete-list, and cleanup-result SHA-256 values are
`1bd8bb8f99f6821066ad5c03f51ca420d40c9960ee6b2fa3bbb881d3d0d9e7b1`,
`e262d1fc2430e2d9d0e031a54e21fe40ae2c6ae89be2f5185070db15c8f01f02`,
and `0b59fa9611a5e098cbd337c63e38ee8cfef941e415dc7f0f258092dca7643eba`.
Provider inventory at `2026-08-27T14:31:41.9870434Z` showed zero active
instances.

## Offline execution-backtest contract (2026-08-27)

`research/src/brazil_rv/execution` is the canonical research execution layer.
It is an offline replay, not a live trading or broker interface. The simulator
uses the 405-minute equity session but receives score-refresh minute indices
explicitly. Current `15,20,...,285` five-minute refreshes and a future dense
one-minute refresh grid therefore share one interface; no code derives cadence
from `decision_idx * 5`.

Raw prediction scores are ranked only over the causal point-in-time activity mask
derived from the hash-bound canonical store. Metric-oriented ensemble ranks
formed with `label_mask` are forbidden for execution because that mask includes
future label-endpoint observability. The archive loader binds sample/date/decision
rows and explicit refresh minutes to the canonical sample index, accepts only
hash-bound discovery-fold inputs through `TRAIN_END`, and requires the completed
source run's fold plus selection-window sample/date hashes to match the archive.
Aggregate OOF inputs remain rejected until a canonical materializer binds every
constituent fold and proves fit-window exclusion per sample. This rejects both
official/test access and in-sample runs relabeled as OOF, while supporting the
present 5-minute cutoffs and future one-minute canonical cutoffs without a
cadence formula. Experiment 51 remains the sole and permanently spent held-out
test read.

The real-data bridge reuses canonical permanent-security assignments and streams
exact observed M1 opens, closes, and `real_volume` only through the training end.
ADV20 and minute-of-day capacity profiles emit before consuming the current
session. Experiment-49 Roll schedules may be used only one completed quarter
late; missing estimates use strictly trailing prior-session Roll inputs. The
same-quarter schedule is not causal for execution decisions.

Actions fill at the next observed minute open. Holdings are marked open-to-open;
half-spread and per-side fees are charged once on traded notional. Participation
and projected-target caps constrain convergence; price-move breaches de-risk only
as recorded liquidity permits. A missing open creates neither a mark nor a fill:
the holding retains its last observed notional and realizes the cumulative
return when the next observed open arrives. A transient rank-validity mask that
cannot support exact signed cap capacity retains the last feasible projected
target. Band and concentrated policies retain the hard exact-gross
neutral/cap projection. A learned policy uses a separate bounded projector that
is still exactly neutral and capped but never scales exposure up, so gross may
remain below target or exactly at zero.

A linear close taper targets zero at the final open, and any remaining terminal
position is force-filled at the recorded spread multiplier and counted,
independent of the ordinary participation/profile gate. When the exact final M1
slot is unobserved, this explicit offline approximation uses the name's last
observed session open and spread without changing the observation mask. Cash
accrual uses an explicit dated CDI input and one configurable margin formula. No
canonical daily CDI execution series currently exists, so runs must supply and
hash-record one.

The baseline policy remains the deterministic no-trade-band policy. The
execution package contains a zero-initialized shared per-name neural policy, a
hash-frozen strictly causal state built inside the minute scan, volume-weighted
position cost basis, direct AdamW optimization over net PnL above all-cash CDI
with optional SAM/replay checkpointing, and hashed five-block purged TRAIN split
generation. The completed learned-policy OOF archive reconstructs those exact
purged folds, verifies all 50 source runs and every sample's fit exclusion, and
rejects any unproven aggregate. This manufacture and calibration do not
establish an accepted policy. The modeling layer also supports the zero-start
horizon-conditioned to-close readout and its TRAIN-only hashed target sidecar;
the three incumbent heads and RNG stream are unchanged at initialization.
Official-validation access and any additional held-out-test access remain
unauthorized. Cluster penalties, round lots, impact, and live routing remain
outside the contract.

## First execution reference — Experiment 52 C0 (2026-08-27)

Experiment 52 ran the frozen 12-cell no-trade-band grid on discovery Folds
C/A/B with exact measured spreads/fees and matched frictionless counterparts.
The rotation rule designated `band_2p0__blend_equal` as C0: band `2.0` and
equal 30/60/120-minute horizon weights. It won all three rotations. All 36
measured cells were net-negative. Across the three C0 fold windows, gross PnL
was `R$2.909m`, measured net PnL was `-R$14.910m`, and the frictionless
counterpart was `+R$3.373m`; spread plus fee drag was `R$18.123m` on
`R$26.534bn` turnover. C0 is the reference future execution policies must beat,
not a live policy, promotion, or prediction-recipe change.

The immutable program root is:

    /lambda/nfs/brazil-rv-east3/quant-data/b3/processed/model_runs/execution_c0_52_e380cd7_20260827T164738Z

Its frozen-design and final-audit SHA-256 values are
`5906da60ec6d1179f20ed368f0e519973c173fc968ee7467936d08e8c1687f41`
and `b7591bb6464da90aa3324dced14ebf04e2e29245043b2719b76fb75bbdbd061f`.
The audit covers all 72 reports plus input hashes, repair provenance, rotation,
C0 designation, and logs. Official-validation and test predictions were not
accessed.

Result/context commit `e4c7736583ce6390eed37b1c953eaf17f52c990e` reached
GitHub before shutdown. Exact paid GH200 instance
`356e43e74aa14abd84ad5bca30f70212` was terminated after the completed root and
audit were secured; two provider reads at `2026-08-27T17:10:06.4015869Z` and
`2026-08-27T17:10:24.4091678Z` confirmed the exact ID absent and zero active
instances.

## Experiment 53 feasible-region contract (2026-08-27)

Experiment 53 is the registered discovery-only follow-up to C0. It tests exactly
48 concentrated equal-blend cells across K `10/20/40`, base band `0.5/1.5`,
cost scale `0/1`, gross `1.0/2.0`, and full/top-half causal ADV20 universes.
Measured, frictionless, and half-spread variants are separately config-hashed.

For this experiment only, the name cap is 5% of gross. Rank-tail selection
extends deterministically when the combined name/ADV caps cannot support a side
target, and every refresh records the extension count. C1 excludes a cell if
its mean deployed gross is below 50% of target on any fold, while retaining its
reports. Among eligible cells, C1 uses C0's measured-only two-fold rotation and
tie rule. C0 and C1 remain references, not live or deployed policies.

Interpretation is constrained in advance: a measured net-positive cell is an
existence proof and lower bound for learned execution; an all-negative map is a
verdict only on this hand-policy family and never establishes that the alpha is
untradeable. Official-validation and held-out-test predictions are excluded.

Experiment 53 completed the frozen 432-report map and designated
`k40__band1p5__c1p0__gross1p0__universe_full` as C1. It won all three
measured-only rotations and deployed `82.7%` to `85.5%` of its gross target,
so it passed the preregistered gross guard. C1 nevertheless lost `R$5.394m`
net and `R$6.804m` relative to all-cash CDI across the three folds: gross PnL
of `R$1.428m` did not cover `R$5.488m` of spread cost plus `R$2.150m` of fees
on `R$10.750bn` turnover. Its fold Sharpes were `-7.188525`, `-8.505339`, and
`-14.026724` on A/B/C. Eight cells were net-positive on all three measured
folds, but all were top-half-ADV K=40 cells rejected by the gross-deployment
guard; no cell beat all-cash CDI on all three folds. Thus C1 is a feasible
hand-policy reference, not an economically accepted policy, and C0 remains the
retained baseline comparator. There was no model, deployment, official-
validation, or test change.

The immutable Experiment-53 root is:

    /lambda/nfs/brazil-rv-east3/quant-data/b3/processed/model_runs/execution_feasible_53_00cb38a_20260827T184000Z

The frozen-design and log-inclusive final-audit SHA-256 values are
`54344ff68ea55da1a13f6d3cb879a1154edb3eeaafb6839358fd1951ad22aecb`
and `bc5811f118cabae6514e889709a5b801551ebdd91f2b3094183107ad9d43f513`.
The audit passed over 2,606 files, all 432 reports and summaries, 2,447,280
selection-extension telemetry rows, analysis tables, manifests, hashes, and
both operational logs. Official-validation and held-out-test access remained
false.

Result/context commit `6862f5c044518f126bec56a1d991112057700a8c`
reached GitHub before shutdown. Exact paid instance
`21d5542246d144978478284e10837a22` was then terminated after the immutable
root and audit were secured. Provider inventory confirmed that exact ID absent
and zero active instances at `2026-08-27T19:14:50.9818266Z` and again at
`2026-08-27T19:15:18.8786125Z`.

## Experiment 54 edge and maker feasibility (2026-08-27)

Experiment 54 is the frozen discovery-only conditional-edge analysis over
Folds C/A/B. It did not run the simulator, train or select a policy, alter C0
or C1, or access official-validation or held-out-test predictions. Fold C alone
froze the state bucket edges; A/B reused them unchanged. The retained analysis
covers 2,219,567 eligible state events, four entry conventions at
`15/30/60/120` minutes, fixed `4.5/7/10`-bps taker hurdles, six conservative
maker schedules, and the preregistered positioning comparison.

At the 7-bps threshold, the optimistic same-fold state-cell taker frontier's
best registered horizon was 120 minutes on every fold and returned
`9.141462`, `7.944564`, and `10.429961` expected NAV bps/day on A/B/C. Two
folds cleared the frozen inclusive 8-bps/day gate, so taker actions remain
VIABLE for a future learned-policy stage and its reward hurdle is all-cash CDI.
This is a state-cell oracle upper bound, not realized simulator PnL, policy
acceptance, or deployment authorization.

The strongest maker frontier was the 120-minute, 15-minute-wait,
half-half-spread-improved limit on Fold A at `3.917194` expected NAV bps/day;
the matched B/C values were `3.739960` and `3.856317`. Aggregate strict-through
fill rates rose from roughly `64.3%–66.8%` at 5 minutes to `81.9%–83.7%` at
30 minutes. Maker results have no automatic gate and require a user decision;
none reached the 8-bps/day framing. The 120-minute informational construction
comparison did not establish a uniformly positive long-only-plus-cash or
long-short tail construction, and it did not change C0/C1.

The immutable root is:

    /lambda/nfs/brazil-rv-east3/quant-data/b3/processed/model_runs/execution_edge_maker_54_047e2d2_20260827T201700Z

Frozen-design, result, program-manifest, and log-inclusive final-audit SHA-256
values are respectively
`b358144752049d149305f4f7ecfc2021c1d224a176716efc4267b88e166018e7`,
`f5bcdfbfd654be29e1c9c758ee0276d9eb386eefd1290b5dadb7be5e11939fb1`,
`129dcfc09273d923948ea2c2dcf8e929885629776b2fc34b53d150ca1764aa86`,
and `d040f12b0a43356a53c1d03070da5a4d7c9f24f1f8cb4054dd73d3251e37c364`.
The audit passed over 21 retained artifacts / 495,173,023 bytes, including raw
OHLC hashes, bucket definitions, 48 latency rows, 276,881 taker conditional
rows, 36 taker frontiers, 163,440 maker conditional rows, 72 maker frontiers,
24 positioning rows, decisions, definitions, and both logs. Official-validation
and test access remained false. At the user's request, the paid GH200 is kept
active after the completed audit for the immediately following experiments.

## Learned-policy OOF and Experiment 55 decision (2026-08-28)

The canonical monitor-free learned-policy OOF archive is complete over the
exact five purged TRAIN folds and ten frozen seeds. It retains all 50 raw and
final-EMA member archives, exact 716-date causal rank-average coverage,
per-sample source-fold proof, loader verification, C/A/B calibration, and
checkpoint-cleanup provenance. Its immutable root and log-inclusive final-audit
SHA-256 are:

    /lambda/nfs/brazil-rv-east3/quant-data/b3/processed/model_runs/learned_policy_oof_53859b8_20260827T220239Z
    2b023ca3368fd06cc2098298c73704b4571a08516acdb93364280dd3d3525451

Experiment 55 trained only the frozen nine C/A/B-by-seed four-head trajectories.
The to-close head was economically promising: overall IC was `0.050540`,
`0.074533`, and `0.066367` on C/A/B, and incremental expected NAV edge versus
the Experiment-54 frontier was `+10.833850`, `+8.752321`, and `+9.168938`
bps/day. It is not adopted because its frozen three-head prediction guardrail
failed: Fold A's IC delta was `-0.001212`, below the `-0.0005` floor. The
conditional 50-run four-head OOF extension was therefore skipped, and the
deployed prediction and execution contracts did not change.

The completed Experiment-55 root and log-inclusive final-audit SHA-256 are:

    /lambda/nfs/brazil-rv-east3/quant-data/b3/processed/model_runs/experiment55_to_close_3b69ba8_20260828T055100Z
    21dd4f4ab1abb22e1445f2d8d1c5fcb3667952701620e68c6ba63183c4989cbd

Both programs kept official validation sealed and did not read the permanently
spent held-out test again. The historical test-spent state is unchanged.

## Experiment 56 four-head OOF and NeuralPolicy result (2026-08-28)

Experiment 56 adopted the Experiment-55 to-close head only as an execution-layer
research input; it did not change the deployed Experiment-45 prediction recipe.
Section A's deterministic four-head total frontier passed the continuation gate
on all three discovery folds at `26.567279`, `21.652961`, and `20.389667`
expected NAV bps/day on C/A/B. Section B completed and audited the exact 50-run,
1,000-epoch four-head OOF manufacture over 716 TRAIN dates. Its final OOF audit
SHA-256 is `d2e84d39fe02dbfeca428783ece7529730965bf2e82b01374d59ae0266d293ec`.

Section C completed all 18 preregistered NeuralPolicy trajectories. Each stopped
after 13 epochs under patience 10; lambda `0.02` was designated on C/A/B, and
nine checkpoints were retained. The frozen strict-positive rule mechanically
graduated the result because pooled daily net excess over all-cash CDI was
`0.0000120245` bps/day. The durable economic interpretation is nevertheless an
all-cash null: designated evaluations had zero turnover, zero round trips, zero
deployed gross, and no target or horizon-ablation effect. The positive pooled
value is float32 CDI-accounting residue rather than learned trading edge. The
stored graduation and paper-preparation flags remain true under the frozen rule,
but no deployed prediction or execution recipe changed.

The immutable completed Section-C root is:

    /lambda/nfs/brazil-rv-east3/quant-data/b3/processed/model_runs/experiment56_section_c_b383d30_20260828T164147Z

Its frozen-design, result, and log-inclusive full-inventory-audit SHA-256 values
are `66ec78c129f4c8dd4f6e80fa8456a204c6ba4b4e3ebfa89359ce3f99c29d1602`,
`3889faeced6fad0c0a655b731dab4ea2f4de52077b9c38c4e7657deccc049e1c`, and
`03c76b9db410fe93149f5b6a7e6b0ab59d96a40cd11d62b2bdebd1e8a0f6c541`.
The final audit covers all run artifacts and diagnostics, nine retained
checkpoint hashes, 200 source/result/log files totaling 418,331,972 bytes, and
official-validation/test access flags of false.

## Experiment 57 cross-fold frontier imitation result (2026-08-28)

Experiment 57 tested whether the four-head oracle frontier survives a realized
cross-fold execution replay and whether the existing NeuralPolicy can imitate
and improve it. Stage 0 selected the strict 7-bps threshold on every C/A/B
rotation, but the realized teacher lost `53.982522` NAV bps/day versus all-cash
CDI at mean deployed gross/NAV `1.543438`. The teacher therefore failed.

The exact nine 20-epoch imitation clones and nine clone-start lambda-0.02
fine-tunes still completed under the frozen contract. Fine-tuning was selected
over its own clone in all nine cells, but the pooled designated evaluation was
again exactly all cash: zero deployed gross and only `0.0000120245` bps/day of
float32 CDI-accounting residual. The actual-deployment guard correctly prevents
graduation. The standing Experiment-57 execution candidate is `none`; no
prediction, execution, or deployment recipe changed.

The immutable Stage 0/1/2 roots are:

    /lambda/nfs/brazil-rv-east3/quant-data/b3/processed/model_runs/experiment57_stage0_60c0b66_20260828T212520Z
    /lambda/nfs/brazil-rv-east3/quant-data/b3/processed/model_runs/experiment57_stage1_60c0b66_20260828T213216Z
    /lambda/nfs/brazil-rv-east3/quant-data/b3/processed/model_runs/experiment57_stage2_60c0b66_20260828T214712Z

Their result SHA-256 values are
`4047be5d6c08ad957f9348d95b4f4ad87b92360db8fd642dd5d05d9112880c55`,
`9228c7165bbadd85c91b9919a1130b38fefa3705499222f5cbe1165ccfca3745`,
and `d7ecc29309bae6a55f926390183d522ced513c5f113f5dec6b2c5e967b345015`;
the log-inclusive full-inventory-audit SHA-256 values are
`453ee7b366b574367b9f8c5e54927cef5d640c1a8eefb661e8f185b58d56a3af`,
`97eafe757b4f02c2548cadcdc6e155d4cfeaba856e799f692475b30623d6b315`,
and `33e2ce573ebef9f8f6d52f2c6c6d6d820b76f653c63fb26291620619c43460c6`.
All fresh-root official-validation and permanently spent test access flags are
false.

## Experiment 58 swing feasibility result (2026-08-29)

Experiment 58 is the canonical TRAIN-only swing screen of the ten-seed
four-head OOF archive. It establishes a modest predictive floor but rejects the
registered daily auction construction. The all-TRAIN last-hour four-head mean
IC at 1/2/3/5/10 sessions is
`0.016150/0.015986/0.014297/0.022234/0.021047`, with positive block-10 lower
bounds at every horizon. C/A/B intervals individually cross zero and daily rank
persistence is only `0.189563`, so the signal is not yet regime-robust or
slow-moving.

No frozen Part-B cell clears all-cash CDI. At the middle 4-bps cost and 2%
borrow, the best cell (four-head mean, K=30, band=0.3) returns `-10.086775`
NAV bps/day with interval `[-18.234071, -1.686089]`; even the best 2-bps-cost
estimate is negative. Patient entry is also unfavorable. Part 0 confirms that
the Experiment-57 intraday loss was dominated by spread plus fee drag
(`52.009366` bps/day) and roughly `6.492246` NAV of daily turnover, not by a
large negative gross signal.

The immutable root and its frozen-design, repaired-result, and log-inclusive
full-inventory-audit SHA-256 values are:

    /lambda/nfs/brazil-rv-east3/quant-data/b3/processed/model_runs/experiment58_swing_7f168eb_20260829T033013Z
    86b9370b1fc63cb4a05d5291a14b0af86efb218b6aa623989061a7a0e1550626
    f72a00024d3a55f73654425d7f7b6d285dd0a87004c6877e7d363b9a218fe95e
    55f0d565d385a2245880d02fc2f95d9da7d4e528eb9f96444492cd1443009eb0

The audit passed over 29 source/result/log artifacts and both access flags are
false. No model, execution, or deployment recipe changed. Any purpose-built
daily-native continuation must be separately preregistered; Experiment 58 does
not start one.

## Persistent Lambda retention cleanup round 5 (2026-09-03)

The canonical Lambda bucket retention state now includes a fifth audited
cleanup round. Exactly 29 files of at least 1 MiB, totaling 2,518,100,236 bytes
(2.345 GiB), were deleted from eight score-free failed experiment roots only.
Each deleted object was first streamed through SHA-256 and proven identical to
its matching object in a completed immutable authority root. All small failed-
root forensic records were retained. Completed roots and raw, interim,
feature, sidecar, and auxiliary data were untouched.

After adding the six immutable cleanup-audit files, the bucket contains 20,461
objects and 124,581,553,143 bytes (116.026 GiB), down from 20,484 objects and
127,099,629,508 bytes (118.371 GiB). The final audit found no planned survivor,
unplanned removal, size change, unexpected addition outside the audit root, or
missing authority. The durable audit root is:

    quant-data/b3/processed/model_runs/_retention/storage_cleanup_20260903_round5

Its cleanup-plan, delete-list, and cleanup-result SHA-256 values are
`7499fbfa01d0b4de5fa0068d537a1da15dbf7f5f5aa81878f4175b7f0b0a808e`,
`d6a1ca2156ac35c9f16a699d9e2962474eaa42450c178aefb2395a5e09e7a065`,
and `ccd1368d2f98a4c77fdc43d193a145680bdf83dd32f27e0bada464ccbc296e89`.
The final deterministic bucket path-and-size inventory SHA-256 is
`a0fa1167e4d731fdc91870d04b9f800fa6ff717d540baf1c709bf92619994b5df`.

## v2 development acceptance and Round-1 parent (2026-09-07)

The current v2 development store remains the V3 store at
`D:\quant-data\b3\processed\v2_daily_store_8021e42_20260906T202315Z`,
manifest SHA-256
`deb9ca8449c5b9a83bf25ac19218069c006e183361b6f6ba836717ade63b491b`.
Its source tier is explicitly development-grade:
`action_terms_source=inferred_cotahist_dismes_v1` and
`schedule_source=reconstructed_v1`. It does not support verified corporate-
action, auction-mark, or executable-borrow claims.

Pass 4h added a five-session hold-through rule for transient eligibility loss
and replaced the 1.8--2.2 gross stop with a labelled report inside the hard
1.5--2.25 interval. The accepted replay is
`D:\quant-data\b3\processed\model_runs\v2_development_acceptance_pass4h_384fd81_20260907T021757Z`,
pipeline-manifest SHA-256
`97a1ed757e87e1e19f2c46999437b31f436470f2dae49d2ce0c5782635607f8f`.
It passed with no reasons: all 16 sealed score panels were reused, every non-
ledger field was bit-identical, every D1--D5 signature was zero, all mean gross
values were inside the hard interval, all stale/unresolved means were below
2%, and protected access remained false/false. F2 inverse-volatility is
reported, not repaired, as `gross_underdeployed` at 1.725810.

Round 1 is completed and sealed at
`D:\quant-data\b3\processed\model_runs\v2_round1_81fe0cb_20260907T023339Z`.
Its result and complete inventory SHA-256 values are
`ca39f11340f13956dca11da7fb6b3a8fa0a38f8a79cb558387d81aff26bf57a0`
and `b5084c817fc478c2759d962af12e282c292cade6f8e715aebed290ba5500ce08`.
The access audit covers all 88 JSON artifacts with access flags and passes
official-validation/test false/false with clean transfer chronology. The
sealed store materializes lending and oddlot; options, rebalance, events, and
fundamentals remain explicitly source-missing and are never imputed or
fabricated.

The registered ladder keeps `a_slow` and `b_intraday`, drops `c_lending` and
`d_all_sidecars`, and designates `b_intraday` as the Round-2 GBDT parent. Its
pooled primary IC is 0.055264 with block-bootstrap interval
[0.027897, 0.087070], persistence is 0.8785/0.7974 at one/five sessions,
spread is 15.558 bps per holding session, and headline net excess is 5.430
bps/day over 375 finite development days. The inverse-volatility naive floor
has pooled IC 0.062256 and net excess 4.357 bps/day; it is a control, not an
eligible ladder parent. The data-span preview is informational: fine-only,
756-session decay, and uniform-pretrain ICs are
0.055264/0.053459/0.053680.

Round 2 was authorized and frozen at the exact Round-1 implementation commit
`cb6a0a5c4de202fe046dba48d9fd6c5af168dbfd` in:

    /lambda/nfs/brazil-rv-east3/quant-data/b3/processed/model_runs/v2_round2_cb6a0a5_20260907T143327Z

Its frozen-design SHA-256 is
`30b75d1b5bd126222ed4d318b8edccb242f84734deeb9222299b3dee55e65d8e`.
The required no-score Arm-A/F1/seed-11 smoke stopped at the first failure:
`torch.compile(fullgraph=True)` exceeded Dynamo's eight-graph recompile limit
when the dynamic active-name axis changed (last guard: 128 to 137 names). No
epoch history, checkpoint, completed training manifest, score, Stage-P run,
registered A/B/C trajectory, or R2 selection exists. Failure-record and
complete failed-root inventory SHA-256 values are
`d830cbb6a139b2c7c18b64a9adef6593095a89954cf3ba7aef5c4f9b2aaacf23`
and `f0c61411b729b9f3aeaff1cab4f133e14e02928994a0b1725bc598c0d0deeb07`.
Official validation, the permanently spent test, and deployment remain
untouched. The next Round-2 attempt requires a bounded dynamic-shape compiler
repair, tests, a new clean commit, and a fresh root; the failed smoke must not
be reused as a registered result. Exact paid instance
`c484c5fd446149f0a7994e9f6d4a0d32` was terminated after all failure evidence
was secured; provider inventories at `2026-09-07T14:57:53.2767470Z` and
`2026-09-07T14:57:58.6356783Z` both confirmed it absent, with no adjacent
instance present.

## Current v2 rev3 evaluation state (2026-09-07)

The characteristic-neutral rev3 evaluation contract is implemented over the
immutable V3 store. Commit `f567e0f6def33bfd49e5f02b4dfab953a94fb589`
repairs only the legacy diagnostic population: D1/D2/D3/D5 use the exact rev2
common active/score/sigma/all-primary-target-valid population, D10 remains
per-horizon, and the neutral path additionally intersects characteristic
validity. Training continues to consume only `target_primary_neutral`; the
legacy scaled target is readout-only. Executable borrow, realized-beta
diagnostics, fixed-width compiled batches, ledger rules, clocks, identities,
and the preregistration are unchanged. Ruff, compileall, and all 866 tests
passed before scoring.

Fresh acceptance is sealed at
`D:\quant-data\b3\processed\model_runs\v2_development_acceptance_rev3_f567e0f_20260907T173213Z`.
Pipeline-manifest/inventory SHA-256 values are
`dacef39764492213599fb9fb696b716583432511d1d8b8b1bfa239f824f1d7fc`
and `22ebbfaeec8306f039a7aadc42b72fd8d1c08f992c3fc7ac334b22ceead8830d`.
All 75 legacy comparisons pass exactly with zero mismatches;
inverse-volatility neutral IC is 0.0041771/0.0067263/0.0046084 in F1/F2/F3,
strictly below the 0.02 engineering bound. Ledger, chronology, native-fast,
entry-defect, and protected-access checks pass. The data tier remains
`development_grade_inferred_actions`; verified action terms, auction execution
marks, and historically executable borrow are still unsupported claims.

Rev3 Round 1 is completed and sealed at
`D:\quant-data\b3\processed\model_runs\v2_round1_rev3_f567e0f_20260907T173815Z`.
Frozen-design/result/access-audit/inventory SHA-256 values are
`6ad0d9b5de9fce20e3b5fa952b9fbfc48fffbf49a903f0bff3dc1bc073e37f16`,
`e51a46e1ddfed7730c178268c083da2f375ed92a660bbbeac4bb3f2619264daf`,
`a2336328dd0a7e85bce1c710724abf2e287806daad335e252bb2fafbde114708`,
and `32eb364d391319d833cd649e628c541e42c5953fc0b5d188029f377658cac35f`.
The 657-artifact audit passes with transfer chronology clean,
official-validation/test access false/false, and no deployment change.

The ladder keeps A/B/D, drops C, and designates `b_intraday` as the Round-2
parent. B has pooled neutral IC 0.0216976 [0.0139469, 0.0327263], exact legacy
IC 0.0319429 [0.0213486, 0.0484990], and headline net excess 1.3382 bps/day
[-6.3396, 12.3077]. C worsens both registered metrics versus B. D improves
over C on its registered F1/F2 support and is kept, but its pooled neutral IC
0.0136093 does not displace B. Parent-B realized beta is directional in F1/F3
(-0.3588/-0.5659) and beta-neutral only in F2 (-0.2450); F3 shortability is
coverage-limited at 0.1127 of active name-days. The informational fine-only,
756-decay, and uniform-pretrain neutral ICs are
0.0216976/0.0219637/0.0212408. The registration stopped before Round 2.
Protected data and deployment remain untouched, and no paid instance was used
for this CPU execution.

## Current v2 rev4 acceptance state (2026-09-08)

Rev4 constructed-book engineering is implemented at commit
`3242ad3910efcf748b9844ad0c5480727eacb886`. It adds a virtual nonlinear
neutral-target view over the immutable V3 store and a hash-bound BOVA11 hedge
ledger. Raw fixed-width COTAHIST establishes the canonical BOVA11 identity as
BDI14, not the proposed BDI02. The canonical hedge artifact is
`D:\quant-data\b3\interim\external\bova11_hedge_close_v1_2009_2024_20260908T115000Z`;
manifest/parquet SHA-256 values are
`858c6fb07e234d8c28219a72efa29775d83ba334f9b457ed24317ab7d270544d`
and `4aa998afb26558c9c2d1cbf9374a6a3881e04fd3a322b769d0c2d6950371a7e9`.

The sealed 16-book CPU acceptance root is
`D:\quant-data\b3\processed\model_runs\v2_development_acceptance_rev4_3242ad3_20260908T131920Z`;
manifest/inventory SHA-256 values are
`175e95ec91e38d546c2857f53c05274efb793e78accc7ae2c55125b375763102`
and `0f16f41c93a896c96e522bccc52f5252e71763e1156638dd7214fece78293d5b`.
It is unsupported with protected access false/false. The inverse-volatility
neutral-IC check passes, but 15 volatility-occupancy bounds and three F3 gross
bounds fail. The occupancy implementation incorrectly narrows candidates to
global top/bottom score halves before quintile balancing; the next registered
revision must rank each side within every volatility quintile across the full
eligible population. F3 remains executable-short coverage-limited.

The additional legacy-identity reason is a disclosed invocation error: the
acceptance was bound to the rev3 Round-1 result rather than the older canonical
legacy reference `v2_round1_81fe0cb_20260907T023339Z`. It was not rerun because
the independent construction bounds already stop the round. Round 1 was not
started; Round 2/GPU work was explicitly withheld, no deployment changed, and
no paid instance was launched.

For future lending coverage, 127 official B3 BDI Chapter 05 PDFs covering
2024-07-01 through 2024-12-30 are archived at
`C:\quant-data\b3\raw\b3\bdi_lending_open_balance\pdf_20240701_20241230`
with download-manifest SHA-256
`f1754e98ad2cd890375a6038907670a4552e96f8c3c232c3bc3f136d7fcce03b`.
They are not yet parsed and no lending sidecar or store was rebuilt.

## Current v2 rev4b acceptance state (2026-09-08)

Rev4b construction and direct-borrow engineering is frozen in commit
`a2d9f6d737fe8ebd5451aca1159ba73e223051e8`. The entry contract now ranks
within five causal volatility quintiles, retains names against their current
quintile, scales small strata, and spills only from unused same-side band
names. Direct borrow uses a separate D+1 lending archive and three registered
availability cells; `borrow_balance` is headline, `borrow_strict` requires a
recent observed trade, and `borrow_open` brackets unlimited supply. Rates are
the last observed taker rate plus 25 bps/year, with no equity floor and a
causal same-day 75th-percentile imputation when the prior-60 rate is absent.

The sealed lending archive is
`C:\quant-data\b3\interim\external\lending_archive_v2_2009_202412_a2d9f6d_20260908T125233Z`;
its manifest SHA-256 is
`5b4c83dd59796baf1cc4c5b2c441c1010b098966dbe72c80a7c7f025e1993e2b`.
Old/new overlap identity passes, the archive contains no protected-period
source row, and only 141 balances whose D+1 availability would fall after
2024-12-30 were excluded. F3 headline balance shortability is 0.7225 of active
name-days and is no longer coverage-limited.

The sealed 16-book CPU acceptance is
`D:\quant-data\b3\processed\model_runs\v2_development_acceptance_rev4b_a2d9f6d_20260908T131054Z`;
manifest/inventory SHA-256 values are
`23a1d513aa8981f40a28d2ea870d525b7c58aecf04ea57829de43739fed6edec`
and `a894e2c096a697f47611bb469ccaee47a3870d21065178c9562bc3a7640ebe33`.
All artifact hashes, chronology, legacy identity, occupancy, neutral-target,
beta, and protected-access checks pass. Acceptance is still `unsupported`
because all 16 books have equity-only mean gross below the fixed 1.50 floor
(range 1.2083--1.4197) and nonzero D4 cap-block signatures (range 8--165).
This is a scored stop, so it was not retried or repaired in place. No rev4b
Round 1 or Round 2 was started, no deployment changed, and no paid instance
was launched.

## Current v2 rev4c acceptance state (2026-09-08)

The canonical rev4c implementation is commit
`155c909f8c80c0e5047bd2e9141dbae33e36c91f`. Equity entries and D4 now use
equity-only gross/net/name limits. BOVA11 has a separate absolute 0.60-NAV
limit and reports capped targets, residual beta, whole-book gross, and
whole-book dollar net. All borrow cells use a separately flagged 2% rate
placeholder before the first causal published rate on 2023-07-11; the sealed
lending archive is unchanged.

The fresh 16-book CPU acceptance root is
`D:\quant-data\b3\processed\model_runs\v2_development_acceptance_rev4c_155c909_20260908T171132Z`;
manifest/final-inventory SHA-256 values are
`aab2330e1fc9ba91ad977619c56da50232c2f8a31ba8e989d87b2793886764d6`
and `99e726a2c621a4dc00284264ae952ce321f6d8051d449653644a0f0166843ee1`.
D1--D5, occupancy, neutral-IC, realized-beta, hedge, chronology, hash, and
protected-access checks pass. Acceptance is still `unsupported`: all 16
headline equity books miss the unchanged 1.50 mean-gross floor, ranging from
1.268828 to 1.498712. The dominant registered shortfall component is the
small-universe term, not hedge/cap interference, and D4 is zero throughout.
This scored stop was not relaxed or retried. Rev4c Round 1 and Round 2 were
not started, no deployment changed, and no paid instance was launched. Two
direct provider inventories returned zero nonterminal instances.

## Current v2 rev4d state (2026-09-08)

Rev4d is frozen in commit
`733ac1e729f98af964e022869a9dfaf617f93d3b`. Its only research change is the
small-stratum scaling threshold from four to two times `k_q + b_q`, the exact
condition needed for disjoint within-volatility-quintile long and short
bands. The proportional rule below that threshold is unchanged. All 899 tests
passed before scores.

The completed 16-book acceptance root is
`D:\quant-data\b3\processed\model_runs\v2_development_acceptance_rev4d_733ac1e_20260908T175620Z`;
manifest/final-inventory SHA-256 values are
`ef789052472044406d5771dcdf02c5f5590637f345f3bf9eaff26e215355c9e6`
and `f607751e8a998b003a8bdd284a0271b847395e54ba31d2bd103a5b26c03c4c86`.
All hard gates pass: D1--D5 are zero, all equity mean-gross values are within
1.5--2.25 (range 1.897--2.005), quota is six per quintile, `small_universe`
is zero, and occupancy, beta, hedge, chronology, hashes, and protected-access
checks pass. The source tier remains explicitly development-grade.

Rev4d Round 1 is completed and sealed at
`D:\quant-data\b3\processed\model_runs\v2_round1_rev4d_733ac1e_20260908T180257Z`.
Frozen-design/result/access-audit/inventory SHA-256 values are
`0e40ead5105cfaf16272f03a6e7272457771ec2d2931d088be0998ab92f6bb21`,
`e782233bd1c1410fc06346ec8590433b6e0743210c970eebdcf325a2505f712c`,
`102d0631e479d55a003c01e9b4d30bb4cf84875ae83840c098118659f99afa67`,
and `fe9b466e6c1e9a38271cce03615b17bef01c48df66395dba462ac7af6c66ffd3`.
`b_intraday` is the sole eligible GBDT rung and designated parent, with pooled
neutral IC .019991 [.011608,.029269] and headline balance net excess -5.604
bps/day [-9.727,1.852]. The result is not a deployment claim. Round 2 remains
unrun pending explicit authorization; official validation, the permanently
spent test, and deployment remain untouched. Two provider reads confirmed
zero nonterminal instances.

## Current v2 rev4e state (2026-09-08)

Rev4e changes only the constructed-book ledger: equity and BOVA11 short-
proceeds collateral earn CDI in the headline, while equity borrower fees are
20% of the contract rate with a 2.5-bps annual floor and 70-bps cap. The
otherwise identical zero-remuneration comparator remains registered. Scores,
models, construction, targets, clocks, folds, and candidate rules are
unchanged. The final replay implementation is commit
`962b9e0c141542bc141fb0df330ae7729178f5f0`.

The 18-panel ledger replay is sealed at
`D:\quant-data\b3\processed\model_runs\v2_round1_rev4e_962b9e0_20260908T192330Z`.
Frozen-design/result/access-audit/inventory/economics-detail SHA-256 values are
`8cc6d14f658af339c49f7a7cd419dabae6a386ef87a5a5902b0e7a14300a3491`,
`19efdaf338e584ae84269c171c34a9a33013fce6d40ccde7cc8b59bb3f199f7a`,
`e89c12934621e897c6884453f3f518fc5daac8ede1d1fccbe78bdd6bbefc7916`,
`2860a82ec90d62bb2083677bd9289bf5eb7c907a441980efd5fa485c4e734261`,
and `74eda616fea82c5d08438be943b3fd82c0c80476c06a5966a78b13498be09b03`.
All non-ledger fields match exactly, no score/model was recomputed, protected
access is false/false, chronology is clean, and deployment is unchanged.
Pooled headline net excess is +1.665 bps/day for momentum and -1.007 for
`b_intraday`, with both intervals spanning zero. `b_intraday` remains the
Round-2 parent. This replay was the input to the subsequently completed
Round 2 below.

Rev4e Round 2 is now completed and sealed at
`/lambda/nfs/brazil-rv-east3/quant-data/b3/processed/model_runs/v2_round2_rev4e_2b40b24_20260908T202100Z`.
The frozen-design/result/access-audit/inventory SHA-256 values are
`0e5e7fa43e221d4ff51bd5ba353dfdad021540c8531fe86738b71bfe8e7ae788`,
`c2549977e94ad2ca00a88642cfc61ab3b1ba98cadd5cb5b9bb2780788cb9ede4`,
`6b5b821ad7d583137b9f9c93754e794d72641f3a564641cedc0ed60361416224`,
and `7d32efb19a539c387c4085e25e660afa748dc230c09760457d1a7e6b850790cc`.
Exactly three Stage-P trajectories and 18 main A/B trajectories completed.
Arm B was selected with pooled neutral IC .024777 and headline net excess
+2.404 bps/day [-.521,9.039]. The equal-rank Arm-B-network/`b_intraday`-GBDT
ensemble is the Round-2 research designation with pooled neutral IC .026531
and headline net excess +1.226 bps/day [-1.691,8.364]. No economics override
fired. All artifact, chronology, and access audits pass; official validation
and the spent test remain untouched, and deployment did not change. Round 3
and 2025 access remain forbidden.

## Research checkpoint (2026-09-09)

The current forward research contract is
`research/preregistrations/v2_research_checkpoint.md` and its accompanying JSON.
It supersedes the old three-fold defaults above: fourteen half-year evaluation
folds cover 2018H1–2024H2, with expanding fits from 2016-07-18, and Stage P ends
2016-06-30. Primary IC and Stage-F checkpoint selection use common D3/D5/D10
neutral-target support; `legacy_primary_ic_1235` preserves the former headline.
The paired parent is the Round-3 fast-off configuration; B6 remains historical.
Screening seeds are 11/29/47. The user-authorized
[Round-4 A4 budget amendment](research/preregistrations/v2_round4_budget_amendment.md)
replaces the unexecuted mandatory 61/79/97 extension with fixed saved-score seed
sensitivity and explicitly provisional development conclusions. It preserves all
fourteen folds and training/evaluation settings. Subsequent routine research uses
three matched seeds; extra seeds require a separately registered decision. The
selected execution policy remains theta=1, D3/D5/D10 equal-notional,
buffer nine, with its in-sample selection label. The repaired development store
and all sealed Round-3 artifacts remain immutable. CPU re-baselining precedes
Round 4; paid compute requires a renewed go, and 2025/2026 access is unauthorized.

The fourteen-fold CPU checkpoint is now accepted and sealed: 70 controls and
56 GBDT cells (1,400 individual head/seed fits), with all unchanged gates passed.
Inventory SHA-256 is
`36aa5f132828aacf700ce728f1d202942c5b11ca9f4b8f9be04df9e8e59da917`;
the root and full readouts are recorded in
[the checkpoint report](docs/v2_RESEARCH_CHECKPOINT.md). Economics retain the
existing resolved-fold pooling rule and explicit candidate-specific exclusions;
engineering acceptance is not a full-calendar or implementability claim.
[Round 4](research/preregistrations/v2_round4.md) is registered as fast_off plus
S0/H/P/L/C, with twelve screening P runs, 252 main F runs and nine selection-only
F runs. A4 permits a working research parent only when its original provisional
choice is unchanged and eligible across the full three-seed panel and all three
leave-one-out panels; otherwise fast_off remains the comparator with an inconclusive
label. This is not six-seed confirmation. Gabriel subsequently
authorized paid Round-4 sessions through A1–A3 after revision and a fresh freeze.
The amended books settle unpriced terminal equity/hedge residuals at their last mark,
retain the uncertainty and 30% haircut labels, and pool full-calendar economics.
Resolved-fold economics remain secondary continuity evidence. S0's tie decision uses
the validity-derived F8–F14 subset; L's informative subset is F9–F14. Every arm receives
momentum-residual diagnostics. The BOVA11 series' last F4 mark is 2019-08-16, 92 sessions
before its end; terminal settlement does not repair those missing hedge returns.
The old CPU seal remains immutable. The proposed 2025-read bar remains unapproved.

Round 4 is now complete under A4/A4.1 as a fixed three-seed development result. The working research parent is `S0`; the research designation is `S0`. P is disqualified across all choice panels after one two-seed F14 book exceeded the unchanged occupancy bound. Original full screening and every fixed seed omission remain reported. No further neural fits were made; all roots are recovered and verified. See [Round 4](docs/v2_ROUND4.md) for exact comparisons, amendments and limits. No new round, 2025/2026 consumer access or deployment is authorized.
