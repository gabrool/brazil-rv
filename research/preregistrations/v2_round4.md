# Round 4: first neural research round on the fourteen-fold checkpoint

**Current budget amendment (2026-09-10):**
[A4](v2_round4_budget_amendment.md) supersedes the unexecuted mandatory six-seed
confirmation stage with fixed three-seed development reporting and saved-score
leave-one-seed-out sensitivity. The original A1–A3 execution below remains the
historical contract of the frozen screening root; its completed artifacts are not
rewritten. No six-seed confirmation or independent replication is claimed under A4.

Registered after the completed CPU report in commit `bd8ce6b`, before any Round-4
neural fit or score. This implements Gabriel's `v2_research_checkpoint.md` and the
[accepted checkpoint contract](v2_research_checkpoint.md). The CPU results are
diagnostics; they do not change the arm roster, paired parent or promotion rule.
Revision A1–A3 incorporates Gabriel's `v2_round4_go_amendments.md`. Paid compute is
authorized after this revision is committed and a fresh execution root is frozen
from it on the compute host. No arm, seed, training setting or execution policy
changes. No 2025/2026 consumer access, deployment change or later research round is granted.

## Accepted inputs and information set

The accepted CPU root is
`D:/quant-data/b3/processed/model_runs/v2_research_checkpoint_d2f7de4_20260909`.
Its CPU result SHA-256 is
`6418e2f156d4ed899d897e5efa4e170f24044f53163dbda5cf503dfa281a091b`;
diagnostics SHA-256 is
`e561cc477d934d31b37a0887c71bea8e30029c6a9f4f761c8c439706af084bb3`;
final inventory SHA-256 is
`36aa5f132828aacf700ce728f1d202942c5b11ca9f4b8f9be04df9e8e59da917`.
All 70 controls and 56 GBDT fold cells passed the unchanged registered engineering
gates. The 2,956-file, 2,460,356,168-byte root is sealed and verified on the host.
See [CPU report](../../docs/v2_RESEARCH_CHECKPOINT.md) and
[acceptance evidence](../../docs/v2_checkpoint_cpu_evidence.json).

Freeze a fresh Round-4 root from a clean commit after this registration. Bind the CPU
inventory/result, repaired store manifest, exact source hashes, execution policy,
feature names, model contracts, protocol payload, this registration's bytes and code
commit. Verify the CPU inventory and all 126 accepted cell markers before freezing.
Resolve external roots through the existing path mapper, verifying relocated hashes;
never infer new data roots from an obsolete document. The operational `.txt` handoff
provided by Gabriel remains authoritative for Lambda launch and storage procedures.

The repaired daily store ends 2024-12-30 (manifest
`db4f751d47133a6611733739ad9bfc15721452218365e338fec61a6b608bea64`).
Sources and exact fold dates are in the [checkpoint JSON](v2_research_checkpoint.json).
There are fourteen half-year evaluation folds, 2018H1–2024H2, with 1,738 sessions.
Every expanding fit starts 2016-07-18, then ten purge sessions, 55 selection sessions,
ten purge sessions and evaluation. F12/F13/F14 preserve the former Round-3 selection,
purge and evaluation dates, with earlier fit starts. Stage P covers 2010-01-04 through
2016-06-30, including its internal selection and 70-session embargo: 1,607 sessions
and approximately 6.5 years in total. Eleven exchange sessions separate P and F.

The decision remains 15:45. No entry bar, future price, revised identity, or future
availability enters a feature. Inputs, validity, ages and point-in-time membership
retain the accepted store's causal definitions. No scaler or threshold is fit on
selection/evaluation outcomes. Historical integrity-only scans of later old-store rows
remain disclosed; no new 2025/2026 payload reaches a consumer in this round.

## Parent and five independent arms

All arms are built on the Round-3 `fast_off` graph: `disable_fast_stream=True`.
The parent retains the twenty current intraday scalars with existing validity/age
handling and its current projection; it does not read native M1 arrays. Historical
fast-on B6 remains sealed and is not refit. No intraday reranking, risk overlay or
execution modification is part of any arm.

| Configuration | Only change versus fast_off | Stage-P graph |
| --- | --- | --- |
| fast_off | Paired parent, re-baselined on all fourteen folds | Fresh parent graph |
| S0 | Remove current intraday values, masks, ages and support, `fast_present`, current projection/normalization and unused fast parameters; no current/native-fast reads | Fresh S0 graph |
| H | Stage-F head weights `(0.25, 0.25, 1, 1, 1)/3.5`, in D1/D2/D3/D5/D10 order | Reuse exact same-seed parent P |
| P | Stage-F `lambda_persistence=0.1` | Reuse exact same-seed parent P |
| L | Add the existing five lending features to the slow path, with canonical validity/age and slow transforms | Fresh L graph |
| C | Concatenate the three stored causal common-state values before the fusion projection/trunk | Fresh C graph |

L fields, in store order: `loan_balance_to_volume_20`, `loan_balance_change_1`,
`loan_balance_change_5`, `loan_rate`, `loan_rate_change_5`. Rates remain decimal.
The all-invalid-sidecar invariant preserves all shared parent batch fields bit-for-bit,
including targets and masks; extra sidecar channels remain invalid. It does not assert
identical outputs from networks with different input widths. Lending balances start
in March 2022, and observed rate availability starts 2023-07-11. Earlier lending
availability/pricing uses existing labelled placeholders. No historical locate is implied.

C fields, in store order: `recent_market_log_return`, `median_raw_daily_volatility`,
`raw_cross_sectional_return_dispersion`. Consume the decision row exactly as stored;
zero invalid values, preserve their provenance, and do not add an evaluation-fitted
normalizer or recompute them from future rows.

S0's changed fusion width and L/C's changed inputs require fresh graph-specific P;
old checkpoints are not forced into those graphs. Every P trajectory keeps uniform
loss, `lambda_persistence=0`, and the prior D1/D2/D3/D5 internal selection metric.
H and P change fine-tuning only. P is trained once per seed/graph, never per fold.
All compatible non-fast parameters use the existing P-to-F transfer rule and learning
rate treatment. Record consumed feature names, checkpoint hashes and transfer audit.

## Fixed training, selection and execution

The [protocol JSON](v2_research_checkpoint.json) fixes the unchanged training settings:
60-session lookback, one GRU layer, hidden width 64, fusion width 128, two trunk blocks,
dropout 0.1, eight date pairs per batch, maximum 20 epochs, patience three, learning
rate 0.0003, transferred-parameter multiplier 0.3, SAM rho 0.125, weight decay 0.01,
EMA decay 0.995, and no time decay or BF16. Raw Patience checkpoints supply evaluation
scores. No optimizer, SAM, transfer-scope, learning-rate or architecture search is added.
Full model payloads in the frozen root disambiguate every parameter.

Stage F selects on the equal daily mean of D3/D5/D10 neutral-target Spearman IC,
with one common active/valid/finite/supported name population across all three heads,
at least twenty names, and three defined correlations. D1/D2 do not restrict that
population. The default uniform head loss uses the original `.mean()` operation
bit-for-bit for value and gradient; only H changes that operation. The former headline
remains `legacy_primary_ic_1235`. Undefined sessions remain on the calendar as missing.
All historical per-head, spread and persistence definitions remain.

The selected execution policy is theta=1, D3/D5/D10, equal notional, buffer nine per
volatility quintile. Carry `execution_parameter_selected_in_sample=true`; theta=1
retains the current missing-score behavior. Use the same ledger, close-proxy fills,
borrow/collateral, costs and risk limits as the accepted CPU root, with A1's terminal
accounting amendment below.
No execution sweep or blend candidate is authorized.

## Stages and exact counts

Screening seeds are 11/29/47. Confirmation seeds are 61/79/97; every comparison that
can promote or change the default must use both candidates' same six-seed panel.
Within each arm/fold, form the existing equal-rank ensemble with identical support.
Pair candidate metrics on their exact common population and common defined economics
sessions. Use the registered 20-session, fold-preserving moving-block bootstrap with
10,000 replications and seed 20260815; no block crosses a fold boundary.

| Stage | Runs | Dependency |
| --- | ---: | --- |
| Serial one-epoch smoke | 6 | One per configuration; seed 11; F1 except L uses F14 so lending is exercised |
| Screening P | 12 | Parent/S0/L/C times three seeds, after all smokes pass |
| Parent F | 42 | Fourteen folds times three seeds, exact same-seed parent P |
| Arm F | 210 | Five arms times fourteen folds times three seeds; after all parent books pass |
| Selection-only diagnostic F | 9 | Parent on F12/F13/F14 with old D1/D2/D3/D5 selection; same new parent P |
| Confirmation P | 3 per needed graph | New seeds; H/P share parent P |
| Confirmation F | 42 per confirmed configuration | Fourteen folds times three new seeds |

The screening fit budget is **12 P + 252 F + nine selection-only F**, plus six
one-epoch smokes. The CPU checkpoint used **1,400 individual head/seed fits**, not
1,400 five-head ensembles. The supplied three-to-nine P estimate and fixed seven-to-nine
GPU hours / US$30 estimate are not valid commitments. Measure runtime after smokes/P
before projecting the full session. After serial smokes, use up to six independent
trajectories concurrently when memory and utilization permit; retain first-failure stop.
Each accepted trajectory must have exactly one compiled training and one selection graph.
Smokes produce no reusable candidate scores or reusable pretraining checkpoint.

Accept the complete fourteen-fold parent book before starting arm fits. Screen all five
arms and retain losing arms at the same detail. Before confirmation scores, freeze its
explicit roster: always fast_off and S0, plus the eligible screening IC leader and any
eligible arm qualifying for the registered positive-economics-interval override against
that leader. If no extra arm qualifies, the minimum confirmation is six P and 84 F;
confirming all six configurations would add twelve P and 252 F. A configuration outside
that roster cannot be promoted without its own matched six-seed confirmation. These
counts preserve mandatory S0 confirmation even when another arm leads screening.

The existing staged planner implements smoke, P, parent, arms, selection, confirmation-P
and confirmation-F. The A1–A3 go authorizes the staged session after fresh freeze. Later
plans bind actual accepted checkpoint hashes and should not be fabricated in advance.
Do not start Round 5 or re-fit H/P/L/C on S0 within this round. If S0 becomes the parent,
any transferred finding is a Round-5 hypothesis requiring 42 screening fits on its
fourteen folds, not the obsolete nine fits.

## Diagnostics and reporting

The CPU four-rung panel and active-row TreeSHAP are complete and sealed. They have zero
selection weight. [Published diagnostics](../../docs/v2_checkpoint_cpu_diagnostics.json)
include all candidates, folds, former three windows, primary/legacy/per-horizon IC,
spread, persistence, turnover, economics, lending-coverage subsets, all six GBDT pairs
and identical per-fold TreeSHAP sample indices. No GBDT replaces the neural parent.

After neural scoring, report every arm's and fast_off's daily cross-sectional correlation with
momentum_12_1 (mean, sample SD, by fold), per-day rank-OLS residual IC, and the share of
the unchanged four-head decile spread attributable to extreme momentum quintiles.
Use the current diagnostic implementation and preserve unknown momentum attribution.
This is score-spread attribution, not constructed-book P&L. Show the sealed B6 results
beside it. The nine old-selection fits isolate Stage-F selection using the same new P,
graph and fit dates; they are not a refit of B6 and cannot become a candidate.
Show momentum's primary IC beside every candidate per fold, and all per-fold paired
deltas. These diagnostics distinguish additional momentum exposure from residual
ranking information; they have zero selection weight and do not create a new gate.

Report settled economics on the full common calendar, with unresolved labels retained.
The former resolved-fold-only figures remain a secondary continuity readout, as
specified in A1 below. The eligibility rule uses amended full-calendar economics.
Report the observed-rate era and the strict held-short subset with no imputed or
placeholder opening short notional. None establishes capacity or locates.

Turnover includes fold-boundary trades. The unchanged non-circular block bootstrap
underweights boundary spikes, so its interval need not contain the whole-panel mean;
disclose this limitation without changing the estimator during this round. Persistence
and turnover are readouts, not new gates. Report all fourteen folds and the former
Round-3 windows; no claim that uncertainty must shrink by a fixed square-root factor.

## Promotion and default decision

Screening can identify provisional choices only. On the confirmed six-seed panel,
negative or undefined constructed-book economics makes a configuration ineligible.
Among eligible candidates, primary IC ranks first. A candidate may override the IC
leader only when its paired primary-IC interval includes zero and its paired economics
interval is strictly positive. If several qualify, use their pooled economics point
estimate to choose the override. Record the full eligibility/leader/override trace.

Separately, S0 becomes the next-round parent unless its paired primary-IC upper 95%
bound versus fast_off on its predeclared informative subset is below zero, subject to the same nonnegative-economics
eligibility and six-seed confirmation. An undefined comparison cannot establish a tie.
If S0 fails eligibility or this comparison, retain fast_off and state the reason and
the retained parent's eligibility. Retaining a research comparator does not make an
ineligible model deployable. Report both the round's designation and the next-round
parent when they differ. Preserve raw M1, v1 and native-fast code; remove superseded
v2 defaults only after a new default is actually accepted.

## Stops, paid session and later work

On any baseline-book D1–D5 flag, equity mean gross outside 1.5–2.25, stale inventory
at least .02, occupancy above two, insolvency or other existing hard-bound failure,
preserve the exact result and stop. Also stop on a failed smoke/compile contract,
off-contract targets/transfer, a replay changing a protected non-ledger field, any
2025/2026 consumer row, an unauthorized paid instance, or a store/build exceeding
the measured 8-GiB invariant. Do not waive a gate. A proven defect or regime-wrong
gate needs a recorded amendment and affected cells in a fresh root. Gabriel's standing
authority permits recommended implementation resolutions. Paid sessions are now
authorized by A1–A3; the holdout read remains unauthorized.

After amended freeze: smoke and P while the host runs the CPU settlement replay;
parent fits and acceptance after the replay passes; arms/selection, evaluation,
confirmation as required, then seal. Every paid session must end with the sealed root
copied to the host, full hash verification, instance termination, and two provider
inventory reads. Never terminate before the verified host copy exists. Preserve partial
sessions and stop evidence; no scored-cell overwrite. Publish `docs/v2_ROUND4.md` with
the parent re-baseline, every arm, paired deltas, diagnostic panels, GBDT context,
S0 decision, promotion trace and read-bar status, then stop for the next research stage.

## A1: terminal residual settlement and common-calendar economics

Source diagnosis: the hash-bound BOVA11 series has no 2019-12-30 close. The existing
terminal intention already requests a complete hedge close, but without a print it
expires and leaves the hedge open. The last observed mark is 96.15 on 2019-08-16,
92 sessions before F4's end (only 34 of F4's 126 sessions have observed BOVA11 marks).
This is a material source gap, not merely a boundary sequencing artifact. A1 closes
the accounting residual; it cannot recover missing hedge returns. Preserve that
limitation and the bound source without inventing prices. The amendment closes a terminal hedge at the
terminal close when present, otherwise at its last known mark as labelled accounting
settlement. Ordinary missing-print hedge rebalances still do not fill.

Equities without a terminal print settle at their last mark at the evaluation boundary,
even if the ten-session grace has not elapsed. The ordinary within-fold convention
remains `last_mark_after_10_sessions`; the additional boundary rule is explicitly
`last_mark_after_10_sessions_with_evaluation_end_acceleration`. It does not synthesize
a print, alter historical prices or add future knowledge to a decision. Freeze conditional
settlement intentions before reading the terminal close; an actual print cancels them.
Keep the 30% adverse settlement haircut scenario, including a hedge settled without
a print, and the 15%-NAV aggregate settlement label. Record equity residual notional
before boundary settlement, the hedge's last-mark settlement notional, the haircut NAV
delta, and actual remaining shares. `terminal_unresolved_inventory_fraction_nav` reports
remaining equity plus the unpriced equity settled at the boundary. `economics_unresolved`
remains true for these uncertain residuals and other existing reasons; it no longer
excludes a fold from amended economics. Accounting validity and hard bounds still gate.

Use `settle_terminal_residuals=True` for every amended book, including sensitivities
and D5 diagnostics. The historical setting remains available only for unchanged prior
registrations and the explicitly requested continuity comparison; no sealed output is
rewritten. Pool primary and paired economics over all 1,738 sessions. A candidate's
`resolved_fold_only_net_excess_bps` is secondary and never the promotion basis.

Replay all 126 sealed CPU panels in a fresh root without refitting scores or models.
Only `economics`, the four ledger-derived diagnostic paths `exposure_daily`,
`exposure_summary`, `realized_beta`, `realized_beta_bova11`, and the four ledger mask
counts `stale_mark_name_days`, `unresolved_action_name_days`, `valuation_scenario_count`,
`actual_risk_breach_dates` may change. All input hashes, outcomes, support, ICs, score
spreads/persistence and other fields must remain bit-identical. Record source and
destination hashes, gate each replayed book, and assert exact continuity of the sealed
resolved-fold pooled and former-three-window economics. The CPU replay may run in
parallel with GPU smokes and P, but parent acceptance requires its completed bound result.

A continuous-path walk-forward, carrying the book across fold boundaries while only
the model changes, is registered as future overlay-replay or Round-5 work. It is not
implemented or evaluated in this round.

## A2: informative fold subsets fixed before scores

The [Round-4 protocol JSON](v2_round4.json) binds counts and consumed-presence hashes
computed only from the repaired store's validity arrays, active membership and causal
history. S0 uses any current intraday validity at t among active names. L uses any
lending validity on a valid slow timestep in t-60 through t-1 for an active decision-time
name, exactly the supplied slow history. H, P, C and the parent use all fourteen folds.

Measured subsets: **S0 F8–F14**, first supported decision 2021-07-19; **L F9–F14**,
first supported decision 2022-03-23. F7 has no valid current intraday input, correcting
the supplied expected F7–F14 range. Each selected fold remains whole; no names/days are
selected retrospectively. Registration and freeze recheck the source manifest and the
same validity-derived payload before any neural score. Use these same subsets during
screening and confirmation, with all-fold deltas retained beside them.

For every pair report the subset corresponding to each member's distinguishing
information beside the all-fold comparison, for every existing paired metric. S0's
simpler-wins-ties bound uses its F8–F14 subset. IC designation and the economics override
continue to use the all-fold rules, and nonnegative economics uses the full calendar.
The same-arm non-informative comparisons can still differ through initialization and
graph effects; zero input coverage is not a promise of equal neural predictions.

## A3 and session budget

Momentum correlation, momentum-residual primary IC and extreme-momentum decile-spread
share are required for every arm at screening and confirmation, pooled and by fold.
Report B6's sealed diagnostics as historical context only. No correlation or residual
threshold is introduced, and no arm is selected by these readouts.

Measure smokes and Stage P before projecting wall time; expanding F14 fits span much
more history than Round 3. Split at useful stage boundaries into sealed, recoverable
sessions when needed. Every paid session ends with verified host recovery, exact-ID
termination and two provider reads. The confirmation roster remains fast_off, S0, the
eligible screening IC leader and every eligible override qualifier. Subsequent sessions
retain the same frozen registration, seeds, settings and hash-bound checkpoint inputs.

The **2025-read bar is unapproved and remains Gabriel's decision**. The proposal is:
confirmed primary-IC lower 95% bound above momentum's point estimate; persistence-1
at least .9 or turnover/cost robustness leaving positive excess under doubled costs;
paired net excess over momentum with lower bound above zero; and the prior Round-3
conditions. This is not an active gate or authorization. Register one candidate's
single-read protocol and any development-to-2025 store extension only after a bar is
approved and met. The historical spent test remains unavailable for candidate selection.

Meaningful-capital execution assumptions precede implementability claims or the read;
contractual event/calendar and prospective execution evidence precede deployment.
The separate intraday risk track starts only after the Round-4 GPU launch and requires
its own registration and coverage/replay prerequisites. It may not alter forecasts.
Inferred event-day action flags are retrospective annotations, never intraday
foreknowledge. Calibrate lower-exposure comparators on fit/selection history and freeze
before evaluation; ex-post matched-gross attribution is descriptive only. The proposed
overlay CVaR/cost thresholds also remain unapproved. No overlay implementation, broad
framework, new architecture search or additional research round is part of this handoff.
