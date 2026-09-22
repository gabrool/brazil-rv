# Brazil-RV: local coding-model handoff

Verified **2026-09-21 22:42:57 America/Sao_Paulo** (2026-09-22 01:42:57 UTC).
This is a dated snapshot, not a launcher. Refresh processes, progress files and
Git state before acting. **Work is already running; do not start duplicate jobs.**

This replaces the obsolete August TCN/cloud handoff, preserved in Git at
`b5e937613da41afe762710b340b07fdd9e03c70e:RESEARCH_HANDOFF.md`. Its old cloud
commands, data, architecture and validation boundaries do not apply now.

## 1. Where we are and what comes next

This is offline Brazilian equity relative-value research, using permanent
security identities, causal historical inputs and economic portfolio evaluation.
It is not a production trading system. The immediate goal is to investigate why
wider attention lost its advantage after the data/refit changes, compare models
fairly on common inputs/accounts/periods, then pursue justified improvements.

The user authorized four steps:

1. **Complete:** propagate the specifically evidenced added-period repairs
   (GUAR/QGEP actions, QGEP/GPC/Wiz histories and distribution targets) into a
   separately accepted complete store. Earlier stores remain immutable.
2. **Running:** attention widths 64/96 × parent stopping patience 5/20, seeds 11/29/47,
   eight fixed development periods. Six patience20 parent trajectories supply
   twelve labelled selections. Each patience5 view sees only its five-stale
   prefix. Identical selected parents permit exact child reuse. There are at most
   96 children and 102 logical jobs including parents, not necessarily 102 fits.
3. **Frozen and queued:** 54 fresh C6/GRU parent/child fits on the same store and
   periods, then common neutral 5% and flexible 45% net-exposure comparisons, with
   beta cap 5% fixed. The existing continuation launches this after attention
   verification. Preserve each model's original distinct recipe.
4. **Not started under this continuation:** choose bounded architecture contrasts
   from the common evidence, including the authorized LSTM investigation where
   justified. The continuation stops for this review; it does not launch an
   unlimited capacity grid.

Foundation, bounded Stage A accounting admission, the previous accepted data
stores, four-period Stage C comparisons, and registered width/depth screens are
already complete/disposed. **Do not restart them.** Historical heartbeat messages
saying “Stage A incomplete / C and D unstarted” are superseded.

### Current best: keep comparison scopes separate

There is **no established global winner on one common corrected continuous
comparison**. These are different evidence sets, not one leaderboard:

| Evidence set | Result | Meaning |
|---|---|---|
| Historical C6, flexible 45% net cap, 1,738 continuous sessions | 6.70751 bps/day above CDI; zero-rate Sharpe 1.88147 | Historical reference; corrected continuous counterpart is unrun |
| Eight-period attention baseline plus separate account/source overlays, neutral 5% net cap | Attention 64: 3.01618 vs attention 96: 1.92001 bps/day above CDI | Working return baseline; independent period accounts, not one continuous account |
| Earlier corrected four-period capacity screen | Two-layer GRU: 1.79771 bps/day above CDI | Positive mean, only 1/3 seeds and 2/4 periods improved; not adopted |

The earlier attention 128/GRU 96 width screens and depth candidates did not pass
their registered gates. The original compiler-corrupted GRU96 results are excluded;
qualified replacement results exist. These findings do not prove scaling cannot
work. No new capacity model has been adopted.

Earlier parent stopping is a demonstrated **partial mechanism** for the attention
reversal: the seed29/F10 patience probe improved seed/ensemble net return by
7.04780/2.36412 bps/day, but remained negative above CDI. That outcome-informed
diagnostic is not an adopted general stopping rule. A real tiny hedge-trade/minimum
fee defect was fixed, but its measured effects were too small to explain the main
reversal. The current matched stopping experiment is the broader test.

Training already uses the full permitted chronological past. Four evaluation
periods never meant training on only four short windows. Current evaluation folds:
**F2/F3/F6/F7/F10/F11/F13/F14**. Reserve **F1/F4/F5/F8/F9/F12** from new corrected
comparisons unless a subsequent explicit decision changes this boundary. They
were used historically: do not call them untouched tests. **No 2025/2026 consumer
reads.** Never splice disjoint periods into a supposedly continuous account.
A fully corrected 1,738-session reference would require currently reserved periods.

## 2. What is running

At the snapshot: **26 completed logical jobs / 24 actual fits**, including all six
parents. Last completed: `TE_wide_p20/F/F3/29`; active: `TE_full_p5/F/F3/47`.
The continuation reported four complete three-seed forecast groups. Refresh these
numbers from disk; they are not a persistent completion claim.

| Job | Pinned checkout and commit | Actual Python PID at snapshot |
|---|---|---:|
| GPU attention | `C:/Brazil-RV/.worktrees/attention-stopping-resume`, `b49b19ff81c1119698379e739d29d4390f2b3a71` | 17812 |
| Finite evaluation/next-stage continuation | `C:/Brazil-RV/.worktrees/scaling-evaluation`, `fdd354a9e5cf4a32e042b5a2aca3e18a87493f21` | 24568 |

GPU entry point: `ops/run_matched_stopping.py` in its pinned checkout.
Continuation entry point:
`C:/quant-data/b3/processed/model_runs/v2_attention_stopping_20260921/continue_comparison.py`.

Windows uv launches an environment-Python wrapper and the actual interpreter.
Three related OS processes can represent **one job**. Inspect full command lines
and parent IDs; never kill or restart based on a stale PID or log alone.

The continuation evaluates completed attention groups on CPU, waits for the GPU
worker to exit, eagerly verifies every distinct new attention checkpoint on its
own evaluation dates, qualifies books, and summarizes results. It then runs the
54 C6/GRU fits sequentially and evaluates neutral/flexible policies. It stops on
errors or at `comparisons_saved_pending_exposure_review_and_capacity_decision`.

It is **one finite offline job**, not a recurring automation. The user disabled
the one-minute heartbeat. Do not recreate it, repeatedly restart chats, or pause
for routine permission questions. Inspect healthy workers briefly and use time
for necessary independent work. Notify meaningful findings, failures or decisions.

The initial attention driver stopped because JSON sorting put the `p20` alias
before its `p5` source fit. `ordered_arms()` now orders reference/patience explicitly;
four tests passed. All six parents and two completed children were reused. Resolve
`scaling_matched_stopping_resume` for exact failure/resume evidence. No learning
rule or model equation changed. The active log is **training_resume.log**, not
the original failed training.log.

### Safe status commands (read-only)

```powershell
$repo = 'C:/Brazil-RV/quant/b3-quant'
Set-Location $repo
$run = Get-Content -LiteralPath "$repo/docs/v2_economic_data_scaling_run.json" -Raw | ConvertFrom-Json
$plan = Get-Content -LiteralPath $run.scaling_matched_stopping_plan.path -Raw | ConvertFrom-Json
$attentionRoot = $plan.root
Get-CimInstance Win32_Process |
    Where-Object { $_.Name -match '^(python|uv)\.exe$' } |
    Select-Object ProcessId,ParentProcessId,Name,CommandLine
nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv
$progress = Get-Content -LiteralPath "$attentionRoot/refits.json" -Raw | ConvertFrom-Json
$progress | Select-Object status,actual_fits,logical_jobs
$progress.completed.Count
$progress.completed | Select-Object -Last 2
Get-Content -LiteralPath "$attentionRoot/training_resume.log" -Tail 8
Get-Content -LiteralPath "$attentionRoot/continuation_progress.json"
Get-Content -LiteralPath "$attentionRoot/continuation.log" -Tail 8
```

## 3. Relevant locations and authority

Repository-relative paths below are under **C:/Brazil-RV/quant/b3-quant**.
Its physical location is **C:/quant/b3-quant**: these are the same repository,
not copies. `C:/Brazil-RV/quant-data` points to `C:/quant-data`, and
`C:/Brazil-RV/Trading` points to `C:/Trading`.

| Path | Purpose |
|---|---|
| `C:/Brazil-RV/AGENTS.md` | Workspace engineering/data/leakage rules |
| `PROJECT_CONTEXT.md` | Canonical startup contract; read before material changes |
| `research/preregistrations/v2_economic_data_scaling.md` | Authorized experiments and gates; reread after each stage/GPU wave |
| `docs/v2_economic_data_scaling_run.json` | Canonical path/hash index; resolve plans/inputs from here |
| `docs/v2_economic_data_scaling_progress.md` | Current decision summary and historical evidence |
| `docs/v2_SCALING_INVESTIGATION.md`, `docs/v2_ATTENTION_REVERSAL_DIAGNOSTICS.md` | Investigation scope/findings |
| `docs/v2_PERFORMANCE_COMPARABILITY.md`, `docs/v2_CAPACITY_RESULTS.md` | Comparable metrics and previous capacity outcomes |
| `docs/v2_scaling_data_inputs.json`, `docs/v2_scaling_store_contract.json` | Current store acceptance and checkpoint compatibility |
| `docs/v2_SCALING_DATA_REPAIRS.md` | Completed added-period data repairs |
| `docs/v2_CORPORATE_ACCOUNT.md`, `docs/v2_ECONOMIC_ACCOUNT_ACCEPTANCE.md` | Account hypotheses and bounded admission |
| `docs/v2_COMPILED_SCORE_FAILURE.md` | Compiler correction and qualified scope |
| `docs/v2_STORAGE_CLEANUP.md`, `docs/v2_strong_cleanup.json` | Cleanup/recovery catalog |
| `research/pyproject.toml`, `research/uv.lock`, `research/.venv` | Research environment, separate from collector tooling |
| `ops/` | Existing frozen-plan drivers, audits and summaries |
| `research/src/brazil_rv/v2/` | Models, training, data consumers and evaluation |
| `research/src/brazil_rv/execution/` | Accounts, allocation, loans, settlement and custody |
| `C:/quant-data/b3/interim/ti` | Active short-path compiler cache; preserve during jobs |
| `C:/quant-data/b3/raw/`, canonical source archives, `C:/Trading/` | Immutable sources/broker installation; do not modify |

Current resolved inputs/outputs, for orientation; resolve pointers before use:

- Fresh-fit store: **C:/quant-data/b3/processed/v2_scaling_store_20260921**.
  Manifest SHA256
  `e0dfad778e09ffac81a6212dbb99a22555f7dbb6ff0020c5400b7025f8862efd`.
  Both `scaling_data_inputs` and `economic_refit_inputs` now select its acceptance.
  Earlier matched/composed/Natura/foundation stores remain immutable.
- Attention root: **C:/quant-data/b3/processed/model_runs/v2_attention_stopping_20260921**.
  Contains plan.json, refits.json, parents/, fits/, evaluation/, refit_economics/,
  ordering_resume/, training/continuation logs and the finite continuation script.
- Common-model root: **C:/quant-data/b3/processed/model_runs/v2_common_model_comparison_20260921**.
  Contains refit_plan.json, neutral/plan.json and flexible/plan.json. Fits and
  training.log will appear as the queued stage runs.
- Evidence/recovery root: **D:/quant-data/b3/processed/model_runs/v2_economic_data_scaling_20260919**.
  The investigation lives under scaling_investigation/. Many canonical dependencies
  are on D:, not merely in the C: junction. Do not scan all quant-data recursively.

Relevant keys in the main run pointer:

```text
scaling_data_inputs                 scaling_input_recovery
scaling_matched_stopping_plan       scaling_matched_stopping_resume
scaling_matched_stopping_evaluation_plan
scaling_continuation_plan           scaling_refit_economics
scaling_common_model_plan           scaling_common_evaluation_plans
economic_account                   stage_d_compiler_acceptance
scaling_expanded_results            scaling_strong_cleanup
```

Qualification/result pointers appear when their stages finish. Do not substitute
an old experiment's report for a missing new result. Use
`brazil_rv.v2.data_repair.bound_json` for hash-bound JSON and
`binding`/`artifacts.write_json_atomic` for new bindings. Preserve frozen plans;
register justified amendments separately, never edit a hash to silence a mismatch.
Current pointers, accepted contracts and live status supersede old narrative queues.

## 4. The “merge to GitHub” rule

**Completed work must reach GitHub main. A local change, local commit, branch or
unmerged PR alone is not completed delivery.** Routine integration is already
authorized; do not repeatedly ask permission to carry it through.

1. Implement the smallest correct change and run appropriate targeted checks.
   Freeze experiments before outcomes. Preserve failed/skipped attempts honestly.
2. Archive/hash recoverable new evidence on D:, verify recovery, and update
   accepted pointers/reports. Keep large datasets/checkpoints out of Git; do not
   duplicate immutable inputs into every recovery package.
3. Commit reviewed files. If work was on a branch/worktree, integrate into main
   and resolve conflicts without losing another job's/user's changes. A normal
   documentation change may be committed directly on main.
4. Push main to **https://github.com/gabrool/brazil-rv.git**.
5. Verify local HEAD, origin/main and the GitHub API SHA agree; inspect worktree
   status. If publication fails, retain work and report the failure. Do not claim
   it is merged or use force-push to conceal a mismatch.

```powershell
Set-Location 'C:/Brazil-RV/quant/b3-quant'
git diff --check
git status --short
# Stage only reviewed task files, then commit/integrate as appropriate.
git push origin main
$localCommit = git rev-parse HEAD
$originCommit = git rev-parse origin/main
$githubCommit = gh api repos/gabrool/brazil-rv/commits/main --jq .sha
if ($localCommit -ne $originCommit -or $localCommit -ne $githubCommit) {
    throw 'Local/origin/GitHub main do not match'
}
git status --short
```

Starting main for this handoff was
`b5e937613da41afe762710b340b07fdd9e03c70e`, verified on GitHub. This document is a
subsequent documentation change; resolve HEAD for new work rather than pinning
everything to that starting SHA.

Financial fits require a **clean, commit-bound runtime checkout**. Keep running
and queued numerical checkouts pinned. Do not edit, pull, merge, reformat or
upgrade dependencies underneath them. Main can advance independently. Their
older frozen commits are intentional. Never bypass the clean-worktree guard.
Exact-byte bindings also matter: CRLF/LF conversion has caused source-hash
mismatches. Restore frozen bytes only after proving normalized source identity
and retaining that evidence; do not weaken provenance checks.

## 5. Running efficiently on the RTX 2060

Verified installed stack: **RTX 2060, 6,144 MiB VRAM; Python 3.12.13;
PyTorch 2.6.0+cu126; triton-windows 3.2.0.post21**. Use
`C:/Brazil-RV/quant/b3-quant/research/.venv` through uv. Windows pins differ from
the old Linux/GH200 environment. Do not install globally or casually upgrade this
working compiler stack. Routine commands use `--no-sync`; rebuild/sync from the
existing lock only when needed and no affected job is running.

The current code already implements the important optimizations:

- **One persistent GPU worker, sequential fits.** Reuse process/cache/immutable
  inputs, with model, optimizer, RNG and compiler state reset at fit boundaries
  by the existing driver. Never compete for 6GB VRAM with a second fit or manual
  GPU verification while the continuation owns the queue.
- **FP16 autocast on Turing; FP32 parameters, moments and losses.**
  `round7_training.autocast_dtype` intentionally selects FP16 here. Do not force
  BF16 because an API reports emulated support. Keep gradient scaling, unscaling
  before SAM perturbation/clipping, and exact same-batch/RNG overflow retries.
  Skipping an update changes the registered learning contract.
- **Inductor default mode on Turing.** `train.compile_forward()` maps expensive
  max-autotune requests to default on this GPU. Keep compiled forward/backward
  and losses, the dynamic date dimension, stable stage-wide name padding and
  dynamic CUDA-graph safeguards. Do not force capture on sparse dynamic paths.
- **Preserve the qualified ATen LayerNorm forward/backward fallback.** A fused
  GRU96 residual-normalization kernel demonstrably read uninitialized temporary
  memory. The correction is in compile_forward(), not an optional speed tweak.
  Finite scores/high GPU utilization do not prove correct forecasts; retain the
  registered eager/saved-score checks for new runs.
- **Exact compact session/security caching.** DateTensorCache stores histories
  once and gathers causally instead of materializing all overlapping 60-session
  windows. Balanced batches visit every fit date once per epoch, with at most
  16 dates per batch. Compact identity-aware padding retains every eligible name;
  it does not authorize security subsampling or truncated history.
- **Single-thread CPU math:** OMP/OpenBLAS/MKL=1 and torch.set_num_threads(1).
  The existing CPU account continuation can overlap GPU training.
- **Short cache path:** C:/quant-data/b3/interim/ti. Default Windows temp paths
  produced an unwritable 264-character Triton artifact; the qualified short path
  fixed it. Do not clear the active cache mid-run.
- **Measure cold and warm time separately.** Use manifests, preparation/cache
  bytes, epoch logs and peak CUDA memory. Recent attention steady epochs are
  roughly 4–6 seconds in observed jobs; initialization, compilation, selection,
  export and verification add time. This is not another graph/fold's fit ETA.

Never gain speed by dropping names, shortening history, shrinking the 60-epoch
maximum/LR schedule, changing seeds/losses or reducing validation. Such changes
require separate frozen contrasts. Reuse passed source/consumer/account proofs;
do not repeat large matrices merely to create a new checkpoint.

Current attention P recipe: LR 1e-4, SAM rho 0.125, executed patience 20 with
patience 5 prefix views. F: ASAM rho 0.2, eta 0.01, patience 5. Both retain max/schedule 60,
minimum improvement 1e-4 and transferred LR multiplier 0.3. EMA half-life 1 is
separately recorded; raw checkpoint selection and earlier ties remain. The C6
slow parent has its own original stopping/selection cadence; do not replace it
with the attention recipe. Read the frozen plans for full recipe details.

### Recovery-only launch example: do not run while the worker exists

First inspect processes and the failing/incomplete fit. Preserve logs/artifacts.
Completed manifests are reused; an incomplete fit can resume only from a matching
resume.pt with optimizer/scaler/RNG state. The trainer rejects incomplete output
without resumable state. Never delete that evidence or bypass the guard.

```powershell
$repo = 'C:/Brazil-RV/quant/b3-quant'
$checkout = 'C:/Brazil-RV/.worktrees/attention-stopping-resume'
$run = Get-Content -LiteralPath "$repo/docs/v2_economic_data_scaling_run.json" -Raw | ConvertFrom-Json
$plan = Get-Content -LiteralPath $run.scaling_matched_stopping_plan.path -Raw | ConvertFrom-Json
$busy = @(Get-CimInstance Win32_Process | Where-Object {
    $_.Name -match '^(python|uv)\.exe$' -and
    $_.CommandLine -match 'run_matched_stopping\.py|compare_common_models\.py|run_economic_refits\.py|brazil_rv\.v2\.train\b'
})
if ($busy.Count) { throw 'Training already exists; inspect it instead of duplicating it' }
Set-Location $checkout
git status --short
# Verify expected commit and frozen driver/runtime hashes before continuing.
$env:PYTHONPATH = "$checkout/research/src;$checkout/ops"
$env:PYTHONUTF8 = '1'
$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:TORCHINDUCTOR_CACHE_DIR = 'C:/quant-data/b3/interim/ti'
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
uv run --project "$repo/research" --no-sync python -u ops/run_matched_stopping.py `
    *> "$($plan.root)/training_recovery_$stamp.log"
```

Explicit PYTHONPATH is essential: the shared environment's editable package
otherwise resolves main instead of necessarily the pinned checkout. Check imported
module paths when isolating a runtime. The current resume driver SHA256 is
`627aa5e2fcd9f3793f32d1774597165d986a97abf84dd012ec09f96c30a951ef`.
Do not invoke --freeze again for an already frozen experiment.

The finite continuation has a different checkout and must remain unique. If it
stops, inspect continuation_progress.json and its log before resuming. Its entry
point is continue_comparison.py under the attention root, with PYTHONPATH pointing
to scaling-evaluation/research/src and scaling-evaluation/ops. It launches
ops/compare_common_models.py itself; do not independently launch that queued job.
Replay scripts can default to older plans: use explicit current bindings.

## 6. Assumptions, safeguards and remaining limits

- Account: ordinary domestic corporate CNPJ, R$10m primary, R$1m/R$5m checks.
  100% historical CDI on eligible settled short proceeds, zero execution brokerage
  and CDI-plus-zero debit spread are favorable **negotiated research hypotheses,
  not obtained quotes**. Do not impose retail loan-turnover penalties, fund
  discounts or an invented required zero-interest admission case.
- Separate dated B3 spot/loan/custody charges, rent/intermediation, brokerage,
  shortfall, free/proceeds income and actual settled debit financing. Never add
  B3 to the old bundled4bp or double-charge invoice adjustments. Preserve
  settlement, prior-close funding, loan cohorts and pending obligations.
- Independent source/arithmetic/causality checks support the bounded contracts;
  they do not prove every historical datum or broker term. Linx's held2021
  BDR/cash valuation and some delivery/fraction/lender terms remain explicit.
  Apply frozen bounds when actual books are exposed; zero-exposure controls are
  not quantitative bounds on held positions.
- Old models/forecasts keep their old model coordinates. Account-only replays
  shallow-copy frozen PolicyData and change admitted inputs explicitly; never
  regenerate static coordinates. New stores require fresh training conditioning
  and compatible new P parents. Equal schema shape is not checkpoint compatibility.
- Preserve all 933 permanent identities, 3,717 accepted dates, 2009 warmup/full 60-session
  history. Tickers are dated attributes. Keep effect/knowledge/custody/payment
  separate. A spot rename is not a loan alias or permission to pool acquired
  companies' filings/history.
- No invented OHLC, quotes, fills, locates or exact endpoints; no forward-filled
  “observed” bars, entry-bar features, future-fitted scalers or current constituents
  used historically. Unit-uncertain denominators stay unsupported until sourced
  evidence resolves them. Do not relax thresholds to obtain better results.
- Freeze contrasts before outcomes. Preserve failed/skipped attempts and separate
  source/account/data/refit attribution. Report net-above-CDI, labelled currency-
  consistent Sharpes, drawdown, seeds/periods, paired 20/40/60-session uncertainty
  (40 primary), turnover and holding information. Validation gains alone do not
  rule out selection bias or implementation errors; investigate concrete causes.
- **No Lambda/cloud compute, deployment, live/forward capture, new subagents,
  automatic extra seeds or unregistered broad grids.** At most two new candidate
  cells per registered capacity wave. Reread registration before it.

## 7. Implementation map, storage and delivery discipline

All entries below are repository-relative; read actual call sites before editing.

| File(s) | Responsibility |
|---|---|
| `ops/run_matched_stopping.py`, `ops/test_matched_stopping.py` | Current attention execution, prefix selection/reuse and ordering tests |
| `ops/compare_common_models.py`, `ops/run_economic_refits.py` | C6/GRU plan and shared fit driver |
| `ops/verify_attention_forecasts.py` | Own-store eager forecast verification |
| `ops/replay_data_refits.py`, `ops/qualify_refit_books.py`, `ops/summarize_scaling_comparison.py` | Explicit-plan books, saved arithmetic and comparisons |
| `research/src/brazil_rv/v2/round7_training.py`, `research/src/brazil_rv/v2/train.py` | Recipes/caches/SAM/resume and compiler/sampling/splits |
| `research/src/brazil_rv/v2/characteristic_model.py`, `research/src/brazil_rv/v2/temporal_pathway.py`, `research/src/brazil_rv/v2/model.py` | Attention/GRU and C6 graphs |
| `research/src/brazil_rv/v2/data.py`, `research/src/brazil_rv/v2/store.py`, `research/src/brazil_rv/v2/round7_preprocessing.py` | Causal consumers, virtual targets, training-only conditioning |
| `research/src/brazil_rv/execution/portfolio_account.py`, `research/src/brazil_rv/execution/stateful_ledger.py`, `research/src/brazil_rv/execution/portfolio_policy.py` | Actual account/allocation paths; supporting loan/custody modules are alongside |

Use uv run --project research --no-sync for targeted pytest/Ruff commands from
main, with PYTHONPATH set to the intended source/ops. Protect important invariants
instead of maximizing test count. Keep code lean; avoid new frameworks,
dependencies, compatibility shims or repeated upstream checks. Evolve canonical
implementations and use Git for history. PROJECT_CONTEXT is for durable contracts
and accepted decisions, not a per-action changelog.

The user authorizes aggressive cleanup of stale files. The last pass reclaimed
**70.55 GB** (10.21 GB deleted, 60.33 GB losslessly compressed), including 52 inactive
checkouts. Only main and the two active checkouts remain. Historical paths in old
reports can therefore be absent; recreate from recorded commits only when needed.
Compressed evidence keeps its decoded hashes/paths, with extra read/decompression
cost for cold artifacts.

Resolve scaling_strong_cleanup for exact catalogs. Metadata recovery:
`D:/quant-data/b3/processed/model_runs/v2_economic_data_scaling_20260919/scaling_investigation/storage/strong_cleanup_20260921.zip`,
SHA256 `61329d534452bc509f1297724b88cf9e872f9148ddbacd6691c15ba4adb031f8`.
Selected weights and forecasts remain online. Retired intermediate objective
weights/optimizer states restore from the prior verified full tar, not that small
metadata ZIP. Do not delete active caches/current parents/unique evidence/raw
archives/accepted stores. Resolve exact intended Windows roots before recursive
deletion, and avoid following junctions into protected locations.

**Next:** refresh the two jobs and let healthy work continue. Investigate actual
failures while reusing completed fits. Review attention verification and matched
comparisons, then the queued C6/GRU/policy results. Check actual exposure limits
and all seeds/periods before choosing a capacity contrast. Deliver recoverable
evidence, a clear statement of the best comparable model and remaining uncertainty,
and verified GitHub-main integration. One audit, cleanup or model wave does not
complete the whole research program.
