# Decision program: CPU checkpoint and GPU restart

This document preserves the authorized Phases 1–3 program for a later session.
The user's latest instruction is to finish CPU work and defer Lambda while they
resolve its account/API problem. **Do not retry launch, restore capacity polling,
or move the full forecasting campaign to local CPU without a later instruction.**
No instance has been launched by this program. The failed approved launch returned
HTTP 401 `global/invalid-api-key` at 2026-09-14T22:52:24Z, before billing/launch.
S3 credentials and the Lambda Cloud API credential are different. Do not embed
either in source, this document, shell history or committed artifacts.

## Read order and authority

1. `C:/Brazil-RV/AGENTS.md` and this repository's `PROJECT_CONTEXT.md`.
2. `docs/v2_decision_progress.md` and the Phase 1/2 reports it links.
3. `docs/v2_PORTFOLIO_DECISION_POSTMORTEM.md` section 13, the original agreed plan.
4. `research/preregistrations/v2_decision_research.md`,
   `v2_decision_phase2.md`, and `v2_decision_phase3.md` in that same directory.
5. This restart document and `docs/v2_decision_cpu_checkpoint.json`.

The user's operational source is `C:/Users/gabri/Downloads/Work in CBrazil-RV..txt`.
Its launcher/host paths remain authoritative. Its old research status and store
pointers have been superseded by subsequent accepted work; use the current
canonical pointers below. No raw data, broker cache or sealed historical results
may be modified. No forward capture, Phase 4 or 2025/2026 consumer read is authorized.

## Durable inputs and locations

Repository: `C:/Brazil-RV/quant/b3-quant`, branch `main`, remote
`https://github.com/gabrool/brazil-rv.git`. Python 3.12, `uv run --project research`.
The workspace path is a junction to `C:/quant/b3-quant`.

Resolve `docs/v2_decision_run.json` for the decision root. At this checkpoint:

`C:/quant-data/b3/processed/model_runs/v2_decision_60dc9a2_20260914T212110Z`

The original `frozen_design.json` SHA-256 is
`907810f9382ceba1a3d2bcf57cfbf8940160d787b5edd2d4d78f8716ba162580`.
It binds all nine copied cache/metadata files to the sealed source inventory.
Three `cache/ARM/policy_data.pkl` files contain all required historical economic
inputs and old OOS forecasts. Each is 699,471,732 logical bytes; NTFS compression
reduces local physical space without changing hashes. Do not rebuild them.

Resolve `docs/v2_data_inputs.json` for the accepted repaired store:

`C:/quant-data/b3/processed/v2_data_store_2d59b9d_20260913T211700Z`

Manifest SHA-256:
`61149d43a23fc55fcd92b43aaa4e74d747fbdf62f852bedbbbdb2cf18281f3cc`.
It has 3,717 dates through 2024-12-30, 933 historical security identities, at most
243 active names, 568,815 active stock-days and 145 scalar fields. Preserve the
full 60-session input history and every PIT-active name. Padding is chosen from
the actual stage population; it is not a universe cap.

The old OOS economic caches cover 2,099 dates, 2016-07-18 through 2024-12-30,
with 363,314 active predicted stock-days. Original caches also remain at:

`/lambda/nfs/brazil-rv-east3/quant-data/b3/processed/model_runs/v2_portfolio_49d3c9a_20260914T143600Z/cache`

`docs/v2_portfolio_policy_recovery.json` binds their sealed inventory and local
forecast/evidence archives. The new decision root is local; do not assume it has
already been uploaded to persistent Lambda storage.

## Phase 3 materials already prepared

- `phase3/parents.json` identifies six verified original parent members and their
  source archive/manifest. All are already extracted under `phase3/parents`.
- The three TE_all parents load unchanged. Three C6 parents explicitly adapt the
  corresponding S0 P parent by copying shared tensors and adding zero-initialized
  cold-family projections. The exact inherited prediction comparison passes.
  Existing adapted parent files must be preserved, not recreated each session.
- `phase3/economic_targets.npz` SHA-256 is
  `f66bdafa1e2e8d5d393eeadbfebd9fb5a9549ff3266da50b751da029ecda184a`.
  It contains 361,572 valid five-session daily residual labels. Each F collator
  separately enforces its permitted fit/endpoint window and fits its own scale.
- `phase3/cpu_acceptance.json` records the real F2 fit-only acceptance: full
  history, unchanged initial neutral predictions and nonzero auxiliary-only
  gradients into both encoders after a head update. Its probe updates are discarded.
- `phase3/mappings/FOLD.json` freezes each arm's old-OOS equal-rank mapping and
  the blend weight chosen exclusively on the earlier selection dates.
- `objective_program.py` implements preparation, CPU checks, compiled GPU smoke
  admission, matched fits and bounded dispatch. `objective_readouts.py` implements
  fixed-policy books, rank/cardinal/tail readouts, the causal blend, paired economic
  and IC intervals, advancement gates, and continuous survivor inventory replay.

The CPU checkpoint JSON contains the exact current hashes and acceptance status.
**No Phase 3 financial fit/result is present. GPU acceptance is still pending.**
The readout path has synthetic matched-account and future-mutation tests; its
first complete execution on real new score artifacts remains a required check.

The real CPU probe covered the first/last F2 fitting cross-sections: 112/143
active names, 111/143 available auxiliary labels and 144 padded slots. Both
architectures retained every active name. The fitted label scale is .0091397613
daily-return units, based on 50,453 permitted fitting labels. C6 has 127,943
parameters including the auxiliary head; TE_all has 1,489,540. Auxiliary-only
gradient norms into the slow input projection were 0.00008595 and 0.00064581,
respectively. These are computation/gradient checks, not evidence of learnability
or improved forecasting. Separate synthetic portfolio tests reproduced identical
neutral/economic daily books exactly when their supplied forecasts were identical.

## Exact remaining experiment

TE_all uses early peer interaction, all accepted families and ASAM .2. C6 uses its
slow GRU with fundamentals/magnitudes and SAM .125. For both, compare fresh neutral
ranking training against the same training plus .25 times an economic Huber loss.
The single auxiliary head shares the encoder. Its target is
`(shareholder_H5 - CDI_H5 - decision_beta * BOVA_excess_H5) / 5`.
Fit-only equal-date RMS scaling preserves magnitudes and zero. Missing auxiliary
labels never remove a name or its neutral ranking loss.

Same compatible parent, seed, sampler, 60-epoch schedule and selection rule within
each pair. Both variants select on neutral IC to isolate the objective change.
The new C6 training recipe differs from its historical recipe, which is why the
fresh matched neutral C6 control is mandatory. Do not reuse old F fits as controls.

The screen is 2 architectures × 2 objectives × 4 folds (F2/F6/F10/F14) × 3 seeds
(11/29/47): **48 new F fits**, plus four disposable two-epoch GPU smoke fits.
No fresh P training and no automatic seed expansion. Report each seed plus an
actually replayed equal-rank ensemble. Retain full histories/module diagnostics.

Use the fixed equal-rank calibrated QP with historical CDI cash, identical risk,
borrow and fill assumptions for both objectives. Each fold starts from cash on
its first new evaluation date and liquidates at its final boundary. Do not insert
old forecasts as burn-in. Blend TE weight comes from {0,.25,.5,.75,1}, selected
on old OOS prior-window utility; exact ties prefer the lower weight. The weight
is identical for new neutral and economic variants. Do not re-rank the convex
combination of normalized rank coordinates.

Gate: positive ensemble paired net and utility; positive utility in at least
three of four folds; no individual seed mean net or utility below −.25 bps/day.
Only surviving architectures receive the other ten folds, with both objectives
and all three seeds: **60 additional fits per survivor**. If neither passes,
there are no confirmation fits. The blend is diagnostic and cannot trigger an
additional architecture campaign.

Where all fourteen new forecast blocks exist, replay a single continuous account
with actual inventory across model changes. Do not splice fold P&Ls or substitute
other models for missing blocks. Continuous results remain descriptive if the
confirmation panel fails. No automatic architecture or policy promotion follows.

## Local continuation commands

From the repository, use one BLAS thread per process. With the current 16 GB PC,
allow at most two cache-heavy fit processes; run two-cache mapping preparation
alone. An overlapping mapping process exhausted memory during imports. Its failed
attempt produced no mapping/result; keep recorded logs and valid completed fits.

```powershell
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$decisionRoot = (Get-Content docs/v2_decision_run.json -Raw | ConvertFrom-Json).root
uv run --project research python -m brazil_rv.v2.objective_readouts mappings --root $decisionRoot
```

The controller screen uses `controller_program plan --kind reliability
--max-parallel 2`, `run_many`, and `controller_program summarize`. Completed
controller manifests are skipped by the corrected plan even though `resume.pt`
is preserved. Never delete completed fits to make a launcher restart work.
Epoch resumes require the actual recorded source commit; completed fits from
earlier commits remain valid under their own provenance and numerical comparisons.

## GPU restart when the user resumes it

First verify clean committed `main` and push. Do not change training code while
jobs that have not imported it are queued. Use this only-approved launch command:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\ops\lambda-gh200.ps1 -Mode Launch -IUnderstandBilling
```

It uses us-east-3, one GH200, and filesystem `brazil-rv-east3`; it prepares the
checkout but does not start training. Record the exact instance ID and its printed
SSH command/IP. Runtime state and the failed-launch log are under
`C:/Users/gabri/AppData/Local/BrazilRV/lambda-gh200`; known-host file is
`known-hosts/<INSTANCE_ID>` and SSH private key is
`C:/Users/gabri/.ssh/lambda_brazil_rv_ed25519`. Use the launcher's host-key options.
Do not guess an old IP or adopt another instance merely because its name matches.

Upload only new derived decision artifacts. Before transfer create a relative-path
SHA-256 inventory, then verify every transferred file on the destination. Reuse
the accepted store and original caches already on the NFS after checking their
bound hashes. Copy the original root `frozen_design.json`, parent/target material,
CPU acceptance and frozen mappings. Preserve all local results. Never overwrite a
different remote root. The approved path resolver maps C:/D:/quant-data to the NFS.

```bash
cd /home/ubuntu/Brazil-RV/quant/b3-quant
export BRAZIL_RV_DATA_ROOTS=/home/ubuntu/Brazil-RV/quant/b3-quant/research/configs/v2/data_roots.lambda_us_east_3.json
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
decision_root=/lambda/nfs/brazil-rv-east3/quant-data/b3/processed/model_runs/v2_decision_60dc9a2_20260914T212110Z

# Before any Phase 3 smoke/financial fit: preserve the local engineering design,
# then bind the final clean implementation and relocated paths. Parent bytes stay exact.
uv run --project research python -m brazil_rv.v2.objective_program prepare --root "$decision_root"
uv run --project research python -m brazil_rv.v2.objective_program plan --root "$decision_root" --smoke --max-parallel 2
uv run --project research python -m brazil_rv.v2.run_many --plan "$decision_root/phase3/smoke_plan.json" --manifest "$decision_root/phase3/smoke_launcher.json"
uv run --project research python -m brazil_rv.v2.objective_program accept-smoke --root "$decision_root"
```

Inspect finite losses/gradients, two-epoch timing, GPU memory, compilation counts,
population equality and cache use. Verify BF16/compiled numerical behavior is
acceptable on the actual device before the financial launch. CPU acceptance is
not a GPU numerical benchmark. Increase concurrency only when the measured device
has room and throughput improves; preserve full history/model complexity. Do not
extrapolate the compile-heavy first epoch as steady-state time.

```bash
uv run --project research python -m brazil_rv.v2.objective_program plan --root "$decision_root" --max-parallel 2
uv run --project research python -m brazil_rv.v2.run_many --plan "$decision_root/phase3/screen_plan.json" --manifest "$decision_root/phase3/screen_launcher.json"
for fold in F2 F6 F10 F14; do
  uv run --project research python -m brazil_rv.v2.objective_readouts evaluate --root "$decision_root" --fold "$fold" || break
done
uv run --project research python -m brazil_rv.v2.objective_readouts summarize --root "$decision_root"
```

If there are survivors, first prepare their remaining mappings with
`objective_readouts mappings --confirmation`; use `objective_program plan
--confirmation` and run its `confirmation_plan.json`. Evaluate F1/F3/F4/F5/F7/F8/
F9/F11/F12/F13 with `objective_readouts evaluate --confirmation`, then summarize
with `--confirmation`. Run `objective_readouts continuous --arm <SURVIVOR>` once
all fourteen blocks and mappings exist. The CLI deliberately refuses partial
continuous histories. Do not call `run_many` with an empty survivor plan.

## Completion checks for the later session

Verify all emitted score/checkpoint hashes, exact seed/date/ISIN axes, full active
population and fit/selection/evaluation chronology. The auxiliary head must export
actual daily-return units. Keep failed/ceiling-stopped seeds in the comparison;
do not retune using screen outcomes. Preserve settlement flags and terminal haircut
sensitivities; tiny dust and materially missing historical fills are different.

After Phase 3, reread the original postmortem section 13, update its acceptance
record, and write one combined LLM-ready review covering Phase 1, the actual account
fixes, every Phase 2 engineering attempt/admission decision, financial results,
parent adaptation, objective screen/confirmation, blend, timing and limitations.
Rank selection remains fixed, so a null result does not rule out economic checkpoint
selection or end-to-end decision training. Phase 4 remains a separate future task.

Recover selected weights, full training histories, diagnostics, scores/masks/axes,
exact books/fills/orders, gates, source bindings and logs. Hash every recovered
artifact and verify the destination before deleting or terminating anything.
Commit/push code and reports; verify local, remote checkout and GitHub commits.
Then terminate only the exact recorded newly launched instance through the Lambda
API or console and verify that ID absent from inventory twice. The launcher itself
never terminates instances. Do not leave a paid host waiting for a later review.

The existing capacity automation remains paused. There is no forward-capture job
to restart. The earlier portfolio instance was already terminated before this
program; do not treat it as the instance for this future launch.

At this checkpoint C: has less than 0.4 GiB free and D: approximately 1.2 GiB.
Do not start a large local extraction blindly. Reuse verified persistent inputs,
recover compressed artifacts selectively, and check actual free space before
copying. The CPU checkpoint record identifies the verified local archive; that
archive is a local recovery copy, not a claim of S3 upload. Existing canonical raw
data, the repaired store and original source archives remain untouched.
