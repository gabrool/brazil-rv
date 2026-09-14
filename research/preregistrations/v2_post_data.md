# Post-data stages A–C

Authorized 13 September 2026. This implements stages A–C of
`docs/v2_POST_DATA_RESEARCH_PLAN.md`. Re-read that document after each stage and
record acceptance and remaining work in `docs/v2_post_data_progress.md`.
Stages D–F, forward capture, and 2025/2026 consumer access are outside this run.

## A: engineering contract

Resolve `docs/v2_data_inputs.json` and bind its accepted manifest. Keep every
eligible security, the complete 60-session window, repaired preprocessing and
unchanged labels, loss, splits and economics. Sealed historical results remain
bound to their original code and inputs.

The new recipe selects raw weights on complete selection dates from epoch 1.
An improvement exceeds 0.0001 mean D3/D5/D10 common-population IC; ties retain
the earlier state. Stop after five consecutive checks without improvement.
The safety ceiling is 60 epochs; warm-up and cosine use the original fixed
60-epoch-equivalent update schedule, independently of stopping. A ceiling hit
with continued improvement is reported before any extension is considered.
P uses the same selection-aware policy. The incumbent S0 control retains its
accepted recipe and newly trains its own compatible P on the repaired store.

Ordinary SAM .125 is the primary new-recipe control. ASAM uses
`epsilon = rho * T^2*g / ||T*g||`, with `T=abs(w)+0.01` for non-bias/non-normalization parameters and
unit metric for biases and normalization parameters. This explicit mixed
metric retains perturbations in zero-initialized outputs; it is not a claim of
exact scale invariance for the whole network. Those adaptive parameters retain
decay .01; every module-owned bias and normalization parameter has zero decay.
Dropout remains .1. Transferred parameter names are explicit; the F multiplier
is .3 for the common recipe, with a 1.0 calibration bridge. New parameters use
the full LR. Exact SAM restoration and identical two-pass RNG are required.
An encoder with neither a valid value nor a known age anywhere in P's fit
population is wholly unexposed and uses full F LR, despite its parameters being
present in the parent file. Preserve its initialization; do not discard other
families' learned representations. Partially exposed encoders and shared fusion
tensors retain their transferred rate. Record these names and exposure evidence.

Compile objective and model with the existing full-graph path where measured
beneficial. CUDA uses BF16 autocast with FP32 parameters, optimizer state,
ranking loss and sensitive reductions. Do not shorten history, omit names,
reduce model dimensions or relax masks to improve a speed benchmark. Verify
forward/backward and mask parity, finite cold-module gradients, and actual
full-fit timing. Keep compilation overhead visible.
On GH200, materialize the canonical fit/selection tensors once in GPU memory,
then gather the same date batches. Preserve FP32 values, all masks and ages;
never refit or approximate preprocessing in the cache. Its finite date scope
is inherited from the dataset's existing access boundary. Record setup time
and bytes, and prove exact tensor equivalence before financial dispatch.

Log every epoch's loss, clipping frequency, complete selection and a fixed
era-spanning clean-fit probe. At selected/terminal states measure full clean-fit
IC. Bounded fit-only probes report module gradients, update/weight norms,
activation scale, perturbed loss/score change, FiLM magnitude and peer-route
sensitivity. Diagnostics preserve RNG and weights; hooks do not enter compiled
training graphs. Never log complete activations or attention matrices.

## B: bounded calibration before financial evaluation

First require independent-date synthetic own-history, same-time peer, lagged
peer and context-conditioned peer tasks using actual full-model rank loss and
the proposed SAM/ASAM optimizer. Own-only and uniform-peer controls distinguish
relational learning from shortcuts. AdamW is an optimization diagnostic.
Keep all failed attempts. A failed engineering gate requires diagnosis, not
silent substitution of component-only acceptance.

Engineering diagnosis after the original label-free-donor task failed: compare
the identical inputs and recipient targets with donors additionally supervised
on their own observed signal. Recipient validation remains on new dates and
uses only recipient labels. The own-only control bypasses both stock mixers
while preserving the donors' own information and supervision. This separates
joint temporal learning from requiring wholly unlabelled source stocks to learn
history through indirect gradients alone. Preserve the original failures and
the post-hoc nature of this diagnosis. A joint-task pass establishes that
conditional capability only; it cannot certify learning from unlabelled market
histories or justify a financial architecture/optimizer change by itself.

The joint lagged task passed seed 11 while joint context had only .340 mean
IC. Before interpreting that aggregate, record validation IC separately for
the two observed regimes. A final bounded confirmation uses joint context
full/uniform at 8192 fresh updates on seeds 11/29, joint lagged full on seed 29
and uniform on seeds 11/29 at 4096, and corrected own-history seed 29 at 4096.
The longer context schedule is a disclosed engineering adaptation. Preserve
terminal weights for later diagnostics; no financial target or model changes.

Engineering disposition before any financial fitting: accept the demonstrated
**conditional** capability, with all 54 attempts retained. ASAM .2 learns true
own history and dynamic peers across two seeds. With joint donor supervision,
lagged IC is .9475/.9557 and context IC .8601/.8573; both context regimes learn
on both seeds. Context uniform controls are .0131/.0135, lagged uniform controls
are -.0110/.3438, and the joint own-only controls remain near zero. The original
unlabelled lagged/context tasks remain failures. This permits the registered
financial calibration/screen, not a claim of universal routing ability or a
guarantee for every optimizer recipe. Do not select the financial optimizer
using synthetic IC or replace actual financial labels with teacher supervision.

Financial calibration uses P and F fit/selection only, initially F2/F14 and
seeds 11/29, on early-attention slow and C1-all representatives. The primary
bracket is LR 1e-4 with SAM .125 and ASAM .2/.5. Add a matched SAM .125 LR 3e-4
bridge and SAM .05 reference. Parent LR-transfer .3 versus 1.0 is a bounded
bridge on F14/seeds 11/29 only, not a full cross-product or a four-fit winner.
For each lane choose by equal-fit selected selection IC; settings within .001
of the maximum tie by the declared order SAM .125, ASAM .2, ASAM .5, LR bridge,
SAM .05. Report complete trajectory stability alongside that deterministic rule.
There are 44 calibration F fits. Use one newly selection-trained SAM .125 P
parent per graph/input/seed across F recipes, keeping parent differences out
of this optimizer comparison. This is not an ASAM-versus-SAM pretraining study.
Do not inspect evaluation panels until
the screen roster/settings are frozen. Selection is development tuning.

## C: fresh matched screen

F2/F6/F10/F14, seeds 11/29/47, twelve F fits per complete cell. Fresh parents
must match the repaired input and graph contracts. Include incumbent S0, S0
under the new common recipe, GRU/C1 slow, temporal-attention slow with early
peer interaction and its parameter-matched late counterpart. Then add repaired
per-name families to the selected reference and early-attention model, with
common FiLM off/on as separate contrasts. Common fields remain available in
the FiLM-on arm; no PCA or data-channel deletion.

The nine screen cells are incumbent S0, common-recipe S0, GRU/C1 slow,
early-attention slow, matched late-attention slow, and the C1/early-attention
per-name-family paths each with FiLM off/on. C1 is the shared family-capable GRU
reference; it is retained even if S0 wins, because it isolates the same family
fusion architecture. The rich-GRU calibration recipe applies to C1 variants;
the slow-attention recipe applies to both attention timings and family variants.
Common S0 uses SAM .125/LR 1e-4. All parents use the same selected SAM .125 P
recipe except the incumbent, which retains its own accepted P recipe. Nine
graphs/input configurations times three seeds require 27 P fits in total;
four are already needed for calibration. Of 108 screen F fits, eight can reuse
exact calibration fits, giving 144 distinct F fits across B+C. No automatic
extension of this roster is allowed merely because a preferred model loses.

Record the calibration choice before new evaluation scores and preserve a paired early/
late recipe comparison alongside any independently tuned practical comparison.
Reuse only exactly compatible completed fits. Do not force the incumbent onto
the new recipe. No automatic extra three seeds or fourteen-fold expansion.

Report paired IC, per-head/composite alignment, seed/fold dispersion, blocked
uncertainty and descriptive unchanged-book economics. Include all attempts and
source limitations; this is repeatedly studied development history, not a new
untouched holdout or a production designation. Secure artifacts and publish the
combined A–C report, then terminate only the recorded research instance after
complete recovery and exact-ID verification.

Paired daily uncertainty uses within-fold moving blocks of 20 sessions with a
60-session sensitivity, 10,000 draws and seed 20260913. Keep same-date names,
heads and seeds together; do not count stock-days as independent samples.
Report each cell versus S0, S0_common versus S0, TE_slow versus TL_slow and
C1_slow, and family/FiLM increments within each architecture. These are
descriptive development screens with nominal intervals, not multiplicity-
adjusted confirmatory claims. No winner automatically triggers stages D–F.
Book readouts run in four independent fold processes; chronological operations
inside each book, sources, masks, costs and paired metrics are unchanged.
