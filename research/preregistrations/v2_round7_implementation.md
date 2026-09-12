# Round 7 implementation resolutions

Registered before new model scores. The supplied Round-7 specification is kept
verbatim in `v2_round7.md`. This file makes its executable choices explicit;
descriptive claims in that document are hypotheses, not engineering guarantees.
The user's instruction to implement the round and prior autonomous continuation
authority apply. Report CPU preflight and archived diagnostics before paid work;
the repaired S0 anchor necessarily follows that report in the first paid session.
Its additional diagnostics are reported before screening. No forward capture,
2025/2026 consumer access, Round 8 or deployment is authorized.

## Data and clocks

- Bind the accepted Round-6 store through `docs/v2_round6_inputs.json`, manifest
  SHA-256 `968df3947b68ec68d62638ee53402c908b9ea2e24dfd27291e156e17d31a31c1`.
  Source archives and sealed results remain immutable. Write one new derived
  store, with bounded memory and an explicit changed-array inventory.
- The corroboration rule applies to retrospective action accounting and targets.
  A filing or provider event after a historical decision cannot enter that
  decision's features. Preserve the existing slow and other protected feature
  arrays. Existing C1 cash terms remain exact. Record evidence date and session
  distance for every accepted/rejected U2 candidate; no price-outcome adjudication.
- DISMES corroboration uses +/-2 sessions, reflecting the documented delayed code
  update. Provider corroboration requires a non-unit split factor within +/-5
  sessions; a nearby cash dividend does not corroborate a unit conversion. IPE
  subject/category and issuer timing use the specified +/-30-session rule.
  New continued-print terms use corrected fixed old factors in their running unit
  history, and apply the same corroboration rule before inferring a new U2.
- Re-admitted BDI 06/07/08 prints require an already established permanent identity
  and its accepted dated ticker mapping. They restore observed prices and claim
  accounting, never create new eligibility. Freeze the existing `active` array.
  The changed-array exception includes recovered raw observation fields, action
  accounting, dependent targets and the audit-only eventual-survival flag.
  Historical decision-causal wealth features remain identical. A separate
  entry-fill mask blocks openings on recovered prints at execution time; it never
  changes earlier intended orders, membership, ranks, marks or printed exits.
  All differences must be explained by repaired prints or action terms.
- Neutral targets can change for peers on a touched date/horizon because ranking
  and neutralization are cross-sectional. Audit both directly affected outcomes
  and these propagated target changes; do not require only the repaired name to
  change. Keep causal risk/exposure inputs fixed for this target repair.
- A PDF creation timestamp alone proves generation, not public availability.
  Historical captures or other publication evidence determine any earlier
  admission. Without evidence, retain the current date-only next-decision bound.
  Do not impose another arbitrary lag on top of a supported publication clock.
- Native fundamentals retain signed earnings yield and revenue growth, natural
  log of strictly positive book-to-market, unchanged standardized SUE, the other
  named physical values and a valid signed-earnings indicator. Median/IQR scaling
  is fit-only, including at Stage P, then clipped to [-5, 5]. Constant fields use
  unit scale; validity remains separate. No 20-name support requirement.

## Architecture and attribution

- Preserve 60 sessions, 32 slow fields, the width-64 GRU and all registered C1
  widths/depths. No top-N truncation. Use a fixed compact axis large enough for
  every historically active name in the relevant stage; inactive slots are masked.
- Retain the existing GRU's value/mask/age/age-known encoding. Core and family
  encoders use the specified three channels; unknown age is represented by -1
  after bounded age encoding, so it cannot be confused with observed age zero.
- Enumerate common/per-name cross-market fields by name before a fit. The text's
  'about 25' is not a field limit: retain every registered common shock/flow field
  plus the three diagnostics, with masks/ages, scaled only on the fit window.
  Common fields enter FiLM only in C1. Fit common scalers on unique dates, not
  duplicate stock rows. S0's A3 retains its original per-name linear encoding.
- Entirely invalid families contribute an exact zero embedding and a separate
  family-valid flag. Nonlinear encoders do not mathematically guarantee full-rank
  learned representations; sensitivity diagnostics remain necessary.
- B1 minus A1 changes the architecture and five-to-three training heads. B7
  quantifies the head-count component within C1; do not call B1 minus A1 a pure
  width or head-independent architectural effect. FiLM and sidecars are disabled
  in slow-only C1; shared context there uses only the slow information set.
- Pretraining reuse requires identical graph, input roster, preprocessing and
  Stage-P recipe. A3 and five-head C1 need their own compatible parents; slow-only
  and all-family C1 do not share an all-family-trained parent. This exceeds the
  draft's approximate fifteen-P-run count. Report the actual job inventory.
- Stage P uses a fixed 60-epoch R schedule for new C1 graphs and A3, with rho .05
  and tail averaging; each graph/input contract shares this parent across its F
  radius/loss controls. S0 uses repaired-store Round-6 P. Thus comparisons change
  the declared full architecture/training pipeline, not just F initialization.
- Attention uses exact masked scaled-dot-product attention, four heads of width
  32, with fused kernels where supported. TabM expands only the shared encoded
  state into eight members; it does not repeat the GRU/family/context encoders.
  Each member's cross-sectional loss is calculated independently, then averaged;
  inference averages member scores. No member-axis mixing in ranks or losses.
- Permutation equivariance is checked with floating-point tolerances and exactly
  matching masks. Reordered reductions need not be bitwise identical. Full-graph
  training and evaluation must each remain bounded to one static graph for a
  configuration; compilation time and peak memory are included in smoke evidence.

## Budget, comparisons and decisions

- Calibration: twelve C1-all rho-.05 fits, F2/F6/F10/F14 and seeds 11/29/47,
  each run for 60 epochs with its schedule defined on 60. Take the equal-fit mean
  selection IC at each epoch, then its trailing five-epoch mean. Let B be the
  earliest multiple of five in [20, 60] whose trailing mean is within 0.001 of
  the maximum trailing mean over epochs 20--60. If no earlier multiple qualifies,
  use 60. Report curves and freeze B before any screening evaluation score.
  Screening schedules are then defined on B; calibration with a 60-epoch
  schedule estimates a budget, not the identical shorter-schedule trajectory.
- Tail average the final ceil(B/4) end-of-epoch parameter states, uniformly.
  This has no contribution from an initial EMA shadow. Save every epoch's raw
  weights and selection daily IC so diagnostics are recoverable. Selection does
  not choose screening/confirmation weights. Reuse calibration as B4 only if B=60
  and every training/input/scoring contract matches; otherwise do not.
- S0 A1/A2/A3 keep their five equal loss heads. C1 uses three, except B7. The
  primary readout always uses the common D3/D5/D10 population; legacy D1/D2/D3/D5
  is S0-only. AdamW is an actual single-pass optimizer update, not rho=0 SAM.
- Archived raw-versus-EMA differences estimate uncertainty of that paired
  comparison, not checkpoint noise alone. Use a paired daily series and Newey-West
  mean standard error with ten session lags, reporting the available observation
  count and EMA endpoint/initialization confounds. Missing epoch checkpoints are
  reported as unavailable, never reconstructed from claims about training logs.
- The chronological tree corrector includes a matched S0-score-only tree control.
  Past-fold labels must mature before the next fold's first decision, with the
  horizon purge enforced by actual dates. Its gain is evidence consistent with a
  representational/optimization limitation, not proof that one is the only cause.
- Screen all fifteen cells unless measured cost invokes the stated cut order.
  A0's screen results reuse four of its 42 anchor fits. Confirmation reuses the
  twelve exact screening fits per advanced cell; only ten remaining folds are
  new. These are disclosed evaluation-selected development panels.
- Advance A1 plus up to four other non-anchor cells: rank by paired mean primary
  IC against A0, admit the leader and those within .002, break exact ties by the
  table order. If more qualify, retain the top four; A1 occupies its mandatory
  slot. Apply engineering/economic eligibility to designation, retain all losing
  results. Six-seed confirmation adds 61/79/97 for the 14-fold leader and A0,
  with compatible Stage P for the added seeds on both sides.
- The IC-first economic override and negative-economics exclusions remain those
  of the standing research contract. A C1 designation additionally requires a
  positive paired point IC against A0 in every leave-one-seed-out panel. If no
  eligible C1 clears that rule, choose the eligible recipe-only comparator when
  it improves A0; otherwise retain A0 and report failure to improve. An adverse
  A1 is not automatically promoted merely because no other model beats it.
- CPU/source audits and implementation proceed together where independent.
  Paid-session throughput estimates come from measured complete concurrent
  smokes, not multiplication by an old early-stopped S0 fit average. Recover and
  hash required artifacts before terminating only the recorded instance ID,
  then verify absence with two provider inventories.
