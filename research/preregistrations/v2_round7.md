# v2 Round 7 — the canonical architecture and a training recipe fit for the problem

Audience: the coding model (repo `main` at `7010049`). This replaces the earlier Round-7
draft. It synthesises the Round-6 results, the coding model's design diagnosis and
literature review, and the reviewer's diagnosis; where the two reviews disagreed, the
resolution is stated in §8. Register as `v2_round7.md` before any score. Paid compute on
Gabriel's go after the pre-flight (§1) and the zero-training diagnostics (§2) are
reported.

## 0. What this round is

Round 6 established that the network at 0.0256 IC uses new information weakly and
unstably: fundamentals +0.0026, C6 +0.0022, three families the tree liked flat or
negative, the parent choice flipping with one seed, the learned family maps rank-one,
the GRU matching a latest-row MLP. Neither review could name a single dominant failure,
and Gabriel's direction is that the initial architecture should have been the canonical
one for this problem. Round 7 therefore does two things at once and attributes the gain
between them: it introduces a canonical per-stock characteristic model built for a
cross-sectional ranking problem with heterogeneous point-in-time inputs (§3), and a
training recipe designed for a target at the noise floor (§4), and it compares both
against S0 in a factorial that separates recipe, architecture and joint inputs (§5).
It runs on a store with the two known label defects repaired (§1), so the parent is
refit once.

Speed: a fixed four-fold screen ranks cells, fourteen folds confirm the few that matter,
fits use the Round-6 compact-axis/BF16/unique-date path. Target: pre-flight and
diagnostics two days, implementation one day, about 22 GH200 hours in two or three
sealed sessions. §7 gives the order in which to cut if time runs over.

---

## 1. Pre-flight (CPU; the store every Round-7 fit uses)

1. **U2 crash-day conversions.** Complete the addendum's audit, then apply the rule: a
   unit conversion is inferred only with corroboration — a DISMES change, a
   provider-archive action within ±5 sessions, or an IPE filing for the issuer within
   ±30 sessions whose category or subject is a desdobramento, grupamento, bonificação or
   conversion; an uncorroborated large move with a quantity surge is
   `large_move_no_action`. C1 cash inference unchanged. Report every reclassified event.
2. **Judicial-recovery parser exclusions.** Admit COTAHIST BDI 06/07/08 rows as
   continued prints for identities already in the universe (AMER3 from 2023-01-20, OIBR4,
   SEQL3 and any others the audit finds); such names remain ineligible for new entries
   under the liquidity rules; verify claim continuity across the code change and report
   the sealed S0 book's affected positions before and after.
3. **Foreign-flow clock** (addendum §5): bulletin creation timestamps by year and
   Wayback captures decide D+2 versus D+3 availability by era; rebuild the family if the
   rule changes.
4. **Native fundamentals family** `fundamentals_native` (nine fields): signed
   `earnings_yield_ttm`, log `book_to_market`, `gross_profitability`,
   `liabilities_to_assets`, `accruals_to_assets`, signed `revenue_growth_yoy`, `sue`
   in its own standardised units, raw `log_market_cap`, and `earnings_negative_flag`.
   Robust scaling (median and IQR) and clipping at ±5 fitted on each fit window only, as
   the magnitudes family does; validity and receipt age carried; not rank-transformed,
   so the 20-name cross-sectional support rule does not apply and sparse observations
   survive. Report, per field, how many name-days the rank rule had suppressed.
5. **One store build** with 1–4 as a new root: every array outside the repaired action
   arrays, the affected targets, the foreign-flow fields and the new family bit-identical
   to `968df394…`; the 8-GiB invariant; the usual audits. Targets change only where an
   action reclassification or a re-admitted print touches them; report the count of
   changed target cells by fold.
6. **S0 anchor on the repaired store**: Stage P (three seeds) and fourteen-fold Stage F
   under the Round-6 recipe exactly (42 fits), so the repair's effect on S0 is a readout
   (Round-6 S0 versus repaired-store S0) and every Round-7 comparison has a same-store
   parent.

Time box: two days. If the U2 audit needs manual adjudication beyond the rule, apply the
rule, list the residual cases, and proceed.

---

## 2. Zero-training diagnostics (report before the GPU session; none is a gate)

On the Round-6 S0 and fundamentals fits (84 fits) and the repaired-store S0 anchor:

1. **Clean fit IC versus selection IC**, same heads, same population, inference mode,
   per fit and per epoch where checkpoints exist: the under- versus over-fitting
   question both reviews asked. Under-fitting shows as fit IC close to selection IC and
   both rising at the selected epoch; over-fitting as fit IC well above.
2. **Paired checkpoint noise.** For each fit, the paired daily IC difference between the
   selected checkpoint and the final EMA, and its serial-dependence-adjusted standard
   error on the 55-session window; this measures the checkpoint-selection noise directly
   rather than inferring it from the absolute daily standard deviation.
3. **Archived EMA versus raw patience**, paired on fourteen folds, reported with the
   confound the coding model identified (28–34% weight on the transferred initial
   parameters in early folds): a practical comparison, not a causal one.
4. **Head-gradient alignment and SAM gap** on fixed fit batches: per-head gradient norms
   and cosine similarities on shared parameters (D1/D2 versus D3/D5/D10), and the loss
   at the SAM-perturbed point minus the loss at the current point.
5. **Residual-corrector control** (CPU): LightGBM on the S0 score rank plus all family
   fields, trained on the evaluation panels of folds ≤ k−1 and scored on fold k
   (chronological, labels matured and purged), paired against S0 alone. If a tree finds
   incremental information on top of S0's score that the neural families did not, the
   representation is the limitation; if it does not, the information is thin.
6. **Sidecar sensitivity**: Jacobian norms of the composite score with respect to each
   family's value channels, by fold, so the rank-one weight maps are read against actual
   output sensitivity.

---

## 3. The canonical architecture, C1

A per-stock characteristic model with heterogeneous encoders, market conditioning, a
cross-sectional context block, a residual trunk and ranking heads for the traded
horizons; permutation-equivariant across names; every input point-in-time as stored.
Widths are the registered values; the coding model reports the exact parameter count.

**Inputs for name i at decision t** (all from the store, unchanged semantics):
temporal window `X_i` (60 × 32 slow fields with validity and age); latest-row core
characteristics `x_i` (the 32 slow fields at t−1, value/validity/age); family vectors
`s_{i,f}` for each family f in {magnitudes, odd-lot, options, events, fundamentals,
fundamentals_native, lending, microstructure, sector, rebalance, cross-market per-name},
each as (value, validity, age); common state `c_t` (about 25 dimensions, fit-window
scaled). The cross-market family's 65 fields are split by their nature: the per-name
fields (exposures, exposure × shock interactions, the ADR return gap) form the
`cross-market per-name` family; the common fields (shocks at 1 and 5 sessions, foreign
flow, the EWZ − BOVA11 gap, VIX) join the three stored common-state diagnostics in `c_t`
and enter only through the FiLM block, never as per-name channels.

**Blocks:**

1. Temporal branch (kept, ablatable): the existing GRU, width 64, over `X_i` → `h_i`.
2. Core encoder: MLP on `x_i` (3·32 → 128 → 128, GELU, LayerNorm) → `e_i`.
3. Family encoders, one per family: MLP (3·n_f → 64 → 32, GELU, LayerNorm) → `e_{i,f}`;
   a family invalid for the name-day contributes a zero vector and its validity flag;
   full-rank by construction, so the rank-one map cannot recur without the data forcing it.
4. Joint state: `z_i = LayerNorm(W_z · concat(h_i, e_i, e_{i,1..F}))`, width 256.
5. Market conditioning (FiLM): `[γ_t, β_t] = MLP(c_t)` (25 → 64 → 512), `γ` and `β`
   initialised to zero so the block starts as the identity; `z'_i = z_i ⊙ (1 + γ_t) + β_t`.
   A common scalar can now change the ranking only by changing which stock features
   matter, which is the mechanism the literature identifies for macro information.
6. Cross-sectional context: the existing gated mean/dispersion pooling of `z'` over the
   active names on date t, concatenated and projected back to 256. (Attention variant:
   one masked multi-head self-attention block over the active names — 4 heads, key width
   32, residual, LayerNorm, padding mask, shared weights across names — in place of the
   pooling.)
7. Trunk: three residual SwiGLU blocks, width 256, inner 256, dropout 0.1. (TabM
   variant: the same trunk as a parameter-efficient ensemble of k = 8 members sharing
   weights with per-member multiplicative adapters on inputs and outputs; the loss is
   the mean over members; scores are the member mean.)
8. Heads: linear D3, D5, D10. D1 and D2 are not produced by C1; the composite is the
   tie-aware rank average of D3/D5/D10 as the adopted policy already uses. The
   `legacy_primary_ic_1235` readout is reported for S0 only.

**Pretraining and transfer:** Stage P on 2010-01-04 → 2016-06-30 with all families
present as stored (absent families masked; coverage by family in the P span reported),
then Stage F per fold with all parameters at the base learning rate (multiplier 1.0,
which E4 showed is equivalent to 0.3 and is simpler). Every C1 variant with a different
graph has its own three-seed Stage P; S0 keeps its own.

**Engineering acceptance before any real fit:** (a) with every family invalid and the
FiLM block at its initial state, C1's outputs depend only on `X_i` and `x_i`; (b) a
synthetic-target test — a known nonlinear target built from permitted inputs (a product
of two family fields plus a threshold on a native fundamental, masked like the real
data) — reaches an IC above 0.5 within the fixed budget on a small real subset; (c)
permuting the name axis permutes the outputs exactly; (d) one compiled training graph
and one selection graph per configuration, as now.

---

## 4. The training recipe, R

Registered as protocol settings; every cell in §5 uses R except the S0 anchor.

- **Fixed update budget, no early stopping.** Training runs B epochs of unique-date
  batches. The learning-rate schedule is defined on B: 5% linear warm-up, cosine decay to
  5% of the peak at the end of B — never on a ceiling that is not reached. The 55-session
  selection window is scored every epoch and *reported*; it does not choose the
  checkpoint.
- **Tail weight averaging.** The scored model is the uniform average of the weights at
  the end of each of the last ⌈B/4⌉ epochs (LayerNorm only, so no statistics to
  recalibrate). This is the clean version of the archived EMA: it starts at a defined
  epoch, so it cannot retain the transferred initial weights.
- **Budget calibration, once, before the screen:** C1 all-families with ρ = 0.05 on the
  four screen folds, three seeds, run to 60 epochs recording clean fit IC and
  selection-window IC every epoch. B is the smallest epoch count after which the mean
  selection IC across those twelve runs stops rising (within 0.001), rounded up to a
  multiple of five, and not less than 20. B is then fixed for every cell, S0 included.
  This is the only use of the selection window for a choice, it is made once, and it is
  recorded before the screen.
- **SAM retained**; ρ ∈ {0.05, 0.10, 0.125} as cells. A single-pass AdamW cell is
  included in the screen as a diagnostic reference for SAM's total contribution and for
  its speed; it is not a candidate unless it wins outright.
- **Loss:** soft Spearman as now, on D3/D5/D10 only for C1 (five equal heads for S0 as
  its own recipe requires). One cell uses the Pearson-with-rank-targets surrogate — the
  same rank labels, no pairwise soft-ranking of predictions — as the loss control. MSE on
  unranked residuals is not run in this round: it changes the target as well as the loss.
- **Weight decay** no longer applied to LayerNorm and bias parameters (the routing
  cleanup the coding model identified), recorded as part of R.
- Dropout 0.1, weight decay 0.01, peak learning rate 3e-4, seeds 11/29/47, batch and
  precision settings from the Round-6 compute amendment — unchanged.

---

## 5. Experiment design

### 5.1 Screening stage: four fixed folds

Folds F2 (2018H2), F6 (2020H2), F10 (2022H2), F14 (2024H2), chosen now for calendar
coverage — the fold F5 that dominates every pooled figure is deliberately excluded.
Three seeds per cell, twelve fits per cell, paired comparisons on the exact common
population of those four folds. Every cell is fitted from its own fresh Stage P where
its graph differs.

| Cell | Architecture | Recipe | Inputs | Isolates |
| --- | --- | --- | --- | --- |
| A0 | S0 | old (patience, ρ 0.125, 5 heads) | slow only | anchor (from §1.6, the same four folds) |
| A1 | S0 | R, ρ 0.125 | slow only | the recipe alone |
| A2 | S0 | R, ρ 0.05 | slow only | recipe + radius on the old graph |
| A3 | S0 + old linear sidecars | R, ρ 0.05 | all families | joint inputs through the old representation |
| B1 | C1 (no attention) | R, ρ 0.125 | slow only | architecture alone |
| B2 | C1 | R, ρ 0.10 | slow only | radius |
| B3 | C1 | R, ρ 0.05 | slow only | radius |
| B4 | C1 | R, ρ 0.05 | all families | joint inputs through the canonical representation |
| B5 | C1 | R, ρ 0.125 | all families | radius × capacity × inputs interaction |
| B6 | C1 without the temporal branch | R, ρ 0.05 | all families | whether the GRU adds anything once the core encoder exists |
| B7 | C1 with five equal heads (D1–D10) | R, ρ 0.05 | all families | the horizon-alignment question |
| B8 | C1, Pearson-on-ranks loss | R, ρ 0.05 | all families | the loss surrogate |
| B9 | C1 with the attention block | R, ρ 0.05 | all families | cross-stock context beyond pooling |
| B10 | C1 with the TabM trunk (k = 8) | R, ρ 0.05 | all families | variance reduction inside the model |
| B11 | C1 | R, AdamW single pass | all families | SAM's total contribution and speed (diagnostic) |

Fifteen cells, 180 fits, plus Stage P for the distinct graphs (S0-on-repaired-store,
C1, C1-no-GRU, C1-attention, C1-TabM: fifteen P runs) and the twelve calibration fits.
Readouts per cell: primary IC on the four folds, paired deltas against A0 and against
the cell's nearest neighbour in the table, seed dispersion, clean fit IC at the end of
the budget, selection-window IC by epoch, momentum-residual IC, fold-reset economics.

### 5.2 Confirmation stage: fourteen folds

Advance to fourteen folds, three seeds: the best cell by paired IC against A0, every
cell within 0.002 of it, and A1 (so the recipe's effect on S0 is measured at full
power) — at most five cells beyond A0, chosen by that rule and recorded before any
confirmation fit. Then the confirmation seeds 61/79/97 for the cell that leads on
fourteen folds and for A0 (six-seed panels on both sides), with the leave-one-seed-out
audit. About 5 × 42 + 2 × 42 fits.

### 5.3 Decision rule, written now

The Round-8 parent (the v3 baseline) is the confirmed leader on `primary_ic` over
fourteen folds against the S0 anchor under the registered IC-first rule with the
economics override and negative-economics ineligibility; persistence and turnover are
weighed in the report. Any C1 variant that beats S0 becomes the baseline even if the
gain is not statistically established, provided its leave-one-seed-out panels agree —
the purpose of this round is to set the canonical starting point, and S0 is not a
privileged default. If nothing beats A1, the recipe alone becomes the baseline. All
screen-fold selections carry the label `screened_on_F2_F6_F10_F14`.

---

## 6. Readouts and reporting

`docs/v2_ROUND7.md`: the §1 repair effects (targets changed by fold; S0 Round-6 versus
repaired); the §2 diagnostics with their limitations; the calibration curves and the
chosen B; the screen table with paired deltas and seed dispersion; the confirmation
table; the six-seed decision; parameter counts and measured fit times per graph; the
momentum-residual diagnostics; economics under the three borrow scenarios; the
continuous book for the designated model; the 2025-bar status. Losing cells at full
detail. The synthetic-target acceptance and the permutation test recorded.

---

## 7. Budget, sessions and where to cut

Measured Round-6 rates were 2.6 minutes per S0 fit at six concurrent trajectories; C1
with all families and the 256-wide trunk should be assumed at two to three times that
until measured on the smoke. Screen: 180 fits + 15 P + 12 calibration ≈ 10–13 GH200
hours. Confirmation: about 294 fits ≈ 9–12 hours. Two or three sealed sessions, each
ending with host copy, hash verification, exact-ID termination and two provider reads;
measure after the smokes and project before committing a session.

If the screen must be cut, drop in this order: B11 (AdamW diagnostic), B7 (five heads),
B8 (loss control), B2 (ρ 0.10), A3. If the confirmation must be cut, confirm three cells
instead of five. Do not cut A0, A1, B1, B3, B4 or B9: they are the attribution chain
(recipe → architecture → inputs → context).

---

## 8. Where the two reviews differed, and what was adopted

Selection noise: the reviewer argued from the absolute daily IC standard deviation; the
coding model correctly noted that paired checkpoint differences are less noisy and must
be measured (§2.2). The redesign of selection (§4) is adopted regardless, because a
55-session window with patience three is unsuited to the problem by the standards of
every comparable study, and because it costs nothing. EMA: the coding model's confound
is real; the recipe uses tail averaging from a defined epoch instead (§4). MSE: dropped
in favour of the Pearson-on-ranks control (§4). SAM: retained per Gabriel; radii 0.05
and 0.10 against 0.125, crossed with the wider model (§5). Capacity: tested early, as
the coding model asked, but inside a canonical redesign rather than as a width change
alone, per Gabriel's direction; the factorial attributes the gain (A1 → B1 → B4).
Joint inputs: every C1 all-families cell answers the interaction question the coding
model raised, with A3 separating inputs from representation. Concatenation versus
addition: the encoders are nonlinear, so the equivalence the coding model noted does
not apply. Attention: one block, as an arm, not the base. Six seeds: confirmation only.
Screening folds: the coding model's four-fold proposal adopted, with the rule that the
screen ranks and fourteen folds decide.

---

## 9. Deferred, so nothing is lost

Temporal representations for the families (previous comparable value, causal change
sequences, short masked histories for fast sidecars) — Round 8, as a data-representation
change on the v3 baseline. Numerical feature embeddings (piecewise-linear or periodic)
for valuation thresholds — Round 8 arm. Target neutralisation versus the book's own
neutralisation, with risk-matched economics — its own target registration after the v3
baseline exists. Cost-aware objective and the calibrated trade-versus-hold policy — the
portfolio round, after the meaningful-capital cost analysis and on the continuous
evaluator. Overlay track — after the portfolio round. Bootstrap boundary weighting —
a statistical audit before the next promotion claim. ASAM — only if the radius results
show parameterisation sensitivity. Lending-rate history — closed unless a new source
appears. The 2025 read — unspent; the bar stands.

---

## Appendix — every item raised since Round 6, and where it lands

| Item | Raised by | Disposition |
| --- | --- | --- |
| Selection window 55 sessions, patience 3, best-checkpoint selection | reviewer; coding model (F7) | §4 fixed budget + tail averaging; §2.1–2.2 measure it |
| LR schedule tied to the 60-epoch cap | coding model | §4 schedule defined on B |
| EMA confound (initial-weight retention) | coding model | §4 tail averaging from a defined epoch; §2.3 archived EMA reported with the caveat |
| SAM radius 0.125 | Gabriel; both | §5 cells at 0.05 / 0.10 / 0.125 crossed with architecture; AdamW as diagnostic B11 |
| Head weights 40% on D1/D2 | coding model (F1) | C1 trains D3/D5/D10 only; B7 is the five-head ablation |
| Training vs selection population mismatch | coding model | §2.4 measured; no rule change |
| Loss surrogate | both | B8 Pearson-on-ranks control; MSE deferred (changes the target) |
| Fit vs selection IC (under/over-fitting) | both | §2.1 |
| Gradient alignment, SAM gap | coding model | §2.4 |
| Weight decay on LayerNorm | coding model | §4 routing cleanup in R |
| Fine-tuning LR multiplier; time decay; fresh P for 2 families | Round 6 | closed (tested) |
| Rank-gauss destroys cardinal information | coding model (F3) | §1.4 `fundamentals_native` family |
| 20-name support rule suppresses sparse fields | coding model | §1.4 native channels exempt; suppression counts reported |
| Families as decision-row scalars only; no history | coding model (F5) | Round 8 (deferred, §9) |
| Rank-one sidecar maps; linear additive adapter | both | §3 nonlinear per-family encoders, concatenation, wide trunk; §2.6 Jacobians |
| Common states need conditioning, not addition | both | §3 FiLM block |
| All families never tested jointly | Gabriel; coding model | B4/B5 and every all-families cell; A3 separates inputs from representation |
| Peer/relational information | both | B9 attention block; pooling now over the joint state (includes sidecars) |
| GRU ≈ MLP | Round 6 | B6 temporal ablation inside C1 |
| Capacity (121k params; trunk inner 48) | Gabriel; coding model | C1 widths (§3); B1 vs A1 attributes architecture; fresh P for every new graph |
| Target neutralisation vs the book | coding model (F6) | deferred to its own registration (§9) |
| U2 crash-day conversions | addendum | §1.1 repaired before any fit |
| Judicial-recovery parser exclusions | Round 6 | §1.2 |
| Foreign-flow publication clock | addendum §5 | §1.3 |
| Bootstrap boundary weighting | Round 6 | deferred statistical audit (§9) |
| Placeholder borrow rates | Round 5 | scenarios reported as before |
| Speed of the round | Gabriel | §5 four-fold screen, §7 cut order, compact-axis/BF16 path |
| Synthetic learnable-task check | coding model | §3 engineering acceptance (b) |
| Residual-corrector control | coding model | §2.5 |
| TabM-style efficient ensemble | coding model | B10 |
| Numerical feature embeddings | coding model | Round 8 arm (§9) |
| Cost-aware objective / trade-vs-hold policy; overlay | earlier rounds | portfolio round after the v3 baseline (§9) |
| 2025 read | standing | unspent; bar unchanged |

## 10. Order and stops

1. Registration `v2_round7.md` (§§1–5; protocol JSON: repaired-store identity, the C1
   contract with widths, the R settings including B once calibrated, the cell table,
   screen folds, advancement and decision rules, labels).
2. §1 pre-flight and store build; §1.6 anchor Stage P and F on the repaired store.
3. §2 diagnostics; report.
4. C1 implementation with the §3 acceptance tests; smokes for every graph; budget
   calibration; B recorded.
5. Screen session(s); advancement recorded.
6. Confirmation session; six-seed decision; seal.
7. `docs/v2_ROUND7.md`; stop. Round 8 is registered on the v3 baseline.

Stops as in `v2_round6.md`: engineering gates bind registered panels; a synthetic-target
or permutation acceptance failure stops before any real fit; any 2025/2026 consumer row;
an unauthorised paid instance; termination before the verified host copy; the 8-GiB
invariant.
