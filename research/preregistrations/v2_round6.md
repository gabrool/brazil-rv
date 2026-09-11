# v2 Round 6 — the information round: one family per arm on S0

Audience: the coding model (repo `main` at `9e28ea0`). Round 5 is accepted as reported:
the store `v2_round5_store_9020bde_20260911T023227Z` (manifest `44a19df1…`; 76 protected
arrays, both axes and 36 tables exact; three lending arrays replaced under
authorisation; ten families; 4.0 GiB peak), the BOVA11 replay, the continuous S0 book,
the CPU screens (770 cells), the sidecar contract with its tests. Register as
`v2_round6.md` before any score; freeze against that store, the sealed S0 panels
(reusable by hash — no refit), and the Round-5 screen readout. Paid compute on Gabriel's
go after §1.

---

## 0. Reading of Round 5

The dataset is built the way the registration asked: every family stamped with its
first eligible decision by the source's own clock, versions by receipt, no second lag,
mutation fixtures per family, protected arrays byte-identical. Three cross-market groups
(oil, ADR premium, foreign flow) and three option fields are masked because the coding
model would not assign a clock it could not prove; §1 resolves the three that the
registration's rule already covers. Coverage after masks: cross-market, magnitudes,
microstructure and odd-lot near-complete from 2010; events and fundamentals 17% of
active name-days in 2010 rising to 63% by 2018 and about 80% from 2020 (the dated CVM
identity bridge, not the model, limits the early years); options 34% in 2018 to 88% in
2024; lending 70–80% from 2020 (informative F6–F14); sector from 2018 (13 folds);
rebalance 2023–2024 only (2 folds).

The tree screens (slow GBDT + one family, 14 folds, 5 seeds, paired on informative
folds, parent 0.0174) are exploratory and unadjusted for ten comparisons, and the tree
is a weaker learner than the network on these inputs (it saw no gain from lending where
the network did). Read as ordering, not as verdicts: magnitudes +0.0032 [0.0010, 0.0058],
odd-lot +0.0031 [0.0009, 0.0051], options +0.0028 [0.0003, 0.0047], lending +0.0032
[−0.0001, 0.0051] on nine folds, events +0.0023 [−0.0016, 0.0054]; microstructure,
cross-market and sector flat; fundamentals −0.0025 [−0.0092, 0.0036]; rebalance
uninformative on two folds. Three positive nominal intervals out of ten, each of the
size of S0's own gain over its parent, is a good result for a data round. The
fundamentals sign is the one thing to look at before spending fits on it (§1.2).

How the network will use the data: the sidecar contract adds each family at the
decision row as a validity-gated, zero-initialised linear residual into the fused
representation before the trunk, so the two residual blocks can form nonlinear
interactions between a family and the slow representation, and an invalid family
leaves the parent's forward pass bit-for-bit unchanged. That is sufficient to use every
family here; it does not need attention (attention is about names seeing other names,
a separate hypothesis for Round 7). Two consequences are registered below: the
projection parameters are additive, so S0's Stage-P checkpoints load unchanged into
every arm (§2.2), and the families' own dynamics reach the model only through the
change/age fields they carry, which is deliberate for this round.

---

## 1. Pre-flight (CPU, before or alongside session 1)

1. **Complete the cross-market family** under the registration's rule and rebuild only
   that family into a new store root (protected arrays and the other nine families
   byte-identical; the extension mechanism already does this in under a minute):
   Brent is retrieved and admitted with `available_at` = the next session's 15:45 (a
   market series whose day-t fixing time is unproven is available at t+1 as its t
   value; later evidence can only move it earlier); `adr_return_gap_1` = the ADR's t−1
   US-session log return in BRL via PTAX minus the name's t−1 B3 log return for
   dual-listed names, and `ewz_minus_bova11_1` as a common shock (log returns, so vendor
   adjustments cancel; the US-close rule already implemented); `foreign_flow_1/5` as the
   differences of consecutive published month-to-date totals at their D+2 publication,
   month-boundary rule stated, methodology-change sessions flagged, family label
   `published_total_difference`, entering through the products with `log_volume_mean_20`
   and the ADR-listed flag. The oil exposure and interaction fields follow from the
   admitted shock. Fixture proofs as for every family.
2. **Fundamentals and events sign diagnostic** (readout, no selection weight): for each
   field, the pooled and by-year univariate Spearman IC against the D5 neutral target
   on the family's supported population, with the sign the literature predicts beside
   it (profitability, earnings yield, SUE positive; leverage, accruals negative; size
   negative to flat; filing age and material-fact recency small). Expected magnitudes are
   0.005–0.02. Wrong signs or zeros across years point at an alignment or unit problem
   in the family and would be fixed before its arm; consistent signs mean the tree
   simply could not use it and the arm runs as planned.
3. **Index releases back to 2018** from B3's own dated news pages (the April-2019 first
   preview and its attachments are an example), same publication proof and parser; only
   the rebalance family changes; needed before the rebalance arm in session 2, not
   before session 1.
4. **Ledger realism, in parallel and not gating any arm:** admit the late-2024
   latest-vintage lending rates under the `latest_vintage` label with the 1,535-overlap
   evidence, for features and as a ledger sensitivity; add the `placeholder_v2`
   borrow-cost scenario (each name's median observed 2023–2024 taker rate shrunk toward
   its liquidity-quintile median; names without an observed rate take the quintile
   median) beside the flat 2% in every economics table; report the 22 stale-name
   settlements by name, date, side, notional, days without a print, later delisting or
   tender, and concentration by fold and liquidity quintile. Also probe the Internet
   Archive for the old B3/BM&FBovespa average-rate pages as a first-vintage source for
   2012–2023 rates and record the result either way.

---

## 2. Protocol

### 2.1 Parent, panels, folds

Parent: S0 (Round-4 graph, seeds 11/29/47, sealed scores hash-bound to the new store's
unchanged slow inputs), fourteen folds, the adopted execution policy, A1 full-calendar
economics on fold-reset panels as the paired basis; the continuous book as a readout for
the designated candidate. Paired comparisons on exact common populations; each family
arm is judged on its informative folds from the store acceptance record (all fourteen
for magnitudes, odd-lot, options, events, fundamentals, microstructure, cross-market;
F6–F14 lending; F2–F14 sector; rebalance's after §1.3), with the all-fold delta beside
it; the A3 momentum diagnostics for every arm; `primary_ic` headline, legacy metric
secondary.

### 2.2 Arms and Stage P

Every family arm = S0 + one family through the sidecar contract, nothing else changed.
Stage P: reuse S0's three Round-4 checkpoints — the sidecar projections are additive
zero-initialised parameters absent from the checkpoint, and the contract's tests show
the family-absent graph reproduces S0 exactly, so the load is compatible by
construction, not forced; record the missing-key load in the transfer audit. A
fresh-Stage-P variant (family present in pretraining, three P runs) is run in session
2 for the best single family and for the combination arm, as its own registered arm.
The three cheap non-data arms reuse S0's checkpoints too.

| Session | Arms | Fits |
| --- | --- | ---: |
| 1 | magnitudes, odd-lot, options, events, fundamentals, lending; fine-tuning multiplier 1.0 vs 0.3 (E4); time decay 756; the point-in-time MLP comparator (E8: a small residual MLP on the latest slow feature vector with the same masks, targets, selection and folds, to tell whether the GRU's temporal modelling is doing work) | 9 × 42 = 378 |
| 2 | cross-market (after §1.1), sector, microstructure, rebalance (after §1.3); C6 = S0 + every family whose session-1 informative-fold paired IC point estimate is positive, one arm; fresh-P variants of the best single family and of C6 | 4 × 42 + 42 + 2 × (42 + 3 P) |

Session-2 rosters are fixed from session-1 results by the rule stated here, before any
session-2 score exists. Losing arms are reported at the same detail as winners.

### 2.3 Readouts added for this round

Per arm: the informative-fold and all-fold paired deltas by fold; the residual IC and
momentum correlation; an inference-time family ablation (the trained arm scored with
the family forced invalid, paired against itself) as the network's own attribution of
the family's contribution; for E8, the MLP's paired delta against S0 read as the
temporal-modelling premium; for E4 and time decay, the paired delta by era (2018–2019,
2020–2021, 2022–2024). Economics for every arm on the fold-reset panels under both
borrow scenarios once §1.4 exists; the continuous book for the designated candidate.

### 2.4 Promotion

Unchanged: `primary_ic` leader among eligible candidates on the fourteen folds, the
economics override when the paired IC interval versus the leader includes zero and the
paired economics interval is strictly positive, negative or undefined economics
ineligible, persistence and turnover reported and weighed. Confirmation seeds 61/79/97
on both sides when the designated candidate's informative-fold paired IC lower bound
against S0 lies within ±0.001 of zero or the override is invoked. Gates bind registered
panels only. The Round-7 parent is the designated candidate (C6 included: a combination
of families is a single model, unlike an ensemble). The leave-one-seed-out audit runs
on every decision as in Round 4.

---

## 3. Budget and sessions

At Round-4 rates (2.6 minutes per fit at six concurrent, 42 fits ≈ 111 minutes),
session 1 is about 16 GH200 hours and session 2 about 11, plus Stage P, evaluation and
transport: roughly US$90 in three sealed sessions, each ending with host copy, hash
verification, exact-ID termination and two provider reads. Measure after the smokes
(one per configuration, one epoch, two compiled graphs, no score) before projecting.
Split session 1 in two if wall-clock requires; nothing depends on all nine finishing
together except the C6 roster.

---

## 4. Not in Round 6

Attention and the hyperparameter pass (Round 7, on the Round-6 parent); the overlay
track and the portfolio round (the continuous evaluator is ready; the meaningful-capital
cost analysis sets what turnover costs first); any blend with a control; the 2025 read.

## 5. Order and stops

1. `v2_round6.md` with §§1–2 and the protocol JSON (arm list, informative folds from the
   acceptance record, Stage-P reuse rule and transfer audit, C6 rule, E8 contract).
2. §1.1, §1.2 and §1.4 on CPU; §1.3 in parallel.
3. Session 1 on Gabriel's go; seal; C6 roster fixed and recorded.
4. Session 2; seal.
5. `docs/v2_ROUND6.md`: arm table, paired deltas on informative and all folds, family
   ablations, E4/E8/time-decay readouts, diagnostics, promotion trace, Round-7 parent,
   2025-bar status. Stop.

Stops as in `v2_round4.md`, with gates bound to registered panels; a family arm whose
smoke shows more than one compiled training graph stops and is diagnosed before any fit.
