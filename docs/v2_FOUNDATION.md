# Model foundation experiments: interim review

2026-09-18. The user authorized all five workstreams. This report distinguishes
completed diagnostics from the new matched experiments, which are still running.
It is intended to be readable independently by an LLM reviewing the repository.

## Scope and experimental discipline

The question is whether input selection, variance reduction and a better allocation
of model capacity can improve the current C6/rich-attention candidates before adding
datasets or novel mechanisms. Follow [the registration](../research/preregistrations/v2_foundation.md)
and section 9 of [the research decision](v2_NEXT_RESEARCH_DECISION.md).

Preserve all 568,815 active stock-days, every eligible stock, full 60-session history,
point-in-time identities, fit-only conditioning and previous label/split contracts.
The immutable repaired store remains unchanged. Experiments use the RTX 2060 with
compiled FP16 neural operations, gradient scaling, FP32 loss/weights and FP64
accounting. No held-out consumer reads or forward capture are performed.

Input wave: three matched attention cells, each with its own three fresh SAM .125
pretraining parents and four folds by three seeds of ASAM .2 fine-tuning. The cells
are cleaned full input, no microstructure, and no oddlot/sector-alpha/rebalance.
The grouped deletion retains microstructure. Both candidate deletions apply to
pretraining and fine-tuning. This avoids silently calling a fine-tuning-only mask
a test of training without that data. C6's core roster is unchanged.

Raw IC selection and existing patience remain the reference. True EMA with a
one-epoch half-life is tracked during necessary F fits; it gets an independent
selection checkpoint on the same prior dates, bounded by the raw trajectory.
Tests verify exact resume and that EMA tracking does not change raw training.
No extra seeds, shorter histories or arbitrary stop budgets are introduced.

## Completed input audit

The earlier 145-field numerical audit found no nonfinite valid values or globally
constant observed field. The new [support inventory](v2_foundation_inventory.json)
adds distinct dates, coverage eras and exact original-checkpoint availability.

- `options:uncovered_call_share` has neither valid values nor known ages anywhere
  in the accepted development store. New compatible graphs exclude this field by
  name. Source files and old checkpoints remain unchanged.
- `put_call_oi_log_ratio` has 442 observed active stock-days on 409 dates;
  `delta_oi_to_volume_1` has 409 on 385 dates. Each has a median of one observed
  name on dates with any support. These are sparse cross-sections across hundreds
  of dates, not hundreds of fully populated cross-sections. Retain them pending
  conditional evidence; low support alone does not justify deletion.
- The bounded within-family rank-correlation audit uses P-fit and F14-fit dates,
  at most 50,000 active stock-days, and one observation per date for common fields.
  No adequately observed tested pair exceeds absolute Spearman .97. Sparse pairs
  with fewer than 100 joint observations remain unassessed. This does not establish
  absence of redundancy, information overlap or unhelpful predictors.
- Retained fields keep their values, masks and known-age semantics. No new stock
  eligibility, minimum-support or staleness filter was added.

The correlation audit consumes no labels and makes no deletion decisions. Its
source-bound result is [the field-overlap evidence](v2_foundation_field_overlap.json); its code
is `ops/audit_foundation_field_overlap.py`. Global support descriptions are not
being presented as an outcome-based causal feature selection method.

## Completed seed-variance diagnostic

This reuses the original Phase-3 neutral F trajectories, fourteen development folds
and seeds 11/29/47. Singles and three-seed books are reused from sealed artifacts;
all three pair forecasts are replayed. The current forward account reproduces the
sealed F2 ensemble exactly for both architectures (zero daily net-bps difference).
The same prior-only equal-rank calibration is held fixed across ensemble sizes.
Thus economics include changed forecast dispersion/confidence under the fixed
allocator; this is not a perfectly isolated statistical-variance experiment.

All singles and all pairs are averaged within their size. No best seed/pair is
selected. These are fold accounts that restart in cash, not uninterrupted accounts;
do not compare their means interchangeably with previous continuous-book numbers.

| Model | Forecast members | IC | Net above CDI, bps/day | Daily turnover, fraction of NAV |
|---|---:|---:|---:|---:|
| C6 | Mean individual | .027954 | 4.264 | .1577 |
| C6 | Mean of all pairs | .028556 | 4.242 | .1519 |
| C6 | Three-seed ensemble | .028735 | 4.245 | .1506 |
| Rich attention | Mean individual | .023509 | 3.783 | .1843 |
| Rich attention | Mean of all pairs | .025095 | 4.143 | .1680 |
| Rich attention | Three-seed ensemble | .025827 | 4.258 | .1597 |

Three seeds versus the average individual, paired circular 40-session blocks within
folds, nominal 95% intervals:

| Model | IC difference | Net difference, bps/day |
|---|---:|---:|
| C6 | +.000781 [.000394, .001194] | -.019 [-.324, .278] |
| Rich attention | +.002318 [.001539, .003093] | +.474 [.015, .915] |

Mean daily cross-sectional correlation of average-horizon ranks between seed pairs
is .914 for C6 and .777 for attention. Attention has more seed disagreement and
benefits more from the existing ensemble in this diagnostic. This supports the
priority of variance reduction; it does not demonstrate that EMA will help, that
more seeds should be trained, or that attention now dominates C6.

Intervals use 10,000 replications; 20/60-session alternatives and full source
bindings are in [the ensemble evidence](v2_foundation_ensemble.json). They are
nominal reused-development intervals, not adjusted for all prior research or
independent replication across overlapping seed pairs.

## Completed fixed forecast-blend diagnostic

The registered equal C6/attention forecast blend has now been replayed on all
fourteen folds. The weight is fixed at .5 before outcomes. Its calibration is
reused from the existing prior-only .5-blend trial; no evaluation dates fit a new
mapping. Forecasts are combined before one joint allocation, not after separate
books have earned returns.

The blend earns 4.948 net bps/day above CDI on the same fold-reset accounts.
Relative to C6, its paired mean differences are +.001181 IC (nominal 95% interval
[-.001494, .003908]) and +.703 net bps/day ([-.791, 2.090]). Relative to attention,
they are +.004089 IC ([.000948, .007170]) and +.690 net bps/day ([-.770, 2.083]).
These 40-session intervals support forecast diversification, but do not establish
economic superiority. Full evidence, alternative block lengths and all source
books are in [the fixed-blend readout](v2_foundation_blend.json).

## Checkpoint averaging and GPU engineering

All 84 original neutral F trajectories have hash-verified selected states and the
preceding states required for the fixed trailing mean of up to three checkpoints.
All 84 averages are prepared. They stop at the raw selected epoch; no late overfit
tail, independent-seed weight averaging or economic continuation is substituted.
Average inference and matched economic results are still pending. The stored
selector IC identifies the original cutoff, not the averaged predictor's IC.
All 84 averaged graph contracts also load strictly into their original model
classes. `ops/score_foundation_averages.py` reuses a compiled graph across compatible
seeds/folds in the historical inference worktree. It must wait for the fit worker;
there is only one GPU worker at a time.

Six disposable full-population GPU checks passed: one complete P epoch and F2 epoch
for each input cell. P padding is 160 names and F2 padding is 144, determined by
their actual eligible populations; later folds may require more. Peak allocated
CUDA memory was 1.62-1.64 GB for P and 1.15-1.16 GB for F2. These are allocated
tensor bytes, not total system GPU usage. Full evidence is in
[the engineering record](v2_foundation_engineering.json).

The first registered full-input P fit completed with selection patience at epoch 9.
Its steady epochs after compilation took approximately 9-10 seconds; preparation,
compilation and scoring are additional costs. This is not a timing estimate for
later, larger F windows or all remaining waves.

The component experiment now has a pooling-only temporal pathway: it retains the
same learned history pooling while omitting the early peer module. Using the old
`none` setting would also replace learned pooling with the last historical state,
confounding the peer ablation. Existing early/late paths retain their behavior.
Sixteen targeted tests cover masks, permutation/padding invariance, date isolation,
finite gradients and compilation, including exact pooling after a peer bypass.
Later waves can bind their own immutable implementation and list only genuinely
new fit cells, reusing eligible controls without refitting them. This does not
change the source of the input wave currently running.

## Accounting evidence acquired; corrections remain outstanding

Original issuer notices have now been downloaded and hash-bound in
[the source manifest](v2_foundation_primary_sources.json). These are historical
accounting evidence, not additional model features.

| Case | Verified issuer terms | Remaining integration work |
|---|---|---|
| Cielo | Trading ceases August 26, 2024. The September 23 redemption notice specifies R$5.89 per share paid September 26 | Replace the generic missing-price disposition with a dated contractual claim; preserve announcement timing, signed obligations and payment timing |
| Copel units | Last unit trading December 22, 2023; cancellation December 26; delivery of one CPLE3 plus four CPLE6 per unit, credited December 28 | Represent two successor legs and delivery timing; combine existing successor holdings without dropping inventory |
| ALLOS | ALSO3 changes to ALOS3 on October 25, 2023 | Bind dated permanent identities and verify quote/holding continuity before adding a successor link |
| ISA | TRPL3/TRPL4 change to ISAE3/ISAE4 on November 18, 2024 | Same identity and quote-continuity verification; a ticker rename is not evidence of cash redemption |
| brMalls | December 19, 2022 notices specify 0.398551577675763 ALSO share plus R$1.62899410177968 per BRML share; reference date January 6, 2023, successor trading January 9, share credit January 11 and cash payment January 20 | Represent stock plus cash with distinct trading/credit/payment dates without liquidating at a stale quote |
| Dommo | December 16, 2022 notice makes the non-electing default 0.0375 PRIO share plus R$0.4625 per DMMO share. January 6 notice specifies PRIO trading January 9, credit January 11 and default cash payment January 17 | Preserve default terms and distinguish the elective cash-only alternative; do not choose the better realized option retrospectively. Borrowed-share election obligations remain a source uncertainty |

Sources: [Cielo trading cessation](https://www.rad.cvm.gov.br/ENETWeb/frmDownloadDocumento.aspx?CodigoInstituicao=1&Tela=ext&descTipo=IPE&numProtocolo=1275808),
[Cielo redemption](https://www.rad.cvm.gov.br/ENETWeb/frmDownloadDocumento.aspx?Tela=ext&numProtocolo=1284473&descTipo=IPE&CodigoInstituicao=1),
[Copel schedule](https://api.mziq.com/mzfilemanager/v2/d/16a31b1b-5ecd-4214-a2e0-308a2393e330/a338bc96-aabd-f8e4-dc9d-3936484bff87),
[ALLOS notice](https://api.mziq.com/mzfilemanager/v2/d/330c258b-6212-45ce-8c13-557ea46cc23a/09dd3df3-2b68-6ef3-8295-a359f82c76fc),
[ISA issuer announcement](https://www.isaenergiabrasil.com.br/centro-de-midia/noticias/trpl4-e-trpl3-agora-sao-isae4-e-isae3-na-b3/).

Additional sources: [brMalls December notice](https://api.mziq.com/mzfilemanager/v2/d/330c258b-6212-45ce-8c13-557ea46cc23a/1830f410-c0a5-44b9-58c9-7c3b4bddeff9?origin=1),
[Dommo December notice reproduced by its IR provider](https://mzgroup.com.br/fatosrelevantes/fato-relevante-periodo-de-opcao-procedimentos-e-data-de-fechamento/),
[Dommo January payment notice](https://api.mziq.com/mzfilemanager/v2/d/848ef34b-7dd8-49fe-b128-ac48ffa6bf38/a180c284-337c-1a35-bbd2-6a062ffbb28b).
The [brMalls delivery notice](https://financenews.com.br/wp-content/uploads/2022/12/022357000101011.pdf)
is an archived issuer PDF mirrored by a news site; its amounts match the direct
issuer-hosted December notice. Its schedule distinguishes trading from custody credit.
The Dommo cash-only election paid R$1.90432468607 on January 13. That is not the
automatic default. An earlier general description of .05 PRIO per DMMO would also
miss the later stock/cash split; use the operative December/January notices.

[B3 circular 188/2022-PRE](https://www.b3.com.br/data/files/C6/01/2D/98/58135810F534EB48AC094EA8/OC%20188-2022%20PRE%20%20DMMO%20%28PT%29.pdf)
clarifies borrowed DMMO stock: qualifying lender cash-election requests close the
loan quantity on December 26 with the cash claim provisioned; otherwise remaining
loans convert to PRIO under the issuer's proportions, carrying cash/fractional
obligations. The general treatment is now documented. Actual historical lender
elections are still unknown and must be a labelled sensitivity where relevant.

A cached index link initially resolved to Cielo's August conversion notice rather
than September's redemption. Reading the actual PDF exposed the mismatch; the
correct September protocol is 1284473. Both original documents are retained with
correct dates. Do not treat search snippets or index titles as contractual proof.

These facts are not yet corrected replay results. In particular, the current
differentiable account supports one successor and rejects an already-held successor;
it is insufficient for the Copel basket without a tested extension. Some delivery,
borrowed-share election and intended broker cash/collateral financing terms remain
to resolve. Later-known
consideration can determine realized cash, but cannot enter earlier decisions.
All affected candidate/control accounts must use the same corrected, separately
bound contract. No raw data or sealed financial result has been overwritten.

The bounded [identity-boundary audit](v2_foundation_identity_boundaries.json) finds
consecutive old/new quote dates on separate ISIN axes for ALLOS and ISA. Neither
has a direct successor action in the accepted store. The new ALLOS axis has 60
quoted but inactive dates before becoming eligible; the ISA preferred axis has
28 by the development cutoff and is never eligible within this store. These
counts warrant checking identity-continuous warm-up as well as settlement. They
do not prove that every other universe requirement was met on those dates.
Do not repair them by relabelling every historical ticker or changing the live
fit store. Verify the dated share-class/ISIN succession first; bind any required
target/history repair separately and report its effect on the comparison.

## Remaining authorized work

1. Finish the matched input wave and its economic/IC readouts, then apply the
   registered improvement/noninferiority gates. A wide interval spanning zero is
   not evidence that deletion is harmless.
2. Score fixed checkpoint averages using compatible original inference code;
   finish EMA comparisons (the fixed cross-architecture blend is complete).
3. Test compact post-pooling readout, then matched encoder/peer contrasts and
   separate width/depth changes according to the registered conditional order.
4. Resolve and quantify the specific accounting cases; do the bounded rich-data
   residual probe on genuine earlier OOS forecasts and the retained roster.
   [Its implementation specification](../research/preregistrations/v2_foundation_residual.md)
   freezes matched score-only/rich ridge and shallow-tree cells, prior selection,
   correction strengths including zero, and equal-date weighting before new outcomes.
5. Confirm admitted candidates only, report continuous accounts and all three
   currency-consistent Sharpes, archive recovery artifacts, and close each branch
   explicitly. No new model is promoted at this interim point.

The [progress record](v2_foundation_progress.md) and [run pointer](v2_foundation_run.json)
identify the source, workers and next actions. Training continues from an immutable
source worktree while documentation and CPU readouts evolve separately.
