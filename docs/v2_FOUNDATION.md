# Model foundation experiments: combined review

2026-09-19. All registered research screens and bounded confidence studies are complete.
It is intended to be readable independently by an LLM reviewing the repository.

## Interpretation and scope audit

The five registered experimental workstreams were exercised. None produced a
candidate satisfying every predeclared screen criterion, so the ten remaining
confirmation folds were not triggered. This is completion of a bounded research
program, not proof that all data/model/accounting problems are fixed. The original
closeout also omitted the promised residual fit-versus-selection diagnostic and
retained stale progress prose; the follow-up audit below corrects that reporting.

The principal conclusions are:

1. **Do not strip out families on the earlier preliminary evidence alone.** Fresh,
   matched removal of microstructure costs .002301 IC and 1.112 net bps/day; joint
   removal of oddlot/sector-alpha/rebalance costs .003061 IC and .344 net. These
   results support keeping those groups in the current attention package. They do
   not prove that every constituent feature helps, or that the same roster is best
   for every architecture. Only the structurally empty option field was deleted.
2. **Architectural changes were not uniformly economically bad.** Matched GRU plus
   early peer attention improves average IC by .001348 and net by .669 bps/day,
   but improves only two of four folds. Wider temporal attention improves net by
   1.338 bps/day but reduces IC by .002271, outside the .001 noninferiority margin.
   Its net gain occurs in three of four folds and all three seeds, but the nominal
   95% economic interval still spans -.430 to +3.019 bps/day.
   These are inconclusive candidates under different failure mechanisms, not
   evidence of universal inferiority. Promoting either now would change the gates
   after observing outcomes. A new economically primary validation question would
   require an explicit new design and acknowledgement of this selection history.
3. **Cross-stock interaction and readout capacity have conditional value.** Removing
   early peer attention while preserving learned history pooling hurts IC and net.
   Cutting the readout width materially hurts IC, despite slightly better net.
   Thus the large readout cannot simply be dismissed as useless parameter bloat.
   These one-component contrasts are stronger evidence than comparing the whole
   C6 and attention packages, but do not identify a globally optimal architecture.
4. **Forecast diversification is more promising than the tested weight smoothing.**
   The existing three-seed attention ensemble adds .474 net bps/day over the mean
   single seed; C6 has little economic ensemble gain. The fixed C6/attention blend
   adds about .70 over either full-history fold-reset reference, but its economic
   intervals cross zero. Neither the single tested EMA horizon nor trailing
   checkpoint averaging improves enough to advance. More seeds or a smoothing
   sweep do not follow automatically.
5. **The residual probe does not identify a robust shortcut to unused rich-data
   alpha.** Rich ridge's .784 net gain over C6 is concentrated in one fold; rich
   trees improve their weaker score-only control but do not beat unchanged C6.
   This closes the fixed current-state correction recipes. It does not evaluate
   explicit release-aware histories, other fusion mechanisms or fresh training.
6. **Economic assumptions remain a more material uncertainty than many model
   deltas.** Removing assumed short-proceeds interest reduces both retained
   controls by about 3.04 bps/day in the tested screen. That is a scenario, not a
   finding about the actual broker account. Contractual settlement marks also
   differ materially for some historical books. Resolve these before treating
   headline backtest returns as reliable absolute trading returns.

| Agreed workstream | Completed work | Scope that was not established |
|---|---|---|
| Data and feature selection | Field support/overlap audit, empty-field exclusion, two matched family-removal screens | No exhaustive 145-field search or proof that every retained feature is useful |
| Component attribution and synthesis | Compact readout, matched GRU/attention, early-peer removal | No accepted synthesis; C6 comparison remains a package comparison, so the conditional five-head bridge was not run |
| Variance reduction | All singles/pairs/three-seed diagnostics, fixed model blend, 84 checkpoint averages, true EMA | No new seed sweep or universally optimal averaging recipe |
| Width/depth | Separate compact-readout, wider-encoder and shallower-trunk contrasts; review of 42 original C6 curves | No general scaling law; conditional new C6 capacity fits were not triggered by those curves |
| Existing-rich-data probe | Ridge/tree, each with score-only control and unchanged C6 anchor; 48 cell/fold/seed fits | No admitted corrector and therefore no confirmation; no new data/history mechanism |
| Confidence and delivery | Bounded primary-source event review, settlement-mark/financing sensitivities, metrics, GitHub publication and verified recovery | Full contractual settlement replay, dated identity repairs and actual broker financing remain unresolved |

The broader research decision also defers a repaired-gradient controller retry,
transfer-versus-fresh-training and explicit market/event histories. Those are
outside these five completed foundation workstreams; they must not be described
as experiments performed here. Reporting a completed screen does not settle those
questions or erase outstanding accounting work.

Completion was faster than the conservative estimate because new fits stopped
under the registered selection patience, compatible controls were reused, CPU
probes were small, and no candidate triggered a confirmation campaign. The
60-session history, eligible names, three seeds and screening folds were retained.

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
All 84 averages were prepared and evaluated. They stop at the raw selected epoch; no late overfit
tail, independent-seed weight averaging or economic continuation is substituted.
Their completed inference and matched economic results are reported below. The
stored selector IC identifies the original cutoff, not the averaged predictor's IC.
All 84 averaged graph contracts also load strictly into their original model
classes. `ops/score_foundation_averages.py` reuses a compiled graph across compatible
seeds/folds in the historical inference worktree. It ran after the fit worker;
there was only one GPU worker at a time.

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
change the immutable source of the input wave.

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

## Completed matched screens and decisions

All comparisons below use seeds 11/29/47 and F2/F6/F10/F14. Values are ensemble
changes against the matched full-input, raw-checkpoint attention control, except
where explicitly labelled C6. None passed its registered admission gate.

| Contrast | IC change | Net change, bps/day | Decision |
|---|---:|---:|---|
| Remove microstructure | -.002301 | -1.112 | Retain family |
| Remove oddlot + sector + rebalance | -.003061 | -.344 | Retain group; individual usefulness not established |
| One-epoch-half-life EMA | -.000726 | -.035 | Retain raw checkpoint |
| Readout width/inner width 256 to 128 | -.004912 | +.361 | Forecasting loss exceeds margin |
| Matched GRU encoder with early peer attention | +.001348 | +.669 | Only two of four folds improve either endpoint |
| Remove peer attention, preserve learned pooling | -.003562 | -.625 | Retain peer attention |
| Temporal width 64 to 96 | -.002271 | +1.338 | Forecasting loss exceeds margin |
| Readout depth three to two | -.002282 | -1.048 | Retain depth |

The gates were frozen before outcomes. Improvement requires at least +.001 IC or
+.25 net bps/day, neither endpoint below -.001 IC/-.25 bps, and improvement in at
least two seeds and three folds. Simplification can alternatively pass if both
paired 90% lower bounds exceed those loss margins. Averaging requires improvement.
These gates intentionally reject some positive-PnL point estimates; that is a
research decision rule, not a claim their economic potential is zero. Changing the
gates now to rescue a candidate would require a separately registered future study.
All intervals are nominal reused-development evidence, not untouched-test inference.

Fresh compatible pretraining was used for every graph/input change. The GRU bridge
preserves the rich adapters, FiLM, early peers, learned pooling, readout and three
heads; it is not the original five-head C6 package. A five-head bridge was not run
because no direct GRU-versus-C6 component claim is made. No combined width/depth
candidate, union of removed families, additional seeds or broad optimizer grid was
run. None of the new neural fits hit the 60-epoch ceiling.

Original C6 capacity was conditional on a demonstrated curve bottleneck. Among 42
original histories, one reached epoch 60; its last six selection ICs were nearly
flat (.02083 to .02106), with fit IC around .097. The remaining histories stopped
by patience. This did not trigger a C6 expansion. It does not prove C6 is optimally
sized. [Curve evidence](v2_foundation_c6_curves.json).

## Recovered training and selection diagnostics

The post-closeout audit recovered the omitted diagnostics from saved models and
histories, without fitting new weights or changing any selected checkpoint,
correction strength, outcome forecast or gate. All 48 saved residual models
reproduce their originally recorded selected IC to within 1e-9. Neural histories
match their sealed manifest hashes; the original residual feature/preprocessing,
store-access and training modules are unchanged from their fitted source commit.
The saved scalers and genuine OOS anchors are reused. Only original fit and
selection labels are read for this supplement.

[Training diagnostics](v2_foundation_training_diagnostics.json) record all 120
neural fits and 48 residual cells, selected epochs, peak allocated CUDA memory,
fit time, residual strengths/iterations, source hashes and IC gaps. Neural fit IC
is the original clean-fit probe, not a fresh full-training-set evaluation.

| Neural cell, mean of twelve F fits | Fit-probe IC at selected epoch | Selection IC at selected epoch | Fit-probe IC at final epoch | Selection IC at final epoch |
|---|---:|---:|---:|---:|
| TE_full | .045707 | .042061 | .082212 | .022789 |
| TE_compact | .050317 | .044182 | .077869 | .032644 |
| GRU_early | .049511 | .034190 | .085667 | .021439 |
| TE_wide | .074775 | .045223 | .115291 | .031026 |

Later epochs continue improving the fit probe while selection performance falls.
This supports the continued need for early selection and does not support a blanket
increase in the epoch budget. The wider encoder has a larger selected fit gap,
despite its better average evaluation economics. Selection IC is optimized on its
own dates, so a small selected fit/selection gap is not proof of healthy independent
generalization. Gaps also reflect different historical regimes, not only overfit.

| Residual cell, mean of twelve fits | Fit IC at chosen strength | Selection IC at chosen strength | Fit minus selection |
|---|---:|---:|---:|
| Score-only ridge | .016085 | .028238 | -.012152 |
| Rich ridge | .021981 | .027643 | -.005661 |
| Score-only tree | .025501 | .030490 | -.004989 |
| Rich tree | .033108 | .029857 | +.003251 |

The corresponding unchanged anchors average .017235 on fit and .026057 on selection.
Rich ridge chooses strengths 0/.1/.25/.5/1 in 1/4/1/2/4 fits respectively;
score-only ridge chooses 0/.1/1 in 5/1/6 fits. Both tree cells choose strength 1
in all twelve fits. Selected per-head tree iterations range from 1 to 232 for
rich inputs and 1 to 494 for score-only inputs. These are descriptive selection
results, not extra evaluation-based tuning. The modest residual fit gaps do not
explain away the lack of robust incremental economic gains.

## Averaging and rich-data residual outcomes

All 84 original-neutral trailing-weight averages were scored with compatible
historical inference code. On all fourteen folds, C6 changes IC by -.000123 and net
by -.071 bps/day; attention changes IC by -.000401 and net by -.060. The registered
four-fold screen also fails for both. Retain raw checkpoints; no averaging
confirmation or new averaging-rule sweep was triggered.

The residual probe used genuine per-seed C6 OOS forecasts from July 2016 onward,
including hash-verified P preludes. Parent selection labels mature before the
first prelude forecast. Fit-only accepted scalar conditioning preserves values,
masks and known ages; common scalers sample once per date. Each learner has its
own matched score-only control. Per-head neutral-rank residual targets use only
mature labels, equal total fitting weight per date, and prior selection chooses
one shared strength from 0/.1/.25/.5/1. Predictions retain every eligible name.

| Rich learner | Comparator | IC change | Net change, bps/day | Interpretation |
|---|---|---:|---:|---|
| Ridge | Unchanged C6 | -.000044 | +.784 | Fails fold consistency; gain concentrated in F14 |
| Ridge | Score-only ridge | +.002598 | +.756 | Fails fold consistency |
| Seven-leaf tree | Unchanged C6 | +.000239 | -.096 | Insufficient improvement |
| Seven-leaf tree | Score-only tree | +.001781 | +1.333 | Passes this comparator, but not unchanged C6 |

Neither learner qualifies for confirmation. Improving a weaker correction does not
establish improvement over the original model. This bounded current-state residual
probe does not settle whether different temporal representations of rich data help.
Forty-eight cell/fold/seed fits, three heads each, and 64 matched books completed.
Readout startup failures (store versus cache indices, missing policy metadata and
an incomplete output directory) were corrected and retained in logs. Fitted models
and predictions were unchanged; calendar/population checks then passed.

## Comparability, performance and the earlier six-bps result

[Combined metrics](v2_foundation_combined_metrics.json) contain all thirteen screen
ensembles: three Sharpes, turnover, exposure, win/loss rates and drawdown. Brazilian
Sharpe subtracts CDI from BRL returns; US Sharpe converts returns to USD using
historical PTAX then subtracts calendar-accrued EFFR; zero-rate Sharpe uses BRL
absolute returns. PTAX valuation is not an executable FX-fill assumption.

The four screen folds are noncontiguous, and each account starts in cash. C6's
screen net is .657 bps/day and fresh TE_full's .400. Their concatenated drawdown
and compounded returns describe screened sessions, not continuous 2018–2024.
Do not compare those directly to the earlier all-history four/six-bps numbers.

| Earlier continuous 2018–2024 book | C6 net bps/day | Attention net bps/day |
|---|---:|---:|
| Tight neutral, 5% net cap | 4.000 | 4.497 |
| Flexible net, 45% cap; same 5% beta cap | 6.707 | 5.937 |

Those use earlier sealed forecasts and continuous inventory. Flexible books were
persistently net long, not evidence of learned confidence timing. They remain
research candidates with financing/settlement limitations, not discarded results.
The original all-fourteen-fold reset ensembles earn 4.245/4.258; the fixed equal
forecast blend earns 4.948. Policy, dates, forecast generation and account continuity
must accompany any future headline result.

## Confidence limits and accounting sensitivity

Matched four-fold retained-control replays quantify two independent scenarios:
zero interest on short-sale proceeds changes C6/TE_full net by -3.035/-3.041 bps/day;
a 3% annual debit spread changes them by -.105/-.101. These are scenarios, not
verified broker terms. [Financing evidence](v2_foundation_financing_results.json).

[Settlement exposure](v2_foundation_settlement_exposure.json) records actual synthetic
settlement quantities. [Contractual mark sensitivity](v2_foundation_settlement_marks.json)
compares four verified event terms with the generic marks on those original
quantities, using exact same-session successor quotes. Earlier continuous C6
neutral/flexible nominal differences are +164.911/+250.329 bps of INITIAL NAV;
attention's are +69.433/+261.966. These are cumulative valuation differences, not
daily returns, corrected PnL, or asserted successor sales.

This bounded study has NOT implemented a corrected contractual settlement ledger.
Payment financing, delivery timing, changed inventory/policy paths, loan elections,
other events and dated ALLOS/ISA identity succession remain unresolved. Copel needs
two successor legs; the current account cannot represent that or merge a successor
already held. The source store and historical targets were not silently repaired.
No model is deployment-ready, and absolute economic confidence remains conditional.
These limitations should precede live implementation or strong absolute-PnL claims.

## Runtime, provenance and recovery

| Neural wave | Fits including parents | Summed fit time |
|---|---:|---:|
| Inputs | 45 | 102.2 min |
| Compact readout | 15 | 40.8 min |
| Encoder | 15 | 81.4 min |
| Peer removal | 15 | 22.4 min |
| Separate capacity contrasts | 30 | 73.6 min |

Total: 120 new fits, 1,484 epochs, about 5.34 hours of summed fit time, plus six
disposable engineering checks, score export, preprocessing, readouts and review.
Wall time spans successive registered waves; fit time is not total completion time.
The matched GRU was slower than temporal attention on this RTX 2060 implementation.
Full 60-session histories and all eligible names were preserved throughout.

The saved F manifests also permit the promised resource/selection comparison
without repeating GPU work. Each row contains the same four folds and three seeds;
fit time includes that run's training work and is not a pure per-step benchmark.
Peak memory is allocated CUDA tensor memory, not total device usage.

| Cell | Mean selected F epoch | Mean F fit seconds | Maximum allocated GiB |
|---|---:|---:|---:|
| TE_full | 4.42 | 122.4 | 2.27 |
| TE_no_micro | 6.42 | 132.4 | 2.26 |
| TE_no_weak | 6.58 | 146.5 | 2.23 |
| TE_compact | 9.50 | 144.6 | 2.23 |
| GRU_early | 5.50 | 271.0 | 2.88 |
| TE_no_peer | 5.58 | 83.8 | 1.79 |
| TE_wide | 7.67 | 164.4 | 2.74 |
| TE_shallow | 4.92 | 121.9 | 2.26 |

Fewer parameters therefore did not automatically make a complete selected fit
faster: compact readout trained longer on average. Omitting early peers was faster
but lost forecasting/economic performance. These measurements are conditional on
the present implementation, optimizer and selection rule, not universal complexity
claims about GRUs or attention.

Initial training source d21a93a stayed immutable. Later graphs used 3dbda71;
base readouts used 75a5b75; residual fitting used 19872de; corrected residual
readouts used 81ba9c9; financing used 9cda3ae. Exact artifact hashes and worker
receipts are under the [canonical run](v2_foundation_run.json). Main report changes
do not rebind completed fits. Tests covered EMA/resume invariants, pathway masks
and pooling, residual date weighting, intercept and age semantics; actual runs
checked accepted-store, source, score identities and eligible populations.

No neural, averaging or residual candidate qualified for new confirmation, so no
confirmation fits were run. Preserve the current reference models, full cleaned
roster and raw selection. The most defensible positive findings are existing seed
ensembling and forecast diversity; neither alone establishes a deployable advance.
Recovery verified: 9,755 files, 12.44 GB source content, 10.49 GB compressed on D. Every decompressed member matched its SHA-256 and size; immutable source datasets were not duplicated. The archive includes a verified Git bundle. See [recovery receipt](v2_foundation_recovery.json). The registered bounded program is complete; unresolved full contractual accounting is a stated limitation, not a completed repair.

The subsequent reporting correction, saved-model diagnostics and scope assessment
are preserved in the separate reporting addendum listed in that receipt. The
original 9,755-member archive and experimental outcomes remain unchanged.


