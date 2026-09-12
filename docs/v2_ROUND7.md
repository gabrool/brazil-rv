# Round 7 — implementation, data repairs and experiment results

Status: CPU preflight accepted; new neural experiments have not started. This
document will retain the calibration, all screening cells, confirmation and final
decision when those stages finish. The registered development boundary remains
2024-12-30. No forward capture or 2025/2026 consumer read has occurred.

## Research question and attribution

Round 7 tests whether a joint characteristic model and a better training recipe
can use the available information more effectively. Weak incremental Round-6
results do not prove that architecture was the only limitation. In particular,
univariate relevance and a tree's feature gain do not guarantee stable incremental
portfolio alpha after existing signals, sampling error and costs.

The exact supplied registration is [v2_round7.md](../research/preregistrations/v2_round7.md).
[Implementation resolutions](../research/preregistrations/v2_round7_implementation.md)
make ambiguous counts, clocks, input sharing and calibration choices explicit.
The [machine protocol](../research/preregistrations/v2_round7.json) enumerates the
15 cells and their parameter counts. A0 is a fresh repaired-store S0 under the old
recipe. A1 tests the new recipe on S0. A3 and B4 compare joint inputs through old
linear sidecars and the new characteristic graph. B1 versus A1 also changes five
training heads to three; B7 measures that component within C1.

## Accepted data repairs

[Input acceptance and exact evidence](v2_round7_inputs.json) binds the store
`v2_round7_store_de6476b_20260912T232200Z`, manifest
`1db29cbf7b30244b2327ab7e0c212a796fd25280141a45e4ee28213b2644aa65`.
The build used commit `e609c9f`; the directory label is not the build-code identity.
Peak resident memory was 6.413 GiB, below the 8-GiB build invariant.

Of 470 original U2 candidates, 438 lack the registered unit-action corroboration
and are reclassified as large moves without an inferred conversion. Thirty-two
retain corroboration. The complete event list preserves every accepted/rejected
candidate and its source dates. Existing C1 cash terms remain exact. This is still
development-grade action inference: nearby filing metadata does not establish an
exact contractual ratio or effective date.

The parser recovers 32,530 BDI 06/07/08 quotes for 30 established permanent
identities. These prints restore marks and exits. A separate execution-time fill
mask blocks new entries on recovered prints without changing earlier intended
orders, point-in-time membership, cross-sectional ranks or historical features.
Only 47 recovered quote rows overlap the old active mask. Historical decision-time
wealth, slow features, risk inputs and common features remain protected; retrospective
outcome repair does not import later corroboration into a past decision.

Target reconstruction restores original cent-grid prices and original-precision
action factors before making the repair. Twenty-four evenly spaced source dates
reproduce all twelve original target arrays exactly. Subsequent target changes
are restricted to repair-affected date/horizon windows, including cross-sectional
rank propagation to peers. The acceptance JSON provides all target counts by fold.

All fourteen sealed three-seed S0 books were replayed before and after. Every
before-repair daily table and summary exactly matches its sealed Round-6 reference;
all 28 books pass the engineering gates. Mean net excess changes from **4.5082 to
4.2910 bps/day**, a paired change of **−0.2172 bps/day**, with signals held fixed.
This measures accounting, not retrained-model performance. The continued quotes
touch a SEQL3 short across ten position-days in October 2024: actual marks replace
the last pre-recovery price, and the repaired book exits earlier.

The foreign-flow audit did not establish an earlier reliable publication clock.
PDF generation timestamps often reflect later regeneration; available historical
capture probes did not resolve the uncertainty. The existing publication-date
bound is retained, without an extra lag. Settlement timing and publication timing
are different facts.

## Meaningful data and overly restrictive support rules

The new nine-field native fundamentals family retains physical values, signs,
receipt age and validity. It is not cross-sectionally ranked. Median/IQR scaling
and ±5 clipping use the fit window only; constant fields retain unit scale and
missing values remain explicitly masked. No twenty-name requirement applies.

| Native field | Valid active stock-days | Previously lost to rank support |
| --- | ---: | ---: |
| Signed earnings yield | 133,703 | 17,150 |
| Log positive book-to-market | 136,459 | 19,170 |
| Gross profitability | 276,620 | 477 |
| Liabilities/assets | 303,976 | 221 |
| Accruals/assets | 263,036 | 182 |
| Signed revenue growth | 240,758 | 702 |
| SUE in native standardized units | 236,809 | 84 |
| Log market capitalization | 140,339 | 19,354 |
| Earnings-negative flag | 133,703 | 17,150 |

Counts are field-observations; they are not additive unique stock-days. The
negative-earnings flag inherits the earnings-yield observation population.

The broader audit finds additional sparse-support losses: options 42,538;
sector 34,417; lending 9,747; events 492 field-observations. Cross-market,
magnitudes, microstructure and rebalance lose none at this boundary. A separate
reconstruction of the original odd-lot archive confirms zero losses for its
568,093 share observations and 567,460 exact-five-session changes, with the
publication lag reproduced.

These other ranked inputs retain their registered transforms in the architecture
factorial. Their physical source observations remain available, and the losses
are recorded for a separate representation comparison rather than being treated
as unusable data. Rank support is distinct from source validity, identity and
historical availability. No family is removed from a joint-input model because
of a weak individual screening result or sparse pretraining support.

## Architecture, training and efficiency

C1 keeps the full 60-session, 32-field width-64 GRU; adds a latest-row core MLP,
nonlinear family encoders, a 256-wide joint state, FiLM market conditioning,
cross-sectional context, three residual SwiGLU blocks and D3/D5/D10 heads.
All 44 common fields are retained, with fit-only scaling on unique dates.
Attention uses four heads of width 32. TabM's eight members share the expensive
history/family encoders and expand only at the residual trunk.

| Graph/input example | Parameters |
| --- | ---: |
| S0 slow | 120,902 |
| S0 all families | 166,982 |
| C1 slow | 1,297,475 |
| C1 all families | 1,471,651 |
| C1 all, no GRU | 1,421,923 |
| C1 all, five heads | 1,472,165 |
| C1 attention | 1,013,283 |
| C1 TabM | 1,502,371 |

Recipe R uses fixed unique-date epochs, a schedule defined on the actual budget,
5% warmup and cosine decay, no bias/LayerNorm decay, SAM with the registered radii,
and uniform averaging of the last ceil(B/4) epoch-end states. The SAM second pass
reuses dropout randomness and restores original weights exactly. TabM losses rank
stocks separately within each date/member/head. B11 is genuinely single-pass AdamW.

Training compiles the model and FP32 ranking loss together. GPU autocast is BF16;
parameters, optimizer state, loss and relevant moments remain FP32. Compact name
axes include every active name; there is no top-N selection or shortened history.
Balanced batches include every date exactly once and avoid singleton tails.
Scoring reuses the compiled evaluation model. Epoch-end weights, daily selection
IC, optimizer state, tail weights and RNG state support exact interrupted-run recovery.

CPU engineering covers ten graph/loss combinations. Each captured one full training
graph and one evaluation graph and learned a permitted-input synthetic target to
IC above 0.5 within 10–20 updates. The all-family cases separately learned a masked
product of two family fields plus a native-profitability threshold. The eight
all-family cases all passed in ten updates. These are implementation tests, not
evidence of predictive alpha, GPU throughput or out-of-sample generalization.
CPU uses Dynamo's eager backend; GH200 Inductor/BF16 and full-batch throughput
acceptance remain required before experiment dispatch.

Targeted tests also cover name permutation and padding, invalid-family masking,
fit-only scaling, member-loss independence, exact SAM restore, and three-head
scoring without fabricated D1/D2 predictions. An end-to-end synthetic-store test
interrupts training after a durable epoch and verifies exactly identical final
weights and score arrays after resumption. Three-head and five-head candidates
pair correctly on the same D3/D5/D10 population.

## Diagnostics and remaining experiment stages

All **84 archived diagnostics are complete**. The [full per-fit report](v2_round7_archived_diagnostics.json)
binds the saved weights and daily series. Both checkpoints use CPU FP32 and the
same D3/D5/D10 population. These are means of seed ICs, not rank-ensemble ICs.

| Diagnostic | S0 | Fundamentals |
| --- | ---: | ---: |
| Mean clean fit raw IC | 0.04431 | 0.04897 |
| Mean selection raw IC | 0.03843 | 0.04816 |
| Evaluation mean-seed raw IC | 0.02552 | 0.02764 |
| Evaluation mean-seed EMA IC | 0.02717 | 0.02759 |
| Paired evaluation EMA minus raw | +0.00165 | −0.00005 |
| Newey–West mean SE of that difference | 0.00075 | 0.00117 |
| Median selection paired mean SE | 0.00141 | 0.00261 |
| Mean shared D1/D2 versus D3/D5/D10 gradient cosine | 0.505 | 0.440 |
| Fits with negative combined gradient cosine | 0 / 42 | 0 / 42 |
| Median fixed-batch SAM loss gap, rho 0.125 | 0.03637 | 0.03348 |

The clean fit/selection gaps do not show severe fitting of the training set in
these two saved states, but cannot establish underfitting: selection chose the
raw checkpoint, and the windows cover different dates. Old per-epoch weights
are unavailable. Forty-five common-label dates remain in each nominal 55-session
selection window after the D10 boundary rule. Paired checkpoint uncertainty is
material relative to the incremental gains under investigation.

EMA helps the archived S0 comparison but not the fundamentals comparison. Its
checkpoint time and retained initialization weight differ, so this is not a
causal estimate of averaging alone. The fixed-batch head gradients are positively
aligned in every fit; that does not establish that shorter-horizon heads help
out-of-sample. Actual fundamentals sensitivity is retained per fit alongside
the gradient matrices. These diagnostics inform interpretation and do not alter
the registered factorial or select a new checkpoint.

The fixed chronological residual-tree control is complete on thirteen folds and
1,485 scored dates. Its composite IC is **0.03052 for S0**, **0.01413 for the
score-only corrector**, and **0.01857 for the all-family corrector**. The all-family
corrector minus S0 is **−0.01195**, Newey–West mean SE **0.00577**. Its gain over
the matched score-only tree is **+0.00443**, SE **0.00441**. This control does not
improve S0. The modest, uncertain gain over its own tree control neither establishes
an architectural bottleneck nor proves that the families contain no useful signal.
No hyperparameter or checkpoint was selected from these evaluation results.
This diagnostic ranks the composite against mean traded-horizon rank; it is not
the headline mean of separate D3/D5/D10 ICs and has zero promotion weight.

There are **24 initial Stage-P fits**, not the draft's approximate fifteen: graph,
input roster and head count must match. S0 slow uses its three old-recipe parents;
seven other compatible graph/input contracts need three each. The repaired anchor
has 42 F fits. Twelve calibration fits choose B before any screen score. Screening
adds 168 F fits, or 156 if B=60 allows exact calibration reuse. Confirmation reuses
all screen fits and adds at most 150 F fits for five candidates. The six-seed stage
adds seeds 61/79/97 for the fourteen-fold leader and A0, with compatible parents.

GPU launch, measured concurrency, calibration B, screen/confirmation outcomes,
three-borrow economics, the continuous book, final designation and verified
artifact recovery/instance shutdown are pending. No improvement or promotion is
claimed from the implementation and preflight work alone.

The readout path preserves original-rate economics for model selection and reuses
that accepted ledger for two rate-only sensitivity comparisons. Leave-one-seed-out
forecast stability uses the completed six-seed scores; it does not rerun identical
economic books for a forecast-only gate. The final continuous book carries positions
across model switches and has no influence on the completed designation.
