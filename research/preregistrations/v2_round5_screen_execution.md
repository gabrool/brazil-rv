# Round 5 CPU information-screen execution details

This fixes the otherwise unspecified screen mechanics before any Round-5 screen
outcome is read. The original five-seed, fourteen-fold, one-family-at-a-time
registration remains binding. These screens have zero network selection weight.

- Parent: `a_slow`; each candidate appends exactly one registered Group-A family,
  including the unchanged oddlot family. No intraday predictor, blend or neural
  fit. The store's value/mask/age view supplies every input; missing values are
  NaN under the same masks, not synthetic zero observations.
- Use the existing chronological development folds, their fit/selection purges
  and target-window restrictions. Match the existing CPU parent: no pretraining
  sample concatenation and no time-decay reweighting. Use unchanged `GBDTConfig`
  defaults, five independent horizon models per seed, maximum3000 rounds,
  patience100, and the existing mean-daily-Spearman selection callback. Seeds
  are11,29,47,61,79. Freeze exact configuration in the screen manifest.
- Fit magnitude clipping on each fold's actual fit rows only; apply the frozen
  0.5/99.5 percentile bounds to its selection/evaluation rows. No fit-dependent
  transform is precomputed on the whole cache.
- Report each seed and the equal rank-average five-seed ensemble. Primary
  information readout is equal-head D3/D5/D10 neutral-target Spearman on the
  common supported name population, requiring20 names. Scores exist on fold
  tails; outcomes crossing that fold's permitted target window are masked.
- Measure informative folds from input coverage alone, before fitting. A fold
  needs at least one valid active family observation in both fit and evaluation
  to contribute to the informative-fold readout. Report counts so a sparse fold
  is visible. If no fit observation exists, record the family as unlearnable in
  that fold and reuse the matched parent scores explicitly, without claiming a
  family experiment learned from later availability. Keep every fold in the
  roster, including uninformative folds.
- Paired comparisons use identical seeds, dates, names and targets. Bootstrap
  daily ensemble differences with the existing20-session blocks,10000 draws and
  frozen seed; preserve fold boundaries. Seeds are not independent market days.
- Compute gain and exact TreeSHAP for every trained seed/horizon. Bound the
  explanation set to at most16 evenly spaced eligible ISINs per evaluation day,
  using only activity/family masks, without outcomes. Record the chosen rows.
  Aggregate mean absolute contributions by field, including its age channel.
  TreeSHAP is descriptive attribution, not proof of economic causation.
- These are information screens, not another portfolio-selection sweep. Their
  readout is IC and attribution; the frozen465-book source repair and separate
  S0 continuous-book run provide this round's economic accounting readouts.
- Prepare immutable, hash-bound CPU caches once from the accepted new store.
  Cache only authorized development rows. Reuse the one parent per fold/seed
  across all families. Separate cell outputs permit verified resume after a
  process interruption. Use bounded local workers and measured memory, without
  altering model size, rounds, seeds or chronology to reduce runtime.

An efficient callback may cache unchanged date groups and target ranks only when
tested against the existing calculation. Any such implementation change must
retain the numerical selection rule and be recorded in the code binding.
