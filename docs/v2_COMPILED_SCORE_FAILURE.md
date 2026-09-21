# Compiled forecast failure and qualified correction

Current disposition: stage_d_compiler_acceptance binds the correction and bounded
scope evidence. The canonical width plan now selects capacity_width_recovery;
the original plan/evaluation pointers and every original fit remain preserved.
Only GRU96 results are replaced; C and TE128 remain usable under the scope below.
Earlier investigation statements describe the attempts, not current pending work.

Stage D width execution stopped during score export for GRU_96/F10/29 after
nine optimizer epochs. Twenty earlier width fits have completion manifests;
the failed fit retains its selected checkpoint, all nine epoch checkpoints and
resume state. No further width/depth/LSTM fit is running. Original outputs remain
immutable. The one-minute heartbeat remains paused; implementation continues.

A fresh export of the failed checkpoint is finite. Fresh compiled/eager exports
agree within the frozen numerical tolerance on all78,309 valid scores. However,
a completed same-width/fold control (seed11) fails fresh-versus-saved export
reproduction:76,210 of78,309 values exceed rtol.02/atol.002, maximum6.638671875,
RMS.91614484787. This is materially larger than Float16 rounding. The attempted
score-resume qualification failed and no fresh forecast was installed into an
original fit.

The actual two-date/full-population/full60 control reproduces the interaction:
checkpoint reload, no-op device application and an eager training-mode forward
preserve predictions, but an isolated fit-only optimizer update followed by exact
weight restoration changes the reused compiled output (maximum3.0263671875).
The eager output remains exact, and parameter values and addresses remain exact.
The full restored diagnostic also exposes it; temporary hooks are not required.

Changing only initial contents of compiler temporary allocations changes the
result: NaN initialization leaves only288 of1536 padded output cells finite;
zero/one initialization changes outputs by up to1.6587524414/1.9633789063. All
checkpoint/input bytes remain fixed. This demonstrates a compiled calculation
depending on uninitialized temporary memory. The specific generated operation
and correction are resolved below; training/selection and previous-result scope
remain under investigation.

Saved Stage C economic/model conclusions and unfinished D comparisons are
provisional pending this scope check. This does not revoke the independent source,
data-store or account-arithmetic proofs or authorize their repetition. No model
adoption or new capacity admission follows from the affected results.

Evidence is under stage_d_width_original_plan's directory, `scoring_failure`:
original failure stdout/runtime, fresh exports, failed qualification, saved-only
comparisons, exact executed probe recipes, allocation controls and generated code.
The first allocation inspection used a nonexistent installed PyCodeCache.cache
attribute; its exact failed recipe/stdout remain, and the qualified invocation
uses the installed PyCodeCache.modules API. Initial shell/path/patch attempts are
tooling history, not falsely represented as exact saved execution files.

Controlled allocation tracing initializes every other temporary to zero before
each contrast, eliminating poisoned allocator contents carried between calls.
Its13 contrasts isolate allocation0:10461 in generated module SHA
90501f37fd6e5a0eb0aa0995e7fa2c828ba16858fc573ca277914aa34c64ed3b:
the fused peer residual LayerNorm reduction. The earlier63-test trace did not
control allocator carryover; its apparent21 sites and joint interactions are not
qualified independent defects. Both recipes and complete results remain.

An audit-only synchronization barrier between the reduction's temporary stores
and later loads removes all NaN/zero/one allocation dependence. All1536 outputs
are finite and each variant agrees with eager within the original Float16
tolerance (maximum.00244140625). Disabling in-place buffers alone fails and is
not the implemented fix. Generated kernels and these unsuccessful/successful
contrasts remain evidence, not patched research runtime dependencies.

The maintained correction in v2/train.py routes native_layer_norm and its backward
to ATen's existing kernels within the compiled graph. No package, model equation,
parameter, optimizer, selector or data coordinate changes. It passes the actual
same-checkpoint full-population/full60 allocation controls: all NaN/zero/one
outputs finite and identical across allocation contents, maximum.002197265625
versus eager. One new GPU regression passes forward/backward arithmetic and
allocator poisoning in5.68seconds. Its harmless configuration-serialization
warning is retained. Previous model fits and outputs have not been changed.

The frozen architecture-scope probe compares six F10/seed11 checkpoints:
four C graphs plus both D widths. Each uses all126 original evaluation dates for
eager-versus-saved forecasts and its first actual compiled export batch for
allocator poisoning. This is new verification of the exposed compiler failure;
it does not repeat source, account, history or training campaigns. Other fits and
training/selection graphs require a scope decision from the resulting evidence.

The six-case scope finished in166.572559seconds summed case time. All five other
graphs reproduce every saved126-date F10/11 forecast within frozen rtol.02/atol.002
and remain finite under first-batch NaN allocation poisoning. GRU96 has76,222 of
78,309 saved scores outside tolerance (maximum6.638671875); all9,861 valid poisoned
first-batch scores become nonfinite. The earlier fresh-export comparison counted
76,210 because its separate invocation had slightly different Float16 outputs;
both exact observations are retained, not combined as independent evidence.

The original GRU96 P/11 first16 selection dates, padded160/full60, pass all
NaN/zero/one controls on5,472 valid predictions. Its generated graph nevertheless
contains the same unsafe reduction. Fresh P parents are a conservative scope
decision; no observed P selector mismatch or training-gradient defect is claimed.
The qualified isolated runtime also passes that original parent batch with all
allocation variants identical, maximum.0048828125 versus eager. The separate
new GPU regression checks normalization forward and gradients. No diagnostic
optimizer result is adopted as a new model fit.

The replacement plan preserves the registered two cells, three seeds/four folds,
full history/population, maximum epochs, optimizer and selector. Ten completed
TE128 fits and five remaining TE128 jobs use original runtime3b0f921. Fifteen
GRU96 P/F jobs use normalization@308007a, whose only change from3b0f921 is the
12-line compiler normalization fix. All ten completed original GRU96 fits and
nine failed F10/29 epochs remain excluded evidence. The score-only resume recipe
is retained as rejected evidence and removed from active ops.

The composed evaluation references12 saved primary,132 ordinary and12 funded
TE128 books;78 ordinary and6 fraction-precision skips are retained. Their exact
qualification receipts are copied, with no portfolio or arithmetic replay.
No original GRU96 book enters corrected progress. The declared TE128 directory
junction prevents duplicate fit storage. Resolve stage_d_width_attention_book_reuse.

Read-only inspection once used a nonnumeric PowerShell First argument and later
large output was truncated. These changed no research artifacts; they are
reconstructed tooling descriptions, not saved exact failed execution files.

Future depth/LSTM runtime checkouts a9a775c/67629a7 apply the same12-line compiler
fix to their previously qualified graphs. Their freezer now explicitly binds and
checks the compiler bytes. Both idle code checkouts received lossless NTFS LZX
compression:82,428,016/82,475,858 logical bytes occupy17,507,682/17,534,021bytes;
Git reports both clean. No source/store/fit data was removed or changed. The
capacity archive includes exact per-arm runtime sources and the entire original
failed width root, while skipping its declared duplicate junction.
