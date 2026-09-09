# Pass-5 implementation status

The registered pass is complete through Round 3: repaired store, fresh 16-book
acceptance, Round 1', two smokes, six fresh Stage-P fits, 18 Arm-B fits and nine
matched fast-off fits. All 24 Round-3 aggregate/comparator books pass the gates.
The complete 732-file GPU root is sealed and SHA-256/exact-set verified on the
host. The exact GH200 instance was terminated and its absence confirmed twice. Report publication and closure are recorded in
[the operations evidence](v2_round3_operations.json). The five-minute monitor is paused.

The user requested five-minute heartbeat monitoring while the approved launcher
waits for capacity. Its first launch attempt returned HTTP 400 with no instance
created. The launcher now preserves the structured provider explanation and
recognizes the current documented capacity-race code after inventory reconciliation;
all 13 targeted launcher checks pass. The old logger omitted the specific original
reason, so it is not inferred as a fact. See [the operational evidence](v2_lambda_launch_stop_evidence.json).

Completed evidence:

- [Round-1 rev4f replay](v2_PASS5_REBASELINE.md): 18 sealed panels, protected fields
  exact, corrected economic-beta/notional/action timing and headline gates passed.
- [Round-2 rev4f replay](v2_PASS5_ROUND2_REBASELINE.md): 12 sealed panels, protected
  fields exact. Arm B remains the network and the ensemble the overall designation.
- [R3.1](v2_R31_EXECUTION_SWEEP.md): 240 panels / 960 ledgers, zero D1-D5 flags,
  all 48 baseline daily tables exact. Adopt theta 1, D3/D5/D10, equal notional and
  buffer 9 per quintile, explicitly labelled selected in sample.
- [Intraday archive audit](v2_INTRADAY_COVERAGE.md): causal return consistency,
  completed-session action clocks, observed-support windows, true ages, and
  M1-internal to-close endpoints. Sparse archive gaps remain unknown.
- [Repaired store](v2_REBUILT_STORE.md): 3,717 sessions / 933 ISINs through
  2024-12-30, full-build peak 6.7059 GiB. Untouched arrays match the historical
  authorized slice; the endpoint-dependent audit survival flag reconstructs exactly.
  The independent 20-by-20 native-fast audit has zero value/mask error. Lending v2
  and re-derived oddlot publication masks reproduce independently. Monthly coverage
  is reported for all materialized families. The rebuilt economic-beta arrays are
  byte-identical to rev4f.
- [Fresh acceptance](v2_ACCEPTANCE_REBUILT.md): all 16 books pass and 75 legacy
  control ICs are exact. F3 inverse-volatility and reversal/momentum-blend economics
  remain unresolved under the registered terminal-settlement rule.
- [Round 1'](v2_ROUND1_PRIME.md): all 21 panels pass. Intraday-minus-slow neutral IC
  is -0.004836 [-0.009128, 0.001347]; paired net is -0.874 [-5.439, 7.458] bps/day.
  The repair has not demonstrated a GBDT improvement. Keep the registered
  `b_intraday` parent; this does not reopen rung selection.
- [Round 3](v2_ROUND3.md): retain six-seed Arm B under the registered IC rule;
  designate the network, with no economics override. Neutral IC is 0.025078
  [0.016108, 0.035333] and net excess is 4.436 [1.307, 11.653] bps/day.
  Native-fast pooled and D1 paired IC intervals span zero; it has not earned its
  place on this evidence. The six-seed extension also has no supported paired
  improvement. All four development criteria for the future 2025 read are met,
  but the read remains unspent. Round 4 requires a new registration.

The initial cleanup, including removal of retired daily-swing and GBDT code, is
included in main through commit fb6fc92. Subsequent changes remove duplicate
reader payload scans and per-row calendar construction. The latest reader and
comparison changes pass 51 targeted tests and Ruff. Earlier model/diagnostic tests
verify exact SAM updates, RNG and score bytes with logging enabled. The complete
Round-3 synthetic finalizer test covers 24 books, 27 fine-tune diagnostics, matched
P checkpoints, paired populations, comparator masks and the sealed inventory.

Historical integrity-only reader scans included later old-store ages/masks and
action/reference-price values. No held-out fitting or performance evaluation ran,
but literal earlier no-payload-access statements require this qualification.
[The disclosure and correction](v2_store_comparison_stop_evidence.json) preserve
the original evidence and sealed flags. Corrected readers verify sealed hashes
and headers, then decode payload only through date-bound access. The official
performance read-once rule remains unexercised in this pass.

The effective [Round-3 registration](../research/preregistrations/v2_round3.md)
was executed without old-store checkpoint reuse. The report includes per-horizon
ablation, six-versus-three seeds, gate activations, branch gradients, all economics
scenarios and the unexercised 2025 rule. Six trajectories ran concurrently; median
GPU utilization was 98% in pretraining and 97% in fine-tuning, including compilation
and gaps. Peak sampled memory was 68,605 of 97,871 MiB. Complete host recovery
preceded exact-ID termination and its two absence checks; their timestamps and
provider evidence are preserved in the operations record.

Failed/interrupted roots remain immutable evidence. The memory stop, slow lookup
stop and Windows Update restart were followed by a fresh successful build; no
partial arrays were reused. Raw sources and earlier sealed research roots remain
unchanged. The temporary background S3 upload command was rejected by automatic
approval review with "blocked by policy"; all transfers used the approved
launcher's SSH path and passed full hash and exact-file-set verification.
