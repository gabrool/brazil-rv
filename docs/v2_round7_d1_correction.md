# Round 7 D1 diagnostic correction

The GL/F14 three-seed extension replay stopped because D1 counted a pending printed entry whose entry_fill_allowed mask explicitly prohibited filling. The execution loop correctly respected the corporate-action restriction; the diagnostic incorrectly called it unblocked.

The correction excludes explicitly prohibited entries from this diagnostic count. It changes neither orders nor fills nor restrictions. All 77 ledger tests and Ruff pass, including the extended existing opening-restriction/printed-exit test.

An isolated replay of the exact GL/F14 ensemble changed only 15 report paths containing D1_entry_pending_printed_unblocked_unfilled, each from one to zero. Every other report field matched exactly. See v2_round7_d1_replay_comparison.json for the complete recursive difference and corrected source SHA. This supports a diagnostic correction, not an execution-rule waiver. The original failed aggregate is retained as diagnostic_stop_GL_F14 in the extension root.

Both GPU checkouts remain frozen. Until compute finishes, readouts use an explicit external harness loading this hash-bound corrected ledger before importing the readout module. The final merged implementation incorporates the correction normally. Previously completed zero-D1 readouts remain reusable because the correction only removes explicitly blocked entries from that counter; no economics are modified. Apply the corrected implementation consistently to subsequent original and extension readouts. No model training is repeated.
