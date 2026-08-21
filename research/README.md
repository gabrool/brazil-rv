# Brazil-RV research

The current source tree implements the accepted peer-free, full-TOD TCN with soft
Spearman as its sole objective. Rejected hybrid-loss, auxiliary-training,
recency-weight, and residual-attention implementations are absent; recorded
commits, manifests, and retained analysis artifacts preserve their history.

From the repository root:

    # Install the locked research and test environment
    uv sync --project research --group dev

    # Build and audit the peer-free causal-TOD feature store
    uv run --project research python -m brazil_rv.preprocessing.build

    # Run the two non-overlapping internal folds at seeds 11, 29, and 47
    uv run --project research python -m brazil_rv.modeling.run_discovery_campaign --output-dir <new-campaign-directory>

    # Run a lower-level aligned ensemble comparison
    uv run --project research python -m brazil_rv.modeling.analyze compare --candidate-run <run> --parent-run <run> --candidate-rule <rule> --parent-rule <rule> --output-dir <analysis-directory>

    # Confirm an already-frozen rule on the consumed official validation split
    uv run --project research python -m brazil_rv.modeling.train --selection-window official --selection-rule-file <trajectory-selection.json> --seed 11

    # Open the held-out lockbox only for a completed official run with a frozen rule
    uv run --project research python -m brazil_rv.modeling.evaluate --run-dir <official-run-directory> --split test

Every run follows one fixed 20-epoch SAM trajectory. Raw and EMA validation
predictions are recorded each epoch with EMA decays 0.98, 0.99, and 0.995.
Raw Patience-3 is the frozen primary readout and must be replayed honestly:
checkpoint selection on one odd/even discovery-date parity, reporting only on the
opposite parity in both directions. Retrospective best epoch is diagnostic only.

All future discovery-candidate drivers must call
`modeling.designated_challenger.compare_discovery_screen`. It strictly aligns
observations, uniformly rank-averages members, and emits paired bootstrap,
horizon, and time-of-day reports against both the canonical parent and the fixed
EMA residual challenger. Candidate retention is keyed only to the canonical
parent; the challenger is informational and ensemble weights are never learned.

`modeling.run_discovery_campaign` cannot request official validation or test rows.
See [PROJECT_CONTEXT.md](../PROJECT_CONTEXT.md) for the durable contract and
[RESEARCH_HANDOFF.md](../RESEARCH_HANDOFF.md) for the result and artifact record.
