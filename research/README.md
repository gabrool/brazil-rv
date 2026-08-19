# Brazil-RV research

The current source tree implements the accepted peer-free, full-TOD TCN with soft
Spearman as its sole objective. Rejected hybrid-loss, continuous-target sidecar,
recency-weight, and residual-attention code is intentionally absent. Their recorded
commits and immutable artifacts preserve historical reproduction.

From the repository root:

    # Install the locked research and test environment
    uv sync --project research --group dev

    # Build and audit the peer-free causal-TOD feature store
    uv run --project research python -m brazil_rv.preprocessing.build

    # Run the two non-overlapping internal folds at seeds 11, 29, and 47
    uv run --project research python -m brazil_rv.modeling.run_discovery_campaign --output-dir <new-campaign-directory>

    # Compare a candidate ensemble against its matched parent
    uv run --project research python -m brazil_rv.modeling.analyze compare --candidate-run <run> --parent-run <run> --candidate-rule <rule> --parent-rule <rule> --output-dir <analysis-directory>

    # Confirm the internally frozen rule on the consumed official validation split
    uv run --project research python -m brazil_rv.modeling.train --selection-window official --selection-rule-file <trajectory-selection.json> --seed 11

    # Open the held-out lockbox only for a completed official run with a frozen rule
    uv run --project research python -m brazil_rv.modeling.evaluate --run-dir <official-run-directory> --split test

Every run follows one fixed 20-epoch SAM trajectory. It records raw weights and
raw/EMA validation predictions at every epoch, with EMA decays 0.98, 0.99, and
0.995. It also evaluates last-3/last-5 weight averages and raw-score prediction
averages without retraining. Patience-3 and retrospective best epoch are diagnostic
only.

`modeling.run_discovery_campaign` cannot request official validation or test rows.
It freezes one rule from the mean two-fold three-seed rank-ensemble IC. The
standalone analyzer strictly aligns identities, targets, and masks; uniformly
rank-averages members; and reports paired moving-block intervals plus horizon and
time-of-day guardrails. It never learns ensemble weights.

See [PROJECT_CONTEXT.md](../PROJECT_CONTEXT.md) for the durable contract and
[RESEARCH_HANDOFF.md](../RESEARCH_HANDOFF.md) for the historical result and
artifact record.
