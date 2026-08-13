# Brazil-RV research

The research package builds the canonical feature store, trains full-universe neural models, evaluates current runs, and produces validation-only stock/time attribution.

From the repository root:

```powershell
# Full-universe incumbent TCN
uv run --project research python -m brazil_rv.modeling.train

# Isolated slow-state FiLM
uv run --project research python -m brazil_rv.modeling.train --slow-routing film --seed 29

# Validation evaluation or explicit held-out evaluation
uv run --project research python -m brazil_rv.modeling.evaluate --run-dir <run-directory>
uv run --project research python -m brazil_rv.modeling.evaluate --run-dir <run-directory> --split test

# Validation-only attribution
uv run --project research python -m brazil_rv.modeling.analyze_stock_time_attribution `
  --run-dir <run-directory> `
  --output-dir <output-directory>
```

The incumbent is the width-64 full-receptive-field SwiGLU TCN with selected peers, canonical context masking, late-only routing, soft Spearman at temperature 0.50, and SAM-AdamW at rho 0.125. Training and selection never access the held-out split.

See the repository-root [PROJECT_CONTEXT.md](../PROJECT_CONTEXT.md) for the durable current contract.
