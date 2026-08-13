# Brazil-RV research

The research package builds the current feature store, trains full-universe models, evaluates current-format runs, and produces validation-only stock/time attribution.

From the repository root:

```powershell
# Full-universe incumbent TCN
uv run --project research python -m brazil_rv.modeling.train

# Isolated slow-state FiLM routing alternative
uv run --project research python -m brazil_rv.modeling.train --slow-routing film --seed 29

# Evaluate validation (default) or explicitly open held-out test
uv run --project research python -m brazil_rv.modeling.evaluate --run-dir <run-directory>
uv run --project research python -m brazil_rv.modeling.evaluate --run-dir <run-directory> --split test

# Validation-only attribution
uv run --project research python -m brazil_rv.modeling.analyze_stock_time_attribution `
  --run-dir <run-directory> `
  --output-dir <output-directory>
```

The incumbent is the width-64 full-receptive-field SwiGLU TCN with selected peers, direct current context masking, late-only routing, soft Spearman at temperature 0.50, and SAM-AdamW at rho 0.125. Training and selection never access the held-out test split.

See `C:\Brazil-RV\PROJECT_CONTEXT.md` for the durable data, causality, split, and model contracts.
