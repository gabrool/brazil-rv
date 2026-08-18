# Brazil-RV research

The current research path builds one PIT-clean causal feature store and trains one
fixed full-universe TCN. The held-out test split is never loaded during training
or campaign selection.

From the repository root:

```powershell
# Install runtime plus preprocessing dependencies
uv sync --project research --no-default-groups --group preprocessing

# Build, audit, and atomically promote the peer-free causal-TOD feature store
uv run --project research python -m brazil_rv.preprocessing.build

# One current TCN run
uv run --project research python -m brazil_rv.modeling.train `
  --seed 29 `
  --recency-policy uniform

# Resumable 21-run validation-only campaign
uv run --project research python -m brazil_rv.modeling.run_core_campaign

# Standalone evaluation; test access must be explicit
uv run --project research python -m brazil_rv.modeling.evaluate `
  --run-dir <run-directory>
uv run --project research python -m brazil_rv.modeling.evaluate `
  --run-dir <run-directory> `
  --split test
```

The fixed recipe is a width-64, full-receptive-field SwiGLU causal TCN with
final-state readout, the masked context policy, context-plus-pooled fusion, all
three horizons, soft Spearman at temperature 0.50, and SAM-AdamW at rho 0.125.
The campaign temporarily exposes recency policy and one final-state
cross-equity-attention candidate; losing branches are removed after selection.

See the repository-root [PROJECT_CONTEXT.md](../PROJECT_CONTEXT.md) for the
durable accepted contract.
