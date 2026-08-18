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

# One current hybrid-loss TCN run, using a campaign-built target-scale sidecar
uv run --project research python -m brazil_rv.modeling.train `
  --seed 11 `
  --recency-policy uniform `
  --target-scale-dir <target-scale-directory>

# Resumable two-run hybrid-loss and residual-attention campaign
uv run --project research --no-default-groups --group preprocessing python -m `
  brazil_rv.modeling.run_loss_attention_campaign `
  --source-campaign-dir <completed-PIT-clean-campaign-directory> `
  --output-dir <new-campaign-directory>

# Standalone development evaluation
uv run --project research python -m brazil_rv.modeling.evaluate `
  --run-dir <run-directory>
```

The fixed recipe is a width-64, full-receptive-field SwiGLU causal TCN with
final-state readout, the masked context policy, context-plus-pooled fusion, all
three horizons, and SAM-AdamW at rho 0.125. The current objective adds a
gap-weighted pairwise term at weight 0.25 to soft Spearman. The attention arm
attends only to cross-sectionally residualized equity states; the original state
and macro/context-plus-pooled fusion remain on their existing residual routes.

See the repository-root [PROJECT_CONTEXT.md](../PROJECT_CONTEXT.md) for the
durable accepted contract.
