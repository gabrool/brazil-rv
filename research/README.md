# Brazil-RV research

The current source tree builds one PIT-clean causal feature store and contains the
exact hybrid-loss/residual-attention research snapshot evaluated on 2026-08-18.
Neither candidate was promoted: the accepted incumbent remains the peer-free,
full-TOD, no-attention TCN trained with soft Spearman. The held-out test split was
not loaded during training or campaign selection.

From the repository root:

```powershell
# Install runtime plus preprocessing dependencies
uv sync --project research --no-default-groups --group preprocessing

# Build, audit, and atomically promote the peer-free causal-TOD feature store
uv run --project research python -m brazil_rv.preprocessing.build

# One unpromoted hybrid-loss research run, using a campaign-built target-scale sidecar
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

The accepted recipe is a width-64, full-receptive-field SwiGLU causal TCN with
final-state readout, the masked context policy, context-plus-pooled fusion, all
three horizons, soft Spearman at temperature 0.50, and SAM-AdamW at rho 0.125.
The checked-in experimental objective adds a gap-weighted pairwise term at weight
0.25; it reduced seed-11 IC from 0.041972 to 0.037294. Residualized equity attention
then reduced it to 0.034091. Do not treat either branch as the incumbent.

See the repository-root [PROJECT_CONTEXT.md](../PROJECT_CONTEXT.md) for the
durable accepted contract and [RESEARCH_HANDOFF.md](../RESEARCH_HANDOFF.md) for
the exact experiment history and artifact identities.
