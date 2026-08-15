from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from brazil_rv.preprocessing.contract import GLOBAL_SLOW_CHANNELS

from .context_routing import align_macro_histories, causal_patch_mask
from .contract import (
    CONTEXT_COUNT,
    CONTEXT_GENERIC_DYNAMIC_COUNT,
    CONTEXT_ROUTING_EXCLUDED_GLOBAL_SLOW_CHANNEL,
    CONTEXT_ROUTING_MACRO_EARLY_SOURCE_WIDTH,
    CONTEXT_ROUTING_MACRO_SLOW_INPUT_WIDTH,
    CONTEXT_ROUTING_PATCH_SOURCE_WIDTH,
    CONTEXT_ROUTING_SOURCE_COUNT,
    DYNAMIC_CHANNEL_COUNT,
    EQUITY_COUNT,
    HORIZON_COUNT,
    LOCAL_CONTEXT_COUNT,
    MLP_DEPTH,
    MLP_SWIGLU_WIDTH,
    MLP_WIDTH,
    PATCH_MINUTES,
    PEER_STATE_WIDTH,
    RESIDUAL_DROPOUT,
    RMS_NORM_EPS,
    TABULAR_FEATURE_COUNT,
    TARGETED_FUSION_GATE_BIAS,
    STATE_TOKEN_SLOT,
    TCNArchitecture,
    routing_enabled,
    validate_peer_feature_mode,
)
from .layers import CausalTCNResidualBlock, MuonLinear, SwiGLU


_GLOBAL_SLOW_EXCLUDED_INDEX = GLOBAL_SLOW_CHANNELS.index(
    CONTEXT_ROUTING_EXCLUDED_GLOBAL_SLOW_CHANNEL
)


def apply_context_film(
    hidden: torch.Tensor, gamma_total: torch.Tensor, beta_total: torch.Tensor
) -> torch.Tensor:
    return (1.0 + torch.tanh(gamma_total)) * hidden + beta_total


class _ContextRouting(nn.Module):
    def __init__(self, architecture: TCNArchitecture) -> None:
        super().__init__()
        width = architecture.width
        rank = architecture.context_routing_rank
        slow = architecture.slow_routing
        macro = architecture.macro_temporal_routing
        slow_enabled = slow != "late_only"
        self.equity_slow_projection = (
            nn.Linear(architecture.slow_width, width, bias=False)
            if slow_enabled
            else None
        )
        self.macro_slow_projection = (
            nn.Linear(CONTEXT_ROUTING_MACRO_SLOW_INPUT_WIDTH, width, bias=False)
            if slow_enabled
            else None
        )
        self.slow_condition_norm = nn.LayerNorm(width) if slow_enabled else None
        self.slow_early_input_adapter = (
            nn.Linear(width, width, bias=False)
            if slow in ("early_concat", "early_concat_film")
            else None
        )
        self.slow_film = (
            nn.ModuleList(
                [
                    nn.Linear(width, 2 * width, bias=False)
                    for _ in range(architecture.residual_blocks)
                ]
            )
            if slow in ("film", "early_concat_film")
            else None
        )
        if macro in ("early_concat", "early_concat_film"):
            self.macro_early_input_projection = nn.Linear(
                CONTEXT_ROUTING_SOURCE_COUNT * CONTEXT_ROUTING_MACRO_EARLY_SOURCE_WIDTH,
                width,
                bias=False,
            )
            self.macro_early_activation = nn.SiLU()
            self.macro_early_norm = nn.LayerNorm(width, elementwise_affine=False)
            self.macro_early_output_projection = nn.Linear(width, width, bias=False)
        else:
            self.macro_early_input_projection = None
            self.macro_early_activation = None
            self.macro_early_norm = None
            self.macro_early_output_projection = None
        self.macro_film = (
            nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(
                            CONTEXT_ROUTING_SOURCE_COUNT * width, rank, bias=False
                        ),
                        nn.SiLU(),
                        nn.Linear(rank, 2 * width, bias=False),
                    )
                    for _ in range(architecture.residual_blocks)
                ]
            )
            if macro in ("film", "early_concat_film")
            else None
        )

    def zero_final_projections(self) -> None:
        if self.slow_early_input_adapter is not None:
            nn.init.zeros_(self.slow_early_input_adapter.weight)
        if self.macro_early_output_projection is not None:
            nn.init.zeros_(self.macro_early_output_projection.weight)
        if self.slow_film is not None:
            for layer in self.slow_film:
                nn.init.zeros_(layer.weight)
        if self.macro_film is not None:
            for layer in self.macro_film:
                nn.init.zeros_(layer[-1].weight)


class SharedCausalTCN(nn.Module):
    model_name = "tcn"

    def __init__(
        self,
        architecture: TCNArchitecture,
        peer_features: str = "none",
        equity_count: int = EQUITY_COUNT,
    ) -> None:
        super().__init__()
        if equity_count <= 0:
            raise ValueError("equity_count must be positive")
        self.architecture = architecture
        self.equity_count = equity_count
        self.instrument_count = equity_count + CONTEXT_COUNT
        self.peer_features = validate_peer_feature_mode(self.model_name, peer_features)
        width = architecture.width
        self.input_projection = nn.Linear(
            architecture.patch_input_width, width, bias=False
        )
        self.blocks = nn.ModuleList(
            [
                CausalTCNResidualBlock(
                    width,
                    architecture.kernel_size,
                    dilation,
                    architecture.dropout,
                    architecture.block,
                    architecture.swiglu_hidden_width,
                )
                for dilation in architecture.dilations
            ]
        )
        self.slow_projection = nn.Linear(architecture.slow_width, width, bias=False)
        self.state_norm = nn.LayerNorm(width)
        if architecture.fusion_mode != "none":
            self.fusion_input = nn.Linear(
                architecture.fusion_states * width,
                architecture.fusion_width,
                bias=True,
            )
            self.fusion_output = MuonLinear(
                architecture.fusion_width, width, bias=False
            )
            self.fusion_gate = nn.Linear(2 * width, width, bias=True)
            self.fusion_norm = nn.LayerNorm(width)
        self.dropout = nn.Dropout(architecture.dropout)
        self.prediction_head = nn.Linear(width, architecture.output_horizons, bias=True)
        self.apply(self._initialize_module)
        if architecture.fusion_mode != "none":
            nn.init.zeros_(self.fusion_gate.weight)
            nn.init.constant_(self.fusion_gate.bias, TARGETED_FUSION_GATE_BIAS)
        self.peer_adapter: nn.Linear | None = None
        if self.peer_features != "none":
            with torch.random.fork_rng(devices=[]):
                self.peer_adapter = nn.Linear(PEER_STATE_WIDTH, width, bias=False)
                nn.init.zeros_(self.peer_adapter.weight)

        self.routing: _ContextRouting | None = None
        if routing_enabled(architecture):
            with torch.random.fork_rng(devices=[]):
                self.routing = _ContextRouting(architecture)
                self.routing.apply(self._initialize_module)
                self.routing.zero_final_projections()

        self.scale_logits: nn.Parameter | None = None
        if architecture.readout == "shared_multiscale":
            self.scale_logits = nn.Parameter(torch.zeros(architecture.residual_blocks))
        elif architecture.readout == "horizon_multiscale":
            self.scale_logits = nn.Parameter(
                torch.zeros(architecture.output_horizons, architecture.residual_blocks)
            )
        self.score_mlp: nn.Sequential | None = None
        if architecture.readout == "final_score_mlp":
            with torch.random.fork_rng(devices=[]):
                self.score_mlp = nn.Sequential(
                    nn.Linear(architecture.output_horizons, 2, bias=True),
                    nn.SiLU(),
                    nn.Linear(2, architecture.output_horizons, bias=True),
                )
                self.score_mlp.apply(self._initialize_module)
                nn.init.zeros_(self.score_mlp[-1].weight)
                nn.init.zeros_(self.score_mlp[-1].bias)

    @staticmethod
    def _initialize_module(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            if module.elementwise_affine:
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    @staticmethod
    def _route_has_early(route: str) -> bool:
        return route in ("early_concat", "early_concat_film")

    @staticmethod
    def _route_has_film(route: str) -> bool:
        return route in ("film", "early_concat_film")

    def _macro_sources(
        self,
        values: torch.Tensor,
        masks: torch.Tensor,
        state_position: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        local_start = self.equity_count + 1
        local_stop = self.equity_count + LOCAL_CONTEXT_COUNT
        global_start = self.equity_count + LOCAL_CONTEXT_COUNT + 2
        global_stop = global_start + 2
        return align_macro_histories(
            values[:, local_start:local_stop],
            values[:, global_start:global_stop],
            state_position,
            masks[:, local_start:local_stop],
            masks[:, global_start:global_stop],
        )

    def _slow_condition(
        self,
        slow_features: torch.Tensor,
        instrument_mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.routing is None:
            raise AssertionError("Slow condition requires active routing")
        assert self.routing.equity_slow_projection is not None
        assert self.routing.macro_slow_projection is not None
        assert self.routing.slow_condition_norm is not None
        local_start = self.equity_count + 1
        local_stop = self.equity_count + LOCAL_CONTEXT_COUNT
        global_start = self.equity_count + LOCAL_CONTEXT_COUNT + 2
        global_stop = global_start + 2
        local_slow = slow_features[:, local_start:local_stop]
        global_slow = torch.cat(
            (
                slow_features[
                    :, global_start:global_stop, :_GLOBAL_SLOW_EXCLUDED_INDEX
                ],
                slow_features[
                    :,
                    global_start:global_stop,
                    _GLOBAL_SLOW_EXCLUDED_INDEX + 1 :,
                ],
            ),
            dim=-1,
        )
        macro_ready = torch.cat(
            (
                instrument_mask[:, local_start:local_stop],
                instrument_mask[:, global_start:global_stop],
            ),
            dim=1,
        )
        macro_input = torch.cat(
            (
                local_slow.reshape(local_slow.shape[0], -1),
                global_slow.reshape(global_slow.shape[0], -1),
                macro_ready.to(local_slow.dtype),
            ),
            dim=-1,
        )
        equity = self.routing.equity_slow_projection(
            slow_features[:, : self.equity_count]
        )
        macro = self.routing.macro_slow_projection(macro_input)
        return self.routing.slow_condition_norm(equity + macro[:, None, :])

    def _macro_raw_input(
        self,
        patches: torch.Tensor,
        history_patch_mask: torch.Tensor,
        state_position: torch.Tensor,
    ) -> torch.Tensor:
        macro, available = self._macro_sources(
            patches, history_patch_mask, state_position
        )
        batch_size, source_count, patch_count, _ = macro.shape
        meaningful = macro.reshape(
            batch_size,
            source_count,
            patch_count,
            PATCH_MINUTES,
            DYNAMIC_CHANNEL_COUNT,
        )[..., :CONTEXT_GENERIC_DYNAMIC_COUNT]
        values = meaningful.reshape(
            batch_size,
            source_count,
            patch_count,
            CONTEXT_ROUTING_PATCH_SOURCE_WIDTH,
        )
        values = values * available[..., None].to(values.dtype)
        source_input = torch.cat(
            (values, available[..., None].to(values.dtype)), dim=-1
        )
        return source_input.permute(0, 2, 1, 3).reshape(batch_size, patch_count, -1)

    def _macro_early_input(self, macro_input: torch.Tensor) -> torch.Tensor:
        if self.routing is None:
            raise AssertionError("Macro early input requires active routing")
        assert self.routing.macro_early_input_projection is not None
        assert self.routing.macro_early_activation is not None
        assert self.routing.macro_early_norm is not None
        assert self.routing.macro_early_output_projection is not None
        latent = self.routing.macro_early_input_projection(macro_input)
        latent = self.routing.macro_early_activation(latent)
        latent = self.routing.macro_early_norm(latent)
        return self.routing.macro_early_output_projection(latent)

    def _apply_film(
        self,
        hidden: torch.Tensor,
        block_index: int,
        history_patch_mask: torch.Tensor,
        instrument_mask: torch.Tensor,
        state_position: torch.Tensor,
        slow_condition: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.routing is None:
            raise AssertionError("FiLM requires active routing")
        architecture = self.architecture
        batch_size = history_patch_mask.shape[0]
        streams = hidden.reshape(
            batch_size,
            self.instrument_count,
            architecture.width,
            history_patch_mask.shape[2],
        )
        gamma: torch.Tensor | None = None
        beta: torch.Tensor | None = None
        if self._route_has_film(architecture.slow_routing):
            if slow_condition is None:
                raise AssertionError("Slow FiLM requires the slow condition")
            assert self.routing.slow_film is not None
            parameters = self.routing.slow_film[block_index](slow_condition)
            slow_gamma, slow_beta = parameters.chunk(2, dim=-1)
            slow_valid = history_patch_mask[:, : self.equity_count, :, None].permute(
                0, 1, 3, 2
            )
            gamma = slow_gamma[..., None] * slow_valid.to(slow_gamma.dtype)
            beta = slow_beta[..., None] * slow_valid.to(slow_beta.dtype)
        if self._route_has_film(architecture.macro_temporal_routing):
            sequences = streams.permute(0, 1, 3, 2)
            macro, _ = self._macro_sources(
                sequences, history_patch_mask, state_position
            )
            macro_input = macro.permute(0, 2, 1, 3).reshape(
                batch_size,
                history_patch_mask.shape[2],
                CONTEXT_ROUTING_SOURCE_COUNT * architecture.width,
            )
            assert self.routing.macro_film is not None
            parameters = self.routing.macro_film[block_index](macro_input)
            macro_gamma, macro_beta = parameters.chunk(2, dim=-1)
            macro_gamma = macro_gamma.permute(0, 2, 1)[:, None]
            macro_beta = macro_beta.permute(0, 2, 1)[:, None]
            macro_valid = (
                causal_patch_mask(state_position)[:, None, None, :]
                & instrument_mask[:, : self.equity_count, None, None]
            )
            macro_gamma = macro_gamma * macro_valid.to(macro_gamma.dtype)
            macro_beta = macro_beta * macro_valid.to(macro_beta.dtype)
            gamma = macro_gamma if gamma is None else gamma + macro_gamma
            beta = macro_beta if beta is None else beta + macro_beta
        if gamma is None or beta is None:
            return hidden
        equity = streams[:, : self.equity_count]
        modulated = apply_context_film(equity, gamma, beta)
        return torch.cat((modulated, streams[:, self.equity_count :]), dim=1).reshape(
            batch_size * self.instrument_count,
            architecture.width,
            history_patch_mask.shape[2],
        )

    def _gather_hidden_states(
        self,
        hidden: torch.Tensor,
        batch_size: int,
        instrument_count: int,
        state_position: torch.Tensor,
    ) -> torch.Tensor:
        architecture = self.architecture
        sequence_length = hidden.shape[-1]
        streams = hidden.reshape(
            batch_size, instrument_count, architecture.width, sequence_length
        ).permute(0, 1, 3, 2)
        gather_positions = state_position[:, None].expand(-1, instrument_count)
        if instrument_count == self.instrument_count:
            global_slots = (
                torch.arange(instrument_count, device=hidden.device)
                >= self.equity_count + LOCAL_CONTEXT_COUNT
            )
            gather_positions = torch.where(
                global_slots[None], STATE_TOKEN_SLOT, gather_positions
            )
        gather_index = (gather_positions - 1)[..., None, None].expand(
            -1, -1, 1, architecture.width
        )
        return streams.gather(2, gather_index).squeeze(2)

    def _instrument_states(
        self,
        patches: torch.Tensor,
        history_patch_mask: torch.Tensor,
        instrument_mask: torch.Tensor,
        slow_features: torch.Tensor,
        state_position: torch.Tensor,
        *,
        collect_taps: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        architecture = self.architecture
        batch_size = patches.shape[0]
        instrument_count = (
            self.instrument_count
            if architecture.fusion_mode in ("context_only", "context_pooled")
            else self.equity_count
        )
        patches = patches[:, :instrument_count]
        history_patch_mask = history_patch_mask[:, :instrument_count]
        instrument_mask = instrument_mask[:, :instrument_count]
        slow_features = slow_features[:, :instrument_count]
        masked = patches * history_patch_mask[..., None].to(patches.dtype)
        hidden = self.input_projection(masked).permute(0, 1, 3, 2)

        slow_condition = None
        if self.routing is not None and (
            self._route_has_early(architecture.slow_routing)
            or self._route_has_film(architecture.slow_routing)
        ):
            slow_condition = self._slow_condition(slow_features, instrument_mask)
        if self.routing is not None and self._route_has_early(
            architecture.slow_routing
        ):
            if slow_condition is None:
                raise AssertionError("Slow early routing requires the slow condition")
            slow_early = self.routing.slow_early_input_adapter(slow_condition)[
                ..., None
            ]
            slow_valid = history_patch_mask[:, : self.equity_count, None, :].to(
                slow_early.dtype
            )
            hidden = torch.cat(
                (
                    hidden[:, : self.equity_count] + slow_early * slow_valid,
                    hidden[:, self.equity_count :],
                ),
                dim=1,
            )
        if self.routing is not None and self._route_has_early(
            architecture.macro_temporal_routing
        ):
            macro_input = self._macro_raw_input(
                patches, history_patch_mask, state_position
            )
            macro_early = self._macro_early_input(macro_input).permute(0, 2, 1)[:, None]
            macro_valid = (
                causal_patch_mask(state_position)[:, None, None, :]
                & instrument_mask[:, : self.equity_count, None, None]
            )
            hidden = torch.cat(
                (
                    hidden[:, : self.equity_count]
                    + macro_early * macro_valid.to(macro_early.dtype),
                    hidden[:, self.equity_count :],
                ),
                dim=1,
            )

        hidden = hidden.reshape(
            batch_size * instrument_count,
            architecture.width,
            masked.shape[2],
        )
        taps: list[torch.Tensor] = []
        for block_index, block in enumerate(self.blocks):
            hidden = block(hidden)
            if self.routing is not None and (
                self._route_has_film(architecture.slow_routing)
                or self._route_has_film(architecture.macro_temporal_routing)
            ):
                hidden = self._apply_film(
                    hidden,
                    block_index,
                    history_patch_mask,
                    instrument_mask,
                    state_position,
                    slow_condition,
                )
            if collect_taps:
                taps.append(
                    self._gather_hidden_states(
                        hidden, batch_size, instrument_count, state_position
                    )[:, : self.equity_count]
                )
        raw_state = self._gather_hidden_states(
            hidden, batch_size, instrument_count, state_position
        )
        states = self.state_norm(raw_state + self.slow_projection(slow_features))
        return states, tuple(taps)

    def _add_peer(
        self, equity_states: torch.Tensor, peer_state: torch.Tensor | None
    ) -> torch.Tensor:
        if self.peer_adapter is None:
            if peer_state is not None:
                raise ValueError("Peer state is forbidden when peer features are none")
            return equity_states
        if peer_state is None:
            raise ValueError("Peer state is required for peer-enabled TCN")
        if peer_state.shape != (
            equity_states.shape[0],
            self.equity_count,
            PEER_STATE_WIDTH,
        ):
            raise ValueError("Peer state has the wrong shape")
        peer = self.peer_adapter(peer_state)
        return (
            equity_states + peer[:, :, None, :]
            if equity_states.ndim == 4
            else equity_states + peer
        )

    def _fuse_equity(
        self,
        equity_states: torch.Tensor,
        context_states: torch.Tensor,
        equity_mask: torch.Tensor,
        context_mask: torch.Tensor,
    ) -> torch.Tensor:
        fusion_mode = self.architecture.fusion_mode
        if fusion_mode == "none":
            return equity_states
        shared_parts: list[torch.Tensor] = []
        if fusion_mode in ("context_only", "context_pooled"):
            shared_parts.append(
                (
                    context_states * context_mask[..., None].to(context_states.dtype)
                ).reshape(
                    context_states.shape[0],
                    CONTEXT_COUNT * self.architecture.width,
                )
            )
        if fusion_mode in ("pooled_market", "context_pooled"):
            weight = equity_mask[..., None].to(equity_states.dtype)
            count = weight.sum(dim=1).clamp_min(1.0)
            mean = (equity_states * weight).sum(dim=1) / count
            second_moment = (equity_states.square() * weight).sum(dim=1) / count
            dispersion = torch.sqrt(
                torch.clamp(second_moment - mean.square(), min=1e-6)
            )
            shared_parts.extend((mean, dispersion))
        shared = torch.cat(shared_parts, dim=-1)
        shared = shared[:, None].expand(-1, self.equity_count, -1)
        fusion_input = torch.cat((equity_states, shared), dim=-1)
        fused = self.fusion_output(F.gelu(self.fusion_input(fusion_input)))
        fused = self.dropout(F.gelu(fused))
        gate = torch.sigmoid(
            self.fusion_gate(torch.cat((equity_states, fused), dim=-1))
        )
        return self.fusion_norm(equity_states + gate * fused)

    def _fuse_horizons(
        self,
        equity_states: torch.Tensor,
        context_states: torch.Tensor,
        equity_mask: torch.Tensor,
        context_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, _, horizon_count, width = equity_states.shape
        flattened = equity_states.permute(0, 2, 1, 3).reshape(
            batch_size * horizon_count, self.equity_count, width
        )
        repeated_context = (
            context_states[:, None]
            .expand(-1, horizon_count, -1, -1)
            .reshape(batch_size * horizon_count, context_states.shape[1], width)
        )
        repeated_equity_mask = (
            equity_mask[:, None]
            .expand(-1, horizon_count, -1)
            .reshape(batch_size * horizon_count, self.equity_count)
        )
        repeated_context_mask = (
            context_mask[:, None]
            .expand(-1, horizon_count, -1)
            .reshape(batch_size * horizon_count, context_mask.shape[1])
        )
        fused = self._fuse_equity(
            flattened,
            repeated_context,
            repeated_equity_mask,
            repeated_context_mask,
        )
        return fused.reshape(
            batch_size, horizon_count, self.equity_count, width
        ).permute(0, 2, 1, 3)

    def scale_weights(self) -> torch.Tensor | None:
        return (
            None
            if self.scale_logits is None
            else torch.softmax(self.scale_logits, dim=-1)
        )

    def _readout(
        self,
        states: torch.Tensor,
        taps: tuple[torch.Tensor, ...],
        instrument_mask: torch.Tensor,
        slow_features: torch.Tensor,
        peer_state: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        equity_mask = instrument_mask[:, : self.equity_count]
        context_states = states[:, self.equity_count :]
        context_mask = instrument_mask[:, self.equity_count :]
        readout = self.architecture.readout
        if readout in ("final", "final_score_mlp"):
            equity_states = self._add_peer(states[:, : self.equity_count], peer_state)
            equity_states = self._fuse_equity(
                equity_states, context_states, equity_mask, context_mask
            )
            predictions = self.prediction_head(equity_states)
            if self.score_mlp is not None:
                predictions = predictions + self.score_mlp(predictions)
        else:
            if len(taps) != self.architecture.residual_blocks:
                raise AssertionError("Multiscale readout requires every TCN tap")
            tap_tensor = torch.stack(taps, dim=2)
            weights = self.scale_weights()
            assert weights is not None
            slow = self.slow_projection(slow_features[:, : self.equity_count])
            if readout == "shared_multiscale":
                mixed = (tap_tensor * weights[None, None, :, None]).sum(dim=2)
                equity_states = self.state_norm(mixed + slow)
                equity_states = self._add_peer(equity_states, peer_state)
                equity_states = self._fuse_equity(
                    equity_states, context_states, equity_mask, context_mask
                )
                predictions = self.prediction_head(equity_states)
            else:
                mixed = torch.einsum("hk,bekw->behw", weights, tap_tensor)
                equity_states = self.state_norm(mixed + slow[:, :, None, :])
                equity_states = self._add_peer(equity_states, peer_state)
                equity_states = self._fuse_horizons(
                    equity_states, context_states, equity_mask, context_mask
                )
                predictions = (
                    equity_states * self.prediction_head.weight[None, None, :, :]
                ).sum(dim=-1) + self.prediction_head.bias
        return predictions * equity_mask[..., None].to(predictions.dtype), equity_states

    def forward(
        self,
        patches: torch.Tensor,
        history_patch_mask: torch.Tensor,
        instrument_mask: torch.Tensor,
        slow_features: torch.Tensor,
        state_position: torch.Tensor,
        peer_state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        collect_taps = self.architecture.readout in (
            "shared_multiscale",
            "horizon_multiscale",
        )
        states, taps = self._instrument_states(
            patches,
            history_patch_mask,
            instrument_mask,
            slow_features,
            state_position,
            collect_taps=collect_taps,
        )
        predictions, _ = self._readout(
            states, taps, instrument_mask, slow_features, peer_state
        )
        return predictions

    def extract_diagnostics(
        self,
        patches: torch.Tensor,
        history_patch_mask: torch.Tensor,
        instrument_mask: torch.Tensor,
        slow_features: torch.Tensor,
        state_position: torch.Tensor,
        peer_state: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if torch.compiler.is_compiling():
            raise RuntimeError("Diagnostic extraction is eager-only")
        states, taps = self._instrument_states(
            patches,
            history_patch_mask,
            instrument_mask,
            slow_features,
            state_position,
            collect_taps=True,
        )
        predictions, representation = self._readout(
            states, taps, instrument_mask, slow_features, peer_state
        )
        result = {
            f"block_{index}": tap.detach() for index, tap in enumerate(taps, start=1)
        }
        result["final_pre_head"] = representation.detach()
        result["predictions"] = predictions.detach()
        if self.architecture.fusion_mode in ("context_only", "context_pooled"):
            result["context_states"] = states[:, self.equity_count :].detach()
        return result


class _MLPResidualBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm = nn.RMSNorm(MLP_WIDTH, eps=RMS_NORM_EPS)
        self.feedforward = SwiGLU(MLP_WIDTH, MLP_SWIGLU_WIDTH)
        self.dropout = nn.Dropout(RESIDUAL_DROPOUT)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs + self.dropout(self.feedforward(self.norm(inputs)))


class ResidualTabularMLP(nn.Module):
    model_name = "mlp"

    def __init__(self) -> None:
        super().__init__()
        self.input_projection = nn.Linear(TABULAR_FEATURE_COUNT, MLP_WIDTH, bias=True)
        self.blocks = nn.ModuleList([_MLPResidualBlock() for _ in range(MLP_DEPTH)])
        self.final_norm = nn.RMSNorm(MLP_WIDTH, eps=RMS_NORM_EPS)
        self.prediction_head = nn.Linear(MLP_WIDTH, HORIZON_COUNT, bias=True)
        self.apply(self._initialize_module)

    @staticmethod
    def _initialize_module(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.RMSNorm):
            nn.init.ones_(module.weight)

    def forward(
        self, tabular_features: torch.Tensor, equity_mask: torch.Tensor
    ) -> torch.Tensor:
        hidden = self.input_projection(tabular_features)
        for block in self.blocks:
            hidden = block(hidden)
        predictions = self.prediction_head(self.final_norm(hidden))
        return predictions * equity_mask[..., None].to(predictions.dtype)
