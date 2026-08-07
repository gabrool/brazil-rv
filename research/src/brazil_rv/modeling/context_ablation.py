from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Final

import polars as pl

from .contract import (
    FEATURE_CONTRACT_VERSION,
    GLOBAL_CONTEXT_SYMBOLS,
    LOCAL_CONTEXT_SYMBOLS,
)


@dataclass(frozen=True)
class ContextAblation:
    key: str
    removed_local_symbols: tuple[str, ...]
    removed_global_symbols: tuple[str, ...]
    neutralized_equity_slow_features: tuple[str, ...]
    description: str

    def specification(self) -> dict[str, object]:
        return {
            "key": self.key,
            "description": self.description,
            "removed_local_symbols": list(self.removed_local_symbols),
            "removed_global_symbols": list(self.removed_global_symbols),
            "neutralized_equity_slow_features": list(
                self.neutralized_equity_slow_features
            ),
        }

    def serialized_specification(self) -> str:
        return json.dumps(self.specification(), sort_keys=True, separators=(",", ":"))

    def specification_sha256(self) -> str:
        return hashlib.sha256(self.serialized_specification().encode()).hexdigest()

    def metadata(self) -> dict[str, object]:
        return {
            **self.specification(),
            "serialized_specification": self.serialized_specification(),
            "specification_sha256": self.specification_sha256(),
        }


@dataclass(frozen=True)
class ResolvedContextAblation:
    specification: ContextAblation
    local_slots: tuple[int, ...]
    global_slots: tuple[int, ...]
    equity_slow_indices: tuple[int, ...]

    @property
    def key(self) -> str:
        return self.specification.key

    def metadata(self) -> dict[str, object]:
        return self.specification.metadata()


_DEPENDENCY_BY_LOCAL_SYMBOL: Final = MappingProxyType(
    {
        "WIN$": ("beta_to_WIN",),
        "WDO$": ("beta_to_WDO",),
        "DI1F27": ("beta_to_DI1F27",),
        "DI1F28": ("beta_to_DI1F28",),
        "DI1F29": ("beta_to_DI1F29",),
        "DI1F31": ("beta_to_DI1F31",),
        "DI1$N": (),
    }
)


def _ablation(
    key: str,
    local: tuple[str, ...],
    global_: tuple[str, ...],
    dependencies: tuple[str, ...],
    description: str,
) -> ContextAblation:
    return ContextAblation(key, local, global_, dependencies, description)


_FIXED_DI = ("DI1F27", "DI1F28", "DI1F29", "DI1F31")
_ALL_DI = (*_FIXED_DI, "DI1$N")
_US_EQUITIES = ("ES.v.0", "NQ.v.0")
_US_RATES = ("ZT.v.0", "ZN.v.0")
_COMMODITIES = ("CL.v.0", "HG.v.0")
_GLOBAL_FX = ("6E.v.0", "6M.v.0")
_GLOBAL_NON_RATES = tuple(
    symbol for symbol in GLOBAL_CONTEXT_SYMBOLS if symbol not in _US_RATES
)
_FIXED_DI_BETAS = (
    "beta_to_DI1F27",
    "beta_to_DI1F28",
    "beta_to_DI1F29",
    "beta_to_DI1F31",
)
_ALL_LOCAL_BETAS = ("beta_to_WIN", "beta_to_WDO", *_FIXED_DI_BETAS)

CONTEXT_ABLATIONS = MappingProxyType(
    {
        "none": _ablation(
            "none", (), (), (), "Use every canonical local and global context source."
        ),
        "drop_win": _ablation(
            "drop_win",
            ("WIN$",),
            (),
            ("beta_to_WIN",),
            "Remove the WIN$ local future and its equity beta.",
        ),
        "drop_wdo": _ablation(
            "drop_wdo",
            ("WDO$",),
            (),
            ("beta_to_WDO",),
            "Remove the WDO$ local future and its equity beta.",
        ),
        "drop_di1f27": _ablation(
            "drop_di1f27",
            ("DI1F27",),
            (),
            ("beta_to_DI1F27",),
            "Remove DI1F27 and its equity beta.",
        ),
        "drop_di1f28": _ablation(
            "drop_di1f28",
            ("DI1F28",),
            (),
            ("beta_to_DI1F28",),
            "Remove DI1F28 and its equity beta.",
        ),
        "drop_di1f29": _ablation(
            "drop_di1f29",
            ("DI1F29",),
            (),
            ("beta_to_DI1F29",),
            "Remove DI1F29 and its equity beta.",
        ),
        "drop_di1f31": _ablation(
            "drop_di1f31",
            ("DI1F31",),
            (),
            ("beta_to_DI1F31",),
            "Remove DI1F31 and its equity beta.",
        ),
        "drop_di1n": _ablation(
            "drop_di1n",
            ("DI1$N",),
            (),
            (),
            "Remove the liquidity-selected DI1$N local rate future.",
        ),
        "drop_es": _ablation(
            "drop_es", (), ("ES.v.0",), (), "Remove the ES global equity future."
        ),
        "drop_nq": _ablation(
            "drop_nq", (), ("NQ.v.0",), (), "Remove the NQ global equity future."
        ),
        "drop_zt": _ablation(
            "drop_zt", (), ("ZT.v.0",), (), "Remove the ZT global rate future."
        ),
        "drop_zn": _ablation(
            "drop_zn", (), ("ZN.v.0",), (), "Remove the ZN global rate future."
        ),
        "drop_cl": _ablation(
            "drop_cl", (), ("CL.v.0",), (), "Remove the CL global commodity future."
        ),
        "drop_hg": _ablation(
            "drop_hg", (), ("HG.v.0",), (), "Remove the HG global commodity future."
        ),
        "drop_6e": _ablation(
            "drop_6e", (), ("6E.v.0",), (), "Remove the 6E global FX future."
        ),
        "drop_6m": _ablation(
            "drop_6m", (), ("6M.v.0",), (), "Remove the 6M global FX future."
        ),
        "drop_fixed_di": _ablation(
            "drop_fixed_di",
            _FIXED_DI,
            (),
            _FIXED_DI_BETAS,
            "Remove all four fixed-maturity DI futures and their equity betas.",
        ),
        "drop_all_di": _ablation(
            "drop_all_di",
            _ALL_DI,
            (),
            _FIXED_DI_BETAS,
            "Remove every DI future and all fixed-maturity DI equity betas.",
        ),
        "drop_us_equities": _ablation(
            "drop_us_equities",
            (),
            _US_EQUITIES,
            (),
            "Remove the ES and NQ global equity futures.",
        ),
        "drop_us_rates": _ablation(
            "drop_us_rates",
            (),
            _US_RATES,
            (),
            "Remove the ZT and ZN global rate futures.",
        ),
        "drop_commodities": _ablation(
            "drop_commodities",
            (),
            _COMMODITIES,
            (),
            "Remove the CL and HG global commodity futures.",
        ),
        "drop_global_fx": _ablation(
            "drop_global_fx",
            (),
            _GLOBAL_FX,
            (),
            "Remove the 6E and 6M global FX futures.",
        ),
        "drop_all_local": _ablation(
            "drop_all_local",
            LOCAL_CONTEXT_SYMBOLS,
            (),
            _ALL_LOCAL_BETAS,
            "Remove all seven local contexts and every local-source equity beta.",
        ),
        "drop_all_global": _ablation(
            "drop_all_global",
            (),
            GLOBAL_CONTEXT_SYMBOLS,
            (),
            "Remove all eight global context futures.",
        ),
        "drop_all_context": _ablation(
            "drop_all_context",
            LOCAL_CONTEXT_SYMBOLS,
            GLOBAL_CONTEXT_SYMBOLS,
            _ALL_LOCAL_BETAS,
            "Remove all local and global contexts and every local-source equity beta.",
        ),
        "drop_global_non_rates": _ablation(
            "drop_global_non_rates",
            (),
            _GLOBAL_NON_RATES,
            (),
            "Remove every global context except the ZT and ZN rate futures.",
        ),
        "drop_win_and_global_non_rates": _ablation(
            "drop_win_and_global_non_rates",
            ("WIN$",),
            _GLOBAL_NON_RATES,
            ("beta_to_WIN",),
            "Remove WIN$, its equity beta, and every non-rate global context.",
        ),
        "drop_win_and_global_non_rates_except_es": _ablation(
            "drop_win_and_global_non_rates_except_es",
            ("WIN$",),
            tuple(symbol for symbol in _GLOBAL_NON_RATES if symbol != "ES.v.0"),
            ("beta_to_WIN",),
            "Remove WIN$, its equity beta, and every non-rate global except ES.",
        ),
        "drop_win_and_global_non_rates_except_nq": _ablation(
            "drop_win_and_global_non_rates_except_nq",
            ("WIN$",),
            tuple(symbol for symbol in _GLOBAL_NON_RATES if symbol != "NQ.v.0"),
            ("beta_to_WIN",),
            "Remove WIN$, its equity beta, and every non-rate global except NQ.",
        ),
        "drop_win_and_global_non_rates_except_cl": _ablation(
            "drop_win_and_global_non_rates_except_cl",
            ("WIN$",),
            tuple(symbol for symbol in _GLOBAL_NON_RATES if symbol != "CL.v.0"),
            ("beta_to_WIN",),
            "Remove WIN$, its equity beta, and every non-rate global except CL.",
        ),
        "drop_win_and_global_non_rates_except_hg": _ablation(
            "drop_win_and_global_non_rates_except_hg",
            ("WIN$",),
            tuple(symbol for symbol in _GLOBAL_NON_RATES if symbol != "HG.v.0"),
            ("beta_to_WIN",),
            "Remove WIN$, its equity beta, and every non-rate global except HG.",
        ),
        "drop_win_and_global_non_rates_except_6e": _ablation(
            "drop_win_and_global_non_rates_except_6e",
            ("WIN$",),
            tuple(symbol for symbol in _GLOBAL_NON_RATES if symbol != "6E.v.0"),
            ("beta_to_WIN",),
            "Remove WIN$, its equity beta, and every non-rate global except 6E.",
        ),
        "drop_win_and_global_non_rates_except_6m": _ablation(
            "drop_win_and_global_non_rates_except_6m",
            ("WIN$",),
            tuple(symbol for symbol in _GLOBAL_NON_RATES if symbol != "6M.v.0"),
            ("beta_to_WIN",),
            "Remove WIN$, its equity beta, and every non-rate global except 6M.",
        ),
    }
)

CONTEXT_ABLATION_KEYS = tuple(CONTEXT_ABLATIONS)
INDIVIDUAL_CONTEXT_ABLATIONS = (
    "drop_win",
    "drop_wdo",
    "drop_di1f27",
    "drop_di1f28",
    "drop_di1f29",
    "drop_di1f31",
    "drop_di1n",
    "drop_es",
    "drop_nq",
    "drop_zt",
    "drop_zn",
    "drop_cl",
    "drop_hg",
    "drop_6e",
    "drop_6m",
)
GROUP_CONTEXT_ABLATIONS = (
    "drop_fixed_di",
    "drop_all_di",
    "drop_us_equities",
    "drop_us_rates",
    "drop_commodities",
    "drop_global_fx",
    "drop_all_local",
    "drop_all_global",
    "drop_all_context",
)
STAGE1_CONTEXT_ABLATION_ORDER = (
    "none",
    "drop_fixed_di",
    "drop_all_di",
    "drop_us_equities",
    "drop_us_rates",
    "drop_commodities",
    "drop_global_fx",
    "drop_all_local",
    "drop_all_global",
    "drop_all_context",
    "drop_win",
    "drop_wdo",
    "drop_di1f27",
    "drop_di1f28",
    "drop_di1f29",
    "drop_di1f31",
    "drop_di1n",
    "drop_es",
    "drop_nq",
    "drop_zt",
    "drop_zn",
    "drop_cl",
    "drop_hg",
    "drop_6e",
    "drop_6m",
)


def _validate_registry() -> None:
    expected_keys = (
        "none",
        "drop_win",
        "drop_wdo",
        "drop_di1f27",
        "drop_di1f28",
        "drop_di1f29",
        "drop_di1f31",
        "drop_di1n",
        "drop_es",
        "drop_nq",
        "drop_zt",
        "drop_zn",
        "drop_cl",
        "drop_hg",
        "drop_6e",
        "drop_6m",
        "drop_fixed_di",
        "drop_all_di",
        "drop_us_equities",
        "drop_us_rates",
        "drop_commodities",
        "drop_global_fx",
        "drop_all_local",
        "drop_all_global",
        "drop_all_context",
        "drop_global_non_rates",
        "drop_win_and_global_non_rates",
        "drop_win_and_global_non_rates_except_es",
        "drop_win_and_global_non_rates_except_nq",
        "drop_win_and_global_non_rates_except_cl",
        "drop_win_and_global_non_rates_except_hg",
        "drop_win_and_global_non_rates_except_6e",
        "drop_win_and_global_non_rates_except_6m",
    )
    if CONTEXT_ABLATION_KEYS != expected_keys:
        raise ValueError("Context-ablation registry keys do not match the contract")
    if set(_DEPENDENCY_BY_LOCAL_SYMBOL) != set(LOCAL_CONTEXT_SYMBOLS):
        raise ValueError("Local dependency map does not cover the canonical axis")
    for key, specification in CONTEXT_ABLATIONS.items():
        if specification.key != key or not specification.description:
            raise ValueError(f"Invalid context-ablation identity: {key}")
        local = specification.removed_local_symbols
        global_ = specification.removed_global_symbols
        dependencies = specification.neutralized_equity_slow_features
        if len(local) != len(set(local)) or len(global_) != len(set(global_)):
            raise ValueError(f"Duplicated context symbol in ablation: {key}")
        if len(dependencies) != len(set(dependencies)):
            raise ValueError(f"Duplicated derived dependency in ablation: {key}")
        if not set(local) <= set(LOCAL_CONTEXT_SYMBOLS):
            raise ValueError(f"Unknown local context symbol in ablation: {key}")
        if not set(global_) <= set(GLOBAL_CONTEXT_SYMBOLS):
            raise ValueError(f"Unknown global context symbol in ablation: {key}")
        expected_dependencies = tuple(
            dependency
            for symbol in local
            for dependency in _DEPENDENCY_BY_LOCAL_SYMBOL[symbol]
        )
        if dependencies != expected_dependencies:
            raise ValueError(f"Derived dependencies are incomplete for ablation: {key}")
    if (
        len(STAGE1_CONTEXT_ABLATION_ORDER) != 25
        or len(set(STAGE1_CONTEXT_ABLATION_ORDER)) != 25
    ):
        raise ValueError("Stage-1 context-ablation order must contain 25 unique keys")
    if STAGE1_CONTEXT_ABLATION_ORDER != (
        "none",
        *GROUP_CONTEXT_ABLATIONS,
        *INDIVIDUAL_CONTEXT_ABLATIONS,
    ):
        raise ValueError("Stage-1 context-ablation order changed")


_validate_registry()


def get_context_ablation(key: str) -> ContextAblation:
    try:
        return CONTEXT_ABLATIONS[key]
    except KeyError as error:
        raise ValueError(f"Unknown context ablation: {key}") from error


def resolve_context_ablation(
    specification: ContextAblation,
    *,
    local_symbols: tuple[str, ...],
    global_symbols: tuple[str, ...],
    equity_slow_features: tuple[str, ...],
) -> ResolvedContextAblation:
    for axis_name, axis in (("local", local_symbols), ("global", global_symbols)):
        if len(axis) != len(set(axis)):
            raise ValueError(
                f"Feature-store {axis_name} context axis contains duplicates"
            )
    if local_symbols != LOCAL_CONTEXT_SYMBOLS:
        raise ValueError("Feature-store local context axis is not canonical")
    if global_symbols != GLOBAL_CONTEXT_SYMBOLS:
        raise ValueError("Feature-store global context axis is not canonical")
    if len(equity_slow_features) != len(set(equity_slow_features)):
        raise ValueError("Feature-store equity slow-feature names contain duplicates")
    missing = set(specification.neutralized_equity_slow_features) - set(
        equity_slow_features
    )
    if missing:
        raise ValueError(f"Unknown equity slow-feature dependencies: {sorted(missing)}")
    return ResolvedContextAblation(
        specification=specification,
        local_slots=tuple(
            local_symbols.index(symbol)
            for symbol in specification.removed_local_symbols
        ),
        global_slots=tuple(
            global_symbols.index(symbol)
            for symbol in specification.removed_global_symbols
        ),
        equity_slow_indices=tuple(
            equity_slow_features.index(name)
            for name in specification.neutralized_equity_slow_features
        ),
    )


def resolve_context_ablation_for_store(
    store: Path, key: str
) -> ResolvedContextAblation:
    schema = json.loads((store / "feature_schema.json").read_text(encoding="utf-8"))
    if schema.get("contract_version") != FEATURE_CONTRACT_VERSION:
        raise ValueError("Feature schema has the wrong context-ablation contract")
    local_index = pl.read_parquet(store / "context_index.parquet").sort("context_slot")
    local_symbols = tuple(local_index.get_column("symbol"))
    schema_local = tuple(schema.get("local_context", {}).get("symbols", ()))
    if schema_local != local_symbols:
        raise ValueError("Feature schema and local context index disagree")
    global_index = (
        pl.scan_parquet(store / "global_context_index.parquet")
        .select("global_slot", "continuous_symbol")
        .unique()
        .sort("global_slot")
        .collect()
    )
    global_symbols = tuple(global_index.get_column("continuous_symbol"))
    slow_rows = schema.get("slow_channels")
    if not isinstance(slow_rows, list) or any(
        not isinstance(row, dict) for row in slow_rows
    ):
        raise ValueError("Feature schema is missing equity slow-channel metadata")
    indices = tuple(row.get("index") for row in slow_rows)
    if indices != tuple(range(len(slow_rows))):
        raise ValueError("Feature schema slow-channel indices are not contiguous")
    slow_features = tuple(str(row.get("name")) for row in slow_rows)
    resolved = resolve_context_ablation(
        get_context_ablation(key),
        local_symbols=local_symbols,
        global_symbols=global_symbols,
        equity_slow_features=slow_features,
    )
    exposure_sources = tuple(
        schema.get("local_context", {}).get("exposure_beta_source_symbols", ())
    )
    expected_sources = tuple(
        symbol
        for symbol in LOCAL_CONTEXT_SYMBOLS
        if _DEPENDENCY_BY_LOCAL_SYMBOL[symbol]
    )
    if exposure_sources != expected_sources:
        raise ValueError("Feature schema has ambiguous equity-beta source dependencies")
    for registry_specification in CONTEXT_ABLATIONS.values():
        resolve_context_ablation(
            registry_specification,
            local_symbols=local_symbols,
            global_symbols=global_symbols,
            equity_slow_features=slow_features,
        )
    return resolved


NO_CONTEXT_ABLATION = ResolvedContextAblation(CONTEXT_ABLATIONS["none"], (), (), ())
