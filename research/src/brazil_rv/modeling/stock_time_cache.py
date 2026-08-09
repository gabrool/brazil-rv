from __future__ import annotations

from pathlib import Path


CACHE_VERSION = 3
INFERENCE_CODE_PATHS = (
    "research/src/brazil_rv/preprocessing/contract.py",
    "research/src/brazil_rv/preprocessing/transforms.py",
    "research/src/brazil_rv/modeling/contract.py",
    "research/src/brazil_rv/modeling/context_ablation.py",
    "research/src/brazil_rv/modeling/feature_ablation.py",
    "research/src/brazil_rv/modeling/data.py",
    "research/src/brazil_rv/modeling/layers.py",
    "research/src/brazil_rv/modeling/model.py",
    "research/src/brazil_rv/modeling/engine.py",
    "research/src/brazil_rv/modeling/metrics.py",
    "research/src/brazil_rv/modeling/evaluate.py",
    "research/src/brazil_rv/modeling/stock_time_cache.py",
)


def default_cache_directory(output_dir: Path, analysis_name: str) -> Path:
    return output_dir.resolve().parent / f"_{analysis_name}_cache"


def prediction_cache_directory(
    cache_dir: Path,
    logical_configuration: str,
    seed: int,
) -> Path:
    return cache_dir / "predictions" / f"{logical_configuration}_seed{seed}"


def shared_validation_directory(cache_dir: Path) -> Path:
    return cache_dir / "shared_validation"
