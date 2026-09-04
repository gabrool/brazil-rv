from __future__ import annotations

import json
import subprocess
import sys


def test_all_daily_families_fit_real_axes_below_eight_gib(tmp_path) -> None:
    """Guard the production 4,348 x 933 family-at-a-time RSS contract."""

    script = r"""
import gc
import json
import sys
from pathlib import Path

import numpy as np

from brazil_rv.v2.contract import INTRADAY_DAILY_FEATURES, SIDECAR_FEATURES, SLOW_FEATURES
from brazil_rv.v2.normalization import rank_gauss_panel_into
from brazil_rv.v2.store import close_memmap, peak_rss_bytes
from brazil_rv.v2.targets import build_multi_day_targets_into

root = Path(sys.argv[1])
date_count = 4_348
name_count = 933
active = np.zeros((date_count, name_count), dtype=np.bool_)
widths = [
    len(SLOW_FEATURES),
    len(INTRADAY_DAILY_FEATURES),
    *(len(SIDECAR_FEATURES[group]) for group in sorted(SIDECAR_FEATURES)),
]
for family_index, width in enumerate(widths):
    shape = (date_count, name_count, width)
    raw = np.zeros(shape, dtype=np.float32)
    valid = np.zeros(shape, dtype=np.bool_)
    value_path = root / f"family_{family_index}_values.npy"
    valid_path = root / f"family_{family_index}_valid.npy"
    output = np.lib.format.open_memmap(value_path, mode="w+", dtype=np.float32, shape=shape)
    output_valid = np.lib.format.open_memmap(valid_path, mode="w+", dtype=np.bool_, shape=shape)
    rank_gauss_panel_into(raw, valid, active, output, output_valid)
    close_memmap(output)
    close_memmap(output_valid)
    del raw, valid, output, output_valid
    value_path.unlink()
    valid_path.unlink()
    gc.collect()

target_shape = (date_count, name_count, 5)
target_paths = []
destinations = []
for index, dtype in enumerate((np.float32, np.bool_, np.float32, np.float32, np.bool_, np.float32)):
    path = root / f"target_{index}.npy"
    target_paths.append(path)
    destinations.append(np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=target_shape))
close = np.ones((date_count, name_count), dtype=np.float32)
sigma = np.ones_like(close)
excluded = np.zeros_like(active)
build_multi_day_targets_into(
    close,
    active,
    sigma,
    excluded,
    primary=destinations[0],
    primary_valid=destinations[1],
    normalized_residual=destinations[2],
    raw_midrank=destinations[3],
    raw_valid=destinations[4],
    raw_log_return=destinations[5],
)
for destination in destinations:
    close_memmap(destination)
for path in target_paths:
    path.unlink()
print(json.dumps({"peak_rss_bytes": peak_rss_bytes(), "family_widths": widths}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(completed.stdout.strip().splitlines()[-1])
    assert result["peak_rss_bytes"] < 8 * 1024**3
    assert len(result["family_widths"]) >= 8
