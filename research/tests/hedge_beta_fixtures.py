from pathlib import Path
from collections.abc import Sequence

import numpy as np

from brazil_rv.v2.artifacts import sha256_file, write_json_atomic
from brazil_rv.v2.hedge_beta import HEDGE_BETA_SCHEMA


def write_hedge_beta_fixture(
    root: Path,
    *,
    dates: np.ndarray,
    isins: Sequence[str],
    store_sha256: str,
    bova11_sha256: str,
    values: np.ndarray | None = None,
    valid: np.ndarray | None = None,
) -> dict[str, str]:
    root.mkdir()
    shape = (len(dates), len(isins))
    arrays = {}
    for name, array in (
        ("date_index", dates),
        ("isin_index", np.asarray(isins)),
        ("hedge_beta", np.ones(shape) if values is None else values),
        ("hedge_beta_valid", np.ones(shape, dtype=bool) if valid is None else valid),
    ):
        path = root / f"{name}.npy"
        np.save(path, array, allow_pickle=False)
        arrays[name] = {
            "path": path.name,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    digest = write_json_atomic(
        root / "manifest.json",
        {
            "schema": HEDGE_BETA_SCHEMA,
            "status": "complete",
            "arrays": arrays,
            "store": {"manifest_sha256": store_sha256},
            "bova11": {"manifest_sha256": bova11_sha256},
        },
    )
    return {"hedge_beta_root": str(root), "hedge_beta_manifest_sha256": digest}
