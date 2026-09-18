"""Bounded accounting/eligibility audit at the two verified ticker renames."""

import json
from pathlib import Path

import numpy as np

from brazil_rv.v2.artifacts import sha256_file, write_json_atomic

project = Path(__file__).resolve().parents[1]
root = Path(json.loads((project / "docs/v2_foundation_run.json").read_text())["root"])
design = json.loads((root / "frozen_design.json").read_text())
store = Path(design["store"]["root"])
if sha256_file(store / "manifest.json") != design["store"]["manifest_sha256"]:
    raise ValueError("identity boundary audit store changed")
sources = json.loads((root / "primary_sources/manifest.json").read_text())
dates = np.load(store / "date_index.npy")
names = np.load(store / "isin_index.npy").tolist()
close = np.load(store / "raw_close.npy", mmap_mode="r")
active = np.load(store / "active.npy", mmap_mode="r")
successor = np.load(store / "action_successor_index.npy", mmap_mode="r")
records = []
for old, new, event, source_key in (
    ("BRALSOACNOR5", "BRALOSACNOR5", "2023-10-25", "allos_ticker_20231017"),
    ("BRTRPLACNPR1", "BRISAEACNPR9", "2024-11-18", "isa_ticker_20241118"),
):
    oi, ni = names.index(old), names.index(new)
    source = sources[source_key]
    if sha256_file(Path(source["path"])) != source["sha256"]:
        raise ValueError("issuer rename notice changed")
    old_seen = np.flatnonzero(np.isfinite(close[:, oi]) & (close[:, oi] > 0))
    new_seen = np.flatnonzero(np.isfinite(close[:, ni]) & (close[:, ni] > 0))
    after = dates >= np.datetime64(event)
    lost = after & np.isfinite(close[:, ni]) & (close[:, ni] > 0) & ~active[:, ni]
    next_active = np.flatnonzero(after & active[:, ni])
    records.append(
        {
            "old_isin": old,
            "new_isin": new,
            "issuer_effective_date": event,
            "source": source,
            "old_last_quote_date": str(dates[old_seen[-1]]),
            "new_first_quote_date": str(dates[new_seen[0]]),
            "quotes_on_consecutive_store_sessions": bool(
                new_seen[0] == old_seen[-1] + 1
            ),
            "old_active_days_after_last_quote": int(
                active[old_seen[-1] + 1 :, oi].sum()
            ),
            "new_first_active_date": str(dates[next_active[0]])
            if len(next_active)
            else None,
            "quoted_successor_inactive_dates": [str(d) for d in dates[lost]],
            "existing_direct_successor_actions": int((successor[:, oi] == ni).sum()),
            "interpretation": "Issuer establishes ticker continuity; store uses distinct ISIN axes with no assumed automatic economic succession. Verify dated share-class/ISIN mapping before repairing account/targets/history. Inactive successor dates alone do not prove all universe requirements were met.",
        }
    )
output = {
    "store": design["store"],
    "script_sha256": sha256_file(Path(__file__)),
    "boundaries": records,
    "labels_accessed": False,
    "source_or_store_mutated": False,
}
write_json_atomic(root / "identity_boundaries.json", output)
print(
    {
        r["old_isin"]: {
            "inactive_quoted_successor_dates": len(
                r["quoted_successor_inactive_dates"]
            ),
            "existing_successions": r["existing_direct_successor_actions"],
        }
        for r in records
    }
)
