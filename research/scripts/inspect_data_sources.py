"""Inventory consumed CVM books and cross-period scale discontinuities, read-only."""

import argparse
import json
import pickle
from collections import defaultdict
from pathlib import Path

import polars as pl

from brazil_rv.v2.round5_cvm import load_financial_documents

project = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser()
parser.add_argument("--root", type=Path, required=True)
output = parser.parse_args().root
output.mkdir(parents=True, exist_ok=True)
pointer = json.loads(
    (project / "docs/v2_round5_cvm_final_acceptance.json").read_text()
)["family_manifest"]
manifest = json.loads(Path(pointer["path"]).read_text())
identity = pl.read_parquet(Path(pointer["path"]).parent / "identity.parquet")
documents, evidence = load_financial_documents(
    Path(manifest["source_root"]), set(identity["cnpj"])
)
with (output / "accepted_documents.pkl").open("wb") as handle:
    pickle.dump(documents, handle)
(output / "consumed_sources.json").write_text(
    json.dumps(evidence, default=str, indent=2)
)
by_issuer = defaultdict(list)
for d in documents:
    by_issuer[d["cnpj"], d["cvm_code"]].append(d)
candidates = []
for issuer, docs in by_issuer.items():
    ordered = sorted(docs, key=lambda d: (d["reference"], d["version"], int(d["id"])))
    for before, after in zip(ordered, ordered[1:]):
        changes = []
        for basis in ("con", "ind"):
            for field in ("assets", "equity"):
                old = (
                    before.get("accounts", {})
                    .get(basis, {})
                    .get(field, {})
                    .get("value")
                )
                new = (
                    after.get("accounts", {}).get(basis, {}).get(field, {}).get("value")
                )
                if old and new and 300 < abs(new / old) < 3000:
                    changes.append(
                        dict(
                            field=f"{basis}.{field}",
                            suspect=before["id"],
                            ratio=new / old,
                        )
                    )
                if old and new and 300 < abs(old / new) < 3000:
                    changes.append(
                        dict(
                            field=f"{basis}.{field}",
                            suspect=after["id"],
                            ratio=old / new,
                        )
                    )
        for share in ("ON", "PN"):
            old = (before.get("shares") or {}).get(share)
            new = (after.get("shares") or {}).get(share)
            if old and new and 300 < new / old < 3000:
                changes.append(
                    dict(field=f"shares.{share}", suspect=before["id"], ratio=new / old)
                )
            if old and new and 300 < old / new < 3000:
                changes.append(
                    dict(field=f"shares.{share}", suspect=after["id"], ratio=old / new)
                )
        if changes:
            candidates.append(
                dict(
                    issuer=issuer,
                    before={k: before[k] for k in ("id", "reference", "receipt")},
                    after={k: after[k] for k in ("id", "reference", "receipt")},
                    changes=changes,
                )
            )
(output / "scale_candidates.json").write_text(
    json.dumps(candidates, default=str, indent=2)
)
print(
    json.dumps(
        dict(
            documents=len(documents),
            issuers=len(by_issuer),
            scale_candidate_pairs=len(candidates),
            output=str(output),
        )
    )
)
