"""Resolve AERI's inherited final-session diagnostic against its source lineage."""

import argparse
import base64
import html
import json
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import polars as pl
import requests
from pypdf import PdfReader

from brazil_rv.v2.artifacts import write_json_atomic
from brazil_rv.v2.data_repair import binding, bound_json

PROJECT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--poppler-dir", required=True, type=Path)
    args = parser.parse_args()
    run = json.loads((PROJECT / "docs/v2_economic_data_scaling_run.json").read_text())
    source = bound_json(run["m1_scalar_assembly"])
    store = Path(source["parent"]["root"])
    manifest = bound_json(
        {
            "path": str(store / "manifest.json"),
            "sha256": source["parent"]["manifest_sha256"],
        }
    )
    root = Path(run["root"]) / "aeri_boundary"
    root.mkdir(exist_ok=True)
    shutil.copyfile(__file__, root / "qualified_reproducer.py")
    round7 = bound_json(manifest["metadata"]["round7_repair"]["source_audit"])
    dispositions = bound_json(round7["u2_events"])
    review = next(
        r
        for r in dispositions
        if r["isin"] == "BRAERIACNOR4" and r["date"] == "2024-12-30"
    )
    assert (
        review["classification"] == "large_move_no_action"
        and not review["corroboration"]
    )
    day, name = review["date_index"], review["name_index"]
    expected = {
        "action_has_action": False,
        "action_shares_per_prior_share": 1,
        "action_cash_per_prior_share": 0,
        "action_session_resolved": True,
        "action_successor_index": name,
        "inferred_action_large_move_no_action_mask": True,
        "m1_cotahist_return_consistent_mask": False,
    }
    for key, value in expected.items():
        assert np.load(store / (key + ".npy"), mmap_mode="r")[day, name] == value
    table_rows, tables = {}, {}
    for key in [
        "corporate_actions_verified_terms",
        "corporate_action_alignment_roles",
        "m1_cotahist_level_ratio",
        "cotahist_action_classification",
    ]:
        path = store / manifest["tables"][key]["path"]
        table = pl.read_parquet(path).filter(pl.col("isin") == review["isin"])
        date_key = next(
            c
            for c in ["event_date", "effective_date", "trade_date", "date"]
            if c in table.columns
        )
        selected = table.filter(pl.col(date_key).cast(pl.String) == review["date"])
        table_rows[key] = selected.to_dicts()
        tables[key] = binding(path)
    assert not table_rows["corporate_actions_verified_terms"]
    assert (
        table_rows["corporate_action_alignment_roles"][0]["source"]
        == "inferred_cotahist_dismes_v1"
    )
    # Follow the already bound RAD source manifest, reading only its 2024 files.
    family = bound_json(round7["cvm_family_manifest"])
    cvm_root = Path(round7["cvm_family_manifest"]["path"]).parent.parent
    rad_path = cvm_root / "rad_manifest.json"
    rad = json.loads(rad_path.read_text(encoding="utf8"))
    issuer_rows, source_receipts = [], []
    for rec in rad["files"]:
        path = Path(rec["path"])
        if "2024" not in path.name:
            continue
        data = json.loads(path.read_text(encoding="utf8"))["d"]["dados"]
        matches = []
        for raw in data.split("$&&*"):
            fields = raw.split("$&")
            if len(fields) < 10 or re.sub(r"\D", "", fields[0]) != "025283":
                continue

            def clean(v):
                return html.unescape(re.sub("<[^>]+>", "", v)).strip()

            protocol = re.search(r"NumeroProtocoloEntrega=(\d+)", raw)
            matches.append(
                {
                    "category": clean(fields[2]),
                    "subject": clean(fields[4]),
                    "reference": clean(fields[5]),
                    "receipt": clean(fields[6]),
                    "protocol": protocol[1] if protocol else None,
                    "raw": raw,
                }
            )
        if matches:
            issuer_rows.extend(matches)
            source_receipts.append(binding(path))
    write_json_atomic(
        root / "issuer_2024_rows.json",
        {
            "rows": issuer_rows,
            "source_receipts": source_receipts,
            "rad_manifest": binding(rad_path),
            "family_schema": family.get("schema"),
        },
    )
    pdfs = []
    for protocol in ("1312607", "1314227"):
        row = next(r for r in issuer_rows if r["protocol"] == protocol)
        url = "https://www.rad.cvm.gov.br/ENET/frmExibirArquivoIPEExterno.aspx"
        payload = {
            "codigoInstituicao": "1",
            "numeroProtocolo": protocol,
            "token": "",
            "versaoCaptcha": "",
        }
        path = root / (protocol + ".pdf")
        if not path.exists():
            viewer = requests.get(
                url, params={"NumeroProtocoloEntrega": protocol}, timeout=40
            )
            viewer.raise_for_status()
            (root / (protocol + "_viewer.html")).write_bytes(viewer.content)
            assert 'id="hdnHabilitaCaptcha" value="N"' in viewer.text
            response = requests.post(url + "/ExibirPDF", json=payload, timeout=40)
            response.raise_for_status()
            pdf = base64.b64decode(response.json()["d"], validate=True)
            assert pdf.startswith(b"%PDF-")
            path.write_bytes(pdf)
        path.with_suffix(".txt").write_text(
            "\n".join(page.extract_text() for page in PdfReader(path).pages),
            encoding="utf8",
        )
        subprocess.run(
            [
                str(args.poppler_dir / "pdftoppm.exe"),
                "-f",
                "1",
                "-singlefile",
                "-scale-to",
                "1600",
                "-png",
                str(path),
                str(path.with_suffix("")),
            ],
            check=True,
        )
        pdfs.append(
            {
                "source": binding(path),
                "viewer_url": url + "?NumeroProtocoloEntrega=" + protocol,
                "request": payload,
                "receipt": row["receipt"],
                "reference": row["reference"],
                "protocol": protocol,
            }
        )
    raw = np.load(store / "raw_close.npy", mmap_mode="r")
    with np.load(
        Path(run["m1_scalar_assembly"]["path"]).parent.parent
        / "qualified_control/2024-11-18/raw_control.npz"
    ) as scalar:
        last = int(np.flatnonzero(scalar["date_indices"] == day)[0])
        observed = {
            k: scalar[k][last, name].item()
            for k in [
                "return_consistent",
                "entry",
                "session_close",
                "realized_daily_vol",
                "entry_valid",
                "session_close_valid",
                "fast_present",
            ]
        }
    # A q1/cash0 diagnostic is justified by the recorded source disposition, not
    # by inferring an inverse factor from the price fall. Leave all data immutable.
    write_json_atomic(
        root / "manifest.json",
        {
            "parent": source["parent"],
            "date": review["date"],
            "isin": review["isin"],
            "round7_disposition": review,
            "round7_source": round7["u2_events"],
            "inherited_table_scope": manifest["metadata"]["round7_repair"][
                "audit_table_scope"
            ],
            "tables": tables,
            "rows": json.loads(json.dumps(table_rows, default=str)),
            "accepted_arrays": expected,
            "raw_prices": np.round(
                raw[day - 1 : day + 1, name].astype(float), 2
            ).tolist(),
            "raw_simple_return": float(5.71 / 8.31 - 1),
            "independently_reconstructed_m1_consistency": observed["return_consistent"],
            "target_supported": False,
            "target_rejection": "Pre-decision five-minute return RSS lacks the unchanged support; fast_present false, realized_daily_vol NaN. Exact entry/close exist but do not override support.",
            "proposed_final_diagnostic_patch": {
                "m1_cotahist_return_consistent_mask": True,
                "completed_action_boundary": False,
            },
            "feature_or_target_changes_on_2010_2024_axis": 0,
            "issuer_rows": binding(root / "issuer_2024_rows.json"),
            "issuer_documents": pdfs,
            "executed_reproducer": binding(root / "qualified_reproducer.py"),
            "status": "Lineage resolved: inherited pre-Round7 retrospective inference is not a surviving admitted action. No inferred bonus restored. Final derived diagnostic update must cite Round7 disposition; no current accepted arrays changed.",
            "limitations": "The absence of corroboration is not proof that every issuer event was collected. Debt notices do not supply equity bonus terms and do not identify the cause of the share-price move.",
        },
    )
    print(json.dumps({"issuer_records": len(issuer_rows), "pdfs": pdfs}), flush=True)


if __name__ == "__main__":
    main()
