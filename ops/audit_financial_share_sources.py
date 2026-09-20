"""Audit incremental capital tables and original dated FRE float denominators."""

import csv
from datetime import date, datetime, time as clock, timedelta
import hashlib
import html
import io
import json
from pathlib import Path
import pickle
import re
import time
import zipfile

import numpy as np
import polars as pl
from polars.testing import assert_frame_equal

from audit_cvm_sources import capital_from_html, outstanding
from brazil_rv.v2.artifacts import write_json_atomic
from brazil_rv.v2.data_repair import binding, bound_json

PROJECT = Path(__file__).resolve().parents[1]


def main():
    started = time.perf_counter()
    run = json.loads(
        (PROJECT / "docs/v2_economic_data_scaling_run.json").read_text(encoding="utf8")
    )
    output = Path(run["root"]) / "financial_share_source_audit"
    output.mkdir(exist_ok=False)
    accepted = json.loads(
        (PROJECT / "docs/v2_data_inputs.json").read_text(encoding="utf8")
    )
    family = bound_json(accepted["financial_family"])
    original = bound_json(family["original_family"])
    propagation = bound_json(run["fca_financial_propagation"])
    sources = {}

    def read(path, digest):
        payload = Path(path).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == digest, str(path)
        sources[str(path)] = digest
        return payload

    cache = propagation["artifacts"]["new_issuer_documents"]
    documents, evidence = pickle.loads(read(cache["path"], cache["sha256"]))
    by_id = {d["id"]: d for d in documents}
    final = {}
    overrides = {}
    capital_tables = 0
    for source in evidence["capital_sources"]:
        path = Path(source["manifest_path"])
        manifest = json.loads(read(path, source["manifest_sha256"]))
        d = by_id[source["document_id"]]
        if "disposition" in source:
            row = next(
                r for r in manifest["documents"] if r["document"]["id"] == d["id"]
            )
            assert all(str(d[k]) == str(v) for k, v in row["document"].items())
            overrides[d["id"]] = (
                outstanding(row["paid_in_shares"], row["treasury_shares"])
                if row["disposition"] == "reconciled"
                else None
            )
            for receipt in row["evidence"]:
                read(receipt["path"], receipt["sha256"])
            continue
        assert all(str(d[k]) == str(v) for k, v in manifest["document"].items())
        for page in manifest["sources"]:
            if page["role"] == "capital":
                parsed = capital_from_html(
                    read(path.parent / page["file"], page["sha256"]), d
                )
                assert parsed == manifest["capital"], d["id"]
                final[d["id"]] = parsed["shares"]
                capital_tables += 1
    final.update(overrides)
    assert set(final) == set(by_id)
    assert all(final[k] == by_id[k].get("shares") for k in final)
    annual = bound_json(original["source_manifests"]["annual_manifest.json"])
    headers = {}
    rows = []
    for source in annual["files"]:
        if source["kind"] != "fre":
            continue
        path = Path(source["path"])
        with zipfile.ZipFile(io.BytesIO(read(path, source["sha256"]))) as archive:
            for row in csv.DictReader(
                io.StringIO(archive.read(path.stem + ".csv").decode("latin1")),
                delimiter=";",
            ):
                if row["DT_RECEB"] > "2024-12-30":
                    continue
                identifier = re.sub(r"\D", "", row["ID_DOC"])
                document = {
                    "id": identifier,
                    "cnpj": re.sub(r"\D", "", row["CNPJ_CIA"]),
                    "cvm_code": row["CD_CVM"].zfill(6),
                    "reference": date.fromisoformat(row["DT_REFER"]),
                    "version": int(row["VERSAO"]),
                    "receipt": date.fromisoformat(row["DT_RECEB"]),
                }
                assert identifier not in headers or headers[identifier] == document
                headers[identifier] = document
            name = (
                path.stem.replace("cia_aberta", "cia_aberta_distribuicao_capital")
                + ".csv"
            )
            rows.extend(
                csv.DictReader(
                    io.StringIO(archive.read(name).decode("latin1")), delimiter=";"
                )
            )
    rad = bound_json(original["source_manifests"]["rad_manifest.json"])
    receipts = {}
    for source in rad["files"]:
        if source["group"] != "cadastre":
            continue
        raw = json.loads(read(source["path"], source["sha256"]))["d"]["dados"]
        for record in raw.split("$&&*"):
            match = re.search(r"NumeroSequencialDocumento=(\d+)", record) or re.search(
                r"OpenDownloadDocumentos\('([0-9]+)'", record
            )
            if match is None or match[1] not in headers:
                continue
            parts = [
                html.unescape(re.sub("<[^>]+>", "", x)).strip()
                for x in record.split("$&")
            ]
            stamp = re.search(r"(\d{2}/\d{2}/\d{4})\s+(\d{2}:\d{2})", parts[6])
            instant = datetime.strptime(" ".join(stamp.groups()), "%d/%m/%Y %H:%M")
            if instant.date() > date(2024, 12, 30):
                continue
            d = headers[match[1]]
            assert re.sub(r"\D", "", parts[0]).zfill(6) == d["cvm_code"]
            if match[1] in receipts:
                assert receipts[match[1]] == instant
            receipts[match[1]] = instant
    days = (
        np.load(Path(accepted["store"]["root"]) / "date_index.npy")
        .astype("datetime64[D]")
        .tolist()
    )
    cutoffs = np.array(
        [datetime.combine(d, clock(15, 45)) for d in days], dtype="datetime64[us]"
    )
    reconstructed = []
    details = []
    for row in rows:
        identifier = re.sub(r"\D", "", row["ID_Documento"])
        if identifier not in headers:
            continue
        d = headers[identifier]
        assert (
            re.sub(r"\D", "", row["CNPJ_Companhia"]),
            row["Data_Referencia"],
            int(row["Versao"]),
        ) == (d["cnpj"], str(d["reference"]), d["version"])
        stamp = receipts.get(identifier)
        known = (
            stamp + timedelta(minutes=1)
            if stamp
            else datetime.combine(d["receipt"] + timedelta(days=1), clock())
        )
        index = int(np.searchsorted(cutoffs, np.datetime64(known)))
        if index >= len(days):
            continue
        snapshot = (
            date.fromisoformat(row["Data_Ultima_Assembleia"])
            if row["Data_Ultima_Assembleia"]
            else None
        )
        for cls, label in [("ON", "Ordinarias"), ("PN", "Preferenciais")]:
            raw = row["Quantidade_Acoes_" + label + "_Circulacao"]
            if raw and float(raw) > 0:
                reconstructed.append(
                    {
                        "date": days[index],
                        "cnpj": d["cnpj"],
                        "cvm_code": d["cvm_code"],
                        "class": cls,
                        "free_float_shares": float(raw),
                        "document_id": identifier,
                        "version": d["version"],
                        "reference": d["reference"],
                        "snapshot_date": snapshot,
                    }
                )
                details.append(
                    {
                        "id": identifier,
                        "class": cls,
                        "decision": str(days[index]),
                        "known_upper_bound": str(known),
                        "snapshot_date": str(snapshot),
                        "reported_circulating_shares": raw,
                        "exact_receipt": stamp is not None,
                    }
                )
    source = (
        Path(family["original_family"]["path"]).parent
        / "free_float_observations.parquet"
    )
    read(source, original["files"][source.name]["sha256"])
    before = pl.read_parquet(source)
    after = pl.DataFrame(reconstructed, schema=before.schema)
    assert_frame_equal(
        before.sort(before.columns), after.sort(after.columns), check_exact=True
    )
    write_json_atomic(output / "float_rows.json", details)
    write_json_atomic(
        output / "source_receipts.json",
        [{"path": p, "sha256": s} for p, s in sorted(sources.items())],
    )
    report = {
        "schema": "FINANCIAL_SHARE_SOURCE_AUDIT_V1",
        "new_document_cache": cache,
        "incremental_capital_tables": capital_tables,
        "incremental_note_dispositions": len(overrides),
        "incremental_final_capital_counts": len(final),
        "free_float_source": binding(source),
        "free_float_rows": len(after),
        "free_float_exact_receipts": sum(r["exact_receipt"] for r in details),
        "future_snapshot_rows": sum(
            r["snapshot_date"] is not None and r["snapshot_date"] > r["date"]
            for r in reconstructed
        ),
        "missing_snapshot_rows": sum(r["snapshot_date"] is None for r in reconstructed),
        "source_files": len(sources),
        "mismatches": 0,
        "seconds": time.perf_counter() - started,
        "reproducer": binding(Path(__file__)),
        "artifacts": {
            p.stem: binding(p)
            for p in output.iterdir()
            if p.is_file() and p.suffix == ".json"
        },
        "limits": [
            "FRE quantities are published circulating shares, not daily executable float or locate capacity. Measurement date and filing availability remain distinct.",
            "Only 427 newly extracted capital tables are rechecked; the sealed 21783-document capital audit is reused.",
            "Single-vintage revisions and original-source semantic misstatements remain possible; reconciliation does not make those observations contemporaneous archives.",
            "No source, store, neural tensor or forecast was changed.",
        ],
    }
    write_json_atomic(output / "report.json", report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
