"""Independently reconcile cached account numerators with their original rows."""

from collections import Counter, defaultdict
from datetime import date, datetime
from decimal import Decimal
import io
import json
from pathlib import Path
import pickle
import re
import time
from xml.etree import ElementTree as ET
import zipfile

from lxml import html
import polars as pl

from brazil_rv.v2.artifacts import sha256_file, write_json_atomic
from brazil_rv.v2.data_repair import binding, bound_json

PROJECT = Path(__file__).resolve().parents[1]


def verify_reviewed_units(documents, dispositions):
    """The reviewed after-state is mandatory, even if a parser resolves units."""
    for identifier, row in dispositions.items():
        expected = json.loads(json.dumps(row["accounts_before"]))
        for book in expected.values():
            for account in book.values():
                account["value"] *= float(row["multiplier"])
        actual = json.loads(json.dumps(documents[identifier]["accounts"], default=str))
        assert actual == expected, identifier


def rows_html(payload, reference):
    tree = html.fromstring(payload.decode("utf8"))
    title = " ".join(tree.xpath("//h2//text()")).lower()
    assert "reais" in title, title
    scale = 1000 if re.search(r"reais\s+mil", title) else 1
    rows = [
        [" ".join(c.text_content().split()) for c in r.xpath("./th|./td")]
        for r in tree.xpath("//table//tr")
    ]
    header = next(r for r in rows if len(r) > 2 and r[0] == "Conta")
    for column, label in enumerate(header[2:], 2):
        dates = [
            datetime.strptime(v, "%d/%m/%Y").date()
            for v in re.findall(r"\d{2}/\d{2}/\d{4}", label)
        ]
        if not dates or dates[-1] != reference:
            continue
        for row in rows:
            if (
                len(row) <= column
                or not re.fullmatch(r"\d+(?:\.\d+)*", row[0])
                or not row[column]
            ):
                continue
            yield (
                row[0],
                row[1],
                dates[0] if len(dates) > 1 else None,
                dates[-1],
                Decimal(row[column].replace(".", "").replace(",", ".")) * scale,
                scale,
            )


def rows_zip(payload, document):
    with zipfile.ZipFile(io.BytesIO(payload)) as outer:
        envelope = ET.fromstring(
            outer.read(
                next(
                    n
                    for n in outer.namelist()
                    if n.startswith("FormularioDemonstracaoFinanceira")
                    and n.endswith(".xml")
                )
            )
        )
        assert envelope.findtext("NumeroSequencialDocumento") == document["id"]
        assert int(envelope.findtext("NumeroVersaoDocumento")) == document["version"]
        assert envelope.findtext("DataReferenciaDocumento")[:10] == str(
            document["reference"]
        )
        assert envelope.findtext("CodigoMoeda") == "1"
        scale = {"1": 1, "2": 1000}[envelope.findtext("CodigoEscalaMoeda")]
        nested = [n for n in outer.namelist() if n.lower().endswith((".itr", ".dfp"))]
        if nested:
            assert len(nested) == 1
            with zipfile.ZipFile(io.BytesIO(outer.read(nested[0]))) as inner:
                periods = {
                    n.findtext("NumeroIdentificacaoPeriodo"): n
                    for n in ET.fromstring(
                        inner.read("PeriodoDemonstracaoFinanceira.xml")
                    )
                }
                for node in ET.fromstring(
                    inner.read("InformacaoFinanceiraDemonstracaoFinanceira.xml")
                ):
                    code = node.findtext("PlanoConta/NumeroConta")
                    basis = {"1": "ind", "2": "con"}.get(
                        node.findtext(
                            "PlanoConta/VersaoPlanoConta/CodigoTipoInformacaoFinanceira"
                        )
                    )
                    if basis is None:
                        continue
                    description = node.findtext(
                        "DescricoesContaInformacaoFinanceiraDemonstracaoFinanceira/DescricaoContaInformacaoFinanceiraDemonstracaoFinanceira/DescricaoConta",
                        "",
                    )
                    for cell in node.find(
                        "ColunasInformacaoFinanceiraDemonstracaoFinanceira"
                    ):
                        period = periods[
                            cell.findtext(
                                "PeriodoDemonstracaoFinanceira/NumeroIdentificacaoPeriodo"
                            )
                        ]
                        end = date.fromisoformat(period.findtext("DataFimPeriodo")[:10])
                        flow = code.startswith(("3", "6"))
                        if end != document["reference"] or (
                            flow and period.findtext("NumeroTrimestre") != "0"
                        ):
                            continue
                        start = (
                            date.fromisoformat(
                                period.findtext("DataInicioPeriodo")[:10]
                            )
                            if flow
                            else None
                        )
                        yield (
                            basis,
                            code,
                            description,
                            start,
                            end,
                            Decimal(cell.findtext("ValorConta")) * scale,
                            scale,
                        )
        else:
            names = [
                n
                for n in outer.namelist()
                if re.fullmatch(r"\d+DFP\d{2}-\d{2}-\d{4}v\d+\.xml", n)
            ]
            assert len(names) == 1
            tree = ET.fromstring(outer.read(names[0]))
            assert (
                re.sub(r"\D", "", tree.findtext("DadosEmpresa/CnpjEmpresa"))
                == document["cnpj"]
            )
            annual = tree.find("DadosDFP")
            start = datetime.strptime(
                annual.findtext("DtInicioUltimoExercicioSocial"), "%d/%m/%Y"
            ).date()
            end = datetime.strptime(
                annual.findtext("DtFimUltimoExercicioSocial"), "%d/%m/%Y"
            ).date()
            assert end == document["reference"]
            for basis, tag in [
                ("ind", "DfIndividuais"),
                ("con", "DfConsolidadas"),
            ]:
                group = annual.find("Formulario/" + tag)
                if group is None:
                    continue
                for node in group.iter("Conta"):
                    code = node.findtext("CodigoConta")
                    value = node.findtext("UltimoExercicio")
                    if code and value:
                        yield (
                            basis,
                            code,
                            node.findtext("DescricaoConta", ""),
                            start if code.startswith(("3", "6")) else None,
                            end,
                            Decimal(value.replace(".", "").replace(",", ".")) * scale,
                            scale,
                        )


def main():
    started = time.perf_counter()
    run = json.loads(
        (PROJECT / "docs/v2_economic_data_scaling_run.json").read_text(encoding="utf8")
    )
    output = Path(run["root"]) / "financial_account_audit"
    output.mkdir(exist_ok=False)
    accepted = json.loads(
        (PROJECT / "docs/v2_data_inputs.json").read_text(encoding="utf8")
    )
    family = bound_json(accepted["financial_family"])
    original = bound_json(family["original_family"])
    propagation = bound_json(run["fca_financial_propagation"])
    caches = [
        family["extraction"]["cache"],
        propagation["artifacts"]["new_issuer_documents"],
    ]
    documents, evidence = [], []
    for source in caches:
        assert sha256_file(Path(source["path"])) == source["sha256"]
        ds, ev = pickle.loads(Path(source["path"]).read_bytes())
        documents.extend(ds)
        evidence.append(ev)
    by_id = {d["id"]: d for d in documents}
    assert len(by_id) == len(documents)
    assert all(
        d["reference"] <= date(2024, 12, 31) and d["receipt"] <= date(2024, 12, 31)
        for d in documents
    )
    sources, matches, differences, summaries = {}, [], [], Counter()
    checked = set()

    def read(path, digest):
        path = Path(path)
        data = path.read_bytes()
        import hashlib

        assert hashlib.sha256(data).hexdigest() == digest, str(path)
        sources[str(path)] = digest
        return data

    overrides = {}
    for ev in evidence:
        for key in ("account_units", "repair_units"):
            entry = ev.get(key)
            if entry:
                dispositions = json.loads(read(entry["path"], entry["sha256"]))
                for row in dispositions["documents"]:
                    if row["document"]["id"] in by_id:
                        overrides[row["document"]["id"]] = row

    verify_reviewed_units(by_id, overrides)

    def compare(document, raw, channel):
        identifier = document["id"]
        indexed = defaultdict(list)
        for basis, code, description, start, end, value, scale in raw:
            indexed[(basis, code, start, end)].append((description, value, scale))
        disposition = overrides.get(identifier)
        multiplier = (
            Decimal(str(disposition["multiplier"])) if disposition else Decimal(1)
        )
        if disposition:
            assert all(
                str(document[k]) == str(v) for k, v in disposition["document"].items()
            )
        for basis, accounts in document.get("accounts", {}).items():
            for metric, account in accounts.items():
                candidates = indexed.get(
                    (basis, account["source_code"], account["start"], account["end"]),
                    [],
                )
                actual = account["value"]
                valid = [
                    (desc, value * multiplier, scale)
                    for desc, value, scale in candidates
                    if desc == account["description"]
                ]
                # A later parser may already resolve the reviewed own-note scale.
                if disposition and not any(
                    abs(float(v) - actual) <= max(1e-6, abs(actual) * 2e-15)
                    for _, v, _ in valid
                ):
                    valid = [
                        (desc, value, scale)
                        for desc, value, scale in candidates
                        if desc == account["description"]
                    ]
                ok = any(
                    abs(float(v) - actual) <= max(1e-6, abs(actual) * 2e-15)
                    for _, v, _ in valid
                )
                record = {
                    "id": identifier,
                    "basis": basis,
                    "metric": metric,
                    "source_code": account["source_code"],
                    "start": str(account["start"]),
                    "end": str(account["end"]),
                    "cached": actual,
                    "channel": channel,
                }
                if not ok:
                    differences.append(
                        {
                            **record,
                            "candidates": [(x, str(v), s) for x, v, s in candidates],
                            "multiplier": str(multiplier),
                        }
                    )
                else:
                    summaries[(channel, metric)] += 1
                    matches.append(
                        {
                            **record,
                            "description": account["description"],
                            "scale": valid[0][2],
                            "note_multiplier": float(multiplier),
                        }
                    )
        checked.add(identifier)

    groups = defaultdict(list)
    for d in documents:
        groups[(d["kind"], d["cnpj"], str(d["reference"]), d["version"])].append(d)
    csv_documents = {
        key: ds[0]
        for key, ds in groups.items()
        if len(ds) == 1 and not ds[0].get("recovered_original")
    }
    expected_codes = {
        a["source_code"]
        for d in documents
        for book in d.get("accounts", {}).values()
        for a in book.values()
    }
    annual = bound_json(original["source_manifests"]["annual_manifest.json"])
    for source in annual["files"]:
        if source["kind"] not in ("itr", "dfp"):
            continue
        path = Path(source["path"])
        with zipfile.ZipFile(io.BytesIO(read(path, source["sha256"]))) as archive:
            headers = pl.read_csv(
                io.BytesIO(archive.read(path.stem + ".csv")),
                separator=";",
                encoding="windows-1252",
                infer_schema_length=0,
            )
            for row in headers.iter_rows(named=True):
                identifier = re.sub(r"\D", "", row["ID_DOC"])
                if identifier in by_id:
                    d = by_id[identifier]
                    assert (
                        row["DT_REFER"],
                        int(row["VERSAO"]),
                        re.sub(r"\D", "", row["CNPJ_CIA"]),
                        row["CD_CVM"].zfill(6),
                        row["DT_RECEB"],
                    ) == (
                        str(d["reference"]),
                        d["version"],
                        d["cnpj"],
                        d["cvm_code"],
                        str(d["receipt"]),
                    )
            selected = defaultdict(list)
            for name in archive.namelist():
                m = re.fullmatch(
                    r"(itr|dfp)_cia_aberta_(BPA|BPP|DRE|DFC_MI|DFC_MD)_(con|ind)_\d{4}\.csv",
                    name,
                )
                if not m:
                    continue
                frame = pl.read_csv(
                    io.BytesIO(archive.read(name)),
                    separator=";",
                    encoding="windows-1252",
                    infer_schema_length=0,
                    quote_char=None,
                ).filter(
                    pl.col("CD_CONTA").is_in(expected_codes)
                    & (pl.col("ORDEM_EXERC") == "ÚLTIMO")
                )
                for row in frame.iter_rows(named=True):
                    key = (
                        m[1].upper(),
                        re.sub(r"\D", "", row["CNPJ_CIA"]),
                        row["DT_REFER"],
                        int(row["VERSAO"]),
                    )
                    if key not in csv_documents or row["VL_CONTA"] is None:
                        continue
                    assert row["MOEDA"] == "REAL"
                    scale = {"MIL": 1000, "UNIDADE": 1}[row["ESCALA_MOEDA"]]
                    selected[key].append(
                        (
                            m[3],
                            row["CD_CONTA"],
                            row["DS_CONTA"],
                            date.fromisoformat(row["DT_INI_EXERC"])
                            if row.get("DT_INI_EXERC")
                            else None,
                            date.fromisoformat(row["DT_FIM_EXERC"]),
                            Decimal(row["VL_CONTA"]) * scale,
                            scale,
                        )
                    )
            for key, raw in selected.items():
                compare(csv_documents[key], raw, "annual_csv")
        print(
            json.dumps(
                {
                    "stage": "annual_accounts",
                    "file": path.name,
                    "documents_checked": len(checked),
                    "differences": len(differences),
                }
            ),
            flush=True,
        )

    original_sources = defaultdict(list)
    for ev in evidence:
        for source in ev["original_sources"]:
            original_sources[source["document_id"]].append(source)
    for identifier, document in by_id.items():
        if identifier in checked or not document.get("accounts"):
            continue
        raw = []
        receipts = original_sources[identifier]
        html_sources = [
            s for s in receipts if "original_zips" not in Path(s["manifest_path"]).parts
        ]
        for source in html_sources:
            path = Path(source["manifest_path"])
            manifest = json.loads(read(path, source["manifest_sha256"]))
            for key in ("id", "version", "reference", "cnpj"):
                assert str(manifest["document"][key]) == str(document[key])
            for page in manifest["pages"]:
                raw.extend(
                    (page["basis"], *row)
                    for row in rows_html(
                        read(path.parent / page["file"], page["sha256"]),
                        document["reference"],
                    )
                )
        if raw:
            compare(document, raw, "own_version_html")
        else:
            zips = [
                s for s in receipts if "original_zips" in Path(s["manifest_path"]).parts
            ]
            assert len(zips) == 1, identifier
            source = zips[0]
            path = Path(source["manifest_path"])
            manifest = json.loads(read(path, source["manifest_sha256"]))
            compare(
                document,
                rows_zip(
                    read(path.parent / "source.zip", manifest["sha256"]), document
                ),
                "own_version_zip",
            )
    missing = [
        d["id"] for d in documents if d.get("accounts") and d["id"] not in checked
    ]
    assert not missing, missing
    pl.DataFrame(matches).write_parquet(output / "account_rows.parquet")
    write_json_atomic(output / "differences.json", differences)
    write_json_atomic(
        output / "source_receipts.json",
        [{"path": p, "sha256": s} for p, s in sorted(sources.items())],
    )
    report = {
        "schema": "FINANCIAL_ACCOUNT_SOURCE_AUDIT_V1",
        "caches": caches,
        "documents": len(documents),
        "documents_with_accounts_checked": len(checked),
        "unavailable_accounts": [d["id"] for d in documents if not d.get("accounts")],
        "account_rows_verified": len(matches),
        "differences": len(differences),
        "counts": [
            {"channel": ch, "metric": m, "rows": n}
            for (ch, m), n in sorted(summaries.items())
        ],
        "reviewed_unit_dispositions": len(overrides),
        "source_files": len(sources),
        "artifacts": {
            p.stem: binding(p)
            for p in output.iterdir()
            if p.is_file() and p.suffix in (".json", ".parquet")
        },
        "reproducer": binding(Path(__file__)),
        "seconds": time.perf_counter() - started,
        "limits": [
            "Verifies every cached selected account amount, period, published description, basis and currency scale; it does not prove that every useful non-selected account was admitted.",
            "Own-note unit dispositions reuse their existing source interpretation; raw published values and the reviewed multiplier are both retained.",
            "The source archive is one vintage; own-version identities do not prove historical converter/revision fidelity.",
            "Financial TTM, valuation, free-float and final tensor propagation remain separate audit boundaries.",
        ],
    }
    write_json_atomic(output / "report.json", report)
    print(
        json.dumps(
            {
                k: v
                for k, v in report.items()
                if k not in ("counts", "artifacts", "limits")
            },
            default=str,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
