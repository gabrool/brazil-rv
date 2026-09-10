from datetime import date
import io
import json
import urllib.error
import zipfile

import polars as pl
import pytest

from brazil_rv.v2.round5_cvm_fca import (
    _general_html,
    _generic_html_securities,
    original_fca,
    recover_fca,
    load_fca,
)
from brazil_rv.v2.round5_cvm import build_identity, sha256


def original_fixture(
    tmp_path, *, public_id="70793", inner_version=1, modern=False, inner_cnpj=None
):
    document = dict(
        id="70793",
        cnpj="33592510000154",
        cvm_code="004170",
        kind="FCA",
        reference=date(2018, 1, 1),
        version=1,
        receipt=date(2018, 1, 3),
    )
    company = """<CompanhiaAberta><CodigoCvm>00417-0</CodigoCvm>
      <NumeroCnpjCompanhiaAberta>33.592.510/0001-54</NumeroCnpjCompanhiaAberta>
      <NomeRazaoSocialCompanhiaAberta>VALE S.A.</NomeRazaoSocialCompanhiaAberta>
      </CompanhiaAberta>"""
    envelope = f"""<Documento>{company}<NumeroSequencialDocumento>{public_id}</NumeroSequencialDocumento>
      <NumeroVersaoDocumento>1</NumeroVersaoDocumento>
      <DataReferenciaDocumento>2018-01-01T00:00:00</DataReferenciaDocumento></Documento>"""
    general = f"""<FormularioCadastral>{company}<CodigoSetorAtividadeEmpresa>1030</CodigoSetorAtividadeEmpresa>
      <Documento><NumeroSequencialDocumento>1294</NumeroSequencialDocumento>
      <DataReferenciaDocumento>2018-01-01T00:00:00</DataReferenciaDocumento>
      <NumeroVersaoDocumento>{inner_version}</NumeroVersaoDocumento></Documento></FormularioCadastral>"""
    if inner_cnpj is not None:
        general = general.replace("33.592.510/0001-54", inner_cnpj)
    security = """<ArrayOfValorMobiliario><ValorMobiliario>
      <ValorMobiliarioNegociado><Dominio><CodigoOpcao>1</CodigoOpcao></Dominio>
      <DescricaoOpcaoDominio>Ações</DescricaoOpcaoDominio></ValorMobiliarioNegociado>
      <MercadoNegociacao><DescricaoOpcaoDominio>Bolsa</DescricaoOpcaoDominio></MercadoNegociacao>
      <MercadosNegociacao><MercadoNegociacao>
      <Segmento><DescricaoOpcaoDominio>Novo Mercado</DescricaoOpcaoDominio></Segmento>
      <DataInicioNeg>2017-12-22T00:00:00</DataInicioNeg>
      <DataFimNeg>0001-01-01T00:00:00</DataFimNeg>
      <DataInicioRelc>1968-04-01T00:00:00</DataInicioRelc>
      </MercadoNegociacao></MercadosNegociacao></ValorMobiliario></ArrayOfValorMobiliario>"""
    if modern:
        security = security.replace("Ações</", "Ações Preferenciais</").replace(
            "</ValorMobiliario>",
            "<CodigoNegociacao>VALE5</CodigoNegociacao><ClasseAcao>"
            "<DescricaoOpcaoDominio>Classe A</DescricaoOpcaoDominio></ClasseAcao>"
            "<ComposicaoBDRUnit></ComposicaoBDRUnit></ValorMobiliario>",
        )
    nested = io.BytesIO()
    with zipfile.ZipFile(nested, "w") as archive:
        archive.writestr("FormularioCadastral.xml", general)
        archive.writestr("ValorMobiliarioMercadoNegociacao.xml", security)
    path = tmp_path / "source.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("FormularioCadastral.xml", envelope)
        archive.writestr("00417020180101101.fca", nested.getvalue())
    return document, path


def test_original_generic_equity_is_not_modern_ordinary_label(tmp_path):
    document, path = original_fixture(tmp_path)
    result = original_fca(document, path)
    assert result["legal_name"] == "VALE S.A."
    assert result["sector_code"] == "1030" and result["sector"] is None
    row = result["securities"][0]
    assert row["class"] == "SHARES" and row["ticker"] == ""
    assert row["source_segment_description"] == "Novo Mercado"
    assert row["start"] == "1968-04-01" and row["end"] == "9999-12-31"
    assert row["segment_start"] == "2017-12-22"


def test_historical_establishment_preserves_same_legal_issuer(tmp_path):
    document, path = original_fixture(tmp_path, inner_cnpj="33.592.510/0002-35")
    metadata = original_fca(document, path)
    assert metadata["source_cnpj"] == "33592510000235"
    assert document["cnpj"] == "33592510000154"  # Public header remains exact.
    general = b"""<table><tr><td>Nome Empresarial:</td><td>VALE S.A.</td></tr>
      <tr><td>C.N.P.J.:</td><td>33.592.510/0002-35</td><td>Codigo CVM:</td><td>00417-0</td></tr></table>"""
    assert _general_html(general, document)["source_cnpj"] == "33592510000235"
    with pytest.raises(ValueError, match="CVM registration"):
        _general_html(general, {**document, "cvm_code": "004171"})


@pytest.mark.parametrize("cnpj", ("33592511/0001-54", "33592510"))
def test_original_cannot_borrow_a_different_or_incomplete_legal_root(tmp_path, cnpj):
    document, path = original_fixture(tmp_path, inner_cnpj=cnpj)
    with pytest.raises(ValueError, match="Nested FCA identity differs"):
        original_fca(document, path)


def test_current_reader_recovers_original_ticker_and_preferred_class_immutably(
    tmp_path,
):
    document, path = original_fixture(tmp_path, modern=True)
    metadata = original_fca(document, path)
    row = metadata["securities"][0]
    assert (row["ticker"], row["class"], row["preferred_class"]) == (
        "VALE5",
        "PN",
        "A",
    )
    row["ticker"] = row["preferred_class"] = ""
    metadata.update(sector="Extração Mineral", sector_label_source="exact_fca_html")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "document": {
                    k: str(document[k])
                    for k in ("id", "cnpj", "cvm_code", "reference", "version", "kind")
                },
                "metadata": metadata,
                "sources": [{"path": str(path), "sha256": sha256(path)}],
            }
        ),
        encoding="utf8",
    )
    before = manifest.read_bytes()
    loaded = load_fca(document, tmp_path)
    assert loaded["securities"][0]["ticker"] == "VALE5"
    assert loaded["securities"][0]["preferred_class"] == "A"
    assert loaded["securities"][0]["source_class_description"] == "Classe A"
    assert loaded["sector"] == "Extração Mineral"
    assert manifest.read_bytes() == before


def test_original_preferred_class_joins_actual_b3_pna_only(tmp_path):
    document, path = original_fixture(tmp_path, modern=True)
    parsed = original_fca(document, path)
    sessions = [date(2018, 1, day) for day in (2, 3, 4, 5)]
    securities = parsed["securities"]
    for row in securities:
        row["start"], row["end"] = (
            date.fromisoformat(row["start"]),
            date.fromisoformat(row["end"]),
        )
    document.update(parsed, available_index=2, securities=securities)
    observations = pl.DataFrame(
        {
            "trade_date": [sessions[0]] * 2,
            "isin": ["PNA", "PNB"],
            "ticker": ["VALE5", "VALE6"],
            "issuer_short_name": ["VALE", "VALE"],
            "security_spec_base": ["PNA", "PNB"],
        }
    )
    identity = build_identity([document], observations, sessions, ["PNA", "PNB"])
    assert set(identity["isin"]) == {"PNA"}
    assert set(identity["preferred_class"]) == {"A"}
    assert identity["date"].min() == sessions[2]


@pytest.mark.parametrize("mutation", ["public_id", "inner_version"])
def test_original_fca_requires_both_public_envelope_and_nested_version(
    tmp_path, mutation
):
    args = {"public_id": "70794"} if mutation == "public_id" else {"inner_version": 2}
    document, path = original_fixture(tmp_path, **args)
    with pytest.raises(ValueError, match="differs"):
        original_fca(document, path)


def test_html_general_binds_issuer_and_never_promotes_rendered_class():
    document = {"cnpj": "33592510000154", "cvm_code": "004170"}
    general = """<table><tr><td>Nome Empresarial:</td><td>VALE S.A.</td></tr>
      <tr><td>C.N.P.J.:</td><td>33.592.510/0001-54</td><td>Código CVM:</td><td>00417-0</td></tr>
      <tr><td>Setor de Atividade:</td><td>Extração Mineral</td></tr></table>""".encode()
    assert _general_html(general, document)["sector"] == "Extração Mineral"
    with pytest.raises(ValueError, match="CNPJ differs"):
        _general_html(general, {**document, "cnpj": "00000000000000"})
    securities = """<button id="btnDado_1">Ações Ordinárias</button>
      <div id="divDado_1"><table><tr><td>Bolsa</td><td>B3</td><td>01/04/1968</td><td></td>
      <td>Nível 1</td><td>22/12/2017</td><td></td></tr></table></div>""".encode()
    result = _generic_html_securities(securities)
    assert len(result) == 1
    assert result[0]["class"] == "SHARES" and result[0]["ticker"] == ""
    assert "source_segment_description" not in result[0]


def test_transient_original_response_retries_before_html(tmp_path, monkeypatch):
    document, source = original_fixture(tmp_path)
    calls = []

    class Opener:
        def open(self, request, timeout):
            calls.append(request.full_url)
            if len(calls) == 1:
                return io.BytesIO(b"temporary backend error")
            if len(calls) == 2:
                return io.BytesIO(source.read_bytes())
            raise OSError("HTML unavailable")

    monkeypatch.setattr("urllib.request.build_opener", lambda *args: Opener())
    destination = tmp_path / "snapshot"
    result = recover_fca(document, destination)
    assert result["metadata"]["sector_code"] == "1030"
    assert len(calls) == 3 and "DownloadDocumento" in calls[1]
    assert result["helper_sha256"]
    assert (destination / "manifest.json").exists()
    assert (
        destination / "attempts/0001/original_response_1.bin"
    ).read_bytes() == b"temporary backend error"


def test_total_failure_is_retryable_and_rate_limit_stops_immediately(
    tmp_path, monkeypatch
):
    document, _ = original_fixture(tmp_path)
    calls = []

    class Opener:
        def open(self, request, timeout):
            calls.append(request.full_url)
            raise OSError("network unavailable")

    monkeypatch.setattr("urllib.request.build_opener", lambda *args: Opener())
    destination = tmp_path / "snapshot"
    result = recover_fca(document, destination)
    assert result["status"] == "unavailable"
    assert not (destination / "manifest.json").exists()
    assert (destination / "attempts/0001/result.json").exists()
    recover_fca(document, destination)
    assert len(calls) == 6 and (destination / "attempts/0002/result.json").exists()

    class Limited:
        def open(self, request, timeout):
            calls.append(request.full_url)
            raise urllib.error.HTTPError(
                request.full_url, 429, "rate limited", {}, None
            )

    monkeypatch.setattr("urllib.request.build_opener", lambda *args: Limited())
    with pytest.raises(urllib.error.HTTPError):
        recover_fca(document, destination)
    assert len(calls) == 7
    assert not (destination / "manifest.json").exists()
