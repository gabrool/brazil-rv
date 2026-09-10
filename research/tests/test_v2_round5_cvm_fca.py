from datetime import date
import io
import urllib.error
import zipfile

import pytest

from brazil_rv.v2.round5_cvm_fca import (
    _general_html,
    _generic_html_securities,
    original_fca,
    recover_fca,
)


def original_fixture(tmp_path, *, public_id="70793", inner_version=1):
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
