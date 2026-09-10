from datetime import date
import io
import json
from urllib.parse import parse_qs, urlencode

import pytest

from brazil_rv.v2 import round5_cvm_capital as capital


DOCUMENT = {
    "id": "134555",
    "cnpj": "33000167000101",
    "cvm_code": "009512",
    "reference": date(2023, 12, 31),
    "version": 1,
    "kind": "DFP",
    "receipt": date(2024, 3, 8),
}


def table(unit="Unidade", treasury="104.136.909"):
    rows = [
        [f"Número de Ações ({unit})", "31/12/2023"],
        ["Do Capital Integralizado"],
        ["Ordinárias", "7.442.454.142"],
        ["Preferenciais", "5.602.042.788"],
        ["Total", "13.044.496.930"],
        ["Em Tesouraria"],
        ["Ordinárias", "222.760"],
        ["Preferenciais", treasury],
        ["Total", "104.359.669"],
    ]
    return (
        "<table>"
        + "".join(
            "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>"
            for row in rows
        )
        + "</table>"
    ).encode("utf-8")


def viewer(*, group=False, version=1, captcha="N", code="009512", handler=True):
    query = {
        "NumeroSequencialDocumento": "134555",
        "DataReferencia": "2023-12-31",
        "Versao": version,
        "CodTipoDocumento": "4",
        "Hash": "public-bound-hash",
        "NumeroSequencialRegistroCvm": "1763",
        "CodigoTipoInstituicao": "1",
        "Empresa": "A current display name is not a dated identity",
    }
    child = (
        (
            "frmDadosComposicaoCapitalITR.aspx"
            if group
            else "frmDemonstracaoFinanceiraITR.aspx"
        )
        + "?"
        + urlencode(query)
    )
    fields = {
        "hdnHabilitaCaptcha": captcha,
        "hdnNumeroSequencialDocumento": "134555",
        "hdnCodigoCvm": code,
        "hdnCodigoTipoDocumento": "4",
        "hdnHash": "public-bound-hash",
        "__VIEWSTATE": "original-state",
        "__EVENTVALIDATION": "original-validation",
    }
    body = '<form action="./frmGerenciaPaginaFRE.aspx?NumeroSequencialDocumento=134555&amp;CodigoTipoInstituicao=1">'
    body += "".join(
        f'<input name="{key}" value="{value}">' for key, value in fields.items()
    )
    onchange = "onchange=\"__doPostBack('cmbGrupo','')\"" if handler else ""
    body += f'<select name="cmbGrupo" {onchange}><option value="201">Dados da Empresa</option><option value="345" selected>DFs Consolidadas</option></select>'
    body += '<select name="cmbQuadro"><option value="original-returned-account" selected>DRE</option></select></form>'
    return (body + f"<script>window.frames[0].location='{child}';</script>").encode()


def source_session(monkeypatch, *, initial=None, selected=None, payload=None):
    responses = iter(
        [initial or viewer(), selected or viewer(group=True), payload or table()]
    )
    gets, posts = [], []

    class Opener:
        def open(self, request, timeout):
            assert request.headers["User-agent"] == "Mozilla/5.0"
            if request.get_method() == "POST":
                posts.append(request)
            else:
                gets.append(request.full_url)
            return io.BytesIO(next(responses))

    monkeypatch.setattr(capital.urllib.request, "build_opener", lambda *args: Opener())
    return gets, posts


def test_own_share_quantity_scale_and_treasury_are_independent_of_currency():
    result = capital.parse_capital(table(), DOCUMENT)
    assert result["shares"] == {"ON": 7_442_231_382, "PN": 5_497_905_879}
    thousand = capital.parse_capital(table("Mil"), DOCUMENT)
    assert thousand["shares"] == {
        key: value * 1000 for key, value in result["shares"].items()
    }
    assert thousand["quantity_scale"] == 1000


@pytest.mark.parametrize("treasury", ["", "-", "NaN"])
def test_missing_treasury_is_not_zero(treasury):
    with pytest.raises(ValueError, match="missing|unreported"):
        capital.parse_capital(table(treasury=treasury), DOCUMENT)


def test_negative_treasury_is_not_subtracted_or_made_absolute():
    with pytest.raises(ValueError, match="negative"):
        capital.parse_capital(table(treasury="-181.400"), DOCUMENT)


def test_unissued_class_can_have_zero_outstanding_without_printed_treasury():
    payload = table(treasury="").replace(b"5.602.042.788", b"0")
    result = capital.parse_capital(payload, DOCUMENT)
    assert result["shares"]["PN"] == 0
    assert result["treasury"]["PN"] is None


def disposition_fixture(tmp_path, disposition="reconciled"):
    evidence = tmp_path / "original_note.txt"
    evidence.write_text(
        "Own-period ordinary treasury holdings: 5,207 shares.", encoding="utf8"
    )
    record = {
        "document": capital._identity(DOCUMENT),
        "disposition": disposition,
        "reason": "Own reference/version note establishes positive treasury holdings.",
        "evidence": [{"path": str(evidence), "sha256": capital.sha256(evidence)}],
        "paid_in_shares": {"ON": 90_954_000, "PN": 0},
        "treasury_shares": {"ON": 5207, "PN": 0},
    }
    path = tmp_path / "dispositions.json"
    path.write_text(json.dumps({"documents": [record]}), encoding="utf8")
    return path, evidence


@pytest.mark.parametrize("disposition", ("reconciled", "audited_unavailable"))
def test_source_disposition_preserves_exact_quantity_or_missingness(
    tmp_path, disposition
):
    path, _ = disposition_fixture(tmp_path, disposition)
    result = capital.load_capital_dispositions(path, [DOCUMENT])[DOCUMENT["id"]]
    assert result["shares"] == (
        {"ON": 90_948_793, "PN": 0} if disposition == "reconciled" else None
    )


def test_note_for_another_reference_cannot_repair_this_filing(tmp_path):
    path, _ = disposition_fixture(tmp_path)
    with pytest.raises(ValueError, match="filing identity"):
        capital.load_capital_dispositions(
            path, [{**DOCUMENT, "reference": date(2022, 12, 31)}]
        )


def test_changed_note_bytes_cannot_reuse_a_capital_disposition(tmp_path):
    path, source = disposition_fixture(tmp_path)
    source.write_text("Changed treasury holdings", encoding="utf8")
    with pytest.raises(ValueError, match="hash differs"):
        capital.load_capital_dispositions(path, [DOCUMENT])


@pytest.mark.parametrize("disposition", ("reconciled", "audited_unavailable"))
def test_builder_applies_own_note_audit_even_to_parseable_capital(
    tmp_path, monkeypatch, disposition
):
    import polars as pl

    from brazil_rv.v2 import round5_cvm as cvm

    source_session(monkeypatch)
    capital_root = tmp_path / "capital" / DOCUMENT["id"]
    original = capital.capital_page(DOCUMENT, capital_root)
    path, evidence = disposition_fixture(tmp_path, disposition)
    path.rename(tmp_path / "capital_source_dispositions.json")
    cvm.write_json(
        tmp_path / "calendar_2025_announced.json",
        {
            "source": {"path": str(evidence), "sha256": cvm.sha256(evidence)},
            "available_date": "2024-01-02",
            "base_through": "2024-12-30",
            "through": "2025-12-31",
            "sessions": [],
        },
    )
    day = date(2024, 1, 2)
    identity = pl.DataFrame(
        {
            "date": [day],
            "isin": ["ON"],
            "cnpj": [DOCUMENT["cnpj"]],
            "cvm_code": [DOCUMENT["cvm_code"]],
        }
    )
    accounts = {"con": {"assets": {"value": 123000}}}
    document = {**DOCUMENT, "accounts": accounts}
    monkeypatch.setattr(
        cvm,
        "store_axes_and_identity_observations",
        lambda *_: ([day], ["ON"], pl.DataFrame()),
    )
    monkeypatch.setattr(cvm, "rad_rows", lambda *_: [])
    monkeypatch.setattr(cvm, "fca_documents", lambda *_: [])
    monkeypatch.setattr(cvm, "build_identity", lambda *_: identity)
    monkeypatch.setattr(cvm, "public_float_observations", lambda *_: pl.DataFrame())
    monkeypatch.setattr(cvm, "filing_headers", lambda *_: [])
    monkeypatch.setattr(
        cvm, "event_features", lambda *_: identity.select("date", "cvm_code")
    )
    monkeypatch.setattr(cvm, "load_accounts", lambda *_: [document])
    monkeypatch.setattr(cvm, "capital_change_observations", lambda *_: [])
    monkeypatch.setattr(cvm, "valuation_market", lambda *_: None)

    class ReachedFeatureConsumer(Exception):
        pass

    def inspect_documents(documents, *_):
        assert documents[0]["shares"] == (
            {"ON": 90_948_793, "PN": 0} if disposition == "reconciled" else None
        )
        assert documents[0]["accounts"] == accounts
        assert documents[0]["reference"] == DOCUMENT["reference"]
        raise ReachedFeatureConsumer

    monkeypatch.setattr(cvm, "fundamental_features", inspect_documents)
    with pytest.raises(ReachedFeatureConsumer):
        cvm.build(tmp_path, tmp_path / "unused_store", tmp_path / "output")
    assert capital.load_capital(DOCUMENT, capital_root) == original["capital"]
    sources = json.loads(
        (tmp_path / "output/capital_source_manifests.json").read_text()
    )
    assert len(sources) == 2
    assert sources[-1]["disposition"] == disposition


def test_reference_and_unknown_scale_cannot_borrow_another_period():
    with pytest.raises(ValueError, match="reference date"):
        capital.parse_capital(table(), {**DOCUMENT, "reference": date(2023, 9, 30)})
    with pytest.raises(ValueError, match="quantity scale"):
        capital.parse_capital(table("Reais Mil"), DOCUMENT)


def test_standard_postback_preserves_returned_fields_and_load_is_network_free(
    tmp_path, monkeypatch
):
    gets, posts = source_session(monkeypatch)
    manifest = capital.capital_page(DOCUMENT, tmp_path)
    assert len(gets) == 2 and len(posts) == 1
    fields = parse_qs(posts[0].data.decode(), keep_blank_values=True)
    assert fields["__EVENTTARGET"] == ["cmbGrupo"]
    assert fields["__VIEWSTATE"] == ["original-state"]
    assert fields["__EVENTVALIDATION"] == ["original-validation"]
    assert fields["cmbQuadro"] == ["original-returned-account"]
    assert fields["cmbGrupo"] == ["201"]
    assert (
        capital.load_capital(DOCUMENT, tmp_path)["shares"]
        == manifest["capital"]["shares"]
    )
    assert capital.capital_page(DOCUMENT, tmp_path) == manifest
    assert len(gets) == 2 and len(posts) == 1


@pytest.mark.parametrize("role", ["viewer", "post_form", "group", "capital"])
def test_every_archived_source_is_verified_before_any_share_attachment(
    tmp_path, monkeypatch, role
):
    source_session(monkeypatch)
    manifest = capital.capital_page(DOCUMENT, tmp_path)
    item = next(source for source in manifest["sources"] if source["role"] == role)
    path = tmp_path / item["file"]
    path.write_bytes(path.read_bytes() + b"corruption")
    with pytest.raises(ValueError, match="source hash"):
        capital.load_capital(DOCUMENT, tmp_path)


@pytest.mark.parametrize(
    "change", [{"version": 2}, {"code": "000000"}, {"captcha": "S"}, {"handler": False}]
)
def test_wrong_filing_or_captcha_or_missing_public_handler_stops_before_post(
    tmp_path, monkeypatch, change
):
    gets, posts = source_session(monkeypatch, initial=viewer(**change))
    with pytest.raises(ValueError):
        capital.capital_page(DOCUMENT, tmp_path)
    assert len(gets) == 1 and not posts
    assert not (tmp_path / "manifest.json").exists()


def test_newer_postback_version_cannot_be_relabelled_as_original(tmp_path, monkeypatch):
    gets, _ = source_session(monkeypatch, selected=viewer(group=True, version=2))
    with pytest.raises(ValueError, match="Versao"):
        capital.capital_page(DOCUMENT, tmp_path)
    assert len(gets) == 1


def test_resume_requires_same_document_and_reparsed_capital(tmp_path, monkeypatch):
    source_session(monkeypatch)
    capital.capital_page(DOCUMENT, tmp_path)
    with pytest.raises(ValueError, match="filing identity"):
        capital.load_capital({**DOCUMENT, "version": 2}, tmp_path)
    path = tmp_path / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["capital"]["shares"]["ON"] += 1
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="values differ"):
        capital.load_capital(DOCUMENT, tmp_path)


def test_failed_attempt_bytes_are_retained_on_successful_retry(tmp_path, monkeypatch):
    source_session(monkeypatch, initial=viewer(handler=False))
    with pytest.raises(ValueError):
        capital.capital_page(DOCUMENT, tmp_path)
    original = (tmp_path / "attempts/0001/viewer.html").read_bytes()
    source_session(monkeypatch)
    manifest = capital.capital_page(DOCUMENT, tmp_path)
    assert all(
        item["file"].startswith("attempts/0002/") for item in manifest["sources"]
    )
    assert (tmp_path / "attempts/0001/viewer.html").read_bytes() == original


def test_future_receipt_is_rejected_before_network(tmp_path, monkeypatch):
    gets, posts = source_session(monkeypatch)
    with pytest.raises(ValueError, match="development admission"):
        capital.capital_page({**DOCUMENT, "receipt": date(2025, 1, 2)}, tmp_path)
    assert not gets and not posts
