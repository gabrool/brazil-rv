import io
import zipfile
from datetime import date

import pytest

from brazil_rv.v2.round5_cvm_xml import original_accounts


def original_fixture(tmp_path, quantity_scale="1"):
    document = {
        "id": "37949",
        "version": 1,
        "reference": date(2014, 3, 31),
        "cnpj": "11721921000160",
        "cvm_code": "022217",
    }
    envelope = """<Documento>
    <NumeroSequencialDocumento>37949</NumeroSequencialDocumento>
    <NumeroVersaoDocumento>1</NumeroVersaoDocumento>
    <DataReferenciaDocumento>2014-03-31T00:00:00</DataReferenciaDocumento>
    <CompanhiaAberta><CodigoCvm>02221-7</CodigoCvm>
    <NumeroCnpjCompanhiaAberta>11721921000160</NumeroCnpjCompanhiaAberta></CompanhiaAberta>
    <CodigoMoeda>1</CodigoMoeda><CodigoEscalaMoeda>2</CodigoEscalaMoeda>
    <CodigoEscalaQuantidade>1</CodigoEscalaQuantidade></Documento>"""
    envelope = envelope.replace(
        "<CodigoEscalaQuantidade>1</CodigoEscalaQuantidade>",
        f"<CodigoEscalaQuantidade>{quantity_scale}</CodigoEscalaQuantidade>",
    )
    periods = (
        "<Array>"
        + "".join(
            f"<PeriodoDemonstracaoFinanceira><NumeroIdentificacaoPeriodo>{key}</NumeroIdentificacaoPeriodo>"
            f"<DataInicioPeriodo>2014-01-01</DataInicioPeriodo><DataFimPeriodo>2014-03-31</DataFimPeriodo>"
            f"<NumeroTrimestre>{quarter}</NumeroTrimestre></PeriodoDemonstracaoFinanceira>"
            for key, quarter in ((2, 1), (4, 0))
        )
        + "</Array>"
    )
    accounts = "<Array>"
    for code, columns in (("3.01", ((2, 0), (4, 54520))), ("1", ((2, 836784),))):
        accounts += f"""<InformacaoFinanceiraDemonstracaoFinanceira><PlanoConta>
        <NumeroConta>{code}</NumeroConta><VersaoPlanoConta>
        <CodigoTipoInformacaoFinanceira>2</CodigoTipoInformacaoFinanceira>
        </VersaoPlanoConta></PlanoConta><ColunasInformacaoFinanceiraDemonstracaoFinanceira>"""
        for period, value in columns:
            accounts += f"""<ColunasInformacaoFinanceiraDemonstracaoFinanceira>
            <PeriodoDemonstracaoFinanceira><NumeroIdentificacaoPeriodo>{period}</NumeroIdentificacaoPeriodo>
            </PeriodoDemonstracaoFinanceira><ValorConta>{value}</ValorConta>
            </ColunasInformacaoFinanceiraDemonstracaoFinanceira>"""
        accounts += "</ColunasInformacaoFinanceiraDemonstracaoFinanceira></InformacaoFinanceiraDemonstracaoFinanceira>"
    accounts += "</Array>"
    shares = """<Array><ComposicaoCapitalSocialDemonstracaoFinanceira>
    <PeriodoDemonstracaoFinanceira><NumeroIdentificacaoPeriodo>2</NumeroIdentificacaoPeriodo></PeriodoDemonstracaoFinanceira>
    <QuantidadeAcaoOrdinariaCapitalIntegralizado>99128905</QuantidadeAcaoOrdinariaCapitalIntegralizado>
    <QuantidadeAcaoOrdinariaTesouraria>3985658</QuantidadeAcaoOrdinariaTesouraria>
    <QuantidadeAcaoPreferencialCapitalIntegralizado>0</QuantidadeAcaoPreferencialCapitalIntegralizado>
    <QuantidadeAcaoPreferencialTesouraria>0</QuantidadeAcaoPreferencialTesouraria>
    </ComposicaoCapitalSocialDemonstracaoFinanceira></Array>"""
    inner = io.BytesIO()
    with zipfile.ZipFile(inner, "w") as archive:
        archive.writestr("PeriodoDemonstracaoFinanceira.xml", periods)
        archive.writestr("InformacaoFinanceiraDemonstracaoFinanceira.xml", accounts)
        archive.writestr(
            "ComposicaoCapitalSocialDemonstracaoFinanceiraNegocios.xml", shares
        )
    path = tmp_path / "source.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("FormularioDemonstracaoFinanceiraITR.xml", envelope)
        archive.writestr("022217201403310301.itr", inner.getvalue())
    return document, path


@pytest.mark.parametrize("quantity_scale", ("1", "2"))
def test_original_xml_uses_exact_periods_and_ignores_placeholder_quarter_zeros(
    tmp_path,
    quantity_scale,
):
    document, path = original_fixture(tmp_path, quantity_scale)
    parsed = original_accounts(document, path)
    assert "accounts" not in document  # attachment is atomic
    assert parsed["accounts"]["con"]["revenue"]["value"] == 54520000
    assert parsed["accounts"]["con"]["assets"]["value"] == 836784000
    assert parsed["accounts"]["con"]["assets"]["start"] is None
    assert parsed["shares"] == {
        "ON": 95143247 * (1 if quantity_scale == "1" else 1000),
        "PN": 0,
    }


def test_original_xml_rejects_another_public_version(tmp_path):
    document, path = original_fixture(tmp_path)
    document["version"] = 2
    with pytest.raises(ValueError, match="NumeroVersaoDocumento"):
        original_accounts(document, path)
