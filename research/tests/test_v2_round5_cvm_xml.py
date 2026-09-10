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
        </VersaoPlanoConta></PlanoConta><DescricoesContaInformacaoFinanceiraDemonstracaoFinanceira>
        <DescricaoContaInformacaoFinanceiraDemonstracaoFinanceira><DescricaoConta>{"Ativo Total" if code == "1" else "Receita de Venda de Bens e/ou Serviços"}</DescricaoConta>
        </DescricaoContaInformacaoFinanceiraDemonstracaoFinanceira></DescricoesContaInformacaoFinanceiraDemonstracaoFinanceira>
        <ColunasInformacaoFinanceiraDemonstracaoFinanceira>"""
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


def flat_fixture(tmp_path, quantity_scale="1", inner_version=1):
    document, nested = original_fixture(tmp_path, quantity_scale)
    document["reference"] = date(2014, 12, 31)
    with zipfile.ZipFile(nested) as archive:
        envelope = archive.read("FormularioDemonstracaoFinanceiraITR.xml").decode()
    envelope = envelope.replace("2014-03-31", "2014-12-31")
    payload = f"""<XmlDemonstracoesFinanceiras><DadosEmpresa><CodigoCvm>022217</CodigoCvm>
    <CnpjEmpresa>11721921000160</CnpjEmpresa></DadosEmpresa><Documento>
    <VersaoDocumento>{inner_version}</VersaoDocumento></Documento><DadosDFP>
    <DataReferencia>31/12/2014</DataReferencia><DtInicioUltimoExercicioSocial>01/01/2014</DtInicioUltimoExercicioSocial>
    <DtFimUltimoExercicioSocial>31/12/2014</DtFimUltimoExercicioSocial><Moeda>1</Moeda>
    <EscalaMoeda>2</EscalaMoeda><EscalaQtdAcoes>{quantity_scale}</EscalaQtdAcoes><Formulario>
    <DadosEmpresa><ComposicaoCapital><CaptalIntegralizado><Ordinarias>1000</Ordinarias>
    <Preferenciais>0</Preferenciais></CaptalIntegralizado><Tesouraria><Ordinarias>10</Ordinarias>
    <Preferenciais>0</Preferenciais></Tesouraria></ComposicaoCapital></DadosEmpresa>
    <DfIndividuais><DemonstracaoResultado><Conta><CodigoConta>3.01</CodigoConta>
    <DescricaoConta>Receita</DescricaoConta><UltimoExercicio>1.234,50</UltimoExercicio>
    <PenultimoExercicio>9.999.999</PenultimoExercicio></Conta></DemonstracaoResultado></DfIndividuais>
    <DfConsolidadas><DemonstracaoResultado><Conta><CodigoConta>3.01</CodigoConta>
    <UltimoExercicio/><PenultimoExercicio>99</PenultimoExercicio></Conta></DemonstracaoResultado></DfConsolidadas>
    </Formulario></DadosDFP></XmlDemonstracoesFinanceiras>"""
    path = tmp_path / "flat.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("FormularioDemonstracaoFinanceiraDFP.xml", envelope)
        archive.writestr("022217DFP31-12-2014v1.xml", payload)
    return document, path


@pytest.mark.parametrize("quantity_scale", ("1", "2"))
def test_flat_dfp_uses_own_period_and_independent_quantity_scale(
    tmp_path, quantity_scale
):
    document, path = flat_fixture(tmp_path, quantity_scale)
    parsed = original_accounts(document, path)
    assert parsed["accounts"] == {
        "ind": {
            "revenue": {
                "value": 1234500,
                "start": date(2014, 1, 1),
                "end": date(2014, 12, 31),
                "description": "Receita",
                "source_code": "3.01",
            }
        }
    }
    assert parsed["shares"] == {
        "ON": 990 * (1 if quantity_scale == "1" else 1000),
        "PN": 0,
    }


def test_flat_dfp_inner_version_must_match_public_envelope(tmp_path):
    document, path = flat_fixture(tmp_path, inner_version=2)
    with pytest.raises(ValueError, match="VersaoDocumento"):
        original_accounts(document, path)
