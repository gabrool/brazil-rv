"""Read the original CVM relational XML package when its HTML viewer is absent."""

from __future__ import annotations

import io
import zipfile
from datetime import date, datetime
from pathlib import Path
from xml.etree import ElementTree as ET


def flat_dfp_accounts(document: dict, payload: bytes, envelope: ET.Element) -> dict:
    """Read the alternative official annual XML layout, never prior-year cells."""
    from .round5_cvm import ACCOUNTS, assign_account, digits, net_share_counts

    source = ET.fromstring(payload)
    if source.tag != "XmlDemonstracoesFinanceiras":
        raise ValueError("Original financial package has an unknown XML layout")
    expected = {
        "DadosEmpresa/CodigoCvm": document["cvm_code"],
        "DadosEmpresa/CnpjEmpresa": document["cnpj"],
        "Documento/VersaoDocumento": str(document["version"]),
    }
    for key, value in expected.items():
        if digits(source.findtext(key, "")) != value:
            raise ValueError(f"Flat original financial XML identity differs at {key}")
    annual = source.find("DadosDFP")
    if annual is None:
        raise ValueError("Flat original XML lacks supported annual DFP periods")
    end = datetime.strptime(
        annual.findtext("DtFimUltimoExercicioSocial"), "%d/%m/%Y"
    ).date()
    start = datetime.strptime(
        annual.findtext("DtInicioUltimoExercicioSocial"), "%d/%m/%Y"
    ).date()
    reference = datetime.strptime(annual.findtext("DataReferencia"), "%d/%m/%Y").date()
    if reference != document["reference"] or end != reference:
        raise ValueError("Flat original financial XML reference period differs")
    for flat_name, envelope_name in (
        ("Moeda", "CodigoMoeda"),
        ("EscalaMoeda", "CodigoEscalaMoeda"),
        ("EscalaQtdAcoes", "CodigoEscalaQuantidade"),
    ):
        if annual.findtext(flat_name) != envelope.findtext(envelope_name):
            raise ValueError(
                f"Flat original financial XML scale differs at {flat_name}"
            )
    currency_scale = {"1": 1, "2": 1000}[annual.findtext("EscalaMoeda")]
    parsed = {**document, "accounts": {}}
    form = annual.find("Formulario")
    for field, basis in (("DfIndividuais", "ind"), ("DfConsolidadas", "con")):
        nodes = form.findall(f"{field}/*/Conta")
        parent_descriptions = {
            node.findtext("CodigoConta"): node.findtext("DescricaoConta", "")
            for node in nodes
        }
        for node in nodes:
            code = node.findtext("CodigoConta")
            value = node.findtext("UltimoExercicio", "").strip()
            if code not in ACCOUNTS or not value:
                continue
            assign_account(
                parsed,
                basis,
                code,
                float(value.replace(".", "").replace(",", ".")) * currency_scale,
                start if code.startswith(("3", "6")) else None,
                end,
                node.findtext("DescricaoConta", ""),
                parent_descriptions.get(code.rsplit(".", 1)[0]),
            )
    quantity_scale = {"1": 1, "2": 1000}.get(annual.findtext("EscalaQtdAcoes"))
    capital = form.find("DadosEmpresa/ComposicaoCapital")
    if quantity_scale is not None and capital is not None:
        paid_in = {
            code: capital.findtext(f"CaptalIntegralizado/{label}")
            for code, label in (("ON", "Ordinarias"), ("PN", "Preferenciais"))
        }
        treasury = {
            code: capital.findtext(f"Tesouraria/{label}")
            for code, label in (("ON", "Ordinarias"), ("PN", "Preferenciais"))
        }
        shares = net_share_counts(paid_in, treasury)
        parsed["shares"] = (
            {code: count * quantity_scale for code, count in shares.items()}
            if shares is not None
            else None
        )
        if shares is None:
            parsed["capital_issue"] = {
                "reason": "negative_inconsistent_or_unreported_class_quantities",
                "paid_in": paid_in,
                "treasury": treasury,
                "quantity_scale": quantity_scale,
            }
    if not parsed["accounts"]:
        raise ValueError("Flat original financial XML lacks requested-period accounts")
    parsed["recovered_original"] = True
    return parsed


def validate_original_identity(
    document: dict, archive: zipfile.ZipFile, envelope: ET.Element
) -> str:
    """Bind the exact public filing; recover a blank CNPJ from its own payload."""
    from .round5_cvm import digits

    if envelope.tag != "Documento":
        raise ValueError("Original financial ZIP lacks the public CVM envelope")
    expected = {
        "NumeroSequencialDocumento": str(document["id"]),
        "NumeroVersaoDocumento": str(document["version"]),
        "DataReferenciaDocumento": str(document["reference"]),
        "CompanhiaAberta/CodigoCvm": document["cvm_code"],
    }
    for key, value in expected.items():
        actual = envelope.findtext(key, "")
        actual = actual[:10] if key == "DataReferenciaDocumento" else digits(actual)
        if actual != value:
            raise ValueError(f"Original financial ZIP identity differs at {key}")
    field = "CompanhiaAberta/NumeroCnpjCompanhiaAberta"
    actual = digits(envelope.findtext(field, ""))
    if actual == document["cnpj"]:
        return "public_envelope"
    if actual.strip("0"):
        raise ValueError(f"Original financial ZIP identity differs at {field}")
    # Some historical public envelopes lost the CNPJ while the exact submitted
    # Documento.xml retains it. A different nonzero CNPJ is never overwritten.
    nested = next(
        (n for n in archive.namelist() if n.lower().endswith((".itr", ".dfp"))),
        None,
    )
    if nested is None:
        raise ValueError("Missing public CNPJ lacks an original identity payload")
    with zipfile.ZipFile(io.BytesIO(archive.read(nested))) as inner:
        original = ET.fromstring(inner.read("Documento.xml"))
    if (
        original.tag != "Documento"
        or digits(original.findtext(field, "")) != document["cnpj"]
        or digits(original.findtext("CompanhiaAberta/CodigoCvm", ""))
        != document["cvm_code"]
        or original.findtext("DataReferenciaDocumento", "")[:10]
        != str(document["reference"])
    ):
        raise ValueError("Original inner CNPJ, CVM or reference differs")
    return "original_inner_document"


def original_accounts(document: dict, source: Path) -> dict:
    """Parse one exact original .itr/.dfp, including its own-period capital count.

    Nested XML has local document IDs; the outer CVM envelope supplies the public
    ID. Flow statements use accumulated periods (NumeroTrimestre=0), avoiding
    unused quarter cells that the original package can serialize as zero.
    """
    from .round5_cvm import ACCOUNTS, assign_account, net_share_counts

    with zipfile.ZipFile(source) as outer:
        envelope_name = next(
            name
            for name in outer.namelist()
            if name.startswith("FormularioDemonstracaoFinanceira")
            and name.endswith(".xml")
        )
        envelope = ET.fromstring(outer.read(envelope_name))
        identity_source = validate_original_identity(document, outer, envelope)
        if envelope.findtext("CodigoMoeda") != "1":
            raise ValueError("Original financial ZIP currency is not BRL")
        scale = {"1": 1, "2": 1000}.get(envelope.findtext("CodigoEscalaMoeda"))
        if scale is None:
            raise ValueError("Original financial ZIP currency scale is unavailable")
        nested_name = next(
            (
                name
                for name in outer.namelist()
                if name.lower().endswith((".itr", ".dfp"))
            ),
            None,
        )
        if nested_name is None:
            flat_names = [
                name
                for name in outer.namelist()
                if name.lower().endswith(".xml") and not name.startswith("Formulario")
            ]
            if len(flat_names) != 1:
                raise ValueError(
                    "Original financial package lacks one identifiable payload"
                )
            return flat_dfp_accounts(document, outer.read(flat_names[0]), envelope)
        nested_bytes = outer.read(nested_name)
    parsed = {**document, "accounts": {}, "cnpj_identity_source": identity_source}
    with zipfile.ZipFile(io.BytesIO(nested_bytes)) as inner:
        periods = {
            node.findtext("NumeroIdentificacaoPeriodo"): {
                "start": date.fromisoformat(node.findtext("DataInicioPeriodo")[:10]),
                "end": date.fromisoformat(node.findtext("DataFimPeriodo")[:10]),
                "quarter": int(node.findtext("NumeroTrimestre")),
            }
            for node in ET.fromstring(inner.read("PeriodoDemonstracaoFinanceira.xml"))
        }
        nodes = ET.fromstring(
            inner.read("InformacaoFinanceiraDemonstracaoFinanceira.xml")
        )
        description_path = (
            "DescricoesContaInformacaoFinanceiraDemonstracaoFinanceira/"
            "DescricaoContaInformacaoFinanceiraDemonstracaoFinanceira/DescricaoConta"
        )
        parent_descriptions = {
            (
                node.findtext(
                    "PlanoConta/VersaoPlanoConta/CodigoTipoInformacaoFinanceira"
                ),
                node.findtext("PlanoConta/NumeroConta"),
            ): node.findtext(description_path, "")
            for node in nodes
        }
        for node in nodes:
            code = node.findtext("PlanoConta/NumeroConta")
            if code not in ACCOUNTS:
                continue
            basis = {"1": "ind", "2": "con"}.get(
                node.findtext(
                    "PlanoConta/VersaoPlanoConta/CodigoTipoInformacaoFinanceira"
                )
            )
            if basis is None:
                continue
            description = node.findtext(description_path, "")
            for column in node.find(
                "ColunasInformacaoFinanceiraDemonstracaoFinanceira"
            ):
                period = periods[
                    column.findtext(
                        "PeriodoDemonstracaoFinanceira/NumeroIdentificacaoPeriodo"
                    )
                ]
                if period["end"] != document["reference"]:
                    continue
                flow = code.startswith(("3", "6"))
                if flow and period["quarter"] != 0:
                    continue
                assign_account(
                    parsed,
                    basis,
                    code,
                    float(column.findtext("ValorConta")) * scale,
                    period["start"] if flow else None,
                    period["end"],
                    description,
                    parent_descriptions.get(
                        (
                            node.findtext(
                                "PlanoConta/VersaoPlanoConta/CodigoTipoInformacaoFinanceira"
                            ),
                            code.rsplit(".", 1)[0],
                        )
                    ),
                )
        # Quantity scale is independently encoded; never assume currency scale.
        quantity_scale = {"1": 1, "2": 1000}.get(
            envelope.findtext("CodigoEscalaQuantidade")
        )
        if quantity_scale is not None:
            for node in ET.fromstring(
                inner.read("ComposicaoCapitalSocialDemonstracaoFinanceiraNegocios.xml")
            ):
                period = periods[
                    node.findtext(
                        "PeriodoDemonstracaoFinanceira/NumeroIdentificacaoPeriodo"
                    )
                ]
                if period["end"] == document["reference"]:
                    paid_in = {
                        code: node.findtext(
                            f"QuantidadeAcao{label}CapitalIntegralizado"
                        )
                        for code, label in (("ON", "Ordinaria"), ("PN", "Preferencial"))
                    }
                    treasury = {
                        code: node.findtext(f"QuantidadeAcao{label}Tesouraria")
                        for code, label in (("ON", "Ordinaria"), ("PN", "Preferencial"))
                    }
                    shares = net_share_counts(paid_in, treasury)
                    parsed["shares"] = (
                        {code: count * quantity_scale for code, count in shares.items()}
                        if shares is not None
                        else None
                    )
                    if shares is None:
                        parsed["capital_issue"] = {
                            "reason": "negative_inconsistent_or_unreported_class_quantities",
                            "paid_in": paid_in,
                            "treasury": treasury,
                            "quantity_scale": quantity_scale,
                        }
    if not parsed["accounts"]:
        raise ValueError("Original financial ZIP lacks requested-period accounts")
    parsed["recovered_original"] = True
    return parsed
