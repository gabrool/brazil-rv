"""Read the original CVM relational XML package when its HTML viewer is absent."""

from __future__ import annotations

import io
import zipfile
from datetime import date
from pathlib import Path
from xml.etree import ElementTree as ET


def original_accounts(document: dict, source: Path) -> dict:
    """Parse one exact original .itr/.dfp, including its own-period capital count.

    Nested XML has local document IDs; the outer CVM envelope supplies the public
    ID. Flow statements use accumulated periods (NumeroTrimestre=0), avoiding
    unused quarter cells that the original package can serialize as zero.
    """
    from .round5_cvm import ACCOUNTS, assign_account, digits

    with zipfile.ZipFile(source) as outer:
        envelope_name = next(
            name
            for name in outer.namelist()
            if name.startswith("FormularioDemonstracaoFinanceira")
            and name.endswith(".xml")
        )
        envelope = ET.fromstring(outer.read(envelope_name))
        if envelope.tag != "Documento":
            raise ValueError("Original financial ZIP lacks the public CVM envelope")
        expected = {
            "NumeroSequencialDocumento": str(document["id"]),
            "NumeroVersaoDocumento": str(document["version"]),
            "DataReferenciaDocumento": str(document["reference"]),
            "CompanhiaAberta/CodigoCvm": document["cvm_code"],
            "CompanhiaAberta/NumeroCnpjCompanhiaAberta": document["cnpj"],
        }
        for key, value in expected.items():
            actual = envelope.findtext(key, "")
            actual = actual[:10] if key == "DataReferenciaDocumento" else digits(actual)
            if actual != value:
                raise ValueError(f"Original financial ZIP identity differs at {key}")
        if envelope.findtext("CodigoMoeda") != "1":
            raise ValueError("Original financial ZIP currency is not BRL")
        scale = {"1": 1, "2": 1000}.get(envelope.findtext("CodigoEscalaMoeda"))
        if scale is None:
            raise ValueError("Original financial ZIP currency scale is unavailable")
        nested_name = next(
            name for name in outer.namelist() if name.lower().endswith((".itr", ".dfp"))
        )
        nested_bytes = outer.read(nested_name)
    parsed = {**document, "accounts": {}}
    with zipfile.ZipFile(io.BytesIO(nested_bytes)) as inner:
        periods = {
            node.findtext("NumeroIdentificacaoPeriodo"): {
                "start": date.fromisoformat(node.findtext("DataInicioPeriodo")[:10]),
                "end": date.fromisoformat(node.findtext("DataFimPeriodo")[:10]),
                "quarter": int(node.findtext("NumeroTrimestre")),
            }
            for node in ET.fromstring(inner.read("PeriodoDemonstracaoFinanceira.xml"))
        }
        for node in ET.fromstring(
            inner.read("InformacaoFinanceiraDemonstracaoFinanceira.xml")
        ):
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
            description = node.findtext(
                "DescricoesContaInformacaoFinanceiraDemonstracaoFinanceira/"
                "DescricaoContaInformacaoFinanceiraDemonstracaoFinanceira/DescricaoConta",
                "",
            )
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
                    parsed["shares"] = {
                        code: quantity_scale
                        * (
                            float(
                                node.findtext(
                                    f"QuantidadeAcao{label}CapitalIntegralizado"
                                )
                            )
                            - float(node.findtext(f"QuantidadeAcao{label}Tesouraria"))
                        )
                        for code, label in (("ON", "Ordinaria"), ("PN", "Preferencial"))
                    }
    if not parsed["accounts"]:
        raise ValueError("Original financial ZIP lacks requested-period accounts")
    parsed["recovered_original"] = True
    return parsed
