"""Original cadastre classes and sector vocabulary cannot invent issuer history."""

from copy import deepcopy
from datetime import date

import polars as pl

from brazil_rv.v2 import round5_cvm, round5_cvm_fca
from brazil_rv.v2.round5_cvm import build_identity


def test_sourced_rename_preserves_class_and_separate_sector_clock():
    days, document, observations = fixture()
    source = build_identity([document], observations, days, ["ON", "PN"])
    links = pl.DataFrame(
        [
            dict(
                predecessor_index=0,
                successor_index=2,
                effective_index=2,
                known_index=3,
                evidence_sha256="source",
            )
        ]
    )
    later = source.with_columns(
        pl.when(pl.col("date") >= days[4])
        .then(pl.lit("99"))
        .otherwise(pl.col("sector"))
        .alias("sector"),
        pl.when(pl.col("date") >= days[4])
        .then(pl.lit(days[4]))
        .otherwise(pl.col("sector_known_date"))
        .alias("sector_known_date"),
    )
    result = round5_cvm.continue_issuer_identity(
        later, links, days, ["ON", "PN", "NEW"]
    )
    assert result.filter(pl.col("date") < days[3]).equals(
        later.filter(pl.col("date") < days[3]).sort("date", "isin")
    )
    new = result.filter(pl.col("isin") == "NEW")
    assert new["date"].to_list() == days[3:]
    assert new["class"].to_list() == ["ON", "ON"]
    assert new["sector"].to_list() == ["17", "99"]
    assert new["identity_known_date"].to_list() == [days[3]] * 2
    assert not result.filter(
        (pl.col("isin") == "ON") & (pl.col("date") >= days[3])
    ).height
    shortened = round5_cvm.continue_issuer_identity(
        later.filter(pl.col("date") < days[4]), links, days[:4], ["ON", "PN", "NEW"]
    )
    assert shortened.equals(result.filter(pl.col("date") < days[4]))
    bad = pl.concat([later, new.with_columns(pl.lit("PN").alias("class"))])
    import pytest

    with pytest.raises(ValueError, match="Conflicting rename issuer/class"):
        round5_cvm.continue_issuer_identity(bad, links, days, ["ON", "PN", "NEW"])


def fixture():
    days = [date(2024, 1, d) for d in (2, 3, 4, 5, 8)]
    document = {
        "id": "1",
        "cnpj": "12345678000100",
        "cvm_code": "123456",
        "reference": date(2023, 1, 1),
        "version": 1,
        "receipt": days[0],
        "available_index": 1,
        "legal_name": "EXAMPLE S.A.",
        "sector_code": "17",
        "sector_label": "Historical industry",
        "securities": [
            {
                "ticker": "",
                "class": "SHARES",
                "preferred_class": "",
                "unit_composition": "",
                "start": date(2000, 1, 1),
                "end": date.max,
            }
        ],
    }
    observations = pl.DataFrame(
        {
            "trade_date": [days[0]] * 4 + [days[2]],
            "isin": ["ON", "PN", "PNA", "UNIT", "SUCCESSOR"],
            "ticker": ["ABCD3", "ABCD4", "ABCD5", "ABCD11", "NEW3"],
            "issuer_short_name": ["EXAMPLE"] * 5,
            "security_spec_base": ["ON", "PN", "PNA", "UNT", "ON"],
        }
    )
    return days, document, observations


def test_original_generic_shares_use_independent_prior_classes_and_birth_bounds():
    days, document, observations = fixture()
    identity = build_identity(
        [document], observations, days, observations["isin"].to_list()
    )
    assert set(identity["isin"]) == {"ON", "PN", "PNA"}
    assert identity["date"].min() == days[1]
    assert set(identity["identity_method"]) == {
        "original_generic_shares_exact_legal_spelling"
    }
    assert identity.filter(pl.col("isin") == "PNA")["preferred_class"][0] == "A"
    assert identity.filter(pl.col("isin") == "PN")["preferred_class"][0] == ""


def test_source_bound_correction_cannot_assign_another_security():
    days, document, observations = fixture()
    security = document["securities"][0]
    security.update(ticker="ABCD4", source_bound_isin="PN")
    matched = build_identity([document], observations, days, ["ON", "PN"])
    assert set(matched["isin"]) == {"PN"}
    security["source_bound_isin"] = "ON"
    assert build_identity([document], observations, days, ["ON", "PN"]).is_empty()


def test_explicit_generic_ticker_uses_prior_b3_class_without_name_guessing():
    days, document, observations = fixture()
    document["securities"][0]["ticker"] = "ABCD3"
    document["legal_name"] = "DIFFERENT S.A."
    rows = build_identity([document], observations, days, ["ON", "PN", "UNIT"])
    assert set(rows["isin"]) == {"ON"}
    assert set(rows["class"]) == {"ON"}
    assert set(rows["identity_method"]) == {"original_generic_shares_dated_ticker"}
    # An explicit pair must still fail conflicting issuer claims and cannot
    # attach a ticker born later than the disclosure without a listing boundary.
    other = {**document, "id": "2", "cnpj": "87654321000100", "cvm_code": "654321"}
    import pytest

    with pytest.raises(ValueError, match="Ambiguous dated issuer"):
        build_identity([document, other], observations, days, ["ON"])
    document["securities"][0]["ticker"] = "NEW3"
    assert build_identity([document], observations, days, ["SUCCESSOR"]).is_empty()
    document["securities"][0]["ticker"] = "ABCD11"
    assert build_identity([document], observations, days, ["UNIT"]).is_empty()


def test_explicit_original_class_remains_stricter_than_generic():
    days, document, observations = fixture()
    document["securities"][0]["class"] = "ON"
    identity = build_identity(
        [document], observations, days, observations["isin"].to_list()
    )
    assert set(identity["isin"]) == {"ON"}


def test_source_correction_keeps_its_own_clock_and_other_security_metadata():
    days, document, observations = fixture()
    ordinary = document["securities"][0]
    ordinary.update(ticker="ABCD3", source_identity_known_index=3)
    preferred = deepcopy(ordinary)
    preferred.update(ticker="ABCD4", source_identity_known_index=1)
    document["securities"].append(preferred)
    rows = build_identity([document], observations, days, ["ON", "PN"])
    assert rows.filter(pl.col("isin") == "ON")["date"].min() == days[3]
    assert set(rows.filter(pl.col("isin") == "ON")["identity_known_date"]) == {days[3]}
    assert rows.filter(pl.col("isin") == "PN")["date"].min() == days[1]
    assert set(rows["sector_known_date"]) == {days[1]}
    delayed = deepcopy(document)
    delayed["securities"][0]["source_identity_known_index"] = 4
    later = build_identity([delayed], observations, days, ["ON", "PN"])
    assert later.filter(pl.col("date") < days[3]).equals(
        rows.filter(pl.col("date") < days[3])
    )


def test_explicit_ticker_class_must_match_observation_and_preserve_preferred_suffix():
    days, document, observations = fixture()
    security = document["securities"][0]
    security.update(ticker="ABCD5", **{"class": "ON"})
    assert build_identity([document], observations, days, ["PNA"]).is_empty()
    security["class"] = "PN"
    identity = build_identity([document], observations, days, ["PNA"])
    assert set(identity["preferred_class"]) == {"A"}
    security["preferred_class"] = "B"
    assert build_identity([document], observations, days, ["PNA"]).is_empty()


def test_original_end_date_and_future_version_are_causal():
    days, document, observations = fixture()
    document["securities"][0]["end"] = days[1]
    baseline = build_identity([document], observations, days, ["ON", "PN"])
    assert set(baseline["date"]) == {days[1]}
    future = deepcopy(document)
    future.update(
        id="2", version=2, receipt=days[2], available_index=3, sector_code="18"
    )
    future["securities"][0]["end"] = date.max
    changed = build_identity([document, future], observations, days, ["ON", "PN"])
    assert baseline.equals(changed.filter(pl.col("date") < days[3]))
    assert set(changed.filter(pl.col("date") == days[3])["sector"]) == {"18"}


def test_equivalent_preferred_class_source_labels_match_without_erasing_conflicts():
    days, document, observations = fixture()
    security = document["securities"][0]
    security["class"] = "PN"
    for ticker in ("ABCD5", ""):
        security["ticker"] = ticker
        for label in ("A", "PNA", "Classe A", "Preferencial Classe A"):
            security["preferred_class"] = label
            identity = build_identity([document], observations, days, ["PNA"])
            assert set(identity["isin"]) == {"PNA"}, (ticker, label)
            assert set(identity["preferred_class"]) == {"A"}
            assert identity["date"].min() == days[1]
        for label in ("PNB", "Preferencial Classe B", "unresolved description"):
            security["preferred_class"] = label
            assert build_identity([document], observations, days, ["PNA"]).is_empty()


def test_sector_translation_enters_first_known_decision_without_refiling():
    days, document, observations = fixture()
    document["sector_code"] = None
    evidence = {
        "id": "10",
        "receipt": days[2],
        "available_index": 3,
        "version": 1,
        "sector_code": "17",
        "sector_label": document["sector_label"],
    }
    before = build_identity([document], observations, days, ["ON", "PN"])
    repeated = {**evidence, "id": "11", "receipt": days[3], "available_index": 4}
    after = build_identity(
        [document, evidence, repeated], observations, days, ["ON", "PN"]
    )
    assert before["sector"].null_count() == before.height
    assert before.filter(pl.col("date") < days[3]).equals(
        after.filter(pl.col("date") < days[3])
    )
    known = after.filter(pl.col("date") >= days[3])
    assert set(known["sector"]) == {"17"}
    assert set(known["sector_known_date"]) == {days[3]}
    assert set(known["sector_mapping_id"]) == {"10"}
    assert set(known["fca_id"]) == {"1"}  # No new issuer filing was needed.
    assert set(after["identity_known_date"]) == {days[1]}
    assert set(after["sector_label"]) == {document["sector_label"]}


def test_future_sector_code_conflict_cannot_erase_prior_group_or_explicit_code():
    days, document, observations = fixture()
    annual = {**document, "sector_code": None}
    prior = {
        "id": "10",
        "receipt": days[0],
        "available_index": 1,
        "version": 1,
        "sector_code": "17",
        "sector_label": document["sector_label"],
    }
    future = {
        **prior,
        "id": "11",
        "receipt": days[2],
        "available_index": 3,
        "sector_code": "18",
    }
    before = build_identity([annual, prior], observations, days, ["ON", "PN"])
    after = build_identity([annual, prior, future], observations, days, ["ON", "PN"])
    assert before.filter(pl.col("date") < days[3]).equals(
        after.filter(pl.col("date") < days[3])
    )
    unknown = after.filter(pl.col("date") >= days[3])
    for column in ("sector", "sector_known_date", "sector_mapping_id"):
        assert unknown[column].null_count() == unknown.height
    explicit = build_identity([document, future], observations, days, ["ON", "PN"])
    assert set(explicit["sector"]) == {"17"}
    assert set(explicit["sector_known_date"]) == {days[1]}
    assert set(explicit["sector_mapping_id"]) == {"1"}


def test_original_attachment_uses_listing_bounds_and_ignores_segment_restart(
    tmp_path, monkeypatch
):
    days, header, _ = fixture()
    header = {
        k: v
        for k, v in header.items()
        if k
        not in {
            "securities",
            "sector_code",
            "sector_label",
            "legal_name",
            "available_index",
        }
    }
    monkeypatch.setattr(round5_cvm, "filing_headers", lambda *_: [deepcopy(header)])
    monkeypatch.setattr(
        round5_cvm, "annual_paths", lambda *_: [tmp_path / "fca_2023.zip"]
    )
    rows = {
        "fca_cia_aberta_geral_2023.csv": [
            {
                "ID_Documento": "1",
                "Setor_Atividade": "Industry",
                "Nome_Empresarial": "EXAMPLE S.A.",
            }
        ],
        "fca_cia_aberta_valor_mobiliario_2023.csv": [
            {
                "ID_Documento": "1",
                "Mercado": "Bolsa",
                "Valor_Mobiliario": "Ações Ordinárias",
                "Codigo_Negociacao": "",
                "Sigla_Classe_Acao_Preferencial": "",
                "Composicao_BDR_Unit": "",
                "Data_Inicio_Negociacao": "2000-01-01",
                "Data_Fim_Negociacao": "",
                "Data_Inicio_Listagem": "1980-01-01",
                "Data_Fim_Listagem": "",
            }
        ],
    }
    monkeypatch.setattr(round5_cvm, "read_csv_member", lambda _, name: rows[name])
    source = tmp_path / "fca_originals" / "1"
    source.mkdir(parents=True)
    (source / "manifest.json").write_text("{}", encoding="utf8")
    metadata = {
        "legal_name": "EXAMPLE S.A.",
        "sector_code": "017",
        "sector": "Industry",
        "metadata_source": "original_fca_xml",
        "securities": [
            {
                "ticker": "",
                "class": "SHARES",
                "preferred_class": "",
                "unit_composition": "",
                "start": str(days[0]),
                "end": str(days[2]),
                "listing_start": str(days[0]),
                "listing_end": str(days[2]),
                "segment_start": str(days[3]),
                "segment_end": "9999-12-31",
            }
        ],
    }
    monkeypatch.setattr(round5_cvm_fca, "load_fca", lambda *_: deepcopy(metadata))
    documents = round5_cvm.fca_documents(tmp_path, days)
    assert documents[0]["securities"][0]["class"] == "SHARES"
    assert documents[0]["securities"][0]["start"] == days[0]
    assert documents[0]["securities"][0]["end"] == days[2]
    assert documents[0]["sector_code"] == "17"
    assert documents[0]["sector_label"] == "Industry"
    assert documents[0]["available_index"] == 1
    assert documents[0]["original_fca_source"]["manifest_path"] == str(
        source / "manifest.json"
    )
