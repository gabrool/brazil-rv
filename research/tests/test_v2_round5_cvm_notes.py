from datetime import date

import pytest

from brazil_rv.v2.round5_cvm_notes import propose_paid_in_note


REFERENCE = date(2017, 12, 31)
TEXT = (
    "Em milhões de reais, exceto se indicado de outra forma. "
    "Em 31 de dezembro de 2017, o capital subscrito e integralizado no valor de "
    "R$ 205.432 está representado por 7.442.454.142 ações ordinárias e "
    "5.602.042.788 ações preferenciais, todas nominativas, escriturais e sem valor nominal."
)
PAID = {"ON": 7442454142000, "PN": 5602042788000}


def test_explicit_stock_units_are_independent_of_currency_and_table_scale():
    result = propose_paid_in_note(TEXT, REFERENCE, PAID)
    assert result["paid_in_shares"] == {"ON": 7442454142, "PN": 5602042788}
    assert "treasury_shares" not in result
    assert propose_paid_in_note(TEXT, REFERENCE, result["paid_in_shares"]) == result


@pytest.mark.parametrize(
    "text",
    [
        TEXT.replace("31 de dezembro de 2017", "31 de dezembro de 2016"),
        TEXT.replace("Em 31 de dezembro de 2017,", "DFP - 31/12/2017 -"),
        TEXT.replace("capital subscrito", "capital subscrito da controlada"),
        TEXT.replace("7.442.454.142 ações", "7.442.454 mil ações"),
        TEXT.replace("7.442.454.142 ações", "7.442.454,142 ações"),
        TEXT.replace("capital subscrito", "capital social médio ponderado"),
        TEXT + TEXT.replace("7.442.454.142", "7.442.454.143"),
        "Ações ordinárias (lote de mil ações). " + TEXT,
        "Número de ações em milhares. " + TEXT,
    ],
)
def test_other_period_or_ambiguous_note_never_supplies_a_count(text):
    assert propose_paid_in_note(text, REFERENCE, PAID) is None


def test_only_explicitly_unissued_class_can_omit_its_note_quantity():
    text = (
        "Em 31/12/2017 o capital social está representado por 1.000 ações ordinárias."
    )
    result = propose_paid_in_note(text, REFERENCE, {"ON": 1000000, "PN": 0})
    assert result["paid_in_shares"] == {"ON": 1000, "PN": 0}
    assert propose_paid_in_note(text, REFERENCE, PAID) is None


def test_following_prior_year_sentence_cannot_supply_current_share_classes():
    text = (
        "Em 30 de setembro de 2019 o capital social está representado por "
        "499.200.000 ações nominativas, totalmente integralizadas em ações "
        "ordinárias, conforme assembleia de 30 de abril de 2019. "
        "Em 31 de dezembro de 2018 está representado por 31.200.000 ações "
        "ordinárias e 31.200.000 ações preferenciais."
    )
    assert (
        propose_paid_in_note(text, date(2019, 9, 30), {"ON": 499200000, "PN": 0})
        is None
    )


@pytest.mark.parametrize("prior_date", ["30 de junho de 2014", "30/06/2014"])
def test_current_page_header_does_not_redate_an_explicit_earlier_quarter(prior_date):
    text = (
        "Notas explicativas em 30 de setembro de 2014 (em milhares de reais) "
        f"O capital social em {prior_date} é representado por "
        "5.513.608 ações ordinárias e 8.065.423 ações preferenciais."
    )
    assert propose_paid_in_note(text, date(2014, 9, 30), PAID) is None


def test_embedded_reference_and_explicit_classes_keep_original_evidence():
    text = (
        "O capital social integralizado em 31 de dezembro de 2017 era de "
        "R$23,7 milhões, representado por 726.514 ações, sendo 265.160 "
        "ordinárias e 461.354 preferenciais."
    )
    result = propose_paid_in_note(text, REFERENCE, PAID)
    assert result["paid_in_shares"] == {"ON": 265160, "PN": 461354}
    assert result["normalized_evidence"].startswith("capital social integralizado em")
    assert "treasury_shares" not in result


@pytest.mark.parametrize("prefix", ["autorizado", "da controlada", "médio ponderado"])
def test_embedded_reference_cannot_promote_other_capital(prefix):
    text = (
        f"O capital social {prefix} em 31 de dezembro de 2017 está representado "
        "por 1.000 ações ordinárias."
    )
    assert propose_paid_in_note(text, REFERENCE, {"ON": 1000, "PN": 0}) is None


def test_prior_capital_sentence_cannot_borrow_later_reference():
    text = (
        "O capital social em 31 de dezembro de 2016 está representado por "
        "1.000 ações ordinárias. Em 31 de dezembro de 2017 ocorreu uma reunião."
    )
    assert propose_paid_in_note(text, REFERENCE, {"ON": 1000, "PN": 0}) is None


def test_adjacent_treasury_prose_does_not_set_treasury_or_paid_in_counts():
    text = TEXT + " A Companhia mantinha 10 ações ordinárias em tesouraria."
    result = propose_paid_in_note(text, REFERENCE, PAID)
    assert result["paid_in_shares"] == {"ON": 7442454142, "PN": 5602042788}
    assert "treasury_shares" not in result


def test_separate_authorized_ceiling_is_not_another_paid_in_count():
    text = (
        "O capital social em 31 de dezembro de 2017, totalmente subscrito e "
        "integralizado, é representado por 467.934.646 ações ordinárias "
        "nominativas sem valor nominal e o autorizado é de 529.624.961 "
        "ações ordinárias nominativas sem valor nominal."
    )
    result = propose_paid_in_note(text, REFERENCE, {"ON": 467591824000, "PN": 0})
    assert result["paid_in_shares"] == {"ON": 467934646, "PN": 0}
