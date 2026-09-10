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
ZERO = {"ON": 0, "PN": 0}


def test_explicit_stock_units_are_independent_of_currency_and_table_scale():
    result = propose_paid_in_note(TEXT, REFERENCE, PAID, ZERO)
    assert result["paid_in_shares"] == {"ON": 7442454142, "PN": 5602042788}
    assert "treasury_shares" not in result
    assert (
        propose_paid_in_note(TEXT, REFERENCE, result["paid_in_shares"], ZERO) == result
    )


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
    assert propose_paid_in_note(text, REFERENCE, PAID, ZERO) is None


def test_positive_treasury_needs_its_own_quantity_evidence():
    assert propose_paid_in_note(TEXT, REFERENCE, PAID, {"ON": 1, "PN": 0}) is None


def test_only_explicitly_unissued_class_can_omit_its_note_quantity():
    text = (
        "Em 31/12/2017 o capital social está representado por 1.000 ações ordinárias."
    )
    result = propose_paid_in_note(
        text, REFERENCE, {"ON": 1000000, "PN": 0}, {"ON": 0, "PN": None}
    )
    assert result["paid_in_shares"] == {"ON": 1000, "PN": 0}
    assert propose_paid_in_note(text, REFERENCE, PAID, ZERO) is None
