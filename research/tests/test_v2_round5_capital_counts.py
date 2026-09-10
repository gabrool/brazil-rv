import pytest

from brazil_rv.v2.round5_cvm import net_share_counts


@pytest.mark.parametrize(
    "paid,treasury",
    [
        (100, None),
        (100, ""),
        (100, -5),
        (100, 101),
        (None, 0),
        (-1, 0),
        (100, float("nan")),
    ],
)
def test_unusable_class_quantity_does_not_become_zero_or_inflate_outstanding(
    paid, treasury
):
    assert net_share_counts({"ON": paid, "PN": 0}, {"ON": treasury, "PN": 0}) is None


def test_positive_reported_quantity_and_logically_unissued_class():
    assert net_share_counts({"ON": "100", "PN": "0"}, {"ON": "5", "PN": None}) == {
        "ON": 95,
        "PN": 0,
    }
