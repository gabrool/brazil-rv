from brazil_rv.modeling.official_read import (
    arm_supported,
    deploy_expansion,
    promotion_decision,
)


def _report(ic: float, lower: float) -> dict[str, object]:
    return {
        "candidate": {"ensemble_ic": ic},
        "per_date_delta_bootstrap": {
            "10": {"lower_95": [lower], "upper_95": [lower + 0.001]}
        },
    }


def test_promotion_requires_positive_block10_lower_bound() -> None:
    assert arm_supported(_report(0.05, 0.00001))
    assert not arm_supported(_report(0.06, 0.0))
    assert not arm_supported(_report(0.07, -0.00001))


def test_promotion_uses_support_then_higher_official_ic() -> None:
    assert promotion_decision(_report(0.05, -0.001), _report(0.04, -0.001))[
        "promoted"
    ] is None
    assert promotion_decision(_report(0.05, 0.001), _report(0.06, -0.001))[
        "promoted"
    ] == "challenger"
    assert promotion_decision(_report(0.05, 0.001), _report(0.06, 0.001))[
        "promoted"
    ] == "store_v2"


def test_expansion_deployment_tolerance_is_inclusive() -> None:
    assert deploy_expansion(0.0495, 0.05)
    assert not deploy_expansion(0.049499, 0.05)
