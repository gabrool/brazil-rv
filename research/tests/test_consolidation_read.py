from __future__ import annotations

from brazil_rv.modeling.consolidation_read import (
    ARCHIVED_STORE_V2_IC,
    COMPARATOR_IDENTITIES,
    arm2_supported,
    deploy_arm1_ten_seed,
    deployment_choice,
    derive_consensus,
    sanity_band_passed,
)


def _path(*members: str, gains: dict[str, float]) -> dict[str, object]:
    return {
        "heldout_members": list(members),
        "steps": [
            {"addition": identity, "marginal_ic": gain}
            for identity, gain in gains.items()
        ],
    }


def _report(lower: float) -> dict[str, object]:
    return {
        "per_date_delta_bootstrap": {
            "10": {"lower_95": [lower], "upper_95": [lower + 0.001]}
        }
    }


def test_consensus_uses_repeat_count_then_gain_then_lexical_order() -> None:
    left, middle, right = COMPARATOR_IDENTITIES
    repeated = "e2a|seed_47|final_ema_0995"
    higher_gain = "opt_full|seed_11|final_ema_0995"
    lower_gain = "prune_r1|seed_11|patience3_raw"
    singleton = "discarded|seed_29|final_ema_0995"
    analysis = {
        "named_read_arm": {
            "label": "e2_plus_archive",
            "paths": {
                "fold_c": _path(
                    left,
                    middle,
                    right,
                    repeated,
                    higher_gain,
                    lower_gain,
                    singleton,
                    gains={repeated: 0.2, higher_gain: 0.3, lower_gain: 0.1},
                ),
                "fold_a": _path(
                    left,
                    middle,
                    right,
                    repeated,
                    higher_gain,
                    lower_gain,
                    gains={repeated: 0.4, higher_gain: 0.5, lower_gain: 0.2},
                ),
                "fold_b": _path(
                    left,
                    middle,
                    right,
                    repeated,
                    gains={repeated: 0.6},
                ),
            },
        }
    }

    consensus = derive_consensus(analysis)
    assert consensus["withdrawn"] is False
    members = consensus["members"]
    assert [row["identity"] for row in members[:3]] == list(COMPARATOR_IDENTITIES)
    assert [row["identity"] for row in members[3:]] == [
        repeated,
        higher_gain,
        lower_gain,
    ]
    assert [row["raw_weight"] for row in members] == [3, 3, 3, 3, 2, 2]
    assert abs(sum(row["normalized_weight"] for row in members) - 1.0) < 1e-12


def test_consensus_withdraws_without_repeated_noncomparator() -> None:
    analysis = {
        "named_read_arm": {
            "label": "e2_plus_archive",
            "paths": {
                fold: _path(
                    *COMPARATOR_IDENTITIES,
                    f"single_{fold}|seed_47|final_ema_0995",
                    gains={},
                )
                for fold in ("fold_c", "fold_a", "fold_b")
            },
        }
    }
    assert derive_consensus(analysis)["withdrawn"] is True


def test_frozen_gate_boundaries_are_strict_or_inclusive_as_registered() -> None:
    assert sanity_band_passed(ARCHIVED_STORE_V2_IC + 0.0015)
    assert not sanity_band_passed(ARCHIVED_STORE_V2_IC + 0.0015001)
    assert deploy_arm1_ten_seed(0.0495, 0.05)
    assert not deploy_arm1_ten_seed(0.049499, 0.05)
    assert arm2_supported(_report(0.000001))
    assert not arm2_supported(_report(0.0))


def test_arm2_support_supersedes_arm1_and_sanity_halts_every_deployment() -> None:
    supported = deployment_choice(
        fresh_three_seed_ic=ARCHIVED_STORE_V2_IC,
        fresh_ten_seed_ic=ARCHIVED_STORE_V2_IC - 0.001,
        arm2_report=_report(0.000001),
    )
    assert supported["deployed_recipe"] == "e2_plus_archive"

    unsupported = deployment_choice(
        fresh_three_seed_ic=ARCHIVED_STORE_V2_IC,
        fresh_ten_seed_ic=ARCHIVED_STORE_V2_IC - 0.001,
        arm2_report=_report(0.0),
    )
    assert unsupported["deployed_recipe"] == "store_v2_3_seed"

    halted = deployment_choice(
        fresh_three_seed_ic=ARCHIVED_STORE_V2_IC + 0.002,
        fresh_ten_seed_ic=ARCHIVED_STORE_V2_IC + 0.002,
        arm2_report=_report(0.000001),
    )
    assert halted["deployed_recipe"] is None
    assert halted["deployment_halted_for_review"] is True
