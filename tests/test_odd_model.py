# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
# ruff:file-ignore[ambiguous-unicode-character-string]
"""Tests for fitting and querying a complete autoSAFE ODD."""

from typing import cast

import numpy as np
import pytest

from autosafe.odd import AutoSafeConfig, AutoSafeODD


def test_calibrated_fit_resolves_paper_parameters() -> None:
    anchors = np.array([[0.0], [1.0], [2.0]])
    config = AutoSafeConfig.from_mapping({
        "γ": 1.2,
        "s": 2.0,
        "λ_rel": 1e-4,
        "ζ": 0.5,
        "normalize": False,
    })

    model = AutoSafeODD.fit(
        anchors,
        config=config,
        feature_names=["coordinate"],
    )

    resolved = model.resolved_config.kernel
    assert resolved.median_nearest_neighbor_distance == pytest.approx(1.0)
    assert resolved.maximum_variance == pytest.approx(4.0)
    assert resolved.distance_decay_rate == pytest.approx(1.2)
    assert resolved.variance_floor == pytest.approx(4e-4)
    assert model.feature_names == ("coordinate",)
    assert model.affinity([0.0]) == pytest.approx(1.0)
    assert model.contains([0.0]) is True
    assert model.contains([20.0]) is False


def test_wrapper_stores_and_enforces_observed_ood_data() -> None:
    anchors = np.array([[-6.0, 0.0], [0.0, 0.0], [6.0, 0.0]])
    observed_ood = np.array([[0.02, 0.0]])
    config = {
        "κ": 1.0,
        "η": 1.0,
        "λ_rel": 1e-4,
        "ζ": 0.5,
        "ξ": 0.1,
        "c": 0.9,
        "ood_batch_jump": True,
        "normalize": False,
    }

    model = AutoSafeODD.fit(
        anchors,
        out_of_distribution_data=observed_ood,
        config=config,
    )

    np.testing.assert_array_equal(model.out_of_distribution_data, observed_ood)
    assert model.ood_consistency_result is not None
    affinity = cast("np.ndarray", model.affinity(observed_ood))
    assert float(np.max(affinity)) <= 0.1 + 1e-12


def test_conformal_membership_uses_held_out_id_data() -> None:
    anchors = np.array([[-2.0, 0.0], [0.0, 0.0], [2.0, 0.0]])
    held_out = np.array([
        [-2.2, 0.0],
        [-1.8, 0.0],
        [-0.2, 0.0],
        [0.2, 0.0],
        [1.8, 0.0],
        [2.2, 0.0],
    ])

    model = AutoSafeODD.fit(
        anchors,
        calibration_data=held_out,
        config={"epsilon": 0.2, "normalize": False},
    )

    membership = model.resolved_config.membership
    assert membership.mode == "conformal"
    assert membership.calibration_size == len(held_out)
    assert 0.0 < membership.affinity_threshold < 1.0
    expected = np.asarray(model.log_survival(held_out)) <= (
        membership.log_survival_threshold
    )
    np.testing.assert_array_equal(model.contains(held_out), expected)
    np.testing.assert_array_equal(model.calibration_data, held_out)


def test_ood_data_and_configuration_must_be_supplied_together() -> None:
    anchors = np.array([[0.0], [1.0]])

    with pytest.raises(ValueError, match="requires out_of_distribution_data"):
        AutoSafeODD.fit(
            anchors,
            config={"zeta": 0.5, "xi": 0.1},
        )
    with pytest.raises(ValueError, match="requires OOD consistency"):
        AutoSafeODD.fit(
            anchors,
            out_of_distribution_data=np.array([[0.5]]),
            config={"zeta": 0.5},
        )


def test_conformal_membership_requires_calibration_data() -> None:
    with pytest.raises(ValueError, match="requires held-out calibration_data"):
        AutoSafeODD.fit(
            [[0.0], [1.0]],
            config={"epsilon": 0.2, "normalize": False},
        )


def test_input_feature_counts_and_names_are_validated() -> None:
    anchors = np.array([[0.0, 1.0], [1.0, 2.0]])

    with pytest.raises(ValueError, match="feature_names"):
        AutoSafeODD.fit(
            anchors,
            config={"zeta": 0.5},
            feature_names=["only_one"],
        )
    with pytest.raises(ValueError, match="expected 2"):
        AutoSafeODD.fit(
            anchors,
            out_of_distribution_data=np.array([[0.5]]),
            config={"zeta": 0.5, "xi": 0.1},
        )


def test_conformal_threshold_helper_uses_finite_sample_rank() -> None:
    from autosafe.odd import conformal_membership_threshold

    affinities = np.array([0.1, 0.2, 0.3, 0.4])
    log_survival = np.log1p(-affinities)

    zeta, log_threshold = conformal_membership_threshold(
        log_survival,
        false_exclusion_rate=0.4,
    )

    assert zeta == pytest.approx(0.2)
    assert log_threshold == pytest.approx(np.log1p(-0.2))


@pytest.mark.parametrize(
    ("data", "match"),
    [
        ([0.0, 1.0], "shape"),
        (np.empty((0, 1)), "at least one row"),
        (np.empty((1, 0)), "at least one feature"),
        ([[np.inf]], "finite"),
    ],
)
def test_fit_rejects_invalid_anchor_matrices(data: object, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        AutoSafeODD.fit(data, config={"zeta": 0.5})


def test_fit_rejects_single_anchor_and_invalid_feature_names() -> None:
    with pytest.raises(ValueError, match="at least two"):
        AutoSafeODD.fit([[0.0]], config={"zeta": 0.5})
    with pytest.raises(ValueError, match="unique"):
        AutoSafeODD.fit(
            [[0.0, 1.0], [1.0, 2.0]],
            config={"zeta": 0.5},
            feature_names=["same", "same"],
        )
    with pytest.raises(ValueError, match="empty"):
        AutoSafeODD.fit(
            [[0.0, 1.0], [1.0, 2.0]],
            config={"zeta": 0.5},
            feature_names=["first", ""],
        )


def test_manual_vector_scales_are_validated() -> None:
    with pytest.raises(ValueError, match="expected 2"):
        AutoSafeODD.fit(
            [[0.0, 1.0], [1.0, 2.0]],
            config={"kappa": [1.0], "eta": 1.0, "zeta": 0.5},
        )
    with pytest.raises(ValueError, match="below maximum_variance"):
        AutoSafeODD.fit(
            [[0.0], [1.0]],
            config={"kappa": 1.0, "eta": 1.0, "lambda": 1.0, "zeta": 0.5},
        )

    model = AutoSafeODD.fit(
        [[0.0, 1.0], [1.0, 2.0]],
        config={
            "kappa": [2.0, 3.0],
            "eta": [1.0, 0.5],
            "lambda_rel": 0.1,
            "zeta": 0.5,
        },
    )
    assert model.resolved_config.kernel.variance_floor == pytest.approx((0.2, 0.3))


def test_conformal_threshold_validation() -> None:
    from autosafe.odd import conformal_membership_threshold

    invalid = [
        ([-1.0], 0.0, "in \\(0, 1\\)"),
        ([[-1.0]], 0.5, "one-dimensional"),
        ([], 0.5, "must not be empty"),
        ([np.nan], 0.5, "invalid values"),
        ([-1.0], 0.1, "too small"),
        ([0.0], 0.5, "zeta=0"),
    ]
    for values, epsilon, match in invalid:
        with pytest.raises(ValueError, match=match):
            conformal_membership_threshold(values, epsilon)


def test_resolved_conformal_threshold_must_exceed_xi() -> None:
    with pytest.raises(ValueError, match="resolved zeta"):
        AutoSafeODD.fit(
            [[0.0], [1.0]],
            calibration_data=[[20.0], [21.0], [22.0]],
            out_of_distribution_data=[[20.0]],
            config={
                "epsilon": 0.5,
                "xi": 0.99,
                "ood_batch_jump": True,
                "normalize": False,
            },
        )
