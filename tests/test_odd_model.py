# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
# ruff:file-ignore[ambiguous-unicode-character-string]
"""Tests for fitting and querying a complete autoSAFE ODD."""

import warnings
from typing import cast

import numpy as np
import pytest
from tqdm import TqdmExperimentalWarning

from autosafe.deduplication import deduplicate_points, exact_equality_policy
from autosafe.exceptions import (
    CalibrationSetTooSmallError,
    EmptyParameterError,
    InvalidLogSurvivalError,
    NonFiniteParameterError,
    NonPositiveParameterError,
    ParameterOrderError,
    ParameterOutOfRangeError,
    ShapeMismatchError,
    TooFewAnchorsError,
    UncertifiableThresholdError,
)
from autosafe.neighbors import find_closest_vectors_by_index
from autosafe.odd import AutoSafeConfig, AutoSafeODD
from autosafe.pointsets import subsample_indices
from autosafe.sample import Sample
from autosafe.samples import Samples


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
    with pytest.raises(ShapeMismatchError, match="out_of_distribution_data has shape"):
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
    ("data", "error"),
    [
        ([0.0, 1.0], ShapeMismatchError),
        (np.empty((0, 1)), EmptyParameterError),
        (np.empty((1, 0)), EmptyParameterError),
        ([[np.inf]], NonFiniteParameterError),
    ],
)
def test_fit_rejects_invalid_anchor_matrices(
    data: object, error: type[Exception]
) -> None:
    with pytest.raises(error, match="in_distribution_data"):
        AutoSafeODD.fit(data, config={"zeta": 0.5})


def test_fit_rejects_single_anchor_and_invalid_feature_names() -> None:
    with pytest.raises(TooFewAnchorsError):
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
    with pytest.raises(
        ParameterOrderError,
        match="variance_floor must be strictly less than maximum_variance",
    ):
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

    invalid: list[tuple[object, float, type[Exception]]] = [
        ([-1.0], 0.0, ParameterOutOfRangeError),
        ([[-1.0]], 0.5, ShapeMismatchError),
        ([], 0.5, EmptyParameterError),
        ([np.nan], 0.5, InvalidLogSurvivalError),
        ([-1.0], 0.1, CalibrationSetTooSmallError),
        ([0.0], 0.5, UncertifiableThresholdError),
    ]
    for values, epsilon, error in invalid:
        with pytest.raises(error):
            conformal_membership_threshold(values, epsilon)


def test_resolved_conformal_threshold_must_exceed_xi() -> None:
    with pytest.raises(ParameterOrderError, match="xi must be strictly less than zeta"):
        AutoSafeODD.fit(
            [[0.0], [1.0]],
            calibration_data=[[20.0], [21.0], [22.0]],
            out_of_distribution_data=[[20.0]],
            # A fixed width keeps zeta above zero with only two anchors,
            # so this test exercises the xi < zeta check itself.
            config={
                "epsilon": 0.5,
                "xi": 0.99,
                "s": 3.0,
                "ood_batch_jump": True,
                "normalize": False,
            },
        )


def _calibrated_config() -> AutoSafeConfig:
    return AutoSafeConfig.from_mapping({
        "γ": 1.2,
        "s": 2.0,
        "λ_rel": 1e-4,
        "ζ": 0.5,
    })


def test_fit_deduplicates_normalizes_then_subsamples() -> None:
    rng = np.random.default_rng(3)
    rows = np.repeat(rng.normal(size=(40, 2)), 3, axis=0)
    policy = exact_equality_policy(2)

    model = AutoSafeODD.fit(
        rows,
        config=_calibrated_config(),
        deduplication=policy,
        max_anchors=25,
        anchor_seed=4,
    )

    dedup = deduplicate_points(rows, policy)
    keep = subsample_indices(dedup.n_output, 25, 4)
    assert model.deduplication_result is not None
    assert model.deduplication_result.n_input == rows.shape[0]
    assert model.deduplication_result.n_output == 40
    assert model.deduplication_result.output_digest == dedup.output_digest
    assert model.deduplication_result.groups is None
    np.testing.assert_array_equal(model.anchor_indices, keep)
    np.testing.assert_array_equal(model.in_distribution_data, dedup.points[keep])
    assert len(model.samples) == 25
    # The normalizer is fit on all 40 representatives, not on the 25
    # subsampled anchors.
    reference = AutoSafeODD.fit(dedup.points, config=_calibrated_config())
    assert reference.normalizer is not None
    np.testing.assert_array_equal(
        model.normalized_in_distribution_data,
        reference.normalizer.transform(dedup.points[keep]),
    )


def test_fit_without_preparation_records_nothing() -> None:
    model = AutoSafeODD.fit(
        np.array([[0.0], [1.0], [2.0]]), config=_calibrated_config()
    )

    assert model.deduplication_result is None
    assert model.anchor_indices is None


def test_fit_validates_anchor_preparation() -> None:
    rows = np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 1.0]])
    with pytest.raises(NonPositiveParameterError):
        AutoSafeODD.fit(rows, config=_calibrated_config(), max_anchors=0)
    with pytest.raises(TooFewAnchorsError):
        AutoSafeODD.fit(rows, config=_calibrated_config(), max_anchors=1)
    with pytest.raises(TooFewAnchorsError):
        AutoSafeODD.fit(
            rows[:2],
            config=_calibrated_config(),
            deduplication=exact_equality_policy(2),
        )


def _tqdm_warnings(rows: np.ndarray, *, show_progress: bool) -> int:
    # autosafe filters TqdmExperimentalWarning on import; "always"
    # takes precedence inside the context, so bars would be recorded.
    # The OOD adjustment always draws a bar when enabled, and it builds
    # the per-anchor kernels, so every bar of fit is exercised.
    config = AutoSafeConfig.from_mapping({
        "γ": 1.2,
        "s": 2.0,
        "λ_rel": 1e-4,
        "ζ": 0.5,
        "ξ": 0.1,
        "c": 0.9,
    })
    observed_ood = rows[:3] + 0.01
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        AutoSafeODD.fit(
            rows,
            config=config,
            out_of_distribution_data=observed_ood,
            show_progress=show_progress,
        )
    return sum(issubclass(w.category, TqdmExperimentalWarning) for w in caught)


def test_fit_reuses_calibration_neighbors() -> None:
    rng = np.random.default_rng(6)
    rows = rng.normal(size=(60, 3))

    model = AutoSafeODD.fit(rows, config=_calibrated_config())

    anchors = model.normalized_in_distribution_data
    expected = find_closest_vectors_by_index(anchors, disable_tqdm=True)
    for index, sample in enumerate(model.samples):
        assert sample.closest_sample is not None
        np.testing.assert_array_equal(
            sample.closest_sample[0].x, model.samples[int(expected[index])].x
        )
    rebuilt = Samples(
        [Sample(row) for row in anchors],
        kernel_kwargs=model.samples.kernel_kwargs,
        show_progress=False,
    )
    np.testing.assert_array_equal(
        np.asarray(model.samples(anchors * 1.01)),
        np.asarray(rebuilt(anchors * 1.01)),
    )


def test_samples_reject_misshaped_neighbor_indices() -> None:
    with pytest.raises(ShapeMismatchError):
        Samples(
            [Sample(np.array([0.0])), Sample(np.array([1.0]))],
            kernel_kwargs={"kappa": 1.0, "eta": 1.0, "lam": 1e-4},
            neighbor_indices=np.array([1, 0, 0]),
            show_progress=False,
        )
