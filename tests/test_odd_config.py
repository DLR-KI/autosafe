# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
# ruff:file-ignore[ambiguous-unicode-character-string]
"""Tests for the high-level autoSAFE configuration API."""

from collections.abc import Callable
from typing import Any, Literal, cast

import numpy as np
import pytest

from autosafe.odd import (
    PARAMETER_SPECS,
    AutoSafeConfig,
    CalibratedRBFConfig,
    ConformalMembershipConfig,
    EvaluationConfig,
    FixedMembershipConfig,
    ManualRBFConfig,
    NormalizationConfig,
    OODConsistencyConfig,
)


def test_descriptive_and_paper_aliases_resolve_identically() -> None:
    descriptive = AutoSafeConfig.from_mapping({
        "decay_per_median_gap": 1.5,
        "width_in_median_gaps": 2.5,
        "relative_variance_floor": 1e-4,
        "affinity_threshold": 0.7,
        "max_ood_affinity": 0.2,
        "covariance_shrink_factor": 0.8,
        "local_noise_multiple": 4.0,
    })
    paper = AutoSafeConfig.from_mapping({
        "γ": 1.5,
        "s": 2.5,
        "λ_rel": 1e-4,
        "ζ": 0.7,
        "ξ": 0.2,
        "c": 0.8,
        "m": 4.0,
    })

    assert descriptive == paper
    assert isinstance(paper.kernel, CalibratedRBFConfig)
    assert isinstance(paper.membership, FixedMembershipConfig)
    assert paper.ood is not None
    assert paper.ood.covariance_shrink_factor == pytest.approx(0.8)


@pytest.mark.parametrize("style", ["canonical", "paper"])
def test_config_mapping_round_trip(
    style: Literal["canonical", "paper"],
) -> None:
    config = AutoSafeConfig.from_mapping({
        "κ": [2.0, 3.0],
        "η": [0.5, 0.25],
        "λ": [0.01, 0.02],
        "nearest_neighbor_mode": "per_dimension",
        "ε": 0.1,
        "ξ": 0.05,
        "c": 0.85,
        "ood_max_iterations": 1234,
        "ood_refresh_interval": 20,
        "ood_log_interval": 30,
        "ood_batch_jump": True,
        "normalize": False,
        "m": 2.0,
    })

    encoded = config.to_mapping(style=style)

    assert AutoSafeConfig.from_mapping(encoded) == config


def test_manual_and_calibrated_kernel_parameters_cannot_be_mixed() -> None:
    with pytest.raises(ValueError, match="cannot be mixed"):
        AutoSafeConfig.from_mapping({
            "gamma": 1.0,
            "kappa": 2.0,
            "zeta": 0.5,
        })

    with pytest.raises(ValueError, match="either lambda"):
        AutoSafeConfig.from_mapping({
            "kappa": 2.0,
            "eta": 1.0,
            "lambda": 0.1,
            "lambda_rel": 0.01,
            "zeta": 0.5,
        })


def test_membership_mode_and_threshold_must_agree() -> None:
    with pytest.raises(ValueError, match="define either zeta or epsilon"):
        AutoSafeConfig.from_mapping({"zeta": 0.5, "epsilon": 0.1})
    with pytest.raises(ValueError, match="conflicts with zeta"):
        AutoSafeConfig.from_mapping({
            "membership": {
                "mode": "conformal",
                "affinity_threshold": 0.5,
            }
        })


def test_xi_must_be_below_fixed_zeta() -> None:
    with pytest.raises(ValueError, match="must be below"):
        AutoSafeConfig.from_mapping({"zeta": 0.2, "xi": 0.2})


def test_duplicate_aliases_are_rejected() -> None:
    with pytest.raises(ValueError, match="both define"):
        AutoSafeConfig.from_mapping({
            "gamma": 1.0,
            "kernel": {"decay_per_median_gap": 2.0},
            "zeta": 0.5,
        })


def test_legacy_calibration_c_does_not_claim_plain_c() -> None:
    with pytest.warns(DeprecationWarning, match="deprecated"):
        config = AutoSafeConfig.from_mapping({
            "calibration_c": 1.25,
            "zeta": 0.7,
            "xi": 0.2,
            "c": 0.75,
        })

    assert isinstance(config.kernel, CalibratedRBFConfig)
    assert config.kernel.decay_per_median_gap == pytest.approx(1.25)
    assert config.ood is not None
    assert config.ood.covariance_shrink_factor == pytest.approx(0.75)


def test_manual_config_and_conformal_config_are_inferred() -> None:
    manual = AutoSafeConfig.from_mapping({
        "kappa": 2.0,
        "eta": 0.5,
        "lambda_rel": 0.01,
        "zeta": 0.6,
    })
    conformal = AutoSafeConfig.from_mapping({"epsilon": 0.1})

    assert isinstance(manual.kernel, ManualRBFConfig)
    assert isinstance(conformal.membership, ConformalMembershipConfig)


def test_parameter_registry_exposes_every_paper_symbol() -> None:
    symbols = {spec.symbol for spec in PARAMETER_SPECS}

    assert {"γ", "s", "λ_rel", "κ", "η", "λ", "ζ", "ε", "ξ", "c", "m"} <= symbols


@pytest.mark.parametrize(
    ("factory", "match"),
    [
        (lambda: CalibratedRBFConfig(decay_per_median_gap=True), "real number"),
        (lambda: CalibratedRBFConfig(decay_per_median_gap=np.inf), "finite"),
        (lambda: CalibratedRBFConfig(width_in_median_gaps=0.0), "greater than 0"),
        (lambda: CalibratedRBFConfig(relative_variance_floor=1.0), "in \\(0, 1\\)"),
        (
            lambda: OODConsistencyConfig(
                max_affinity=0.1,
                max_iterations=cast("Any", 1.5),
            ),
            "integer",
        ),
        (lambda: OODConsistencyConfig(max_affinity=0.1, refresh_interval=0), "greater"),
        (
            lambda: ManualRBFConfig(nearest_neighbor_mode=cast("Any", "local")),
            "nearest_neighbor_mode",
        ),
        (lambda: NormalizationConfig(method=cast("Any", "other")), "method"),
        (
            lambda: NormalizationConfig(target_range=cast("Any", (0.0,))),
            "two values",
        ),
        (lambda: NormalizationConfig(target_range=(1.0, 1.0)), "lower bound"),
        (lambda: EvaluationConfig(local_noise_multiple=0.0), "greater than 0"),
    ],
)
def test_configuration_value_validation(
    factory: Callable[[], object],
    match: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=match):
        factory()


@pytest.mark.parametrize(
    "value",
    [
        np.array(2.0),
        np.array([2.0, 3.0]),
    ],
)
def test_manual_scale_accepts_numpy_scalars_and_vectors(value: np.ndarray) -> None:
    config = ManualRBFConfig(maximum_variance=cast("Any", value))
    expected = (2.0, 3.0) if value.ndim else 2.0

    assert config.maximum_variance == expected


@pytest.mark.parametrize(
    ("value", "match"),
    [
        (np.ones((1, 1)), "one-dimensional"),
        ([], "must not be empty"),
    ],
)
def test_manual_scale_rejects_invalid_sequences(value: object, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        ManualRBFConfig(maximum_variance=cast("Any", value))


def test_mapping_rejects_invalid_names_and_modes() -> None:
    with pytest.raises(TypeError, match="names must be strings"):
        AutoSafeConfig.from_mapping(cast("Any", {1: 2}))
    with pytest.raises(ValueError, match="unknown autoSAFE parameter"):
        AutoSafeConfig.from_mapping({"zetaa": 0.5})
    with pytest.raises(ValueError, match=r"kernel\.mode"):
        AutoSafeConfig.from_mapping({"kernel_mode": "other", "zeta": 0.5})
    with pytest.raises(ValueError, match="calibrated parameters"):
        AutoSafeConfig.from_mapping({
            "kernel_mode": "manual",
            "gamma": 1.0,
            "zeta": 0.5,
        })
    with pytest.raises(ValueError, match="manual parameters"):
        AutoSafeConfig.from_mapping({
            "kernel_mode": "calibrated",
            "kappa": 1.0,
            "zeta": 0.5,
        })
    with pytest.raises(ValueError, match=r"membership\.mode"):
        AutoSafeConfig.from_mapping({"membership_mode": "other", "zeta": 0.5})
    with pytest.raises(ValueError, match="requires zeta or epsilon"):
        AutoSafeConfig.from_mapping({"kernel_mode": "auto"})
    with pytest.raises(ValueError, match="require xi"):
        AutoSafeConfig.from_mapping({
            "zeta": 0.5,
            "ood_shrink_factor": 0.8,
        })


def test_calibrated_serialization_and_invalid_style() -> None:
    config = AutoSafeConfig.from_mapping({"zeta": 0.5})

    assert config.to_mapping(style="paper")["ζ"] == pytest.approx(0.5)
    canonical = config.to_mapping()
    kernel = cast("dict[str, object]", canonical["kernel"])
    assert kernel["mode"] == "calibrated"
    assert AutoSafeConfig.from_mapping(canonical) == config
    with pytest.raises(ValueError, match="style"):
        config.to_mapping(style=cast("Any", "compact"))
