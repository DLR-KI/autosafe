# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT

import pathlib
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

import autosafe.odd.openodd as exporter
from autosafe.odd import (
    AutoSafeConfig,
    AutoSafeODD,
    OpenODDCategoricalFeature,
    OpenODDExportMetadata,
    OpenODDNumericFeature,
    OpenODDValidationError,
    build_openodd_yaml,
    dump_openodd_yaml,
    parse_openodd_yaml,
    validate_openodd_document,
    write_openodd_yaml,
)


def _model() -> AutoSafeODD:
    config = AutoSafeConfig.from_mapping({
        "kappa": 0.5,
        "eta": 1.0,
        "zeta": 0.8,
        "normalization.method": "zscore",
        "nearest_neighbor_mode": "per_dimension",
    })
    return AutoSafeODD.fit(
        [
            [10.0, 1.0, 0.0],
            [12.0, 1.0, 0.0],
            [20.0, 0.0, 1.0],
            [22.0, 0.0, 1.0],
        ],
        config=config,
        feature_names=("size", "color_red", "color_blue"),
    )


def _features() -> tuple[
    OpenODDNumericFeature,
    OpenODDCategoricalFeature,
]:
    return (
        OpenODDNumericFeature(index=0, concept_id="size", unit_type="area"),
        OpenODDCategoricalFeature(
            indices=(1, 2),
            concept_id="color",
            literals={"red": (1.0, 0.0), "blue": (0.0, 1.0)},
        ),
    )


def test_export_is_standard_yaml_with_autosafe_only_in_metadata() -> None:
    document = build_openodd_yaml(
        _model(),
        features=_features(),
        metadata=OpenODDExportMetadata(
            title="Example ODD",
            description="Derived from observed conditions",
        ),
    )

    validate_openodd_document(document)
    root = document["ODD"]["generated_odd"]
    assert root["TITLE"] == "Example ODD"
    assert root["ACTIVE"] is True
    assert root["METADATA"]["autosafe.asam_openodd_version"] == "1.0.0"
    assert (
        root["METADATA"]["autosafe.requested_parameters.membership.affinity_threshold"]
        == "0.8"
    )
    assert document["TAXONOMY"]["operational_conditions"] == {
        "size": "float area",
        "color": ["red", "blue"],
    }
    for module in document["MODULES"].values():
        assert set(module["INCLUDE_AND"]) == {"size", "color"}
        assert "affinity" not in module["INCLUDE_AND"]


def test_yaml_round_trip_and_write_validation(tmp_path: pathlib.Path) -> None:
    model = _model()
    document = build_openodd_yaml(model, features=_features())
    text = dump_openodd_yaml(document)

    assert parse_openodd_yaml(text) == document
    path = write_openodd_yaml(
        model,
        tmp_path / "generated.yaml",
        features=_features(),
    )
    assert (
        parse_openodd_yaml(path.read_text(encoding="utf-8"))["ODD"] == document["ODD"]
    )


def test_validator_rejects_unknown_condition_reference() -> None:
    document = build_openodd_yaml(_model(), features=_features())
    first_module = next(iter(document["MODULES"].values()))
    first_module["INCLUDE_AND"]["not_in_taxonomy"] = True

    with pytest.raises(OpenODDValidationError, match="unknown concept"):
        validate_openodd_document(document)


def test_validator_rejects_duplicate_yaml_keys() -> None:
    text = """
TAXONOMY:
  conditions:
    speed: float velocity
ODD:
  main:
    TITLE: First
    TITLE: Second
    ACTIVE: true
    INCLUDE_AND:
      speed: "[0 .. 10] km/h"
"""
    with pytest.raises(OpenODDValidationError, match="duplicate YAML key"):
        parse_openodd_yaml(text)


def test_feature_mapping_must_cover_every_model_dimension() -> None:
    with pytest.raises(ValueError, match="every model dimension"):
        build_openodd_yaml(
            _model(),
            features=(OpenODDNumericFeature(index=0, concept_id="size"),),
        )


def test_writer_rejects_non_yaml_suffix(tmp_path: pathlib.Path) -> None:
    with pytest.raises(ValueError, match=r"\.yaml"):
        write_openodd_yaml(_model(), tmp_path / "generated.txt")


@pytest.mark.parametrize(
    ("factory", "match"),
    [
        (lambda: OpenODDNumericFeature(index=-1, concept_id="x"), "index"),
        (
            lambda: OpenODDNumericFeature(index=0, concept_id="not valid"),
            "concept_id",
        ),
        (
            lambda: OpenODDNumericFeature(
                index=0,
                concept_id="x",
                primitive_type=cast("Any", "number"),
            ),
            "primitive_type",
        ),
        (
            lambda: OpenODDNumericFeature(index=0, concept_id="x", unit_type="custom"),
            "built-in unit type",
        ),
        (lambda: OpenODDNumericFeature(index=0, concept_id="x", unit="m"), "unit_type"),
        (
            lambda: OpenODDNumericFeature(
                index=0,
                concept_id="x",
                unit_type="length",
                unit="m s",
            ),
            "whitespace",
        ),
        (
            lambda: OpenODDCategoricalFeature(
                indices=(),
                concept_id="color",
                literals={"red": ()},
            ),
            "at least one",
        ),
        (
            lambda: OpenODDCategoricalFeature(
                indices=(-1,),
                concept_id="color",
                literals={"red": (1.0,)},
            ),
            "non-negative",
        ),
        (
            lambda: OpenODDCategoricalFeature(
                indices=(0, 0),
                concept_id="color",
                literals={"red": (1.0, 1.0)},
            ),
            "duplicate",
        ),
        (
            lambda: OpenODDCategoricalFeature(
                indices=(0,),
                concept_id="color",
                literals={},
            ),
            "must not be empty",
        ),
        (
            lambda: OpenODDCategoricalFeature(
                indices=(0, 1),
                concept_id="color",
                literals={"red": (1.0,)},
            ),
            "expected 2",
        ),
        (
            lambda: OpenODDCategoricalFeature(
                indices=(0,),
                concept_id="color",
                literals={"red": (np.inf,)},
            ),
            "finite",
        ),
        (lambda: OpenODDExportMetadata(title=" "), "title"),
        (
            lambda: OpenODDExportMetadata(extra={"autosafe": "replacement"}),
            "reserved",
        ),
    ],
)
def test_export_configuration_validation(
    factory: Callable[[], object],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        factory()


def test_default_feature_names_are_sanitized_and_deduplicated() -> None:
    odd = SimpleNamespace(
        in_distribution_data=np.zeros((2, 2)),
        feature_names=("a b", "a@b"),
    )

    features = exporter._default_features(cast("Any", odd))

    assert [feature.concept_id for feature in features] == ["a_b", "a_b_2"]
    fallback = exporter._safe_identifier("123", "feature")
    assert fallback == "feature_123"
    assert exporter._safe_identifier("***", "feature") == "feature"


def test_feature_collection_validation() -> None:
    with pytest.raises(ValueError, match="at least one"):
        exporter._validate_features((), 1)
    with pytest.raises(ValueError, match="concept IDs"):
        exporter._validate_features(
            (
                OpenODDNumericFeature(index=0, concept_id="same"),
                OpenODDNumericFeature(index=1, concept_id="same"),
            ),
            2,
        )


def test_plain_metadata_and_ood_result_conversion() -> None:
    assert exporter._plain(np.array([1, 2])) == [1, 2]
    assert exporter._plain(np.int64(2)) == 2
    assert exporter._plain((np.int64(1),)) == [1]
    model = _model()
    model.ood_consistency_result = {"iterations": np.int64(2)}

    assert exporter._resolved_parameters(model)["ood_consistency"] == {"iterations": 2}


def _dummy_odd(covariance: object) -> SimpleNamespace:
    sample = SimpleNamespace(
        x=np.array([0.0, 0.0]),
        kernel=SimpleNamespace(sigma=covariance),
    )
    return SimpleNamespace(
        in_distribution_data=np.zeros((1, 2)),
        resolved_config=SimpleNamespace(
            membership=SimpleNamespace(affinity_threshold=0.5),
        ),
        samples=SimpleNamespace(samples=[sample]),
    )


def test_full_covariance_boxes_and_kernel_errors() -> None:
    lower, upper, full_count = exporter._normalized_kernel_boxes(
        cast("Any", _dummy_odd(np.array([[2.0, 0.5], [0.5, 1.0]]))),
    )

    assert lower.shape == upper.shape == (1, 2)
    assert full_count == 1
    with pytest.raises(ValueError, match="initialized"):
        exporter._normalized_kernel_boxes(cast("Any", _dummy_odd(None)))
    with pytest.raises(ValueError, match="positive definite"):
        exporter._normalized_kernel_boxes(
            cast("Any", _dummy_odd(np.array([[1.0, 2.0], [2.0, 1.0]]))),
        )


def test_nonfinite_original_bounds_are_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        exporter,
        "_normalized_kernel_boxes",
        lambda _odd: (np.array([[np.nan]]), np.array([[1.0]]), 0),
    )
    with pytest.raises(ValueError, match="finite"):
        exporter._original_coordinate_boxes(
            cast("Any", SimpleNamespace(normalizer=None)),
        )


def test_integer_and_categorical_regions_can_be_unrepresentable() -> None:
    integer = OpenODDNumericFeature(
        index=0,
        concept_id="count",
        primitive_type="integer",
    )
    assert exporter._numeric_expression(
        integer,
        np.array([0.1]),
        np.array([0.9]),
    ) is None
    categorical = OpenODDCategoricalFeature(
        indices=(0,),
        concept_id="color",
        literals={"red": (1.0,)},
    )
    assert exporter._region_conditions(
        (categorical,),
        np.array([0.0]),
        np.array([0.5]),
    ) is None


def test_build_skips_unrepresentable_and_duplicate_regions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _model()
    integer_features = (
        OpenODDNumericFeature(index=0, concept_id="size", primitive_type="integer"),
        OpenODDNumericFeature(index=1, concept_id="red"),
        OpenODDNumericFeature(index=2, concept_id="blue"),
    )
    monkeypatch.setattr(
        exporter,
        "_original_coordinate_boxes",
        lambda _odd: (
            np.array([[0.1, 0.0, 0.0]]),
            np.array([[0.9, 1.0, 1.0]]),
            0,
        ),
    )
    with pytest.raises(ValueError, match="no OpenODD region"):
        build_openodd_yaml(model, features=integer_features)

    monkeypatch.setattr(
        exporter,
        "_original_coordinate_boxes",
        lambda _odd: (
            np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
            np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]),
            0,
        ),
    )
    document = build_openodd_yaml(model, features=integer_features)
    assert len(document["MODULES"]) == 1
