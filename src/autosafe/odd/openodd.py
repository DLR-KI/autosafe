# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""Export a fitted autoSAFE model as ASAM OpenODD 1.0.0 YAML."""

import dataclasses
import math
import pathlib
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, cast

import numpy as np

from autosafe.odd.model import AutoSafeODD
from autosafe.odd.openodd_schema import (
    OPENODD_VERSION,
    dump_openodd_yaml,
    validate_openodd_document,
)
from autosafe.typing import NPMatrix, OpenODDFeature

_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]*$")
_BUILTIN_UNIT_TYPES = frozenset({
    "length",
    "area",
    "volume",
    "angle",
    "force",
    "weight",
    "duration",
    "time",
    "count",
    "fraction",
    "temperature",
    "frequency",
    "charge",
    "illuminance",
    "luminous_flux",
    "sound_intensity",
    "cloud_coverage",
    "grains",
    "electric_potential",
    "electric_current",
    "electric_current_density",
    "power",
    "data_size",
    "velocity",
    "precipitation_rate",
    "occurrence",
    "bandwidth",
    "pressure",
    "torque",
    "acceleration",
    "risk",
    "reliability",
    "confidence",
    "percentile",
})


def _validate_identifier(value: str, name: str) -> None:
    if not _IDENTIFIER_PATTERN.fullmatch(value):
        raise ValueError(
            f"{name} must start with a letter and contain only letters, "
            "digits, dots, hyphens, or underscores"
        )


def _indices(value: Sequence[int], name: str) -> tuple[int, ...]:
    result = tuple(value)
    if not result:
        raise ValueError(f"{name} must contain at least one feature index")
    if any(isinstance(index, bool) or index < 0 for index in result):
        raise ValueError(f"{name} must contain non-negative integers")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must not contain duplicate indices")
    return result


@dataclass(frozen=True, slots=True)
class OpenODDNumericFeature:
    """Map one model dimension to an OpenODD numeric concept."""

    index: int
    concept_id: str
    primitive_type: Literal["float", "integer"] = "float"
    unit_type: str | None = None
    unit: str | None = None

    def __post_init__(self) -> None:
        """Validate the numeric feature mapping.

        Raises:
            ValueError: If the mapping is not valid.
        """
        if isinstance(self.index, bool) or self.index < 0:
            raise ValueError("index must be a non-negative integer")
        _validate_identifier(self.concept_id, "concept_id")
        if self.primitive_type not in {"float", "integer"}:
            raise ValueError("primitive_type must be 'float' or 'integer'")
        if self.unit_type is not None:
            _validate_identifier(self.unit_type, "unit_type")
            if self.unit_type not in _BUILTIN_UNIT_TYPES:
                raise ValueError(
                    f"unit_type must be an ASAM built-in unit type; "
                    f"received {self.unit_type!r}"
                )
        if self.unit is not None and self.unit_type is None:
            raise ValueError("unit requires a unit_type")
        if self.unit is not None and any(char.isspace() for char in self.unit):
            raise ValueError("unit must not contain whitespace")


@dataclass(frozen=True, slots=True)
class OpenODDCategoricalFeature:
    """Map encoded model dimensions to an OpenODD categorical concept.

    ``literals`` maps each OpenODD literal to its encoded prototype in
    the dimensions named by ``indices``.  One-hot encodings are the
    common case, but any finite encoding can be supplied.
    """

    indices: tuple[int, ...]
    concept_id: str
    literals: Mapping[str, Sequence[float]]

    def __post_init__(self) -> None:
        """Validate and freeze the categorical encoding.

        Raises:
            ValueError: If the mapping is not valid.
        """
        normalized_indices = _indices(self.indices, "indices")
        _validate_identifier(self.concept_id, "concept_id")
        if not self.literals:
            raise ValueError("literals must not be empty")
        normalized_literals: dict[str, tuple[float, ...]] = {}
        for literal, prototype in self.literals.items():
            _validate_identifier(literal, "categorical literal")
            values = tuple(float(value) for value in prototype)
            if len(values) != len(normalized_indices):
                raise ValueError(
                    f"literal {literal!r} has {len(values)} encoded values; "
                    f"expected {len(normalized_indices)}"
                )
            if not np.isfinite(values).all():
                raise ValueError(f"literal {literal!r} must contain only finite values")
            normalized_literals[literal] = values
        object.__setattr__(self, "indices", normalized_indices)
        object.__setattr__(self, "literals", normalized_literals)


@dataclass(frozen=True, slots=True)
class OpenODDExportMetadata:
    """Human-readable and reproducibility metadata for an export."""

    title: str = "Generated operational design domain"
    description: str = "Data-derived operational design domain"
    root_id: str = "generated_odd"
    extra: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate export metadata.

        Raises:
            ValueError: If the metadata is not valid.
        """
        if not self.title.strip():
            raise ValueError("title must not be empty")
        _validate_identifier(self.root_id, "root_id")
        if "autosafe" in self.extra:
            raise ValueError("extra metadata must not replace reserved 'autosafe'")


def _safe_identifier(value: str, fallback: str) -> str:
    result = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_.-")
    if not result or not result[0].isalpha():
        result = f"{fallback}_{result}" if result else fallback
    return result


def _default_features(odd: AutoSafeODD) -> tuple[OpenODDFeature, ...]:
    count = odd.in_distribution_data.shape[1]
    raw_names = odd.feature_names or tuple(f"feature_{index}" for index in range(count))
    used: set[str] = set()
    result: list[OpenODDFeature] = []
    for index, raw_name in enumerate(raw_names):
        base = _safe_identifier(raw_name, f"feature_{index}")
        concept_id = base
        suffix = 2
        while concept_id in used:
            concept_id = f"{base}_{suffix}"
            suffix += 1
        used.add(concept_id)
        result.append(
            OpenODDNumericFeature(
                index=index,
                concept_id=concept_id,
            )
        )
    return tuple(result)


def _validate_features(
    features: Sequence[OpenODDFeature],
    feature_count: int,
) -> tuple[OpenODDFeature, ...]:
    result = tuple(features)
    if not result:
        raise ValueError("at least one OpenODD feature is required")
    concept_ids = [feature.concept_id for feature in result]
    if len(concept_ids) != len(set(concept_ids)):
        raise ValueError("OpenODD feature concept IDs must be unique")
    covered: list[int] = []
    for feature in result:
        if isinstance(feature, OpenODDNumericFeature):
            covered.append(feature.index)
        else:
            covered.extend(feature.indices)
    expected = list(range(feature_count))
    if sorted(covered) != expected:
        raise ValueError(
            "OpenODD features must map every model dimension exactly once; "
            f"received {sorted(covered)!r}, expected {expected!r}"
        )
    return result


def _plain(value: object) -> object:  # ruff:ignore[too-many-return-statements]
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _plain(dataclasses.asdict(value))
    if isinstance(value, np.ndarray):
        return cast("Any", value).tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    if isinstance(value, list):
        return [_plain(item) for item in value]
    return value


def _flatten_metadata(
    value: Mapping[str, object],
    prefix: str = "",
) -> dict[str, str]:
    result: dict[str, str] = {}
    for key, item in value.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        plain = _plain(item)
        if plain is None:
            continue
        if isinstance(plain, Mapping):
            result.update(
                _flatten_metadata(
                    cast("Mapping[str, object]", plain),
                    path,
                )
            )
        elif isinstance(plain, list):
            result.update(
                _flatten_metadata(
                    {str(index): child for index, child in enumerate(plain)},
                    path,
                )
            )
        elif isinstance(plain, bool):
            result[path] = str(plain).lower()
        else:
            result[path] = str(plain)
    return result


def _resolved_parameters(odd: AutoSafeODD) -> dict[str, object]:
    resolved = odd.resolved_config
    result = {
        "kernel": cast("dict[str, object]", _plain(resolved.kernel)),
        "membership": cast("dict[str, object]", _plain(resolved.membership)),
    }
    if odd.ood_consistency_result is not None:
        result["ood_consistency"] = cast(
            "dict[str, object]",
            _plain(odd.ood_consistency_result),
        )
    return result


def _normalized_kernel_boxes(odd: AutoSafeODD) -> tuple[NPMatrix, NPMatrix, int]:
    threshold = odd.resolved_config.membership.affinity_threshold
    squared_radius = -2.0 * math.log(threshold)
    feature_count = odd.in_distribution_data.shape[1]
    lower: list[np.ndarray] = []
    upper: list[np.ndarray] = []
    full_covariance_count = 0
    for sample in odd.samples.samples:
        kernel = sample.kernel
        sigma = getattr(kernel, "sigma", None)
        if sigma is None:
            raise ValueError("all fitted samples must have initialized RBF kernels")
        covariance = np.asarray(sigma, dtype=np.float64)
        diagonal = np.diag(np.diag(covariance))
        if np.allclose(covariance, diagonal):
            variances = np.diag(covariance)
            half_width = np.sqrt(squared_radius * variances / feature_count)
        else:
            eigenvalues = np.linalg.eigvalsh(covariance)
            minimum = float(np.min(eigenvalues))
            if minimum <= 0.0:
                raise ValueError("kernel covariance must be positive definite")
            half_width = np.full(
                feature_count,
                math.sqrt(squared_radius * minimum / feature_count),
                dtype=np.float64,
            )
            full_covariance_count += 1
        center = np.asarray(sample.x, dtype=np.float64)
        lower.append(center - half_width)
        upper.append(center + half_width)
    return np.stack(lower), np.stack(upper), full_covariance_count


def _original_coordinate_boxes(
    odd: AutoSafeODD,
) -> tuple[NPMatrix, NPMatrix, int]:
    lower, upper, full_covariance_count = _normalized_kernel_boxes(odd)
    if odd.normalizer is not None:
        lower = np.asarray(odd.normalizer.inverse_transform(lower), dtype=np.float64)
        upper = np.asarray(odd.normalizer.inverse_transform(upper), dtype=np.float64)
    if not np.isfinite(lower).all() or not np.isfinite(upper).all():
        raise ValueError("exported OpenODD bounds must be finite")
    return np.minimum(lower, upper), np.maximum(lower, upper), full_covariance_count


def _number(value: float) -> str:
    result = format(value, ".15g")
    return "0" if result == "-0" else result


def _numeric_expression(
    feature: OpenODDNumericFeature,
    lower: np.ndarray,
    upper: np.ndarray,
) -> str | None:
    low = float(lower[feature.index])
    high = float(upper[feature.index])
    if feature.primitive_type == "integer":
        low = float(math.ceil(low))
        high = float(math.floor(high))
        if low > high:
            return None
    unit = f" {feature.unit}" if feature.unit is not None else ""
    return f"[{_number(low)} .. {_number(high)}]{unit}"


def _categorical_expression(
    feature: OpenODDCategoricalFeature,
    lower: np.ndarray,
    upper: np.ndarray,
) -> list[str]:
    indices = np.asarray(feature.indices, dtype=np.int64)
    low = lower[indices]
    high = upper[indices]
    result: list[str] = []
    for literal, prototype in feature.literals.items():
        encoded = np.asarray(prototype, dtype=np.float64)
        if np.all(encoded >= low) and np.all(encoded <= high):
            result.append(literal)
    return result


def _region_conditions(
    features: Sequence[OpenODDFeature],
    lower: np.ndarray,
    upper: np.ndarray,
) -> dict[str, object] | None:
    conditions: dict[str, object] = {}
    for feature in features:
        if isinstance(feature, OpenODDNumericFeature):
            expression = _numeric_expression(feature, lower, upper)
            if expression is None:
                return None
            conditions[feature.concept_id] = expression
        else:
            literals = _categorical_expression(feature, lower, upper)
            if not literals:
                return None
            conditions[feature.concept_id] = literals
    return conditions


def _taxonomy(features: Sequence[OpenODDFeature]) -> dict[str, object]:
    concepts: dict[str, object] = {}
    for feature in features:
        if isinstance(feature, OpenODDNumericFeature):
            definition = feature.primitive_type
            if feature.unit_type is not None:
                definition = f"{definition} {feature.unit_type}"
            concepts[feature.concept_id] = definition
        else:
            concepts[feature.concept_id] = list(feature.literals)
    return {"operational_conditions": concepts}


def build_openodd_yaml(  # ruff:ignore[too-many-locals]
    odd: AutoSafeODD,
    *,
    features: Sequence[OpenODDFeature] | None = None,
    metadata: OpenODDExportMetadata | None = None,
) -> dict[str, Any]:
    """Build and validate an ASAM OpenODD 1.0.0 YAML mapping.

    The fitted kernel level set is represented as a union of
    axis-aligned boxes.  Every emitted box is inside one kernel's
    membership-level ellipsoid, so the YAML ODD is a conservative inner
    approximation of the fitted autoSAFE ODD.  The ODD body contains
    only standard OpenODD concepts and expressions.  autoSAFE tuning
    and derivation details are kept in module ``METADATA``.

    Returns:
        dict[str, Any]: Validated YAML-serializable document.

    Raises:
        ValueError: If the model cannot be represented by the supplied
            feature mappings.
    """
    export_metadata = metadata or OpenODDExportMetadata()
    feature_count = odd.in_distribution_data.shape[1]
    mappings = _validate_features(
        features if features is not None else _default_features(odd),
        feature_count,
    )
    lower, upper, full_covariance_count = _original_coordinate_boxes(odd)

    modules: dict[str, object] = {}
    references: dict[str, bool] = {}
    seen_conditions: set[str] = set()
    skipped_regions = 0
    for index, (box_lower, box_upper) in enumerate(zip(lower, upper, strict=True)):
        conditions = _region_conditions(mappings, box_lower, box_upper)
        if conditions is None:
            skipped_regions += 1
            continue
        fingerprint = repr(conditions)
        if fingerprint in seen_conditions:
            continue
        seen_conditions.add(fingerprint)
        module_id = f"{export_metadata.root_id}.region_{index + 1:06d}"
        references[module_id] = True
        modules[module_id] = {
            "TITLE": f"Derived operational region {index + 1}",
            "ACTIVE": True,
            "INCLUDE_AND": conditions,
        }
    if not modules:
        raise ValueError(
            "no OpenODD region can be represented by the supplied feature mapping"
        )

    autosafe_metadata = {
        "asam_openodd_version": OPENODD_VERSION,
        "derivation": {
            "representation": "union_of_axis_aligned_inner_boxes",
            "conservative_inner_approximation": True,
            "source_kernel_count": len(odd.samples.samples),
            "exported_region_count": len(modules),
            "categorically_unrepresentable_regions": skipped_regions,
            "full_covariance_isotropic_boxes": full_covariance_count,
        },
        "requested_parameters": odd.config.to_mapping(style="canonical"),
        "paper_notation": odd.config.to_mapping(style="paper"),
        "parameter_sources": dict(odd.config.parameter_sources),
        "resolved_parameters": _resolved_parameters(odd),
    }
    root_metadata = _flatten_metadata({
        **dict(export_metadata.extra),
        "autosafe": autosafe_metadata,
    })
    document: dict[str, Any] = {
        "TAXONOMY": _taxonomy(mappings),
        "ODD": {
            export_metadata.root_id: {
                "TITLE": export_metadata.title,
                "DESCRIPTION": export_metadata.description,
                "ACTIVE": True,
                "METADATA": root_metadata,
                "INCLUDE_OR": references,
            }
        },
        "MODULES": modules,
    }
    validate_openodd_document(document)
    return document


def write_openodd_yaml(
    odd: AutoSafeODD,
    path: pathlib.Path | str,
    *,
    features: Sequence[OpenODDFeature] | None = None,
    metadata: OpenODDExportMetadata | None = None,
) -> pathlib.Path:
    """Build, validate, and write an ASAM OpenODD 1.0.0 YAML file.

    Returns:
        pathlib.Path: Written YAML path.

    Raises:
        ValueError: If the suffix is not ``.yaml`` or ``.yml``, or the
            model cannot be represented.
    """
    output_path = pathlib.Path(path)
    if output_path.suffix.casefold() not in {".yaml", ".yml"}:
        raise ValueError("OpenODD output path must end in .yaml or .yml")
    document = build_openodd_yaml(
        odd,
        features=features,
        metadata=metadata,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(dump_openodd_yaml(document), encoding="utf-8")
    return output_path


__all__ = [
    "OpenODDCategoricalFeature",
    "OpenODDExportMetadata",
    "OpenODDFeature",
    "OpenODDNumericFeature",
    "build_openodd_yaml",
    "write_openodd_yaml",
]
