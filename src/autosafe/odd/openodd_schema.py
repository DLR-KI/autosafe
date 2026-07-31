# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""Validation for the ASAM OpenODD 1.0.0 YAML mapping.

The ASAM specification defines the YAML mapping normatively in prose
and examples, but does not publish a JSON Schema for it.  This module
therefore provides a derived structural schema plus semantic checks for
the taxonomy, module references, expression types, and module graph.
"""

import pathlib
import re
from collections.abc import Mapping, Sequence
from typing import Any, cast

import yaml
import yaml.resolver
from jsonschema import Draft202012Validator

OPENODD_VERSION = "1.0.0"
"""ASAM OpenODD version implemented by the YAML validator."""

_SECTION_NAMES = frozenset({
    "INCLUDE_AND",
    "INCLUDE_OR",
    "EXCLUDE_AND",
    "EXCLUDE_OR",
})
_MODULE_FIELDS = frozenset({
    "TITLE",
    "DESCRIPTION",
    "ACTIVE",
    "METADATA",
    "LABEL",
    "LABELS",
    "TAGS",
    *_SECTION_NAMES,
})
_PRIMITIVE_PATTERN = r"^(?:float|integer)(?:\s+[A-Za-z][A-Za-z0-9_.-]*)?$|^boolean$"
_NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
_FACTOR = rf"(?:{_NUMBER}|[A-Za-z][A-Za-z0-9_.-]*)"
_TERM = rf"{_FACTOR}(?:\s*[*\/]\s*{_FACTOR})?"
_UNIT = r"(?:\s+[^\s\[\]]+)?"
_BOUND_EXPRESSION = re.compile(rf"^(?:>=|>|<=|<)\s*{_TERM}{_UNIT}$")
_EQUAL_NUMERIC_EXPRESSION = re.compile(rf"^{_TERM}{_UNIT}$")
_RANGE_EXPRESSION = re.compile(rf"^[\[(]\s*{_TERM}\s*\.\.\s*{_TERM}\s*[\])]{_UNIT}$")

OPENODD_1_0_YAML_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "urn:autosafe:schema:asam-openodd:1.0.0:yaml",
    "title": "ASAM OpenODD 1.0.0 YAML mapping",
    "type": "object",
    "minProperties": 1,
    "properties": {
        "IMPORT": {
            "type": "array",
            "minItems": 1,
            "items": {"type": "string", "minLength": 1},
        },
        "TAXONOMY": {
            "type": "object",
            "minProperties": 1,
            "additionalProperties": {"$ref": "#/$defs/taxonomyNode"},
        },
        "unit_types": {
            "type": "array",
            "minItems": 1,
            "items": {"type": "string", "minLength": 1},
        },
        "conversion": {"type": "object", "minProperties": 1},
        "COD": {"type": "object"},
        "OD": {"type": "object"},
        "ODD": {"$ref": "#/$defs/moduleCollection"},
        "TOD": {"$ref": "#/$defs/moduleCollection"},
        "MODULES": {"$ref": "#/$defs/moduleCollection"},
    },
    "additionalProperties": False,
    "anyOf": [
        {"required": ["TAXONOMY"]},
        {"required": ["IMPORT"]},
    ],
    "$defs": {
        "scalar": {
            "type": ["string", "number", "integer", "boolean"],
        },
        "expression": {
            "oneOf": [
                {"$ref": "#/$defs/scalar"},
                {
                    "type": "array",
                    "minItems": 1,
                    "items": {"$ref": "#/$defs/scalar"},
                },
            ],
        },
        "taxonomyNode": {
            "oneOf": [
                {
                    "type": "string",
                    "minLength": 1,
                },
                {
                    "type": "array",
                    "minItems": 1,
                    "uniqueItems": True,
                    "items": {"type": "string", "minLength": 1},
                },
                {
                    "type": "object",
                    "minProperties": 1,
                    "additionalProperties": {"$ref": "#/$defs/taxonomyNode"},
                },
            ],
        },
        "conditionMap": {
            "type": "object",
            "minProperties": 1,
            "properties": {
                "METADATA": {"$ref": "#/$defs/metadata"},
                "AND": {"$ref": "#/$defs/conditionMap"},
                "OR": {"$ref": "#/$defs/conditionMap"},
            },
            "additionalProperties": {"$ref": "#/$defs/expression"},
        },
        "module": {
            "type": "object",
            "required": ["TITLE", "ACTIVE"],
            "properties": {
                "TITLE": {"type": "string", "minLength": 1},
                "DESCRIPTION": {"type": "string"},
                "ACTIVE": {"type": "boolean"},
                "METADATA": {"$ref": "#/$defs/metadata"},
                "LABEL": {"$ref": "#/$defs/stringList"},
                "LABELS": {"$ref": "#/$defs/stringList"},
                "TAGS": {"$ref": "#/$defs/stringList"},
                "INCLUDE_AND": {"$ref": "#/$defs/conditionMap"},
                "INCLUDE_OR": {"$ref": "#/$defs/conditionMap"},
                "EXCLUDE_AND": {"$ref": "#/$defs/conditionMap"},
                "EXCLUDE_OR": {"$ref": "#/$defs/conditionMap"},
            },
            "additionalProperties": False,
            "allOf": [
                {
                    "anyOf": [
                        {"required": ["INCLUDE_AND"]},
                        {"required": ["INCLUDE_OR"]},
                        {"required": ["EXCLUDE_AND"]},
                        {"required": ["EXCLUDE_OR"]},
                    ],
                },
                {
                    "not": {
                        "required": ["INCLUDE_AND", "INCLUDE_OR"],
                    },
                },
                {
                    "not": {
                        "required": ["EXCLUDE_AND", "EXCLUDE_OR"],
                    },
                },
            ],
        },
        "moduleCollection": {
            "type": "object",
            "minProperties": 1,
            "additionalProperties": {"$ref": "#/$defs/module"},
        },
        "metadata": {
            "type": "object",
            "additionalProperties": {
                "type": "string",
            },
        },
        "stringList": {
            "oneOf": [
                {"type": "string", "minLength": 1},
                {
                    "type": "array",
                    "minItems": 1,
                    "uniqueItems": True,
                    "items": {"type": "string", "minLength": 1},
                },
            ],
        },
    },
}
"""Derived JSON Schema for YAML-decoded ASAM OpenODD 1.0.0 data."""

Draft202012Validator.check_schema(OPENODD_1_0_YAML_SCHEMA)


class OpenODDValidationError(ValueError):
    """Raised when a document violates the OpenODD YAML mapping."""


class _UniqueKeyLoader(yaml.SafeLoader):
    """Safe YAML loader rejecting duplicate mapping keys."""


def _construct_mapping(
    loader: _UniqueKeyLoader,
    node: yaml.MappingNode,
    *,
    deep: bool = False,
) -> dict[object, object]:
    """Construct a mapping while rejecting YAML duplicate keys.

    Returns:
        dict[object, object]: Constructed unique-key mapping.

    Raises:
        OpenODDValidationError: If a key occurs more than once.
    """
    result: dict[object, object] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise OpenODDValidationError(
                f"duplicate YAML key {key!r} at line {key_node.start_mark.line + 1}"
            )
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_mapping,
)


def _path_text(parts: Sequence[object]) -> str:
    """Format a JSON-style validation path.

    Returns:
        str: Human-readable path.
    """
    return "$" + "".join(
        f"[{part}]" if isinstance(part, int) else f".{part}" for part in parts
    )


def _schema_validate(document: object) -> None:
    validator = Draft202012Validator(OPENODD_1_0_YAML_SCHEMA)
    errors = sorted(
        validator.iter_errors(document),
        key=lambda error: tuple(str(part) for part in error.absolute_path),
    )
    if not errors:
        return
    error = errors[0]
    path = _path_text(tuple(error.absolute_path))
    raise OpenODDValidationError(f"{path}: {error.message}")


def _taxonomy_concepts(
    taxonomy: Mapping[str, object],
) -> tuple[dict[str, str], dict[str, tuple[object, ...]], set[str]]:
    numeric_or_boolean: dict[str, str] = {}
    categorical: dict[str, tuple[object, ...]] = {}
    all_names: set[str] = set()

    def visit(node: Mapping[str, object], prefix: str = "") -> None:
        for name, value in node.items():
            dotted = f"{prefix}.{name}" if prefix else name
            all_names.update((name, dotted))
            if isinstance(value, str) and re.fullmatch(_PRIMITIVE_PATTERN, value):
                primitive = value.split(maxsplit=1)[0]
                numeric_or_boolean[name] = primitive
                numeric_or_boolean[dotted] = primitive
            elif isinstance(value, list):
                values = tuple(value)
                categorical[name] = values
                categorical[dotted] = values
            elif isinstance(value, Mapping):
                visit(cast("Mapping[str, object]", value), dotted)

    visit(taxonomy)
    return numeric_or_boolean, categorical, all_names


def _labels(module: Mapping[str, object]) -> set[str]:
    result: set[str] = set()
    for key in ("LABEL", "LABELS"):
        value = module.get(key)
        if isinstance(value, str):
            result.add(value)
        elif isinstance(value, list):
            result.update(cast("list[str]", value))
    return result


def _module_collections(
    document: Mapping[str, object],
) -> tuple[dict[str, Mapping[str, object]], set[str]]:
    modules: dict[str, Mapping[str, object]] = {}
    roots: set[str] = set()
    for collection_name in ("ODD", "TOD", "MODULES"):
        collection = document.get(collection_name)
        if not isinstance(collection, Mapping):
            continue
        for module_id, value in collection.items():
            if module_id in modules:
                raise OpenODDValidationError(
                    f"module ID {module_id!r} is not globally unique"
                )
            modules[str(module_id)] = cast("Mapping[str, object]", value)
            if collection_name in {"ODD", "TOD"}:
                roots.add(str(module_id))
    return modules, roots


def _valid_numeric_expression(value: object) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return True
    if not isinstance(value, str):
        return False
    return bool(
        _BOUND_EXPRESSION.fullmatch(value)
        or _EQUAL_NUMERIC_EXPRESSION.fullmatch(value)
        or _RANGE_EXPRESSION.fullmatch(value)
    )


def _validate_condition(  # ruff:ignore[too-many-arguments]
    name: str,
    value: object,
    *,
    path: str,
    primitive: Mapping[str, str],
    categorical: Mapping[str, tuple[object, ...]],
    known_references: set[str],
) -> str | None:
    """Validate one condition and return a referenced module, if any.

    Returns:
        str | None: Referenced module or label ID, if applicable.

    Raises:
        OpenODDValidationError: If the condition is invalid.
    """
    if name not in known_references:
        raise OpenODDValidationError(
            f"{path}: condition references unknown concept, module, or label {name!r}"
        )
    primitive_type = primitive.get(name)
    if primitive_type in {"float", "integer"}:
        if not _valid_numeric_expression(value):
            raise OpenODDValidationError(
                f"{path}: numeric concept {name!r} requires an OpenODD "
                "numeric expression"
            )
        if (
            primitive_type == "integer"
            and isinstance(value, float)
            and not value.is_integer()
        ):
            raise OpenODDValidationError(
                f"{path}: integer concept {name!r} cannot equal {value!r}"
            )
        return None
    if primitive_type == "boolean":
        if not isinstance(value, bool):
            raise OpenODDValidationError(
                f"{path}: boolean concept {name!r} requires true or false"
            )
        return None
    domain = categorical.get(name)
    if domain is not None:
        observed = value if isinstance(value, list) else [value]
        unknown = [item for item in observed if item not in domain]
        if unknown:
            raise OpenODDValidationError(
                f"{path}: categorical concept {name!r} uses unknown "
                f"literal(s) {unknown!r}"
            )
        return None
    return name if isinstance(value, bool) else None


def _walk_conditions(
    section: Mapping[str, object],
    *,
    path: str,
    primitive: Mapping[str, str],
    categorical: Mapping[str, tuple[object, ...]],
    known_references: set[str],
) -> set[str]:
    references: set[str] = set()
    for name, value in section.items():
        child_path = f"{path}.{name}"
        if name == "METADATA":
            continue
        if name in {"AND", "OR"}:
            if not isinstance(value, Mapping):  # covered by JSON Schema
                raise OpenODDValidationError(f"{child_path}: must be a mapping")
            references.update(
                _walk_conditions(
                    cast("Mapping[str, object]", value),
                    path=child_path,
                    primitive=primitive,
                    categorical=categorical,
                    known_references=known_references,
                )
            )
            continue
        reference = _validate_condition(
            name,
            value,
            path=child_path,
            primitive=primitive,
            categorical=categorical,
            known_references=known_references,
        )
        if reference is not None:
            references.add(reference)
    return references


def _validate_module_graph(
    graph: Mapping[str, set[str]],
    roots: set[str],
) -> None:
    for module_id, dependencies in graph.items():
        invalid_roots = dependencies & roots
        if invalid_roots:
            raise OpenODDValidationError(
                f"module {module_id!r} depends on root module(s) "
                f"{sorted(invalid_roots)!r}"
            )

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(module_id: str) -> None:
        if module_id in visiting:
            raise OpenODDValidationError(
                f"module dependency cycle contains {module_id!r}"
            )
        if module_id in visited:
            return
        visiting.add(module_id)
        for dependency in graph.get(module_id, set()):
            if dependency in graph:
                visit(dependency)
        visiting.remove(module_id)
        visited.add(module_id)

    for module_id in graph:
        visit(module_id)


def validate_openodd_document(document: object) -> None:
    """Validate decoded YAML against the ASAM OpenODD 1.0.0 mapping.

    Structural validation is performed with
    :data:`OPENODD_1_0_YAML_SCHEMA`.  Additional checks cover global
    identifiers, typed expressions, references, root dependencies, and
    module dependency cycles.

    Args:
        document (object): YAML-decoded document.

    Raises:
        OpenODDValidationError: If the document is invalid.
    """
    _schema_validate(document)
    mapping = cast("Mapping[str, object]", document)
    taxonomy = mapping.get("TAXONOMY")
    taxonomy_mapping = (
        cast("Mapping[str, object]", taxonomy) if isinstance(taxonomy, Mapping) else {}
    )
    primitive, categorical, concept_names = _taxonomy_concepts(taxonomy_mapping)
    modules, roots = _module_collections(mapping)
    labels: set[str] = set()
    for module in modules.values():
        labels.update(_labels(module))

    duplicate_names = (set(modules) & concept_names) | (labels & set(modules))
    duplicate_names |= labels & concept_names
    if duplicate_names:
        raise OpenODDValidationError(
            "taxonomy concepts, modules, and labels must have distinct IDs; "
            f"duplicates: {sorted(duplicate_names)!r}"
        )

    known_references = concept_names | set(modules) | labels
    graph: dict[str, set[str]] = {}
    for module_id, module in modules.items():
        if set(module) - _MODULE_FIELDS:  # covered by schema, defensive
            raise OpenODDValidationError(f"module {module_id!r} has unknown fields")
        dependencies: set[str] = set()
        for section_name in _SECTION_NAMES:
            section = module.get(section_name)
            if isinstance(section, Mapping):
                dependencies.update(
                    _walk_conditions(
                        cast("Mapping[str, object]", section),
                        path=f"$.{module_id}.{section_name}",
                        primitive=primitive,
                        categorical=categorical,
                        known_references=known_references,
                    )
                )
        graph[module_id] = dependencies & set(modules)
    _validate_module_graph(graph, roots)


def parse_openodd_yaml(text: str) -> dict[str, Any]:
    """Parse and validate an ASAM OpenODD 1.0.0 YAML document.

    Args:
        text (str): YAML source text.

    Returns:
        dict[str, Any]: Validated YAML mapping.

    Raises:
        OpenODDValidationError: If parsing or validation fails.
    """
    try:
        document = yaml.load(
            text,
            Loader=_UniqueKeyLoader,  # ruff:ignore[unsafe-yaml-load]
        )
    except OpenODDValidationError:
        raise
    except yaml.YAMLError as error:
        raise OpenODDValidationError(f"invalid YAML: {error}") from error
    validate_openodd_document(document)
    return cast("dict[str, Any]", document)


def load_openodd_yaml(path: pathlib.Path | str) -> dict[str, Any]:
    """Load and validate an ASAM OpenODD 1.0.0 YAML file.

    Returns:
        dict[str, Any]: Validated YAML mapping.
    """
    source = pathlib.Path(path).read_text(encoding="utf-8")
    return parse_openodd_yaml(source)


def dump_openodd_yaml(document: Mapping[str, object]) -> str:
    """Validate and serialize an ASAM OpenODD 1.0.0 YAML document.

    Returns:
        str: YAML source.
    """
    validate_openodd_document(document)
    return yaml.safe_dump(
        dict(document),
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False,
    )


__all__ = [
    "OPENODD_1_0_YAML_SCHEMA",
    "OPENODD_VERSION",
    "OpenODDValidationError",
    "dump_openodd_yaml",
    "load_openodd_yaml",
    "parse_openodd_yaml",
    "validate_openodd_document",
]
