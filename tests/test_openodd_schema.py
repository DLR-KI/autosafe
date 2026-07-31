# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""Exhaustive structural and semantic OpenODD validator tests."""

import copy
import pathlib
from typing import cast

import pytest

import autosafe.odd.openodd_schema as schema
from autosafe.odd import (
    OpenODDValidationError,
    dump_openodd_yaml,
    load_openodd_yaml,
    parse_openodd_yaml,
    validate_openodd_document,
)


def _document() -> dict[str, object]:
    return {
        "TAXONOMY": {
            "conditions": {
                "speed": "float velocity",
                "count": "integer count",
                "enabled": "boolean",
                "weather": ["sun", "rain"],
            },
        },
        "ODD": {
            "root": {
                "TITLE": "Root ODD",
                "ACTIVE": True,
                "INCLUDE_OR": {"region": True},
            },
        },
        "MODULES": {
            "region": {
                "TITLE": "Operational region",
                "ACTIVE": True,
                "INCLUDE_AND": {
                    "speed": "[0 .. 10] km/h",
                    "count": 2,
                    "enabled": True,
                    "weather": ["sun"],
                },
            },
        },
    }


def _modules(
    document: dict[str, object],
    collection: str,
) -> dict[str, dict[str, object]]:
    return cast("dict[str, dict[str, object]]", document[collection])


def test_validator_accepts_nested_conditions_metadata_and_labels() -> None:
    document = _document()
    root = _modules(document, "ODD")["root"]
    root["LABEL"] = "root_label"
    region = _modules(document, "MODULES")["region"]
    region["LABELS"] = ["region_label"]
    region["INCLUDE_AND"] = {
        "METADATA": {"source": "test"},
        "AND": {"speed": ">= 0 km/h", "root_label": True},
    }

    validate_openodd_document(document)


def test_schema_errors_include_array_paths() -> None:
    with pytest.raises(OpenODDValidationError, match=r"\$\.IMPORT\[0\]"):
        validate_openodd_document({"IMPORT": [""]})


def test_module_ids_must_be_globally_unique() -> None:
    document = _document()
    modules = _modules(document, "MODULES")
    modules["root"] = modules.pop("region")
    _modules(document, "ODD")["root"]["INCLUDE_OR"] = {"root": True}

    with pytest.raises(OpenODDValidationError, match="globally unique"):
        validate_openodd_document(document)


@pytest.mark.parametrize(
    ("name", "value", "match"),
    [
        ("speed", True, "numeric concept"),
        ("speed", {"minimum": 0}, "numeric concept"),
        ("count", 1.5, "integer concept"),
        ("enabled", "true", "boolean concept"),
        ("weather", "snow", "unknown literal"),
    ],
)
def test_typed_condition_errors(name: str, value: object, match: str) -> None:
    document = _document()
    _modules(document, "MODULES")["region"]["INCLUDE_AND"] = {name: value}

    if isinstance(value, dict):
        assert schema._valid_numeric_expression(value) is False
        return
    with pytest.raises(OpenODDValidationError, match=match):
        validate_openodd_document(document)


def test_condition_namespaces_must_be_distinct() -> None:
    document = _document()
    modules = _modules(document, "MODULES")
    modules["speed"] = modules.pop("region")
    _modules(document, "ODD")["root"]["INCLUDE_OR"] = {"speed": True}

    with pytest.raises(OpenODDValidationError, match="distinct IDs"):
        validate_openodd_document(document)


def test_modules_cannot_depend_on_roots() -> None:
    document = _document()
    _modules(document, "MODULES")["region"]["INCLUDE_AND"] = {"root": True}

    with pytest.raises(OpenODDValidationError, match="depends on root"):
        validate_openodd_document(document)


def test_module_dependency_cycles_are_rejected() -> None:
    document = _document()
    _modules(document, "ODD")["root"]["INCLUDE_OR"] = {"first": True}
    document["MODULES"] = {
        "first": {
            "TITLE": "First",
            "ACTIVE": True,
            "INCLUDE_OR": {"second": True},
        },
        "second": {
            "TITLE": "Second",
            "ACTIVE": True,
            "INCLUDE_OR": {"first": True},
        },
    }

    with pytest.raises(OpenODDValidationError, match="dependency cycle"):
        validate_openodd_document(document)


def test_defensive_semantic_guards(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(OpenODDValidationError, match="must be a mapping"):
        schema._walk_conditions(
            {"AND": True},
            path="$.test",
            primitive={},
            categorical={},
            known_references={"AND"},
        )

    document = _document()
    _modules(document, "MODULES")["region"]["UNKNOWN"] = True
    monkeypatch.setattr(schema, "_schema_validate", lambda _document: None)
    with pytest.raises(OpenODDValidationError, match="unknown fields"):
        validate_openodd_document(document)


def test_invalid_yaml_is_wrapped() -> None:
    with pytest.raises(OpenODDValidationError, match="invalid YAML"):
        parse_openodd_yaml("TAXONOMY: [")


def test_load_and_dump_round_trip(tmp_path: pathlib.Path) -> None:
    source = dump_openodd_yaml(_document())
    path = tmp_path / "odd.yaml"
    path.write_text(source, encoding="utf-8")

    assert load_openodd_yaml(path) == _document()
    assert parse_openodd_yaml(source) == _document()


def test_valid_numeric_expression_forms() -> None:
    assert schema._valid_numeric_expression(1)
    assert schema._valid_numeric_expression(">= 2 m/s")
    assert schema._valid_numeric_expression("2 * scale m/s")
    assert schema._valid_numeric_expression("(0 .. 2] m/s")


def test_validation_does_not_mutate_document() -> None:
    document = _document()
    original = copy.deepcopy(document)

    validate_openodd_document(document)

    assert document == original
