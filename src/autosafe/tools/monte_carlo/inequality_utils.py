# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""Utilities for handling inequality constraints in Monte Carlo ODD."""

from pathlib import Path

import numpy as np
import polytope as pc
import yaml

from autosafe.exceptions import (
    EmptyParameterError,
    InputFileNotFoundError,
    LengthMismatchError,
    MissingConfigFieldError,
    ParameterTypeError,
    PolytopeCreationError,
    UnknownOptionError,
)


def load_yaml_odd_config(config_path: str | Path) -> dict:
    """Load ODD configuration from YAML file.

    Args:
        config_path (str | Path): Path to YAML configuration file

    Returns:
        dict: Dictionary with ODD configuration

    Raises:
        InputFileNotFoundError: If config file doesn't exist
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise InputFileNotFoundError(config_path)

    with Path(config_path).open(encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Allow legacy dataset YAML that use a top-level `limits` mapping
    if isinstance(config, dict) and "type" not in config:
        return config

    return validate_odd_config(config)


def validate_odd_config(config: dict) -> dict:
    """Validate ODD configuration.

    Args:
        config (dict): ODD configuration dictionary

    Returns:
        dict: Validated configuration

    Raises:
        MissingConfigFieldError: If a required field is missing.
        UnknownOptionError: If the ODD type is unknown.
        ParameterTypeError: If the constraints are not a list.
        EmptyParameterError: If no constraint is given.
    """
    for required in ("type", "dim"):
        if required not in config:
            raise MissingConfigFieldError("ODD", (required,))

    config_type = config["type"]
    valid_types = ["box", "polytope"]

    if config_type not in valid_types:
        raise UnknownOptionError("type", config_type, valid_types)

    if config_type == "polytope":
        if "constraints" not in config:
            raise MissingConfigFieldError("polytope", ("constraints",))

        constraints = config["constraints"]
        if not isinstance(constraints, list):
            raise ParameterTypeError("constraints", constraints, list)

        if len(constraints) == 0:
            raise EmptyParameterError("constraints")

    elif config_type == "box":
        if "lower_bounds" not in config or "upper_bounds" not in config:
            raise MissingConfigFieldError("box", ("lower_bounds", "upper_bounds"))

    return config


def create_polytope_from_constraints(
    dim: int, constraints: list[dict]
) -> tuple[pc.Region, list[str]]:
    """Create a polytope from inequality constraints.

    Args:
        dim (int): Dimension of the space
        constraints (list[dict]): List of constraint dictionaries

    Returns:
        tuple[pc.Region, list[str]]: Polytope Region, list of constraint
        strings

    Raises:
        PolytopeCreationError: If the constraints do not define a valid
            polytope.
    """
    if not constraints:
        # Create a full-dimensional unit box as default
        return create_default_unit_box(dim), []

    constraint_matrices: list[list[float]] = []
    constraint_vectors: list[float] = []
    constraint_strings: list[str] = []

    for constraint in constraints:
        coefficients, _relation, bound, description = _normalize_constraint(
            constraint,
            dim,
        )
        constraint_matrices.append(coefficients)
        constraint_vectors.append(bound)
        constraint_strings.append(description)

    # Create polytope region
    if not constraint_matrices:
        return create_default_unit_box(dim), constraint_strings

    a_matrix = np.array(constraint_matrices)
    b_vector = np.array(constraint_vectors)

    try:
        polytope = pc.Region([pc.Polytope(a_matrix, b_vector)])
    except (ValueError, TypeError) as err:
        raise PolytopeCreationError from err
    else:
        return polytope, constraint_strings


def _normalize_constraint(
    constraint: dict,
    dim: int,
) -> tuple[list[float], str, float, str]:
    """Normalize one constraint into the canonical half-space form.

    Args:
        constraint (dict): Constraint mapping with ``coefficients``,
            ``relation`` and ``bound``.
        dim (int): Dimension of the space.

    Returns:
        tuple[list[float], str, float, str]: coefficients, relation,
            bound, and description.

    Raises:
        ParameterTypeError: If the constraint is not a dictionary.
        UnknownOptionError: If the constraint type or relation is
            unsupported.
        MissingConfigFieldError: If a constraint field is missing.
        LengthMismatchError: If the coefficients do not match the
            dimension.
    """
    if not isinstance(constraint, dict):
        raise ParameterTypeError("constraint", constraint, dict)

    constraint_type = constraint.get("type", "linear")
    if constraint_type != "linear":
        raise UnknownOptionError("type", constraint_type, ("linear",))

    try:
        coefficients = list(constraint["coefficients"])
        relation = constraint["relation"]
        bound = float(constraint["bound"])
    except KeyError as err:
        raise MissingConfigFieldError(
            "constraint", ("coefficients", "relation", "bound")
        ) from err

    if len(coefficients) != dim:
        raise LengthMismatchError("coefficients", len(coefficients), dim)

    # Keep the constraint as written, for the description: negating
    # the coefficients and bound while keeping the original relation
    # symbol names the opposite half-space.
    written_coefficients = list(coefficients)
    written_bound = bound

    if relation in {">=", ">"}:
        coefficients = [-c for c in coefficients]
        bound = -bound
    elif relation not in {"<=", "<"}:
        raise UnknownOptionError("relation", relation, (">=", ">", "<=", "<"))

    description = (
        "("
        + " ".join(f"{written_coefficients[i]:g}x{i + 1}" for i in range(dim))
        + f") {relation} {written_bound:g}"
    )
    return coefficients, relation, bound, description


def create_default_unit_box(dim: int) -> pc.Region:
    """Create a default unit box for fallback.

    Args:
        dim (int): Space dimension.

    Returns:
        pc.Region: Unit-box polytope region.
    """
    a_matrix = np.vstack((np.eye(dim), -np.eye(dim)))
    b_vector = np.ones(2 * dim)
    return pc.Region([pc.Polytope(a_matrix, b_vector)])


def convert_to_positive_form(
    coefficients: list[float], relation: str, bound: float
) -> tuple[list[float], str, float]:
    """Convert any constraint to the form a*x <= b.

    Args:
        coefficients (list[float]): Constraint coefficients.
        relation (str): Constraint relation.
        bound (float): Constraint bound.

    Returns:
        tuple[list[float], str, float]: Canonical coefficients,
            relation, and bound.

    Raises:
        UnknownOptionError: If the relation is unsupported.
    """
    if relation in {">=", ">"}:
        # x1 >= x2 + 4 becomes -x1 + x2 + 4 <= 0
        new_coefficients = [-c for c in coefficients]
        new_bound = -bound
        new_relation = "<="
    elif relation in {"<=", "<", "==", "="}:
        new_coefficients = coefficients
        new_bound = bound
        new_relation = "<="
    else:
        raise UnknownOptionError(
            "relation", relation, (">=", ">", "<=", "<", "==", "=")
        )

    return new_coefficients, new_relation, new_bound


class ODDFactory:
    """Factory for creating different types of ODD regions."""

    def __init__(self, config: dict) -> None:
        self.config = config
        self.dim = config["dim"]

    def create_odd(self) -> tuple[pc.Region, str]:
        """Create ODD region from configuration.

        Returns:
            tuple[pc.Region, str]: Polytope Region, string description
            of constraints.

        Raises:
            UnknownOptionError: If the configuration type is
                unsupported.
        """
        odd_type = self.config["type"]

        if odd_type == "box":
            lower_bounds = self.config["lower_bounds"]
            upper_bounds = self.config["upper_bounds"]

            if isinstance(lower_bounds, (int, float)):
                lower_bounds = [lower_bounds] * self.dim
            if isinstance(upper_bounds, (int, float)):
                upper_bounds = [upper_bounds] * self.dim

            a_matrix = np.vstack((np.eye(self.dim), -np.eye(self.dim)))
            b_vector = np.concatenate((upper_bounds, [-lb for lb in lower_bounds]))

            region = pc.Region([pc.Polytope(a_matrix, b_vector)])
            description = f"Box: {lower_bounds} <= x <= {upper_bounds}"

        elif odd_type == "polytope":
            polytope, constraint_strings = create_polytope_from_constraints(
                self.dim, self.config["constraints"]
            )
            description = "Polytope with constraints: " + ", ".join(constraint_strings)
            region = polytope

        else:
            raise UnknownOptionError("type", odd_type, ("box", "polytope"))

        return region, description
