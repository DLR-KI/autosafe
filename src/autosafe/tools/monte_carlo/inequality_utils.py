# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""Utilities for handling inequality constraints in Monte Carlo ODD."""

from pathlib import Path

import numpy as np
import polytope as pc
import yaml


def load_yaml_odd_config(config_path: str | Path) -> dict:
    """Load ODD configuration from YAML file.

    Args:
        config_path (str | Path): Path to YAML configuration file

    Returns:
        dict: Dictionary with ODD configuration

    Raises:
        FileNotFoundError: If config file doesn't exist
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

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
        ValueError: If configuration is invalid
    """
    if "type" not in config:
        raise ValueError("Configuration must specify 'type' field")

    if "dim" not in config:
        raise ValueError("Configuration must specify 'dim' field")

    config_type = config["type"]
    valid_types = ["box", "polytope"]

    if config_type not in valid_types:
        raise ValueError(
            f"Invalid ODD type: {config_type}. Must be one of: {valid_types}"
        )

    if config_type == "polytope":
        if "constraints" not in config:
            raise ValueError("Polytope configuration must specify 'constraints' field")

        constraints = config["constraints"]
        if not isinstance(constraints, list):
            raise ValueError("Constraints must be a list")

        if len(constraints) == 0:
            raise ValueError("At least one constraint must be specified")

    elif config_type == "box":
        if "lower_bounds" not in config or "upper_bounds" not in config:
            raise ValueError(
                "Box configuration must specify 'lower_bounds' and 'upper_bounds'"
            )

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
        ValueError: If constraints are invalid
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
        return polytope, constraint_strings
    except (ValueError, TypeError) as err:
        raise ValueError("Failed to create polytope from constraints") from err


def _normalize_constraint(
    constraint: dict,
    dim: int,
) -> tuple[list[float], str, float, str]:
    """Normalize one constraint into the canonical half-space form.

    Returns:
        tuple[list[float], str, float, str]: coefficients, relation,
            bound, and description.

    Raises:
        ValueError: If the constraint is invalid.
    """
    if not isinstance(constraint, dict):
        raise ValueError(f"Each constraint must be a dict, got {type(constraint)}")

    constraint_type = constraint.get("type", "linear")
    if constraint_type != "linear":
        raise ValueError(f"Unsupported constraint type: {constraint_type}")

    try:
        coefficients = list(constraint["coefficients"])
        relation = constraint["relation"]
        bound = float(constraint["bound"])
    except KeyError as err:
        raise ValueError(
            "Linear constraint must have 'coefficients', 'relation', and "
            "'bound' fields",
        ) from err

    if len(coefficients) != dim:
        raise ValueError(
            f"Coefficient dimension {len(coefficients)} does not match "
            f"space dimension {dim}",
        )

    if relation in {">=", ">"}:
        coefficients = [-c for c in coefficients]
        bound = -bound
    elif relation not in {"<=", "<"}:
        raise ValueError(f"Unsupported relation: {relation}")

    description = (
        "("
        + " ".join(f"{coefficients[i]:g}x{i + 1}" for i in range(dim))
        + f") {relation} {bound:g}"
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
        ValueError: If the relation is unsupported.
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
        raise ValueError(f"Unsupported relation: {relation}")

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
            ValueError: If the configuration type is unsupported.
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
            raise ValueError(f"Unknown ODD type: {odd_type}")

        return region, description
