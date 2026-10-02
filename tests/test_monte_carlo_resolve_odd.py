# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""Regression tests for ``_resolve_odd``'s custom-polytope merge.

The merge branch was unreachable for the whole life of the feature:
``MonteCarloConfig`` is a ``TypedDict``, so at runtime it is a plain
``dict``, and ``hasattr(a_dict, "custom_odd_config")`` is always False --
dicts do not expose their keys as attributes. Every ``type: polytope``
ODD therefore fell through to the box branch and its inequality
constraints were silently discarded.

Nothing covered ``_resolve_odd``, which is why it survived. These tests
pin the merge so it cannot regress to box-only again.
"""

from pathlib import Path
from typing import Any, cast

import numpy as np
import polytope as pc
import pytest

from autosafe.tools.monte_carlo.sample import _resolve_odd

_HALF_SPACE = {
    "type": "polytope",
    "dim": 2,
    # Encodes the half-space "x1 minus x2 is at least 4".
    "constraints": [
        {
            "type": "linear",
            "coefficients": [1.0, -1.0],
            "relation": ">=",
            "bound": 4.0,
        }
    ],
}


def _box(limit: float = 10.0) -> pc.Region:
    """Build the square ``[-limit, limit]^2`` as a polytope region.

    Args:
        limit (float): Half-width of the square.

    Returns:
        pc.Region: The axis-aligned box region.
    """
    bounds = np.array([[-limit, limit], [-limit, limit]])
    return pc.Region([pc.box2poly(bounds)])


def _config(**overrides: Any) -> Any:  # ruff:ignore[any-type]
    """Build a minimal Monte Carlo config mapping.

    Args:
        overrides (Any): Keys to override on the base mapping.

    Returns:
        Any: The config mapping accepted by ``_resolve_odd``.
    """
    config: dict[str, Any] = {"odd_type": "box", "custom_odd_config": None}
    config.update(overrides)
    return cast("Any", config)


def test_resolve_odd_applies_polytope_constraints():
    """A polytope config must add its half-space to the base box.

    This is the regression: the merged region gains one row over the
    four box rows, and points on the wrong side of ``x1 - x2 >= 4`` fall
    outside the ODD. Before the fix the branch never ran, so the result
    was the bare box and every such point was wrongly counted inside.
    """
    base = _box()
    assert len(base.list_poly[0].A) == 4

    odd, description = _resolve_odd(_config(custom_odd_config=_HALF_SPACE), base)

    assert len(odd.list_poly[0].A) == 5, "half-space was dropped"
    assert description is not None
    assert "(1x1 -1x2) >= 4" in description
    # x1 - x2 = 5 >= 4 -> inside; x1 - x2 = -5 -> outside.
    assert np.array([5.0, 0.0]) in odd
    assert np.array([0.0, 5.0]) not in odd


def test_resolve_odd_without_custom_config_returns_base_box():
    """No custom config leaves the base region and description untouched."""
    base = _box()
    odd, description = _resolve_odd(_config(), base)

    assert odd is base
    assert description is None


def test_resolve_odd_accepts_yaml_path(tmp_path: Path):
    """``custom_odd_config`` may be a path to a YAML file.

    Args:
        tmp_path (Path): Pytest temporary directory fixture.
    """
    import yaml

    config_path = tmp_path / "odd.yaml"
    config_path.write_text(yaml.safe_dump(_HALF_SPACE), encoding="utf-8")

    odd, description = _resolve_odd(_config(custom_odd_config=str(config_path)), _box())

    assert len(odd.list_poly[0].A) == 5
    assert description is not None


def test_resolve_odd_rejects_unsupported_custom_config_type():
    """Anything but a YAML path or a mapping is rejected explicitly."""
    with pytest.raises(ValueError, match="YAML path or a mapping"):
        _resolve_odd(_config(custom_odd_config=42), _box())
