# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT

import numpy as np
import pytest

from autosafe.tools.monte_carlo._sample import cast_to_array


def test_cast_to_array_from_scalar():
    """Test casting from scalar to numpy array."""
    dims = [1, 3, 20]
    values = [-1.5, 0.0, 1, 1.0, 5, 30.2]
    for dim in dims:
        for value in values:
            result = cast_to_array(dim=dim, value=value)
            expected = np.full((dim,), value)
            assert np.array_equal(result, expected)


def test_cast_to_array_from_sequence_with_one_item():
    """Test casting from single-item sequence to numpy array."""
    dims = [1, 3, 20]
    values = [-1.5, 0.0, 1, 1.0, 5, 30.2]
    for dim in dims:
        for value in values:
            result = cast_to_array(dim=dim, value=[value])
            expected = np.full((dim,), value)
            assert np.array_equal(result, expected)


def test_cast_to_array_from_sequence_with_multiple_items():
    """Test casting from multi-item sequence to numpy array."""
    test_cases = [
        (3, [1.0, 2.0, 3.0]),
        (4, [0.5, 1.5, 2.5, 3.5]),
        (5, [-1, 0, 1, 2, 3]),
    ]
    for dim, value in test_cases:
        result = cast_to_array(dim=dim, value=value)
        expected = np.array(value)
        assert np.array_equal(result, expected)


def test_cast_to_array_from_numpy_array():
    """Test casting from numpy array to numpy array."""
    test_cases = [
        (3, np.array([1.0, 2.0, 3.0])),
        (4, np.array([0.5, 1.5, 2.5, 3.5])),
        (5, np.array([-1, 0, 1, 2, 3])),
    ]
    for dim, value in test_cases:
        result = cast_to_array(dim=dim, value=value)
        expected = value
        assert np.array_equal(result, expected)


def test_cast_to_array_dimension_mismatch():
    """Test casting with dimension mismatch raises an error."""
    test_cases = [
        (3, [1.0, 2.0], "three"),
        (4, [0.5, 1.5, 2.5], "four"),
        (5, [-1, 0, 1, 2], "five"),
    ]
    for dim, value, name in test_cases:
        with pytest.raises(
            IndexError,
            match=f"Length of {name} must match dim if {name} is a list or array.",
        ):
            cast_to_array(dim=dim, value=value, name=name)
