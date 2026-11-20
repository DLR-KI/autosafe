# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""Tests for the membership agreement statistics."""

import numpy as np
import pytest

from autosafe.exceptions import (
    LengthMismatchError,
    ParameterOutOfRangeError,
    ShapeMismatchError,
)
from autosafe.tools.evaluate.metrics import membership_agreement, subset_agreement


def test_membership_agreement_counts_and_ratios() -> None:
    reference = np.array([True, True, True, False, False, True])
    other = np.array([True, True, False, True, False, False])

    agreement = membership_agreement(reference, other)

    assert (agreement.n_reference, agreement.n_other, agreement.n_both) == (4, 3, 2)
    assert agreement.precision == pytest.approx(2 / 3)
    assert agreement.recall == pytest.approx(2 / 4)
    assert agreement.iou == pytest.approx(2 / 5)


def test_empty_denominators_count_as_one() -> None:
    nothing = np.zeros(4, dtype=bool)
    something = np.array([True, False, False, False])

    both_empty = membership_agreement(nothing, nothing)
    assert (both_empty.precision, both_empty.recall, both_empty.iou) == (1.0, 1.0, 1.0)
    other_empty = membership_agreement(something, nothing)
    assert other_empty.precision == pytest.approx(1.0)
    assert other_empty.recall == pytest.approx(0.0)
    assert other_empty.iou == pytest.approx(0.0)


def test_subset_agreement_verdict() -> None:
    first = np.ones(1000, dtype=bool)
    second = first.copy()
    second[:4] = False

    close = subset_agreement(first, second, tolerance=0.005)
    assert close.agree
    assert close.passed
    assert close.first_precision == pytest.approx(0.996)
    assert close.second_precision == pytest.approx(1.0)
    assert close.iou == pytest.approx(0.996)
    assert close.first_vs_second == membership_agreement(second, first)
    assert close.second_vs_first == membership_agreement(first, second)
    assert close.first_vs_second.precision == close.first_precision
    assert close.second_vs_first.precision == close.second_precision
    assert close.second_vs_first.n_both == 996

    second[:6] = False
    far = subset_agreement(first, second, tolerance=0.005)
    assert not far.agree
    assert not far.passed
    assert far.tolerance == pytest.approx(0.005)


def test_agreement_validates_input() -> None:
    with pytest.raises(LengthMismatchError):
        membership_agreement(np.ones(3, dtype=bool), np.ones(4, dtype=bool))
    with pytest.raises(ShapeMismatchError):
        membership_agreement(np.ones((2, 2), dtype=bool), np.ones(4, dtype=bool))
    with pytest.raises(ParameterOutOfRangeError):
        subset_agreement(np.ones(2, dtype=bool), np.ones(2, dtype=bool), tolerance=1.5)
