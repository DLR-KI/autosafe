# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""Tests for the continuous ``score`` of the ODD boundary methods."""

from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from autosafe.exceptions import ScoreNotSupportedError
from autosafe.odd.comparison import (
    ClusteredSuperlevelSetMonitor,
    GaussianMixtureBoundary,
    KMeansBoundaries,
    KNNMonitor,
    ODDBoundaryMethod,
    OneClassSVMBoundary,
    SuperlevelSetMonitor,
    SVDDBoundary,
)

#: Each method with the membership rule its score is documented to follow.
Rule = Callable[[Any, npt.NDArray[np.float64]], npt.NDArray[np.bool_]]


def _reference_and_tests() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(8)
    reference = rng.normal(size=(3, 120))
    test = rng.normal(size=(3, 200)) * 2.0
    return reference, test


CASES: list[tuple[str, Callable[[], ODDBoundaryMethod], Rule]] = [
    (
        "kde",
        SuperlevelSetMonitor,
        lambda m, s: s > np.log(m.gamma),
    ),
    (
        "gmm",
        lambda: GaussianMixtureBoundary(random_state=0),
        lambda m, s: s > np.log(m.gamma),
    ),
    (
        "knn",
        KNNMonitor,
        lambda m, s: s > -m.gamma,
    ),
    (
        "ocsvm",
        lambda: OneClassSVMBoundary(gamma=0.5, nu=0.1),
        lambda m, s: s >= m.threshold,
    ),
    (
        "svdd",
        lambda: SVDDBoundary(kernel="poly", nu=0.1),
        lambda m, s: s >= m.threshold,
    ),
    (
        "clustered_kde",
        lambda: ClusteredSuperlevelSetMonitor(n_clusters=2),
        lambda _m, s: s > 0.0,
    ),
]


@pytest.mark.parametrize(("name", "factory", "rule"), CASES, ids=[c[0] for c in CASES])
def test_score_reproduces_membership(
    name: str, factory: Callable[[], ODDBoundaryMethod], rule: Rule
) -> None:
    reference, test = _reference_and_tests()
    method = factory().fit(reference)

    scores = method.score(test)

    assert scores.shape == (test.shape[1],), name
    assert scores.dtype == np.float64
    inside = method.evaluate_batch(test)
    assert inside.any()
    assert not inside.all()
    np.testing.assert_array_equal(rule(method, scores), inside)


def test_density_score_is_log_pdf() -> None:
    reference, test = _reference_and_tests()
    monitor = SuperlevelSetMonitor().fit(reference)

    np.testing.assert_array_equal(np.exp(monitor.score(test)), monitor.pdf(test))


def test_methods_without_score_raise() -> None:
    reference, test = _reference_and_tests()
    method = KMeansBoundaries(n_clusters=2).fit(reference)

    with pytest.raises(ScoreNotSupportedError):
        method.score(test)
