# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""Registration tests for the gmm/oneclass_svm/svdd comparison methods.

Mirrors tests/test_evaluate_comparison.py's create_comparison_monitor
coverage, scoped to the three new R2 baseline methods.
"""

import warnings

import numpy as np
import pytest

from autosafe.odd.comparison.mixture import GaussianMixtureBoundary
from autosafe.odd.comparison.oneclass import OneClassSVMBoundary, SVDDBoundary
from autosafe.tools.evaluate.comparison import (
    create_comparison_monitor,
    get_available_method_names,
    validate_method_names,
)


def test_get_available_method_names_includes_new_baselines():
    names = get_available_method_names()
    assert "gmm" in names
    assert "oneclass_svm" in names
    assert "svdd" in names


def test_validate_method_names_accepts_new_baselines():
    validate_method_names(["gmm", "oneclass_svm", "svdd"])


def test_create_comparison_monitor_gmm_default():
    monitor = create_comparison_monitor("gmm")
    assert isinstance(monitor, GaussianMixtureBoundary)
    assert monitor.random_state == 0
    assert monitor.n_components_range == (1, 10)


def test_create_comparison_monitor_gmm_with_params():
    monitor = create_comparison_monitor(
        "gmm", random_state=5, n_components_range=(1, 3)
    )
    assert isinstance(monitor, GaussianMixtureBoundary)
    assert monitor.random_state == 5
    assert monitor.n_components_range == (1, 3)


def test_create_comparison_monitor_oneclass_svm_default():
    monitor = create_comparison_monitor("oneclass_svm")
    assert isinstance(monitor, OneClassSVMBoundary)
    assert monitor.gamma == "scale"
    assert monitor.nu == pytest.approx(0.05)
    assert monitor.auto_select is False


def test_create_comparison_monitor_oneclass_svm_with_params():
    monitor = create_comparison_monitor("oneclass_svm", nu=0.2, gamma=1.0)
    assert isinstance(monitor, OneClassSVMBoundary)
    assert monitor.nu == pytest.approx(0.2)
    assert monitor.gamma == pytest.approx(1.0)


def test_create_comparison_monitor_svdd_default_warns_rbf():
    with pytest.warns(UserWarning, match="equivalent"):
        monitor = create_comparison_monitor("svdd")
    assert isinstance(monitor, SVDDBoundary)
    assert monitor.kernel == "rbf"


def test_create_comparison_monitor_svdd_poly_no_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        monitor = create_comparison_monitor("svdd", kernel="poly", degree=2)
    assert isinstance(monitor, SVDDBoundary)
    assert monitor.kernel == "poly"
    assert monitor.degree == 2


def test_new_baselines_fit_and_evaluate_through_the_factory():
    rng = np.random.default_rng(0)
    ref = rng.normal(size=(2, 30))
    test_points = rng.normal(size=(2, 20))

    gmm = create_comparison_monitor("gmm", random_state=0, n_components_range=(1, 2))
    gmm.fit(ref)
    assert gmm.evaluate_batch(test_points).shape == (20,)

    ocsvm = create_comparison_monitor("oneclass_svm")
    ocsvm.fit(ref)
    assert ocsvm.evaluate_batch(test_points).shape == (20,)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        svdd = create_comparison_monitor("svdd")
    svdd.fit(ref)
    assert svdd.evaluate_batch(test_points).shape == (20,)
