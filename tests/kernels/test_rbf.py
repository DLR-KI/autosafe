# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT


import copy
import math
import re
import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
from _pytest.mark.structures import ParameterSet

from autosafe.exceptions import (
    ArrayDtypeError,
    KernelParametersNotSetError,
    MissingParameterError,
    NonPositiveParameterError,
    NotPositiveSemidefiniteError,
    ParameterOrderError,
    ParameterTypeError,
    ShapeMismatchError,
    SigmaNotInvertibleError,
    UnknownOptionError,
)
from autosafe.kernels import rbf
from autosafe.kernels.rbf import (
    SIGMA_FLOOR_RATIO,
    GaussianKernel,
    RBFKernel,
    _invert_sigma,
    _validate_and_broadcast_param,
    diagonal_widths,
    rbf_diagonal_sigma,
)
from autosafe.typing import FloatType


class TestRBFKernel:
    """Test suite for the RBF kernel."""

    kernel = RBFKernel(
        x_i=np.array([1.0, 2.0, 3.0], dtype=FloatType),
        sigma=np.eye(3, dtype=FloatType),
    )

    def test_rbf_kernel_same_vector(self):
        kernel = copy.deepcopy(self.kernel)
        assert np.allclose(kernel(kernel.x_i), 1.0), (
            "RBF kernel should return 1.0 for the same vector."
        )

    def test_rbf_kernel_different_vector(self):
        kernel = copy.deepcopy(self.kernel)
        assert kernel(kernel.x_i + kernel.x_i) < 1.0, (
            "RBF kernel should return a value less than 1.0 for different vectors."
        )
        assert kernel(-kernel.x_i) < 1.0, (
            "RBF kernel should return a value less than 1.0 for different vectors."
        )

    @staticmethod
    def test_rbf_kernel_init_fails_on_wrong_sigma_shape():
        with pytest.raises(ShapeMismatchError, match="sigma has shape"):
            RBFKernel(
                x_i=np.array([1.0, 2.0, 3.0], dtype=FloatType),
                sigma=np.eye(2, dtype=FloatType),
            )

    @staticmethod
    def test_rbf_kernel_init_fails_on_non_psd_sigma():
        with pytest.raises(NotPositiveSemidefiniteError):
            RBFKernel(
                x_i=np.array([1.0, 2.0], dtype=FloatType),
                sigma=np.array([[-1.0, 0.0], [0.0, -1.0]], dtype=FloatType),
            )
        with pytest.raises(NotPositiveSemidefiniteError):
            RBFKernel(
                x_i=np.array([1.0, 2.0], dtype=FloatType),
                sigma=np.array([[1.0, 0.0], [0.0, 0.0]], dtype=FloatType),
            )

    @staticmethod
    def test_rbf_kernel_init_fails_on_non_psd():
        with pytest.raises(NotPositiveSemidefiniteError):
            RBFKernel(
                x_i=np.array([1.0, 2.0], dtype=FloatType),
                sigma=-1.0,
            )

    @staticmethod
    def test_rbf_kernel_init_fails_wrong_sigma_type():
        with pytest.raises(UnknownOptionError, match="unknown sigma "):
            RBFKernel(
                x_i=np.array([1.0, 2.0], dtype=FloatType),
                sigma="identity",  # ty: ignore[invalid-argument-type]
            )

    @staticmethod
    def test_rbf_kernel_init_accepts_scalar_sigma():
        kernel = RBFKernel(
            x_i=np.array([1.0, 2.0], dtype=FloatType),
            sigma=1.2,
        )
        expected_sigma = 1.2 * np.eye(2, dtype=FloatType)
        assert kernel.sigma is not None
        assert np.allclose(kernel.sigma, expected_sigma)
        assert kernel.sigma_inv is not None
        assert np.allclose(kernel.sigma_inv, np.linalg.inv(expected_sigma))

    @staticmethod
    def test_rbf_kernel_init_sigma_eye():
        kernel = RBFKernel(x_i=np.array([1.0, 2.0, 3.0], dtype=FloatType), sigma="eye")
        assert kernel.sigma is not None
        assert kernel.sigma_inv is not None
        assert np.array_equal(kernel.sigma, np.eye(3, dtype=FloatType)), (
            "Sigma should be the identity matrix."
        )
        assert np.array_equal(kernel.sigma_inv, np.eye(3, dtype=FloatType)), (
            "Sigma inverse should also be the identity matrix."
        )

    @staticmethod
    def test_rbf_kernel_call_fails_with_no_sigma():
        kernel = RBFKernel(x_i=np.array([1.0, 2.0, 3.0], dtype=FloatType))
        with pytest.raises(
            KernelParametersNotSetError,
            match="kernel parameter sigma has not been set",
        ):
            kernel(np.array([1.0, 2.0, 3.0], dtype=FloatType))

    @staticmethod
    def test_rbf_kernel_init_sigma_defaults_to_none():
        kernel = RBFKernel(x_i=np.array([1.0, 2.0, 3.0], dtype=FloatType))
        assert kernel.sigma is None, "Default sigma should be None."
        assert kernel.sigma_inv is None, "Default sigma inverse should also be None."

    def test_rbf_kernel_sigma_ii_x_nn_not_array(self):
        kernel = copy.deepcopy(self.kernel)
        with pytest.raises(ParameterTypeError, match="x_nn must be of type"):
            kernel._sigma_ii(x_nn=3)  # ty: ignore[invalid-argument-type]

    def test_rbf_kernel_sigma_ii_x_nn_wrong_dtype(self):
        kernel = copy.deepcopy(self.kernel)
        with pytest.raises(ArrayDtypeError, match="x_nn must have dtype"):
            kernel._sigma_ii(x_nn=np.array([1.0, 2.0, 3.0], dtype=np.float32))  # ty: ignore[invalid-argument-type]

    def test_rbf_kernel_sigma_ii_x_nn_shape_mismatch(self):
        kernel = copy.deepcopy(self.kernel)
        dim = kernel.x_i.shape[0]
        with pytest.raises(ShapeMismatchError, match="x_nn has shape"):
            kernel._sigma_ii(x_nn=np.ones((dim + 1, dim), dtype=FloatType))

    def test_rbf_kernel_sigma_ii_success_vector(self):
        kernel = copy.deepcopy(self.kernel)
        dim = kernel.x_i.shape[0]
        x_nn = np.ones((dim,), dtype=FloatType)
        kernel._sigma_ii(x_nn=x_nn)

        assert kernel.sigma is not None, "Sigma should be set."
        assert kernel.sigma_inv is not None, "Sigma inverse should be set."
        assert kernel.sigma.shape == (
            dim,
            dim,
        ), "Sigma should have the correct shape."

    @staticmethod
    def test_rbf_kernel_set_sigma_via_kappa_eta_xnn():
        dim = 3
        x_nn = np.ones((dim,), dtype=FloatType)
        kernel1 = RBFKernel(
            x_i=np.ones((dim,), dtype=FloatType),
            x_nn=x_nn,
            kappa=1.0,
            eta=1.0,
        )
        kernel2 = RBFKernel(
            x_i=np.ones((dim,), dtype=FloatType),
            x_nn=x_nn,
            kappa=2.0,
            eta=1.0,
        )
        kernel3 = RBFKernel(
            x_i=np.array([1.0, 2.0, 3.0], dtype=FloatType),
            x_nn=x_nn,
            kappa=1.0,
            eta=1.0,
        )

        assert kernel1.sigma is not None, "Sigma should be set."
        assert kernel1.sigma_inv is not None, "Sigma inverse should be set."
        assert kernel1.sigma.shape == (
            dim,
            dim,
        ), "Sigma should have the correct shape."
        assert np.allclose(kernel1.sigma, np.eye(dim, dtype=FloatType)), (
            "Sigma should be the identity matrix."
        )
        assert np.allclose(kernel1.sigma_inv, np.eye(dim, dtype=FloatType)), (
            "Sigma inverse should also be the identity matrix."
        )
        assert kernel2.sigma is not None, "Sigma should be set."
        assert kernel2.sigma_inv is not None, "Sigma inverse should be set."
        assert kernel2.sigma.shape == (
            dim,
            dim,
        ), "Sigma should have the correct shape."
        assert np.allclose(kernel2.sigma, 2 * np.eye(dim, dtype=FloatType)), (
            "Sigma should be the scaled identity matrix."
        )
        assert kernel3.sigma is not None, "Sigma should be set."
        assert kernel3.sigma_inv is not None, "Sigma inverse should be set."
        assert kernel3.sigma.shape == (
            dim,
            dim,
        ), "Sigma should have the correct shape."
        assert not np.allclose(kernel3.sigma, np.eye(dim, dtype=FloatType)), (
            "Sigma should not be the identity matrix."
        )
        assert not np.allclose(kernel3.sigma_inv, np.eye(dim, dtype=FloatType)), (
            "Sigma inverse should not be the identity matrix."
        )

    def test_rbf_kernel_sigma_ii_is_eye_like_for_x_nn_kappa_eta_scalar_equals_x_i(self):
        kernel = copy.deepcopy(self.kernel)
        dim = kernel.x_i.shape[0]
        x_nn = copy.deepcopy(kernel.x_i)

        kappas = [0.5, 1.0, 2.0]
        etas = [0.1, 1.0, 10.0]

        for kappa, eta in zip(kappas, etas, strict=False):
            kernel.update(x_nn=x_nn, kappa=kappa, eta=eta)
            expected_sigma = kappa * np.eye(dim, dtype=FloatType)
            expected_sigma_inv = np.linalg.inv(expected_sigma)
            assert kernel.sigma is not None, "Sigma should be set."
            assert kernel.sigma_inv is not None, "Sigma inverse should be set."
            assert kernel.sigma.shape == (
                dim,
                dim,
            ), "Sigma should have the correct shape."
            assert kernel.sigma_inv.shape == (
                dim,
                dim,
            ), "Sigma inverse should have the correct shape."
            assert np.allclose(kernel.sigma, expected_sigma), (
                "Sigma should be kappa * np.eye."
            )
            assert np.allclose(kernel.sigma_inv, expected_sigma_inv), (
                "Sigma inverse should be the inverse of kappa * np.eye."
            )

    def test_rbf_kernel_sigma_ii_is_scaled_for_x_nn_kappa_eta_scalar_far_away(self):
        kernel = copy.deepcopy(self.kernel)
        dim = kernel.x_i.shape[0]
        x_nn = 10 * np.ones((dim,), dtype=FloatType)

        kappas = [0.5, 1.0, 2.0]
        etas = [0.1, 1.0, 10.0]

        for kappa, eta in zip(kappas, etas, strict=False):
            kernel.update(x_nn=x_nn, kappa=kappa, eta=eta)
            dist = np.abs(kernel.x_i - x_nn)
            lam_val = SIGMA_FLOOR_RATIO * kappa
            expected_sigma = np.diag((kappa - lam_val) * np.exp(-eta * dist) + lam_val)
            expected_sigma_inv = np.linalg.inv(expected_sigma)
            assert kernel.sigma is not None, "Sigma should be set."
            assert kernel.sigma_inv is not None, "Sigma inverse should be set."
            assert kernel.sigma.shape == (
                dim,
                dim,
            ), "Sigma should have the correct shape."
            assert kernel.sigma_inv.shape == (
                dim,
                dim,
            ), "Sigma inverse should have the correct shape."
            assert np.allclose(kernel.sigma, expected_sigma), (
                "Sigma should be ((kappa-lam)*exp(-eta*|x_i-x_nn|)+lam)*np.eye."
            )
            assert np.all(np.diag(kernel.sigma) < kappa), (
                "All diagonal elements of sigma should be less than kappa."
            )
            assert np.allclose(kernel.sigma_inv, expected_sigma_inv), (
                "Sigma inverse should be the inverse of sigma."
            )

    def test_rbf_kernel_sigma_ii_is_eye_like_for_x_nn_kappa_eta_vector_equals_x_i(self):
        kernel = copy.deepcopy(self.kernel)
        dim = kernel.x_i.shape[0]
        x_nn = copy.deepcopy(kernel.x_i)

        ks = [0.5, 1.0, 10.0]  # negative kappa is invalid with the affine-floor law
        for k in ks:
            kappas = k * np.array([0.5, 1.0, 2.0], dtype=FloatType)
            etas = k * np.array([0.1, 1.0, 10.0], dtype=FloatType)

            kernel.update(x_nn=x_nn, kappa=kappas, eta=etas)
            expected_sigma = np.diag(kappas)
            expected_sigma_inv = np.linalg.inv(expected_sigma)
            assert kernel.sigma is not None, "Sigma should be set."
            assert kernel.sigma_inv is not None, "Sigma inverse should be set."
            assert kernel.sigma.shape == (
                dim,
                dim,
            ), "Sigma should have the correct shape."
            assert kernel.sigma_inv.shape == (
                dim,
                dim,
            ), "Sigma inverse should have the correct shape."
            assert np.allclose(kernel.sigma, expected_sigma), (
                "Sigma should be kappa * np.eye."
            )
            assert np.allclose(kernel.sigma_inv, expected_sigma_inv), (
                "Sigma inverse should be the inverse of kappa * np.eye."
            )

    def test_rbf_kernel_sigma_ii_is_scaled_for_x_nn_kappa_eta_vector_far_away(self):
        kernel = copy.deepcopy(self.kernel)
        dim = kernel.x_i.shape[0]
        x_nn = 10 * np.ones((dim,), dtype=FloatType)

        ks = [0.5, 1.0, 10.0]  # negative kappa is invalid with the affine-floor law
        for k in ks:
            kappas = k * np.array([0.5, 1.0, 2.0], dtype=FloatType)
            etas = k * np.array([0.1, 1.0, 10.0], dtype=FloatType)

            kernel.update(x_nn=x_nn, kappa=kappas, eta=etas)
            dist = np.abs(kernel.x_i - x_nn)
            lam_vals = SIGMA_FLOOR_RATIO * kappas
            expected_sigma = np.diag(
                (kappas - lam_vals) * np.exp(-etas * dist) + lam_vals
            )
            expected_sigma_inv = np.linalg.inv(expected_sigma)
            assert kernel.sigma is not None, "Sigma should be set."
            assert kernel.sigma_inv is not None, "Sigma inverse should be set."
            assert kernel.sigma.shape == (
                dim,
                dim,
            ), "Sigma should have the correct shape."
            assert kernel.sigma_inv.shape == (
                dim,
                dim,
            ), "Sigma inverse should have the correct shape."
            assert np.allclose(kernel.sigma, expected_sigma), (
                "Sigma should be ((kappa-lam)*exp(-eta*|x_i-x_nn|)+lam)*np.eye."
            )
            assert np.all(np.diag(kernel.sigma) <= kappas), (
                "All diagonal elements of sigma should be <= kappa."
            )
            assert np.allclose(kernel.sigma_inv, expected_sigma_inv), (
                "Sigma inverse should be the inverse of sigma."
            )

    def test_rbf_kernel_sigma_ii_success_matrix(self):
        kernel = copy.deepcopy(self.kernel)
        dim = kernel.x_i.shape[0]
        x_nn = np.ones((dim, dim), dtype=FloatType)
        kernel._sigma_ii(x_nn=x_nn)

        assert kernel.sigma is not None, "Sigma should be set."
        assert kernel.sigma_inv is not None, "Sigma inverse should be set."
        assert kernel.sigma.shape == (
            dim,
            dim,
        ), "Sigma should have the correct shape."

    def test_rbf_kernel_compare_kernels(self):
        kernel1 = copy.deepcopy(self.kernel)
        kernel2 = RBFKernel(
            x_i=np.array([1.0, 2.0, 3.0], dtype=FloatType),
            sigma=np.eye(3, dtype=FloatType),
        )
        kernel3 = RBFKernel(
            x_i=np.array([3.0, 2.0, 1.0], dtype=FloatType),
            sigma=np.eye(3, dtype=FloatType),
        )
        kernel4 = RBFKernel(
            x_i=np.array([1.0, 2.0, 3.0, 4.0], dtype=FloatType),
            sigma=np.eye(4, dtype=FloatType),
        )
        kernel5 = RBFKernel(
            x_i=np.array([1.0, 2.0, 3.0], dtype=FloatType),
            sigma=2 * np.eye(3, dtype=FloatType),
        )
        kernel6 = RBFKernel(
            x_i=np.array([1.0, 2.0, 3.0], dtype=FloatType),
        )
        kernel7 = RBFKernel(
            x_i=np.array([1.0, 2.0, 3.0], dtype=FloatType),
        )
        kernel8 = RBFKernel(
            x_i=np.array([1.0, 2.0, 3.0, 4.0], dtype=FloatType),
            sigma=np.eye(4, dtype=FloatType),
        )
        kernel8.x_i = kernel1.x_i  # Create impossible kernel
        kernel9 = "kernel"

        assert kernel1 == kernel2, "Kernels with the same parameters should be equal."
        assert kernel1 != kernel3, (
            "Kernels with different parameters should not be equal."
        )
        assert kernel1 != kernel4, (
            "Kernels with different dimensions should not be equal."
        )
        assert kernel1 != kernel5, (
            "Kernels with different sigma values should not be equal."
        )
        assert kernel6 == kernel7, "Kernels with default sigma should be equal."
        assert kernel2 not in {
            kernel6,
            kernel3,
        }, "Kernels with one undefined sigma should not be equal."
        assert kernel1 != kernel8, (
            "Kernels with different dimensions should not be equal."
        )
        assert kernel1 != kernel9, "Kernel should not be equal to a non-kernel object."

        # Test for hashing consistency
        assert kernel1 in {kernel1, kernel2}, (
            "Kernels with the same parameters should be equal."
        )
        assert kernel1 not in {
            kernel3,
            kernel4,
            kernel5,
            kernel6,
            kernel7,
            kernel8,
            kernel9,
        }, "Kernels with different parameters should not be equal."

    def test_rbf_kernel_set_sigma_sigma_not_array(self):
        kernel = copy.deepcopy(self.kernel)
        with pytest.raises(NotPositiveSemidefiniteError):
            kernel.update(sigma=-0.1)

    def test_rbf_kernel_set_sigma_scalar_success(self):
        kernel = copy.deepcopy(self.kernel)
        kernel.update(sigma=0.8)
        expected_sigma = 0.8 * np.eye(3, dtype=FloatType)
        assert kernel.sigma is not None
        assert np.allclose(kernel.sigma, expected_sigma)
        assert kernel.sigma_inv is not None
        assert np.allclose(kernel.sigma_inv, np.linalg.inv(expected_sigma))

    def test_rbf_kernel_set_sigma_sigma_wrong_dtype(self):
        kernel = copy.deepcopy(self.kernel)
        with pytest.warns(
            UserWarning,
            match=re.escape(
                "sigma is not of dtype FloatType. Converting to FloatType."
            ),
        ):
            kernel.update(sigma=np.eye(3, dtype=np.float32))  # ty: ignore[invalid-argument-type]

    def test_rbf_kernel_set_sigma_success_eye(self):
        kernel = copy.deepcopy(self.kernel)
        dim = kernel.x_i.shape[0]
        kernel.update(sigma="eye")
        assert kernel.sigma is not None, "Sigma should be set."
        assert kernel.sigma_inv is not None, "Sigma inverse should be set."
        assert kernel.sigma.shape == (
            dim,
            dim,
        ), "Sigma should have the correct shape."
        assert np.array_equal(kernel.sigma, np.eye(dim, dtype=FloatType)), (
            "Sigma should be the identity matrix."
        )

    def test_rbf_kernel_set_sigma_sigma_wrong_dims(self):
        kernel = copy.deepcopy(self.kernel)
        dim = kernel.x_i.shape[0]
        with pytest.raises(ShapeMismatchError, match="sigma has shape"):
            kernel.update(sigma=np.eye(dim + 1, dtype=FloatType))

    @staticmethod
    def test_rbf_kernel_set_sigma_sigma_not_psd():
        kernel = RBFKernel(
            x_i=np.array([1.0, 2.0], dtype=FloatType),
            sigma=None,
        )
        with pytest.raises(NotPositiveSemidefiniteError):
            kernel.update(sigma=np.array([[-1.0, 0.0], [0.0, -1.0]], dtype=FloatType))

        kernel = RBFKernel(
            x_i=np.array([1.0, 2.0], dtype=FloatType),
            sigma=None,
        )
        with pytest.raises(NotPositiveSemidefiniteError):
            kernel.update(sigma=np.array([[1.0, 0.0], [0.0, 0.0]], dtype=FloatType))

    def test_rbf_kernel_set_sigma_x_nn_bot_not_defined(self):
        kernel = copy.deepcopy(self.kernel)
        with pytest.raises(
            MissingParameterError, match="x_nn is required unless sigma is given"
        ):
            kernel.update()

    @pytest.mark.parametrize(
        ("error", "match", "bad_argument"),
        [
            pytest.param(
                UnknownOptionError,
                "unknown sigma ",
                {"sigma": "not_a_valid_string"},
                id="sigma-unknown-string",
            ),
            pytest.param(
                ParameterTypeError,
                "x_nn must be of type",
                {"x_nn": 3},
                id="x_nn-not-array",
            ),
            pytest.param(
                ParameterTypeError,
                "sigma must be of type",
                {"sigma": [[1, 2], [3, 4]]},
                id="sigma-not-array",
            ),
        ],
    )
    def test_rbf_kernel_update_rejects_bad_argument(
        self,
        error: type[Exception],
        match: str,
        bad_argument: dict[str, object],
    ) -> None:
        kernel = copy.deepcopy(self.kernel)
        with pytest.raises(error, match=match):
            kernel.update(**bad_argument)  # ty: ignore[invalid-argument-type]

    def test_rbf_kernel_set_sigma_x_nn_wrong_dtype(self):
        kernel = copy.deepcopy(self.kernel)
        with pytest.warns(
            UserWarning,
            match=re.escape("x_nn is not of dtype FloatType. Converting to FloatType."),
        ):
            kernel.update(x_nn=np.array([1.0, 2.0, 3.0], dtype=np.float32))  # ty: ignore[invalid-argument-type]

    def test_rbf_kernel_set_sigma_x_nn_shape_mismatch(self):
        kernel = copy.deepcopy(self.kernel)
        dim = kernel.x_i.shape[0]
        with pytest.raises(ShapeMismatchError, match="x_nn has shape"):
            kernel.update(x_nn=np.ones((dim + 1, dim), dtype=FloatType))

    def test_rbf_kernel_set_sigma_x_nn_success(self):
        kernel = copy.deepcopy(self.kernel)
        dim = kernel.x_i.shape[0]
        x_nn = np.ones((dim, dim), dtype=FloatType)
        kernel.update(x_nn=x_nn)
        assert kernel.sigma is not None, "Sigma should be set."
        assert kernel.sigma_inv is not None, "Sigma inverse should be set."
        assert kernel.sigma.shape == (
            dim,
            dim,
        ), "Sigma should have the correct shape."

    @staticmethod
    def test_rbf_kernel_gaussian_kernel_alias():
        kernel1 = RBFKernel(
            x_i=np.array([1.0, 2.0, 3.0], dtype=FloatType),
            sigma=np.eye(3, dtype=FloatType),
        )
        kernel2 = GaussianKernel(
            x_i=np.array([1.0, 2.0, 3.0], dtype=FloatType),
            sigma=np.eye(3, dtype=FloatType),
        )
        assert kernel1 == kernel2, "RBFKernel and GaussianKernel should be equal."

    def test_str_contains_all_needed_info(self):
        kernel = copy.deepcopy(self.kernel)
        kernel_str = str(kernel)
        assert "RBFKernel" in kernel_str, (
            "String representation should contain class name."
        )
        assert "x_i" in kernel_str, "String representation should contain x_i."
        assert str(kernel.x_i) in kernel_str, (
            "String representation should contain x_i value."
        )
        assert "sigma" in kernel_str, "String representation should contain sigma."
        assert str(kernel.sigma) in kernel_str, (
            "String representation should contain sigma value."
        )

    def test_repr_contains_all_needed_info(self):
        kernel = copy.deepcopy(self.kernel)
        kernel_repr = repr(kernel)
        assert "RBFKernel" in kernel_repr, (
            "Repr representation should contain class name."
        )
        assert "x_i" in kernel_repr, "Repr representation should contain x_i."
        assert repr(kernel.x_i) in kernel_repr, (
            "Repr representation should contain x_i value."
        )
        assert "sigma" in kernel_repr, "Repr representation should contain sigma."
        assert repr(kernel.sigma) in kernel_repr, (
            "Repr representation should contain sigma value."
        )

    def test_kernel_can_be_recreated_from_repr(self):
        from numpy import array  # ruff:ignore[unused-import]

        kernel = copy.deepcopy(self.kernel)
        kernel_repr = repr(kernel)
        recreated_kernel = eval(kernel_repr)
        assert kernel == recreated_kernel, (
            "Kernel recreated from repr should be equal to the original."
        )


def test_validate_and_broadcast_param_success_int():
    param = 3
    name = "test_param"
    dim = 5
    result = _validate_and_broadcast_param(param, name, dim)
    expected = np.array([3, 3, 3, 3, 3], dtype=FloatType)
    assert np.array_equal(result, expected), (
        "Integer parameter should be broadcasted correctly."
    )


def test_validate_and_broadcast_param_success_float():
    param = math.pi
    name = "test_param"
    dim = 4
    result = _validate_and_broadcast_param(param, name, dim)
    expected = np.array([math.pi, math.pi, math.pi, math.pi], dtype=FloatType)
    assert np.array_equal(result, expected), (
        "Float parameter should be broadcasted correctly."
    )


@pytest.mark.parametrize(
    "param",
    [
        pytest.param(FloatType(math.e), id="floattype"),
        pytest.param(math.e, id="float-cast-to-floattype"),
    ],
)
def test_validate_and_broadcast_param_floattype(param: float) -> None:
    name = "test_param"
    dim = 6
    result = _validate_and_broadcast_param(param, name, dim)
    expected = np.array(
        [math.e, math.e, math.e, math.e, math.e, math.e],
        dtype=FloatType,
    )
    assert result.dtype == expected.dtype, "Result should have FloatType dtype."
    assert np.array_equal(result, expected), (
        "FloatType parameter should be broadcasted correctly."
    )


def test_validate_and_broadcast_param_success_vector():
    param = np.array([math.e, math.pi, -1], dtype=FloatType)
    name = "test_param"
    dim = 3
    result = _validate_and_broadcast_param(param, name, dim)
    expected = np.array([math.e, math.pi, -1], dtype=FloatType)
    assert np.array_equal(result, expected), (
        "Vector parameter should be returned correctly."
    )


def test_validate_and_broadcast_param_success_vector_with_dtype_casting():
    param = np.array([math.e, math.pi, -1], dtype=np.float32)
    name = "test_param"
    dim = 3
    result = _validate_and_broadcast_param(param, name, dim)  # ty: ignore[invalid-argument-type]
    expected = np.array([math.e, math.pi, -1], dtype=FloatType)
    assert np.allclose(result, expected), (
        "Vector parameter should be returned correctly."
    )


def test_validate_and_broadcast_param_shape_mismatch():
    param = np.array([1.0, 2.0, 3.0], dtype=FloatType)
    name = "test_param"
    dim = 6
    with pytest.raises(ShapeMismatchError, match="test_param has shape"):
        _validate_and_broadcast_param(param, name, dim)


def test_validate_and_broadcast_param_wrong_type():
    param = "three"
    name = "test_param"
    dim = 6
    with pytest.raises(ParameterTypeError, match="test_param must be of type"):
        _validate_and_broadcast_param(param, name, dim)  # ty: ignore[invalid-argument-type]


def test_fix_sigma_matrix():
    # A subnormal eigenvalue passes the psd check, but its reciprocal
    # overflows, so the inverse is not finite and sigma gets repaired.
    kernel = RBFKernel(x_i=np.zeros(2, dtype=FloatType))
    with pytest.warns(UserWarning, match="sigma matrix was not invertible"):
        kernel.update(sigma=np.diag([1e-320, 1.0]).astype(FloatType))
    assert kernel.sigma is not None
    assert kernel.sigma_inv is not None
    eps = np.finfo(FloatType).eps
    np.testing.assert_allclose(kernel.sigma, np.diag([eps, 1.0]))
    assert np.isfinite(kernel.sigma_inv).all()
    np.testing.assert_allclose(kernel.sigma @ kernel.sigma_inv, np.eye(2), atol=1e-12)


def test_fix_sigma_matrix_in_constructor():
    with pytest.warns(UserWarning, match="sigma matrix was not invertible"):
        kernel = RBFKernel(
            x_i=np.zeros(2, dtype=FloatType),
            sigma=np.diag([1e-320, 1.0]).astype(FloatType),
        )
    assert kernel.sigma_inv is not None
    assert np.isfinite(kernel.sigma_inv).all()


def _failing_repair(sigma: np.ndarray) -> np.ndarray:
    """Stand-in for ``_fix_sigma_matrix`` whose result is still singular.

    Whether the real eigenvalue repair succeeds on an all-subnormal sigma
    depends on the CPU (``jnp.linalg.eigh`` returns NaN on some machines
    and finite values on others), so the failure path is forced here.

    Args:
        sigma (np.ndarray): Sigma matrix to "repair".

    Returns:
        np.ndarray: A zero matrix of the same shape.
    """
    return np.zeros_like(sigma)


_SINGULAR_SIGMA = np.diag([1e-320, 1.0]).astype(FloatType)


def test_unrepairable_sigma_update_raises_and_keeps_sigma(
    monkeypatch: pytest.MonkeyPatch,
):
    kernel = RBFKernel(x_i=np.zeros(2, dtype=FloatType), sigma="eye")
    monkeypatch.setattr(rbf, "_fix_sigma_matrix", _failing_repair)
    with pytest.raises(SigmaNotInvertibleError, match="could not be repaired"):
        kernel.update(sigma=_SINGULAR_SIGMA)
    np.testing.assert_array_equal(kernel.sigma, np.eye(2))
    np.testing.assert_array_equal(kernel.sigma_inv, np.eye(2))


def test_unrepairable_sigma_in_constructor_raises(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(rbf, "_fix_sigma_matrix", _failing_repair)
    with pytest.raises(SigmaNotInvertibleError, match="could not be repaired"):
        RBFKernel(x_i=np.zeros(2, dtype=FloatType), sigma=_SINGULAR_SIGMA)


@pytest.mark.filterwarnings("ignore:sigma matrix was not invertible")
def test_subnormal_scalar_sigma_never_yields_a_nan_kernel():
    """Repaired to a finite kernel or rejected, depending on the CPU."""
    kernel = RBFKernel(x_i=np.zeros(2, dtype=FloatType), sigma="eye")
    try:
        kernel.update(sigma=1e-320)
    except SigmaNotInvertibleError:
        np.testing.assert_array_equal(kernel.sigma, np.eye(2))
    assert kernel.sigma is not None
    assert kernel.sigma_inv is not None
    assert np.isfinite(kernel.sigma).all()
    assert np.isfinite(kernel.sigma_inv).all()


_SIGMA_NON_DIAG = np.array(
    [[2.0, 0.3, 0.0], [0.3, 2.0, 0.0], [0.0, 0.0, 2.0]], dtype=FloatType
)
_X_I_3 = np.array([1.0, 2.0, 3.0], dtype=FloatType)


def test_rbf_call_single_point_non_diagonal_sigma():
    kernel = RBFKernel(x_i=_X_I_3, sigma=_SIGMA_NON_DIAG)
    assert np.isclose(float(kernel(kernel.x_i)), 1.0)
    assert float(kernel(kernel.x_i + 1.0)) < 1.0


def test_rbf_call_matrix_input_diagonal_sigma():
    kernel = RBFKernel(x_i=_X_I_3, sigma=np.eye(3, dtype=FloatType))
    pts = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=FloatType)
    result = kernel(pts)
    assert result.shape == (2,)
    assert np.isclose(float(result[0]), 1.0)
    assert float(result[1]) < 1.0


def test_rbf_call_matrix_input_non_diagonal_sigma():
    kernel = RBFKernel(x_i=_X_I_3, sigma=_SIGMA_NON_DIAG)
    pts = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=FloatType)
    result = kernel(pts)
    assert result.shape == (2,)
    assert np.isclose(float(result[0]), 1.0)
    assert float(result[1]) < 1.0


def test_refresh_sigma_cache_raises_when_sigma_is_none():
    import copy

    kernel = copy.deepcopy(RBFKernel(x_i=_X_I_3, sigma=np.eye(3, dtype=FloatType)))
    kernel.sigma = None
    with pytest.raises(
        KernelParametersNotSetError,
        match="kernel parameter sigma has not been set",
    ):
        kernel._refresh_sigma_cache()


def test_get_sigma_inv_diag_jax_raises_when_sigma_inv_none():
    import copy

    kernel = copy.deepcopy(RBFKernel(x_i=_X_I_3, sigma=np.eye(3, dtype=FloatType)))
    kernel.sigma_inv = None
    kernel._sigma_inv_diag_jax = None
    with pytest.raises(
        KernelParametersNotSetError,
        match="kernel parameter sigma_inv has not been set",
    ):
        kernel._get_sigma_inv_diag_jax()


def test_get_sigma_inv_full_jax_raises_when_sigma_inv_none():
    import copy

    kernel = copy.deepcopy(RBFKernel(x_i=_X_I_3, sigma=_SIGMA_NON_DIAG))
    kernel.sigma_inv = None
    kernel._sigma_inv_full_jax = None
    with pytest.raises(
        KernelParametersNotSetError,
        match="kernel parameter sigma_inv has not been set",
    ):
        kernel._get_sigma_inv_full_jax()


def _reference_widths(
    anchors: npt.NDArray[np.float64],
    neighbors: npt.NDArray[np.float64],
    kappa: float | npt.NDArray[np.float64],
    eta: float | npt.NDArray[np.float64],
    lam: float | npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    return (kappa - lam) * np.exp(-eta * np.abs(neighbors - anchors)) + lam


_KAPPA, _ETA, _LAM = 0.7, 1.3, 0.7e-4


def _random_anchor_neighbor(
    n_points: int, dim: int, seed: int
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n_points, dim)), rng.normal(size=(n_points, dim))


def _widths_cases() -> list[ParameterSet]:
    anchors, neighbors = _random_anchor_neighbor(200, 7, 0)
    return [
        pytest.param(anchors, neighbors, id="random"),
        pytest.param(anchors, anchors.copy(), id="zero-displacement"),
        pytest.param(anchors, anchors + 1e3, id="underflow"),
    ]


@pytest.mark.parametrize(("anchors", "neighbors"), _widths_cases())
def test_diagonal_widths_matches_formula(
    anchors: npt.NDArray[np.float64], neighbors: npt.NDArray[np.float64]
) -> None:
    widths = diagonal_widths(anchors, neighbors, kappa=_KAPPA, eta=_ETA, lam=_LAM)
    assert widths.dtype == np.float64
    assert widths.shape == anchors.shape
    assert np.array_equal(
        widths, _reference_widths(anchors, neighbors, _KAPPA, _ETA, _LAM)
    )


def test_diagonal_widths_per_feature_parameters():
    anchors, neighbors = _random_anchor_neighbor(30, 5, 1)
    dim = anchors.shape[1]
    kappa = np.full(dim, 0.5)
    eta = np.linspace(0.5, 2, dim)
    widths = diagonal_widths(anchors, neighbors, kappa=kappa, eta=eta, lam=None)
    expected = _reference_widths(
        anchors, neighbors, kappa, eta, SIGMA_FLOOR_RATIO * kappa
    )
    assert np.array_equal(widths, expected)


def test_diagonal_widths_validates():
    anchors, neighbors = _random_anchor_neighbor(10, 4, 2)
    with pytest.raises(NonPositiveParameterError):
        diagonal_widths(anchors, neighbors, kappa=1.0, eta=1.0, lam=0.0)
    with pytest.raises(ParameterOrderError):
        diagonal_widths(anchors, neighbors, kappa=0.5, eta=1.0, lam=0.5)
    with pytest.raises(ShapeMismatchError):
        diagonal_widths(anchors, neighbors[:, :3])
    with pytest.raises(ShapeMismatchError):
        diagonal_widths(anchors, neighbors, kappa=np.ones(3))


def test_kernel_update_uses_the_batched_widths():
    anchors, neighbors = _random_anchor_neighbor(50, 6, 3)
    expected = diagonal_widths(anchors, neighbors, kappa=_KAPPA, eta=_ETA, lam=_LAM)
    for i, widths in enumerate(expected):
        a = np.asarray(anchors[i], dtype=np.float64).reshape(-1)
        n = np.asarray(neighbors[i], dtype=np.float64).reshape(-1)
        k = RBFKernel(x_i=a, kappa=_KAPPA, eta=_ETA, lam=_LAM)
        k.update(x_nn=n, kappa=_KAPPA, eta=_ETA, lam=_LAM)
        assert k.sigma is not None
        assert k.sigma_inv is not None
        assert np.array_equal(np.diag(k.sigma), widths)
        assert np.array_equal(k.sigma_inv, np.diag(1.0 / np.diag(k.sigma)))
        assert k.sigma_is_diagonal
        off = ~np.eye(6, dtype=bool)
        assert not np.any(k.sigma[off])
        assert not np.any(k.sigma_inv[off])


def test_per_dimension_update_uses_the_batched_widths():
    rng = np.random.default_rng(4)
    dim = 6
    x_i = rng.normal(size=dim)
    x_nn = rng.normal(size=(dim, dim))
    k = RBFKernel(x_i=x_i, kappa=_KAPPA, eta=_ETA, lam=_LAM)
    k.update(x_nn=x_nn)
    expected = _reference_widths(x_i, np.diag(x_nn), _KAPPA, _ETA, _LAM)
    assert np.array_equal(np.diag(k.sigma), expected)


def _outcome(
    call: Callable[[], tuple[Any, Any]],
) -> tuple[type[Exception] | None, tuple[Any, Any] | None, bool]:
    """Run ``call`` and report how it ended.

    Args:
        call (Callable[[], tuple[Any, Any]]): The call to run.

    Returns:
        tuple[type[Exception] | None, tuple[Any, Any] | None, bool]:
            The sigma error class raised (or None), the result, and
            whether the repair warning was issued.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            result = call()
        except SigmaNotInvertibleError as error:
            return type(error), None, False
    repaired = any(
        issubclass(w.category, UserWarning)
        and "sigma matrix was not invertible" in str(w.message)
        for w in caught
    )
    return None, result, repaired


def test_subnormal_width_falls_back_to_repair() -> None:
    # 1 / width overflows, so the dense repair path runs. Whether the
    # repair succeeds or fails depends on the JAX backend, so the
    # kernel must match the repair applied to the same widths.
    kappa, eta, lam = 1e-309, 1.0, 1e-310
    x_i, x_nn = np.zeros(3), np.full(3, 1e3)
    widths = diagonal_widths(x_i[None], x_nn[None], kappa=kappa, eta=eta, lam=lam)[0]
    error, reference, ref_repaired = _outcome(lambda: _invert_sigma(np.diag(widths)))

    k = RBFKernel(x_i=x_i, kappa=kappa, eta=eta, lam=lam)

    def update() -> tuple[object, object]:
        k.update(x_nn=x_nn, kappa=kappa, eta=eta, lam=lam)
        return k.sigma, k.sigma_inv

    k_error, result, k_repaired = _outcome(update)
    assert k_error is error
    assert k_repaired == ref_repaired
    if error is None:
        assert reference is not None
        assert result is not None
        assert np.array_equal(result[0], reference[0])
        assert np.array_equal(result[1], reference[1])


def test_rbf_diagonal_sigma_overflow_raises_custom_error() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(SigmaNotInvertibleError):
            rbf_diagonal_sigma(
                np.zeros((2, 3)),
                np.full((2, 3), 1e3),
                kappa=1e-309,
                eta=1.0,
                lam=1e-310,
            )


def test_per_dimension_column_vector_neighbor():
    rng = np.random.default_rng(5)
    dim = 4
    x_i = rng.normal(size=dim)
    x_nn = rng.normal(size=(dim, 1))
    k = RBFKernel(x_i=x_i, kappa=_KAPPA, eta=_ETA, lam=_LAM)
    k.update(x_nn=x_nn)
    expected = _reference_widths(x_i, x_nn[:, 0], _KAPPA, _ETA, _LAM)
    assert np.array_equal(np.diag(k.sigma), expected)
