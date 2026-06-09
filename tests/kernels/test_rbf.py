# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT


import copy
import math
import re

import numpy as np
import pytest

from autosafe.kernels.rbf import (
    GaussianKernel,
    RBFKernel,
    _validate_and_broadcast_param,
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
        with pytest.raises(
            ValueError, match=re.escape("sigma must have compatible dimensions to x_i.")
        ):
            RBFKernel(
                x_i=np.array([1.0, 2.0, 3.0], dtype=FloatType),
                sigma=np.eye(2, dtype=FloatType),
            )

    @staticmethod
    def test_rbf_kernel_init_fails_on_non_psd_sigma():
        with pytest.raises(
            ValueError, match=re.escape("sigma must be positive semidefinite (psd).")
        ):
            RBFKernel(
                x_i=np.array([1.0, 2.0], dtype=FloatType),
                sigma=np.array([[-1.0, 0.0], [0.0, -1.0]], dtype=FloatType),
            )
        with pytest.raises(
            ValueError, match=re.escape("sigma must be positive semidefinite (psd).")
        ):
            RBFKernel(
                x_i=np.array([1.0, 2.0], dtype=FloatType),
                sigma=np.array([[1.0, 0.0], [0.0, 0.0]], dtype=FloatType),
            )

    @staticmethod
    def test_rbf_kernel_init_fails_on_non_psd():
        with pytest.raises(
            ValueError, match=re.escape("sigma must be positive semidefinite (psd).")
        ):
            RBFKernel(
                x_i=np.array([1.0, 2.0], dtype=FloatType),
                sigma=-1.0,
            )

    @staticmethod
    def test_rbf_kernel_init_fails_wrong_sigma_type():
        with pytest.raises(
            ValueError,
            match=re.escape("sigma must be either 'eye', a square matrix, or None."),
        ):
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
            ValueError,
            match=re.escape("Kernel has no free parameter matrix (sigma) defined."),
        ):
            kernel(np.array([1.0, 2.0, 3.0], dtype=FloatType))

    @staticmethod
    def test_rbf_kernel_init_sigma_defaults_to_none():
        kernel = RBFKernel(x_i=np.array([1.0, 2.0, 3.0], dtype=FloatType))
        assert kernel.sigma is None, "Default sigma should be None."
        assert kernel.sigma_inv is None, "Default sigma inverse should also be None."

    def test_rbf_kernel_sigma_ii_x_nn_not_array(self):
        kernel = copy.deepcopy(self.kernel)
        with pytest.raises(ValueError, match=re.escape("x_nn must be a numpy array.")):
            kernel._sigma_ii(x_nn=3)  # ty: ignore[invalid-argument-type]

    def test_rbf_kernel_sigma_ii_x_nn_wrong_dtype(self):
        kernel = copy.deepcopy(self.kernel)
        with pytest.raises(
            ValueError, match=re.escape("x_nn must be of dtype FloatType.")
        ):
            kernel._sigma_ii(x_nn=np.array([1.0, 2.0, 3.0], dtype=np.float32))

    def test_rbf_kernel_sigma_ii_x_nn_shape_mismatch(self):
        kernel = copy.deepcopy(self.kernel)
        dim = kernel.x_i.shape[0]
        with pytest.raises(
            ValueError,
            match=re.escape("x_nn must have the same number of rows as x_i."),
        ):
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
            expected_sigma = np.diag(kappa * np.exp(-eta * dist))
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
                "Sigma should be (kappa * exp(-eta * |x_i - x_nn|)) * np.eye."
            )
            assert np.all(np.diag(kernel.sigma) < kappa), (
                "All diagonal elements of sigma should be less than or equal to kappa."
            )
            assert np.allclose(kernel.sigma_inv, expected_sigma_inv), (
                "Sigma inverse should be the inverse of sigma."
            )

    def test_rbf_kernel_sigma_ii_is_eye_like_for_x_nn_kappa_eta_vector_equals_x_i(self):
        kernel = copy.deepcopy(self.kernel)
        dim = kernel.x_i.shape[0]
        x_nn = copy.deepcopy(kernel.x_i)

        ks = [-1.0, 0.5, 1.0, 10.0]
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

        ks = [-1.0, 0.5, 1.0, 10.0]
        for k in ks:
            kappas = k * np.array([0.5, 1.0, 2.0], dtype=FloatType)
            etas = k * np.array([0.1, 1.0, 10.0], dtype=FloatType)

            kernel.update(x_nn=x_nn, kappa=kappas, eta=etas)
            dist = np.abs(kernel.x_i - x_nn)
            expected_sigma = np.diag(kappas * np.exp(-etas * dist))
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
                "Sigma should be (kappa * exp(-eta * |x_i - x_nn|)) * np.eye."
            )
            assert np.all(np.diag(kernel.sigma) <= kappas), (
                "All diagonal elements of sigma should be less than or equal to kappa."
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
        with pytest.raises(
            ValueError,
            match=re.escape("sigma must be positive semidefinite (psd)."),
        ):
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
            kernel.update(sigma=np.eye(3, dtype=np.float32))

    def test_rbf_kernel_set_sigma_sigma_not_valid_string(self):
        kernel = copy.deepcopy(self.kernel)
        with pytest.raises(
            ValueError,
            match=re.escape("sigma must be either 'eye', a square matrix, or None."),
        ):
            kernel.update(sigma="not_a_valid_string")  # ty: ignore[invalid-argument-type]

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
        with pytest.raises(
            ValueError, match=re.escape("sigma must have compatible dimensions to x_i.")
        ):
            kernel.update(sigma=np.eye(dim + 1, dtype=FloatType))

    @staticmethod
    def test_rbf_kernel_set_sigma_sigma_not_psd():
        kernel = RBFKernel(
            x_i=np.array([1.0, 2.0], dtype=FloatType),
            sigma=None,
        )
        with pytest.raises(
            ValueError, match=re.escape("sigma must be positive semidefinite (psd).")
        ):
            kernel.update(sigma=np.array([[-1.0, 0.0], [0.0, -1.0]], dtype=FloatType))

        kernel = RBFKernel(
            x_i=np.array([1.0, 2.0], dtype=FloatType),
            sigma=None,
        )
        with pytest.raises(
            ValueError, match=re.escape("sigma must be positive semidefinite (psd).")
        ):
            kernel.update(sigma=np.array([[1.0, 0.0], [0.0, 0.0]], dtype=FloatType))

    def test_rbf_kernel_set_sigma_x_nn_bot_not_defined(self):
        kernel = copy.deepcopy(self.kernel)
        with pytest.raises(
            ValueError, match=re.escape("x_nn must be provided if sigma is not set.")
        ):
            kernel.update()

    def test_rbf_kernel_set_sigma_x_nn_not_array(self):
        kernel = copy.deepcopy(self.kernel)
        with pytest.raises(ValueError, match=re.escape("x_nn must be a numpy array.")):
            kernel.update(x_nn=3)  # ty: ignore[invalid-argument-type]

    def test_rbf_kernel_set_sigma_not_array(self):
        kernel = copy.deepcopy(self.kernel)
        with pytest.raises(ValueError, match=re.escape("sigma must be a numpy array.")):
            kernel.update(sigma=[[1, 2], [3, 4]])  # ty: ignore[invalid-argument-type]

    def test_rbf_kernel_set_sigma_x_nn_wrong_dtype(self):
        kernel = copy.deepcopy(self.kernel)
        with pytest.warns(
            UserWarning,
            match=re.escape("x_nn is not of dtype FloatType. Converting to FloatType."),
        ):
            kernel.update(x_nn=np.array([1.0, 2.0, 3.0], dtype=np.float32))

    def test_rbf_kernel_set_sigma_x_nn_shape_mismatch(self):
        kernel = copy.deepcopy(self.kernel)
        dim = kernel.x_i.shape[0]
        with pytest.raises(
            ValueError,
            match=re.escape("x_nn must have the same number of rows as x_i."),
        ):
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
        from numpy import array  # noqa: F401

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


def test_validate_and_broadcast_param_success_floattype():
    param = FloatType(math.e)
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
    result = _validate_and_broadcast_param(param, name, dim)
    expected = np.array([math.e, math.pi, -1], dtype=FloatType)
    assert np.allclose(result, expected), (
        "Vector parameter should be returned correctly."
    )


def test_validate_and_broadcast_param_shape_mismatch():
    param = np.array([1.0, 2.0, 3.0], dtype=FloatType)
    name = "test_param"
    dim = 6
    with pytest.raises(
        ValueError, match=re.escape("test_param must have the same dimension as x_i.")
    ):
        _validate_and_broadcast_param(param, name, dim)


def test_validate_and_broadcast_param_wrong_type():
    param = "three"
    name = "test_param"
    dim = 6
    with pytest.raises(
        ValueError, match=re.escape("test_param must be a vector or a scalar.")
    ):
        _validate_and_broadcast_param(param, name, dim)  # ty: ignore[invalid-argument-type]


def test_validate_and_broadcast_param_cast_floattype():
    param = math.e
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


def test_fix_sigma_matrix():
    x_i = np.array(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=FloatType
    )  # Singular matrix

    RBFKernel(x_i=x_i)


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
        RuntimeError, match="_refresh_sigma_cache called before sigma is set"
    ):
        kernel._refresh_sigma_cache()


def test_get_sigma_inv_diag_jax_raises_when_sigma_inv_none():
    import copy

    kernel = copy.deepcopy(RBFKernel(x_i=_X_I_3, sigma=np.eye(3, dtype=FloatType)))
    kernel.sigma_inv = None
    kernel._sigma_inv_diag_jax = None
    with pytest.raises(RuntimeError, match="sigma_inv has not been set"):
        kernel._get_sigma_inv_diag_jax()


def test_get_sigma_inv_full_jax_raises_when_sigma_inv_none():
    import copy

    kernel = copy.deepcopy(RBFKernel(x_i=_X_I_3, sigma=_SIGMA_NON_DIAG))
    kernel.sigma_inv = None
    kernel._sigma_inv_full_jax = None
    with pytest.raises(RuntimeError, match="sigma_inv has not been set"):
        kernel._get_sigma_inv_full_jax()
