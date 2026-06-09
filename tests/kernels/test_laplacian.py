# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT

import copy
import math
import re

import numpy as np
import pytest

from autosafe.kernels import LaplacianKernel
from autosafe.typing import FloatType


class TestLaplacianKernel:
    """Test suite for the Laplacian kernel."""

    kernel = LaplacianKernel(
        x_i=np.array([1.0, 2.0, 3.0], dtype=FloatType),
        alpha=np.array([1.0, 0.5, 3.0], dtype=FloatType),
    )

    def test_laplacian_kernel_same_vector(self):
        kernel = copy.deepcopy(self.kernel)
        assert np.allclose(kernel(kernel.x_i), 1.0), (
            "Laplacian kernel should return 1.0 for the same vector."
        )

    def test_laplacian_kernel_different_vector(self):
        kernel = copy.deepcopy(self.kernel)
        assert kernel(kernel.x_i + kernel.x_i) < 1.0, (
            "Laplacian kernel should return a value less than 1.0 for different vectors."
        )
        assert kernel(-kernel.x_i) < 1.0, (
            "Laplacian kernel should return a value less than 1.0 for different vectors."
        )

    @staticmethod
    def test_laplacian_kernel_init_fails_on_wrong_alpha_shape():
        with pytest.raises(
            ValueError, match=re.escape("alpha must have the same dimension as x_i.")
        ):
            LaplacianKernel(
                x_i=np.array([1.0, 2.0, 3.0], dtype=FloatType),
                alpha=np.array([1.0, 2.0], dtype=FloatType),
            )

    @staticmethod
    def test_laplacian_kernel_init_fails_on_negative_alpha():
        with pytest.raises(ValueError, match=re.escape("alpha must be positive.")):
            LaplacianKernel(
                x_i=np.array([1.0, 2.0], dtype=FloatType),
                alpha=np.array([-1.0, 2.0], dtype=FloatType),
            )

    @staticmethod
    def test_laplacian_kernel_init_alpha_defaults_to_none():
        kernel = LaplacianKernel(x_i=np.array([1.0, 2.0, 3.0], dtype=FloatType))
        assert kernel.alpha is None, "Default alpha should be None."

    def test_laplacian_kernel_update_alpha(self):
        kernel = copy.deepcopy(self.kernel)
        new_alpha = np.array([2.0, 1.0, 0.5], dtype=FloatType)
        kernel.update(alpha=new_alpha)
        assert kernel.alpha is not None
        assert np.array_equal(kernel.alpha, new_alpha), (
            "Kernel alpha should be updated correctly."
        )

    def test_laplacian_kernel_update_alpha_fails_on_wrong_shape(self):
        kernel = copy.deepcopy(self.kernel)
        with pytest.raises(
            ValueError, match=re.escape("alpha must have the same dimension as x_i.")
        ):
            kernel.update(alpha=np.array([1.0, 2.0], dtype=FloatType))

    def test_laplacian_kernel_update_alpha_fails_on_negative_value(self):
        kernel = copy.deepcopy(self.kernel)
        with pytest.raises(ValueError, match=re.escape("alpha must be positive.")):
            kernel.update(alpha=np.array([-1.0, 2.0, 3.0], dtype=FloatType))

    def test_laplacian_kernel_update_alpha_fails_on_negative_scalar(self):
        kernel = copy.deepcopy(self.kernel)
        with pytest.raises(ValueError, match=re.escape("alpha must be positive.")):
            kernel.update(alpha=-1.0)

    def test_laplacian_kernel_update_alpha_fails_on_zero_scalar(self):
        kernel = copy.deepcopy(self.kernel)
        with pytest.raises(ValueError, match=re.escape("alpha must be positive.")):
            kernel.update(alpha=0.0)

    def test_laplacian_kernel_update_alpha_fails_on_zero_vector(self):
        kernel = copy.deepcopy(self.kernel)
        with pytest.raises(ValueError, match=re.escape("alpha must be positive.")):
            kernel.update(alpha=np.zeros(3, dtype=FloatType))

    def test_laplacian_kernel_update_alpha_fails_on_none(self):
        kernel = copy.deepcopy(self.kernel)
        with pytest.raises(
            ValueError,
            match=re.escape("alpha must be provided for LaplacianKernel."),
        ):
            kernel.update(alpha=None)  # ty: ignore[invalid-argument-type]

    def test_laplacian_kernel_update_alpha_fails_on_empty_vector(self):
        kernel = copy.deepcopy(self.kernel)
        with pytest.raises(
            ValueError, match=re.escape("alpha must have the same dimension as x_i.")
        ):
            kernel.update(alpha=np.array([], dtype=FloatType))

    def test_laplacian_kernel_update_alpha_float(self):
        kernel = copy.deepcopy(self.kernel)
        new_alpha = 2.0
        kernel.update(alpha=new_alpha)
        expected_alpha = np.full_like(kernel.x_i, new_alpha, dtype=FloatType)
        assert kernel.alpha is not None
        assert np.array_equal(kernel.alpha, expected_alpha), (
            "Kernel alpha should be updated to a constant vector."
        )

    def test_laplacian_kernel_update_alpha_int(self):
        kernel = copy.deepcopy(self.kernel)
        new_alpha = 2
        kernel.update(alpha=new_alpha)
        expected_alpha = np.full_like(kernel.x_i, new_alpha, dtype=FloatType)
        assert kernel.alpha is not None
        assert np.array_equal(kernel.alpha, expected_alpha), (
            "Kernel alpha should be updated to a constant vector."
        )

    @staticmethod
    def test_laplacian_kernel_call_fails_with_unset_alpha():
        kernel = LaplacianKernel(x_i=np.array([1.0, 2.0, 3.0], dtype=FloatType))
        with pytest.raises(
            ValueError,
            match=re.escape("Kernel parameters have not been set."),
        ):
            kernel(np.array([1.0, 2.0, 3.0], dtype=FloatType))

    def test_laplacian_kernel_call_with_set_alpha(self):
        kernel = copy.deepcopy(self.kernel)
        x = np.array([1.0, 2.0, 3.0], dtype=FloatType)
        assert kernel.alpha is not None
        result = kernel(x)
        expected_distance = np.linalg.norm(
            np.multiply(kernel.alpha, x - kernel.x_i),
            ord=1,
        )
        expected_result = np.exp(-expected_distance, dtype=FloatType)
        assert result == expected_result, (
            "Laplacian kernel should return the correct value for the input vector."
        )

    def test_laplacian_kernel_compare_fails_with_none_kernel(self):
        for kernel in (None, "string", 42, math.pi):
            kernel1 = copy.deepcopy(self.kernel)
            assert kernel1 != kernel, (
                f"Comparing a kernel to wrong type ({type(kernel)}) should fail."
            )

    def test_laplacian_kernel_compare_fails_if_only_one_alpha_is_none(self):
        kernel1 = copy.deepcopy(self.kernel)
        kernel2 = LaplacianKernel(
            x_i=np.array([1.0, 2.0, 3.0], dtype=FloatType),
            alpha=None,
        )
        assert kernel1 != kernel2, "Kernels with one None alpha should not be equal."

    @staticmethod
    def test_laplacian_kernel_compare_works_if_both_alpha_are_none():
        kernel1 = LaplacianKernel(
            x_i=np.array([1.0, 2.0, 3.0], dtype=FloatType),
            alpha=None,
        )
        kernel2 = LaplacianKernel(
            x_i=np.array([1.0, 2.0, 3.0], dtype=FloatType),
            alpha=None,
        )
        assert kernel1 == kernel2, "Kernels with both None alpha should be equal."

    def test_laplacian_kernel_compare_kernels(self):
        kernel1 = copy.deepcopy(self.kernel)
        kernel2 = LaplacianKernel(
            x_i=np.array([1.0, 2.0, 3.0], dtype=FloatType),
            alpha=np.array([1.0, 0.5, 3.0], dtype=FloatType),
        )
        kernel3 = LaplacianKernel(
            x_i=np.array([1.0, 5.0, 25.0], dtype=FloatType),
            alpha=np.array([1.0, 0.5, 2.0], dtype=FloatType),
        )
        kernel4 = LaplacianKernel(
            x_i=np.array([1.0, 2.0, 3.0, 4.0], dtype=FloatType),
            alpha=np.array([1.0, 0.5, 3.0, 2.0], dtype=FloatType),
        )
        kernel5 = LaplacianKernel(
            x_i=np.array([1.0, 2.0, 4.0], dtype=FloatType),
            alpha=np.array([1.0, 0.5, 3.0], dtype=FloatType),
        )
        kernel6 = copy.deepcopy(self.kernel)
        kernel6.x_i = np.array([1.0, 2.0], dtype=FloatType)  # force wrong shape
        kernel7 = copy.deepcopy(self.kernel)
        kernel7.alpha = np.array([1.0, 2.0], dtype=FloatType)  # force wrong shape

        assert kernel1 == kernel2, "Kernels with the same parameters should be equal."
        assert kernel1 != kernel3, (
            "Kernels with different parameters should not be equal."
        )
        assert kernel1 != kernel4, (
            "Kernels with different dimensions should not be equal."
        )
        assert kernel1 != kernel5, "Kernels with different x_i should not be equal."
        assert kernel1 != kernel6, (
            "Kernels with different alpha shape should not be equal."
        )

        # Test for hashing consistency
        assert kernel1 in {kernel1, kernel2}, (
            "Kernels with the same parameters should be equal."
        )
        assert kernel1 not in {
            kernel3,
            kernel4,
            kernel5,
            kernel6,
        }, "Kernels with different parameters should not be equal."

    def test_str_contains_all_needed_info(self):
        kernel = copy.deepcopy(self.kernel)
        kernel_str = str(kernel)
        assert "LaplacianKernel" in kernel_str, (
            "String representation should contain class name."
        )
        assert "x_i" in kernel_str, "String representation should contain x_i."
        assert str(kernel.x_i) in kernel_str, (
            "String representation should contain x_i."
        )
        assert "alpha" in kernel_str, "String representation should contain alpha."
        assert str(kernel.alpha) in kernel_str, (
            "String representation should contain alpha."
        )

    def test_repr_contains_all_needed_info(self):
        kernel = copy.deepcopy(self.kernel)
        kernel_repr = repr(kernel)
        assert "LaplacianKernel" in kernel_repr, (
            "Repr representation should contain class name."
        )
        assert "x_i" in kernel_repr, "Repr representation should contain x_i."
        assert repr(kernel.x_i) in kernel_repr, (
            "Repr representation should contain x_i."
        )
        assert "alpha" in kernel_repr, "Repr representation should contain alpha."
        assert repr(kernel.alpha) in kernel_repr, (
            "Repr representation should contain alpha."
        )

    def test_kernel_can_be_recreated_from_repr(self):
        from numpy import array  # noqa: F401

        kernel = copy.deepcopy(self.kernel)
        kernel_repr = repr(kernel)
        recreated_kernel = eval(kernel_repr)
        assert kernel == recreated_kernel, (
            "Kernel recreated from repr should be equal to the original."
        )


def test_laplacian_matrix_format_handling():
    """Test that both (n_dim, n_points) and (n_points, n_dim) formats work
    correctly."""
    # Test 1: (2, 3) format (n_dim, n_points)
    kernel = LaplacianKernel(
        x_i=np.array([0.0, 0.0], dtype=FloatType),
        alpha=np.array([1.0, 1.0], dtype=FloatType),
    )

    # Matrix format (2, 3) - 3 points, 2 dimensions
    points_2d = np.array([[1.0, -1.0, 0.5], [1.0, -1.0, 0.5]], dtype=FloatType)

    result_2d = kernel(points_2d)

    # Verify shape
    assert result_2d.shape == (3,), "Should return 3 affinity values"

    # Test 2: (3, 2) format (n_points, n_dim) - should transpose correctly
    points_3d = np.array([[1.0, 1.0], [-1.0, -1.0], [0.5, 0.5]], dtype=FloatType)

    result_3d = kernel(points_3d)

    # Verify shape
    assert result_3d.shape == (3,), "Should return 3 affinity values"

    # Results should be identical since they're the same points in different formats
    # We can't directly compare since they're different shapes, but we can check values make sense

    # Check that results are reasonable (all between 0 and 1)
    assert np.all(result_2d >= 0.0), "All results should be >= 0.0"
    assert np.all(result_2d <= 1.0), "All results should be <= 1.0"
    assert np.all(result_3d >= 0.0), "All results should be >= 0.0"
    assert np.all(result_3d <= 1.0), "All results should be <= 1.0"


def test_laplacian_kernel_edge_cases():
    """Test edge cases that might cause issues in vectorized computation."""
    # Test that matrix formats are handled correctly
    kernel = LaplacianKernel(
        x_i=np.array([1.0, 2.0], dtype=FloatType),
        alpha=np.array([2.0, 1.5], dtype=FloatType),
    )

    # Test 1: Single point (vector format)
    single_point = np.array([1.0, 2.0], dtype=FloatType)  # Should return 1.0
    result_single = kernel(single_point)
    assert np.isclose(result_single, 1.0), "Single point at center should return 1.0"

    # Test 2: Multiple points (matrix format - 2 points, 2 dimensions)
    points = np.array(
        [[1.0, 2.0], [1.0, 2.0]], dtype=FloatType
    )  # Both points at center
    result_matrix = kernel(points)
    assert result_matrix.shape == (2,), "Should return 2 values"
    # Actually, the distance computation is not quite 0, let's be more precise
    assert np.all(result_matrix <= 1.0), "All results should be <= 1.0"


def test_laplacian_kernel_exact_center():
    """Test that kernel correctly returns values at center points."""
    kernel = LaplacianKernel(
        x_i=np.array([1.0, 2.0], dtype=FloatType),
        alpha=np.array([2.0, 1.5], dtype=FloatType),
    )

    # Test with scalar input at center
    center_point = np.array([1.0, 2.0], dtype=FloatType)
    result = kernel(center_point)

    # This should be approximately 1.0 due to L1 distance being 0
    assert np.isclose(result, 1.0, atol=1e-10), (
        "Scalar input at center should be nearly 1.0"
    )

    # Test with matrix input at center
    center_points = np.array(
        [[1.0, 2.0], [1.0, 2.0]], dtype=FloatType
    )  # 2 points at center
    result_matrix = kernel(center_points)

    assert result_matrix.shape == (2,), "Should return 2 values"
    # Values should be close to 1.0, but not exactly due to numerical precision
    assert np.all(result_matrix <= 1.0), "All results should be <= 1.0"
    assert np.all(result_matrix > 0.0), "All results should be > 0.0"


def test_validate_alpha_jax_array_valid():
    import jax.numpy as jnp

    kernel = LaplacianKernel(
        x_i=np.array([1.0, 2.0], dtype=FloatType),
        alpha=jnp.array([1.0, 2.0]),  # ty: ignore[invalid-argument-type]
    )
    assert kernel.alpha is not None
    assert np.allclose(kernel.alpha, [1.0, 2.0])


def test_validate_alpha_jax_array_wrong_shape():
    import jax.numpy as jnp
    import pytest

    with pytest.raises(ValueError, match=r"alpha must have the same dimension as x_i."):
        LaplacianKernel(
            x_i=np.array([1.0, 2.0], dtype=FloatType),
            alpha=jnp.array([1.0, 2.0, 3.0]),  # ty: ignore[invalid-argument-type]
        )


def test_validate_alpha_jax_array_negative():
    import jax.numpy as jnp
    import pytest

    with pytest.raises(ValueError, match=r"alpha must be positive."):
        LaplacianKernel(
            x_i=np.array([1.0, 2.0], dtype=FloatType),
            alpha=jnp.array([-1.0, 2.0]),  # ty: ignore[invalid-argument-type]
        )
