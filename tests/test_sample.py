# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT

import re

import numpy as np
import pytest

from autosafe.kernels.laplacian import LaplacianKernel
from autosafe.kernels.rbf import RBFKernel
from autosafe.sample import Sample


def test_sample_creation():
    """Test the creation of a Sample object."""
    x = np.array([1.0, 2.0, 3.0])
    kernel = RBFKernel(x_i=np.array([1.0, 2.0, 3.0]), sigma="eye")
    sample = Sample(x=x, kernel=kernel)

    assert np.array_equal(sample.x, x)
    assert sample.kernel == kernel
    assert sample.kernel is not None
    assert sample.kernel.x_i.shape == x.shape
    assert sample.kernel.x_i is not None
    assert sample.kernel.x_i is not None
    assert sample.kernel.x_i.shape == x.shape


def test_sample_creation_no_kernel():
    """Test the creation of a Sample object without a kernel."""
    x = np.array([1.0, 2.0, 3.0])
    sample = Sample(x=x)

    assert np.array_equal(sample.x, x)
    assert sample.kernel is None


def test_sample_creation_with_single_closest_sample():
    """Test the creation of a Sample object with a single closest sample."""
    x = np.array([1.0, 2.0, 3.0])
    closest_sample = Sample(x=np.array([0.0, 0.0, 0.0]))
    sample = Sample(x=x, closest_sample=[closest_sample])

    assert np.array_equal(sample.x, x)
    assert sample.kernel is None
    assert sample.closest_sample is not None
    assert len(sample.closest_sample) == 1
    assert np.array_equal(sample.closest_sample[0].x, closest_sample.x)


def test_sample_creation_with_multiple_closest_samples():
    """Test the creation of a Sample object with multiple closest samples."""
    x = np.array([1.0, 2.0, 3.0])
    closest_samples = [
        Sample(x=np.array([0.0, 0.0, 0.0])),
        Sample(x=np.array([1.0, 1.0, 1.0])),
    ]
    sample = Sample(x=x, closest_sample=closest_samples)

    assert np.array_equal(sample.x, x)
    assert sample.kernel is None
    assert sample.closest_sample is not None
    assert len(sample.closest_sample) == 2
    assert np.array_equal(sample.closest_sample[0].x, closest_samples[0].x)
    assert np.array_equal(sample.closest_sample[1].x, closest_samples[1].x)


def test_sample_creation_x_as_list():
    x = [1.0, 2.0, 3.0]
    sample = Sample(x=x)
    sample_arr = Sample(x=np.array(x))
    assert np.array_equal(sample.x, np.array(x))
    assert sample == sample_arr


def test_sample_creation_invalid_x_type():
    """Test the creation of a Sample object with an invalid x type."""
    with pytest.raises(TypeError):
        _ = Sample(x="invalid")  # ty: ignore[invalid-argument-type]


def test_sample_creation_x_not_1d():
    """Test the creation of a Sample object with a non-1D x array."""
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    with pytest.raises(
        ValueError, match=re.escape("x must be a one-dimensional array.")
    ):
        _ = Sample(x=x)


def test_sample_creation_invalid_kernel_type():
    """Test the creation of a Sample object with an invalid kernel type."""
    x = np.array([1.0, 2.0, 3.0])
    with pytest.raises(TypeError):
        _ = Sample(x=x, kernel="invalid")  # ty: ignore[invalid-argument-type]


def test_sample_creation_invalid_kernel_shape():
    """Test the creation of a Sample object with an invalid kernel shape."""
    x = np.array([1.0, 2.0, 3.0])
    kernel = RBFKernel(x_i=np.array([1.0, 2.0]), sigma="eye")
    with pytest.raises(
        ValueError, match=re.escape("Kernel center x_i must have the same shape as x.")
    ):
        _ = Sample(x=x, kernel=kernel)


def test_sample_creation_invalid_closest_sample_type():
    """Test the creation of a Sample object with an invalid closest_sample
    type."""
    x = np.array([1.0, 2.0, 3.0])
    with pytest.raises(TypeError):
        _ = Sample(x=x, closest_sample=["invalid"])  # ty: ignore[invalid-argument-type]


def test_sample_returns_same_value_than_kernel():
    """Test that the sample returns the same value as the kernel when
    evaluated."""
    x = np.array([1.0, 2.0, 3.0])
    kernel = RBFKernel(x_i=np.array([1.0, 2.0, 3.0]), sigma="eye")
    sample = Sample(x=x, kernel=kernel)

    assert sample.kernel is not None
    assert sample(x) == kernel(x)
    assert sample(x) == sample.kernel(x)
    assert sample(sample.x) == kernel(sample.x)
    assert sample(sample.x) == sample.kernel(sample.x)


def test_sample_call_fails_without_kernel():
    """Test that calling a Sample without a kernel raises a ValueError."""
    x = np.array([1.0, 2.0, 3.0])
    sample = Sample(x=x)

    with pytest.raises(
        ValueError, match=re.escape("Kernel is not defined for this sample.")
    ):
        _ = sample(x)


def test_sample_setters():
    """Test the setters of the Sample class."""
    x = np.array([1.0, 2.0, 3.0])
    sample = Sample(x=x)

    new_x = np.array([4.0, 5.0, 6.0])
    sample.x = new_x
    assert np.array_equal(sample.x, new_x)

    kernel = LaplacianKernel(x_i=np.array([4.0, 5.0, 6.0]), alpha=None)
    sample.kernel = kernel
    assert sample.kernel == kernel

    closest_samples = [
        Sample(x=np.array([1.0, 1.0, 1.0])),
        Sample(x=np.array([2.0, 2.0, 2.0])),
    ]
    sample.closest_sample = closest_samples
    assert sample.closest_sample == closest_samples


def test_sample_repr_contains_all_info():
    """Test the __repr__ method of the Sample class."""
    x = np.array([1.0, 2.0, 3.0])
    kernel = RBFKernel(x_i=np.array([1.0, 2.0, 3.0]), sigma="eye")
    closest_samples = [
        Sample(x=np.array([0.0, 0.0, 0.0])),
        Sample(x=np.array([1.0, 1.0, 1.0])),
    ]
    sample = Sample(x=x, kernel=kernel, closest_sample=closest_samples)

    repr_str = repr(sample)
    assert repr_str.startswith("Sample(")
    assert f"x={sample.x!r}" in repr_str
    assert f"kernel={sample.kernel!r}" in repr_str
    assert repr_str.endswith(")")


def test_sample_repr_allows_eval():
    """Test that the __repr__ method of the Sample class allows eval."""
    from numpy import array  # noqa: F401

    x = np.array([1.0, 2.0, 3.0])
    kernel = RBFKernel(x_i=np.array([1.0, 2.0, 3.0]), sigma="eye")
    closest_samples = [
        Sample(x=np.array([0.0, 0.0, 0.0])),
        Sample(x=np.array([1.0, 1.0, 1.0])),
    ]
    sample = Sample(x=x, kernel=kernel, closest_sample=closest_samples)

    evaluated_sample = eval(repr(sample))

    assert isinstance(evaluated_sample, Sample)
    assert np.allclose(evaluated_sample.x, sample.x)
    assert evaluated_sample.kernel == sample.kernel


def test_sample_str_contains_all_info():
    """Test the __str__ method of the Sample class."""
    x = np.array([1.0, 2.0, 3.0])
    kernel = RBFKernel(x_i=np.array([1.0, 2.0, 3.0]), sigma="eye")
    closest_samples = [
        Sample(x=np.array([0.0, 0.0, 0.0])),
        Sample(x=np.array([1.0, 1.0, 1.0])),
    ]
    sample = Sample(x=x, kernel=kernel, closest_sample=closest_samples)

    str_str = str(sample)
    assert str_str.startswith("Sample")
    assert f"x={sample.x!s}" in str_str
    assert f"kernel={sample.kernel!s}" in str_str


def test_sample_equality():
    """Test the __eq__ method of the Sample class."""
    x = np.array([1.0, 2.0, 3.0])
    kernel1 = RBFKernel(x_i=np.array([1.0, 2.0, 3.0]), sigma="eye")
    kernel2 = RBFKernel(x_i=np.array([1.0, 2.0, 3.0]), sigma="eye")
    kernel3 = RBFKernel(x_i=np.array([1.0, 2.0]), sigma="eye")

    sample1 = Sample(x=x, kernel=kernel1)
    sample2 = Sample(x=x, kernel=kernel2)
    sample3 = Sample(x=np.array([1.0, 2.0]), kernel=kernel3)
    sample4 = Sample(x=np.array([4.0, 5.0, 6.0]), kernel=kernel1)
    sample5 = Sample(x=x)

    assert sample1 == sample2, "Samples with same x and kernel should be equal."
    assert sample1 != sample3, "Samples with different kernels should not be equal."
    assert sample1 != sample4, "Samples with different x should not be equal."
    assert sample1 != sample5, "Samples with and without kernel should not be equal."
    assert sample5 == Sample(x=x), "Samples without kernel but same x should be equal."
    assert sample1 != "invalid", "Sample should not be equal to an invalid type."

    # Test for hashing consistency
    assert sample1 in {sample1, sample2}, (
        "Samples with same x and kernel should be equal."
    )
    assert sample1 not in {sample3, sample4, sample5}, (
        "Samples with different x or kernel should not be equal."
    )
