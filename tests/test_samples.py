# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
import math
import re
from typing import Any

import jax
import msgspec.json
import numpy as np
import pytest

from autosafe.kernels.rbf import RBFKernel
from autosafe.sample import Sample
from autosafe.samples import (
    Samples,
    find_closest_vectors_by_index,
    find_closest_vectors_by_index_per_dimension,
)
from autosafe.tools.serializers.msgspec import decode_hook, encode_hook
from autosafe.typing import (
    ClosestSampleModeType,
    KernelType,
)

CLOSEST_SAMPLE_MODES = [
    "global",
    "per_dimension",
]

KERNELS = [
    ("RBF", None),
    ("RBF", {"sigma": "eye"}),
    ("RBF", {"kappa": 2.0, "eta": 0.5}),
    ("RBF", {"kappa": 0.1, "eta": 4.0}),
    ("Laplacian", {"alpha": 0.1}),
    ("Laplacian", {"alpha": 1.0}),
    ("Laplacian", {"alpha": 10.0}),
]

testdata = [
    (mode, kernel_cls, kernel_kwargs)
    for mode in CLOSEST_SAMPLE_MODES
    for kernel_cls, kernel_kwargs in KERNELS
]


@pytest.mark.parametrize(
    ("closest_sample_mode", "kernel_cls", "kernel_kwargs"), testdata
)
def test_successful_init_single_sample_sample(
    closest_sample_mode: ClosestSampleModeType,
    kernel_cls: KernelType,
    kernel_kwargs: dict[str, Any] | None,
):  # Sample
    """Test successful initialization with a single Sample object."""
    sample = Sample(x=[0.5, 1.0, -0.5])
    samples = Samples(
        sample,
        closest_sample_mode=closest_sample_mode,
        kernel_cls=kernel_cls,
        kernel_kwargs=kernel_kwargs,
    )
    assert len(samples.samples) == 1
    assert len(samples) == 1
    assert samples.dim == 3
    assert samples.shape == (1, 3)


@pytest.mark.parametrize(
    ("closest_sample_mode", "kernel_cls", "kernel_kwargs"), testdata
)
def test_successful_init_multiple_samples_sample(
    closest_sample_mode: ClosestSampleModeType,
    kernel_cls: KernelType,
    kernel_kwargs: dict[str, Any] | None,
):  # list[Sample]
    """Test successful initialization with multiple Sample objects."""
    sample_list = [
        Sample(x=[0.5, 1.0]),
        Sample(x=[-0.5, 0.0]),
        Sample(x=[1.5, -1.0]),
    ]
    samples = Samples(
        sample_list,
        closest_sample_mode=closest_sample_mode,
        kernel_cls=kernel_cls,
        kernel_kwargs=kernel_kwargs,
    )
    assert len(samples.samples) == 3
    assert len(samples) == 3
    assert samples.dim == 2
    assert samples.shape == (3, 2)


@pytest.mark.parametrize(
    ("closest_sample_mode", "kernel_cls", "kernel_kwargs"), testdata
)
def test_successful_init_single_sample_list(
    closest_sample_mode: ClosestSampleModeType,
    kernel_cls: KernelType,
    kernel_kwargs: dict[str, Any] | None,
):  # list[float]
    """Test successful initialization with a single sample."""
    sample = [0.5, 1.0, -0.5, 1.0]
    samples = Samples(
        sample,
        closest_sample_mode=closest_sample_mode,
        kernel_cls=kernel_cls,
        kernel_kwargs=kernel_kwargs,
    )
    assert len(samples.samples) == 1
    assert len(samples) == 1
    assert samples.dim == 4
    assert samples.shape == (1, 4)


@pytest.mark.parametrize(
    ("closest_sample_mode", "kernel_cls", "kernel_kwargs"), testdata
)
def test_successful_init_multiple_samples_list(
    closest_sample_mode: ClosestSampleModeType,
    kernel_cls: KernelType,
    kernel_kwargs: dict[str, Any] | None,
):  # list[list[float]]
    """Test successful initialization with multiple samples."""
    sample_list = [[0.5, 1.0], [-0.5, 0.0], [1.5, -1.0]]
    samples = Samples(
        sample_list,
        closest_sample_mode=closest_sample_mode,
        kernel_cls=kernel_cls,
        kernel_kwargs=kernel_kwargs,
    )
    assert len(samples.samples) == 3
    assert samples.dim == 2
    assert samples.shape == (3, 2)


@pytest.mark.parametrize(
    ("closest_sample_mode", "kernel_cls", "kernel_kwargs"), testdata
)
def test_successful_init_single_sample_array(
    closest_sample_mode: ClosestSampleModeType,
    kernel_cls: KernelType,
    kernel_kwargs: dict[str, Any] | None,
):  # Vector
    """Test successful initialization with a single sample array."""
    dim = 12
    sample_array = np.zeros(dim)
    samples = Samples(
        sample_array,
        closest_sample_mode=closest_sample_mode,
        kernel_cls=kernel_cls,
        kernel_kwargs=kernel_kwargs,
    )
    assert len(samples.samples) == 1
    assert len(samples) == 1
    assert samples.dim == dim
    assert samples.shape == (1, dim)


@pytest.mark.parametrize(
    ("closest_sample_mode", "kernel_cls", "kernel_kwargs"), testdata
)
def test_successful_init_multiple_samples_list_vector(
    closest_sample_mode: ClosestSampleModeType,
    kernel_cls: KernelType,
    kernel_kwargs: dict[str, Any] | None,
):  # list[Vector]
    """Test successful initialization with multiple samples as list of
    Vectors."""
    sample_list = [
        np.array([0.5, 1.0]),
        np.array([-0.5, 0.0]),
        np.array([1.5, -1.0]),
    ]
    samples = Samples(
        sample_list,
        closest_sample_mode=closest_sample_mode,
        kernel_cls=kernel_cls,
        kernel_kwargs=kernel_kwargs,
    )
    assert len(samples.samples) == 3
    assert len(samples) == 3
    assert samples.dim == 2
    assert samples.shape == (3, 2)


@pytest.mark.parametrize(
    ("closest_sample_mode", "kernel_cls", "kernel_kwargs"), testdata
)
def test_successful_init_multiple_samples_array(
    closest_sample_mode: ClosestSampleModeType,
    kernel_cls: KernelType,
    kernel_kwargs: dict[str, Any] | None,
):  # Matrix
    """Test successful initialization with multiple samples array."""
    num_samples = 5
    dim = 3
    rng = np.random.default_rng()
    sample_array = rng.random((num_samples, dim))
    samples = Samples(
        sample_array,
        closest_sample_mode=closest_sample_mode,
        kernel_cls=kernel_cls,
        kernel_kwargs=kernel_kwargs,
    )
    assert len(samples.samples) == num_samples
    assert len(samples) == num_samples
    assert samples.dim == dim
    assert samples.shape == (num_samples, dim)


@pytest.mark.parametrize(
    ("closest_sample_mode", "kernel_cls", "kernel_kwargs"), testdata
)
def test_init_wrong_type_raises_typeerror(
    closest_sample_mode: ClosestSampleModeType,
    kernel_cls: KernelType,
    kernel_kwargs: dict[str, Any] | None,
):
    """Test that initialization with wrong type raises TypeError."""
    with pytest.raises(TypeError):
        _ = Samples(42)  # type: ignore

    with pytest.raises(TypeError):
        _ = Samples("invalid input")  # type: ignore

    with pytest.raises(TypeError):
        _ = Samples([Sample(x=[0.5]), "invalid"])  # type: ignore

    with pytest.raises(TypeError):
        _ = Samples(
            [Sample(x=[0.5]), "invalid"],  # ty: ignore[invalid-argument-type]
            closest_sample_mode=closest_sample_mode,
            kernel_cls=kernel_cls,
            kernel_kwargs=kernel_kwargs,
        )


@pytest.mark.parametrize(
    ("closest_sample_mode", "kernel_cls", "kernel_kwargs"), testdata
)
def test_wrong_dimension_raises_valueerror(
    closest_sample_mode: ClosestSampleModeType,
    kernel_cls: KernelType,
    kernel_kwargs: dict[str, Any] | None,
):
    """Test that initialization with samples of different dimensions raises
    ValueError."""
    with pytest.raises(
        ValueError, match=re.escape("All samples must have the same dimension.")
    ):
        _ = Samples(
            [Sample(x=[0.5, 1.0]), Sample(x=[-0.5])],
            closest_sample_mode=closest_sample_mode,
            kernel_cls=kernel_cls,
            kernel_kwargs=kernel_kwargs,
        )

    with pytest.raises(ValueError, match=re.escape("Each sample must be a 1D array.")):
        _ = Samples(
            [[0.5, 1.0], [0.0]],
            closest_sample_mode=closest_sample_mode,
            kernel_cls=kernel_cls,
            kernel_kwargs=kernel_kwargs,
        )

    with pytest.raises(ValueError, match=re.escape("Each sample must be a 1D array.")):
        _ = Samples(
            [np.array([0.5, 1.0]), np.array([0.0])],
            closest_sample_mode=closest_sample_mode,
            kernel_cls=kernel_cls,
            kernel_kwargs=kernel_kwargs,
        )

    with pytest.raises(ValueError, match=re.escape("Input array must be 1D or 2D.")):
        _ = Samples(
            np.zeros((3, 3, 3)),
            closest_sample_mode=closest_sample_mode,
            kernel_cls=kernel_cls,
            kernel_kwargs=kernel_kwargs,
        )


@pytest.mark.parametrize(
    ("closest_sample_mode", "kernel_cls", "kernel_kwargs"), testdata
)
def test_wrong_dimension_append_raises_valueerror_on_append(
    closest_sample_mode: ClosestSampleModeType,
    kernel_cls: KernelType,
    kernel_kwargs: dict[str, Any] | None,
):
    """Test that appending samples of different dimensions raises
    ValueError."""
    samples = Samples(
        [0.5, 1.0],
        closest_sample_mode=closest_sample_mode,
        kernel_cls=kernel_cls,
        kernel_kwargs=kernel_kwargs,
    )

    with pytest.raises(
        ValueError, match=re.escape("All samples must have the same dimension.")
    ):
        samples.append(Sample(x=[-0.5]))

    with pytest.raises(ValueError, match=re.escape("Each sample must be a 1D array.")):
        samples.append([[0.0]])

    with pytest.raises(ValueError, match=re.escape("Input array must be 1D or 2D.")):
        samples.append(np.array([0.0]))


@pytest.mark.parametrize(
    ("closest_sample_mode", "kernel_cls", "kernel_kwargs"), testdata
)
def test_samples_equals(
    closest_sample_mode: ClosestSampleModeType,
    kernel_cls: KernelType,
    kernel_kwargs: dict[str, Any] | None,
):
    """Test the equality operator."""
    sample_list = [
        Sample(x=[0.5, 1.0]),
        Sample(x=[-0.5, 0.0]),
        Sample(x=[1.5, -1.0]),
    ]
    samples1 = Samples(
        sample_list,
        closest_sample_mode=closest_sample_mode,
        kernel_cls=kernel_cls,
        kernel_kwargs=kernel_kwargs,
    )
    samples2 = Samples(
        sample_list,
        closest_sample_mode=closest_sample_mode,
        kernel_cls=kernel_cls,
        kernel_kwargs=kernel_kwargs,
    )
    samples3 = Samples(
        sample_list[:1],
        closest_sample_mode=closest_sample_mode,
        kernel_cls=kernel_cls,
        kernel_kwargs=kernel_kwargs,
    )
    samples4 = Samples(
        [*sample_list, Sample(x=[2.0, 2.0])],
        closest_sample_mode=closest_sample_mode,
        kernel_cls=kernel_cls,
        kernel_kwargs=kernel_kwargs,
    )
    samples5 = Samples(
        [Sample(x=[0.5, 1.0]), Sample(x=[-0.5, 0.0])],
        closest_sample_mode=closest_sample_mode,
        kernel_cls=kernel_cls,
        kernel_kwargs=kernel_kwargs,
    )
    samples6 = 0
    samples7 = "invalid"

    assert samples1 == samples2, "Identical samples should be equal."
    assert samples1 != samples3, "Samples with different content should not be equal."
    assert samples1 != samples4, "Samples with different content should not be equal."
    assert samples1 != samples5, (
        "Samples with different number of samples should not be equal."
    )
    assert samples1 != samples6, "Samples should not be equal to an integer."
    assert samples1 != samples7, "Samples should not be equal to a string."

    # Test for hashing consistency
    assert samples1 in {samples1, samples2}, (
        "Identical samples should be equal in a set."
    )
    assert samples1 not in {samples3, samples4, samples5, samples6, samples7}, (
        "Different samples should not be equal in a set."
    )


@pytest.mark.parametrize(
    ("closest_sample_mode", "kernel_cls", "kernel_kwargs"), testdata
)
def test_samples_unique_samples_have_non_eye_kernel(
    closest_sample_mode: ClosestSampleModeType,
    kernel_cls: KernelType,
    kernel_kwargs: dict[str, Any] | None,
):
    """Test that samples with unique positions have non-identity kernels."""
    sample_list = [
        Sample(x=[0.5, 1.0]),
        Sample(x=[-0.5, 0.0]),
        Sample(x=[1.5, -1.0]),
    ]
    samples = Samples(
        sample_list,
        closest_sample_mode=closest_sample_mode,
        kernel_cls=kernel_cls,
        kernel_kwargs=kernel_kwargs,
    )

    for sample in sample_list:
        assert sample.kernel is not None, "Kernel should be set."
        if kernel_cls == "RBF" and (
            kernel_kwargs is None or kernel_kwargs.get("sigma") != "eye"
        ):
            assert isinstance(sample.kernel, RBFKernel)
            assert samples.dim is not None
            assert samples.dim == sample.kernel.x_i.shape[0]
            assert not np.allclose(
                sample.kernel.sigma,  # ty: ignore[invalid-argument-type]
                np.eye(samples.dim),
            ), "Kernel should not be identity matrix for unique samples."


@pytest.mark.parametrize(
    ("closest_sample_mode", "kernel_cls", "kernel_kwargs"), testdata
)
def test_samples_call_returns_floatingtype(
    closest_sample_mode: ClosestSampleModeType,
    kernel_cls: KernelType,
    kernel_kwargs: dict[str, Any] | None,
):
    """Test that calling Samples returns a floating point number."""
    sample_list = [
        Sample(x=[0.5, 1.0]),
        Sample(x=[-0.5, 0.0]),
        Sample(x=[1.5, -1.0]),
    ]
    samples = Samples(
        sample_list,
        closest_sample_mode=closest_sample_mode,
        kernel_cls=kernel_cls,
        kernel_kwargs=kernel_kwargs,
    )

    for sample in sample_list:
        result = samples(sample.x)
        assert isinstance(result, jax.Array), (
            "Calling Samples should return a JAX array."
        )
        assert result.ndim == 0, "Calling Samples should return a 0-d JAX array."
        assert result >= 0.0, "Result should be non-negative."
        assert result <= 1.0, "Result should be less than or equal to 1.0."

    result = samples(np.array([3.0, -7.0]))
    assert isinstance(result, jax.Array), "Calling Samples should return a JAX array."
    assert result.ndim == 0, "Calling Samples should return a 0-d JAX array."
    assert result >= 0.0, "Result should be non-negative."
    assert result <= 1.0, "Result should be less than or equal to 1.0."


@pytest.mark.parametrize(
    ("closest_sample_mode", "kernel_cls", "kernel_kwargs"), testdata
)
def test_samples_call_returns_vector_of_affinities_for_matrix_input(
    closest_sample_mode: ClosestSampleModeType,
    kernel_cls: KernelType,
    kernel_kwargs: dict[str, Any] | None,
):
    """Test evaluating multiple input vectors at the same time."""
    sample_list = [
        Sample(x=[0.5, 1.0]),
        Sample(x=[-0.5, 0.0]),
        Sample(x=[1.5, -1.0]),
    ]
    samples = Samples(
        sample_list,
        closest_sample_mode=closest_sample_mode,
        kernel_cls=kernel_cls,
        kernel_kwargs=kernel_kwargs,
    )

    input_vectors = np.array([
        [0.5, 1.0],
        [3.0, -7.0],
        [-0.5, 0.0],
        [1.5, -1.0],
    ])

    # First, pass all vectors one by one and collect affinities
    scalar_affinities = []
    for vec in input_vectors:
        affinity = samples(vec)
        assert isinstance(affinity, jax.Array), (
            "Calling Samples should return a JAX array."
        )
        assert affinity.ndim == 0, (
            "Calling Samples with a vector should return a 0-d JAX array."
        )
        assert affinity >= 0.0, "Result should be non-negative."
        assert affinity <= 1.0, "Result should be less than or equal to 1.0."
        scalar_affinities.append(affinity)

    # Now, pass all vectors at once as a matrix
    matrix_affinities = samples(input_vectors)
    assert isinstance(matrix_affinities, jax.Array), (
        "Calling Samples with a matrix should return a JAX array."
    )
    assert matrix_affinities.shape == (input_vectors.shape[0],), (
        "Resulting affinity array should have correct shape."
    )

    # Last, compare both results
    for i in range(input_vectors.shape[0]):
        assert np.isclose(
            np.asarray(scalar_affinities[i]),
            np.asarray(matrix_affinities[i]),
        ), "Affinities from scalar and matrix calls should match."


@pytest.mark.parametrize(
    ("closest_sample_mode", "kernel_cls", "kernel_kwargs"), testdata
)
def test_result_cannot_be_bigger_than_one(
    closest_sample_mode: ClosestSampleModeType,
    kernel_cls: KernelType,
    kernel_kwargs: dict[str, Any] | None,
):
    """Test that the result of calling Samples cannot be bigger than
    1.0."""
    sample_list = [
        Sample(x=[0.5, 1.0]),
        Sample(x=[0.5, 1.0]),
    ]
    samples = Samples(
        sample_list,
        closest_sample_mode=closest_sample_mode,
        kernel_cls=kernel_cls,
        kernel_kwargs=kernel_kwargs,
    )

    result = samples(np.array([0.5, 1.0]))
    assert isinstance(result, jax.Array), "Calling Samples should return a JAX array."
    assert result.ndim == 0, "Calling Samples should return a 0-d JAX array."
    assert np.allclose(np.asarray(result), 1.0), "Result should be 1.0."

    result = samples(np.array([3.0, -7.0]))
    assert isinstance(result, jax.Array), "Calling Samples should return a JAX array."
    assert result.ndim == 0, "Calling Samples should return a 0-d JAX array."
    assert result >= 0.0, "Result should be non-negative."
    assert result <= 1.0, "Result should be less than or equal to 1.0."


@pytest.mark.parametrize(
    ("closest_sample_mode", "kernel_cls", "kernel_kwargs"), testdata
)
def test_samples_get_assigned_correct_closest_sample(
    closest_sample_mode: ClosestSampleModeType,  # noqa: ARG001
    kernel_cls: KernelType,
    kernel_kwargs: dict[str, Any] | None,
):
    """Test that samples get assigned the correct closest sample."""
    sample_list = [
        Sample(x=[0.0, 0.0]),
        Sample(x=[1.0, 1.0]),
        Sample(x=[3.0, 3.0]),
    ]
    samples = Samples(
        sample_list,
        closest_sample_mode="global",
        kernel_cls=kernel_cls,
        kernel_kwargs=kernel_kwargs,
    )

    assert samples.samples[0].closest_sample == [
        sample_list[1],
    ], "Closest sample for first sample is incorrect."
    assert samples.samples[1].closest_sample == [
        sample_list[0],
    ], "Closest sample for second sample is incorrect."
    assert samples.samples[2].closest_sample == [
        sample_list[1],
    ], "Closest sample for third sample is incorrect."


@pytest.mark.parametrize(
    ("closest_sample_mode", "kernel_cls", "kernel_kwargs"), testdata
)
def test_samples_get_assigned_correct_closest_sample_per_dimension(
    closest_sample_mode: ClosestSampleModeType,  # noqa: ARG001
    kernel_cls: KernelType,
    kernel_kwargs: dict[str, Any] | None,
):
    """Test that samples get assigned the correct closest sample per
    dimension."""
    sample_list = [
        Sample(x=[0.0, 0.0]),
        Sample(x=[1.0, 3.0]),
        Sample(x=[6.0, 5.5]),
        Sample(x=[-0.1, 6.5]),
    ]
    samples = Samples(
        sample_list,
        closest_sample_mode="per_dimension",
        kernel_cls=kernel_cls,
        kernel_kwargs=kernel_kwargs,
    )

    assert samples.samples[0].closest_sample == [
        sample_list[3],
        sample_list[1],
    ], "Closest samples for first sample are incorrect."
    assert samples.samples[1].closest_sample == [
        sample_list[0],
        sample_list[2],
    ], "Closest samples for second sample are incorrect."
    assert samples.samples[2].closest_sample == [
        sample_list[1],
        sample_list[3],
    ], "Closest samples for third sample are incorrect."
    assert samples.samples[3].closest_sample == [
        sample_list[0],
        sample_list[2],
    ], "Closest samples for fourth sample are incorrect."


@pytest.mark.parametrize(
    ("closest_sample_mode", "kernel_cls", "kernel_kwargs"), testdata
)
def test_samples_repr_contains_all_info(
    closest_sample_mode: ClosestSampleModeType,
    kernel_cls: KernelType,
    kernel_kwargs: dict[str, Any] | None,
):
    """Test the __repr__ method of the Samples class."""
    sample_list = [
        Sample(x=[0.5, 1.0]),
        Sample(x=[-0.5, 0.0]),
        Sample(x=[1.5, -1.0]),
    ]
    samples = Samples(
        sample_list,
        closest_sample_mode=closest_sample_mode,
        kernel_cls=kernel_cls,
        kernel_kwargs=kernel_kwargs,
    )

    repr_str = repr(samples)
    assert repr_str.startswith("Samples(")
    assert f"samples={samples.samples!r}" in repr_str
    assert repr_str.endswith(")")


@pytest.mark.parametrize(
    ("closest_sample_mode", "kernel_cls", "kernel_kwargs"), testdata
)
def test_samples_repr_allows_eval(
    closest_sample_mode: ClosestSampleModeType,
    kernel_cls: KernelType,
    kernel_kwargs: dict[str, Any] | None,
):
    """Test that Samples' __repr__ allows eval."""
    from numpy import array  # noqa: F401

    from autosafe.kernels.laplacian import LaplacianKernel  # noqa: F401
    from autosafe.kernels.rbf import RBFKernel  # noqa: F401

    sample_list = [
        Sample(x=[0.5, 1.0]),
        Sample(x=[-0.5, 0.0]),
        Sample(x=[1.5, -1.0]),
    ]
    samples = Samples(
        sample_list,
        closest_sample_mode=closest_sample_mode,
        kernel_cls=kernel_cls,
        kernel_kwargs=kernel_kwargs,
    )

    evaluated_samples = eval(repr(samples))
    assert evaluated_samples == samples


@pytest.mark.parametrize(
    ("closest_sample_mode", "kernel_cls", "kernel_kwargs"), testdata
)
def test_samples_str_contains_all_info(
    closest_sample_mode: ClosestSampleModeType,
    kernel_cls: KernelType,
    kernel_kwargs: dict[str, Any] | None,
):
    """Test the __str__ method of the Samples class."""
    sample_list = [
        Sample(x=[1.0, 1.0]),
        Sample(x=[0.0, 0.0]),
        Sample(x=[-1.0, -1.0]),
    ]
    samples = Samples(
        sample_list,
        closest_sample_mode=closest_sample_mode,
        kernel_cls=kernel_cls,
        kernel_kwargs=kernel_kwargs,
    )

    str_repr = str(samples)
    assert str_repr.startswith("Samples based on list of samples:")
    assert str(samples.samples) in str_repr


@pytest.mark.parametrize(
    ("closest_sample_mode", "kernel_cls", "kernel_kwargs"), testdata
)
def test_json_dumps_loads(
    closest_sample_mode: ClosestSampleModeType,
    kernel_cls: KernelType,
    kernel_kwargs: dict[str, Any] | None,
):
    """Test JSON serialization and deserialization of Samples."""
    sample_list = [
        Sample(x=[0.5, 1.0]),
        Sample(x=[-0.5, 0.0]),
        Sample(x=[1.5, -1.0]),
    ]
    samples = Samples(
        sample_list,
        closest_sample_mode=closest_sample_mode,
        kernel_cls=kernel_cls,
        kernel_kwargs=kernel_kwargs,
    )

    json_bytes = msgspec.json.encode(
        samples,
        enc_hook=encode_hook,
    )
    loaded_samples = msgspec.json.decode(
        json_bytes,
        type=Samples,
        dec_hook=decode_hook,
    )

    assert loaded_samples == samples, (
        "Deserialized Samples should be equal to the original."
    )


@pytest.mark.parametrize(
    ("closest_sample_mode", "kernel_cls", "kernel_kwargs"), testdata
)
def test___iter__(
    closest_sample_mode: ClosestSampleModeType,
    kernel_cls: KernelType,
    kernel_kwargs: dict[str, Any] | None,
):
    """Test the __iter__ method."""
    sample_list = [
        Sample(x=[0.5, 1.0]),
        Sample(x=[-0.5, 0.0]),
        Sample(x=[1.5, -1.0]),
    ]
    samples = Samples(
        sample_list,
        closest_sample_mode=closest_sample_mode,
        kernel_cls=kernel_cls,
        kernel_kwargs=kernel_kwargs,
    )

    for i, sample in enumerate(samples):
        assert sample == sample_list[i]


@pytest.mark.parametrize(
    ("closest_sample_mode", "kernel_cls", "kernel_kwargs"), testdata
)
def test___getitem__(
    closest_sample_mode: ClosestSampleModeType,
    kernel_cls: KernelType,
    kernel_kwargs: dict[str, Any] | None,
):
    """Test the __getitem__ method."""
    sample_list = [
        Sample(x=[0.5, 1.0]),
        Sample(x=[-0.5, 0.0]),
        Sample(x=[1.5, -1.0]),
    ]
    samples = Samples(
        samples=sample_list,
        closest_sample_mode=closest_sample_mode,
        kernel_cls=kernel_cls,
        kernel_kwargs=kernel_kwargs,
    )

    assert samples[0] == sample_list[0]
    assert samples[1] == sample_list[1]
    assert samples[2] == sample_list[2]

    with pytest.raises(IndexError):
        _ = samples[3]


@pytest.mark.parametrize(
    ("closest_sample_mode", "kernel_cls", "kernel_kwargs"), testdata
)
def test_find_closest_samples(
    closest_sample_mode: ClosestSampleModeType,
    kernel_cls: KernelType,
    kernel_kwargs: dict[str, Any] | None,
):
    """Test the find_closest_sample method."""
    sample_list = [
        Sample(x=[0.0, 0.0]),
        Sample(x=[1.0, 1.0]),
        Sample(x=[3.0, 3.0]),
    ]
    samples = Samples(
        samples=sample_list,
        closest_sample_mode=closest_sample_mode,
        kernel_cls=kernel_cls,
        kernel_kwargs=kernel_kwargs,
    )

    samples._find_closest_samples()

    # global: 1 closest sample globally
    # per_dimension: D=2 closest samples (one per dim)
    if closest_sample_mode == "global":
        assert samples.samples[0].closest_sample == [sample_list[1]]
        assert samples.samples[1].closest_sample == [sample_list[0]]
        assert samples.samples[2].closest_sample == [sample_list[1]]
    else:
        assert samples.samples[0].closest_sample == [sample_list[1], sample_list[1]]
        assert samples.samples[1].closest_sample == [sample_list[0], sample_list[0]]
        assert samples.samples[2].closest_sample == [sample_list[1], sample_list[1]]


def test_find_closest_vectors_by_index():
    """Test the find_closest_vectors_by_index function."""
    sample_array = np.array([
        [0.0, 0.0],
        [1.0, 1.0],
        [3.0, 3.0],
    ])

    closest_indices = find_closest_vectors_by_index(sample_array)

    expected_indices = np.array([1, 0, 1])
    assert np.array_equal(closest_indices, expected_indices), (
        "Closest vector indices do not match expected values."
    )


def test_find_closest_vectors_by_index_per_dimension():
    """Test the find_closest_vectors_by_index_per_dimension function."""
    sample_array = np.array([
        [0.0, 0.0],
        [1.0, 2.0],
        [6.0, 3.0],
        [-0.1, 7.0],
    ])

    closest_indices_per_dim = find_closest_vectors_by_index_per_dimension(
        sample_array,
    )

    expected_indices_per_dim = np.array([
        [3, 1],
        [0, 2],
        [1, 1],
        [0, 2],
    ]).T
    assert np.array_equal(closest_indices_per_dim, expected_indices_per_dim), (
        "Closest vector indices per dimension do not match expected values."
    )


def test_affinity_batched_matches_loop():
    """Batched (all-diagonal-RBF) path matches per-sample loop.

    N is chosen to NOT be divisible by DEFAULT_ANCHOR_CHUNK (256) so that
    the anchor-padding validity mask is exercised.
    """
    # 259 samples: 256 + 3, so the last anchor tile is partially padded
    rng = np.random.default_rng(42)
    pts = rng.standard_normal((259, 3))
    s = Samples(pts)

    x_pts = rng.standard_normal((20, 3))
    batched = np.asarray(s(x_pts))
    loop = np.array([float(s(x)) for x in x_pts])
    assert np.allclose(batched, loop, atol=1e-10), (
        "Batched affinity must match per-sample loop (anchor mask fix)."
    )


def test_samples_call_empty_samples_1d():
    """Empty Samples returns 0.0 for a 1-D query."""
    s = Samples([])
    result = s(np.array([1.0, 2.0, 3.0]))
    assert math.isclose(float(result), 0.0)


def test_samples_call_empty_samples_matrix():
    """Empty Samples returns zero vector for a 2-D query."""
    s = Samples([])
    pts = np.array([[1.0, 2.0], [3.0, 4.0]])  # shape (2, 2)
    result = s(pts)
    assert result.shape[0] > 0
    assert math.isclose(float(np.asarray(result).sum()), 0.0)


def test_affinity_full_dense_basic():
    """affinity_full_dense with N < anchor_chunk covers full_dense tile and padding else-branch."""
    import jax.numpy as jnp

    from autosafe._affinity import affinity_full_dense

    rng = np.random.default_rng(0)
    n, d = 3, 2
    anchors = jnp.array(rng.standard_normal((n, d)))
    sigma_inv_stack = jnp.broadcast_to(jnp.eye(d), (n, d, d))
    x = jnp.array(rng.standard_normal((5, d)))
    result = affinity_full_dense(anchors, sigma_inv_stack, x)
    assert result.shape == (5,)
    assert bool(jnp.all(result >= 0.0))
    assert bool(jnp.all(result <= 1.0))


def test_get_tile_unknown_variant_raises():
    """_build_tile raises ValueError for unknown variant."""
    from autosafe._affinity import _build_tile

    with pytest.raises(ValueError, match="unknown variant"):
        _build_tile("not_a_real_variant_xyz")
