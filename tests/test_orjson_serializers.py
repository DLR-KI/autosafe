# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT


import numpy as np
import pytest

from autosafe.kernels.rbf import RBFKernel
from autosafe.sample import Sample
from autosafe.samples import Samples
from autosafe.tools.serializers.orjson import serializer


def test_autosafe_orjson_serializer_type_error():
    """Test the orjson serializer function with an unsupported type."""
    with pytest.raises(TypeError):
        serializer(42)


def test_autosafe_orjson_serializer_kernel():
    """Test the orjson serializer function with a Kernel object."""
    kernel = RBFKernel(x_i=np.array([1.0, 2.0]), sigma="eye")
    serialized = serializer(kernel)
    assert isinstance(serialized, str)
    assert serialized == repr(kernel)


def test_autosafe_orjson_serializer_sample():
    """Test the orjson serializer function with a Sample object."""
    sample = Sample(
        x=np.array([1.0, 2.0]),
        kernel=RBFKernel(x_i=np.array([1.0, 2.0]), sigma="eye"),
    )
    serialized = serializer(sample)
    assert isinstance(serialized, str)
    assert serialized == repr(sample)


def test_autosafe_orjson_serializer_samples():
    """Test the orjson serializer function with a Samples object."""
    sample1 = Sample(
        x=np.array([1.0, 2.0]),
        kernel=RBFKernel(x_i=np.array([1.0, 2.0]), sigma="eye"),
    )
    sample2 = Sample(
        x=np.array([3.0, 4.0]),
        kernel=RBFKernel(x_i=np.array([3.0, 4.0]), sigma="eye"),
    )
    samples = Samples(samples=[sample1, sample2], kernel_cls="RBF")
    serialized = serializer(samples)
    assert isinstance(serialized, str)
    assert serialized == repr(samples)


def test_autosafe_orjson_serializer_jax_array():
    """Test the orjson serializer function with a jax.Array."""
    import jax.numpy as jnp

    arr = jnp.array([1.0, 2.0, 3.0])
    result = serializer(arr)
    assert result == [1.0, 2.0, 3.0]
