# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT

import pathlib

import msgspec.json
import numpy as np

from autosafe.kernels.laplacian import LaplacianKernel
from autosafe.kernels.rbf import RBFKernel
from autosafe.sample import Sample
from autosafe.samples import Samples
from autosafe.tools.serializers.msgspec import (
    decode_hook,
    encode_hook,
)

MSGSPEC_ENCODER = msgspec.json.Encoder(enc_hook=encode_hook)


def test_msgspec_hooks_pathlib():
    """Test the msgspec hooks with a pathlib.Path object."""
    path = pathlib.Path("/some/path/to/file.txt")
    encoded = MSGSPEC_ENCODER.encode(path)
    decoded = msgspec.json.decode(encoded, type=pathlib.Path, dec_hook=decode_hook)
    assert decoded == path


def test_msgspec_hooks_numpy_array():
    """Test the msgspec hooks with a numpy.ndarray object."""
    array = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    encoded = MSGSPEC_ENCODER.encode(array)
    decoded = msgspec.json.decode(
        encoded,
        type=np.ndarray,
        dec_hook=decode_hook,
    )
    assert np.array_equal(decoded, array)
    assert decoded.dtype == array.dtype
    assert decoded.shape == array.shape


def test_msgspec_hooks_rbf_kernel():
    """Test the msgspec hooks with an RBFKernel object."""
    kernel1 = RBFKernel(
        x_i=np.array([1.0, 2.0]),
        sigma="eye",
        kappa=1.0,
        eta=0.5,
    )
    kernel2 = RBFKernel(
        x_i=np.array([3.0, 4.0]),
        sigma=np.array([[1.0, 2.0], [0.0, 1.0]]),
        kappa=np.array([1.0, 2.0]),
        eta=np.array([0.5, 1.5]),
    )
    kernel3 = RBFKernel(
        x_i=np.array([5.0, 6.0]),
        sigma=np.array([[2.0, 0.0], [0.0, 2.0]]),
    )

    for kernel in (kernel1, kernel2, kernel3):
        encoded = MSGSPEC_ENCODER.encode(kernel)
        decoded = msgspec.json.decode(
            encoded,
            type=RBFKernel,
            dec_hook=decode_hook,
        )
        assert decoded == kernel
        assert isinstance(decoded, RBFKernel)


def test_msgspec_hooks_laplacian_kernel():
    """Test the msgspec hooks with a LaplacianKernel object."""
    kernel1 = LaplacianKernel(
        x_i=np.array([1.0, 2.0]),
        alpha=np.array([0.5, 1.5]),
    )
    kernel2 = LaplacianKernel(
        x_i=np.array([3.0, 4.0]),
        alpha=7.0,
    )
    kernel3 = LaplacianKernel(x_i=np.array([5.0, 6.0]))

    for kernel in (kernel1, kernel2, kernel3):
        encoded = MSGSPEC_ENCODER.encode(kernel)
        decoded = msgspec.json.decode(
            encoded,
            type=LaplacianKernel,
            dec_hook=decode_hook,
        )
        assert decoded == kernel
        assert isinstance(decoded, LaplacianKernel)


def test_msgspec_hooks_sample():
    """Test the msgspec hooks with a Sample object."""
    sample1 = Sample(
        x=np.array([1.0, 2.0]),
        kernel=RBFKernel(x_i=np.array([1.0, 2.0]), sigma="eye"),
    )
    sample2 = Sample(
        x=np.array([3.0, 4.0]),
        kernel=LaplacianKernel(x_i=np.array([3.0, 4.0]), alpha=2.0),
    )
    sample3 = Sample(
        x=np.array([5.0, 6.0]),
        kernel=None,
    )

    for sample in (sample1, sample2, sample3):
        encoded = MSGSPEC_ENCODER.encode(sample)
        decoded = msgspec.json.decode(
            encoded,
            type=Sample,
            dec_hook=decode_hook,
        )
        assert decoded == sample
        assert isinstance(decoded, Sample)


def test_msgspec_hooks_samples():
    """Test the msgspec hooks with a Samples object."""
    sample11 = Sample(
        x=np.array([1.0, 2.0]),
        kernel=RBFKernel(x_i=np.array([3.0, 4.0]), sigma="eye"),
    )
    sample12 = Sample(
        x=np.array([3.0, 4.0]),
        kernel=RBFKernel(
            x_i=np.array([5.0, 6.0]), sigma=np.array([[1.0, 1.0], [0.0, 0.1]])
        ),
    )
    sample21 = Sample(
        x=np.array([3.0, 4.0]),
        kernel=LaplacianKernel(x_i=np.array([3.0, 4.0]), alpha=2.0),
    )
    sample22 = Sample(
        x=np.array([5.0, 6.0]),
        kernel=LaplacianKernel(x_i=np.array([5.0, 6.0]), alpha=np.array([1.0, 0.5])),
    )
    samples1 = Samples(
        samples=[sample11, sample12],
        closest_sample_mode="global",
        kernel_cls="RBF",
        kernel_kwargs={"sigma": "eye"},
    )
    samples2 = Samples(
        samples=[sample21, sample22],
        closest_sample_mode="per_dimension",
        kernel_cls="Laplacian",
        kernel_kwargs={"alpha": 1.0},
    )

    for samples in (samples1, samples2):
        encoded = MSGSPEC_ENCODER.encode(samples)
        decoded = msgspec.json.decode(
            encoded,
            type=Samples,
            dec_hook=decode_hook,
        )
        assert decoded == samples
        assert isinstance(decoded, Samples)


def test_msgspec_encode_hook_jax_array():
    """Test encode_hook with a jax.Array (covers the jax.Array branch)."""
    from typing import cast

    import jax.numpy as jnp

    arr = jnp.array([1.0, 2.0])
    result = cast("dict[str, object]", encode_hook(arr))
    assert isinstance(result, dict)
    assert "__ndarray__" in result
    assert result["dtype"] == "float64"
    assert result["__ndarray__"] == [1.0, 2.0]
