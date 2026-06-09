# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""Helper functions for autoSAFE."""

import pathlib

import jax
import numpy as np

from autosafe.kernels.laplacian import LaplacianKernel
from autosafe.kernels.rbf import RBFKernel
from autosafe.sample import Sample
from autosafe.samples import Samples
from autosafe.tools.serializers.kernels import (
    decode_laplacian_kernel,
    decode_rbf_kernel,
    encode_laplacian_kernel,
    encode_rbf_kernel,
)
from autosafe.tools.serializers.np_array import (
    decode_np_array,
    encode_np_array,
)
from autosafe.tools.serializers.path import decode_path, encode_path
from autosafe.tools.serializers.sample import decode_sample, encode_sample
from autosafe.tools.serializers.samples import decode_samples, encode_samples


def decode_hook(expected_type: type, obj: object) -> object:  # noqa: PLR0911
    """Decode serialized representations back to autoSAFE objects.

    Args:
        expected_type (type): The expected type of the object to decode.
        obj (object): The serialized representation of the object.

    Returns:
        object: The decoded autoSAFE object.
    """
    if expected_type is pathlib.Path:
        return decode_path(obj)
    if expected_type is np.ndarray:
        return decode_np_array(obj)
    if expected_type is RBFKernel:
        return decode_rbf_kernel(obj)
    if expected_type is LaplacianKernel:
        return decode_laplacian_kernel(obj)
    if expected_type is Sample:
        return decode_sample(obj)
    if expected_type is Samples:
        return decode_samples(obj)
    return obj  # pragma: no cover


def encode_hook(obj: object) -> object:  # noqa: PLR0911
    """Encode autoSAFE objects to serializable representations.

    Args:
        obj (object): The object to encode.

    Returns:
        object: The encoded representation of the object.

    Raises:
        NotImplementedError: If the object type is not supported.
    """
    if isinstance(obj, RBFKernel):
        return encode_rbf_kernel(obj)
    if isinstance(obj, LaplacianKernel):
        return encode_laplacian_kernel(obj)
    if isinstance(obj, Sample):
        return encode_sample(obj)
    if isinstance(obj, Samples):
        return encode_samples(obj)
    if isinstance(obj, np.ndarray):
        return encode_np_array(obj)
    if isinstance(obj, jax.Array):
        return encode_np_array(np.asarray(obj))
    if isinstance(obj, pathlib.Path):
        return encode_path(obj)
    raise NotImplementedError(
        f"{obj} of type {type(obj)} is not supported in {encode_hook.__name__}"
    )  # pragma: no cover
