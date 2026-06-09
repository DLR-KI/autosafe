# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""Functions to decode and encode the Sample class."""

from typing import Any, cast

import numpy as np

from autosafe.kernels.kernel import Kernel
from autosafe.sample import Sample
from autosafe.tools.serializers.kernels import (
    decode_laplacian_kernel,
    decode_rbf_kernel,
)
from autosafe.tools.serializers.np_array import decode_np_array
from autosafe.typing import Vector


def decode_sample(
    obj: object,
) -> Sample:
    """Convert a serialized Sample back to a Sample.

    Args:
        obj (object): The serialized Sample.

    Returns:
        Sample: The reconstructed Sample.
    """
    obj = cast("tuple[object, tuple[Any, ...] | None]", obj)

    x_raw = decode_np_array(obj[0]) if isinstance(obj[0], dict) else obj[0]
    x: list[float] | np.ndarray = (
        cast("list[float]", x_raw)
        if isinstance(x_raw, list)
        else np.asarray(x_raw, dtype=float)
    )
    kernel: Kernel | None = None
    if obj[1] is not None:
        if obj[1][0] == "RBF":
            kernel = decode_rbf_kernel(obj[1])
        elif obj[1][0] == "Laplacian":
            kernel = decode_laplacian_kernel(obj[1])
    return Sample(x=x, kernel=kernel)


def encode_sample(sample: Sample) -> tuple[Vector, Kernel | None]:
    """Convert a Sample to a serializable tuple.

    Args:
        sample (Sample): The Sample to convert.

    Returns:
        tuple[Vector, Kernel | None]: The serialized representation of
            the Sample.
    """
    return (sample.x, sample.kernel)
