# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""Helper functions for autoSAFE."""

import pathlib

import jax
import numpy as np

from autosafe.kernels.kernel import Kernel
from autosafe.sample import Sample
from autosafe.samples import Samples


def serializer(obj: object) -> str:
    """Serializer for the orjson library for autoSAFE objects.

    Args:
        obj (object): The object to serialize.

    Returns:
        str: The serialized object.

    Raises:
        TypeError: If the object is not serializable by this function.
    """
    if isinstance(obj, Kernel):
        return repr(obj)
    if isinstance(obj, Sample):
        return repr(obj)
    if isinstance(obj, Samples):
        return repr(obj)
    if isinstance(obj, jax.Array):
        return np.asarray(obj).tolist()
    if isinstance(obj, pathlib.Path):
        return str(obj)
    raise TypeError
