# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""Functions to decode and encode numpy arrays."""

from typing import Any, TypedDict, cast

import numpy as np
import numpy.typing as npt


class NumpyArrayDict(TypedDict):
    """Dictionary representation of a numpy array for serialization.

    Attributes:
        __ndarray__ (list[float]): The flattened array data.
        dtype (str): The data type of the array.
    """

    __ndarray__: list[float]
    dtype: str


def decode_np_array(obj: object) -> npt.NDArray[Any]:
    """Convert a object back to a numpy array.

    Args:
        obj (object): The dictionary representation of the numpy
            array.

    Returns:
        np.ndarray: The reconstructed numpy array.
    """
    array_obj = cast("NumpyArrayDict", obj)
    return np.array(array_obj["__ndarray__"], dtype=array_obj["dtype"])


def encode_np_array(array: npt.NDArray[Any]) -> NumpyArrayDict:
    """Convert a numpy array to a NumpyArrayDict.

    Args:
        array (np.ndarray): The numpy array to convert.

    Returns:
        NumpyArrayDict: The dictionary representation of the numpy
            array.
    """
    return {"__ndarray__": array.tolist(), "dtype": str(array.dtype)}
