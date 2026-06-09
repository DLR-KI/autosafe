# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""Import functions to turn data into a autoSAFE ODD."""

from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import msgspec.json
import numpy.typing as npt
import polars as pl

from autosafe.samples import Samples
from autosafe.tools.serializers.msgspec import decode_hook
from autosafe.typing import ClosestSampleModeType, FloatType, KernelType

if TYPE_CHECKING:
    from autosafe.typing import Matrix


def from_csv(
    file: Path | str,
    closest_sample_mode: ClosestSampleModeType = "global",
    kernel_cls: KernelType = "RBF",
    kernel_kwargs: dict[str, Any] | None = None,
) -> Samples:
    """Import from a CSV file.

    This function imports sample data from a CSV file.

    Args:
        file (Path | str): Path to the CSV file.
        closest_sample_mode (ClosestSampleModeType): The closest sample
            mode to use.
        kernel_cls (KernelType): The kernel type to use.
        kernel_kwargs (dict[str, Any]): The kernel parameters.

    Returns:
        Samples: The imported samples.
    """
    return from_polars(
        data=pl.read_csv(file),
        closest_sample_mode=closest_sample_mode,
        kernel_cls=kernel_cls,
        kernel_kwargs=kernel_kwargs,
    )


def from_polars(
    data: pl.DataFrame,
    closest_sample_mode: ClosestSampleModeType = "global",
    kernel_cls: KernelType = "RBF",
    kernel_kwargs: dict[str, Any] | None = None,
) -> Samples:
    """Import from a Polars DataFrame.

    This function imports sample data from a Polars DataFrame.

    Args:
        data (pl.DataFrame): The Polars DataFrame containing the
            samples.
        closest_sample_mode (ClosestSampleModeType): The closest sample
            mode to use.
        kernel_cls (KernelType): The kernel type to use.
        kernel_kwargs (dict[str, Any]): The kernel parameters.

    Returns:
        Samples: The imported samples.
    """
    return from_numpy(
        data=data.to_numpy(),
        closest_sample_mode=closest_sample_mode,
        kernel_cls=kernel_cls,
        kernel_kwargs=kernel_kwargs,
    )


def from_numpy(
    data: npt.NDArray,
    closest_sample_mode: ClosestSampleModeType = "global",
    kernel_cls: KernelType = "RBF",
    kernel_kwargs: dict[str, Any] | None = None,
) -> Samples:
    """Import from a NumPy array.

    This function imports sample data from a NumPy array.

    Args:
        data (npt.NDArray): The NumPy array containing the samples.
        closest_sample_mode (ClosestSampleModeType): The closest sample
            mode to use.
        kernel_cls (KernelType): The kernel type to use.
        kernel_kwargs (dict[str, Any]): The kernel parameters.

    Returns:
        Samples: The imported samples.
    """
    samples_array = cast("Matrix", data.astype(FloatType))

    return Samples(
        samples=samples_array,
        closest_sample_mode=closest_sample_mode,
        kernel_cls=kernel_cls,
        kernel_kwargs=kernel_kwargs,
    )


def from_json(file: Path | str) -> Samples:
    """Import from a JSON file.

    This function imports a Samples object from a JSON file dumped
    with msgspec.

    Args:
        file (Path | str): Path to the JSON file.

    Returns:
        Samples: The imported samples.
    """
    return msgspec.json.decode(
        Path(file).read_text(encoding="utf-8"),
        type=Samples,
        dec_hook=decode_hook,
    )
