# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""Functions to decode and encode the Samples class."""

from typing import TYPE_CHECKING, Any, cast

from autosafe.sample import Sample
from autosafe.samples import Samples
from autosafe.tools.serializers.sample import decode_sample
from autosafe.typing import ClosestSampleModeType

if TYPE_CHECKING:
    from autosafe.typing import KernelType


def decode_samples(
    obj: object,
) -> Samples:
    """Convert a serialized Samples back to a Samples.

    Args:
        obj (object): The serialized Samples.

    Returns:
        Samples: The reconstructed Samples.
    """
    obj = cast(
        "tuple[list[object], ClosestSampleModeType, KernelType, dict[str, Any]]", obj
    )
    samples = [decode_sample(sample_obj) for sample_obj in obj[0]]
    return Samples(
        samples=samples,
        closest_sample_mode=obj[1],
        kernel_cls=obj[2],
        kernel_kwargs=obj[3],
        skip_updates=True,
    )


def encode_samples(
    samples: Samples,
) -> tuple[list[Sample], ClosestSampleModeType, str | None, dict[str, Any]]:
    """Convert a Samples to a serializable tuple.

    Args:
        samples (Samples): The Samples to convert.

    Returns:
        tuple[
            list[Sample],
            ClosestSampleModeType, str | None,
            dict[str, Any]
        ]: The serialized representation of the Samples.
    """
    kernel_cls: str | None = None
    if samples.kernel_cls.__name__ == "RBFKernel":
        kernel_cls = "RBF"
    elif samples.kernel_cls.__name__ == "LaplacianKernel":
        kernel_cls = "Laplacian"
    return (
        samples.samples,
        samples.closest_sample_mode,
        kernel_cls,
        samples.kernel_kwargs,
    )
