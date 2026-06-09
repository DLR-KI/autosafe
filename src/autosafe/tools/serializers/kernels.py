# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""Functions to decode and encode kernels."""

from typing import cast

from autosafe.kernels.laplacian import LaplacianKernel
from autosafe.kernels.rbf import RBFKernel
from autosafe.tools.serializers.np_array import (
    decode_np_array,
)
from autosafe.typing import KernelScaleParam, NPSquareMatrix, NPVector


def decode_rbf_kernel(
    obj: object,
) -> RBFKernel:
    """Convert a serialized RBFKernel back to an RBFKernel.

    Args:
        obj (object): The serialized RBFKernel.

    Returns:
        RBFKernel: The reconstructed RBFKernel.
    """
    obj = cast("tuple[str, object, object, dict | float, dict | float]", obj)
    x_i = decode_np_array(obj[1])
    sigma = decode_np_array(obj[2])
    kappa = decode_np_array(obj[3]) if isinstance(obj[3], dict) else obj[3]
    eta = decode_np_array(obj[4]) if isinstance(obj[4], dict) else obj[4]
    return RBFKernel(x_i=x_i, sigma=sigma, kappa=kappa, eta=eta)


def decode_laplacian_kernel(
    obj: object,
) -> LaplacianKernel:
    """Convert a serialized LaplacianKernel back to a LaplacianKernel.

    Args:
        obj (object): The serialized LaplacianKernel.

    Returns:
        LaplacianKernel: The reconstructed LaplacianKernel.
    """
    obj = cast("tuple[str, object, dict | float]", obj)
    x_i = decode_np_array(obj[1])
    alpha = decode_np_array(obj[2]) if isinstance(obj[2], dict) else obj[2]
    return LaplacianKernel(x_i=x_i, alpha=alpha)


def encode_rbf_kernel(
    kernel: RBFKernel,
) -> tuple[str, NPVector, NPSquareMatrix | None, KernelScaleParam, KernelScaleParam]:
    """Convert an RBFKernel to a serializable tuple.

    Args:
        kernel (RBFKernel): The RBFKernel to convert.

    Returns:
        tuple[
            str,
            NPVector,
            NPSquareMatrix | None,
            KernelScaleParam,
            KernelScaleParam
        ]: The serialized representation of the RBFKernel.
    """
    return ("RBF", kernel.x_i, kernel.sigma, kernel.kappa, kernel.eta)


def encode_laplacian_kernel(
    kernel: LaplacianKernel,
) -> tuple[str, NPVector, NPVector | None]:
    """Convert a LaplacianKernel to a serializable tuple.

    Args:
        kernel (LaplacianKernel): The LaplacianKernel to convert.

    Returns:
        tuple[str, NPVector, NPVector | None]: The serialized
            representation of the LaplacianKernel.
    """
    return ("Laplacian", kernel.x_i, kernel.alpha)
