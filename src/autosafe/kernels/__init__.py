# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""Kernel imports for the autoSAFE framework."""

from autosafe.kernels.kernel import Kernel
from autosafe.kernels.laplacian import LaplacianKernel
from autosafe.kernels.rbf import RBFKernel
from autosafe.typing import KernelType

KernelDict: dict[KernelType, type[Kernel]] = {
    "RBF": RBFKernel,
    "Laplacian": LaplacianKernel,
}
