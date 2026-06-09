# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""Abstract base class for kernels in the autoSAFE framework."""

from abc import ABC, abstractmethod
from typing import overload

from autosafe.typing import Affinity, AffinityVector, Matrix, NPMatrix, NPVector, Vector


class Kernel(ABC):
    """Abstract base class for kernels.

    Args:
        x_i (NPVector): The center point of the kernel.
        **kwargs: Additional keyword arguments for kernel
            parameters.

    Attributes:
        x_i (NPVector): The center point of the kernel.
    """

    x_i: NPVector

    @abstractmethod
    def __init__(self, x_i: NPVector, **kwargs: object) -> None: ...

    @abstractmethod
    def update(self, *args, **kwargs) -> None:
        """Update the kernel parameters."""

    @overload
    @abstractmethod
    def __call__(self, x: Vector | NPVector) -> Affinity: ...

    @overload
    @abstractmethod
    def __call__(self, x: Matrix | NPMatrix) -> AffinityVector: ...

    @abstractmethod
    def __call__(
        self, x: Vector | Matrix | NPVector | NPMatrix
    ) -> Affinity | AffinityVector:
        """Evaluate the kernel function at vector x.

        Args:
            x (Vector | Matrix | NPVector | NPMatrix): The input vector
                at which to evaluate the kernel.

        Returns:
            Affinity | AffinityVector: The result of the kernel function
                evaluation.
        """

    @abstractmethod
    def __eq__(self, value: object) -> bool:
        """Check equality of two kernel instances.

        Args:
            value (object): The object to compare with this kernel.

        Returns:
            bool: True if the kernels are equal, False otherwise.
        """

    @abstractmethod
    def __hash__(self) -> int:
        """Hash the kernel instance.

        Returns:
            int: The hash value of the kernel instance.
        """

    @abstractmethod
    def __repr__(self) -> str:
        """String representation of the kernel.

        Returns:
            str: A string representation of the kernel.
        """

    @abstractmethod
    def __str__(self) -> str:
        """String representation of the kernel.

        Returns:
            str: A string representation of the kernel.
        """
