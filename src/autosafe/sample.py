# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""Class for a single sample in the autoSAFE framework.

A sample is a point in the n-dimensional ODD space, defined as the
vector of ODD parameter values, the kernel function and optionally the
closest sample in the ODD space.
"""

from typing import cast, overload

import numpy as np

from autosafe.kernels.kernel import Kernel
from autosafe.typing import (
    Affinity,
    AffinityVector,
    FloatType,
    Matrix,
    NPMatrix,
    NPVector,
    Vector,
)


class Sample:
    """Class representing a single sample in the autoSAFE framework.

    A sample is a point in the n-dimensional ODD space, defined as the
    vector of ODD parameter values, the kernel function and optionally
    the closest sample in the ODD space.

    Args:
        x (Vector | list[float]): The sample point in the ODD space.
        kernel (Kernel | None): The kernel function used to define the
            sample. Might be None if the kernel is not yet defined.
        closest_sample (list["Sample"] | None): The closest samples in
            the ODD space per dimension, if available; otherwise,
            defaults to None.
    """

    def __init__(
        self,
        x: NPVector | NPMatrix | list[float],
        kernel: Kernel | None = None,
        closest_sample: list["Sample"] | None = None,
    ) -> None:
        self.x = x  # type: ignore[assignment] # <https://github.com/python/mypy/issues/3004>
        self.kernel = kernel
        self.closest_sample = closest_sample

    @property
    def x(self) -> Vector:
        """Get the sample point in the ODD space.

        Returns:
            Vector: The sample point in the ODD space.
        """
        return self.__x

    @x.setter
    def x(self, value: NPVector | NPMatrix | list[float]) -> None:
        """Set the sample point in the ODD space.

        Args:
            value (NPVector | NPMatrix | list[float]): The sample point
                in the ODD space.

        Raises:
            TypeError: If the value is not a numpy array.
            ValueError: If the value is not a one-dimensional array.
        """
        if not isinstance(value, (np.ndarray, list)):
            raise TypeError("x must be a numpy array or a list of floats.")
        if isinstance(value, list):
            value_ = np.array(value, dtype=FloatType)
        else:
            value_ = value.astype(FloatType)
        if value_.ndim != 1:
            raise ValueError("x must be a one-dimensional array.")
        self.__x = cast("Vector", value_)

    @property
    def kernel(self) -> Kernel | None:
        """Get the kernel function of the sample.

        Returns:
            Kernel | None: The kernel function of the sample, or None
                if the sample does not have a defined kernel.
        """
        return self.__kernel

    @kernel.setter
    def kernel(self, value: Kernel | None) -> None:
        """Set the kernel function of the sample.

        Args:
            value (Kernel | None): The kernel function to set for the
                sample. If None, the sample will not have a defined
                kernel.

        Raises:
            TypeError: If the value is not an instance of Kernel or
                None.
            ValueError: If the kernel center x_i does not have the same
                shape as x.
        """
        if value is not None and not isinstance(value, Kernel):
            raise TypeError("Kernel must be an instance of Kernel class or None.")
        if value is not None and value.x_i.shape != self.x.shape:
            raise ValueError("Kernel center x_i must have the same shape as x.")
        self.__kernel = value

    @property
    def closest_sample(self) -> list["Sample"] | None:
        """Get the closest samples in the ODD space per dimension.

        Returns:
            list[Sample] | None: The closest samples in the ODD space
                per dimension, or None if not available.
        """
        return self.__closest_sample

    @closest_sample.setter
    def closest_sample(self, value: list["Sample"] | None) -> None:
        """Set the closest samples in the ODD space per dimension.

        Args:
            value (list[Sample] | None): The closest samples in the
                ODD space per dimension, or None if not available.

        Raises:
            TypeError: If the value is not a list of Sample instances.
        """
        if value is not None and not all(isinstance(s, Sample) for s in value):
            raise TypeError(
                "Closest sample must be a list of Sample instances or None.",
            )
        self.__closest_sample = value

    @overload
    def __call__(self, x: Vector | NPVector) -> Affinity: ...

    @overload
    def __call__(self, x: Matrix | NPMatrix) -> AffinityVector: ...

    def __call__(
        self, x: Vector | Matrix | NPVector | NPMatrix
    ) -> Affinity | AffinityVector:
        """Evaluate the kernel function at vector x.

        Args:
            x (Vector | Matrix | NPVector | NPMatrix): The input vector
                or matrix at which to evaluate the kernel.

        Returns:
            Affinity | AffinityVector: The value of the kernel function
                at x.

        Raises:
            ValueError: If the kernel is not defined for this sample.
        """
        if self.__kernel is None:
            raise ValueError("Kernel is not defined for this sample.")
        return self.__kernel(x)

    def __repr__(self) -> str:
        """Return a string representation of the sample.

        Returns:
            str: A string representation of the sample, including its
                vector, kernel function. We cannot contain the closest
                sample here to avoid recursion issues.
        """
        return f"Sample(x={self.x!r}, kernel={self.__kernel!r})"

    def __str__(self) -> str:
        """Return a string representation of the sample.

        Returns:
            str: A string representation of the sample, including its
                vector, kernel function. We cannot contain the closest
                sample here to avoid recursion issues.
        """
        return f"Sample with x={self.x!s}, kernel={self.__kernel!s}"

    def __eq__(self, value: object) -> bool:
        """Check if two samples are equal.

        Two samples are considered equal if they have the same vector
        and the same kernel function.

        Args:
            value (object): The object to compare with this sample.

        Returns:
            bool: True if the samples are equal, False otherwise.
        """
        return (
            isinstance(value, Sample)
            and self.x.shape == value.x.shape
            and np.allclose(self.x, value.x)
            and self.kernel == value.kernel
        )

    def __hash__(self) -> int:
        """Hash the sample instance.

        Returns:
            int: The hash value of the sample instance.
        """
        x_bytes = self.x.tobytes()
        x_shape = self.x.shape
        kernel_hash = hash(self.kernel) if self.kernel is not None else None
        return hash((x_bytes, x_shape, kernel_hash))
