# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""Implementation of the Laplacian kernel."""

from typing import cast, overload

import jax
import jax.numpy as jnp
import numpy as np

from autosafe import _jax_config  # noqa: F401
from autosafe.kernels.kernel import Kernel
from autosafe.typing import (
    Affinity,
    AffinityVector,
    KernelScaleParam,
    Matrix,
    NPFloatType,
    NPMatrix,
    NPVector,
    Vector,
)


def _validate_alpha(
    x_i: NPVector,
    alpha: KernelScaleParam | None,
) -> NPVector | None:
    """Validate the alpha parameter.

    Args:
        x_i (NPVector): The kernel center point.
        alpha (KernelScaleParam | None): The scaling vector for the
            kernel.

    Returns:
        NPVector: The validated alpha vector.

    Raises:
        ValueError: If alpha is not positive or does not match the
            dimension of x_i.
    """
    if isinstance(alpha, (float, int)):
        if alpha <= 0:
            raise ValueError("alpha must be positive.")
        alpha = cast("NPVector", np.full_like(x_i, alpha, dtype=NPFloatType))
    elif isinstance(alpha, jax.Array):
        alpha = np.asarray(alpha, NPFloatType)
        if alpha.shape != x_i.shape:
            raise ValueError("alpha must have the same dimension as x_i.")
        if not np.all(alpha > 0):
            raise ValueError("alpha must be positive.")
    elif alpha is not None:
        if alpha.shape != x_i.shape:
            raise ValueError("alpha must have the same dimension as x_i.")
        if not np.all(alpha > 0):
            raise ValueError("alpha must be positive.")
        alpha = alpha.astype(NPFloatType)
    return alpha


class LaplacianKernel(Kernel):
    r"""Implementation of the Laplacian kernel.

    The Laplacian kernel is defined as:

    .. math::
        k(x, x_i) = \exp(-\|\alpha\odot(x - x_i)\|_1)

    where :math:`\|x - x_i\|_1` is the L1 norm (Manhattan distance)
    between vectors :math:`x` and :math:`x_i` and :math:`\alpha` is a
    scaling vector.

    Args:
        x_i (NPVector): The kernel center point.
        alpha (KernelScaleParam | None): The scaling vector for the
            kernel.
    """

    def __init__(self, x_i: NPVector, alpha: KernelScaleParam | None = None) -> None:
        self.x_i = np.asarray(x_i, NPFloatType)

        # JAX cache fields
        self._x_i_jax: jax.Array | None = None
        self._alpha_jax: jax.Array | None = None

        self.alpha = _validate_alpha(self.x_i, alpha)

    def _get_x_i_jax(self) -> jax.Array:
        if self._x_i_jax is None:
            self._x_i_jax = jnp.asarray(self.x_i)
        return self._x_i_jax

    def _get_alpha_jax(self) -> jax.Array:
        if self._alpha_jax is None and self.alpha is not None:
            self._alpha_jax = jnp.asarray(self.alpha)
        return cast("jax.Array", self._alpha_jax)

    def update(  # pylint: disable=W0221
        self,
        *,
        x_nn: NPVector | None = None,  # pylint: disable=W0613  # noqa: ARG002
        alpha: KernelScaleParam,
    ) -> None:
        """Update the kernel parameters.

        The Laplacian kernel only has one parameter to update, the
        scaling vector `alpha`. The nearest neighbor(s) `x_nn` are not
        used in the update, but are included for consistency with other
        kernel implementations.

        Args:
            x_nn (NPVector | None): The nearest neighbor(s) to
                the kernel center point. If x_nn is a collection of
                nearest neighbors, the matrix is expected to contain a
                column-wise collection of nearest neighbor coordinates.
            alpha (KernelScaleParam): The new scaling vector for the
                kernel.

        Raises:
            ValueError: If `alpha` is None. While this is not allowed
                per type hints anyway, it is included for clarity as
                None is allowed in the constructor.
        """
        if alpha is None:
            raise ValueError("alpha must be provided for LaplacianKernel.")
        self.alpha = _validate_alpha(self.x_i, alpha)
        # Invalidate JAX cache
        self._x_i_jax = None
        self._alpha_jax = None

    @overload
    def __call__(self, x: Vector | NPVector) -> Affinity: ...

    @overload
    def __call__(self, x: Matrix | NPMatrix) -> AffinityVector: ...

    def __call__(
        self, x: Vector | Matrix | NPVector | NPMatrix
    ) -> Affinity | AffinityVector:
        """Evaluate the Laplacian kernel at vector x.

        Args:
            x (Vector | Matrix | NPVector | NPMatrix): The input vector
                or matrix at which to evaluate the kernel.

        Returns:
            Affinity | AffinityVector: The value of the Laplacian
                kernel at x as a JAX array.

        Raises:
            ValueError: If the kernel parameters have not been set.
        """
        if self.alpha is None:
            raise ValueError("Kernel parameters have not been set.")
        dim = self.x_i.shape[0]
        x_j = jnp.asarray(x)
        x_i_j = self._get_x_i_jax()
        alpha_j = self._get_alpha_jax()

        if x_j.ndim == 1 or x_j.shape == (dim,):
            diff = x_j - x_i_j
            dist = jnp.sum(jnp.abs(alpha_j * diff))
            return jnp.exp(-dist)

        # 2D input: normalise to (n_points, D) using the same R2 rule
        x2 = x_j if x_j.shape[1] == dim else x_j.T  # (n_points, D)
        diff = x2 - x_i_j[None, :]
        dist = jnp.sum(jnp.abs(alpha_j[None, :] * diff), axis=1)
        return jnp.exp(-dist)

    def __eq__(self, value: object) -> bool:
        """Check equality of two LaplacianKernel instances.

        Two LaplacianKernel instances are considered equal if they have
        the same kernel center point and the same scaling vector.

        Args:
            value (object): The object to compare with.

        Returns:
            bool: True if the objects are equal, False otherwise.
        """
        if not isinstance(value, LaplacianKernel):
            return False
        if self.alpha is None or value.alpha is None:
            return self.alpha is value.alpha
        if self.x_i.shape != value.x_i.shape or self.alpha.shape != value.alpha.shape:
            return False
        if not np.allclose(self.alpha, value.alpha):
            return False
        return np.allclose(self.x_i, value.x_i)

    def __hash__(self) -> int:
        """Hash the LaplacianKernel instance.

        Returns:
            int: The hash value of the LaplacianKernel instance.
        """
        x_i_bytes = self.x_i.tobytes()
        x_i_shape = self.x_i.shape
        alpha_bytes = self.alpha.tobytes() if self.alpha is not None else b"None"
        alpha_shape = self.alpha.shape if self.alpha is not None else b"None"
        return hash((x_i_bytes, x_i_shape, alpha_bytes, alpha_shape))

    def __repr__(self) -> str:
        """String representation of the LaplacianKernel.

        Returns:
            str: A string representation of the LaplacianKernel.
        """
        return f"LaplacianKernel(x_i={self.x_i!r}, alpha={self.alpha!r})"

    def __str__(self) -> str:
        """String representation of the LaplacianKernel.

        Returns:
            str: A string representation of the LaplacianKernel.
        """
        return f"LaplacianKernel with x_i={self.x_i!s}, alpha={self.alpha!s}"
