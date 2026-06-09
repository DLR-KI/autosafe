# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""Implementation of the Radial Basis Function (RBF) kernel."""

import warnings
from typing import Literal, cast, overload

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt

from autosafe import _jax_config  # noqa: F401
from autosafe.kernels.kernel import Kernel
from autosafe.typing import (
    Affinity,
    AffinityVector,
    KernelScaleParam,
    Matrix,
    NPFloatType,
    NPMatrix,
    NPSquareMatrix,
    NPVector,
    Vector,
)


def _to_np(a: object) -> npt.NDArray[np.float64]:
    """Convert array-like (including jax.Array) to NumPy float64.

    Args:
        a (object): The array-like object to convert.

    Returns:
        npt.NDArray[np.float64]: Input as a NumPy float64 array.
    """
    return np.asarray(a, NPFloatType)


def _validate_and_broadcast_param(
    param: NPVector | NPFloatType | float,
    name: str,
    dim: int,
) -> NPVector:
    """Validate and broadcast a parameter to the correct dimension.

    This function checks if the parameter is a vector or a scalar and
    ensures it has the correct dimension. If it is a scalar, it
    broadcasts it to a vector of the specified dimension.

    Args:
        param (npt.NDArray | NPVector | NPFloatType | float): The
            parameter to validate and broadcast.
        name (str): The name of the parameter for error messages.
        dim (int): The expected dimension of the parameter.

    Returns:
        NPVector: The validated and broadcasted parameter as a vector.

    Raises:
        ValueError: If the parameter is not a vector or a scalar, or if
            it does not match the expected dimension.
    """
    # 0-d arrays (NumPy or JAX) are treated as scalars
    if isinstance(param, (np.ndarray, jax.Array)) and np.asarray(param).ndim == 0:
        param = float(param)
    if isinstance(param, (NPFloatType, float, int)):
        return cast("NPVector", NPFloatType(param) * np.ones(dim, dtype=NPFloatType))
    if isinstance(param, (np.ndarray, jax.Array)):
        param_np = _to_np(param)
        if param_np.shape != (dim,):
            raise ValueError(f"{name} must have the same dimension as x_i.")
        return cast("NPVector", param_np)
    raise ValueError(f"{name} must be a vector or a scalar.")


def _fix_sigma_matrix(sigma: NPSquareMatrix) -> NPSquareMatrix:
    """Ensure the sigma matrix is positive definite.

    This function checks the eigenvalues of the sigma matrix and
    adjusts any non-positive eigenvalues to a small positive value to
    ensure the matrix is positive definite.

    Args:
        sigma (NPSquareMatrix): The sigma matrix to check and adjust.

    Returns:
        NPSquareMatrix: The adjusted sigma matrix that is positive
            definite.
    """
    # Use jnp.linalg.eigh; results converted back to numpy for storage.
    sigma_j = jnp.asarray(sigma)
    eigvals, eigvecs = jnp.linalg.eigh(sigma_j)
    eigvals_fixed = jnp.maximum(eigvals, float(np.finfo(NPFloatType).eps))
    sigma_fixed = eigvecs @ jnp.diag(eigvals_fixed) @ eigvecs.T
    return cast("NPSquareMatrix", np.asarray(sigma_fixed, NPFloatType))


class RBFKernel(Kernel):
    r"""Implementation of the Radial Basis Function (RBF) kernel.

    The Radial Basis Function (RBF) kernel, also known as Gaussian
    kernel is defined as:

    .. math::
        k(x, x_i) = \exp(-\frac{1}{2} (x - x_i)^\top \Sigma^{-1} (x - x_i))

    where :math:`\Sigma` is the positive semidefinite (psd) free
    parameter matrix.

    Args:
        x_i (NPVector): The kernel center point, also called anchor point.
        sigma (NPSquareMatrix | Literal["eye"] | None): The free parameter
            matrix for the kernel, defaults to the identity matrix.
        x_nn (NPVector | NPMatrix | None): The nearest neighbor(s) to
            the kernel center point. If x_nn is a collection of nearest
            neighbors, the matrix is expected to contain a column-wise
            collection of nearest neighbor coordinates.
        kappa (KernelScaleParam, optional): Scaling factor for the
            kernel, defaults to 1.0.
        eta (KernelScaleParam, optional): Exponential decay factor for
            the kernel, defaults to 1.0.

    Raises:
        ValueError: If `sigma` is not psd or does not match the
            dimension of `x_i`. Or if `sigma` is not one of the
            expected types (`"eye"`, a square matrix, or `None`).
    """  # noqa: W505

    def __init__(
        self,
        x_i: NPVector,
        sigma: NPSquareMatrix | Literal["eye"] | NPFloatType | float | None = None,
        x_nn: NPVector | NPMatrix | None = None,
        kappa: KernelScaleParam = 1.0,
        eta: KernelScaleParam = 1.0,
    ) -> None:
        self.x_i = _to_np(x_i)

        self.kappa = kappa
        self.eta = eta

        # Initialise JAX cache fields before any sigma branch.
        self._sigma_is_diagonal: bool = False
        self._x_i_jax: jax.Array | None = None
        self._sigma_inv_diag_jax: jax.Array | None = None
        self._sigma_inv_full_jax: jax.Array | None = None

        if sigma is None:
            self.sigma = None
            self.sigma_inv = None
            if x_nn is not None:
                self.update(x_nn=x_nn, kappa=kappa, eta=eta)
        elif isinstance(sigma, str) and sigma == "eye":
            self.sigma = np.eye(len(self.x_i), dtype=NPFloatType)
            self.sigma_inv = np.asarray(
                jnp.linalg.inv(jnp.asarray(self.sigma)), NPFloatType
            )
            self._refresh_sigma_cache()
        elif isinstance(sigma, (NPFloatType, float, int)):
            sigma_scalar = NPFloatType(sigma)
            if sigma_scalar <= 0:
                raise ValueError("sigma must be positive semidefinite (psd).")
            self.sigma = np.eye(len(self.x_i), dtype=NPFloatType) * sigma_scalar
            self.sigma_inv = np.asarray(
                jnp.linalg.inv(jnp.asarray(self.sigma)), NPFloatType
            )
            self._refresh_sigma_cache()
        elif isinstance(sigma, (np.ndarray, jax.Array)):
            sigma_np = _to_np(sigma)
            if sigma_np.shape != (self.x_i.shape[0], self.x_i.shape[0]):
                raise ValueError("sigma must have compatible dimensions to x_i.")
            if not np.all(
                np.asarray(jnp.linalg.eigvals(jnp.asarray(sigma_np)).real) > 0
            ):
                raise ValueError("sigma must be positive semidefinite (psd).")
            self.sigma = sigma_np.astype(NPFloatType)
            self.sigma_inv = np.asarray(
                jnp.linalg.inv(jnp.asarray(self.sigma)), NPFloatType
            )
            self._refresh_sigma_cache()
        else:
            raise ValueError("sigma must be either 'eye', a square matrix, or None.")

    def _refresh_sigma_cache(self) -> None:
        """Recompute diagonal flag and invalidate cached JAX arrays.

        Raises:
            RuntimeError: If called before sigma has been assigned.
        """
        if self.sigma is None:
            raise RuntimeError("_refresh_sigma_cache called before sigma is set")
        self._sigma_is_diagonal = bool(
            np.allclose(self.sigma, np.diag(np.diag(self.sigma)))
        )
        self._x_i_jax = None
        self._sigma_inv_diag_jax = None
        self._sigma_inv_full_jax = None

    def _get_x_i_jax(self) -> jax.Array:
        if self._x_i_jax is None:
            self._x_i_jax = jnp.asarray(self.x_i)
        return self._x_i_jax

    def _get_sigma_inv_diag_jax(self) -> jax.Array:
        if self._sigma_inv_diag_jax is None:
            if self.sigma_inv is None:
                raise RuntimeError("sigma_inv has not been set")
            self._sigma_inv_diag_jax = jnp.asarray(np.diag(self.sigma_inv))
        return self._sigma_inv_diag_jax

    def _get_sigma_inv_full_jax(self) -> jax.Array:
        if self._sigma_inv_full_jax is None:
            if self.sigma_inv is None:
                raise RuntimeError("sigma_inv has not been set")
            self._sigma_inv_full_jax = jnp.asarray(self.sigma_inv)
        return self._sigma_inv_full_jax

    def update(  # pylint: disable=W0221  # noqa: C901,PLR0912
        self,
        *,
        x_nn: Vector | Matrix | NPVector | NPMatrix | None = None,
        sigma: NPSquareMatrix | Literal["eye"] | NPFloatType | float | None = None,
        kappa: KernelScaleParam | None = None,
        eta: KernelScaleParam | None = None,
    ) -> None:
        r"""Set the free parameter matrix (sigma) of the kernel.

        The sigma matrix :math:`\Sigma` can either be set directly or
        computed based on the nearest neighbor points :math:`x_{nn}`.

        If `sigma` is provided, it must be a square matrix with the
        same dimension as `x_i`. If `sigma` is not provided, it will be
        computed using the nearest neighbor points and the parameters
        `kappa` and `eta`. If `kappa` or `eta` are provided, they will
        overwrite the existing values.

        Args:
            x_nn (Vector | Matrix | NPVector | NPMatrix | None): The
                nearest neighbor(s) to the kernel center point. If x_nn
                is a collection of nearest neighbors, the matrix is
                expected to contain a column-wise collection of
                coordinates.
            sigma (NPSquareMatrix | Literal["eye"] | None): The free
                parameter matrix for the kernel.
            kappa (KernelScaleParam, optional): Scaling factor for the
                kernel, defaults to 1.0.
            eta (KernelScaleParam, optional): Exponential decay factor
                for the kernel, defaults to 1.0.

        Raises:
            ValueError: If `sigma` is not a numpy array, if it does not
                have compatible dimensions to `x_i`, or if it is not
                positive semidefinite (psd). Also raised if `x_nn` is
                not provided when `sigma` is not set, or if `x_nn` is
                not a numpy array or does not match the expected dtype
                and dimensions.
        """
        if kappa is not None:
            self.kappa = kappa
        if eta is not None:
            self.eta = eta

        if sigma is not None:
            if isinstance(sigma, str) and sigma == "eye":
                sigma = cast(
                    "NPSquareMatrix", np.eye(self.x_i.shape[0], dtype=NPFloatType)
                )
            elif isinstance(sigma, str):
                raise ValueError(
                    "sigma must be either 'eye', a square matrix, or None.",
                )
            elif isinstance(sigma, (NPFloatType, float, int)):
                sigma_scalar = NPFloatType(sigma)
                if sigma_scalar <= 0:
                    raise ValueError("sigma must be positive semidefinite (psd).")
                sigma = cast(
                    "NPSquareMatrix",
                    np.eye(self.x_i.shape[0], dtype=NPFloatType) * sigma_scalar,
                )
            elif isinstance(sigma, (np.ndarray, jax.Array)):
                if isinstance(sigma, np.ndarray) and sigma.dtype != NPFloatType:
                    warnings.warn(
                        message=(
                            "sigma is not of dtype FloatType. Converting to FloatType."
                        ),
                        category=UserWarning,
                        stacklevel=2,
                    )
                sigma = _to_np(sigma)
                if sigma.shape != (self.x_i.shape[0], self.x_i.shape[0]):
                    raise ValueError("sigma must have compatible dimensions to x_i.")
                if not np.all(
                    np.asarray(jnp.linalg.eigvals(jnp.asarray(sigma)).real) > 0
                ):
                    raise ValueError("sigma must be positive semidefinite (psd).")
            else:
                raise ValueError("sigma must be a numpy array.")
            sigma = np.asarray(sigma, NPFloatType)
        else:
            if x_nn is None:
                raise ValueError("x_nn must be provided if sigma is not set.")
            if not isinstance(x_nn, (np.ndarray, jax.Array)):
                raise ValueError("x_nn must be a numpy array.")
            if isinstance(x_nn, np.ndarray) and x_nn.dtype != NPFloatType:
                warnings.warn(
                    message="x_nn is not of dtype FloatType. Converting to FloatType.",
                    category=UserWarning,
                    stacklevel=2,
                )
            x_nn = _to_np(x_nn)
            if x_nn.shape[0] != self.x_i.shape[0]:
                raise ValueError("x_nn must have the same number of rows as x_i.")
            sigma = self._sigma_ii(x_nn, self.kappa, self.eta)

        self.sigma = np.asarray(sigma, NPFloatType)
        sigma_inv_j = jnp.linalg.inv(jnp.asarray(self.sigma))
        sigma_inv = np.asarray(sigma_inv_j, NPFloatType)
        if not np.isfinite(sigma_inv).all():
            self.sigma = _fix_sigma_matrix(self.sigma)
            sigma_inv = np.asarray(jnp.linalg.inv(jnp.asarray(self.sigma)), NPFloatType)
            warnings.warn(
                message=(
                    "sigma matrix was not invertible and has been adjusted "
                    "to be positive definite."
                ),
                category=UserWarning,
                stacklevel=2,
            )
        self.sigma_inv = sigma_inv
        self._refresh_sigma_cache()

    def _sigma_ii(
        self,
        x_nn: NPVector | NPMatrix,
        kappa: KernelScaleParam = 1.0,
        eta: KernelScaleParam = 1.0,
    ) -> NPSquareMatrix:
        r"""Calculates the diagonal elements of the sigma matrix.

        The diagonal elements of the sigma matrix are computed based on
        the distance between the kernel center point and the nearest
        neighbor point.

        .. math::
            \sigma_{ii} = \kappa \cdot \exp(-\eta \cdot d)

        where :math:`d` is the distance between the kernel center point
        and the nearest neighbor point, :math:`kappa` is a scaling
        factor, and :math:`eta` is an exponential decay factor.

        Args:
            x_nn (NPVector | NPMatrix): The nearest neighbor(s) to the
                kernel center point. If x_nn is a collection of nearest
                neighbors, the matrix is expected to contain a
                column-wise collection of nearest neighbor
                coordinates.
            kappa (KernelScaleParam, optional): Scaling factor for the
                kernel, defaults to 1.0.
            eta (KernelScaleParam, optional): Exponential decay factor
                for the kernel, defaults to 1.0.

        Returns:
            NPSquareMatrix: The diagonal elements of the sigma matrix.

        Raises:
            ValueError: If `x_nn` is not a numpy array or if the dtype
                does not match `NPFloatType`.
        """
        if not isinstance(x_nn, (np.ndarray, jax.Array)):
            raise ValueError("x_nn must be a numpy array.")
        if isinstance(x_nn, np.ndarray) and x_nn.dtype != NPFloatType:
            raise ValueError("x_nn must be of dtype FloatType.")
        x_nn = _to_np(x_nn)

        kappa = _validate_and_broadcast_param(kappa, "kappa", self.x_i.shape[0])
        eta = _validate_and_broadcast_param(eta, "eta", self.x_i.shape[0])

        if x_nn.shape[0] != self.x_i.shape[0]:
            raise ValueError("x_nn must have the same number of rows as x_i.")

        dim = self.x_i.shape[0]
        kappa_ = kappa.reshape((dim, 1))
        eta_ = eta.reshape((dim, 1))

        if x_nn.ndim == 2:  # noqa: PLR2004
            # Per-dimension mode: x_nn is (dim, dim) where column j is
            # the nearest neighbor found in dimension j.  Only the
            # diagonal d[i, i] = neighbour_i[i] - x_i[i] drives
            # sigma[i, i].
            d_full = x_nn - np.tile(self.x_i.reshape(-1, 1), (1, dim))
            d_diag = np.diag(d_full).reshape(dim, 1)
            sigma_diag = np.squeeze(kappa_ * np.exp(-eta_ * np.abs(d_diag)))
        else:
            # Global mode: single nearest neighbour vector (dim,) or
            # (dim, 1).
            d = x_nn.reshape((dim, -1)) - self.x_i.reshape((dim, 1))
            sigma_diag = np.mean(kappa_ * np.exp(-eta_ * np.abs(d)), axis=1)

        return cast("NPSquareMatrix", np.diag(sigma_diag))

    @overload
    def __call__(self, x: Vector | NPVector) -> Affinity: ...

    @overload
    def __call__(self, x: Matrix | NPMatrix) -> AffinityVector: ...

    def __call__(
        self, x: Matrix | Vector | NPMatrix | NPVector
    ) -> Affinity | AffinityVector:
        """Evaluate the Radial Basis Function (RBF) kernel at vector x.

        Args:
            x (Vector | NPVector | Matrix | NPMatrix): The input vector
                or matrix at which to evaluate the kernel.

        Returns:
            Affinity | AffinityVector: The value of the RBF kernel at
                x as a JAX array (0-d scalar or (n_points,) vector).

        Raises:
            ValueError: If the kernel has no free parameter matrix
                (sigma) defined.
        """
        if self.sigma is None or self.sigma_inv is None:
            raise ValueError("Kernel has no free parameter matrix (sigma) defined.")
        dim = self.x_i.shape[0]
        x_j = jnp.asarray(x)
        x_i_j = self._get_x_i_jax()

        if x_j.ndim == 1 or x_j.shape == (dim,):  # single point -> 0-d
            diff = x_j - x_i_j
            if self._sigma_is_diagonal:
                s = self._get_sigma_inv_diag_jax()
                mahal = jnp.maximum(jnp.sum(diff * diff * s), 0.0)
            else:
                si = self._get_sigma_inv_full_jax()
                mahal = jnp.maximum(diff @ si @ diff, 0.0)
            return jnp.exp(-0.5 * mahal)

        x2 = x_j if x_j.shape[1] == dim else x_j.T  # (n_points, D)
        diff = x2 - x_i_j[None, :]
        if self._sigma_is_diagonal:
            s = self._get_sigma_inv_diag_jax()
            mahal = jnp.maximum(jnp.sum(diff * diff * s[None, :], axis=1), 0.0)
        else:
            si = self._get_sigma_inv_full_jax()
            mahal = jnp.maximum(jnp.einsum("md,de,me->m", diff, si, diff), 0.0)
        return jnp.exp(-0.5 * mahal)

    def __eq__(self, value: object) -> bool:
        """Check equality of two RBFKernel instances.

        Two RBFKernel instances are considered equal if they have the
        same kernel center point and the same free parameter matrix.

        Args:
            value (object): The object to compare with this RBFKernel.

        Returns:
            bool: True if the RBFKernel instances are equal, False
                otherwise.
        """
        if not isinstance(value, RBFKernel):
            return False
        if self.x_i.shape != value.x_i.shape:
            return False
        if not np.allclose(self.x_i, value.x_i):
            return False
        if self.sigma is None or value.sigma is None:
            return bool(self.sigma is None and value.sigma is None)
        if self.sigma.shape != value.sigma.shape:
            return False
        return np.allclose(self.sigma, value.sigma)

    def __hash__(self) -> int:
        """Hash the RBFKernel instance.

        Returns:
            int: The hash value of the RBFKernel instance.
        """
        x_i_bytes = self.x_i.tobytes()
        x_i_shape = self.x_i.shape
        sigma_bytes = self.sigma.tobytes() if self.sigma is not None else b"None"
        sigma_shape = self.sigma.shape if self.sigma is not None else b"None"
        return hash((x_i_bytes, x_i_shape, sigma_bytes, sigma_shape))

    def __repr__(self) -> str:
        """String representation of the RBFKernel.

        Returns:
            str: A string representation of the RBFKernel.
        """
        return f"RBFKernel(x_i={self.x_i!r}, sigma={self.sigma!r})"

    def __str__(self) -> str:
        """String representation of the RBFKernel.

        Returns:
            str: A string representation of the RBFKernel.
        """
        return f"RBFKernel with x_i={self.x_i!s}, sigma={self.sigma!s}"


GaussianKernel = RBFKernel
