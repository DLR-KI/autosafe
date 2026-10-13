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

from autosafe import _jax_config  # ruff:ignore[unused-import]
from autosafe.exceptions import (
    ArrayDtypeError,
    DeprecatedAliasConflictError,
    KernelParametersNotSetError,
    MissingParameterError,
    NonFiniteParameterError,
    NonPositiveParameterError,
    NotPositiveSemidefiniteError,
    ParameterOrderError,
    ParameterTypeError,
    ShapeMismatchError,
    SigmaNotInvertibleError,
    TooFewAnchorsError,
    UnknownOptionError,
    ZeroNearestNeighborDistancesError,
)
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

# Relative lower bound for the sigma law :
# sigma_kk = (kappa - lam) * exp(-eta * d) + lam  # ruff:ignore[commented-out-code]
# with lam = SIGMA_FLOOR_RATIO * kappa.
# lam is defined relative to kappa so the law stays scale-equivariant.
SIGMA_FLOOR_RATIO = float(np.exp(-10.0))

#: Name of the anchor-count schedule s_N = sqrt(ln N) for the width
#: multiple, the schedule of the paper's calibrated convergence theorem
#: and the default of every calibration entry point.
WIDTH_SCHEDULE_SQRT_LOG_N: Literal["sqrt_log_n"] = "sqrt_log_n"

#: A width multiple: a fixed positive number or the name of a schedule.
WidthMultiple = float | Literal["sqrt_log_n"]

#: A covariance specification: a matrix, ``"eye"``, or a scalar
#: multiple of the identity.
SigmaSpec = NPSquareMatrix | Literal["eye"] | NPFloatType | float


def resolve_width_multiple(s: float | str, n_anchors: int) -> float:
    """Return the numeric width multiple for an anchor count.

    A fixed ``s`` is returned unchanged. The schedule
    ``"sqrt_log_n"`` evaluates s_N = sqrt(ln N), which grows without
    bound while s_N * d_tilde_N still shrinks, as the calibrated
    convergence theorem requires. ``N`` is the number of anchors that
    instantiate kernels, i.e. after de-duplication and subsampling.

    Args:
        s (float | str): Fixed width multiple or schedule name.
        n_anchors (int): Number of anchors that instantiate kernels.

    Returns:
        float: The width multiple to use for ``n_anchors`` anchors.

    Raises:
        UnknownOptionError: If ``s`` is a string but not a known
            schedule.
        TooFewAnchorsError: If the schedule gets fewer than 2 anchors.
        ParameterTypeError: If ``s`` is neither a number nor a string.
        NonFiniteParameterError: If ``s`` is not finite.
        NonPositiveParameterError: If ``s`` is not positive.
    """
    if isinstance(s, str):
        if s != WIDTH_SCHEDULE_SQRT_LOG_N:
            raise UnknownOptionError("s", s, (WIDTH_SCHEDULE_SQRT_LOG_N,))
        if n_anchors < 2:  # ruff:ignore[magic-value-comparison]
            raise TooFewAnchorsError(2, n_anchors)
        return float(np.sqrt(np.log(n_anchors)))
    if isinstance(s, bool) or not isinstance(s, (int, float, np.floating)):
        raise ParameterTypeError("s", s, (float, str))
    s_value = float(s)
    if not np.isfinite(s_value):
        raise NonFiniteParameterError("s")
    if s_value <= 0.0:
        raise NonPositiveParameterError("s", s_value)
    return s_value


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
        param (NPVector | NPFloatType | float): The
            parameter to validate and broadcast.
        name (str): The name of the parameter for error messages.
        dim (int): The expected dimension of the parameter.

    Returns:
        NPVector: The validated and broadcasted parameter as a vector.

    Raises:
        ShapeMismatchError: If the parameter is a vector that does not
            match the expected dimension.
        ParameterTypeError: If the parameter is not a vector or a
            scalar.
    """
    # 0-d arrays (NumPy or JAX) are treated as scalars
    if isinstance(param, (np.ndarray, jax.Array)) and np.asarray(param).ndim == 0:
        param = float(param)
    if isinstance(param, (NPFloatType, float, int)):
        return cast("NPVector", NPFloatType(param) * np.ones(dim, dtype=NPFloatType))
    if isinstance(param, (np.ndarray, jax.Array)):
        param_np = _to_np(param)
        if param_np.shape != (dim,):
            raise ShapeMismatchError(name, param_np.shape, (dim,))
        return cast("NPVector", param_np)
    raise ParameterTypeError(name, param, (float, np.ndarray))


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


def _invert_sigma(sigma: NPSquareMatrix) -> tuple[NPSquareMatrix, NPSquareMatrix]:
    """Invert sigma, repairing it first if the inverse is not finite.

    Args:
        sigma (NPSquareMatrix): The sigma matrix to invert.

    Returns:
        tuple[NPSquareMatrix, NPSquareMatrix]: The (possibly repaired)
            sigma matrix and its inverse.

    Raises:
        SigmaNotInvertibleError: If the inverse is still not finite
            after the repair.
    """
    sigma_inv = np.asarray(jnp.linalg.inv(jnp.asarray(sigma)), NPFloatType)
    if np.isfinite(sigma_inv).all():
        return sigma, sigma_inv
    sigma = _fix_sigma_matrix(sigma)
    sigma_inv = np.asarray(jnp.linalg.inv(jnp.asarray(sigma)), NPFloatType)
    if not np.isfinite(sigma_inv).all():
        raise SigmaNotInvertibleError(sigma.shape[0])
    warnings.warn(
        message=(
            "sigma matrix was not invertible and has been adjusted "
            "to be positive definite."
        ),
        category=UserWarning,
        stacklevel=3,
    )
    return sigma, sigma_inv


class RBFKernel(Kernel):
    r"""Implementation of the Radial Basis Function (RBF) kernel.

    The Radial Basis Function (RBF) kernel, also known as Gaussian
    kernel is defined as:

    .. math::
        k(x, x_i) = \exp(-\frac{1}{2} (x - x_i)^\top \Sigma^{-1} (x - x_i))

    where :math:`\Sigma` is the positive semidefinite (psd) free
    parameter matrix.

    A sigma without a finite inverse is repaired with a warning; if
    the repair fails,
    :class:`~autosafe.exceptions.SigmaNotInvertibleError` is raised.

    Args:
        x_i (NPVector): The kernel center point, also called anchor
            point.
        sigma (SigmaSpec | None): The free parameter matrix for the
            kernel, defaults to the identity matrix.
        x_nn (NPVector | NPMatrix | None): The nearest neighbor(s) to
            the kernel center point. If x_nn is a collection of nearest
            neighbors, the matrix is expected to contain a column-wise
            collection of nearest neighbor coordinates.
        kappa (KernelScaleParam, optional): Scaling factor for the
            kernel, defaults to 1.0.
        eta (KernelScaleParam, optional): Exponential decay factor for
            the kernel, defaults to 1.0.
        lam (KernelScaleParam | None, optional): Affine lower bound
            for sigma. If None, uses SIGMA_FLOOR_RATIO * kappa.

    Attributes:
        sigma_is_diagonal (bool): Whether sigma is set and
            (numerically) diagonal.

    Raises:
        UnknownOptionError: If `sigma` is a string other than
            ``"eye"``.
        NotPositiveSemidefiniteError: If `sigma` is not psd.
        ShapeMismatchError: If `sigma` does not match the dimension of
            `x_i`.
        ParameterTypeError: If `sigma` is not one of the expected
            types (`"eye"`, a square matrix, a scalar, or `None`).
    """  # ruff:ignore[doc-line-too-long]

    def __init__(  # ruff:ignore[too-many-arguments, too-many-positional-arguments]
        self,
        x_i: NPVector,
        sigma: SigmaSpec | None = None,
        x_nn: NPVector | NPMatrix | None = None,
        kappa: KernelScaleParam = 1.0,
        eta: KernelScaleParam = 1.0,
        lam: KernelScaleParam | None = None,
    ) -> None:
        self.x_i = _to_np(x_i)

        self.kappa = kappa
        self.eta = eta
        self.lam = lam

        # Initialize JAX cache fields before any sigma branch.
        self._sigma_is_diagonal: bool = False
        self._x_i_jax: jax.Array | None = None
        self._sigma_inv_diag_jax: jax.Array | None = None
        self._sigma_inv_full_jax: jax.Array | None = None

        if sigma is None:
            self.sigma = None
            self.sigma_inv = None
            if x_nn is not None:
                self.update(x_nn=x_nn, kappa=kappa, eta=eta, lam=lam)
        elif isinstance(sigma, str) and sigma == "eye":
            self.sigma = np.eye(len(self.x_i), dtype=NPFloatType)
            self.sigma_inv = np.asarray(
                jnp.linalg.inv(jnp.asarray(self.sigma)), NPFloatType
            )
            self._refresh_sigma_cache()
        elif isinstance(sigma, str):
            raise UnknownOptionError("sigma", sigma, ("eye",))
        elif isinstance(sigma, (NPFloatType, float, int)):
            sigma_scalar = NPFloatType(sigma)
            if sigma_scalar <= 0:
                raise NotPositiveSemidefiniteError("sigma")
            self.sigma, self.sigma_inv = _invert_sigma(
                np.eye(len(self.x_i), dtype=NPFloatType) * sigma_scalar
            )
            self._refresh_sigma_cache()
        elif isinstance(sigma, (np.ndarray, jax.Array)):
            sigma_np = _to_np(sigma)
            if sigma_np.shape != (self.x_i.shape[0], self.x_i.shape[0]):
                raise ShapeMismatchError(
                    "sigma", sigma_np.shape, (self.x_i.shape[0], self.x_i.shape[0])
                )
            if not np.all(
                np.asarray(jnp.linalg.eigvals(jnp.asarray(sigma_np)).real) > 0
            ):
                raise NotPositiveSemidefiniteError("sigma")
            self.sigma, self.sigma_inv = _invert_sigma(sigma_np.astype(NPFloatType))
            self._refresh_sigma_cache()
        else:
            raise ParameterTypeError("sigma", sigma, (str, np.ndarray, float))

    def _refresh_sigma_cache(self) -> None:
        """Recompute diagonal flag and invalidate cached JAX arrays.

        Raises:
            KernelParametersNotSetError: If called before sigma has
                been assigned.
        """
        if self.sigma is None:
            raise KernelParametersNotSetError("sigma")
        self._sigma_is_diagonal = bool(
            np.allclose(self.sigma, np.diag(np.diag(self.sigma)))
        )
        self._x_i_jax = None
        self._sigma_inv_diag_jax = None
        self._sigma_inv_full_jax = None

    @property
    def sigma_is_diagonal(self) -> bool:
        """Whether sigma is (numerically) diagonal.

        Returns:
            bool: True if sigma is set and has no off-diagonal entries.
        """
        return self._sigma_is_diagonal

    def _get_x_i_jax(self) -> jax.Array:
        if self._x_i_jax is None:
            self._x_i_jax = jnp.asarray(self.x_i)
        return self._x_i_jax

    def _get_sigma_inv_diag_jax(self) -> jax.Array:
        if self._sigma_inv_diag_jax is None:
            if self.sigma_inv is None:
                raise KernelParametersNotSetError("sigma_inv")
            self._sigma_inv_diag_jax = jnp.asarray(np.diag(self.sigma_inv))
        return self._sigma_inv_diag_jax

    def _get_sigma_inv_full_jax(self) -> jax.Array:
        if self._sigma_inv_full_jax is None:
            if self.sigma_inv is None:
                raise KernelParametersNotSetError("sigma_inv")
            self._sigma_inv_full_jax = jnp.asarray(self.sigma_inv)
        return self._sigma_inv_full_jax

    def update(  # pylint: disable=W0221  # ruff:ignore[complex-structure, too-many-branches]
        self,
        *,
        x_nn: Vector | Matrix | NPVector | NPMatrix | None = None,
        sigma: SigmaSpec | None = None,
        kappa: KernelScaleParam | None = None,
        eta: KernelScaleParam | None = None,
        lam: KernelScaleParam | None = None,
    ) -> None:
        r"""Set the free parameter matrix (sigma) of the kernel.

        The sigma matrix :math:`\Sigma` can either be set directly or
        computed based on the nearest neighbor points :math:`x_{nn}`.

        If `sigma` is provided, it must be a square matrix with the
        same dimension as `x_i`. If `sigma` is not provided, it will be
        computed using the nearest neighbor points and the parameters
        `kappa` and `eta`. If `kappa` or `eta` are provided, they will
        overwrite the existing values.

        A sigma without a finite inverse is repaired with a warning. If
        the repair fails,
        :class:`~autosafe.exceptions.SigmaNotInvertibleError` is raised
        and the kernel's sigma is left unchanged.

        Args:
            x_nn (Vector | Matrix | NPVector | NPMatrix | None): The
                nearest neighbor(s) to the kernel center point. If x_nn
                is a collection of nearest neighbors, the matrix is
                expected to contain a column-wise collection of
                coordinates.
            sigma (SigmaSpec | None): The free parameter matrix for the
                kernel.
            kappa (KernelScaleParam | None): Scaling factor for the
                kernel, defaults to 1.0.
            eta (KernelScaleParam | None): Exponential decay factor
                for the kernel, defaults to 1.0.
            lam (KernelScaleParam | None, optional): Affine lower bound
                for sigma. If None, uses SIGMA_FLOOR_RATIO * kappa.

        Raises:
            UnknownOptionError: If `sigma` is a string other than
                ``"eye"``.
            ParameterTypeError: If `sigma` is not a string, numpy
                array or scalar, or if `x_nn` is not a numpy array.
            ShapeMismatchError: If `sigma` or `x_nn` does not have
                dimensions compatible with `x_i`.
            NotPositiveSemidefiniteError: If `sigma` is not positive
                semidefinite (psd).
            MissingParameterError: If neither `sigma` nor `x_nn` is
                provided.
        """
        if kappa is not None:
            self.kappa = kappa
        if eta is not None:
            self.eta = eta
        if lam is not None:
            self.lam = lam

        if sigma is not None:
            if isinstance(sigma, str) and sigma == "eye":
                sigma = np.eye(self.x_i.shape[0], dtype=NPFloatType)
            elif isinstance(sigma, str):
                raise UnknownOptionError("sigma", sigma, ("eye",))
            elif isinstance(sigma, (NPFloatType, float, int)):
                sigma_scalar = NPFloatType(sigma)
                if sigma_scalar <= 0:
                    raise NotPositiveSemidefiniteError("sigma")
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
                    raise ShapeMismatchError(
                        "sigma", sigma.shape, (self.x_i.shape[0], self.x_i.shape[0])
                    )
                if not np.all(
                    np.asarray(jnp.linalg.eigvals(jnp.asarray(sigma)).real) > 0
                ):
                    raise NotPositiveSemidefiniteError("sigma")
            else:
                raise ParameterTypeError("sigma", sigma, (str, np.ndarray, float))
            sigma = np.asarray(sigma, NPFloatType)
        else:
            if x_nn is None:
                raise MissingParameterError("x_nn", unless="sigma")
            if not isinstance(x_nn, (np.ndarray, jax.Array)):
                raise ParameterTypeError("x_nn", x_nn, np.ndarray)
            if isinstance(x_nn, np.ndarray) and x_nn.dtype != NPFloatType:
                warnings.warn(
                    message="x_nn is not of dtype FloatType. Converting to FloatType.",
                    category=UserWarning,
                    stacklevel=2,
                )
            x_nn = _to_np(x_nn)
            if x_nn.shape[0] != self.x_i.shape[0]:
                raise ShapeMismatchError(
                    "x_nn", x_nn.shape, (self.x_i.shape[0], *x_nn.shape[1:])
                )
            sigma = self._sigma_ii(x_nn, self.kappa, self.eta, self.lam)

        self.sigma, self.sigma_inv = _invert_sigma(np.asarray(sigma, NPFloatType))
        self._refresh_sigma_cache()

    def _sigma_ii(
        self,
        x_nn: NPVector | NPMatrix,
        kappa: KernelScaleParam = 1.0,
        eta: KernelScaleParam = 1.0,
        lam: KernelScaleParam | None = None,
    ) -> NPSquareMatrix:
        r"""Calculates the diagonal elements of the sigma matrix.

        The diagonal elements of the sigma matrix are computed based on
        the distance between the kernel center point and the nearest
        neighbor point using the affine-floor law:

        .. math::
            \sigma_{ii} = (\kappa - \lambda) \cdot \exp(-\eta \cdot d)
            + \lambda

        where :math:`\lambda = \text{SIGMA\_FLOOR\_RATIO} \cdot \kappa`
        when unset, :math:`d` is the distance between the kernel center
        point and the nearest neighbor point, :math:`\kappa` is a
        scaling factor, and :math:`\eta` is an exponential decay factor.
        The lower bound :math:`\lambda` keeps :math:`\Sigma` positive
        definite for isolated anchors.

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
            lam (KernelScaleParam | None, optional): Affine lower bound.
                If None, uses SIGMA_FLOOR_RATIO * kappa per dimension.

        Returns:
            NPSquareMatrix: The diagonal elements of the sigma matrix.

        Raises:
            ParameterTypeError: If `x_nn` is not a numpy array.
            ArrayDtypeError: If the dtype of `x_nn` does not match
                `NPFloatType`.
            ShapeMismatchError: If `x_nn` does not have as many rows as
                `x_i`.
            NonPositiveParameterError: If `lam` is not positive.
            ParameterOrderError: If `lam` >= `kappa` for any dimension.
        """
        if not isinstance(x_nn, (np.ndarray, jax.Array)):
            raise ParameterTypeError("x_nn", x_nn, np.ndarray)
        if isinstance(x_nn, np.ndarray) and x_nn.dtype != NPFloatType:
            raise ArrayDtypeError("x_nn", x_nn.dtype, np.dtype(NPFloatType))
        x_nn = _to_np(x_nn)

        kappa_ = _validate_and_broadcast_param(kappa, "kappa", self.x_i.shape[0])
        eta_ = _validate_and_broadcast_param(eta, "eta", self.x_i.shape[0])

        if x_nn.shape[0] != self.x_i.shape[0]:
            raise ShapeMismatchError(
                "x_nn", x_nn.shape, (self.x_i.shape[0], *x_nn.shape[1:])
            )

        dim = self.x_i.shape[0]

        if lam is None:
            lam_ = SIGMA_FLOOR_RATIO * kappa_
        else:
            lam_ = _validate_and_broadcast_param(lam, "lam", dim)
            if not np.all(lam_ > 0.0):
                raise NonPositiveParameterError("lam", lam_)
        if not np.all(lam_ < kappa_):
            raise ParameterOrderError("lam", "kappa")

        kappa_col = kappa_.reshape((dim, 1))
        eta_col = eta_.reshape((dim, 1))
        lam_col = lam_.reshape((dim, 1))

        if x_nn.ndim == 2:  # ruff:ignore[magic-value-comparison]
            # Per-dimension mode: x_nn is (dim, dim) where column j is
            # the nearest neighbor found in dimension j.  Only the
            # diagonal d[i, i] = neighbor_i[i] - x_i[i] drives
            # sigma[i, i].
            d_full = x_nn - np.tile(self.x_i.reshape(-1, 1), (1, dim))
            d_diag = np.diag(d_full).reshape(dim, 1)
            sigma_diag = np.squeeze(
                (kappa_col - lam_col) * np.exp(-eta_col * np.abs(d_diag)) + lam_col
            )
        else:
            # Global mode: single nearest neighbor vector (dim,) or
            # (dim, 1).
            d = x_nn.reshape((dim, -1)) - self.x_i.reshape((dim, 1))
            sigma_diag = np.mean(
                (kappa_col - lam_col) * np.exp(-eta_col * np.abs(d)) + lam_col,
                axis=1,
            )

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
            x (Matrix | Vector | NPMatrix | NPVector): The input vector
                or matrix at which to evaluate the kernel.

        Returns:
            Affinity | AffinityVector: The value of the RBF kernel at
                x as a JAX array (0-d scalar or (n_points,) vector).

        Raises:
            KernelParametersNotSetError: If the kernel has no free
                parameter matrix (sigma) defined.
        """
        if self.sigma is None or self.sigma_inv is None:
            raise KernelParametersNotSetError("sigma")
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


def calibrate_rbf_scale_median(
    d_nn: NPMatrix,
    *,
    gamma: float = 1.0,
    s: WidthMultiple = WIDTH_SCHEDULE_SQRT_LOG_N,
    c: float | None = None,
) -> tuple[NPVector, NPVector]:
    """Derive scale-invariant kappa, eta from NN distances.

    Implements the calibration rule eta_j = gamma / median_i(d_ij) and
    kappa_j = (s * median_i(d_ij))**2, where d_ij is the per-dimension
    nearest-neighbor distance of anchor i in dimension j. See
    docs/bandwidth-calibration.md for the derivation and the proof of
    scale invariance.

    Dimensions whose median distance is zero (e.g. discrete variables
    with duplicated values) are floored to the median over the
    non-degenerate dimensions so that sigma stays positive definite.

    Args:
        d_nn (NPMatrix): Per-dimension nearest-neighbor distances,
            shape (n_anchors, n_dims), non-negative.
        gamma (float): Decay constant; eta_j = gamma / median_j.
        s (WidthMultiple): Width multiple or schedule name (see
            ``resolve_width_multiple``, N = n_anchors); kappa_j =
            (s * median_j)**2.
        c (float | None): Deprecated alias for ``gamma``.

    Returns:
        tuple[NPVector, NPVector]: (kappa, eta) arrays of shape
            (n_dims,).

    Raises:
        DeprecatedAliasConflictError: If gamma and c disagree.
        NonPositiveParameterError: If gamma is not positive.
        ShapeMismatchError: If d_nn is not 2D.
        ZeroNearestNeighborDistancesError: If all per-dimension
            medians are zero.
    """
    if c is not None:
        gamma_is_default = bool(np.isclose(gamma, 1.0))
        gamma_matches_alias = bool(np.isclose(gamma, c))
        if not gamma_is_default and not gamma_matches_alias:
            raise DeprecatedAliasConflictError("gamma", "c")
        warnings.warn(
            "'c' is deprecated for RBF calibration; use 'gamma'",
            DeprecationWarning,
            stacklevel=2,
        )
        gamma = c
    if gamma <= 0.0:
        raise NonPositiveParameterError("gamma", gamma)

    d = np.asarray(d_nn, dtype=NPFloatType)
    if d.ndim != 2:  # ruff:ignore[magic-value-comparison]
        raise ShapeMismatchError("d_nn", d.shape, (None, None))
    s_value = resolve_width_multiple(s, d.shape[0])
    med = np.median(d, axis=0)
    positive = med > 0.0
    if not positive.any():
        raise ZeroNearestNeighborDistancesError(per_dimension=True)
    floor = float(np.median(med[positive]))
    med = np.where(positive, med, floor)
    kappa = (s_value * med) ** 2
    eta = gamma / med
    return kappa.astype(NPFloatType), eta.astype(NPFloatType)


def calibrate_rbf_scale_d_tilde(
    d_nn_l2: NPVector,
    *,
    gamma: float = 1.0,
    s: WidthMultiple = WIDTH_SCHEDULE_SQRT_LOG_N,
    c: float | None = None,
) -> tuple[float, float]:
    """Derive scalar (kappa, eta) from full-space NN distances.

    Implements eta = gamma / d_tilde and kappa = (s * d_tilde)**2 where
    d_tilde is the median of the positive full-space (L2) nearest-
    neighbor distances. Exact-duplicate anchors (distance 0) are
    excluded from the median. See docs/bandwidth-calibration.md.

    Args:
        d_nn_l2 (NPVector): Full-space NN distances, shape (n_anchors,),
            non-negative.
        gamma (float): Decay constant; eta = gamma / d_tilde.
        s (WidthMultiple): Width multiple or schedule name (see
            ``resolve_width_multiple``, N = n_anchors); kappa =
            (s * d_tilde)**2.
        c (float | None): Deprecated alias for ``gamma``.

    Returns:
        tuple[float, float]: (kappa, eta) scalars.

    Raises:
        DeprecatedAliasConflictError: If gamma and c disagree.
        NonPositiveParameterError: If gamma is not positive.
        ShapeMismatchError: If d_nn_l2 is not 1D.
        ZeroNearestNeighborDistancesError: If d_nn_l2 has no positive
            entries.
    """
    if c is not None:
        gamma_is_default = bool(np.isclose(gamma, 1.0))
        gamma_matches_alias = bool(np.isclose(gamma, c))
        if not gamma_is_default and not gamma_matches_alias:
            raise DeprecatedAliasConflictError("gamma", "c")
        warnings.warn(
            "'c' is deprecated for RBF calibration; use 'gamma'",
            DeprecationWarning,
            stacklevel=2,
        )
        gamma = c
    if gamma <= 0.0:
        raise NonPositiveParameterError("gamma", gamma)

    d = np.asarray(d_nn_l2, dtype=NPFloatType)
    if d.ndim != 1:
        raise ShapeMismatchError("d_nn_l2", d.shape, (None,))
    s_value = resolve_width_multiple(s, d.shape[0])
    positive = d[d > 0.0]
    if positive.size == 0:
        raise ZeroNearestNeighborDistancesError(per_dimension=False)
    d_tilde = float(np.median(positive))
    return float((s_value * d_tilde) ** 2), float(gamma / d_tilde)


GaussianKernel = RBFKernel
