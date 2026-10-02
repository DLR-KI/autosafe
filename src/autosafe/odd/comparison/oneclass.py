# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""One-Class SVM and SVDD ODD boundary comparison methods.

``SVDDBoundary`` implements the Tax--Duin Support Vector Data
Description dual directly (Tax & Duin, 2004) rather than aliasing it to
``sklearn.svm.OneClassSVM``. For a kernel with constant diagonal --- the
RBF kernel, where ``k(x, x) = 1`` for every ``x`` --- the SVDD dual and
the one-class SVM dual are the *same* quadratic program (Scholkopf et
al., 2001): both maximize/minimize ``alpha^T K alpha`` subject to
``0 <= alpha_i <= C`` and ``sum_i alpha_i = 1``, so an RBF-SVDD row
would duplicate the ``OneClassSVMBoundary`` row exactly. SVDD is a
genuinely distinct baseline only with a non-constant-diagonal kernel
(e.g. polynomial). See ``SVDDBoundary`` for the runtime warning this
implies.
"""

import warnings
from typing import Literal, TypedDict, cast

import numpy as np
import numpy.typing as npt
import scipy.optimize
from sklearn.svm import OneClassSVM
from typing_extensions import Self

from autosafe.odd.comparison.base import (
    DecisionBoundary,
    ODDBoundaryMethod,
)
from autosafe.typing import FloatType, Matrix, NPFloatType, NPMatrix, NPVector, Vector

DEFAULT_NU = 0.05
DEFAULT_GAMMA: float | Literal["scale"] = "scale"
DEFAULT_THRESHOLD = NPFloatType(0.0)

# Small internal grid mirroring the "ocsvm" branch of
# experiments/benchmark/run_baseline_comparison.py::_score: select the
# (nu, gamma) combination with the largest mean decision-function
# margin on the reference (in-distribution) points themselves.
_NU_GRID = (0.01, 0.05, 0.1)
_GAMMA_GRID: tuple[float | Literal["scale"], ...] = ("scale", 1.0, 10.0)

SVDDKernel = Literal["rbf", "poly", "linear"]


class _SVDDKernelKwargs(TypedDict):
    """Per-key kernel arguments for the module kernel helpers.

    A ``TypedDict`` rather than ``dict[str, float | int]`` so that
    splatting it preserves each key's own type; a plain union
    widens ``degree`` to ``int | float`` and fails the type check.
    """

    gamma: float
    degree: int
    coef0: float


DEFAULT_SVDD_KERNEL: SVDDKernel = "rbf"
DEFAULT_DEGREE = 3
DEFAULT_COEF0 = 0.0
_FREE_SV_TOLERANCE = 1e-6
_QP_TOLERANCE = 1e-12
_QP_MAX_ITER = 1000


def _resolve_rbf_gamma(gamma: float | Literal["scale"], data: NPMatrix) -> float:
    """Resolve a scalar RBF gamma, matching sklearn's ``"scale"`` rule.

    Args:
        gamma (float | Literal["scale"]): Fixed gamma value, or
            ``"scale"`` for ``1 / (n_features * X.var())``, matching
            ``sklearn.svm.OneClassSVM``'s convention exactly so
            ``SVDDBoundary`` and ``OneClassSVMBoundary`` build the same
            kernel matrix under matched hyperparameters.
        data (NPMatrix): Reference points, shape (n_samples,
            n_features).

    Returns:
        float: Resolved scalar gamma.
    """
    if gamma != "scale":
        return float(gamma)
    n_features = data.shape[1]
    variance = float(np.var(data))
    if variance <= 0.0:
        return 1.0
    return 1.0 / (n_features * variance)


def _select_ocsvm_hyperparameters(
    data_t: NPMatrix,
) -> tuple[float | Literal["scale"], float, OneClassSVM]:
    """Select (gamma, nu) by best mean margin on the reference points.

    Mirrors the ``"ocsvm"`` branch of
    ``experiments/benchmark/run_baseline_comparison.py::_score``
    branch verbatim: for each (nu, gamma) combination in the small
    internal grid, fit ``OneClassSVM`` and keep the one maximizing the
    mean decision-function value on the fitting data itself.

    Args:
        data_t (NPMatrix): Reference points, shape (n_samples,
            n_features).

    Returns:
        tuple[float | Literal["scale"], float, OneClassSVM]: Selected
        gamma, selected nu, and the fitted estimator.

    Raises:
        RuntimeError: If no candidate could be fit (unreachable given
            a non-empty static grid, kept only to satisfy the type
            checker that ``best`` is not ``None`` below).
    """
    best: tuple[float | Literal["scale"], float, OneClassSVM] | None = None
    best_margin = -np.inf
    for nu in _NU_GRID:
        for gamma in _GAMMA_GRID:
            svm = OneClassSVM(kernel="rbf", nu=nu, gamma=gamma).fit(data_t)
            margin = float(np.mean(svm.decision_function(data_t)))
            if margin > best_margin:
                best_margin = margin
                best = (gamma, nu, svm)
    if best is None:
        raise RuntimeError("hyperparameter grid search produced no candidate")
    return best


class OneClassSVMBoundary(ODDBoundaryMethod):
    """One-Class SVM ODD boundary with an RBF kernel.

    Score is ``sklearn.svm.OneClassSVM.decision_function``; membership
    is the ``score >= threshold`` superlevel set (``threshold = 0.0``
    by default, matching scikit-learn's own convention). The raw score
    is exposed via :meth:`decision_function` so the same threshold-
    sweep machinery used elsewhere in the comparison framework applies
    without modification.

    Attributes:
        gamma (float | Literal["scale"]): RBF kernel gamma.
        nu (float): Upper bound on the fraction of margin errors /
            lower bound on the fraction of support vectors.
        auto_select (bool): If True, ``fit`` selects (gamma, nu) from
            a small internal grid by best mean margin on the reference
            points, mirroring the ``ocsvm`` branch of
            ``run_baseline_comparison.py::_score``,
            instead of using the constructor's fixed values.
        threshold (FloatType): Decision threshold in score space.
        svm (OneClassSVM | None): Fitted estimator.
        method_type (str): Method identifier property.
        decision_boundary (DecisionBoundary): Decision-boundary
            metadata property.
    """

    def __init__(
        self,
        gamma: float | Literal["scale"] = DEFAULT_GAMMA,
        nu: float = DEFAULT_NU,
        *,
        auto_select: bool = False,
        threshold: FloatType | None = None,
    ) -> None:
        """Initialize the one-class SVM boundary.

        Args:
            gamma (float | Literal["scale"]): RBF kernel gamma
                (default ``"scale"``, matching scikit-learn).
            nu (float): Nu parameter (default 0.05).
            auto_select (bool): Select (gamma, nu) from the internal
                grid during ``fit`` instead of using the fixed values
                above.
            threshold (FloatType | None): Decision threshold in score
                space; defaults to 0.0.
        """
        self.gamma = gamma
        self.nu = nu
        self.auto_select = auto_select
        self.threshold = threshold if threshold is not None else DEFAULT_THRESHOLD
        self.svm: OneClassSVM | None = None
        self.ref_points: Matrix | NPMatrix | None = None
        self.trained = False

    @property
    def method_type(self) -> str:
        """method_type property for the one-class SVM method.

        Returns:
            str: The method name.
        """
        return "oneclass_svm"

    @property
    def decision_boundary(self) -> DecisionBoundary:
        """decision_boundary property for the one-class SVM monitor.

        Returns:
            DecisionBoundary: Structured decision-boundary metadata.
        """
        if not self.trained or self.svm is None:
            return DecisionBoundary(
                type="oneclass_svm",
                parameters={"gamma": self.gamma, "nu": self.nu},
                coverage={},
                conservatism=None,
            )

        return DecisionBoundary(
            type="oneclass_svm",
            parameters={
                "gamma": self.svm.gamma,
                "nu": self.svm.nu,
                "auto_select": self.auto_select,
                "threshold": float(self.threshold),
            },
            coverage=self._estimate_coverage(),
            conservatism=self._calculate_conservatism(),
        )

    def _estimate_coverage(self) -> dict[str, FloatType]:
        """Estimate coverage metrics for this boundary.

        Returns:
            dict[str, FloatType]: Coverage metrics for the fitted
                boundary.
        """
        if not self.trained or self.ref_points is None:
            return {}
        train_scores = self.decision_function(self.ref_points)
        above_threshold = (train_scores >= self.threshold).mean()
        return {
            "coverage_ratio": NPFloatType(above_threshold),
            "n_above_threshold": NPFloatType((train_scores >= self.threshold).sum()),
        }

    def _calculate_conservatism(self) -> FloatType:
        """Calculate conservatism for this boundary.

        Returns:
            FloatType: Conservatism score in the range [0, 1].
        """
        if not self.trained or self.ref_points is None:
            return NPFloatType(0.5)
        coverage = self._estimate_coverage().get("coverage_ratio")
        if coverage is None:
            return NPFloatType(0.5)
        return NPFloatType(1.0 - coverage)

    def fit(self, reference_points: Matrix | NPMatrix) -> Self:
        """Fit the one-class SVM to reference ODD data.

        Args:
            reference_points (Matrix | NPMatrix): Reference ODD points.

        Returns:
            Self: Self for method chaining.
        """
        self.ref_points = reference_points
        data_t = np.asarray(reference_points, dtype=float).T

        if self.auto_select:
            self.gamma, self.nu, self.svm = _select_ocsvm_hyperparameters(data_t)
        else:
            self.svm = OneClassSVM(kernel="rbf", nu=self.nu, gamma=self.gamma)
            self.svm.fit(data_t)

        self.trained = True
        return self

    def decision_function(
        self, test_points: Matrix | NPMatrix
    ) -> npt.NDArray[np.float64]:
        """Return the raw signed score, positive inside the boundary.

        Args:
            test_points (Matrix | NPMatrix): Points to evaluate.

        Returns:
            npt.NDArray[np.float64]: Decision-function values.

        Raises:
            RuntimeError: If the monitor is not fitted.
        """
        if not self.trained or self.svm is None:
            raise RuntimeError("OneClassSVMBoundary not fitted yet")
        return cast(
            "npt.NDArray[np.float64]",
            self.svm.decision_function(np.asarray(test_points, dtype=float).T),
        )

    def __call__(self, test_point: Vector | NPVector) -> bool:
        """Check if a test point is inside the boundary.

        Args:
            test_point (Vector | NPVector): Point to evaluate.

        Returns:
            bool: True if ``decision_function(test_point) >=
                threshold``.
        """
        test_point_matrix = np.asarray(test_point, dtype=float).reshape(-1, 1)
        score = self.decision_function(test_point_matrix)
        return bool(score[0] >= self.threshold)

    def evaluate_batch(self, test_points: Matrix | NPMatrix) -> npt.NDArray[np.bool_]:
        """Vectorized membership evaluation for multiple test points.

        Args:
            test_points (Matrix | NPMatrix): Points to evaluate.

        Returns:
            npt.NDArray[np.bool_]: Boolean membership array.
        """
        return self.decision_function(test_points) >= self.threshold


def _pairwise_kernel(  # ruff:ignore[too-many-arguments]
    x: NPMatrix,
    y: NPMatrix,
    *,
    kernel: SVDDKernel,
    gamma: float,
    degree: int,
    coef0: float,
) -> npt.NDArray[np.float64]:
    """Compute a pairwise kernel matrix between two point sets.

    Args:
        x (NPMatrix): First point set, shape (n, d).
        y (NPMatrix): Second point set, shape (m, d).
        kernel (SVDDKernel): Kernel name.
        gamma (float): Resolved (scalar) gamma for "rbf"/"poly".
        degree (int): Polynomial degree for "poly".
        coef0 (float): Polynomial offset for "poly".

    Returns:
        npt.NDArray[np.float64]: Kernel matrix, shape (n, m).

    Raises:
        ValueError: If ``kernel`` is not one of the supported names.
    """
    if kernel == "linear":
        return cast("npt.NDArray[np.float64]", x @ y.T)
    if kernel == "poly":
        return cast("npt.NDArray[np.float64]", (gamma * (x @ y.T) + coef0) ** degree)
    if kernel == "rbf":
        sq_x = np.sum(x**2, axis=1)[:, np.newaxis]
        sq_y = np.sum(y**2, axis=1)[np.newaxis, :]
        sq_dist = np.maximum(sq_x + sq_y - 2.0 * (x @ y.T), 0.0)
        return cast("npt.NDArray[np.float64]", np.exp(-gamma * sq_dist))
    raise ValueError(f"unknown SVDD kernel {kernel!r}")


def _kernel_self(
    x: NPMatrix, *, kernel: SVDDKernel, gamma: float, degree: int, coef0: float
) -> npt.NDArray[np.float64]:
    """Compute ``k(x_i, x_i)`` for every row of ``x``.

    Args:
        x (NPMatrix): Points, shape (n, d).
        kernel (SVDDKernel): Kernel name.
        gamma (float): Resolved (scalar) gamma for "rbf"/"poly".
        degree (int): Polynomial degree for "poly".
        coef0 (float): Polynomial offset for "poly".

    Returns:
        npt.NDArray[np.float64]: Self-kernel values, shape (n,).

    Raises:
        ValueError: If ``kernel`` is not one of the supported names.
    """
    if kernel == "linear":
        return np.sum(x**2, axis=1)
    if kernel == "poly":
        return (gamma * np.sum(x**2, axis=1) + coef0) ** degree
    if kernel == "rbf":
        return np.ones(x.shape[0], dtype=float)
    raise ValueError(f"unknown SVDD kernel {kernel!r}")


def _solve_svdd_dual(
    kernel_matrix: npt.NDArray[np.float64],
    kernel_diag: npt.NDArray[np.float64],
    c_bound: float,
) -> npt.NDArray[np.float64]:
    """Solve the Tax--Duin SVDD dual for the support weights ``alpha``.

    Maximizes ``sum_i alpha_i k(x_i, x_i) - alpha^T K alpha`` subject to
    ``0 <= alpha_i <= c_bound`` and ``sum_i alpha_i = 1``, equivalently
    minimizing ``alpha^T K alpha - alpha^T diag(K)``.

    Args:
        kernel_matrix (npt.NDArray[np.float64]): Reference-reference
            kernel matrix ``K``, shape (n, n).
        kernel_diag (npt.NDArray[np.float64]): ``k(x_i, x_i)`` per
            reference point, shape (n,).
        c_bound (float): Per-point upper bound ``C = 1 / (nu * n)``.

    Returns:
        npt.NDArray[np.float64]: Solved dual weights ``alpha``, shape
        (n,).
    """
    n = kernel_matrix.shape[0]

    def objective(alpha: npt.NDArray[np.float64]) -> float:
        return float(alpha @ kernel_matrix @ alpha - alpha @ kernel_diag)

    def objective_grad(alpha: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return 2.0 * (kernel_matrix @ alpha) - kernel_diag

    constraints = (
        {
            "type": "eq",
            "fun": lambda alpha: float(np.sum(alpha) - 1.0),
            "jac": lambda _alpha: np.ones(n),
        },
    )
    bounds = [(0.0, c_bound)] * n
    x0 = np.full(n, 1.0 / n)

    result = scipy.optimize.minimize(
        objective,
        x0,
        jac=objective_grad,
        bounds=bounds,
        constraints=constraints,
        method="SLSQP",
        options={"maxiter": _QP_MAX_ITER, "ftol": _QP_TOLERANCE},
    )
    return np.clip(np.asarray(result.x, dtype=float), 0.0, c_bound)


class SVDDBoundary(ODDBoundaryMethod):
    """Support Vector Data Description (SVDD) ODD boundary.

    Implements the Tax--Duin (2004) SVDD dual directly via a
    constrained QP (``scipy.optimize``), rather than aliasing it to
    ``sklearn.svm.OneClassSVM``. Score = ``R^2 - ||phi(x) - a||^2``
    (the squared distance from the enclosing hypersphere's radius),
    positive inside, matching ``OneClassSVMBoundary``'s sign
    convention so the same ``score >= threshold`` rule applies.

    .. warning::
        With ``kernel="rbf"`` (the default), ``k(x, x) = 1`` for every
        ``x``, which collapses the SVDD dual onto exactly the same
        quadratic program as the one-class SVM dual (Scholkopf et al.,
        2001): the two produce identical decision boundaries under
        matched ``(nu, gamma)``. An RBF ``SVDDBoundary`` is therefore
        **not** independent evidence from ``OneClassSVMBoundary`` --
        do not report both. Use ``kernel="poly"`` (or another
        non-constant-diagonal kernel) for a genuinely distinct
        comparison; the constructor warns at runtime when
        ``kernel="rbf"`` for exactly this reason.

    Attributes:
        kernel (SVDDKernel): Kernel name ("rbf", "poly", or "linear").
        gamma (float | Literal["scale"]): Kernel gamma for "rbf"/
            "poly".
        nu (float): Upper bound on the fraction of margin errors /
            lower bound on the fraction of support vectors, same
            convention as ``OneClassSVMBoundary``.
        degree (int): Polynomial degree for "poly".
        coef0 (float): Polynomial offset for "poly".
        threshold (FloatType): Decision threshold in score space.
        alpha (npt.NDArray[np.float64] | None): Solved dual weights.
        method_type (str): Method identifier property.
        decision_boundary (DecisionBoundary): Decision-boundary
            metadata property.
    """

    def __init__(  # ruff:ignore[too-many-arguments, too-many-positional-arguments]
        self,
        kernel: SVDDKernel = DEFAULT_SVDD_KERNEL,
        gamma: float | Literal["scale"] = DEFAULT_GAMMA,
        nu: float = DEFAULT_NU,
        degree: int = DEFAULT_DEGREE,
        coef0: float = DEFAULT_COEF0,
        threshold: FloatType | None = None,
    ) -> None:
        """Initialize the SVDD boundary.

        Args:
            kernel (SVDDKernel): Kernel name, one of "rbf", "poly",
                "linear". Default "rbf".
            gamma (float | Literal["scale"]): Kernel gamma for "rbf"/
                "poly" (default "scale", matching scikit-learn).
            nu (float): Nu parameter (default 0.05), same convention as
                ``OneClassSVMBoundary.nu``.
            degree (int): Polynomial degree for "poly" (default 3).
            coef0 (float): Polynomial offset for "poly" (default 0.0).
            threshold (FloatType | None): Decision threshold in score
                space; defaults to 0.0.

        Warns:
            UserWarning: If ``kernel="rbf"``, stating the provable
                equivalence to ``OneClassSVMBoundary`` under matched
                hyperparameters (see class docstring).
        """
        if kernel == "rbf":
            warnings.warn(
                "SVDDBoundary(kernel='rbf') is mathematically equivalent to "
                "OneClassSVMBoundary under matched (nu, gamma): the RBF "
                "kernel's constant diagonal k(x, x) = 1 collapses the "
                "Tax-Duin SVDD dual onto the one-class SVM dual "
                "(Tax & Duin, 2004; Scholkopf et al., 2001). Do not report "
                "both as independent baselines under RBF -- use kernel="
                "'poly' (or another non-constant-diagonal kernel) for a "
                "genuinely distinct comparison.",
                UserWarning,
                stacklevel=2,
            )
        self.kernel = kernel
        self.gamma = gamma
        self.nu = nu
        self.degree = degree
        self.coef0 = coef0
        self.threshold = threshold if threshold is not None else DEFAULT_THRESHOLD
        self.alpha: npt.NDArray[np.float64] | None = None
        self.ref_points: Matrix | NPMatrix | None = None
        self._ref_points_t: npt.NDArray[np.float64] | None = None
        self._resolved_gamma: float | None = None
        self._r_squared: float | None = None
        self._w_quad: float | None = None
        self._free_sv_index: int | None = None
        self.trained = False

    @property
    def method_type(self) -> str:
        """method_type property for the SVDD method.

        Returns:
            str: The method name.
        """
        return "svdd"

    @property
    def decision_boundary(self) -> DecisionBoundary:
        """decision_boundary property for the SVDD monitor.

        Returns:
            DecisionBoundary: Structured decision-boundary metadata.
        """
        if not self.trained:
            return DecisionBoundary(
                type="svdd",
                parameters={
                    "kernel": self.kernel,
                    "gamma": self.gamma,
                    "nu": self.nu,
                },
                coverage={},
                conservatism=None,
            )

        return DecisionBoundary(
            type="svdd",
            parameters={
                "kernel": self.kernel,
                "gamma": self._resolved_gamma,
                "nu": self.nu,
                "degree": self.degree,
                "coef0": self.coef0,
                "r_squared": self._r_squared,
                "threshold": float(self.threshold),
            },
            coverage=self._estimate_coverage(),
            conservatism=self._calculate_conservatism(),
        )

    def _kernel_kwargs(self) -> _SVDDKernelKwargs:
        """Resolved kernel keyword arguments for ``_pairwise_kernel``.

        Returns:
            _SVDDKernelKwargs: ``gamma``, ``degree``, ``coef0``.
        """
        gamma = self._resolved_gamma if self._resolved_gamma is not None else 1.0
        return {"gamma": gamma, "degree": self.degree, "coef0": self.coef0}

    def _estimate_coverage(self) -> dict[str, FloatType]:
        """Estimate coverage metrics for this boundary.

        Returns:
            dict[str, FloatType]: Coverage metrics for the fitted
                boundary.
        """
        if not self.trained or self.ref_points is None:
            return {}
        train_scores = self.decision_function(self.ref_points)
        above_threshold = (train_scores >= self.threshold).mean()
        return {
            "coverage_ratio": NPFloatType(above_threshold),
            "n_above_threshold": NPFloatType((train_scores >= self.threshold).sum()),
        }

    def _calculate_conservatism(self) -> FloatType:
        """Calculate conservatism for this boundary.

        Returns:
            FloatType: Conservatism score in the range [0, 1].
        """
        if not self.trained or self.ref_points is None:
            return NPFloatType(0.5)
        coverage = self._estimate_coverage().get("coverage_ratio")
        if coverage is None:
            return NPFloatType(0.5)
        return NPFloatType(1.0 - coverage)

    def fit(self, reference_points: Matrix | NPMatrix) -> Self:
        """Fit the SVDD dual to reference ODD data.

        Args:
            reference_points (Matrix | NPMatrix): Reference ODD points.

        Returns:
            Self: Self for method chaining.
        """
        self.ref_points = reference_points
        data_t = np.asarray(reference_points, dtype=float).T
        n = data_t.shape[0]

        self._resolved_gamma = (
            _resolve_rbf_gamma(self.gamma, data_t) if self.kernel != "linear" else 1.0
        )
        kernel_kwargs = self._kernel_kwargs()

        kernel_matrix = _pairwise_kernel(
            data_t, data_t, kernel=self.kernel, **kernel_kwargs
        )
        kernel_diag = _kernel_self(data_t, kernel=self.kernel, **kernel_kwargs)

        c_bound = 1.0 / (self.nu * n)
        alpha = _solve_svdd_dual(kernel_matrix, kernel_diag, c_bound)

        # Free (non-bound) support vectors satisfy 0 < alpha_i < C and
        # lie exactly on the hypersphere boundary; R^2 is averaged over
        # all of them (rather than taken from a single one) so the
        # anchor is not overly sensitive to which particular free SV
        # the QP solve happens to report closest to a bound.
        tol = max(_FREE_SV_TOLERANCE, 1e-4 * c_bound)
        free_mask = (alpha > tol) & (alpha < c_bound - tol)
        free_indices = np.flatnonzero(free_mask)
        if free_indices.size == 0:
            # Degenerate solve (no strictly interior alpha): fall back
            # to the index farthest from either bound.
            free_indices = np.array([np.argmin(np.minimum(alpha, c_bound - alpha))])

        f_values = kernel_matrix @ alpha
        w_quad = float(alpha @ kernel_matrix @ alpha)
        dist_sq_free = kernel_diag[free_indices] - 2.0 * f_values[free_indices] + w_quad
        r_squared = float(np.mean(dist_sq_free))

        self.alpha = alpha
        self._ref_points_t = data_t
        self._r_squared = r_squared
        self._w_quad = w_quad
        self._free_sv_index = int(free_indices[0])
        self.trained = True
        return self

    def decision_function(
        self, test_points: Matrix | NPMatrix
    ) -> npt.NDArray[np.float64]:
        """Return the raw signed score, positive inside the boundary.

        Score = ``R^2 - ||phi(x) - a||^2``, expanded as ``R^2 -
        (k(x, x) - 2 * alpha @ K(ref, x) + alpha @ K(ref, ref) @
        alpha)``. The last (quadratic-in-alpha) term is a scalar fixed
        at fit time (``_w_quad``), so evaluating new points only needs
        the reference-to-test kernel block, not the full training
        kernel matrix.

        Args:
            test_points (Matrix | NPMatrix): Points to evaluate.

        Returns:
            npt.NDArray[np.float64]: Decision-function values.

        Raises:
            RuntimeError: If the monitor is not fitted.
        """
        if (
            not self.trained
            or self.alpha is None
            or self._ref_points_t is None
            or self._r_squared is None
            or self._w_quad is None
        ):
            raise RuntimeError("SVDDBoundary not fitted yet")

        test_t = np.asarray(test_points, dtype=float).T
        kernel_kwargs = self._kernel_kwargs()
        k_test = _pairwise_kernel(
            self._ref_points_t, test_t, kernel=self.kernel, **kernel_kwargs
        )
        k_test_diag = _kernel_self(test_t, kernel=self.kernel, **kernel_kwargs)

        f_test = self.alpha @ k_test
        dist_sq = k_test_diag - 2.0 * f_test + self._w_quad
        return self._r_squared - dist_sq

    def __call__(self, test_point: Vector | NPVector) -> bool:
        """Check if a test point is inside the boundary.

        Args:
            test_point (Vector | NPVector): Point to evaluate.

        Returns:
            bool: True if ``decision_function(test_point) >=
                threshold``.
        """
        test_point_matrix = np.asarray(test_point, dtype=float).reshape(-1, 1)
        score = self.decision_function(test_point_matrix)
        return bool(score[0] >= self.threshold)

    def evaluate_batch(self, test_points: Matrix | NPMatrix) -> npt.NDArray[np.bool_]:
        """Vectorized membership evaluation for multiple test points.

        Args:
            test_points (Matrix | NPMatrix): Points to evaluate.

        Returns:
            npt.NDArray[np.bool_]: Boolean membership array.
        """
        return self.decision_function(test_points) >= self.threshold


__all__ = ["OneClassSVMBoundary", "SVDDBoundary"]
