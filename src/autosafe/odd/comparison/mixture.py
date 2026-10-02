# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""Gaussian-mixture ODD boundary comparison method."""

import numpy as np
from sklearn.mixture import GaussianMixture

from autosafe.odd.comparison.base import DecisionBoundary
from autosafe.odd.comparison.density import SuperlevelSetMonitor
from autosafe.typing import FloatType, Matrix, NPFloatType, NPMatrix

DEFAULT_N_COMPONENTS_RANGE = (1, 10)


class GaussianMixtureBoundary(SuperlevelSetMonitor):
    """Superlevel-set ODD boundary using a Gaussian mixture model (GMM).

    Implements the same threshold-based membership rule as
    ``SuperlevelSetMonitor`` (``ODD = {x | score(x) > gamma}``), but
    replaces the KDE score with the log-likelihood of a
    ``sklearn.mixture.GaussianMixture`` fit. Rather than reimplementing
    gamma auto-detection, coverage estimation, or the ``__call__`` /
    ``evaluate_batch`` superlevel-set logic, this class subclasses
    ``SuperlevelSetMonitor`` and overrides only what differs: how the
    reference model is fit (``fit``) and how a score is produced from
    it (``pdf``).

    The component count is not fixed: ``fit`` selects it by BIC over
    ``n_components_range``, because a hardcoded ``n_components`` is the
    obvious reviewer objection to a GMM baseline. The selected count is
    stored in ``n_components`` and surfaced in ``decision_boundary``.

    ``random_state`` has no default and must always be passed
    explicitly: determinism is the paper's central claim, and a
    baseline whose fit silently varies run-to-run would undercut the
    comparison it is meant to support.

    Attributes:
        random_state (int): Seed for ``GaussianMixture``'s EM
            initialization. Required so two fits with the same value
            are bit-identical.
        n_components_range (tuple[int, int]): Inclusive
            ``(min, max)`` component count searched by BIC.
        gmm (GaussianMixture | None): Fitted mixture model.
        n_components (int | None): BIC-selected component count.
        method_type (str): Method identifier property.
        decision_boundary (DecisionBoundary): Decision-boundary
            metadata property.
    """

    def __init__(
        self,
        random_state: int,
        n_components_range: tuple[int, int] = DEFAULT_N_COMPONENTS_RANGE,
        gamma: FloatType | None = None,
        sigmoid_weight: FloatType | None = None,
    ) -> None:
        """Initialize the Gaussian-mixture boundary.

        Args:
            random_state (int): Seed forwarded to ``GaussianMixture``.
                Required (no default) so determinism is explicit at
                every call site.
            n_components_range (tuple[int, int]): Inclusive
                ``(min, max)`` component count searched by BIC during
                ``fit``.
            gamma (FloatType | None): Likelihood (log-space) threshold
                for the superlevel set. Auto-calibrated from data when
                ``None``, exactly as in ``SuperlevelSetMonitor``.
            sigmoid_weight (FloatType | None): Forwarded to
                ``SuperlevelSetMonitor`` for the score transition.
        """
        super().__init__(gamma=gamma, bandwidth=None, sigmoid_weight=sigmoid_weight)
        self.random_state = random_state
        self.n_components_range = n_components_range
        self.gmm: GaussianMixture | None = None
        self.n_components: int | None = None

    @property
    def method_type(self) -> str:
        """method_type property for the Gaussian-mixture method.

        Returns:
            str: The method name.
        """
        return "gmm"

    @property
    def decision_boundary(self) -> DecisionBoundary:
        """decision_boundary property for the Gaussian-mixture monitor.

        Returns:
            DecisionBoundary: Structured decision-boundary metadata.
        """
        if not self.trained or self.gmm is None:
            return DecisionBoundary(
                type="gmm",
                parameters={
                    "gamma": float(self.gamma),
                    "n_components_range": list(self.n_components_range),
                    "random_state": self.random_state,
                },
                coverage={},
                conservatism=None,
            )

        return DecisionBoundary(
            type="gmm",
            parameters={
                "gamma": float(self.gamma),
                "n_components": self.n_components,
                "n_components_range": list(self.n_components_range),
                "random_state": self.random_state,
                "sigmoid_weight": self.sigmoid_weight,
            },
            coverage=self._estimate_coverage(),
            conservatism=self._calculate_conservatism(),
        )

    def fit(self, reference_points: Matrix | NPMatrix) -> "GaussianMixtureBoundary":
        """Fit a BIC-selected Gaussian mixture to reference ODD data.

        Args:
            reference_points (Matrix | NPMatrix): Reference ODD points.

        Returns:
            GaussianMixtureBoundary: Self for method chaining.
        """
        self.ref_points = reference_points
        data_t = np.asarray(reference_points, dtype=float).T
        n_samples = data_t.shape[0]

        lo, hi = self.n_components_range
        lo = max(1, min(lo, n_samples))
        max_components = max(lo, min(hi, n_samples))

        best_gmm: GaussianMixture | None = None
        best_bic = np.inf
        best_n = lo
        for n_components in range(lo, max_components + 1):
            candidate = GaussianMixture(
                n_components=n_components,
                random_state=self.random_state,
            )
            candidate.fit(data_t)
            bic = candidate.bic(data_t)
            if bic < best_bic:
                best_bic = bic
                best_gmm = candidate
                best_n = n_components

        self.gmm = best_gmm
        self.n_components = best_n
        self.trained = True  # must be set before pdf() / suggest_reasonable_gamma()

        # Always calibrate gamma from data, as in SuperlevelSetMonitor.
        self.gamma = self.suggest_reasonable_gamma(percentile=np.float64(75.0))

        return self

    def suggest_reasonable_gamma(
        self,
        percentile: FloatType | None = None,
    ) -> FloatType:
        """Suggest a reasonable gamma threshold from GMM log-likelihood.

        Overrides ``SuperlevelSetMonitor.suggest_reasonable_gamma``,
        which gates on ``self.kde is None`` -- always true here, since
        this class never builds a ``KernelDensity`` -- and would
        otherwise silently skip calibration. The percentile-threshold
        logic itself is unchanged.

        Args:
            percentile (FloatType | None): Percentile of score values
                to use as the threshold.

        Returns:
            FloatType: Suggested gamma value based on data
            distribution.
        """
        if percentile is None:
            percentile = NPFloatType(75.0)

        if self.ref_points is None or self.gmm is None:
            return self.gamma

        pdf_values = self.pdf(self.ref_points)
        if len(pdf_values) == 0:
            return self.gamma

        candidate = NPFloatType(np.percentile(pdf_values, percentile))
        self.candidate_gamma = candidate
        return candidate

    def _calculate_conservatism(self) -> FloatType:
        """Calculate conservatism for the Gaussian-mixture boundary.

        Overrides ``SuperlevelSetMonitor._calculate_conservatism``,
        which gates on ``self.kde is None`` -- always true here -- and
        would otherwise always report the neutral 0.5 default instead
        of a data-derived value.

        Returns:
            FloatType: Conservatism score in the range [0, 1].
        """
        if not self.trained or self.gmm is None or self.ref_points is None:
            return NPFloatType(0.5)

        try:
            pdf_values = self.pdf(self.ref_points)
        except (RuntimeError, ValueError):
            return NPFloatType(0.5)

        if len(pdf_values) == 0:
            return NPFloatType(0.5)

        max_pdf = pdf_values.max()
        if max_pdf <= 0.0:
            return NPFloatType(0.5)

        normalized_gamma = self.gamma / max_pdf
        return NPFloatType(max(0.0, min(1.0, normalized_gamma)))

    def pdf(self, test_points: Matrix | NPMatrix) -> np.ndarray:
        """Score test points by GMM log-likelihood (exponentiated).

        Args:
            test_points (Matrix | NPMatrix): Points to evaluate.

        Returns:
            np.ndarray: Score values at test points, in the same
            (exponentiated log-likelihood) convention as
            ``SuperlevelSetMonitor.pdf``.

        Raises:
            RuntimeError: If the monitor is not fitted.
        """
        if not self.trained or self.gmm is None:
            raise RuntimeError("GaussianMixtureBoundary not fitted yet")

        log_likelihood = self.gmm.score_samples(test_points.T)
        return np.exp(log_likelihood * self.sigmoid_weight)


__all__ = ["GaussianMixtureBoundary"]
