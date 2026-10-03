# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
# Research-driver relaxations (grid magic numbers, typer boolean flags,
# driver-length rules); scoped per file since experiment scripts are not
# library code.
# ruff: file-ignore[magic-value-comparison]
"""Ground-truth ODD registry for the benchmark experiments.

Each registered ODD exposes:
    -   ``contains(x)`` -> boolean membership for points ``x`` of shape
        (M, dim);
    -   ``sample_id(n, rng)`` -> ``n`` points uniformly inside the
        region;
    -   ``bbox`` -> (lower, upper) taxonomy bounds.

The registry intentionally includes non-convex cases (``annulus2d``,
``twoblobs2d``, ``banana2d``) that the box/polytope YAML ODDs of the
main package cannot express, so they live here rather than in
``autosafe.tools.monte_carlo``.
"""

from collections.abc import Callable

import numpy as np
import numpy.typing as npt

NPArray = npt.NDArray[np.float64]
NPBool = npt.NDArray[np.bool_]


class SyntheticODD:
    """A ground-truth ODD defined by a membership predicate on a box."""

    def __init__(
        self,
        name: str,
        dim: int,
        lower: NPArray,
        upper: NPArray,
        predicate: Callable[[NPArray], NPBool],
    ) -> None:
        """Initialize the synthetic ODD.

        Args:
            name (str): Registry key.
            dim (int): Dimensionality.
            lower (NPArray): Per-dimension lower taxonomy bounds.
            upper (NPArray): Per-dimension upper taxonomy bounds.
            predicate (Callable): Maps (M, dim) -> (M,) bool membership.
        """
        self.name = name
        self.dim = dim
        self.lower = np.asarray(lower, dtype=float)
        self.upper = np.asarray(upper, dtype=float)
        self._predicate = predicate

    @property
    def bbox(self) -> tuple[NPArray, NPArray]:
        """The taxonomy bounds as (lower, upper)."""
        return self.lower, self.upper

    def contains(self, x: NPArray) -> NPBool:
        """Return ODD membership: in the taxonomy box AND satisfying R.

        Membership requires both the ontology predicate ``R`` and the
        taxonomy ``X = [lower, upper]`` (paper Def. 3.1). Enforcing the
        box matters for validation points drawn on the enlarged box: a
        point outside ``X`` that happens to satisfy ``R`` is *not* in
        the ODD, and labeling it otherwise corrupts every metric.

        Args:
            x (NPArray): Points, shape (M, dim) or (dim,).

        Returns:
            NPBool: Membership mask, shape (M,).
        """
        x = np.atleast_2d(np.asarray(x, dtype=float))
        in_box = np.all((x >= self.lower) & (x <= self.upper), axis=1)
        return np.asarray(self._predicate(x), dtype=bool) & in_box

    def sample_id(self, n: int, rng: np.random.Generator) -> NPArray:
        """Draw ``n`` points uniformly inside the region by rejection.

        Args:
            n (int): Number of in-distribution points to return.
            rng (np.random.Generator): Seeded RNG.

        Returns:
            NPArray: Array of shape (n, dim) with points inside the ODD.
        """
        out: list[NPArray] = []
        got = 0
        # Oversample generously; non-convex regions can have low
        # acceptance.
        while got < n:
            size = (max(n * 4, 1024), self.dim)
            batch = rng.uniform(self.lower, self.upper, size=size)
            inside = batch[self.contains(batch)]
            out.append(inside)
            got += len(inside)
        return np.vstack(out)[:n]

    def enlarged_bbox(self, factor: float = 2.0) -> tuple[NPArray, NPArray]:
        """Return the bbox enlarged by ``factor`` about its center.

        Args:
            factor (float): Scale factor of the box's side lengths.

        Returns:
            tuple[NPArray, NPArray]: The enlarged (lower, upper) bounds.
        """
        center = 0.5 * (self.lower + self.upper)
        half = 0.5 * (self.upper - self.lower) * factor
        return center - half, center + half

    def sample_validation(
        self, n: int, rng: np.random.Generator, factor: float = 2.0
    ) -> tuple[NPArray, NPBool]:
        """Sample ``n`` uniform points on the enlarged box with labels.

        Args:
            n (int): Number of validation points.
            rng (np.random.Generator): Seeded RNG.
            factor (float): Box enlargement factor (paper uses 2x).

        Returns:
            tuple[NPArray, NPBool]: (points (n, dim), labels (n,)).
        """
        lo, hi = self.enlarged_bbox(factor)
        pts = rng.uniform(lo, hi, size=(n, self.dim))
        return pts, self.contains(pts)


def _linear2d(x: NPArray) -> NPBool:
    return x[:, 1] >= x[:, 0] - 3.0


def _annulus2d(x: NPArray) -> NPBool:
    r2 = x[:, 0] ** 2 + x[:, 1] ** 2
    return (r2 >= 1.0) & (r2 <= 16.0)


def _twoblobs2d(x: NPArray) -> NPBool:
    d1 = np.linalg.norm(x - np.array([-2.5, -2.5]), axis=1)
    d2 = np.linalg.norm(x - np.array([2.5, 2.5]), axis=1)
    return (d1 <= 1.5) | (d2 <= 1.5)


def _banana2d(x: NPArray) -> NPBool:
    return np.abs(x[:, 1] - 0.4 * x[:, 0] ** 2 + 2.0) <= 1.0


def _poly5d(x: NPArray) -> NPBool:
    return (x[:, 1] >= x[:, 0] - 3.0) & (np.sum(x**2, axis=1) <= 60.0)


def _poly10d(x: NPArray) -> NPBool:
    return (x[:, 1] >= x[:, 0] - 3.0) & (np.sum(x**2, axis=1) <= 200.0)


def make_corr2d(rho: float, w: float = 0.6) -> "SyntheticODD":
    """Rotated anisotropic band ``|x2 - rho*x1| <= w`` on ``[-5, 5]^2``.

    Coupling strength grows with ``rho``: ``rho=0`` is an axis-aligned
    band (a diagonal kernel suffices), ``rho>0`` rotates it so the local
    data covariance acquires off-diagonal structure that only a
    full-covariance kernel can align to. Used by the
    covariance-structure experiment (diagonal/isotropic-vs-full
    ablation).

    Args:
        rho (float): band slope (coupling strength).
        w (float): band half-width in the ``x2 - rho*x1`` coordinate.

    Returns:
        SyntheticODD: the registered-style band ODD.
    """

    def _pred(x: NPArray) -> NPBool:
        return np.abs(x[:, 1] - rho * x[:, 0]) <= w

    return SyntheticODD(f"corr2d_rho{rho:g}", 2, *_box(2, -5.0, 5.0), _pred)


def _box(dim: int, lo: float, hi: float) -> tuple[NPArray, NPArray]:
    return np.full(dim, lo), np.full(dim, hi)


_REGISTRY: dict[str, SyntheticODD] = {
    "linear2d": SyntheticODD("linear2d", 2, *_box(2, -5.0, 5.0), _linear2d),
    "annulus2d": SyntheticODD("annulus2d", 2, *_box(2, -5.0, 5.0), _annulus2d),
    "twoblobs2d": SyntheticODD("twoblobs2d", 2, *_box(2, -5.0, 5.0), _twoblobs2d),
    "banana2d": SyntheticODD("banana2d", 2, *_box(2, -5.0, 5.0), _banana2d),
    "poly5d": SyntheticODD("poly5d", 5, *_box(5, -5.0, 5.0), _poly5d),
    "poly10d": SyntheticODD("poly10d", 10, *_box(10, -5.0, 5.0), _poly10d),
    "corr2d": make_corr2d(1.0),
}


def get_odd(name: str) -> SyntheticODD:
    """Return the registered :class:`SyntheticODD` named ``name``.

    Args:
        name (str): Registered ODD name, e.g. ``"linear2d"``.

    Returns:
        SyntheticODD: The registered ODD.

    Raises:
        KeyError: If ``name`` is not a registered ODD.
    """
    if name not in _REGISTRY:
        raise KeyError(f"unknown synthetic ODD {name!r}; have {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def registry_names() -> list[str]:
    """Return the sorted list of registered ODD names.

    Returns:
        list[str]: Registered names, sorted.
    """
    return sorted(_REGISTRY)
