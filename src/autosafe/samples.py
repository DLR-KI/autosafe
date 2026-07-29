# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""Class for the set of all samples.

The set of samples is a collection of `Sample` objects, each
representing a point in the n-dimensional ODD space. The total set
allows to query the samples and their kernels, giving predictions as to
how likely any given vector is in the ODD.
"""

from collections.abc import Iterator
from typing import Any, cast, overload

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
import tqdm.rich

from autosafe import (
    _affinity,
    _jax_config,  # ruff:ignore[unused-import]
)
from autosafe.kernels import KernelDict
from autosafe.kernels.rbf import RBFKernel
from autosafe.neighbors import (
    find_closest_vectors_by_index,
    find_closest_vectors_by_index_per_dimension,
)
from autosafe.pointsets import rows_in
from autosafe.sample import Sample
from autosafe.typing import (
    Affinity,
    AffinityVector,
    ClosestSampleModeType,
    FloatType,
    KernelType,
    Matrix,
    NPFloatType,
    NPMatrix,
    NPVector,
    SampleLike,
    Vector,
)


class Samples:
    """Class representing a set of samples in the autoSAFE framework.

    The set of samples is a collection of `Sample` objects, each
    representing a point in the n-dimensional ODD space. The total set
    allows to query the samples and their kernels, giving predictions as
    to how likely any given vector is in the ODD.

    To initialize the set of samples, the `Samples` class requires a
    list of vectors, either as a list of lists, list of Vector or an n*d
    array.

    Args:
        samples (SampleLike): An object representing the sample points
            in the ODD space.
        closest_sample_mode (ClosestSampleModeType): Mode to determine
            how closest samples are found. "global" finds the closest
            sample globally, while "per_dimension" finds the closest
            sample for each dimension separately.
        kernel_cls (KernelType): The kernel class to use for the
            samples.
        kernel_kwargs (dict[str, Any] | None): Additional keyword
            arguments to pass to the kernel class constructor.
        skip_updates (bool): Whether to skip updating closest samples
            and kernels after initialization.

    Attributes:
        samples (list[Sample]): The list of samples in the set.
        dim (int | None): The dimension of the samples in the set.
        closest_sample_mode (ClosestSampleModeType): Mode to determine
            how closest samples are found.
        kernel_cls_str (str): The string representation of the kernel
            class.
        kernel_cls (KernelType): The kernel class used for the samples.
        kernel_kwargs (dict[str, Any]): Additional keyword arguments
            passed to the kernel class constructor.
        shape (tuple[int, int]): The shape of the samples set, as a
            tuple of (number of samples, dimension of samples).
    """

    def __init__(
        self,
        samples: SampleLike,
        *,
        closest_sample_mode: ClosestSampleModeType = "global",
        kernel_cls: KernelType = "RBF",
        kernel_kwargs: dict[str, Any] | None = None,
        skip_updates: bool = False,
    ) -> None:
        self.samples: list[Sample] = []
        self.dim: int | None = None
        self.closest_sample_mode = closest_sample_mode
        self.kernel_cls_str = kernel_cls
        self.kernel_cls = KernelDict[kernel_cls]
        self.kernel_kwargs = kernel_kwargs if kernel_kwargs is not None else {}
        self._batch_cache_valid = False
        self.append(samples, skip_updates=skip_updates)

    def append(self, samples: SampleLike, *, skip_updates: bool = False) -> None:
        """Append new samples to the list of samples.

        Args:
            samples (SampleLike): The sample-like objects to append to
                the list.
            skip_updates (bool): Whether to skip updating closest
                samples and kernels after appending.
        """
        samples = self._convert_to_samples(samples)
        self.samples.extend(samples)

        if len(self.samples) > 1 and not skip_updates:
            self._find_closest_samples()
            self.refresh_kernels()
        self._batch_cache_valid = False

    def _convert_to_samples(self, samplelike: SampleLike) -> list[Sample]:
        """Convert various input types to a list of Sample.

        A single sample can be of type Sample, list of floats or Vector.
        Samples can be a list of the aforementioned types, or a 2D
        array. Thus, we need to check for these types and convert
        accordingly. All other types will raise a TypeError. We also
        need to ensure that all samples have the same dimension. If not,
        a ValueError is raised.

        Args:
            samplelike (SampleLike): The input to convert to a Sample.

        Returns:
            Sample: The converted Sample object.

        Raises:
            TypeError: If the input type is not supported.
            ValueError: If the dimensions of the samples do not match.
        """
        if isinstance(samplelike, Sample):
            self.__check_dim(samplelike.x)
            return [samplelike]

        if isinstance(samplelike, (np.ndarray, jax.Array)):
            samplelike_array = np.squeeze(np.asarray(samplelike))
            if samplelike_array.ndim == 1:
                return [self.__to_sample(samplelike_array)]
            if samplelike_array.ndim == 2:  # ruff:ignore[magic-value-comparison]
                return [self.__to_sample(row) for row in samplelike_array]
            raise ValueError("Input array must be 1D or 2D.")

        if isinstance(samplelike, list):
            if all(isinstance(s, Sample) for s in samplelike):
                for s in cast("list[Sample]", samplelike):
                    self.__check_dim(s.x)
                return cast("list[Sample]", samplelike)

            if all(isinstance(s, (float, int)) for s in samplelike):
                return [self.__to_sample(samplelike)]

            if all(isinstance(s, (list, np.ndarray)) for s in samplelike):
                return [self.__to_sample(s) for s in samplelike]

        raise TypeError(f"Unsupported input type for samples: {type(samplelike)}")

    def _find_closest_samples(self) -> None:
        """Find the closest samples in the ODD space per dimension.

        Raises:
            ValueError: If the closest_sample_mode is invalid.
        """
        samples_array = cast("Matrix", np.array([sample.x for sample in self.samples]))

        if self.closest_sample_mode == "global":
            closest_indices = find_closest_vectors_by_index(samples_array)
        elif self.closest_sample_mode == "per_dimension":
            closest_indices = find_closest_vectors_by_index_per_dimension(samples_array)
        else:
            raise ValueError(
                f"Invalid closest_sample_mode: {self.closest_sample_mode}"
            )  # pragma: no cover

        for idx, sample in tqdm.rich.tqdm(
            enumerate(self.samples),
            total=len(self.samples),
            desc="Assigning closest samples",
        ):
            if self.closest_sample_mode == "global":
                sample.closest_sample = [self.samples[closest_indices[idx]]]
            else:  # per_dimension
                sample.closest_sample = [
                    self.samples[closest_idx] for closest_idx in closest_indices[:, idx]
                ]

    def refresh_kernels(
        self, kernel_kwargs_override: dict[str, Any] | None = None
    ) -> None:
        """Update the kernels of the samples in the set.

        Args:
            kernel_kwargs_override (dict[str, Any] | None): If provided,
                these kwargs are passed to the kernel instead of
                ``self.kernel_kwargs``. Used by the calibration path to
                pass resolved (concrete) kappa/eta arrays while keeping
                the meta form in ``self.kernel_kwargs`` for cache-key
                equality checks.

        Raises:
            ValueError: If a sample does not have a closest sample.
        """
        kwargs = (
            self.kernel_kwargs
            if kernel_kwargs_override is None
            else kernel_kwargs_override
        )
        for sample in tqdm.rich.tqdm(self.samples, desc="Updating kernels"):
            # Update the kernel of each sample
            if sample.kernel is None:
                sample.kernel = self.kernel_cls(x_i=sample.x, **kwargs)
            # Only for type checkers, this will not happen at runtime.
            if sample.closest_sample is None:  # pragma: no cover
                raise ValueError(f"No closest sample found for sample {sample}.")

            if (
                self.closest_sample_mode == "per_dimension"
                and len(sample.closest_sample) > 1
            ):
                # Per-dimension mode: (n_dims, n_dims) matrix, column
                # j = nearest neighbor in dimension j. Lets the kernel
                # set sigma[i,i] from the dim-i neighbor's distance in
                # dim i.
                x_nn_matrix = cast(
                    "Matrix",
                    np.stack([nn.x for nn in sample.closest_sample], axis=1),
                )
                sample.kernel.update(x_nn=x_nn_matrix, **kwargs)
            else:
                sample.kernel.update(x_nn=sample.closest_sample[0].x, **kwargs)
        self._batch_cache_valid = False

    def _build_batch_arrays(self) -> None:
        """Build and cache anchor/inv-diag arrays for batch affinity."""
        self._anchors_np = np.stack([s.x for s in self.samples]).astype(
            NPFloatType
        )  # (N, D)
        diags: list[NPVector] = []
        all_diag_rbf = True
        for s in self.samples:
            k = s.kernel
            if (
                isinstance(k, RBFKernel)
                and k.sigma_inv is not None
                and k._sigma_is_diagonal  # ruff:ignore[private-member-access]
            ):
                diags.append(np.diag(k.sigma_inv))
            else:
                all_diag_rbf = False
                break
        self._all_kernels_diagonal_rbf = all_diag_rbf
        self._inv_diag_np = (
            np.stack(diags).astype(NPFloatType) if all_diag_rbf else None
        )
        self._batch_cache_valid = True

    def __to_sample(self, arr: Any) -> Sample:  # ruff:ignore[any-type]
        """Convert an object to a Sample.

        Args:
            arr (Any): The object to convert.

        Returns:
            Sample: The converted Sample object.
        """
        arr_ = np.array(arr, dtype=NPFloatType).squeeze()
        self.__check_dim(arr_)
        return Sample(x=arr_)  # type: ignore[arg-type]

    def __check_dim(self, sample_array: np.ndarray | jax.Array) -> None:
        """Check if the sample's dimension matches the set dimension.

        Args:
            sample_array (np.ndarray | jax.Array): The sample array to
                check.

        Raises:
            ValueError: If the array is not 1d or does not match the
                dimension of the rest of the samples.
        """
        if sample_array.ndim != 1:
            raise ValueError("Each sample must be a 1D array.")
        if self.dim is None:
            self.dim = sample_array.shape[0]
        if self.dim != sample_array.shape[0]:
            raise ValueError("All samples must have the same dimension.")

    @property
    def shape(self) -> tuple[int, int]:
        """The shape of the samples set.

        Returns:
            tuple[int, int]: A tuple representing the number of samples
                and their dimension.
        """
        return len(self.samples), self.dim if self.dim is not None else 0

    @overload
    def __call__(self, x: Vector | NPVector) -> Affinity: ...

    @overload
    def __call__(self, x: Matrix | NPMatrix) -> AffinityVector: ...

    def __call__(
        self, x: "Vector | Matrix | NPVector | NPMatrix"
    ) -> "Affinity | AffinityVector":
        """Evaluate vector x's affinity with respect to the samples.

        Args:
            x (Vector | Matrix | NPVector | NPMatrix): The input vector
                or matrix (multiple vectors as a column matrix) for
                which to evaluate the affinity.

        Returns:
            Affinity | AffinityVector: The affinity of the vector x or
                the vector of affinities with respect to the samples, a
                JAX array value between 0 and 1.
        """
        if len(self.samples) == 0:
            x_j = jnp.asarray(x)
            if x_j.ndim == 1:
                return FloatType(0.0)
            n_pts = x_j.shape[0] if x_j.shape[1] == self.dim else x_j.shape[1]
            return jnp.zeros(n_pts, dtype=FloatType)

        if not self._batch_cache_valid:
            self._build_batch_arrays()

        x_j = jnp.asarray(x)
        is_single = x_j.ndim == 1
        x_eval = (
            x_j[None, :]
            if is_single
            else (x_j if x_j.shape[1] == self.dim else x_j.T)  # (n_points, D)
        )

        if self._all_kernels_diagonal_rbf:
            anchors = jnp.asarray(self._anchors_np)
            inv_diag = jnp.asarray(self._inv_diag_np)
            result = _affinity.affinity_diag(anchors, inv_diag, x_eval)
        else:
            prod = jnp.ones(x_eval.shape[0], dtype=FloatType)
            for s in self.samples:
                prod *= 1.0 - jnp.asarray(s(x_eval))
            result = 1.0 - prod

        return result[0] if is_single else result

    def affinity_dual(
        self, x: "Vector | Matrix | NPVector | NPMatrix"
    ) -> "tuple[Affinity, Affinity] | tuple[AffinityVector, AffinityVector]":
        """Evaluate affinity in linear and log space in one pass.

        Args:
            x (Vector | Matrix | NPVector | NPMatrix): Input vector or
                matrix of evaluation points.

        Returns:
            tuple: (alpha, survival). alpha is the linear-space
                affinity (may saturate at exactly 1.0 for many
                anchors); survival = log(1 - alpha) computed stably
                (-inf only on exact anchor hits). See
                docs/log-space-affinity.md.
        """
        if len(self.samples) == 0:
            x_j = jnp.asarray(x)
            if x_j.ndim == 1:
                return FloatType(0.0), FloatType(0.0)
            n_pts = x_j.shape[0] if x_j.shape[1] == self.dim else x_j.shape[1]
            zeros = jnp.zeros(n_pts, dtype=FloatType)
            return zeros, zeros

        if not self._batch_cache_valid:
            self._build_batch_arrays()

        x_j = jnp.asarray(x)
        is_single = x_j.ndim == 1
        x_eval = (
            x_j[None, :] if is_single else (x_j if x_j.shape[1] == self.dim else x_j.T)
        )

        if self._all_kernels_diagonal_rbf:
            anchors = jnp.asarray(self._anchors_np)
            inv_diag = jnp.asarray(self._inv_diag_np)
            alpha, survival = _affinity.affinity_diag_dual(anchors, inv_diag, x_eval)
        else:
            prod = jnp.ones(x_eval.shape[0], dtype=FloatType)
            log_acc = jnp.zeros(x_eval.shape[0], dtype=FloatType)
            for s in self.samples:
                k = jnp.clip(jnp.asarray(s(x_eval)), 0.0, 1.0)
                prod *= 1.0 - k
                log_acc += jnp.log1p(-k)
            alpha = 1.0 - prod
            survival = log_acc

        if is_single:
            return alpha[0], survival[0]
        return alpha, survival

    def batch_arrays(
        self,
    ) -> tuple[NPMatrix, npt.NDArray[np.float64] | None, bool]:
        """Return the cached anchor/inverse-diagonal batch arrays.

        Builds the cache first if it is stale. Exposed so the OOD
        consistency algorithm can read the fast-path arrays without
        reaching into private attributes.

        Returns:
            tuple[NPMatrix, npt.NDArray[np.float64] | None, bool]:
                Anchors of shape (n_anchors, n_dims); the stacked
                inverse-covariance diagonals of the same shape, or None
                when the fast path does not apply; and whether every
                kernel is a diagonal RBF.
        """
        if not self._batch_cache_valid:
            self._build_batch_arrays()
        return self._anchors_np, self._inv_diag_np, self._all_kernels_diagonal_rbf

    def invalidate_batch_cache(self) -> None:
        """Mark the cached batch arrays stale."""
        self._batch_cache_valid = False

    def sync_kernel_cache(self, index: int) -> None:
        """Refresh the batch cache after one kernel's sigma changed.

        On the all-diagonal fast path this patches a single row, which
        is O(n_dims) instead of an O(n_anchors * n_dims) rebuild. Else
        the cache is simply invalidated.

        Args:
            index (int): Index of the kernel whose sigma was replaced.
        """
        kern = self.samples[index].kernel
        if (
            self._batch_cache_valid
            and self._all_kernels_diagonal_rbf
            and self._inv_diag_np is not None
            and isinstance(kern, RBFKernel)
            and kern.sigma_inv is not None
        ):
            self._inv_diag_np[index] = np.diag(kern.sigma_inv)
        else:
            self._batch_cache_valid = False

    def enforce_ood_consistency(  # ruff:ignore[too-many-arguments]
        self,
        ood_points: "Matrix | NPMatrix",
        xi: float,
        shrink_factor: float = 0.9,
        max_iterations: int = 1_000_000,
        *,
        refresh_interval: int = 1000,
        log_interval: int = 1000,
        batch_jump: bool = False,
    ) -> dict[str, object]:
        """Enforce alpha(x) <= xi on all OOD points.

        Repeatedly selects the globally most-violated OOD point, finds
        its dominant kernel, and shrinks that kernel's covariance by
        ``shrink_factor`` until the constraint holds. Selection depends
        only on affinity values, so the result is independent of the
        ordering of ``ood_points`` (exact ties are broken by the lowest
        index and have measure zero for generic data).

        PRECONDITION: no OOD point may coincide with an anchor point. At
        an anchor the affinity is ``exp(0) = 1`` for every covariance,
        so the exit condition is unreachable and the loop cannot end.
        This is checked up front and raises
        :class:`~autosafe.exceptions.OODAnchorCoincidenceError`.

        Selection uses the log-survival ``L(x) = log(1 - alpha(x))``
        rather than ``alpha`` directly, because ``alpha`` saturates at
        exactly 1.0 for many points at once and ``argmax`` over such
        ties (or over NaN) degrades the tie rule. ``L`` is maintained
        incrementally---only one kernel changes per iteration, so the
        update is exact and costs O((M + N) * n) instead of the
        O(M * N * n) full sweep---with an exact refresh every
        ``refresh_interval`` iterations to bound accumulated round-off.

        Note: shrinking bypasses the sigma lower bound lam; the
        conditioning bound cond(Sigma) <= 1/SIGMA_FLOOR_RATIO holds only
        for unadjusted kernels. See docs/ood-consistency.md for the
        mathematics and docs/bandwidth-calibration.md for the floor.

        Args:
            ood_points (Matrix | NPMatrix): OOD samples, shape
                (M, n_dims), in the SAME (normalized) coordinate system
                as the anchors.
            xi (float): Maximum allowed OOD affinity, in (0, 1).
            shrink_factor (float): Covariance scale factor, in (0, 1).
            max_iterations (int): Safety cap; exceeding it raises.
            refresh_interval (int): Recompute L exactly every this many
                iterations; <= 0 disables periodic refresh.
            log_interval (int): Write a progress line every this many
                iterations; <= 0 disables it.
            batch_jump (bool): If True, apply the closed-form number of
                shrinks per iteration instead of one. This is a faithful
                acceleration only while the selected point and its
                dominant kernel stay the same, so the result may
                over-shrink relative to the one-shrink-per-iteration
                procedure; the constraint is always still satisfied.
                Default False, and all reported numbers use the default.

        Returns:
            dict[str, object]: Summary with keys ``iterations``,
                ``max_ood_affinity`` (final), ``adjusted_kernels``
                (mapping kernel index -> shrink count),
                ``exact_recomputations`` and ``batch_jump``.
        """
        return ood_consistency.enforce_ood_consistency(
            self,
            ood_points,
            xi,
            shrink_factor,
            max_iterations,
            refresh_interval=refresh_interval,
            log_interval=log_interval,
            batch_jump=batch_jump,
        )

    def __eq__(self, value: object) -> bool:
        """Check if two Samples instances are equal.

        Args:
            value (object): The object to compare with.

        Returns:
            bool: True if the two Samples instances are equal, False
                otherwise.
        """
        if not isinstance(value, Samples):
            return False
        if len(self.samples) != len(value.samples):
            return False
        return all(
            s1 == s2 for s1, s2 in zip(self.samples, value.samples, strict=False)
        )

    def __hash__(self) -> int:
        """Hash the Samples instance.

        Returns:
            int: The hash value of the Samples instance.
        """
        samples_hashes = tuple(hash(sample) for sample in self.samples)
        return hash((samples_hashes,))

    def __repr__(self) -> str:
        """Return a string representation of the sample.

        Returns:
            str: A string representation of the samples class object
                listing all contained samples.
        """
        return (
            f"Samples(samples={self.samples!r}, "
            f"closest_sample_mode={self.closest_sample_mode!r}, "
            f"kernel_cls={self.kernel_cls_str!r}, "
            f"kernel_kwargs={self.kernel_kwargs!r})"
        )

    def __str__(self) -> str:
        """Return a string representation of the sample.

        Returns:
            str: A string representation of the samples class object
                listing all contained samples.
        """
        return (
            f"Samples based on list of samples: {self.samples!s} "
            f"with kernel class {self.kernel_cls_str!s} "
            f"and kernel kwargs {self.kernel_kwargs!s}"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self) -> Iterator[Sample]:
        return iter(self.samples)

    def __getitem__(self, index: int) -> Sample:
        return self.samples[index]


__all__ = [
    "SampleLike",
    "Samples",
    "find_closest_vectors_by_index",
    "find_closest_vectors_by_index_per_dimension",
    "rows_in",
]
