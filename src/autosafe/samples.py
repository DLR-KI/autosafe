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
from typing import Any, TypeAlias, cast, overload

import faiss
import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
import tqdm.rich

from autosafe import (
    _affinity,
    _jax_config,  # noqa: F401
)
from autosafe.kernels import KernelDict
from autosafe.kernels.rbf import RBFKernel
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
    Vector,
)

SampleLike: TypeAlias = (
    Sample
    | list[float]
    | Vector
    | NPVector
    | list[Sample]
    | list[list[float]]
    | list[Vector]
    | list[NPVector]
    | Matrix
)


def find_closest_vectors_by_index(
    sample_array: Matrix | NPMatrix,
    disable_tqdm: bool = False,  # noqa: FBT001, FBT002
) -> npt.NDArray[np.int64]:
    """Find the closest vector for each vector in the matrix.

    Find the closest vector for each vector in the matrix using L2 norm.
    Then, return the index of the closest vector for each vector.

    Args:
        sample_array (Matrix | NPMatrix): Matrix with m vectors of
            dimension n.
        disable_tqdm (bool): Whether to disable the tqdm progress bar.

    Returns:
        npt.NDArray[np.int64]: Index of the closest vector for each
            vector
    """
    # FAISS only works with float32
    sample_array_float32 = np.ascontiguousarray(sample_array, dtype=np.float32)
    dimension = sample_array_float32.shape[1]

    # Create FAISS index for exact L2 search
    index = faiss.IndexFlatL2(dimension)
    index.add(sample_array_float32)  # pyright: ignore[reportCallIssue]

    closest_indices: npt.NDArray[np.int64] = np.array([], dtype=np.int64)

    # Fake tqdm progress bar for consistency
    for _ in tqdm.rich.tqdm(
        range(1),
        desc="Finding closest samples",
        disable=disable_tqdm,
    ):
        # Query 2 nearest neighbors (first is self, second is closest)
        _, indices = index.search(sample_array_float32, k=2)  # pyright: ignore[reportCallIssue]
        closest_indices = indices[:, 1].astype(
            np.int64
        )  # Take the second neighbor (skip self)
    return closest_indices


def find_closest_vectors_by_index_per_dimension(
    sample_array: Matrix | NPMatrix,
) -> npt.NDArray[np.int64]:
    """Find the closest vector for each vector per dimension.

    Find the closest vector for each vector in the matrix using L2 norm.
    Then, for each dimension, return the index of the closest vector
    based on that dimension alone.

    Args:
        sample_array (Matrix | NPMatrix): Matrix with m vectors of
            dimension n.

    Returns:
        npt.NDArray[np.int64]: Index of the closest vector per dimension
            for each vector
    """
    # Find closest vector per dimension
    # For each dimension, we need to find the closest vector based on
    # that dimension alone
    m, n = sample_array.shape
    closest_per_dim = np.zeros((n, m), dtype=np.int64)

    for dim in tqdm.rich.tqdm(range(n), desc="Finding closest samples per dimension"):
        closest_per_dim[dim, :] = find_closest_vectors_by_index(
            sample_array=sample_array[:, dim].reshape(-1, 1),
            disable_tqdm=True,
        )

    return closest_per_dim


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
            if samplelike_array.ndim == 2:  # noqa: PLR2004
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

    def refresh_kernels(self) -> None:
        """Update the kernels of the samples in the set.

        Raises:
            ValueError: If a sample does not have a closest sample.
        """
        for sample in tqdm.rich.tqdm(self.samples, desc="Updating kernels"):
            # Update the kernel of each sample
            if sample.kernel is None:
                sample.kernel = self.kernel_cls(x_i=sample.x, **self.kernel_kwargs)
            # Only for type checkers, this will not happen at runtime.
            if sample.closest_sample is None:  # pragma: no cover
                raise ValueError(f"No closest sample found for sample {sample}.")

            if (
                self.closest_sample_mode == "per_dimension"
                and len(sample.closest_sample) > 1
            ):
                # Per-dimension mode: (n_dims, n_dims) matrix, column
                # j = nearest neighbour in dimension j. Lets the kernel
                # set sigma[i,i] from the dim-i neighbour's distance in
                # dim i.
                x_nn_matrix = cast(
                    "Matrix",
                    np.stack([nn.x for nn in sample.closest_sample], axis=1),
                )
                sample.kernel.update(x_nn=x_nn_matrix, **self.kernel_kwargs)
            else:
                sample.kernel.update(
                    x_nn=sample.closest_sample[0].x, **self.kernel_kwargs
                )
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
                and k._sigma_is_diagonal  # noqa: SLF001
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

    def __to_sample(self, arr: Any) -> Sample:  # noqa: ANN401
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
        """Get the shape of the samples set.

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
