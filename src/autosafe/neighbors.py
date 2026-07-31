# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""Nearest-neighbor search over anchor point sets.

Used to parameterize each anchor's kernel from the distance to its
closest neighbor, globally or per dimension.
"""

import faiss
import numpy as np
import numpy.typing as npt
import tqdm.rich

from autosafe.typing import Matrix, NPMatrix


def find_closest_vectors_by_index(
    sample_array: Matrix | NPMatrix,
    disable_tqdm: bool = False,  # ruff:ignore[boolean-type-hint-positional-argument, boolean-default-value-positional-argument]
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
