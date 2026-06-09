# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""Collection of custom type definitions for autoSAFE."""

from collections.abc import Sequence
from typing import Annotated, Literal, TypeAlias

import jax  # noqa: F401
import jax.numpy as jnp
import numpy as np
from annotated_types import Ge, Le
from jaxtyping import Array, Float

KernelType: TypeAlias = Literal["RBF", "Laplacian"]
"""Kernel implementation selector."""

ClosestSampleModeType: TypeAlias = Literal["global", "per_dimension"]
"""Nearest-anchor search strategy: global or per-dimension."""

# JAX compute types: used for __call__ return values and computation
FloatType: TypeAlias = jnp.float64
"""JAX scalar float dtype used for all kernel computations."""

Vector: TypeAlias = Float[Array, "n"]  # noqa: F821, TC008
"""1-D JAX float array of length n."""

Matrix: TypeAlias = Float[Array, "n m"]  # noqa: F722
"""2-D JAX float array of shape (n, m)."""

SquareMatrix: TypeAlias = Float[Array, "n n"]  # noqa: F722
"""Square 2-D JAX float array of shape (n, n)."""

Affinity = Annotated[Float[Array, ""], Ge(0), Le(1)]  # noqa: F722
"""Scalar JAX affinity value constrained to [0, 1]."""

AffinityVector = Annotated[Float[Array, "n"], Ge(0), Le(1)]  # noqa: F821
"""1-D JAX array of affinity values, each constrained to [0, 1]."""

# NumPy types: used for stored state, FAISS, serializers, hashing
NPFloatType: TypeAlias = np.float64
"""NumPy scalar float dtype; mirrors FloatType for NumPy arrays."""

NPVector: TypeAlias = np.ndarray[tuple[int], np.dtype[np.float64]]
"""1-D NumPy float64 array; mirrors Vector for NumPy arrays."""

NPMatrix: TypeAlias = np.ndarray[tuple[int, int], np.dtype[np.float64]]
"""2-D NumPy float64 array of shape (n, m).

Mirrors Matrix for NumPy arrays.
"""

NPSquareMatrix: TypeAlias = np.ndarray[tuple[int, int], np.dtype[np.float64]]
"""Square 2-D NumPy float64 array of shape (n, n).

Mirrors SquareMatrix for NumPy arrays.
"""

NPAffinity: TypeAlias = Annotated[
    np.ndarray[tuple[()], np.dtype[np.float64]], Ge(0), Le(1)
]
"""0-D NumPy float64 array (scalar) constrained to [0, 1].

Mirrors Affinity for NumPy arrays.
"""

NPAffinityVector: TypeAlias = Annotated[
    np.ndarray[tuple[int], np.dtype[np.float64]], Ge(0), Le(1)
]
"""1-D NumPy float64 affinity vector; values in [0, 1].

Mirrors AffinityVector for NumPy arrays.
"""

# Compound aliases for repeated unions
KernelScaleParam: TypeAlias = NPVector | float
"""Per-dimension scale vector or scalar for all dimensions."""

BoundSpec: TypeAlias = Vector | Sequence[float] | float
"""ODD/sampling bound: JAX vector, sequence, or scalar."""

__all__ = [
    "Affinity",
    "AffinityVector",
    "Array",
    "BoundSpec",
    "ClosestSampleModeType",
    "Float",
    "FloatType",
    "KernelScaleParam",
    "KernelType",
    "Matrix",
    "NPAffinity",
    "NPAffinityVector",
    "NPFloatType",
    "NPMatrix",
    "NPSquareMatrix",
    "NPVector",
    "SquareMatrix",
    "Vector",
]
