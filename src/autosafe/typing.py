# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""Collection of custom type definitions for autoSAFE."""

from collections.abc import Sequence
from typing import TYPE_CHECKING, Annotated, Literal, TypeAlias, Union

import jax  # ruff:ignore[unused-import]
import jax.numpy as jnp
import numpy as np
from annotated_types import Ge, Le
from jaxtyping import Array, Float

if TYPE_CHECKING:
    from autosafe.odd.config import (
        CalibratedRBFConfig,
        ConformalMembershipConfig,
        FixedMembershipConfig,
        ManualRBFConfig,
    )
    from autosafe.odd.openodd import (
        OpenODDCategoricalFeature,
        OpenODDNumericFeature,
    )
    from autosafe.sample import Sample

KernelType: TypeAlias = Literal["RBF", "Laplacian"]
"""Kernel implementation selector."""

ClosestSampleModeType: TypeAlias = Literal["global", "per_dimension"]
"""Nearest-anchor search strategy: global or per-dimension."""

StandardVariant: TypeAlias = Literal["diag", "full_dense"]
"""Standard batched-affinity implementation selector."""

DualVariant: TypeAlias = Literal["diag_dual", "full_dense_dual"]
"""Dual affinity/log-survival implementation selector."""

MethodName: TypeAlias = Literal[
    "hull_single",
    "knn",
    "kmeans",
    "density_single",
    "hull_clustered",
    "density_clustered",
    "dbscan_cluster",
]
"""Supported ODD comparison method name."""

# JAX compute types: used for __call__ return values and computation
FloatType: TypeAlias = jnp.float64
"""JAX scalar float dtype used for all kernel computations."""

Vector: TypeAlias = Float[Array, "n"]  # ruff:ignore[undefined-name, quoted-type-alias]
"""1-D JAX float array of length n."""

Matrix: TypeAlias = Float[Array, "n m"]  # ruff:ignore[forward-annotation-syntax-error]
"""2-D JAX float array of shape (n, m)."""

SquareMatrix: TypeAlias = Float[Array, "n n"]  # ruff:ignore[forward-annotation-syntax-error]
"""Square 2-D JAX float array of shape (n, n)."""

Affinity = Annotated[Float[Array, ""], Ge(0), Le(1)]  # ruff:ignore[forward-annotation-syntax-error]
"""Scalar JAX affinity value constrained to [0, 1]."""

AffinityVector = Annotated[Float[Array, "n"], Ge(0), Le(1)]  # ruff:ignore[undefined-name]
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

ScaleValue: TypeAlias = float | tuple[float, ...]
"""Scalar or per-dimension kernel scale stored in immutable form."""

SampleLike: TypeAlias = Union[
    "Sample",
    list[float],
    Vector,
    NPVector,
    list["Sample"],
    list[list[float]],
    list[Vector],
    list[NPVector],
    Matrix,
]
"""Input forms accepted by :class:`autosafe.samples.Samples`."""

KernelConfig: TypeAlias = Union["CalibratedRBFConfig", "ManualRBFConfig"]
"""Supported high-level kernel configuration."""

MembershipConfig: TypeAlias = Union[
    "FixedMembershipConfig",
    "ConformalMembershipConfig",
]
"""Fixed or data-calibrated ODD membership configuration."""

OpenODDFeature: TypeAlias = Union[
    "OpenODDNumericFeature",
    "OpenODDCategoricalFeature",
]
"""Supported mappings from fitted dimensions to OpenODD concepts."""

__all__ = [
    "Affinity",
    "AffinityVector",
    "Array",
    "BoundSpec",
    "ClosestSampleModeType",
    "DualVariant",
    "Float",
    "FloatType",
    "KernelConfig",
    "KernelScaleParam",
    "KernelType",
    "Matrix",
    "MembershipConfig",
    "MethodName",
    "NPAffinity",
    "NPAffinityVector",
    "NPFloatType",
    "NPMatrix",
    "NPSquareMatrix",
    "NPVector",
    "OpenODDFeature",
    "SampleLike",
    "ScaleValue",
    "SquareMatrix",
    "StandardVariant",
    "Vector",
]
