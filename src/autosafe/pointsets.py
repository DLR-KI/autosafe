# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""Set predicates over arrays of points.

General-purpose helpers that answer questions about how two point
sets relate, independent of any kernel or ODD machinery.
"""

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from autosafe.exceptions import RowShapeMismatchError

if TYPE_CHECKING:
    from autosafe.typing import NPMatrix


def rows_in(query: "NPMatrix", reference: "NPMatrix") -> npt.NDArray[np.bool_]:
    """Exact row-membership mask of ``query`` against ``reference``.

    Comparison is by float **value** over whole rows (``np.unique`` with
    ``axis=0``), so ``-0.0`` matches ``0.0`` and a row with ``NaN``
    never matches anything---both are the correct semantics here, since
    the quantity that matters is whether the Mahalanobis distance
    ``q = (x - x_i)^T Sigma^-1 (x - x_i)`` is exactly zero.

    Args:
        query (NPMatrix): Rows to test, shape (Q, n_dims).
        reference (NPMatrix): Rows to test against, shape (R, n_dims).

    Returns:
        npt.NDArray[np.bool_]: Shape (Q,); ``True`` where the query row
            equals some reference row.

    Raises:
        RowShapeMismatchError: If the two arrays disagree on the number
            of columns.
    """
    q = np.atleast_2d(np.asarray(query, dtype=float))
    r = np.atleast_2d(np.asarray(reference, dtype=float))
    if q.shape[0] == 0:
        return np.zeros((0,), dtype=bool)
    if r.shape[0] == 0:
        return np.zeros((q.shape[0],), dtype=bool)
    if q.shape[1] != r.shape[1]:
        raise RowShapeMismatchError(q.shape[1], r.shape[1])
    combined = np.vstack([r, q])
    _, inverse = np.unique(combined, axis=0, return_inverse=True)
    # NumPy 2.0 briefly changed the shape of `inverse` under axis=;
    # flatten defensively so this works on every 2.x.
    inverse = np.asarray(inverse).reshape(-1)
    n_ref = r.shape[0]
    reference_groups = np.unique(inverse[:n_ref])
    return np.isin(inverse[n_ref:], reference_groups)
