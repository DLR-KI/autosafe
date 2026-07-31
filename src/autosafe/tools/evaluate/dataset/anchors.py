# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""Anchor-coordinate extraction from a built ODD."""

from typing import TYPE_CHECKING

import numpy as np

from autosafe.typing import NPMatrix

if TYPE_CHECKING:
    from autosafe.samples import Samples


def _extract_anchor_points(odd: "Samples") -> NPMatrix:
    """Extract anchor points from an autoSAFE ODD object.

    Args:
        odd (Samples): Affinity ODD object.

    Returns:
        NPMatrix: Anchor points as `(n_points, n_dims)` array.
    """
    return np.array([np.array(sample.x, dtype=float) for sample in odd.samples])
