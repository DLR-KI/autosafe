# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT

import pytest

from autosafe.kernels import Kernel


def test_kernel_abstract_methods_raises_typeerror():
    """Test that abstract methods raise TypeError when called."""
    with pytest.raises(TypeError):
        _ = Kernel()  # ty: ignore[missing-argument]
