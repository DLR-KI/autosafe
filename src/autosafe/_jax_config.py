# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""JAX x64 bootstrap: be imported before any jnp array is created."""

import os

import jax

_JAX_CONFIGURED = False


def _ensure_jax_x64() -> None:
    global _JAX_CONFIGURED  # noqa: PLW0603
    if not _JAX_CONFIGURED:
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
        jax.config.update("jax_enable_x64", True)  # noqa: FBT003
        jax.config.update("jax_default_matmul_precision", "highest")
        _JAX_CONFIGURED = True


_ensure_jax_x64()
