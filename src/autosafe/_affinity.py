# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""Batched JAX affinity kernels with anchor-validity masking."""

import functools
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from autosafe import _jax_config  # noqa: F401  # MUST precede first jnp use

if TYPE_CHECKING:
    from collections.abc import Callable

DEFAULT_ANCHOR_CHUNK = 256
DEFAULT_POINT_CHUNK = 4096


@functools.cache
def _get_tile(
    variant: str,
) -> "Callable[..., jax.Array | tuple[jax.Array, jax.Array]]":
    """Return a jit-compiled per-tile affinity kernel.

    `valid` masks padded anchors to identity (1.0) so padding never
    biases the affinity product.

    Args:
        variant (str): Kernel variant to compile; one of "diag",
            "full_dense", "diag_dual", or "full_dense_dual". The dual
            variants return a tuple (prod_factor, sum_log) per tile.

    Returns:
        Callable[..., jax.Array | tuple[jax.Array, jax.Array]]:
            jit-compiled tile function for the requested variant.
            Standard variants return jax.Array; dual variants return
            tuple[jax.Array, jax.Array].

    Raises:
        ValueError: If `variant` is not a recognized variant name.
    """
    if variant == "diag":

        @jax.jit
        def _tile(
            x_block: jax.Array,
            a_block: jax.Array,
            s_block: jax.Array,
            valid: jax.Array,
        ) -> jax.Array:
            # x_block (mb,D)  a_block (nb,D)
            # s_block (nb,D)  valid (nb,) in {0.,1.}
            diff = x_block[None, :, :] - a_block[:, None, :]  # (nb,mb,D)
            mahal = jnp.sum(diff * diff * s_block[:, None, :], axis=-1)  # (nb,mb)
            mahal = jnp.maximum(mahal, 0.0)
            k = jnp.exp(-0.5 * mahal)
            factor = jnp.where(valid[:, None] > 0, 1.0 - k, 1.0)  # padded -> 1.0
            return jnp.prod(factor, axis=0)  # (mb,)

        return _tile
    if variant == "full_dense":

        @jax.jit
        def _tile(  # type: ignore[misc]
            x_block: jax.Array,
            a_block: jax.Array,
            sigma_inv_block: jax.Array,
            valid: jax.Array,
        ) -> jax.Array:
            # sigma_inv_block (nb,D,D)  # noqa: ERA001
            diff = x_block[None, :, :] - a_block[:, None, :]
            mahal = jnp.einsum("nmd,nde,nme->nm", diff, sigma_inv_block, diff)
            mahal = jnp.maximum(mahal, 0.0)
            k = jnp.exp(-0.5 * mahal)
            factor = jnp.where(valid[:, None] > 0, 1.0 - k, 1.0)
            return jnp.prod(factor, axis=0)

        return _tile
    if variant == "diag_dual":

        @jax.jit
        def _tile(  # type: ignore[misc]
            x_block: jax.Array,
            a_block: jax.Array,
            s_block: jax.Array,
            valid: jax.Array,
        ) -> tuple[jax.Array, jax.Array]:
            diff = x_block[None, :, :] - a_block[:, None, :]  # (nb,mb,D)
            mahal = jnp.sum(diff * diff * s_block[:, None, :], axis=-1)
            mahal = jnp.maximum(mahal, 0.0)
            k = jnp.exp(-0.5 * mahal)
            factor = jnp.where(valid[:, None] > 0, 1.0 - k, 1.0)
            # log(1 - exp(-z)): branch at ln 2 for full-range stability;
            # z == 0 yields -inf (exact anchor hit).
            z = 0.5 * mahal
            log1m_k = jnp.where(
                z <= jnp.log(2.0),
                jnp.log(-jnp.expm1(-z)),
                jnp.log1p(-jnp.exp(-z)),
            )
            log_term = jnp.where(valid[:, None] > 0, log1m_k, 0.0)
            return jnp.prod(factor, axis=0), jnp.sum(log_term, axis=0)

        return _tile
    if variant == "full_dense_dual":

        @jax.jit
        def _tile(  # type: ignore[misc]
            x_block: jax.Array,
            a_block: jax.Array,
            sigma_inv_block: jax.Array,
            valid: jax.Array,
        ) -> tuple[jax.Array, jax.Array]:
            diff = x_block[None, :, :] - a_block[:, None, :]
            mahal = jnp.einsum("nmd,nde,nme->nm", diff, sigma_inv_block, diff)
            mahal = jnp.maximum(mahal, 0.0)
            k = jnp.exp(-0.5 * mahal)
            factor = jnp.where(valid[:, None] > 0, 1.0 - k, 1.0)
            z = 0.5 * mahal
            log1m_k = jnp.where(
                z <= jnp.log(2.0),
                jnp.log(-jnp.expm1(-z)),
                jnp.log1p(-jnp.exp(-z)),
            )
            log_term = jnp.where(valid[:, None] > 0, log1m_k, 0.0)
            return jnp.prod(factor, axis=0), jnp.sum(log_term, axis=0)

        return _tile
    raise ValueError(f"unknown variant {variant!r}")


def _affinity(  # noqa: PLR0913, PLR0917
    anchors: jax.Array,
    params: jax.Array,
    x: jax.Array,
    anchor_chunk: int,
    point_chunk: int,
    variant: str,
) -> jax.Array:
    """Compute batched affinity using jit-compiled tiling.

    Args:
        anchors (jax.Array): (N, D) array of anchor positions.
        params (jax.Array): (N, D) diag-inv OR (N, D, D) dense
            sigma_inv per anchor.
        x (jax.Array): (M, D) array of evaluation points.
        anchor_chunk (int): Tile size along the anchor dimension.
        point_chunk (int): Tile size along the point dimension.
        variant (str): "diag" for diagonal covariance, "full_dense"
            for dense.

    Returns:
        jax.Array: (M,) affinity values = 1 - prod_i(1 - k_i).
            Padding uses fixed tile shapes (no recompilation) and a
            validity mask (no bias).
    """
    tile = _get_tile(variant)
    n, d = anchors.shape
    m = x.shape[0]
    result = jnp.zeros(m, dtype=anchors.dtype)

    for p_start in range(0, m, point_chunk):
        x_block = x[p_start : p_start + point_chunk]
        mb = x_block.shape[0]
        pad_p = point_chunk - mb
        if pad_p > 0:  # pad points to fixed size; sliced off via [:mb] below
            x_block = jnp.concatenate(
                [x_block, jnp.zeros((pad_p, d), dtype=x.dtype)], axis=0
            )

        acc = jnp.ones(point_chunk, dtype=anchors.dtype)
        for a_start in range(0, n, anchor_chunk):
            a_block = anchors[a_start : a_start + anchor_chunk]
            p_block = params[a_start : a_start + anchor_chunk]
            nb = a_block.shape[0]
            pad_a = anchor_chunk - nb
            if pad_a > 0:
                valid = jnp.concatenate([
                    jnp.ones(nb, anchors.dtype),
                    jnp.zeros(pad_a, anchors.dtype),
                ])
                a_block = jnp.concatenate(
                    [a_block, jnp.zeros((pad_a, d), dtype=anchors.dtype)], axis=0
                )
                if variant == "diag":
                    p_block = jnp.concatenate(
                        [p_block, jnp.ones((pad_a, d), dtype=params.dtype)], axis=0
                    )
                else:
                    p_block = jnp.concatenate(
                        [
                            p_block,
                            jnp.broadcast_to(
                                jnp.eye(d, dtype=params.dtype), (pad_a, d, d)
                            ),
                        ],
                        axis=0,
                    )
            else:
                valid = jnp.ones(anchor_chunk, anchors.dtype)
            acc *= tile(x_block, a_block, p_block, valid)

        result = result.at[p_start : p_start + mb].set((1.0 - acc)[:mb])

    jax.block_until_ready(result)
    return result


def _affinity_dual(  # noqa: PLR0913, PLR0914, PLR0917
    anchors: jax.Array,
    params: jax.Array,
    x: jax.Array,
    anchor_chunk: int,
    point_chunk: int,
    variant: str,
) -> tuple[jax.Array, jax.Array]:
    """Compute lin. affinity & log-survival in one batched pass.

    Args:
        anchors (jax.Array): (N, D) array of anchor positions.
        params (jax.Array): (N, D) diag-inv OR (N, D, D) dense
            sigma_inv per anchor.
        x (jax.Array): (M, D) array of evaluation points.
        anchor_chunk (int): Tile size along the anchor dimension.
        point_chunk (int): Tile size along the point dimension.
        variant (str): "diag_dual" or "full_dense_dual".

    Returns:
        tuple[jax.Array, jax.Array]: (alpha, survival), each (M,).
            alpha = 1 - prod_i(1 - k_i) (linear-space, may saturate at
            1.0); survival = sum_i log(1 - k_i) = log(1 - alpha),
            computed stably (no underflow; -inf only on exact anchor
            hits). See docs/log-space-affinity.md.
    """
    tile = _get_tile(variant)
    n, d = anchors.shape
    m = x.shape[0]
    result = jnp.zeros(m, dtype=anchors.dtype)
    surv_result = jnp.zeros(m, dtype=anchors.dtype)

    for p_start in range(0, m, point_chunk):
        x_block = x[p_start : p_start + point_chunk]
        mb = x_block.shape[0]
        pad_p = point_chunk - mb
        if pad_p > 0:
            x_block = jnp.concatenate(
                [x_block, jnp.zeros((pad_p, d), dtype=x.dtype)], axis=0
            )

        acc = jnp.ones(point_chunk, dtype=anchors.dtype)
        log_acc = jnp.zeros(point_chunk, dtype=anchors.dtype)
        for a_start in range(0, n, anchor_chunk):
            a_block = anchors[a_start : a_start + anchor_chunk]
            p_block = params[a_start : a_start + anchor_chunk]
            nb = a_block.shape[0]
            pad_a = anchor_chunk - nb
            if pad_a > 0:
                valid = jnp.concatenate([
                    jnp.ones(nb, anchors.dtype),
                    jnp.zeros(pad_a, anchors.dtype),
                ])
                a_block = jnp.concatenate(
                    [a_block, jnp.zeros((pad_a, d), dtype=anchors.dtype)], axis=0
                )
                if variant == "diag_dual":
                    p_block = jnp.concatenate(
                        [p_block, jnp.ones((pad_a, d), dtype=params.dtype)], axis=0
                    )
                else:
                    p_block = jnp.concatenate(
                        [
                            p_block,
                            jnp.broadcast_to(
                                jnp.eye(d, dtype=params.dtype), (pad_a, d, d)
                            ),
                        ],
                        axis=0,
                    )
            else:
                valid = jnp.ones(anchor_chunk, anchors.dtype)
            prod_part, log_part = tile(x_block, a_block, p_block, valid)
            acc *= prod_part
            log_acc += log_part

        result = result.at[p_start : p_start + mb].set((1.0 - acc)[:mb])
        surv_result = surv_result.at[p_start : p_start + mb].set(log_acc[:mb])

    jax.block_until_ready(result)
    jax.block_until_ready(surv_result)
    return result, surv_result


def affinity_diag(
    anchors: jax.Array,
    inv_diag: jax.Array,
    x: jax.Array,
    anchor_chunk: int = DEFAULT_ANCHOR_CHUNK,
    point_chunk: int = DEFAULT_POINT_CHUNK,
) -> jax.Array:
    """Compute affinity for all-diagonal-RBF anchors.

    Args:
        anchors (jax.Array): (N, D) array of anchor positions.
        inv_diag (jax.Array): (N, D) array of sigma_inv diagonal
            entries per anchor.
        x (jax.Array): (M, D) array of evaluation points.
        anchor_chunk (int): Tile size along the anchor dimension.
        point_chunk (int): Tile size along the point dimension.

    Returns:
        jax.Array: (M,) affinity values in [0, 1].
    """
    return _affinity(anchors, inv_diag, x, anchor_chunk, point_chunk, "diag")


def affinity_full_dense(
    anchors: jax.Array,
    sigma_inv_stack: jax.Array,
    x: jax.Array,
    anchor_chunk: int = DEFAULT_ANCHOR_CHUNK,
    point_chunk: int = DEFAULT_POINT_CHUNK,
) -> jax.Array:
    """Compute affinity for full (dense) sigma_inv anchors.

    Args:
        anchors (jax.Array): (N, D) array of anchor positions.
        sigma_inv_stack (jax.Array): (N, D, D) stack of sigma_inv
            matrices.
        x (jax.Array): (M, D) array of evaluation points.
        anchor_chunk (int): Tile size along the anchor dimension.
        point_chunk (int): Tile size along the point dimension.

    Returns:
        jax.Array: (M,) affinity values in [0, 1].
    """
    return _affinity(
        anchors, sigma_inv_stack, x, anchor_chunk, point_chunk, "full_dense"
    )


def affinity_diag_dual(
    anchors: jax.Array,
    inv_diag: jax.Array,
    x: jax.Array,
    anchor_chunk: int = DEFAULT_ANCHOR_CHUNK,
    point_chunk: int = DEFAULT_POINT_CHUNK,
) -> tuple[jax.Array, jax.Array]:
    """Compute lin. affinity & log-survival for diagonal-RBF anchors.

    Args:
        anchors (jax.Array): (N, D) array of anchor positions.
        inv_diag (jax.Array): (N, D) array of sigma_inv diagonal
            entries per anchor.
        x (jax.Array): (M, D) array of evaluation points.
        anchor_chunk (int): Tile size along the anchor dimension.
        point_chunk (int): Tile size along the point dimension.

    Returns:
        tuple[jax.Array, jax.Array]: (alpha, survival), each (M,).
            alpha = 1 - prod_i(1 - k_i) (linear-space, may saturate);
            survival = sum_i log(1 - k_i), computed stably in log-space.
            See docs/log-space-affinity.md.
    """
    return _affinity_dual(anchors, inv_diag, x, anchor_chunk, point_chunk, "diag_dual")


def affinity_full_dense_dual(
    anchors: jax.Array,
    sigma_inv_stack: jax.Array,
    x: jax.Array,
    anchor_chunk: int = DEFAULT_ANCHOR_CHUNK,
    point_chunk: int = DEFAULT_POINT_CHUNK,
) -> tuple[jax.Array, jax.Array]:
    """Compute lin. affinity & log-survival for full-covariance anchors.

    Args:
        anchors (jax.Array): (N, D) array of anchor positions.
        sigma_inv_stack (jax.Array): (N, D, D) stack of sigma_inv
            matrices.
        x (jax.Array): (M, D) array of evaluation points.
        anchor_chunk (int): Tile size along the anchor dimension.
        point_chunk (int): Tile size along the point dimension.

    Returns:
        tuple[jax.Array, jax.Array]: (alpha, survival), each (M,).
            alpha = 1 - prod_i(1 - k_i) (linear-space, may saturate);
            survival = sum_i log(1 - k_i), computed stably in log-space.
            See docs/log-space-affinity.md.
    """
    return _affinity_dual(
        anchors, sigma_inv_stack, x, anchor_chunk, point_chunk, "full_dense_dual"
    )
