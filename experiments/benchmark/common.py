# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
# Research-driver relaxations (grid magic numbers, typer boolean flags,
# driver-length rules); scoped per file since experiment scripts are not
# library code.
# ruff:file-ignore[magic-value-comparison, suspicious-subprocess-import, start-process-with-partial-path, too-many-arguments]
"""Shared benchmark helpers (build, score, metrics, IO).

Reuses the autoSAFE public API; the only autoSAFE-internal call is
``Samples._find_closest_samples`` (to assign each kernel's nearest
neighbor), mirroring the production construction path. The calibration
distance is computed independently with an exact ``KDTree`` — there is
no ``Samples._closest_indices`` attribute to read.
"""

import json
import subprocess
import time
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import numpy.typing as npt
from scipy.spatial import KDTree
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)

from autosafe.kernels.rbf import (
    SIGMA_FLOOR_RATIO,
    RBFKernel,
    calibrate_rbf_scale_d_tilde,
)
from autosafe.preprocessing import RangeNormalizer
from autosafe.sample import Sample
from autosafe.samples import Samples

NPArray = npt.NDArray[np.float64]


def build_odd(
    points: NPArray,
    *,
    mode: str,
    gamma: float = 1.0,
    s: float = 3.0,
    kappa: float = 1.0,
    eta: float = 1.0,
    lam: float | None = None,
) -> Samples:
    """Build an autoSAFE RBF ODD from anchor points.

    Args:
        points (NPArray): (N, D) ID anchor points (already in the
            desired coordinate system).
        mode (str): ``"fixed"`` -> kappa=eta=1,
            lam=SIGMA_FLOOR_RATIO*kappa (= e^-10, the paper's Monte
            Carlo parameters); ``"calibrated"``
            -> kappa=(s*d_tilde)**2, eta=gamma/d_tilde,
            lam=SIGMA_FLOOR_RATIO*kappa; ``"manual"`` -> the explicit
            ``kappa``/``eta``/``lam`` given.
        gamma (float): Calibrated-mode decay constant (ignored
            otherwise).
        s (float): Calibrated-mode width multiple (ignored otherwise).
        kappa (float): Manual-mode scale (ignored otherwise).
        eta (float): Manual-mode decay (ignored otherwise).
        lam (float | None): Manual-mode floor; defaults to
            ``SIGMA_FLOOR_RATIO * kappa`` when None.

    Returns:
        Samples: The constructed ODD; call ``odd(X)`` or
            ``odd.affinity_dual(X)``.

    Raises:
        ValueError: If fewer than 2 anchors or unknown mode.
    """
    pts = np.ascontiguousarray(np.asarray(points, dtype=float))
    if pts.shape[0] < 2:
        raise ValueError("autoSAFE calibration requires N >= 2 anchors")
    odd = Samples(
        [Sample(x=row.copy()) for row in pts],
        closest_sample_mode="global",
        kernel_cls="RBF",
        skip_updates=True,
    )
    odd._find_closest_samples()  # ruff:ignore[private-member-access]  -- assigns each kernel's x_nn (exact)
    if mode == "fixed":
        # Camera-ready MCM floor is relative (lam_rel = e^-10), not the
        # old absolute 1e-15; with kappa=1, lam = SIGMA_FLOOR_RATIO * 1
        # = e^-10.
        override: dict[str, object] = {
            "kappa": 1.0,
            "eta": 1.0,
            "lam": SIGMA_FLOOR_RATIO,
        }
    elif mode == "calibrated":
        # Distance from each anchor to its nearest OTHER anchor (k=2 ->
        # col 1), exactly the d_l2 the production calibrator consumes.
        d_l2 = KDTree(pts).query(pts, k=2)[0][:, 1]
        kappa_c, eta_c = calibrate_rbf_scale_d_tilde(d_l2, gamma=gamma, s=s)
        override = {
            "kappa": kappa_c,
            "eta": eta_c,
            "lam": SIGMA_FLOOR_RATIO * kappa_c,
        }
    elif mode == "manual":
        override = {
            "kappa": float(kappa),
            "eta": float(eta),
            "lam": float(lam) if lam is not None else SIGMA_FLOOR_RATIO * float(kappa),
        }
    else:
        raise ValueError(f"unknown mode {mode!r}")
    odd.refresh_kernels(kernel_kwargs_override=override)
    return odd


def kernel_sigma(sample: Sample) -> NPArray:
    """Covariance ``sigma`` of an anchor's RBF kernel.

    Args:
        sample (Sample): An anchor of a built ODD.

    Returns:
        NPArray: (D, D) kernel covariance.

    Raises:
        TypeError: If the anchor has no RBF kernel.
    """
    kern = sample.kernel
    if not isinstance(kern, RBFKernel):
        raise TypeError(f"anchor kernel is {type(kern).__name__}, not RBFKernel")
    return np.asarray(kern.sigma, dtype=float)


def score_autosafe(odd: Samples, x: NPArray) -> NPArray:
    """Continuous membership score: linear-space affinity in [0, 1].

    Args:
        odd (Samples): The constructed ODD.
        x (NPArray): (M, D) query points.

    Returns:
        NPArray: (M,) affinity ``alpha`` of each query point.
    """
    alpha, _ = odd.affinity_dual(np.asarray(x, dtype=float))
    return np.asarray(alpha, dtype=float)


def normalize_fit_apply(
    id_points: NPArray, *others: NPArray
) -> tuple[RangeNormalizer, list[NPArray]]:
    """Fit a [-1, 1] min-max normalizer on ID points; apply to all.

    Min-max (not IQR) keeps synthetic ground-truth geometry exactly
    mappable. The returned transformed arrays are
    ``[id_points, *others]``.

    Args:
        id_points (NPArray): ID anchor points (used to fit).
        others (NPArray): Additional point arrays to transform
            identically.

    Returns:
        tuple[RangeNormalizer, list[NPArray]]: fitted normalizer and the
            list of transformed arrays (ID first).
    """
    norm = RangeNormalizer(target_range=(-1.0, 1.0), method="minmax")
    norm.fit(np.asarray(id_points, dtype=float))
    transformed = [
        np.asarray(norm.transform(np.asarray(a, dtype=float)), dtype=float)
        for a in (id_points, *others)
    ]
    return norm, transformed


def classification_metrics(scores: NPArray, labels: npt.NDArray[np.bool_]) -> dict:
    """Threshold-free metrics for a continuous score vs. binary labels.

    Args:
        scores (NPArray): Higher = more "inside". Shape (M,).
        labels (npt.NDArray[np.bool_]): Ground-truth membership, shape
            (M,).

    Returns:
        dict: ``aupr``, ``auroc``, ``max_f1``, ``p_at_r90``,
            ``r_at_p95``. Degenerate single-class cases return NaN for
            the affected keys.
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=bool)
    out = {
        "aupr": float("nan"),
        "auroc": float("nan"),
        "max_f1": float("nan"),
        "p_at_r90": float("nan"),
        "r_at_p95": float("nan"),
    }
    if labels.all() or (~labels).all():
        return out  # metrics undefined with a single class present
    out["aupr"] = float(average_precision_score(labels, scores))
    out["auroc"] = float(roc_auc_score(labels, scores))
    precision, recall, _ = precision_recall_curve(labels, scores)
    f1 = np.where(
        (precision + recall) > 0, 2 * precision * recall / (precision + recall), 0.0
    )
    out["max_f1"] = float(np.max(f1))
    mask_r = recall >= 0.9
    out["p_at_r90"] = float(np.max(precision[mask_r])) if mask_r.any() else 0.0
    mask_p = precision >= 0.95
    out["r_at_p95"] = float(np.max(recall[mask_p])) if mask_p.any() else 0.0
    return out


def pr_curves(
    scores: NPArray, labels: npt.NDArray[np.bool_], thresholds: NPArray
) -> tuple[NPArray, NPArray]:
    """Precision and recall of ``scores >= t`` for each threshold ``t``.

    Args:
        scores (NPArray): Continuous membership scores, shape (M,).
        labels (npt.NDArray[np.bool_]): Ground-truth membership, shape
            (M,).
        thresholds (NPArray): Thresholds to sweep, shape (T,).

    Returns:
        tuple[NPArray, NPArray]: (precision, recall) each shape (T,).
            Precision is defined as 1.0 where no point is predicted
            inside.
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=bool)
    pos = labels.sum()
    prec = np.empty(len(thresholds))
    rec = np.empty(len(thresholds))
    for i, t in enumerate(thresholds):
        pred = scores >= t
        tp = int(np.count_nonzero(pred & labels))
        pp = int(np.count_nonzero(pred))
        prec[i] = tp / pp if pp > 0 else 1.0
        rec[i] = tp / pos if pos > 0 else 0.0
    return prec, rec


def curve_r2(curve_target: NPArray, curve_pred: NPArray) -> float:
    """Curve R^2 with ``curve_target`` as the reference.

    Matches the paper's definition:
    ``R^2 = 1 - SS_res/SS_tot`` with the underlying-ODD curve as the
    regression target.

    Args:
        curve_target (NPArray): Reference curve (e.g. ODD-referenced),
            (T,).
        curve_pred (NPArray): Comparison curve (e.g. hull-referenced),
            (T,).

    Returns:
        float: The curve-R^2 (1.0 for identical curves; can be
            negative).
    """
    target = np.asarray(curve_target, dtype=float)
    pred = np.asarray(curve_pred, dtype=float)
    ss_res = float(np.sum((target - pred) ** 2))
    ss_tot = float(np.sum((target - np.mean(target)) ** 2))
    if np.isclose(ss_tot, 0.0):
        return 1.0 if np.isclose(ss_res, 0.0) else float("nan")
    return 1.0 - ss_res / ss_tot


def write_dat(path: Path, columns: Mapping[str, npt.ArrayLike]) -> None:
    """Write whitespace-separated, pgfplots-ready columns with a header.

    Args:
        path (Path): Output ``.dat`` path (parent dirs are created).
        columns (Mapping[str, ArrayLike]): Ordered column-name ->
            values.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    names = list(columns)
    arrays = [np.atleast_1d(np.asarray(columns[n])) for n in names]
    n_rows = max((a.shape[0] for a in arrays), default=0)
    lines = [" ".join(names)]
    for r in range(n_rows):
        cells = []
        for a in arrays:
            v = a[r] if r < a.shape[0] else ""
            cells.append(v if isinstance(v, str) else f"{float(v):.8g}")
        lines.append(" ".join(str(c) for c in cells))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _git_hash() -> str:
    """Return the current git commit hash, or 'unknown'."""
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return "unknown"


def write_config(outdir: Path, config: dict, *, start_time: float) -> None:
    """Write ``config.json`` with config, git hash, and wall time.

    Args:
        outdir (Path): Output directory (created if needed).
        config (dict): Experiment configuration to record.
        start_time (float): ``time.perf_counter()`` at experiment start.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    payload = {
        **config,
        "git_hash": _git_hash(),
        "wall_time_seconds": time.perf_counter() - start_time,
    }
    (outdir / "config.json").write_text(json.dumps(payload, default=str, indent=2))


def default_thresholds(count: int = 256) -> NPArray:
    """Return a uniform ζ grid on [0, 1] for curve sweeps.

    Args:
        count (int): Number of thresholds.

    Returns:
        NPArray: ``count`` evenly spaced thresholds from 0 to 1.
    """
    return np.linspace(0.0, 1.0, count)


def level_set_mask(odd: Samples, x: NPArray, zeta: float) -> npt.NDArray[np.bool_]:
    """Boolean membership of ``x`` in the ODD at threshold ``zeta``.

    Args:
        odd (Samples): The constructed ODD.
        x (NPArray): (M, D) query points.
        zeta (float): Membership threshold; inside iff
            ``alpha >= zeta``.

    Returns:
        npt.NDArray[np.bool_]: (M,) membership mask.
    """
    return np.asarray(score_autosafe(odd, x) >= zeta, dtype=bool)


def iou(mask_a: npt.NDArray[np.bool_], mask_b: npt.NDArray[np.bool_]) -> float:
    """Intersection-over-union of two boolean masks on a shared grid.

    Args:
        mask_a (npt.NDArray[np.bool_]): (M,) mask over the evaluation
            points.
        mask_b (npt.NDArray[np.bool_]): (M,) mask over the SAME points.

    Returns:
        float: intersection over union; 1.0 when both masks are empty.
    """
    a = np.asarray(mask_a, dtype=bool)
    b = np.asarray(mask_b, dtype=bool)
    union = int(np.count_nonzero(a | b))
    return 1.0 if union == 0 else int(np.count_nonzero(a & b)) / union


def boundary_error(
    x: NPArray, labels: npt.NDArray[np.bool_], pred: npt.NDArray[np.bool_]
) -> dict:
    """Mean/max distance of misclassified points to the true boundary.

    The distance-to-boundary of a misclassified evaluation point is
    approximated by its Euclidean distance to the nearest evaluation
    point of the OPPOSITE true label: a false positive is truly outside,
    so its nearest truly-inside point sits just across the boundary (and
    vice versa for false negatives). This upper-bounds the true boundary
    distance by at most the local evaluation-grid spacing, so it
    tightens with grid density and needs no closed-form boundary
    geometry. Distances are in the coordinate frame of ``x`` — pass the
    RAW (ground-truth) frame for interpretable units.

    Args:
        x (NPArray): (M, D) evaluation points.
        labels (npt.NDArray[np.bool_]): (M,) ground-truth membership.
        pred (npt.NDArray[np.bool_]): (M,) predicted membership.

    Returns:
        dict: ``be_fp_mean``, ``be_fp_max``, ``be_fn_mean``,
            ``be_fn_max``, ``n_fp``, ``n_fn`` (0.0 distances when the
            corresponding error set is empty).
    """
    x = np.asarray(x, dtype=float)
    labels = np.asarray(labels, dtype=bool)
    pred = np.asarray(pred, dtype=bool)
    out = {
        "be_fp_mean": 0.0,
        "be_fp_max": 0.0,
        "be_fn_mean": 0.0,
        "be_fn_max": 0.0,
        "n_fp": 0,
        "n_fn": 0,
    }
    if labels.all() or (~labels).all():
        return out  # no boundary crossing representable on this grid
    tree_in = KDTree(x[labels])
    tree_out = KDTree(x[~labels])
    fp = pred & ~labels  # truly outside -> distance to nearest inside point
    fn = ~pred & labels  # truly inside -> distance to nearest outside point
    if fp.any():
        d = tree_in.query(x[fp], k=1, workers=1)[0]
        out["be_fp_mean"] = float(np.mean(d))
        out["be_fp_max"] = float(np.max(d))
        out["n_fp"] = int(fp.sum())
    if fn.any():
        d = tree_out.query(x[fn], k=1, workers=1)[0]
        out["be_fn_mean"] = float(np.mean(d))
        out["be_fn_max"] = float(np.max(d))
        out["n_fn"] = int(fn.sum())
    return out


def confusion_at(scores: NPArray, labels: npt.NDArray[np.bool_], zeta: float) -> dict:
    """Confusion metrics for ``scores >= zeta`` vs. labels.

    Complements the threshold-free :func:`classification_metrics`;
    reports the operating-point numbers reviewers ask for (precision,
    recall, FP rate, IoU).

    Args:
        scores (NPArray): (M,) continuous membership scores.
        labels (npt.NDArray[np.bool_]): (M,) ground-truth membership.
        zeta (float): Membership threshold.

    Returns:
        dict: ``tp``, ``fp``, ``tn``, ``fn``, ``precision``, ``recall``,
            ``specificity``, ``fpr``, ``f1``, ``iou``.
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=bool)
    pred = scores >= zeta
    tp = int(np.count_nonzero(pred & labels))
    fp = int(np.count_nonzero(pred & ~labels))
    tn = int(np.count_nonzero(~pred & ~labels))
    fn = int(np.count_nonzero(~pred & labels))
    prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 1.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    iou_val = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 1.0
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": prec,
        "recall": rec,
        "specificity": spec,
        "fpr": fpr,
        "f1": f1,
        "iou": iou_val,
    }
