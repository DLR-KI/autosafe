# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""Dataset-mode wiring for resolution-cell de-duplication.

Ships disabled by default: every function here is reached only when
the caller passes an explicit
:class:`~autosafe.deduplication.DeduplicationPolicy`. With no policy,
``evaluate_dataset_mode`` never imports or calls anything in this
module, so the existing (non-de-duplicated) pipeline is
byte-identical.

Implements the pipeline order fixed by Algorithm 1 for the dataset
workflow:

1.  Load raw data with stable record ids (the row index).
2.  Resolve/validate the de-duplication policy (done by the caller
    before invoking this module, via :class:`DeduplicationPolicy`
    construction).
3.  Reserve split-conformal calibration records: they retain their
    observed frequency and never instantiate kernels.
4.  Subtract OOD rows (exact match, in raw coordinates).
5.  De-duplicate the ID candidates and, independently, the OOD set under
    the same policy.
6.  Fit/apply the normalizer on the retained ID representatives only.
7.  Re-check ID/OOD disjointness in the (now normalized) space used by
    the OOD consistency loop.
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import polars as pl

from autosafe.deduplication import (
    DeduplicationPolicy,
    DeduplicationResult,
    check_label_conflicts,
    deduplicate_points,
)
from autosafe.exceptions import EmptyAnchorPoolError, OODAnchorCoincidenceError
from autosafe.pointsets import rows_in
from autosafe.preprocessing import RangeNormalizer, create_robust_normalization_pipeline
from autosafe.tools.evaluate.dataset.normalization import _numeric_array
from autosafe.tools.evaluate.dataset.odd_cache import _dedup_filename_tag
from autosafe.tools.experiments.utils import DatasetLoadOptions, load_dataset

if TYPE_CHECKING:
    from autosafe.typing import NPMatrix


@dataclass(frozen=True)
class DatasetDeduplicationOutcome:
    """Result of applying dataset-level de-duplication to one dataset.

    Attributes:
        id_result (DeduplicationResult): De-duplication result for the
            ID anchor candidates, in raw coordinates, after calibration
            reservation and OOD subtraction.
        ood_result (DeduplicationResult | None): De-duplication result
            for the OOD constraint set under the same policy, or
            ``None`` when no OOD file was given.
        id_points_normalized (NPMatrix): ID representatives after
            fitting/applying the normalizer to them.
        ood_points_normalized (NPMatrix | None): OOD representatives in
            the same normalized space, or ``None``.
        normalizer (RangeNormalizer | None): The normalizer fit on the
            ID representatives (or the supplied YAML normalizer),
            or ``None`` when normalization is disabled.
        calibration_record_ids (tuple[int, ...]): Raw row indices
            reserved for split-conformal calibration, held out before
            de-duplication and never instantiated as kernels.
        calibration_points_raw (NPMatrix): Raw coordinates of the
            reserved calibration records, in original acquisition
            multiplicity (never de-duplicated).
        label_conflicts (tuple[tuple[int, ...], ...]): Cell-id tuples
            occupied by both ``id_result`` and ``ood_result``.
        filename_tag (str): The ``-dedup<digest>`` cache tag derived
            from the algorithm version, the resolved policy, and the ID
            representative coordinates.
    """

    id_result: DeduplicationResult
    ood_result: "DeduplicationResult | None"
    id_points_normalized: "NPMatrix"
    ood_points_normalized: "NPMatrix | None"
    normalizer: "RangeNormalizer | None"
    calibration_record_ids: tuple[int, ...]
    calibration_points_raw: "NPMatrix"
    label_conflicts: tuple[tuple[int, ...], ...]
    filename_tag: str


def _load_raw_numeric(dataset_path: Path) -> "npt.NDArray[np.float64]":
    """Load a dataset's numeric columns without normalization.

    Args:
        dataset_path (Path): Dataset file path.

    Returns:
        npt.NDArray[np.float64]: Shape ``(n_rows, n_dims)`` raw values.
    """
    df, _ = load_dataset(dataset_path, options=DatasetLoadOptions(normalize=False))
    return _numeric_array(df)


def run_dataset_deduplication(  # ruff:ignore[too-many-arguments, too-many-locals]
    dataset_path: Path,
    *,
    policy: DeduplicationPolicy,
    ood_path: "Path | None",
    yaml_normalizer: "RangeNormalizer | None",
    normalize_data: bool,
    n_calibration_reserved: int = 0,
    strict_label_conflict: bool = False,
) -> DatasetDeduplicationOutcome:
    """Run the full dedup-enabled dataset-mode pre-processing pipeline.

    Args:
        dataset_path (Path): ID dataset path.
        policy (DeduplicationPolicy): Fully resolved de-duplication
            policy, applied in raw coordinates.
        ood_path (Path | None): Optional OOD CSV, de-duplicated
            independently under the same policy.
        yaml_normalizer (RangeNormalizer | None): External normalizer
            fit on YAML ground-truth bounds. When given, it is reused
            as-is (it is not data-fit, so de-duplication order does not
            affect it). When ``None`` and ``normalize_data`` is true, a
            fresh IQR normalizer is fit on the retained ID
            representatives only.
        normalize_data (bool): Whether to normalize at all.
        n_calibration_reserved (int): Number of raw ID rows (the last
            this many by row index) to reserve for split-conformal
            calibration before de-duplication. They retain their
            observed frequency and are excluded from the anchor
            candidate pool entirely.
        strict_label_conflict (bool): Raise instead of only reporting
            when an ID and an OOD resolution cell coincide.

    Returns:
        DatasetDeduplicationOutcome: The de-duplicated, normalized ID
            and OOD point sets plus provenance and cache-tagging
            inputs.

    Raises:
        ValueError: If ``n_calibration_reserved`` is negative or is not
            smaller than the number of raw ID rows.
        EmptyAnchorPoolError: If subtracting the OOD rows (and any
            reserved calibration rows) leaves no ID candidates.
        OODAnchorCoincidenceError: If, after de-duplication and
            normalization, an OOD representative coincides exactly with
            an ID representative.
    """
    id_raw = _load_raw_numeric(dataset_path)
    n_id_raw = id_raw.shape[0]

    if n_calibration_reserved < 0 or n_calibration_reserved >= max(n_id_raw, 1):
        raise ValueError(
            f"n_calibration_reserved={n_calibration_reserved} must be in "
            f"[0, {n_id_raw}) for a dataset with {n_id_raw} raw rows"
        )

    all_ids = np.arange(n_id_raw)
    if n_calibration_reserved > 0:
        calibration_idx = all_ids[n_id_raw - n_calibration_reserved :]
        candidate_idx = all_ids[: n_id_raw - n_calibration_reserved]
    else:
        calibration_idx = all_ids[:0]
        candidate_idx = all_ids

    calibration_points_raw = id_raw[calibration_idx]
    calibration_record_ids = tuple(int(i) for i in calibration_idx)
    candidate_raw = id_raw[candidate_idx]
    candidate_record_ids = [int(i) for i in candidate_idx]

    ood_raw: npt.NDArray[np.float64] | None = None
    if ood_path is not None:
        ood_raw = _load_raw_numeric(ood_path)
        keep = ~rows_in(candidate_raw, ood_raw)
        candidate_raw = candidate_raw[keep]
        candidate_record_ids = [
            rid
            for rid, keep_row in zip(candidate_record_ids, keep, strict=False)
            if keep_row
        ]

    if candidate_raw.shape[0] == 0:
        raise EmptyAnchorPoolError(dataset_path)

    id_result = deduplicate_points(
        candidate_raw, policy, record_ids=candidate_record_ids
    )
    ood_result = (
        deduplicate_points(ood_raw, policy, record_ids=list(range(ood_raw.shape[0])))
        if ood_raw is not None
        else None
    )

    label_conflicts = (
        check_label_conflicts(id_result, ood_result, strict=strict_label_conflict)
        if ood_result is not None
        else ()
    )

    normalizer: RangeNormalizer | None = None
    if yaml_normalizer is not None:
        normalizer = yaml_normalizer
    elif normalize_data:
        # Accepted policy gate: fit on the retained representatives,
        # i.e. de-duplicate first. Fitting on the full (undeduplicated)
        # raw pool would let sampling density back into the geometry
        # through the normalization statistics.
        normalizer = create_robust_normalization_pipeline(
            target_range=(-1.0, 1.0), method="iqr"
        ).fit(id_result.points)

    id_points_normalized = (
        np.asarray(normalizer.transform(id_result.points), dtype=float)
        if normalizer is not None
        else id_result.points
    )
    ood_points_normalized: npt.NDArray[np.float64] | None = None
    if ood_result is not None:
        ood_points_normalized = (
            np.asarray(normalizer.transform(ood_result.points), dtype=float)
            if normalizer is not None
            else ood_result.points
        )
        coincident = rows_in(ood_points_normalized, id_points_normalized)
        if np.any(coincident):
            raise OODAnchorCoincidenceError(
                int(coincident.sum()),
                int(ood_points_normalized.shape[0]),
                int(np.argmax(coincident)),
            )

    filename_tag = _dedup_filename_tag(policy, id_result.points)

    return DatasetDeduplicationOutcome(
        id_result=id_result,
        ood_result=ood_result,
        id_points_normalized=id_points_normalized,
        ood_points_normalized=ood_points_normalized,
        normalizer=normalizer,
        calibration_record_ids=calibration_record_ids,
        calibration_points_raw=calibration_points_raw,
        label_conflicts=label_conflicts,
        filename_tag=filename_tag,
    )


def _provenance_rows(
    result: DeduplicationResult, *, label: str
) -> list[dict[str, object]]:
    """Flatten one de-duplication result into provenance table rows.

    Args:
        result (DeduplicationResult): De-duplication result to flatten.
        label (str): ``"id"`` or ``"ood"``, recorded per row.

    Returns:
        list[dict[str, object]]: One row per (cell, source-record).
    """
    rows: list[dict[str, object]] = []
    for cell_index, (group, representative_id) in enumerate(
        zip(result.groups, result.representative_ids, strict=False)
    ):
        for rank, member_id in enumerate(group):
            rows.append({
                "label": label,
                "cell_index": cell_index,
                "member_record_id": str(member_id),
                "member_rank": rank,
                "is_representative": member_id == representative_id,
            })
    return rows


def write_dedup_provenance(
    dataset_path: Path,
    *,
    outcome: DatasetDeduplicationOutcome,
) -> tuple[Path, Path, str]:
    """Write the representative-to-source-row provenance artifact.

    Source-row groups can be large, so they are kept in a separate
    machine-readable Parquet file (never embedded in the ODD JSON), plus
    a small JSON summary of counts and digests for quick inspection.

    Args:
        dataset_path (Path): ID dataset path; the artifact is written
            beside it.
        outcome (DatasetDeduplicationOutcome): Result of
            :func:`run_dataset_deduplication`.

    Returns:
        tuple[Path, Path, str]: (parquet_path, json_summary_path,
            provenance_digest) where ``provenance_digest`` is the
            SHA-256 hex digest of the Parquet file's bytes.
    """
    rows = _provenance_rows(outcome.id_result, label="id")
    if outcome.ood_result is not None:
        rows.extend(_provenance_rows(outcome.ood_result, label="ood"))
    for rank, record_id in enumerate(outcome.calibration_record_ids):
        rows.append({
            "label": "calibration",
            "cell_index": -1,
            "member_record_id": str(record_id),
            "member_rank": rank,
            "is_representative": True,
        })

    provenance_df = pl.DataFrame(
        rows,
        schema={
            "label": pl.Utf8,
            "cell_index": pl.Int64,
            "member_record_id": pl.Utf8,
            "member_rank": pl.Int64,
            "is_representative": pl.Boolean,
        },
    )

    parquet_path = dataset_path.with_name(
        f"{dataset_path.stem}-dedup-provenance{outcome.filename_tag}.parquet"
    )
    provenance_df.write_parquet(parquet_path)
    provenance_digest = hashlib.sha256(parquet_path.read_bytes()).hexdigest()

    summary = {
        "policy": {
            "resolution": outcome.id_result.policy.resolution,
            "origin": outcome.id_result.policy.origin,
            "coordinate_space": outcome.id_result.policy.coordinate_space,
            "representative_metric": outcome.id_result.policy.representative_metric,
            "tie_break": outcome.id_result.policy.tie_break,
        },
        "id_n_input": outcome.id_result.n_input,
        "id_n_output": outcome.id_result.n_output,
        "id_n_duplicates": outcome.id_result.n_duplicates,
        "id_cell_count_distribution": outcome.id_result.cell_count_distribution(),
        "id_input_digest": outcome.id_result.input_digest,
        "id_output_digest": outcome.id_result.output_digest,
        "ood_n_input": (
            outcome.ood_result.n_input if outcome.ood_result is not None else None
        ),
        "ood_n_output": (
            outcome.ood_result.n_output if outcome.ood_result is not None else None
        ),
        "label_conflicts_count": len(outcome.label_conflicts),
        "label_conflicts_sample": [list(c) for c in outcome.label_conflicts[:10]],
        "calibration_reserved_count": len(outcome.calibration_record_ids),
        "provenance_parquet_path": str(parquet_path),
        "provenance_digest": provenance_digest,
    }
    json_path = dataset_path.with_name(
        f"{dataset_path.stem}-dedup-summary{outcome.filename_tag}.json"
    )
    json_path.write_text(json.dumps(summary, indent=2, default=str))

    return parquet_path, json_path, provenance_digest


__all__ = [
    "DatasetDeduplicationOutcome",
    "run_dataset_deduplication",
    "write_dedup_provenance",
]
