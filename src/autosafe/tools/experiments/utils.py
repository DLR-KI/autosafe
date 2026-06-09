# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""Utility functions for experiments."""

import dataclasses
import datetime
import logging
import pathlib
import sys
from collections.abc import Callable
from typing import Any, Literal, cast

import numpy as np
import orjson
import polars as pl
import polars.selectors as cs

from autosafe.preprocessing import create_robust_normalization_pipeline
from autosafe.tools.experiments.core import DatasetConfig, DatasetType, ExperimentType

RANGE_FILTER_BOUNDS = 2


@dataclasses.dataclass(frozen=True)
class DatasetLoadOptions:
    """Options used when loading a dataset.

    Attributes:
        file_type (str | None): Optional explicit file type (e.g.,
            "csv", "json", "parquet", "numpy"). If None, inferred from
            file extension.
        filters (dict[str, Any] | None): Optional column filters to
            apply after loading. Keys are column names, values are
            filter conditions.
        normalize (bool): Whether to apply normalization to numeric
            columns.
        normalization_method (Literal['iqr', 'minmax']): Method for
            robust normalization. "iqr" scales based on interquartile
            range, while "minmax" scales to a fixed range.
        polars_kwargs (dict[str, Any] | None): Additional keyword
            arguments to pass to Polars read functions (e.g.,
            `read_csv`, `read_parquet`).
    """

    file_type: str | None = None
    filters: dict[str, Any] | None = None
    normalize: bool = True
    normalization_method: Literal["iqr", "minmax"] = "iqr"
    polars_kwargs: dict[str, Any] | None = None


def load_dataset(
    file_path: pathlib.Path,
    options: DatasetLoadOptions | None = None,
) -> tuple[pl.DataFrame, str]:
    """Load and process a dataset from various file formats.

    Args:
        file_path (pathlib.Path): Path to the dataset file.
        options (DatasetLoadOptions | None): Optional loading options.

    Returns:
        tuple[pl.DataFrame, str]: DataFrame, dataset_type

    Raises:
        NotImplementedError: If NumPy array loading is requested.
        ValueError: If file format is unsupported.
    """
    options = options or DatasetLoadOptions()
    file_type = options.file_type
    filters = options.filters
    normalize = options.normalize
    normalization_method = options.normalization_method
    polars_kwargs = options.polars_kwargs

    if file_type is None:
        file_type = file_path.suffix.lower()[1:]  # Remove dot

    if file_type == "csv":
        df = pl.read_csv(file_path, **(polars_kwargs or {}))
    elif file_type in {"json", "jsonl"}:
        df = pl.read_ndjson(file_path, **(polars_kwargs or {}))
    elif file_type == "parquet":
        df = pl.read_parquet(file_path, **(polars_kwargs or {}))
    elif file_type == "npy":
        raise NotImplementedError("NumPy array loading not yet implemented")
    else:
        raise ValueError(f"Unsupported file type: {file_type}")

    # Apply filters if provided
    if filters:
        df = apply_filters(df, filters)

    if normalize:
        df = normalize_dataset(
            df,
            method=normalization_method,
        )

    return df, file_type


def normalize_dataset(
    df: pl.DataFrame,
    method: Literal["iqr", "minmax"] = "iqr",
) -> pl.DataFrame:
    """Normalize numeric columns to improve kernel-affinity stability.

    Args:
        df (pl.DataFrame): Input dataset.
        method (Literal["iqr", "minmax"]): Robust normalization method.

    Returns:
        pl.DataFrame: Normalized numeric columns and unchanged
        non-numeric columns.
    """
    numeric_columns = df.select(cs.numeric()).columns
    if not numeric_columns:
        return df

    numeric_data = df.select(numeric_columns).to_numpy().astype(float)
    normalizer = create_robust_normalization_pipeline(
        target_range=(-1.0, 1.0),
        method=method,
    )
    normalized = np.asarray(normalizer.fit_transform(numeric_data))

    normalized_df = pl.DataFrame(
        normalized,
        schema=numeric_columns,
        orient="row",
    )

    return df.with_columns([normalized_df[col].alias(col) for col in numeric_columns])


def apply_filters(df: pl.DataFrame, filters: dict[str, Any]) -> pl.DataFrame:
    """Apply filtering conditions to the dataset.

    Args:
        df (pl.DataFrame): Input dataframe.
        filters (dict[str, Any]): Column filters to apply.

    Returns:
        pl.DataFrame: Filtered DataFrame if any filter expressions are
            provided.
    """
    filter_expr = None

    for column, condition in filters.items():
        if column not in df.columns:
            continue

        if isinstance(condition, (tuple, list)) and len(condition) == (
            RANGE_FILTER_BOUNDS
        ):
            # Assume it's a range filter
            col_min, col_max = condition
            column_expr = (pl.col(column) >= col_min) & (pl.col(column) <= col_max)
        else:
            # Assume equality filter
            column_expr = pl.col(column) == condition

        filter_expr = column_expr if filter_expr is None else filter_expr & column_expr

    if filter_expr is not None:
        return df.filter(filter_expr)

    return df


def calculate_dataset_statistics(df: pl.DataFrame) -> dict[str, Any]:
    """Calculate basic statistics for the dataset.

    Args:
        df (pl.DataFrame): Input dataframe.

    Returns:
        dict[str, Any]: Mapping of summary statistics.
    """
    numeric_df = df.select(cs.numeric())
    return {
        "shape": df.shape,
        "columns": df.columns,
        "dtypes": {
            col: str(dtype) for col, dtype in zip(df.columns, df.dtypes, strict=False)
        },
        "null_counts": df.null_count().to_dict(),
        "means": numeric_df.mean().to_dict(),
        "stds": numeric_df.std().to_dict(),
        "mins": numeric_df.min().to_dict(),
        "maxs": numeric_df.max().to_dict(),
    }


def find_dataset_bounds(
    df: pl.DataFrame,
    numeric_columns: list[str] | None = None,
) -> tuple[list[float], list[float]]:
    """Find minimum and maximum values for all numeric columns.

    Args:
        df (pl.DataFrame): Input dataframe.
        numeric_columns (list[str] | None): Optional numeric columns
            to inspect.

    Returns:
        tuple[list[float], list[float]]: lists of minimum and maximum
            values.
    """
    if numeric_columns is None:
        numeric_columns = df.select(cs.numeric()).columns

    min_values = [df.select(pl.col(col).min()).item() for col in numeric_columns]
    max_values = [df.select(pl.col(col).max()).item() for col in numeric_columns]

    return cast("list[float]", min_values), cast("list[float]", max_values)


def save_results(
    results: dict[str, Any] | object,
    export_path: pathlib.Path,
    default: Callable[[object], object] | None = None,
) -> None:
    """Save experiment results to JSON file using orjson.

    Args:
        results (dict[str, Any] | object): Result object or plain
            mapping.
        export_path (pathlib.Path): Destination JSON path.
        default (Callable[[object], object] | None): Optional serializer
            callback.
    """
    # Create parent directory if it doesn't exist
    export_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert results to dict if needed
    to_dict = getattr(results, "to_dict", None)
    data = to_dict() if callable(to_dict) else results

    # Write results
    with export_path.open("wb") as f:
        f.write(
            orjson.dumps(
                data,
                option=orjson.OPT_INDENT_2,
                default=default,
            )
        )


def setup_logging(
    log_level: int = logging.INFO,
    log_file: pathlib.Path | None = None,
    name: str = "autosafe.experiments",
) -> logging.Logger:
    """Setup logging for experiments with optional file output.

    Args:
        log_level (int): Logging level.
        log_file (pathlib.Path | None): Optional file path.
        name (str): Logger name.

    Returns:
        logging.Logger: Configured logger.
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # Remove any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Add console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)

    # Add file handler if requested
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(str(log_file))
        file_handler.setLevel(log_level)
        logger.addHandler(file_handler)

    # Format
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)

    return logger


def load_experiment_results(export_path: pathlib.Path) -> dict[str, Any]:
    """Load experiment results from a JSON file.

    Args:
        export_path (pathlib.Path): Path to the JSON file.

    Returns:
        dict[str, Any]: Parsed JSON payload.
    """
    with export_path.open("rb") as f:
        return orjson.loads(f.read())


def create_dataset_config(
    file_path: pathlib.Path,
    config: DatasetConfig | None = None,
) -> DatasetConfig:
    """Create a DatasetConfig from a file path with automatic detection.

    Args:
        file_path (pathlib.Path): Dataset file path.
        config (DatasetConfig | None): Optional existing dataset
            configuration.

    Returns:
        DatasetConfig: Dataset configuration with inferred dataset type.
    """
    if config is not None:
        config.file_path = file_path
        return config

    detected_type = get_dataset_type(file_path)
    dataset_type = (
        DatasetType.POLARS if detected_type == "parquet" else DatasetType(detected_type)
    )
    return DatasetConfig(file_path=file_path, dataset_type=dataset_type)


def get_dataset_type(file_path: pathlib.Path) -> str:
    """Automatically detect dataset type from file extension.

    Args:
        file_path (pathlib.Path): Dataset file path.

    Returns:
        str: Normalized dataset type string.

    Raises:
        ValueError: If the extension is unknown.
    """
    ext = file_path.suffix.lower()

    if ext == ".csv":
        return "csv"
    if ext in {".json", ".jsonl"}:
        return "json"
    if ext == ".parquet":
        return "parquet"
    if ext in {".npy", ".npz"}:
        return "numpy"
    raise ValueError(f"Unknown dataset type for extension: {ext}")


def construct_experiment_id(
    experiment_type: ExperimentType,
    dataset_path: pathlib.Path,
    timestamp: datetime.datetime | None = None,
    suffix: str | None = None,
) -> str:
    """Construct a standardized experiment ID.

    Args:
        experiment_type (ExperimentType): Experiment type.
        dataset_path (pathlib.Path): Dataset file path.
        timestamp (datetime.datetime | None): Optional timestamp.
        suffix (str | None): Optional suffix.

    Returns:
        str: Standardized experiment identifier.
    """
    if timestamp is None:
        timestamp = datetime.datetime.now(datetime.timezone.utc)

    timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")
    dataset_stem = dataset_path.stem.replace("-", "_")
    suffix_str = f"_{suffix}" if suffix else ""

    return f"{experiment_type.value}_{timestamp_str}_{dataset_stem}{suffix_str}"


__all__ = [
    "DatasetLoadOptions",
    "apply_filters",
    "calculate_dataset_statistics",
    "construct_experiment_id",
    "create_dataset_config",
    "find_dataset_bounds",
    "get_dataset_type",
    "load_dataset",
    "load_experiment_results",
    "normalize_dataset",
    "save_results",
    "setup_logging",
]
