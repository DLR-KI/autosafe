# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT

import pathlib
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from autosafe import ROOT_FOLDER
from autosafe.deduplication import DeduplicationPolicy, deduplicate_points
from autosafe.sample import Sample
from autosafe.samples import Samples
from autosafe.tools.importers import from_csv, from_json, from_numpy, from_polars
from autosafe.typing import ClosestSampleModeType, FloatType, KernelType

CSV_FILES = [
    ROOT_FOLDER / "data" / "iris.csv",
    ROOT_FOLDER / "data" / "WineQT.csv",
    ROOT_FOLDER / "data" / "breast-cancer-wisconsin.csv",  # Leads to singular matrix
]

CLOSEST_SAMPLE_MODES = [
    "global",
    "per_dimension",
]

KERNELS = [
    ("RBF", None),
    ("RBF", {"sigma": "eye"}),
    ("RBF", {"kappa": 0.1, "eta": 4.0}),
    ("Laplacian", {"alpha": 0.5}),
]

testdata = [
    (file, mode, kernel_cls, kernel_kwargs)
    for file in CSV_FILES
    for mode in CLOSEST_SAMPLE_MODES
    for kernel_cls, kernel_kwargs in KERNELS
]


@pytest.mark.filterwarnings("ignore:sigma matrix was not invertible")
@pytest.mark.parametrize(
    ("file", "closest_sample_mode", "kernel_cls", "kernel_kwargs"), testdata
)
def test_from_csv(
    file: Path | str,
    closest_sample_mode: ClosestSampleModeType,
    kernel_cls: KernelType,
    kernel_kwargs: dict[str, Any] | None,
):
    """Test the csv importer."""
    samples = from_csv(
        file=file,
        closest_sample_mode=closest_sample_mode,
        kernel_cls=kernel_cls,
        kernel_kwargs=kernel_kwargs,
    )
    assert isinstance(samples, Samples)

    data = np.genfromtxt(file, delimiter=",", skip_header=1, dtype=FloatType)

    assert samples.shape == data.shape


def test_dataset_no_singular_matrix_warning_with_affine_floor():
    """The affine-floor sigma law prevents singular matrices on isolated anchors.

    The breast-cancer-wisconsin dataset previously triggered the
    "sigma matrix was not invertible" UserWarning. With the affine lower
    bound (SIGMA_FLOOR_RATIO * kappa) that warning must no longer be emitted.
    """
    import warnings

    file = ROOT_FOLDER / "data" / "breast-cancer-wisconsin.csv"
    kernel_cls = "RBF"
    kernel_kwargs = None

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        # Must NOT raise a UserWarning about singular matrices
        samples = from_csv(
            file=file, kernel_cls=kernel_cls, kernel_kwargs=kernel_kwargs
        )

    assert isinstance(samples, Samples)


def test_from_json(tmp_path: pathlib.Path):
    """Test from_json imports from JSON file."""
    # First create Samples and export it
    original = Samples(
        samples=[Sample(x=[1.0, 2.0, 3.0])],
        closest_sample_mode="global",
        kernel_cls="RBF",
    )
    json_file = tmp_path / "samples.json"

    # Export using msgspec
    import msgspec.json

    from autosafe.tools.serializers.msgspec import encode_hook

    json_file.write_bytes(msgspec.json.Encoder(enc_hook=encode_hook).encode(original))

    # Now import it back
    imported = from_json(str(json_file))
    assert isinstance(imported, Samples)
    assert len(imported.samples) == 1


def test_from_polars():
    """Test from_polars imports from Polars DataFrame."""
    import polars as pl

    df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})
    samples = from_polars(df)
    assert isinstance(samples, Samples)
    assert samples.shape == (3, 2)


def test_from_numpy():
    """Test from_numpy imports from NumPy array."""
    import numpy as np

    data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    samples = from_numpy(data)
    assert isinstance(samples, Samples)
    assert samples.shape == (3, 2)


def test_from_numpy_dedup_disabled_by_default_matches_no_kwarg():
    """Omitting dedup_policy and passing dedup_policy=None are identical."""
    data = np.array([[1.0, 2.0], [1.0, 2.0], [3.0, 4.0]])
    default_samples = from_numpy(data)
    explicit_none_samples = from_numpy(data, dedup_policy=None)
    assert default_samples.shape == explicit_none_samples.shape == (3, 2)


def test_from_numpy_dedup_enabled_collapses_duplicates():
    """dedup_policy set collapses exact duplicates before Samples is built."""
    data = np.array([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0], [9.0, 9.0]])
    policy = DeduplicationPolicy(resolution=(0.5, 0.5), origin=(0.0, 0.0))
    samples = from_numpy(data, dedup_policy=policy)
    assert samples.shape == (2, 2)


def test_from_csv_dedup_matches_dataset_workflow_representatives(tmp_path: Path):
    """The importer and the dataset workflow agree on representatives.

    Both ultimately call ``deduplicate_points`` on the same raw numeric
    array; this checks that from_csv's wiring does not diverge from
    calling the core function directly on the same CSV-derived array.
    """
    rows = [
        "x0,x1",
        "0.100000,0.200000",
        "0.100000,0.200000",
        "5.000000,5.000000",
        "5.010000,5.010000",
    ]
    csv_path = tmp_path / "dedup_source.csv"
    csv_path.write_text("\n".join(rows), encoding="utf-8")

    policy = DeduplicationPolicy(resolution=(0.5, 0.5), origin=(0.0, 0.0))
    samples = from_csv(csv_path, dedup_policy=policy)

    import polars as pl

    raw = pl.read_csv(csv_path).to_numpy().astype(float)
    expected = deduplicate_points(raw, policy)

    got_points = np.array([np.asarray(s.x, dtype=float) for s in samples.samples])
    np.testing.assert_array_equal(
        got_points[np.lexsort(got_points.T[::-1])],
        expected.points[np.lexsort(expected.points.T[::-1])],
    )
