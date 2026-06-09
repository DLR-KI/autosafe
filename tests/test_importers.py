# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT

import pathlib
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from autosafe import ROOT_FOLDER
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


def test_dataset_with_singular_matrix_triggers_warning():
    """Test that importing a dataset that leads to a singular matrix triggers a
    warning."""
    file = ROOT_FOLDER / "data" / "breast-cancer-wisconsin.csv"
    kernel_cls = "RBF"
    kernel_kwargs = None

    with pytest.warns(UserWarning, match="sigma matrix was not invertible"):
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
