# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""Coverage tests for autosafe.tools.exporters package."""

import pathlib
from pathlib import Path

import msgspec

from autosafe.sample import Sample
from autosafe.tools.exporters import to_json


def test_to_json_with_path(tmp_path: pathlib.Path):
    """Test to_json with Path argument."""
    obj = Sample(x=[1.0, 2.0])
    output_file = tmp_path / "output.json"

    to_json(obj, output_file)

    assert output_file.exists()
    # Verify the file can be read back
    content = output_file.read_bytes()
    decoded = msgspec.json.decode(content)
    assert decoded is not None


def test_to_json_with_string(tmp_path: pathlib.Path):
    """Test to_json with string path argument."""
    obj = Sample(x=[1.0, 2.0])
    output_file = str(tmp_path / "output.json")

    to_json(obj, output_file)

    assert Path(output_file).exists()
    # Verify the file can be read back
    content = Path(output_file).read_bytes()
    decoded = msgspec.json.decode(content)
    assert decoded is not None
