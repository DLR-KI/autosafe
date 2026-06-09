# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""Type annotations for the autoSAFE framework."""

import logging
import pathlib
import warnings

from tqdm import TqdmExperimentalWarning

from autosafe import _jax_config  # noqa: F401
from autosafe.kernels.laplacian import LaplacianKernel
from autosafe.kernels.rbf import RBFKernel
from autosafe.sample import Sample
from autosafe.samples import Samples
from autosafe.tools.exporters import to_json
from autosafe.tools.importers import (
    from_csv,
    from_json,
    from_numpy,
    from_polars,
)

# Disable tqdm ExperimentalWarnings
warnings.filterwarnings(
    action="ignore",
    category=TqdmExperimentalWarning,
)

# Disable polytope logging as it will warn about missing solvers
logging.getLogger("polytope").setLevel(logging.ERROR)

PACKAGE_FOLDER = pathlib.Path(__file__).parent
ROOT_FOLDER: pathlib.Path = PACKAGE_FOLDER.parent.parent


__all__ = [
    "LaplacianKernel",
    "RBFKernel",
    "Sample",
    "Samples",
    "from_csv",
    "from_json",
    "from_numpy",
    "from_polars",
    "to_json",
]
