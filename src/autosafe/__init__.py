# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""Type annotations for the autoSAFE framework."""

import logging
import pathlib
import warnings

from tqdm import TqdmExperimentalWarning

from autosafe import _jax_config  # ruff:ignore[unused-import]
from autosafe.kernels.laplacian import LaplacianKernel
from autosafe.kernels.rbf import RBFKernel
from autosafe.odd import (
    OPENODD_1_0_YAML_SCHEMA,
    OPENODD_VERSION,
    PARAMETER_SPECS,
    AutoSafeConfig,
    AutoSafeODD,
    CalibratedRBFConfig,
    ConformalMembershipConfig,
    EvaluationConfig,
    FixedMembershipConfig,
    ManualRBFConfig,
    NormalizationConfig,
    OODConsistencyConfig,
    OpenODDCategoricalFeature,
    OpenODDExportMetadata,
    OpenODDNumericFeature,
    OpenODDValidationError,
    build_openodd_yaml,
    dump_openodd_yaml,
    load_openodd_yaml,
    parse_openodd_yaml,
    validate_openodd_document,
    write_openodd_yaml,
)
from autosafe.sample import Sample
from autosafe.samples import Samples
from autosafe.tools.exporters import to_json
from autosafe.tools.importers import (
    from_csv,
    from_json,
    from_numpy,
    from_polars,
)
from autosafe.typing import OpenODDFeature

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
    "OPENODD_1_0_YAML_SCHEMA",
    "OPENODD_VERSION",
    "PARAMETER_SPECS",
    "AutoSafeConfig",
    "AutoSafeODD",
    "CalibratedRBFConfig",
    "ConformalMembershipConfig",
    "EvaluationConfig",
    "FixedMembershipConfig",
    "LaplacianKernel",
    "ManualRBFConfig",
    "NormalizationConfig",
    "OODConsistencyConfig",
    "OpenODDCategoricalFeature",
    "OpenODDExportMetadata",
    "OpenODDFeature",
    "OpenODDNumericFeature",
    "OpenODDValidationError",
    "RBFKernel",
    "Sample",
    "Samples",
    "build_openodd_yaml",
    "dump_openodd_yaml",
    "from_csv",
    "from_json",
    "from_numpy",
    "from_polars",
    "load_openodd_yaml",
    "parse_openodd_yaml",
    "to_json",
    "validate_openodd_document",
    "write_openodd_yaml",
]
