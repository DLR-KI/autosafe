# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""Operational Design Domain construction and comparison APIs."""

from autosafe.odd.config import (
    PARAMETER_SPECS,
    AutoSafeConfig,
    CalibratedRBFConfig,
    ConformalMembershipConfig,
    EvaluationConfig,
    FixedMembershipConfig,
    ManualRBFConfig,
    NormalizationConfig,
    OODConsistencyConfig,
    ParameterSpec,
    ResolvedAutoSafeConfig,
    ResolvedKernelConfig,
    ResolvedMembershipConfig,
)
from autosafe.odd.membership import conformal_membership_threshold
from autosafe.odd.model import AutoSafeODD
from autosafe.odd.openodd import (
    OpenODDCategoricalFeature,
    OpenODDExportMetadata,
    OpenODDNumericFeature,
    build_openodd_yaml,
    write_openodd_yaml,
)
from autosafe.odd.openodd_schema import (
    OPENODD_1_0_YAML_SCHEMA,
    OPENODD_VERSION,
    OpenODDValidationError,
    dump_openodd_yaml,
    load_openodd_yaml,
    parse_openodd_yaml,
    validate_openodd_document,
)
from autosafe.typing import OpenODDFeature

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
    "ManualRBFConfig",
    "NormalizationConfig",
    "OODConsistencyConfig",
    "OpenODDCategoricalFeature",
    "OpenODDExportMetadata",
    "OpenODDFeature",
    "OpenODDNumericFeature",
    "OpenODDValidationError",
    "ParameterSpec",
    "ResolvedAutoSafeConfig",
    "ResolvedKernelConfig",
    "ResolvedMembershipConfig",
    "build_openodd_yaml",
    "conformal_membership_threshold",
    "dump_openodd_yaml",
    "load_openodd_yaml",
    "parse_openodd_yaml",
    "validate_openodd_document",
    "write_openodd_yaml",
]
