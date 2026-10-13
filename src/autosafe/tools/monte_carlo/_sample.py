# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""Utils module for the Monte Carlo sampling."""

import warnings
from pathlib import Path
from typing import Literal, cast

import numpy as np
import numpy.typing as npt
import orjson
import polytope as pc
import yaml

from autosafe import ROOT_FOLDER
from autosafe.exceptions import (
    ConfigFileNotMappingError,
    DimensionLengthError,
    ParameterOrderError,
    ParameterTypeError,
    ShapeMismatchError,
)
from autosafe.tools.monte_carlo.dicts import (
    KernelConfig,
    MonteCarloConfig,
)
from autosafe.typing import (
    BoundSpec,
    KernelType,
    NPFloatType,
    NPVector,
)


def _load_sampling_config_file(config_file_path: Path) -> MonteCarloConfig:
    """Load Monte Carlo config from JSON or YAML file.

    Supported file formats:

    - ``.json``: legacy sampling config format
    - ``.yaml``/``.yml``: unified config with optional inline ``odd``
        section

    The unified YAML format can embed ODD constraints under either
    ``odd`` or ``odd_config``. This section is mapped to
    ``custom_odd_config`` in the internal config structure.

    Args:
        config_file_path (Path): Path to the configuration file.

    Returns:
        MonteCarloConfig: Parsed Monte Carlo configuration.

    Raises:
        ConfigFileNotMappingError: If a YAML file does not contain a
            mapping.
    """
    suffix = config_file_path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        loaded = yaml.safe_load(config_file_path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ConfigFileNotMappingError(config_file_path)
        config_raw = cast("dict[str, object]", loaded)
    else:
        config_raw = cast(
            "dict[str, object]",
            orjson.loads(config_file_path.read_bytes()),
        )

    if "odd" in config_raw and config_raw.get("odd") is not None:
        config_raw["custom_odd_config"] = config_raw.pop("odd")
    elif "odd_config" in config_raw and config_raw.get("odd_config") is not None:
        config_raw["custom_odd_config"] = config_raw.pop("odd_config")

    return cast("MonteCarloConfig", config_raw)


def create_config(  # ruff:ignore[too-many-arguments, too-many-positional-arguments]
    dim: int | None = None,
    odd_type: Literal["box"] | None = None,
    odd_limits: float | None = None,
    box_limits: float | None = None,
    odd_anchors: int | None = None,
    samples: int | None = None,
    kernel_type: KernelType = "RBF",
    kernel_params: str | None = None,
    filename: str = "sampling_results.json",
    config_file: str | None = None,
    odd_config: str | None = None,
) -> MonteCarloConfig:
    """Create a MonteCarloConfig object from the given parameters.

    Args:
        dim (int | None): The dimension of the space to sample. Must be
            greater than 0.
        odd_type (Literal["box"] | None): The type of the ODD polytope.
            Currently, only "box" is supported.
        odd_limits (float | None): The limits of the ODD box in each
            dimension. This allows to set a symmetric box around the
            origin.
        box_limits (float | None): The limits of the sampling box in
            each dimension. This allows to set a symmetric box around
            the origin.
        odd_anchors (int | None): The number of samples used as anchors
            for the autoSAFE representation of the ODD.
        samples (int | None): The number of samples used to
            quantitatively evaluate the autoSAFE's algorithm
            performance. Recommended to be significantly greater than
            the number of anchors.
        kernel_type (KernelType): The type of the kernel to use. Default
            is "RBF".
        kernel_params (str | None): The parameters for the
            kernel as a JSON string.
        filename (str): Path to the output JSON file where the results
            will be saved.
        config_file (str | None): Path to the configuration file. If
            provided, all other parameters are ignored.
        odd_config (str | None): Path to a YAML file or inline JSON
            describing custom ODD inequalities.

    Returns:
        MonteCarloConfig: The created configuration object.
    """
    if config_file is not None:
        if any(
            param is not None
            for param in (dim, odd_type, odd_limits, box_limits, odd_anchors, samples)
        ):
            warnings.warn(
                "Config file overrides command line parameters.",
                UserWarning,
                stacklevel=2,
            )
        config_file_path = Path(config_file)
        if not config_file_path.is_absolute():
            config_file_path = ROOT_FOLDER / config_file_path

        config = _load_sampling_config_file(config_file_path)

        if "kernel_config" in config:
            warnings.warn(
                "'kernel_config' in Monte Carlo config files is deprecated. "
                "Set kernel_type/kernel_kwargs in the experiment spec instead.",
                DeprecationWarning,
                stacklevel=2,
            )
        if "samples" in config:
            warnings.warn(
                "'samples' in Monte Carlo config files is deprecated. "
                "Set samples/n_samples in the experiment spec instead.",
                DeprecationWarning,
                stacklevel=2,
            )

        config.setdefault(
            "kernel_config",
            KernelConfig(type="RBF", params={}),
        )
        config.setdefault("samples", 1000)

        if not Path(config["filename"]).is_absolute():
            config["filename"] = (
                config_file_path.parent / "results" / config["filename"]
            )
    else:
        dim = dim if dim is not None else 2
        odd_limits = odd_limits if odd_limits is not None else 5.0
        box_limits = box_limits if box_limits is not None else 10.0
        odd_anchors = odd_anchors if odd_anchors is not None else 20
        samples = samples if samples is not None else 1000
        filename_ = Path(filename if filename is not None else "sampling_results.json")

        config = MonteCarloConfig(
            dim=dim,
            odd_type=odd_type if odd_type is not None else "box",
            odd_lower_limits=min(-odd_limits, odd_limits),
            odd_upper_limits=max(-odd_limits, odd_limits),
            box_lower_limits=min(-box_limits, box_limits),
            box_upper_limits=max(-box_limits, box_limits),
            odd_anchors=odd_anchors if odd_anchors is not None else 20,
            samples=samples if samples is not None else 1000,
            kernel_config=KernelConfig(
                type=kernel_type,
                params=orjson.loads(kernel_params) if kernel_params else {},
            ),
            filename=filename_ if filename_.is_absolute() else ROOT_FOLDER / filename_,
            custom_odd_config=odd_config,
        )

    return config


def cast_to_array(
    dim: int,
    value: NPVector | BoundSpec,
    name: str = "bound",
) -> NPVector:
    """Helper function to cast a bound value to a numpy array.

    Args:
        dim (int): The designated dimension of the array.
        value (NPVector | BoundSpec): The value to cast. If a float is
            provided, it will be broadcasted to all dimensions. If a
            list or array is provided, its length must match the
            dimension. Only exception are arrays or list of length 1,
            which will be broadcasted to all dimensions.
        name (str): The name of the bound, used for error messages.

    Returns:
        NPVector: The casted numpy array.

    Raises:
        DimensionLengthError: If the length of the provided list or
            array does not match the dimension and is not of length 1.
    """
    if isinstance(value, (list, np.ndarray)):
        if len(value) == 1:
            scalar = float(np.asarray(value, dtype=float).reshape(-1)[0])
            return np.full(dim, scalar, dtype=NPFloatType)
        if len(value) != dim:
            raise DimensionLengthError(name, len(value), dim)
        return np.asarray(value, dtype=NPFloatType)
    return np.full(dim, value, dtype=NPFloatType)


def create_box(
    dim: int | None = None,
    *,
    lower_bounds: BoundSpec = -1.0,
    upper_bounds: BoundSpec = 1.0,
) -> pc.Region:
    """Create a sample box for Monte Carlo sampling.

    Args:
        dim (int | None): The dimension of the box.
        lower_bounds (BoundSpec): A vector or list of lower bounds for
            each dimension, or a single float value to be used as the
            lower bound for all dimensions. Default is -1.0.
        upper_bounds (BoundSpec): A vector or list of upper bounds for
            each dimension, or a single float value to be used as the
            upper bound for all dimensions. Default is 1.0.

    Returns:
        pc.Region: The created box as a convex polyhedron.

    Raises:
        ParameterTypeError: If dim is None and the bounds are not given
            as list or array.
        ShapeMismatchError: If the dimensions of the lower and upper
            bounds do not match.
        ParameterOrderError: If any lower bound is greater than the
            corresponding upper bound.
    """
    if dim is None:
        for name, bounds in (
            ("lower_bounds", lower_bounds),
            ("upper_bounds", upper_bounds),
        ):
            if not isinstance(bounds, (list, np.ndarray)):
                raise ParameterTypeError(name, bounds, (list, np.ndarray))

        lower_arr: npt.NDArray[NPFloatType] = np.asarray(
            lower_bounds, dtype=NPFloatType
        )
        upper_arr: npt.NDArray[NPFloatType] = np.asarray(
            upper_bounds, dtype=NPFloatType
        )

        if lower_arr.shape != upper_arr.shape:
            raise ShapeMismatchError("upper_bounds", upper_arr.shape, lower_arr.shape)
        if np.any(lower_arr > upper_arr):
            raise ParameterOrderError("lower_bounds", "upper_bounds", strict=False)

        dim = int(lower_arr.shape[0])
    else:
        lower_arr = cast_to_array(dim, lower_bounds, name="lower_bounds")
        upper_arr = cast_to_array(dim, upper_bounds, name="upper_bounds")

    A = np.vstack((np.eye(dim), -np.eye(dim)))  # ruff:ignore[non-lowercase-variable-in-function]
    b = np.hstack((upper_arr, -lower_arr))

    return pc.Region([pc.Polytope(A, b)])
