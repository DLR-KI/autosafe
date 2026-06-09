# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""Main entry point for the monte carlo sampling."""

import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal, cast

import numpy as np
import numpy.typing as npt
import orjson
import polytope as pc
import tqdm.rich
import typer
from loguru import logger
from tqdm import TqdmExperimentalWarning

from autosafe import ROOT_FOLDER
from autosafe.samples import Samples
from autosafe.tools.monte_carlo import MC_APP
from autosafe.tools.monte_carlo._sample import (
    create_box,
    create_config,
)
from autosafe.tools.monte_carlo.dicts import (
    MonteCarloConfig,
    PolytopeDict,
    RegionDict,
    ResultStats,
    SamplingResult,
)
from autosafe.tools.monte_carlo.inequality_utils import (
    ODDFactory,
    load_yaml_odd_config,
)
from autosafe.tools.serializers.orjson import serializer
from autosafe.typing import KernelType, Vector

if TYPE_CHECKING:
    from autosafe.typing import Matrix

warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)


def _resolve_odd(
    config: MonteCarloConfig,
    odd_base: pc.Region,
) -> tuple[pc.Region, str | None]:
    """Resolve the configured ODD region.

    Args:
        config (MonteCarloConfig): The Monte Carlo sampling
            configuration.
        odd_base (pc.Region): The base ODD region defined by the box
            limits.

    Returns:
        tuple[pc.Region, str | None]: The resolved ODD region and an
            optional description.

    Raises:
        ValueError: If the custom ODD configuration cannot be parsed.
        NotImplementedError: If the ODD type is not supported.
    """
    if hasattr(config, "custom_odd_config") and config.get("custom_odd_config"):
        custom_odd_config = config["custom_odd_config"]
        if isinstance(custom_odd_config, str) and custom_odd_config.endswith((
            ".yaml",
            ".yml",
        )):
            odd_config = load_yaml_odd_config(custom_odd_config)
        elif isinstance(custom_odd_config, dict):
            odd_config = custom_odd_config
        else:
            raise ValueError(
                "custom_odd_config must be either a YAML path or a mapping",
            )

        odd_factory = ODDFactory(odd_config)
        odd_custom, odd_description = odd_factory.create_odd()
        base_poly = cast("pc.Polytope", odd_base.list_poly[0])
        custom_poly = cast("pc.Polytope", odd_custom.list_poly[0])
        odd = odd_base.__class__([
            base_poly.__class__(
                np.vstack([base_poly.A, custom_poly.A]),
                np.hstack([base_poly.b, custom_poly.b]),
            ),
        ])
        return odd, odd_description

    if config["odd_type"] == "box":
        return odd_base, None

    raise NotImplementedError(
        f"ODD polytope type '{config['odd_type']}' is not implemented.",
    )


def _find_anchor_points(
    config: MonteCarloConfig,
    odd: pc.Region,
    rng: np.random.Generator,
) -> list[Vector]:
    """Find anchor points that lie inside the configured ODD.

    Args:
        config (MonteCarloConfig): The Monte Carlo sampling
            configuration.
        odd (pc.Region): The ODD region to find anchor points within.
        rng (np.random.Generator): Random number generator for sampling
            points.

    Returns:
        list[Vector]: Anchor points inside the ODD.
    """
    anchors: list[Vector] = []
    with tqdm.rich.tqdm(
        total=config["odd_anchors"],
        desc=f"Finding anchor points within ODD ({config['odd_anchors']} points)",
    ) as pbar:
        while len(anchors) < config["odd_anchors"]:
            point = cast(
                "Vector",
                rng.uniform(
                    config["box_lower_limits"],
                    config["box_upper_limits"],
                    size=(config["dim"],),
                ),
            )
            if point in odd:
                anchors.append(point)
                pbar.update(1)
    return anchors


def run_single_sampling(config: MonteCarloConfig) -> None:
    """Run a single MC sampling based on the provided configuration.

    Args:
        config (MonteCarloConfig): The configuration for the Monte
            Carlo sampling.
    """
    box = create_box(
        dim=config["dim"],
        lower_bounds=config["box_lower_limits"],
        upper_bounds=config["box_upper_limits"],
    )

    odd_base = create_box(
        dim=config["dim"],
        lower_bounds=config["odd_lower_limits"],
        upper_bounds=config["odd_upper_limits"],
    )

    odd, odd_description = _resolve_odd(config, odd_base)
    if odd_description is not None:
        logger.info(f"Created custom ODD: {odd_description}")

    # If dim is None, infer it from the box limits. We know that box
    # limits are set correctly by now.
    if config["dim"] is None:
        config["dim"] = len(cast("list", config["box_lower_limits"]))

    rng = np.random.default_rng()

    # Fake progress bar for anchor point finding. This is done to have a
    # consistent output format and to provide feedback on the anchor
    # point finding process, which can take some time for higher
    # dimensions and complex ODD shapes.
    for _ in tqdm.rich.tqdm(
        range(1),
        desc=f"Finding anchor points for ODD in {config['dim']} dimensions",
    ):
        points = cast(
            "Matrix",
            rng.uniform(
                config["box_lower_limits"],
                config["box_upper_limits"],
                size=(config["samples"], config["dim"]),
            ),
        )
        anchors = _find_anchor_points(config, odd, rng)

    autosafe_odd = Samples(
        samples=anchors,
        kernel_cls=config["kernel_config"]["type"],
        kernel_kwargs=config["kernel_config"]["params"],
    )

    # Fake progress bar for affinity calculation and ODD membership
    # This is done to have a consistent output format
    sampling_results: list[SamplingResult] = []
    for _ in tqdm.rich.tqdm(
        range(1),
        desc=f"Computing affinity scores for {points.shape[0]} test points",
    ):
        affinities = autosafe_odd(points)
        points_in_odd: npt.NDArray[np.bool] = odd.contains(points.T)
        sampling_results = [
            SamplingResult(
                coordinates=points[idx, :],
                in_odd=bool(points_in_odd[idx]),
                affinity=float(affinities[idx]),
            )
            for idx in range(points.shape[0])
        ]

    result_stats = ResultStats(
        box=RegionDict(
            list_poly=[
                PolytopeDict(A=cast("pc.Polytope", p).A, b=cast("pc.Polytope", p).b)
                for p in box.list_poly
            ],
        ),
        odd=RegionDict(
            list_poly=[
                PolytopeDict(A=cast("pc.Polytope", p).A, b=cast("pc.Polytope", p).b)
                for p in odd.list_poly
            ],
        ),
        anchors=anchors,
        config=config,
        autosafe_odd=autosafe_odd,
        total_samples=len(sampling_results),
        sampling_results=sampling_results,
    )

    file = Path(config["filename"])
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_bytes(
        orjson.dumps(
            result_stats,
            option=(orjson.OPT_INDENT_2 | orjson.OPT_SERIALIZE_NUMPY),
            default=serializer,
        ),
    )


@MC_APP.command(
    name="sample",
    help="""
    Main function to run the Monte Carlo sampling.

    Using Monte Carlo sampling, this function allows to quantitatively
    evaluate the affinity algorithm of autoSAFE on a randomly created
    set of samples. This is useful to verify the behavior of the
    affinity algorithm in higher dimensions.
    The results are saved to a JSON file for further evaluation.
    """,
)
def sample(  # noqa: PLR0913,PLR0917
    dim: Annotated[
        int | None,
        typer.Option(
            help="Number of dimensions. Default, if not config file is given, is 2.0.",
        ),
    ] = None,
    odd_type: Annotated[
        Literal["box"] | None,
        typer.Option(
            help="Polytope type of the ODD. All shapes other than 'box' "
            "must be set via the config file. Default, if not config "
            "file is given, is 'box'.",
        ),
    ] = None,
    odd_limits: Annotated[
        float | None,
        typer.Option(
            help="Limits of the ODD box in each dimension. "
            "This allows to set a symmetric box around the origin. "
            "Default, if not config file is given, is 5.0.",
        ),
    ] = None,
    box_limits: Annotated[
        float | None,
        typer.Option(
            help="Limits of the sampling box in each dimension. "
            "This allows to set a symmetric box around the origin. "
            "Default, if not config file is given, is 10.0.",
        ),
    ] = None,
    odd_anchors: Annotated[
        int | None,
        typer.Option(
            help="Number of samples used as anchors for the autoSAFE "
            "representation of the ODD. Default, if not config file is "
            "given, is 20.",
        ),
    ] = None,
    samples: Annotated[
        int | None,
        typer.Option(
            help="Number of samples used to quantitatively evaluate the "
            "autoSAFE's algorithm performance. Recommended to be "
            "significantly greater than the number of anchors. "
            "Default, if not config file is given, is 1000.",
        ),
    ] = None,
    kernel_type: Annotated[
        KernelType,
        typer.Option(
            help="The type of the kernel to use. Default is 'RBF'.",
        ),
    ] = "RBF",
    kernel_params: Annotated[
        str | None,
        typer.Option(
            help="The parameters for the kernel as a JSON string. "
            "If None, default parameters are used.",
        ),
    ] = None,
    filename: Annotated[
        str,
        typer.Option(
            help="Path to the output JSON file where the results will be saved.",
        ),
    ] = "sampling_results.json",
    config_file: Annotated[
        str | None,
        typer.Option(
            help="Path to the configuration file. If provided, all "
            "other parameters are ignored.",
        ),
    ] = None,
    odd_config: Annotated[
        str | None,
        typer.Option(
            help="Path to YAML file or inline JSON with custom ODD configuration. "
            "Allows defining inequality constraints and complex polytopes.",
        ),
    ] = None,
    config_file_folder: Annotated[
        str | None,
        typer.Option(
            help="Path to the folder where the configuration files are "
            "located. If provided, the Monte-Carlo-sampling is ran for "
            "all configuration files in this folder overwriting all "
            "other parameters.",
        ),
    ] = None,
) -> None:
    """Main function to run the Monte Carlo sampling.

    Using Monte Carlo sampling, this function allows to quantitatively
    evaluate the affinity algorithm of autoSAFE on a randomly created
    set of samples. This is useful to verify the behavior of the
    affinity algorithm in higher dimensions.
    The results are saved to a JSON file for further evaluation.

    Args:
        dim (int | None): The dimension of the space to sample. Must be
            greater than 0.
        odd_type (Literal["box"] | None): The type of the ODD polytope.
            Currently, "box" is supported, and custom polytopes can be
            defined via odd_config.
        odd_config (str | None): Path to YAML file or inline JSON with
            custom ODD configuration. Allows defining inequality
            constraints and complex polytopes beyond simple boxes.
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
        kernel_params (str | None): The parameters for the kernel as a
            JSON string. If None, default parameters are used.
        filename (str): Path to the output JSON file where the results
            will be saved.
        config_file (str | None): Path to the configuration file. If
            provided, all other parameters are ignored.
        config_file_folder (str | None): Path to the folder where the
            configuration files are located. If provided, the Monte-
            Carlo-sampling is ran for all configuration files in this
            folder overwriting all other parameters.

    Raises:
        ValueError: If the provided config file folder is not a
            directory or does not contain any configuration files.
    """
    if config_file_folder is not None:
        config_folder_path = Path(config_file_folder)
        if not config_folder_path.is_absolute():
            config_folder_path = ROOT_FOLDER / config_folder_path
        if not config_folder_path.is_dir():
            raise ValueError("The provided config file folder is not a directory.")

        config_files = [
            *config_folder_path.glob("*.json"),
            *config_folder_path.glob("*.yaml"),
            *config_folder_path.glob("*.yml"),
        ]
        if len(config_files) == 0:
            raise ValueError("No configuration files found in the provided folder.")

        for config_path in config_files:
            logger.info(
                f"Processing configuration file: {config_path}",
            )
            config = create_config(config_file=str(config_path))
            run_single_sampling(config)

    else:
        config = create_config(
            dim=dim,
            odd_type=odd_type,
            odd_limits=odd_limits,
            box_limits=box_limits,
            odd_anchors=odd_anchors,
            samples=samples,
            kernel_type=kernel_type,
            kernel_params=kernel_params,
            filename=filename,
            config_file=config_file,
            odd_config=odd_config,
        )
        run_single_sampling(config)


def main() -> None:
    """Main function to run the Monte Carlo sampling and evaluation."""
    sample(
        dim=2,
        odd_type="box",
        odd_limits=5.0,
        box_limits=10.0,
        odd_anchors=200,
        samples=10000,
        kernel_type="RBF",
        kernel_params=None,
        filename=str(ROOT_FOLDER / "sampling_results.json"),
        config_file=None,
        config_file_folder=None,
    )


if __name__ == "__main__":
    main()
