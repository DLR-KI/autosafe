# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""Core experiment classes and configuration."""

import dataclasses
import datetime
import pathlib
from collections.abc import Callable
from importlib import import_module
from typing import Any, cast

import orjson
import yaml

from autosafe.tools.experiments.config import (
    DatasetConfig,
    KernelExperimentConfig,
)
from autosafe.tools.experiments.results import (
    BenchmarkResult,
    EvaluationResult,
    ExperimentResult,
)
from autosafe.tools.experiments.types import (
    DatasetType,
    ExperimentType,
)


@dataclasses.dataclass
class _BatchExecutionContext:
    """Managing batch experiment execution with resume support.

    Attributes:
        completed (set[str]): Set of completed item identifiers.
        failed (dict[str, str]): Mapping of failed item identifiers to
            error messages.
        run_item (Callable[[dict[str, Any]], dict[str, Any]]): Callback
            to execute a batch item.
        resume (bool): Whether to skip already completed items.
        stop_on_error (bool): Whether to stop execution on first error.
        state_path (pathlib.Path): Path to the state file for progress
            persistence.
    """

    completed: set[str]
    failed: dict[str, str]
    run_item: Callable[[dict[str, Any]], dict[str, Any]]
    resume: bool
    stop_on_error: bool
    state_path: pathlib.Path


class ExperimentManager:
    """Main class for managing and running experiments."""

    def __init__(self, config: KernelExperimentConfig) -> None:
        """Initialize the experiment manager.

        Args:
            config (KernelExperimentConfig): Kernel experiment
                configuration.
        """
        self.config = config
        self.results: list[ExperimentResult] = []
        self.experiment_count = 0
        self.last_result: ExperimentResult | None = None

    def create_experiment_id(self) -> str:
        """Generate a unique experiment ID.

        Returns:
            str: generated experiment identifier.
        """
        self.experiment_count += 1
        timestamp = datetime.datetime.now(datetime.timezone.utc).strftime(
            "%Y%m%d_%H%M%S"
        )
        return f"exp_{timestamp}_{self.experiment_count:03d}"

    def run_evaluation_experiment(
        self,
        dataset_config: DatasetConfig,
        export_dir: pathlib.Path | None = None,
    ) -> EvaluationResult:
        """Run a Monte Carlo evaluation experiment.

        Uses the in-process evaluation API.

        Args:
            dataset_config (DatasetConfig): Dataset configuration.
            export_dir (pathlib.Path | None): Optional export directory.

        Returns:
            EvaluationResult: Result of the evaluation experiment.
        """
        experiment_id = self.create_experiment_id()

        evaluation_module = import_module("autosafe.tools.experiments.evaluation")

        result_instance = evaluation_module.evaluate_experiment(
            dataset_path=dataset_config.file_path,
            request=evaluation_module.EvaluationExperimentConfig(
                kernel_type=self.config.kernel_type,
                kernel_kwargs=self.config.kernel_kwargs,
                n_samples=self.config.evaluation_samples,
                export_dir=export_dir,
            ),
        )
        result_instance.experiment_id = experiment_id

        # Store result
        self.last_result = result_instance
        self.results.append(result_instance)

        return result_instance

    def run_benchmark_experiment(
        self,
        dataset_config: DatasetConfig,
        export_dir: pathlib.Path | None = None,
    ) -> BenchmarkResult:
        """Run a kernel benchmarking experiment.

        Uses the in-process benchmarking API.

        Args:
            dataset_config (DatasetConfig): Dataset configuration.
            export_dir (pathlib.Path | None): Optional export directory.

        Returns:
            BenchmarkResult: Benchmark result.
        """
        experiment_id = self.create_experiment_id()

        evaluation_module = import_module("autosafe.tools.experiments.evaluation")

        result_instance = evaluation_module.run_benchmark_experiment(
            dataset_path=dataset_config.file_path,
            request=evaluation_module.BenchmarkExperimentConfig(
                kernel_type=self.config.kernel_type,
                kernel_kwargs=self.config.kernel_kwargs,
                n_samples=self.config.n_samples,
                export_dir=export_dir,
            ),
        )
        result_instance.experiment_id = experiment_id

        # Store result
        self.last_result = result_instance
        self.results.append(result_instance)

        return result_instance

    @staticmethod
    def save_results(
        result: ExperimentResult,
        export_dir: pathlib.Path | None = None,
    ) -> pathlib.Path:
        """Save experiment results to JSON file.

        Args:
            result (ExperimentResult): Experiment result to save.
            export_dir (pathlib.Path | None): Optional export directory.

        Returns:
            pathlib.Path: path to the written JSON file.
        """
        if export_dir is None:
            export_dir = result.dataset_path.parent / "results"

        export_dir.mkdir(parents=True, exist_ok=True)

        # Create export filename
        timestamp_str = result.timestamp.strftime("%Y-%m-%dT%H-%M-%S")
        filename = (
            f"{result.experiment_id}_{result.experiment_type.value}_"
            f"{timestamp_str}.json"
        )
        export_file = export_dir / filename

        export_file.write_bytes(
            orjson.dumps(
                result.to_dict(),
                option=(orjson.OPT_INDENT_2 | orjson.OPT_SERIALIZE_NUMPY),
            )
        )

        result.add_export(export_file)
        return export_file

    @staticmethod
    def _load_batch_state(state_path: pathlib.Path) -> dict[str, Any]:
        """Load resumable batch execution state from disk.

        Args:
            state_path (pathlib.Path): State file path.

        Returns:
            dict[str, Any]: Parsed state dictionary.
        """
        if not state_path.exists():
            return {
                "completed": [],
                "failed": {},
                "updated_at": None,
            }
        return orjson.loads(state_path.read_bytes())

    @staticmethod
    def _save_batch_state(
        state_path: pathlib.Path,
        state: dict[str, Any],
    ) -> None:
        """Persist resumable batch execution state.

        Args:
            state_path (pathlib.Path): State file path.
            state (dict[str, Any]): State dictionary to persist.
        """
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state["updated_at"] = datetime.datetime.now(
            datetime.timezone.utc,
        ).isoformat()
        state_path.write_bytes(orjson.dumps(state, option=orjson.OPT_INDENT_2))

    @staticmethod
    def _handle_batch_item(
        item: dict[str, Any],
        run_item: Callable[[dict[str, Any]], dict[str, Any]],
    ) -> None:
        """Execute a batch item and raise failures to the caller.

        Args:
            item (dict[str, Any]): Batch item specification.
            run_item (Callable[[dict[str, Any]], dict[str, Any]]):
                Callback to execute the batch item.
        """
        _ = run_item(item)

    def _process_batch_item(
        self,
        raw_item: object,
        idx: int,
        ctx: "_BatchExecutionContext",
    ) -> None:
        """Process one batch-spec entry and persist progress.

        Args:
            raw_item (object): Raw batch item specification.
            idx (int): Index of the item in the batch.
            ctx (_BatchExecutionContext): Execution context with
                progress tracking.

        Raises:
            RuntimeError: If stop-on-error is enabled and execution
                fails.
        """
        if not isinstance(raw_item, dict):
            ctx.failed[f"item_{idx:04d}"] = "Spec entry must be a mapping"
            return

        item = cast("dict[str, Any]", raw_item)
        item_id = str(item.get("id", f"item_{idx:04d}"))
        if ctx.resume and item_id in ctx.completed:
            return

        try:
            self._handle_batch_item(item, ctx.run_item)
            ctx.completed.add(item_id)
            ctx.failed.pop(item_id, None)
        except Exception as exc:
            ctx.failed[item_id] = str(exc)
            if ctx.stop_on_error:
                self._save_batch_state(
                    ctx.state_path,
                    {
                        "completed": sorted(ctx.completed),
                        "failed": ctx.failed,
                    },
                )
                raise RuntimeError(str(exc)) from exc

        self._save_batch_state(
            ctx.state_path,
            {
                "completed": sorted(ctx.completed),
                "failed": ctx.failed,
            },
        )

    def run_batch_spec(
        self,
        spec_path: pathlib.Path,
        run_item: Callable[[dict[str, Any]], dict[str, Any]],
        *,
        state_path: pathlib.Path | None = None,
        resume: bool = True,
        stop_on_error: bool = False,
    ) -> dict[str, Any]:
        """Run experiments declared in a YAML spec with resume support.

        Args:
            spec_path (pathlib.Path): YAML file describing experiments.
            run_item (Any): Callback that executes one spec entry.
            state_path (pathlib.Path | None): Optional state file path.
            resume (bool): Skip entries already marked completed.
            stop_on_error (bool): Stop after first failure.

        Returns:
            dict[str, Any]: Summary including completed/failed counts.

        Raises:
            FileNotFoundError: If spec file does not exist.
            TypeError: If spec format is invalid.
        """
        if not spec_path.exists():
            raise FileNotFoundError(f"Spec file not found: {spec_path}")

        loaded = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            experiments = loaded.get("experiments", [])
        elif isinstance(loaded, list):
            experiments = loaded
        else:
            raise TypeError("Spec must be a list or a mapping with 'experiments'")

        if not isinstance(experiments, list):
            raise TypeError("Spec field 'experiments' must be a list")

        if state_path is None:
            state_path = spec_path.with_suffix(".state.json")
        state = self._load_batch_state(state_path)

        completed = set(state.get("completed", []))
        failed = dict(state.get("failed", {}))
        ctx = _BatchExecutionContext(
            completed=completed,
            failed=failed,
            run_item=run_item,
            resume=resume,
            stop_on_error=stop_on_error,
            state_path=state_path,
        )

        for idx, raw_item in enumerate(experiments, start=1):
            self._process_batch_item(raw_item, idx, ctx)

        return {
            "state_path": state_path,
            "completed_count": len(completed),
            "failed_count": len(failed),
            "failed": failed,
        }


def create_experiment_manager(config: KernelExperimentConfig) -> ExperimentManager:
    """Create and return an experiment manager.

    Args:
        config (KernelExperimentConfig): Kernel experiment
            configuration.

    Returns:
        New experiment manager instance.
    """
    return ExperimentManager(config)


__all__ = [
    "BenchmarkResult",
    "DatasetConfig",
    "DatasetType",
    "EvaluationResult",
    "ExperimentManager",
    "ExperimentResult",
    "ExperimentType",
    "KernelExperimentConfig",
    "create_experiment_manager",
]
