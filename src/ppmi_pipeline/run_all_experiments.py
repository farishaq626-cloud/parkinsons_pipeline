"""Sequential Paper 2 batch execution and non-clinical summary aggregation."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .config import SOFTWARE_VERSION
from .exceptions import DataFileNotFoundError, MissingColumnError
from .logging_config import configure_logging
from .paper2_config import (
    PAPER2_FEATURE_BLOCKS,
    ResolvedPaper2Config,
    load_paper2_config,
)
from .paper2_experiment import Paper2ExperimentResult, Paper2ExperimentRunner
from .synthetic_generator import PAPER2_BATCH_CONFIGS

LOGGER = logging.getLogger("ppmi_pipeline.run_all_experiments")


@dataclass(slots=True)
class BatchExperimentResult:
    """Hold the unified batch summary and provenance paths."""

    master_summary: pd.DataFrame
    master_summary_path: Path
    metadata_path: Path
    run_directories: dict[str, Path]
    config_hashes: dict[str, str]


def run_all_experiments(
    config_dir: str | Path = "configs/paper2",
    output_path: str | Path = "outputs/paper2_master_summary.csv",
    config_names: Sequence[str] = PAPER2_BATCH_CONFIGS,
    reuse_completed: bool = False,
) -> BatchExperimentResult:
    """Execute every declared Paper 2 experiment and aggregate its outputs.

    Execution is sequential and fail-fast: no master summary is published if a
    configuration fails. The unified CSV contains aggregate metrics and feature
    stability only; patient-level predictions remain in ignored local run
    directories.

    Args:
        config_dir: Directory containing fully resolved or inheriting YAML files.
        output_path: Destination for the atomic master-summary CSV.
        config_names: Ordered YAML filenames to execute.
        reuse_completed: Reuse the newest complete run whose resolved
            configuration hash matches the current YAML.

    Returns:
        Master summary, metadata path, configuration hashes, and run directories.

    Raises:
        DataFileNotFoundError: If a requested configuration is absent.
        ValueError: If no configurations are supplied or outputs are malformed.
    """
    directory = Path(config_dir).resolve()
    names = tuple(config_names)
    if not names:
        raise ValueError("At least one Paper 2 batch configuration is required.")
    summary_frames: list[pd.DataFrame] = []
    run_directories: dict[str, Path] = {}
    config_hashes: dict[str, str] = {}
    database_versions: dict[str, str] = {}
    random_seeds: dict[str, int] = {}

    for filename in names:
        config_path = directory / filename
        if not config_path.exists():
            raise DataFileNotFoundError(
                f"Paper 2 batch configuration was not found: {config_path}"
            )
        LOGGER.info("Starting Paper 2 batch configuration: %s", filename)
        config = load_paper2_config(config_path)
        result = _load_latest_completed_run(config) if reuse_completed else None
        if result is None:
            result = Paper2ExperimentRunner(config).run_from_configured_path()
        else:
            LOGGER.info(
                "Reusing completed Paper 2 run for %s: %s",
                filename,
                result.run_directory,
            )
        experiment_name = Path(filename).stem
        summary_frames.append(build_experiment_summary(experiment_name, config, result))
        run_directories[experiment_name] = result.run_directory
        config_hashes[experiment_name] = config.sha256
        database_versions[experiment_name] = str(
            config.values["study"]["database_version"]
        )
        random_seeds[experiment_name] = int(config.values["study"]["random_seed"])

    master = pd.concat(summary_frames, ignore_index=True, sort=False)
    master = master.sort_values(
        ["batch_experiment", "outcome_type", "metric", "specification_key"],
        kind="stable",
    ).reset_index(drop=True)
    destination = Path(output_path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    _atomic_csv_write(master, destination)
    metadata_path = destination.with_name("paper2_master_metadata.json")
    metadata = {
        "artifact_type": "paper2_non_patient_level_master_summary",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "software_version": SOFTWARE_VERSION,
        "config_directory": str(directory),
        "config_order": list(names),
        "config_sha256": config_hashes,
        "database_versions": database_versions,
        "random_seeds": random_seeds,
        "run_directories": {name: str(path) for name, path in run_directories.items()},
        "master_summary_path": str(destination),
        "master_summary_sha256": _file_sha256(destination),
        "master_summary_rows": int(len(master)),
        "contains_patient_level_data": False,
    }
    _atomic_text_write(
        metadata_path,
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
    )
    LOGGER.info("Paper 2 master summary written to %s", destination)
    return BatchExperimentResult(
        master_summary=master,
        master_summary_path=destination,
        metadata_path=metadata_path,
        run_directories=run_directories,
        config_hashes=config_hashes,
    )


def build_experiment_summary(
    experiment_name: str,
    config: ResolvedPaper2Config,
    result: Paper2ExperimentResult,
) -> pd.DataFrame:
    """Build a metric-long summary for one YAML experiment.

    Args:
        experiment_name: Human-readable source YAML stem.
        config: Resolved configuration and hash.
        result: Completed experiment outputs.

    Returns:
        One row per specification and aggregate metric.

    Raises:
        MissingColumnError: If required experiment artifacts are malformed.
        ValueError: If manifest JSON cannot be interpreted.
    """
    manifest_required = {
        "configuration_id",
        "task",
        "feature_blocks",
        "imputation_strategy",
        "model_family",
        "model_parameters",
    }
    confidence_required = {
        "configuration_id",
        "metric",
        "estimate",
        "ci_lower",
        "ci_upper",
        "confidence_level",
        "valid_resamples",
    }
    _require_columns(
        result.specification_manifest,
        manifest_required,
        "specification_manifest",
    )
    _require_columns(
        result.metric_confidence_intervals,
        confidence_required,
        "metric_confidence_intervals",
    )
    manifest = result.specification_manifest.copy()
    task_values = manifest["task"].map(_json_mapping)
    manifest["outcome_type"] = task_values.map(lambda value: value["outcome_type"])
    manifest["horizon_months"] = task_values.map(
        lambda value: int(value["horizon_months"])
    )
    manifest["tolerance_days"] = task_values.map(
        lambda value: int(value["tolerance_days"])
    )
    manifest["endpoint"] = task_values.map(lambda value: value["endpoint"])
    manifest["progression_threshold"] = task_values.map(
        lambda value: value.get("progression_threshold")
    )
    manifest["medication_state"] = task_values.map(
        lambda value: value.get("medication_state")
    )
    manifest["selected_feature_blocks"] = manifest["feature_blocks"].map(
        lambda value: "+".join(_json_sequence(value))
    )
    for block in PAPER2_FEATURE_BLOCKS:
        manifest[f"feature_block__{block}"] = manifest["feature_blocks"].map(
            lambda value, name=block: int(name in _json_sequence(value))
        )

    stability = _stability_summary(result.feature_stability)
    cohort = _retained_cohort_summary(result.cohort_flow)
    summary = result.metric_confidence_intervals.merge(
        manifest.drop(columns=["task", "feature_blocks"]),
        on="configuration_id",
        how="left",
        validate="many_to_one",
    )
    summary = summary.merge(
        stability,
        on="configuration_id",
        how="left",
        validate="many_to_one",
    ).merge(
        cohort,
        on="configuration_id",
        how="left",
        validate="many_to_one",
    )
    summary.insert(0, "batch_experiment", experiment_name)
    summary.insert(
        1,
        "specification_key",
        experiment_name + ":" + summary["configuration_id"].astype(str),
    )
    validation = config.values["validation"]
    summary["validation_strategy"] = validation["split_strategy"]
    summary["outer_splits"] = int(validation["outer_splits"])
    summary["outer_repeats"] = int(validation["outer_repeats"])
    summary["inner_splits"] = int(validation["inner_splits"])
    summary["config_sha256"] = config.sha256
    summary["database_version"] = config.values["study"]["database_version"]
    summary["random_seed"] = int(config.values["study"]["random_seed"])
    summary["source_file_sha256"] = result.provenance.get("source_file_sha256")
    summary["software_version"] = result.provenance.get(
        "software_version", SOFTWARE_VERSION
    )
    summary["metric_direction"] = summary["metric"].map(_metric_direction)
    if summary["outcome_type"].isna().any():
        raise ValueError("A confidence-interval row did not match a specification.")
    return summary


def build_parser() -> argparse.ArgumentParser:
    """Build the Paper 2 batch command-line parser.

    Returns:
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(
        description="Execute and aggregate the four Paper 2 YAML experiments."
    )
    parser.add_argument("--config-dir", type=Path, default=Path("configs/paper2"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/paper2_master_summary.csv"),
    )
    parser.add_argument(
        "--log-path",
        type=Path,
        default=Path("results/paper2/paper2_batch.log"),
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        default="INFO",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Reuse the newest complete run with a provenance hash matching each "
            "resolved configuration."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Execute the master Paper 2 batch.

    Args:
        argv: Optional command-line argument sequence.

    Returns:
        Zero after successful execution.
    """
    arguments = build_parser().parse_args(argv)
    configure_logging(arguments.log_path, arguments.log_level)
    run_all_experiments(
        arguments.config_dir,
        arguments.output,
        reuse_completed=arguments.resume,
    )
    return 0


def _load_latest_completed_run(
    config: ResolvedPaper2Config,
) -> Paper2ExperimentResult | None:
    """Load the newest complete run matching a resolved configuration hash.

    Args:
        config: Current resolved experiment configuration.

    Returns:
        Reconstructed experiment result, or ``None`` if no valid complete run
        exists. Incomplete or hash-mismatched directories are never reused.
    """
    configured = Path(config.values["output"]["directory"])
    if not configured.is_absolute():
        configured = config.path.parents[2] / configured
    if not configured.exists():
        return None
    artifact_names = {
        "cohort_flow": "cohort_flow.csv",
        "fold_metrics": "fold_metrics.csv",
        "metric_confidence_intervals": "metric_confidence_intervals.csv",
        "oof_predictions": "oof_predictions.csv",
        "fold_assignments": "fold_assignments.csv",
        "feature_records": "feature_records.csv",
        "feature_stability": "feature_stability.csv",
        "feature_manifest": "feature_manifest.csv",
        "specification_manifest": "specification_manifest.csv",
        "tuning_results": "tuning_results.csv",
    }
    candidates = sorted(
        configured.glob(f"*_{config.sha256[:12]}"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for run_directory in candidates:
        provenance_path = run_directory / "provenance.json"
        resolved_path = run_directory / "resolved_config.yaml"
        required_paths = [
            provenance_path,
            resolved_path,
            *(run_directory / name for name in artifact_names.values()),
        ]
        if not all(path.is_file() for path in required_paths):
            continue
        try:
            provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if provenance.get("resolved_config_sha256") != config.sha256:
            continue
        tables = {
            field: pd.read_csv(run_directory / filename)
            for field, filename in artifact_names.items()
        }
        return Paper2ExperimentResult(
            run_directory=run_directory,
            provenance=provenance,
            **tables,
        )
    return None


def _stability_summary(stability: pd.DataFrame) -> pd.DataFrame:
    """Collapse per-feature stability into specification-level diagnostics."""
    required = {
        "configuration_id",
        "feature",
        "mean_importance",
        "inclusion_frequency",
        "sign_consistency",
        "rank_stability",
    }
    _require_columns(stability, required, "feature_stability")
    rows: list[dict[str, Any]] = []
    for configuration_id, group in stability.groupby("configuration_id", sort=True):
        ranked = group.sort_values(
            ["rank_stability", "mean_importance"],
            ascending=False,
            kind="stable",
        )
        rows.append(
            {
                "configuration_id": configuration_id,
                "feature_count": int(group["feature"].nunique()),
                "mean_sign_consistency": float(group["sign_consistency"].mean()),
                "mean_inclusion_frequency": float(group["inclusion_frequency"].mean()),
                "mean_rank_stability": float(group["rank_stability"].mean()),
                "top_stable_features": ";".join(ranked["feature"].head(5)),
            }
        )
    return pd.DataFrame(rows)


def _retained_cohort_summary(flow: pd.DataFrame) -> pd.DataFrame:
    """Extract final fixed-horizon patient counts for every specification."""
    required = {"configuration_id", "flow_component", "stage", "patients_retained"}
    _require_columns(flow, required, "cohort_flow")
    selected = flow.loc[
        flow["flow_component"].eq("fixed_horizon")
        & flow["stage"].eq("fixed_horizon_follow_up"),
        ["configuration_id", "patients_retained"],
    ].copy()
    if selected.empty:
        raise ValueError("cohort_flow has no fixed_horizon_follow_up stage.")
    selected["retained_patients"] = pd.to_numeric(
        selected.pop("patients_retained"), errors="raise"
    ).astype(int)
    return selected.drop_duplicates("configuration_id")


def _require_columns(table: pd.DataFrame, required: set[str], name: str) -> None:
    """Raise a descriptive error for malformed experiment artifacts."""
    missing = sorted(required.difference(table.columns))
    if missing:
        raise MissingColumnError(
            f"{name} is missing required columns: " + ", ".join(missing)
        )


def _json_mapping(value: object) -> dict[str, Any]:
    """Decode a JSON object stored in an experiment manifest."""
    decoded = json.loads(str(value))
    if not isinstance(decoded, dict):
        raise ValueError("Task manifest JSON must decode to an object.")
    return decoded


def _json_sequence(value: object) -> list[str]:
    """Decode a JSON list stored in an experiment manifest."""
    decoded = json.loads(str(value))
    if not isinstance(decoded, list) or any(
        not isinstance(item, str) for item in decoded
    ):
        raise ValueError("Feature-block manifest JSON must decode to a string list.")
    return decoded


def _metric_direction(metric: str) -> str:
    """Return whether larger or smaller values indicate better performance."""
    lower_is_better = {"mae", "rmse", "brier_score"}
    return "lower_is_better" if metric in lower_is_better else "higher_is_better"


def _atomic_csv_write(table: pd.DataFrame, path: Path) -> None:
    """Write one CSV atomically through a same-directory temporary file."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    table.to_csv(temporary, index=False)
    temporary.replace(path)


def _atomic_text_write(path: Path, content: str) -> None:
    """Write one text artifact atomically."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Calculate the SHA-256 hash of an output artifact."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
