"""End-to-end orchestration and provenance capture for Paper 2 experiments."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import logging
import platform
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .adapter import harmonize_paper2_schema, resolve_feature_columns
from .cohort import CohortBuilder, CohortDefinition
from .config import SOFTWARE_VERSION
from .data_utils import create_fixed_horizon_feature_dataset
from .evaluation import (
    patient_bootstrap_confidence_intervals,
    summarize_feature_stability,
)
from .exceptions import DataFileNotFoundError
from .paper2_config import (
    ResolvedPaper2Config,
    expand_experiment_grid,
)
from .paper2_validation import NestedPatientValidationRunner
from .phenotypes import EndpointDefinition, MedicationState, PhenotypeBuilder

LOGGER = logging.getLogger("ppmi_pipeline.paper2_experiment")


@dataclass(slots=True)
class Paper2ExperimentResult:
    """Hold combined outputs from a configuration-grid execution."""

    run_directory: Path
    provenance: dict[str, Any]
    cohort_flow: pd.DataFrame
    fold_metrics: pd.DataFrame
    metric_confidence_intervals: pd.DataFrame
    oof_predictions: pd.DataFrame
    fold_assignments: pd.DataFrame
    feature_records: pd.DataFrame
    feature_stability: pd.DataFrame
    feature_manifest: pd.DataFrame
    specification_manifest: pd.DataFrame
    tuning_results: pd.DataFrame


class Paper2ExperimentRunner:
    """Execute YAML-defined Paper 2 tasks without UI or legacy coupling."""

    def __init__(self, config: ResolvedPaper2Config) -> None:
        """Initialize a configuration-driven experiment.

        Args:
            config: Resolved and validated Paper 2 YAML configuration.
        """
        self.config = config
        self.values = config.values
        self.specifications = expand_experiment_grid(config)

    def run_from_configured_path(self) -> Paper2ExperimentResult:
        """Load the configured CSV/XLSX input and execute the experiment.

        Returns:
            Combined experiment artifacts.

        Raises:
            DataFileNotFoundError: If the configured data path is missing.
            ValueError: If the file type is unsupported.
        """
        data_config = self.values["data"]
        configured_path = data_config.get("path")
        if not configured_path:
            raise DataFileNotFoundError(
                "data.path must be supplied locally before a Paper 2 run."
            )
        source_path = Path(configured_path)
        if not source_path.is_absolute():
            source_path = (self.config.path.parents[2] / source_path).resolve()
        if not source_path.exists():
            raise DataFileNotFoundError(f"Paper 2 input was not found: {source_path}")
        if source_path.suffix.lower() == ".csv":
            data = pd.read_csv(source_path)
        elif source_path.suffix.lower() == ".xlsx":
            data = pd.read_excel(
                source_path,
                sheet_name=data_config.get("sheet_name", 0),
            )
        else:
            raise ValueError("Paper 2 input must be a .csv or .xlsx file.")
        return self.run(data, source_path=source_path)

    def run(
        self,
        data: pd.DataFrame,
        source_path: str | Path | None = None,
    ) -> Paper2ExperimentResult:
        """Run all YAML-expanded specifications on one longitudinal table.

        Args:
            data: Source long-format clinical data.
            source_path: Optional source path used only to calculate provenance.

        Returns:
            Combined in-memory outputs and their local run directory.
        """
        harmonized = harmonize_paper2_schema(
            data,
            self.values["data"]["column_map"],
            self.values["feature_blocks"],
            self.values["data"].get("column_constants"),
        )
        cohort_config = self.values["cohort"]
        cohort_result = CohortBuilder(
            CohortDefinition(
                eligible_groups=tuple(cohort_config["eligible_groups"]),
                baseline_event_ids=tuple(cohort_config["baseline_event_ids"]),
            )
        ).select(harmonized)
        endpoint_config = self.values["endpoint"]
        phenotype_result = PhenotypeBuilder(
            EndpointDefinition(
                name=str(endpoint_config["name"]),
                score_column=str(endpoint_config.get("score_column", "SCORE")),
                required_medication_state=MedicationState(
                    endpoint_config["required_medication_state"]
                ),
                medication_state_column=str(
                    endpoint_config.get("medication_state_column", "MEDICATION_STATE")
                ),
                minimum_score=endpoint_config.get("minimum_score"),
                maximum_score=endpoint_config.get("maximum_score"),
            )
        ).apply(cohort_result.data)

        combined_flows: list[pd.DataFrame] = []
        metric_frames: list[pd.DataFrame] = []
        confidence_frames: list[pd.DataFrame] = []
        prediction_frames: list[pd.DataFrame] = []
        assignment_frames: list[pd.DataFrame] = []
        feature_frames: list[pd.DataFrame] = []
        feature_manifest_frames: list[pd.DataFrame] = []
        specification_rows: list[dict[str, Any]] = []
        tuning_frames: list[pd.DataFrame] = []
        validation_config = self.values["validation"]

        base_flow = pd.concat(
            [
                cohort_result.flow.assign(flow_component="cohort"),
                phenotype_result.flow.assign(flow_component="phenotype"),
            ],
            ignore_index=True,
            sort=False,
        )
        for specification in self.specifications:
            fixed_horizon = create_fixed_horizon_feature_dataset(
                phenotype_result.data,
                specification.task,
                self.values["feature_blocks"],
                specification.feature_blocks,
            )
            feature_manifest_frames.append(
                fixed_horizon.feature_manifest.assign(
                    configuration_id=specification.configuration_id
                )
            )
            specification_rows.append(
                {
                    "configuration_id": specification.configuration_id,
                    "task_fingerprint": specification.task.fingerprint(),
                    "task": json.dumps(specification.task.to_dict(), sort_keys=True),
                    "feature_blocks": json.dumps(specification.feature_blocks),
                    "imputation_strategy": specification.imputation_strategy,
                    "model_family": specification.model_family,
                    "model_parameters": json.dumps(
                        specification.model_parameters, sort_keys=True
                    ),
                }
            )
            task_flow = fixed_horizon.cohort_flow.assign(
                flow_component="fixed_horizon",
                configuration_id=specification.configuration_id,
            )
            combined_flows.append(
                pd.concat(
                    [
                        base_flow.assign(
                            configuration_id=specification.configuration_id
                        ),
                        task_flow,
                    ],
                    ignore_index=True,
                    sort=False,
                )
            )
            features = [
                "Baseline_Score",
                *resolve_feature_columns(
                    self.values["feature_blocks"], specification.feature_blocks
                ),
            ]
            features = list(dict.fromkeys(features))
            runner = NestedPatientValidationRunner(
                outcome_type=specification.task.outcome_type,
                target_column=specification.task.target_column,
                feature_columns=features,
                model_family=specification.model_family,
                parameter_grid=specification.model_parameters,
                patient_column="PATNO",
                outer_splits=int(validation_config["outer_splits"]),
                outer_repeats=int(validation_config["outer_repeats"]),
                inner_splits=int(validation_config["inner_splits"]),
                imputation_strategy=specification.imputation_strategy,
                threshold_selection=str(
                    validation_config.get("threshold_selection", "youden")
                ),
                fixed_threshold=float(validation_config.get("fixed_threshold", 0.5)),
                random_state=int(self.values["study"]["random_seed"]),
                configuration_id=specification.configuration_id,
            )
            validation_result = runner.run(fixed_horizon.dataset)
            confidence = patient_bootstrap_confidence_intervals(
                validation_result.oof_predictions,
                outcome_type=specification.task.outcome_type,
                n_resamples=int(validation_config.get("bootstrap_resamples", 1_000)),
                confidence_level=float(validation_config.get("confidence_level", 0.95)),
                random_state=int(self.values["study"]["random_seed"]),
            ).assign(configuration_id=specification.configuration_id)
            metric_frames.append(validation_result.fold_metrics)
            confidence_frames.append(confidence)
            prediction_frames.append(validation_result.oof_predictions)
            assignment_frames.append(validation_result.fold_assignments)
            feature_frames.append(validation_result.feature_records)
            tuning_frames.append(validation_result.tuning_results)

        fold_metrics = pd.concat(metric_frames, ignore_index=True)
        metric_confidence_intervals = pd.concat(confidence_frames, ignore_index=True)
        oof_predictions = pd.concat(prediction_frames, ignore_index=True)
        fold_assignments = pd.concat(assignment_frames, ignore_index=True)
        feature_records = pd.concat(feature_frames, ignore_index=True)
        tuning_results = pd.concat(tuning_frames, ignore_index=True)
        feature_manifest = pd.concat(
            feature_manifest_frames, ignore_index=True
        ).drop_duplicates(["configuration_id", "feature"])
        specification_manifest = pd.DataFrame(specification_rows)
        feature_stability = pd.concat(
            [
                summarize_feature_stability(group).assign(configuration_id=identifier)
                for identifier, group in feature_records.groupby(
                    "configuration_id", sort=True
                )
            ],
            ignore_index=True,
        )
        cohort_flow = pd.concat(combined_flows, ignore_index=True, sort=False)

        run_directory = self._run_directory()
        provenance = self._provenance(source_path, run_directory)
        self._persist(
            run_directory,
            provenance,
            cohort_flow,
            fold_metrics,
            metric_confidence_intervals,
            oof_predictions,
            fold_assignments,
            feature_records,
            feature_stability,
            feature_manifest,
            specification_manifest,
            tuning_results,
        )
        LOGGER.info(
            "Completed %d Paper 2 specifications in %s",
            len(self.specifications),
            run_directory,
        )
        return Paper2ExperimentResult(
            run_directory=run_directory,
            provenance=provenance,
            cohort_flow=cohort_flow,
            fold_metrics=fold_metrics,
            metric_confidence_intervals=metric_confidence_intervals,
            oof_predictions=oof_predictions,
            fold_assignments=fold_assignments,
            feature_records=feature_records,
            feature_stability=feature_stability,
            feature_manifest=feature_manifest,
            specification_manifest=specification_manifest,
            tuning_results=tuning_results,
        )

    def _run_directory(self) -> Path:
        """Create a timestamped local output directory."""
        configured = Path(self.values["output"]["directory"])
        if not configured.is_absolute():
            configured = self.config.path.parents[2] / configured
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        destination = configured / f"{timestamp}_{self.config.sha256[:12]}"
        destination.mkdir(parents=True, exist_ok=False)
        return destination

    def _provenance(
        self,
        source_path: str | Path | None,
        run_directory: Path,
    ) -> dict[str, Any]:
        """Create a non-clinical execution provenance record."""
        source_hash = None
        if source_path is not None:
            source_hash = _file_sha256(Path(source_path))
        return {
            "created_at_utc": datetime.now(UTC).isoformat(),
            "software_version": SOFTWARE_VERSION,
            "python_version": platform.python_version(),
            "dependency_versions": {
                name: importlib.metadata.version(name)
                for name in ("numpy", "pandas", "scikit-learn", "PyYAML")
            },
            "study_name": self.values["study"]["name"],
            "database_version": self.values["study"]["database_version"],
            "random_seed": self.values["study"]["random_seed"],
            "resolved_config_sha256": self.config.sha256,
            "resolved_config_path": str(self.config.path),
            "source_file_sha256": source_hash,
            "specification_count": len(self.specifications),
            "configuration_ids": [
                item.configuration_id for item in self.specifications
            ],
            "run_directory": str(run_directory),
            "patient_level_outputs_are_restricted": True,
        }

    def _persist(
        self,
        output_dir: Path,
        provenance: dict[str, Any],
        cohort_flow: pd.DataFrame,
        fold_metrics: pd.DataFrame,
        confidence_intervals: pd.DataFrame,
        oof_predictions: pd.DataFrame,
        fold_assignments: pd.DataFrame,
        feature_records: pd.DataFrame,
        feature_stability: pd.DataFrame,
        feature_manifest: pd.DataFrame,
        specification_manifest: pd.DataFrame,
        tuning_results: pd.DataFrame,
    ) -> None:
        """Persist all reproducibility artifacts without raw clinical records."""
        (output_dir / "provenance.json").write_text(
            json.dumps(provenance, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (output_dir / "resolved_config.yaml").write_text(
            yaml.safe_dump(self.values, sort_keys=True),
            encoding="utf-8",
        )
        cohort_flow.to_csv(output_dir / "cohort_flow.csv", index=False)
        fold_metrics.to_csv(output_dir / "fold_metrics.csv", index=False)
        confidence_intervals.to_csv(
            output_dir / "metric_confidence_intervals.csv", index=False
        )
        oof_predictions.to_csv(output_dir / "oof_predictions.csv", index=False)
        fold_assignments.to_csv(output_dir / "fold_assignments.csv", index=False)
        feature_records.to_csv(output_dir / "feature_records.csv", index=False)
        feature_stability.to_csv(output_dir / "feature_stability.csv", index=False)
        feature_manifest.to_csv(output_dir / "feature_manifest.csv", index=False)
        specification_manifest.to_csv(
            output_dir / "specification_manifest.csv", index=False
        )
        tuning_results.to_csv(output_dir / "tuning_results.csv", index=False)


def _file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Calculate a source-file hash without exposing file contents."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()
