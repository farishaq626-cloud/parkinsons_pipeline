"""YAML loading, validation, hashing, and grid expansion for Paper 2."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

import yaml

from .exceptions import ConfigurationError, DataFileNotFoundError
from .task_spec import OutcomeType, TaskSpecification

PAPER2_FEATURE_BLOCKS = (
    "demographics_disease_history",
    "baseline_motor",
    "cognition_neuropsychology",
    "olfaction_sleep_autonomic",
    "biofluid_biomarkers",
    "imaging_dat_spect",
    "genetic_variables",
)
REQUIRED_TOP_LEVEL_SECTIONS = {
    "study",
    "data",
    "cohort",
    "endpoint",
    "task_grid",
    "feature_blocks",
    "validation",
    "models",
    "output",
}
ENVIRONMENT_REFERENCE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


@dataclass(frozen=True, slots=True)
class ResolvedPaper2Config:
    """Hold a resolved YAML configuration and its stable fingerprint."""

    path: Path
    values: dict[str, Any]
    sha256: str


@dataclass(frozen=True, slots=True)
class ExperimentSpecification:
    """Represent one expanded task, imputation, and model combination."""

    configuration_id: str
    task: TaskSpecification
    imputation_strategy: str
    model_family: str
    model_parameters: dict[str, list[Any]]
    feature_blocks: tuple[str, ...]


def load_paper2_config(path: str | Path) -> ResolvedPaper2Config:
    """Load a Paper 2 YAML file, including optional local inheritance.

    Args:
        path: YAML configuration path. An ``extends`` field may refer to a base
            YAML file relative to this file.

    Returns:
        Validated resolved values and their canonical SHA-256 hash.

    Raises:
        DataFileNotFoundError: If the requested configuration is absent.
        ConfigurationError: If YAML content or required fields are invalid.
    """
    config_path = Path(path).resolve()
    values = _load_with_inheritance(config_path, visited=set())
    values = _expand_environment(values)
    _validate_config(values)
    canonical = json.dumps(
        values, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return ResolvedPaper2Config(
        path=config_path,
        values=values,
        sha256=hashlib.sha256(canonical).hexdigest(),
    )


def expand_experiment_grid(
    config: ResolvedPaper2Config | Mapping[str, Any],
) -> list[ExperimentSpecification]:
    """Expand YAML task, imputation, model, and feature-set dimensions.

    Args:
        config: Resolved configuration or equivalent mapping.

    Returns:
        Deterministically ordered experiment specifications.

    Raises:
        ConfigurationError: If the expanded task grid is inconsistent.
    """
    values = config.values if isinstance(config, ResolvedPaper2Config) else dict(config)
    _validate_config(values)
    task_grid = values["task_grid"]
    validation = values["validation"]
    feature_sets = task_grid.get("feature_sets", [list(PAPER2_FEATURE_BLOCKS)])
    if not isinstance(feature_sets, list) or not feature_sets:
        raise ConfigurationError("task_grid.feature_sets must be a non-empty list.")
    if any(not isinstance(blocks, list) or not blocks for blocks in feature_sets):
        raise ConfigurationError(
            "Every task_grid.feature_sets entry must be a non-empty list."
        )

    specifications: list[ExperimentSpecification] = []
    dimensions = product(
        task_grid["horizons_months"],
        task_grid["tolerance_days"],
        task_grid["outcome_types"],
        validation["imputation_strategies"],
        values["models"].items(),
        feature_sets,
    )
    for horizon, tolerance, outcome, imputation, model_item, blocks in dimensions:
        model_family, model_config = model_item
        outcome_type = OutcomeType(str(outcome).lower())
        threshold = (
            task_grid.get("progression_threshold")
            if outcome_type is OutcomeType.BINARY
            else None
        )
        task = TaskSpecification(
            name=(
                f"{values['study']['name']}_{values['endpoint']['name']}_"
                f"{horizon}m_{outcome_type.value}"
            ),
            endpoint=str(values["endpoint"]["name"]),
            outcome_type=outcome_type,
            horizon_months=int(horizon),
            tolerance_days=int(tolerance),
            baseline_event_ids=tuple(values["cohort"]["baseline_event_ids"]),
            progression_threshold=threshold,
            medication_state=values["endpoint"].get("required_medication_state"),
        )
        normalized_blocks = tuple(str(block) for block in blocks)
        unknown_blocks = sorted(
            set(normalized_blocks).difference(PAPER2_FEATURE_BLOCKS)
        )
        if unknown_blocks:
            raise ConfigurationError(
                "feature_sets references unknown blocks: " + ", ".join(unknown_blocks)
            )
        grid_key = f"{outcome_type.value}_parameter_grid"
        parameter_grid = model_config.get(
            grid_key,
            model_config.get("parameter_grid", {}),
        )
        if not isinstance(parameter_grid, Mapping):
            raise ConfigurationError(
                f"models.{model_family}.{grid_key} must be a mapping."
            )
        payload = {
            "task": task.to_dict(),
            "imputation_strategy": imputation,
            "model_family": model_family,
            "model_parameters": parameter_grid,
            "feature_blocks": normalized_blocks,
        }
        identifier = hashlib.sha256(
            json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()[:16]
        specifications.append(
            ExperimentSpecification(
                configuration_id=identifier,
                task=task,
                imputation_strategy=str(imputation),
                model_family=str(model_family),
                model_parameters={
                    str(key): list(value) for key, value in parameter_grid.items()
                },
                feature_blocks=normalized_blocks,
            )
        )
    if not specifications:
        raise ConfigurationError("Paper 2 experiment grid expanded to zero runs.")
    return specifications


def _load_with_inheritance(
    path: Path,
    visited: set[Path],
) -> dict[str, Any]:
    """Load YAML and recursively merge an optional base configuration."""
    if not path.exists():
        raise DataFileNotFoundError(f"Paper 2 configuration was not found: {path}")
    if path in visited:
        raise ConfigurationError(f"Circular YAML inheritance detected at {path}.")
    visited.add(path)
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise ConfigurationError(f"Invalid YAML configuration: {path}") from error
    if not isinstance(loaded, dict):
        raise ConfigurationError(f"YAML configuration must contain a mapping: {path}")

    base_reference = loaded.pop("extends", None)
    if base_reference is None:
        return loaded
    base_path = (path.parent / str(base_reference)).resolve()
    base = _load_with_inheritance(base_path, visited)
    return _deep_merge(base, loaded)


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge mappings while replacing scalar and list values."""
    merged = copy.deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _expand_environment(value: Any) -> Any:
    """Resolve explicit ``${NAME}`` references in a YAML value tree.

    Args:
        value: Scalar, sequence, or mapping loaded from YAML.

    Returns:
        A recursively copied value with environment references substituted.

    Raises:
        ConfigurationError: If a referenced environment variable is unset.
    """
    if isinstance(value, Mapping):
        return {key: _expand_environment(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_environment(item) for item in value]
    if not isinstance(value, str):
        return value

    missing = sorted(
        {
            name
            for name in ENVIRONMENT_REFERENCE.findall(value)
            if name not in os.environ
        }
    )
    if missing:
        raise ConfigurationError(
            "Paper 2 configuration requires unset environment variables: "
            + ", ".join(missing)
        )
    return ENVIRONMENT_REFERENCE.sub(lambda match: os.environ[match.group(1)], value)


def _validate_config(values: Mapping[str, Any]) -> None:
    """Validate safety-critical Paper 2 configuration fields."""
    missing = sorted(REQUIRED_TOP_LEVEL_SECTIONS.difference(values))
    if missing:
        raise ConfigurationError(
            "Paper 2 configuration is missing sections: " + ", ".join(missing)
        )
    study = values["study"]
    for field in ("name", "database_version", "random_seed"):
        if field not in study:
            raise ConfigurationError(f"study.{field} is required.")
    database_version = str(study["database_version"]).strip()
    if not database_version or database_version.upper() in {
        "UNKNOWN",
        "REPLACE_WITH_PPMI_SNAPSHOT",
    }:
        raise ConfigurationError(
            "study.database_version must identify the exact PPMI snapshot or "
            "synthetic fixture; placeholder values are forbidden."
        )
    if isinstance(study["random_seed"], bool) or not isinstance(
        study["random_seed"], int
    ):
        raise ConfigurationError("study.random_seed must be an integer.")

    data = values["data"]
    if not isinstance(data.get("column_map"), Mapping):
        raise ConfigurationError("data.column_map must be a mapping.")
    required_canonical = {
        "PATNO",
        "EVENT_ID",
        "VISIT_DATE",
        "DIAGNOSTIC_GROUP",
        "SCORE",
    }
    missing_canonical = sorted(required_canonical.difference(data["column_map"]))
    if missing_canonical:
        raise ConfigurationError(
            "data.column_map is missing canonical fields: "
            + ", ".join(missing_canonical)
        )
    constants = data.get("column_constants", {})
    if not isinstance(constants, Mapping):
        raise ConfigurationError("data.column_constants must be a mapping.")
    unsupported_constants = sorted(set(constants).difference({"MEDICATION_STATE"}))
    if unsupported_constants:
        raise ConfigurationError(
            "data.column_constants contains unsupported fields: "
            + ", ".join(unsupported_constants)
        )

    cohort = values["cohort"]
    for field in ("eligible_groups", "baseline_event_ids"):
        if not isinstance(cohort.get(field), list) or not cohort[field]:
            raise ConfigurationError(f"cohort.{field} must be a non-empty list.")

    endpoint = values["endpoint"]
    for field in ("name", "required_medication_state"):
        if not endpoint.get(field):
            raise ConfigurationError(f"endpoint.{field} is required.")

    blocks = values["feature_blocks"]
    missing_blocks = sorted(set(PAPER2_FEATURE_BLOCKS).difference(blocks))
    if missing_blocks:
        raise ConfigurationError(
            "feature_blocks must declare every Paper 2 domain; missing: "
            + ", ".join(missing_blocks)
        )
    for name, columns in blocks.items():
        if name not in PAPER2_FEATURE_BLOCKS:
            raise ConfigurationError(f"Unknown Paper 2 feature block: {name}")
        if not isinstance(columns, list) or any(
            not isinstance(column, str) for column in columns
        ):
            raise ConfigurationError(
                f"feature_blocks.{name} must be a list of columns."
            )

    validation = values["validation"]
    if validation.get("split_strategy") != "patient_isolated":
        negative_control = validation.get("methodological_negative_control", False)
        if not negative_control:
            raise ConfigurationError(
                "Row-level splitting is forbidden unless explicitly marked as a "
                "methodological_negative_control."
            )
        raise ConfigurationError(
            "The production Paper 2 runner does not execute row-level negative "
            "controls. Use a separately labelled audit script."
        )
    for field in ("outer_splits", "outer_repeats", "inner_splits"):
        value = validation.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 2:
            raise ConfigurationError(f"validation.{field} must be at least 2.")
    allowed_imputation = {"median", "most_frequent", "constant"}
    strategies = validation.get("imputation_strategies")
    if not isinstance(strategies, list) or not strategies:
        raise ConfigurationError(
            "validation.imputation_strategies must be a non-empty list."
        )
    invalid = sorted(set(strategies).difference(allowed_imputation))
    if invalid:
        raise ConfigurationError(
            "Unsupported imputation strategies: " + ", ".join(invalid)
        )

    task_grid = values["task_grid"]
    for field in ("horizons_months", "tolerance_days", "outcome_types"):
        if not isinstance(task_grid.get(field), list) or not task_grid[field]:
            raise ConfigurationError(f"task_grid.{field} must be a non-empty list.")
    if "binary" in {str(value).lower() for value in task_grid["outcome_types"]}:
        threshold = task_grid.get("progression_threshold")
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
            raise ConfigurationError(
                "task_grid.progression_threshold is required for binary outcomes."
            )
    if not isinstance(values["models"], Mapping) or not values["models"]:
        raise ConfigurationError("models must contain at least one model family.")
    supported_models = {"elasticnet", "random_forest", "hist_gradient_boosting"}
    unknown_models = sorted(set(values["models"]).difference(supported_models))
    if unknown_models:
        raise ConfigurationError(
            "Unsupported model families: " + ", ".join(unknown_models)
        )
    for model_name, model_config in values["models"].items():
        if not isinstance(model_config, Mapping):
            raise ConfigurationError(f"models.{model_name} must be a mapping.")
        for grid_name in (
            "parameter_grid",
            "continuous_parameter_grid",
            "binary_parameter_grid",
        ):
            if grid_name in model_config and not isinstance(
                model_config[grid_name], Mapping
            ):
                raise ConfigurationError(
                    f"models.{model_name}.{grid_name} must be a mapping."
                )
            if grid_name in model_config:
                invalid_parameters = [
                    name
                    for name, candidates in model_config[grid_name].items()
                    if not isinstance(candidates, list) or not candidates
                ]
                if invalid_parameters:
                    raise ConfigurationError(
                        f"models.{model_name}.{grid_name} values must be "
                        "non-empty lists: " + ", ".join(invalid_parameters)
                    )

    output = values["output"]
    if not isinstance(output, Mapping) or not output.get("directory"):
        raise ConfigurationError("output.directory is required.")
