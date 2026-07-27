"""Canonical v2.1.7 entry point for the PPMI methodology framework."""

from __future__ import annotations

import copy
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from adapter import harmonize_schema
from config import FIXED_HORIZON_CONFIG
from data_utils import create_fixed_horizon_dataset
from etl import PPMIDataLoader
from exceptions import ConfigurationError
from logging_config import configure_logging
from modeling import ExecutionHarnessModel
from validation import ValidationFramework
from visualization import ResultVisualizer

LOGGER = logging.getLogger("ppmi_pipeline.main")


def load_config(config_override: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Return and validate an isolated fixed-horizon configuration.

    Args:
        config_override: Optional complete in-memory configuration for a
            documented execution. When omitted, ``FIXED_HORIZON_CONFIG``
            remains the canonical default source of settings.

    Returns:
        A deep copy of ``FIXED_HORIZON_CONFIG`` from ``config.py``.

    Raises:
        ConfigurationError: If required fixed-horizon settings are absent.
    """
    config = copy.deepcopy(
        FIXED_HORIZON_CONFIG if config_override is None else dict(config_override)
    )
    required = {
        "data_path",
        "sheet_name",
        "score_column",
        "target_horizon_days",
        "window_tolerance_days",
        "progression_threshold",
        "target_column",
        "n_splits",
        "modeling_results_dir",
        "log_path",
        "log_level",
        "logistic_regression",
    }
    missing = sorted(required.difference(config))
    if missing:
        raise ConfigurationError(
            "Fixed-horizon configuration is missing: " + ", ".join(missing)
        )

    path_fields = ("data_path", "modeling_results_dir", "log_path")
    for field in path_fields:
        if not isinstance(config[field], (str, Path)):
            raise ConfigurationError(
                f"Configuration field '{field}' must be a string or pathlib.Path."
            )

    sheet_name = config["sheet_name"]
    if isinstance(sheet_name, bool) or not isinstance(sheet_name, (str, int)):
        raise ConfigurationError(
            "Configuration field 'sheet_name' must be a worksheet name "
            "or integer index."
        )

    for field in ("score_column", "target_column", "log_level"):
        value = config[field]
        if not isinstance(value, str) or not value.strip():
            raise ConfigurationError(
                f"Configuration field '{field}' must be a non-empty string."
            )

    numeric_level = logging.getLevelName(config["log_level"].upper())
    if not isinstance(numeric_level, int):
        raise ConfigurationError(
            f"Configuration field 'log_level' is invalid: {config['log_level']!r}."
        )

    positive_integer_fields = ("target_horizon_days", "n_splits")
    for field in positive_integer_fields:
        value = config[field]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ConfigurationError(
                f"Configuration field '{field}' must be a positive integer."
            )
    if config["n_splits"] < 2:
        raise ConfigurationError("Configuration field 'n_splits' must be at least 2.")

    tolerance = config["window_tolerance_days"]
    if isinstance(tolerance, bool) or not isinstance(tolerance, int) or tolerance < 0:
        raise ConfigurationError(
            "Configuration field 'window_tolerance_days' must be a "
            "non-negative integer."
        )

    threshold = config["progression_threshold"]
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise ConfigurationError(
            "Configuration field 'progression_threshold' must be numeric."
        )

    model_config = config["logistic_regression"]
    if not isinstance(model_config, Mapping):
        raise ConfigurationError(
            "Configuration field 'logistic_regression' must be a mapping."
        )
    required_model_fields = {"c", "l1_ratio", "max_iter", "random_state"}
    missing_model_fields = sorted(required_model_fields.difference(model_config))
    if missing_model_fields:
        raise ConfigurationError(
            "Logistic-regression configuration is missing: "
            + ", ".join(missing_model_fields)
        )

    c_value = model_config["c"]
    if (
        isinstance(c_value, bool)
        or not isinstance(c_value, (int, float))
        or c_value <= 0
    ):
        raise ConfigurationError(
            "Logistic-regression field 'c' must be a positive number."
        )
    l1_ratio = model_config["l1_ratio"]
    if (
        isinstance(l1_ratio, bool)
        or not isinstance(l1_ratio, (int, float))
        or not 0 <= l1_ratio <= 1
    ):
        raise ConfigurationError(
            "Logistic-regression field 'l1_ratio' must be between 0 and 1."
        )
    max_iter = model_config["max_iter"]
    if isinstance(max_iter, bool) or not isinstance(max_iter, int) or max_iter <= 0:
        raise ConfigurationError(
            "Logistic-regression field 'max_iter' must be a positive integer."
        )
    random_state = model_config["random_state"]
    if isinstance(random_state, bool) or not isinstance(random_state, int):
        raise ConfigurationError(
            "Logistic-regression field 'random_state' must be an integer."
        )
    return config


def create_progression_label(
    dataset: pd.DataFrame,
    threshold: float,
    target_column: str,
) -> pd.DataFrame:
    """Create a binary execution target from fixed-horizon score change.

    A patient is labelled as progressing when ``Delta_Score`` is greater than
    or equal to the configured methodology threshold. The threshold is
    centralized in ``config.py`` and must be reported with every execution.

    Args:
        dataset: Fixed-horizon patient-level dataset containing ``Delta_Score``.
        threshold: Minimum score increase denoting progression.
        target_column: Name of the generated binary target column.

    Returns:
        A copy of ``dataset`` with a binary progression-label column.

    Raises:
        ConfigurationError: If the threshold is invalid or only one label class
            is produced.
        ValueError: If ``Delta_Score`` is absent or contains missing values.
    """
    if "Delta_Score" not in dataset.columns:
        raise ValueError("Cannot create progression label; Delta_Score is missing.")
    if not isinstance(threshold, (int, float)) or isinstance(threshold, bool):
        raise ConfigurationError("progression_threshold must be a numeric value.")
    if dataset["Delta_Score"].isna().any():
        raise ValueError(
            "Cannot create progression label from missing Delta_Score values."
        )
    labelled = dataset.copy()
    labelled[target_column] = (labelled["Delta_Score"] >= threshold).astype("int64")
    if labelled[target_column].nunique() != 2:
        raise ConfigurationError(
            "Configured progression_threshold produced one target class. "
            "Adjust config.py or provide a cohort with both outcome classes."
        )
    return labelled


def run_pipeline(
    config_override: Mapping[str, Any] | None = None,
) -> ExecutionHarnessModel:
    """Execute the canonical fixed-horizon computational workflow.

    Execution order is: configuration, ETL, schema harmonisation, fixed-horizon
    dataset construction, patient-isolated validation, ElasticNet modelling,
    and publication-quality visualisation.

    Args:
        config_override: Optional in-memory configuration used for one
            explicitly requested analysis run. It is validated through
            :func:`load_config` and does not modify ``config.py``.

    Returns:
        A fitted ``ExecutionHarnessModel`` containing diagnostic reports.

    Raises:
        ConfigurationError: If configured modelling settings are incompatible
            with the available fixed-horizon cohort.
        ValueError: If the input data cannot support the requested execution.
    """
    config = load_config(config_override)
    configure_logging(config["log_path"], level=config["log_level"])
    LOGGER.info("Starting the fixed-horizon PPMI methodology pipeline.")

    clinical_data = PPMIDataLoader().load(
        config["data_path"], sheet_name=config["sheet_name"]
    )
    harmonized_data = harmonize_schema(clinical_data, config["score_column"])
    horizon_dataset, dropoff_summary = create_fixed_horizon_dataset(
        harmonized_data,
        target_horizon_days=config["target_horizon_days"],
        window_tolerance=config["window_tolerance_days"],
    )
    labelled_dataset = create_progression_label(
        horizon_dataset,
        threshold=config["progression_threshold"],
        target_column=config["target_column"],
    )

    validation = ValidationFramework(
        labelled_dataset,
        target=config["target_column"],
        n_splits=config["n_splits"],
    )
    model = ExecutionHarnessModel.train_and_evaluate(
        labelled_dataset,
        target=config["target_column"],
        n_splits=config["n_splits"],
        c=config["logistic_regression"]["c"],
        l1_ratio=config["logistic_regression"]["l1_ratio"],
        max_iter=config["logistic_regression"]["max_iter"],
        random_state=config["logistic_regression"]["random_state"],
        output_dir=config["modeling_results_dir"],
        validation_framework=validation,
    )
    visualizer = ResultVisualizer(config["modeling_results_dir"])
    stability_paths = visualizer.plot_stability_importance()
    heatmap_paths = visualizer.plot_coef_heatmap()

    LOGGER.info("Fixed-horizon cohort summary: %s", dropoff_summary)
    LOGGER.info(
        "Per-fold model metrics:\n%s", model.fold_metrics_.to_string(index=False)
    )
    LOGGER.info("Stability plot outputs: %s", stability_paths)
    LOGGER.info("Coefficient heatmap outputs: %s", heatmap_paths)
    LOGGER.info("Fixed-horizon PPMI methodology pipeline completed successfully.")
    return model


def main() -> None:
    """Run the default methodology pipeline for script and console entry points."""
    run_pipeline()


if __name__ == "__main__":
    main()
