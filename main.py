"""Canonical entry point for fixed-horizon PPMI prognostic modelling."""

from __future__ import annotations

import copy
import logging
from collections.abc import Mapping
from typing import Any

import pandas as pd

from adapter import harmonize_schema
from config import FIXED_HORIZON_CONFIG
from data_utils import create_fixed_horizon_dataset
from etl import PPMIDataLoader
from exceptions import ConfigurationError
from logging_config import configure_logging
from modeling import PrognosticModel
from validation import ValidationFramework
from visualization import ResultVisualizer


LOGGER = logging.getLogger("ppmi_pipeline.main")


def load_config(config_override: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Return an isolated copy of the canonical fixed-horizon configuration.

    Args:
        config_override: Optional in-memory override for a documented analysis
            run. When omitted, ``FIXED_HORIZON_CONFIG`` remains the canonical
            default source of settings.

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
        "score_column",
        "target_horizon_days",
        "window_tolerance_days",
        "progression_threshold",
        "target_column",
        "n_splits",
        "modeling_results_dir",
        "log_path",
        "logistic_regression",
    }
    missing = sorted(required.difference(config))
    if missing:
        raise ConfigurationError(
            "Fixed-horizon configuration is missing: " + ", ".join(missing)
        )
    return config


def create_progression_label(
    dataset: pd.DataFrame,
    threshold: float,
    target_column: str,
) -> pd.DataFrame:
    """Create a binary motor-progression label from fixed-horizon score change.

    A patient is labelled as progressing when ``Delta_Score`` is greater than
    or equal to the configured clinically declared threshold. The threshold is
    intentionally centralised in ``config.py`` and must be reported with every
    scientific use of this pipeline.

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
        raise ValueError("Cannot create progression label from missing Delta_Score values.")
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
) -> PrognosticModel:
    """Execute the canonical fixed-horizon PPMI prognostic workflow.

    Execution order is: configuration, ETL, schema harmonisation, fixed-horizon
    dataset construction, patient-isolated validation, ElasticNet modelling,
    and publication-quality visualisation.

    Args:
        config_override: Optional in-memory configuration used for one
            explicitly requested analysis run. It is validated through
            :func:`load_config` and does not modify ``config.py``.

    Returns:
        A fitted ``PrognosticModel`` containing metrics and stability reports.

    Raises:
        ConfigurationError: If configured modelling settings are incompatible
            with the available fixed-horizon cohort.
        ValueError: If clinical data cannot support the requested analysis.
    """
    config = load_config(config_override)
    configure_logging(config["log_path"], level=config["log_level"])
    LOGGER.info("Starting fixed-horizon PPMI prognostic pipeline.")

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
    model = PrognosticModel.train_and_evaluate(
        labelled_dataset,
        target=config["target_column"],
        n_splits=config["n_splits"],
        c=config["logistic_regression"]["c"],
        l1_ratio=config["logistic_regression"]["l1_ratio"],
        random_state=config["logistic_regression"]["random_state"],
        output_dir=config["modeling_results_dir"],
        validation_framework=validation,
    )
    visualizer = ResultVisualizer(config["modeling_results_dir"])
    stability_paths = visualizer.plot_stability_importance()
    heatmap_paths = visualizer.plot_coef_heatmap()

    LOGGER.info("Fixed-horizon cohort summary: %s", dropoff_summary)
    LOGGER.info("Per-fold model metrics:\n%s", model.fold_metrics_.to_string(index=False))
    LOGGER.info("Stability plot outputs: %s", stability_paths)
    LOGGER.info("Coefficient heatmap outputs: %s", heatmap_paths)
    LOGGER.info("Fixed-horizon PPMI prognostic pipeline completed successfully.")
    return model


if __name__ == "__main__":
    run_pipeline()
