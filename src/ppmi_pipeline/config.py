"""Versioned configuration for the PPMI-Pipeline v2.1.8 methodology framework."""

from __future__ import annotations

from pathlib import Path
from typing import Any

SOFTWARE_VERSION = "2.1.8"
PROJECT_ROOT = Path.cwd()
PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_DUMMY_DATA_PATH = PACKAGE_ROOT / "sample_data" / "dummy_ppmi.csv"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results"
MODELING_RESULTS_DIR = DEFAULT_RESULTS_DIR / "modeling"
PIPELINE_LOG_PATH = PROJECT_ROOT / "pipeline.log"

DEFAULT_RANDOM_STATE = 42
DEFAULT_N_SPLITS = 5
DEFAULT_TARGET_HORIZON_DAYS = 365
DEFAULT_WINDOW_TOLERANCE_DAYS = 90
DEFAULT_PROGRESSION_THRESHOLD = 5.0
DEFAULT_FIGURE_DPI = 300
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_SCORE_COLUMN = "updrs3_score"
# Backward-compatible alias retained for existing imports.
DEFAULT_PROGNOSTIC_SCORE_COLUMN = DEFAULT_SCORE_COLUMN

DEFAULT_LOGISTIC_REGRESSION_CONFIG: dict[str, Any] = {
    "c": 1.0,
    "l1_ratio": 0.5,
    "max_iter": 5_000,
    "random_state": DEFAULT_RANDOM_STATE,
}
DEFAULT_DUMMY_DATA_CONFIG: dict[str, Any] = {
    "random_seed": DEFAULT_RANDOM_STATE,
    "n_patients": 50,
    "visits": (("BL", 0), ("V01", 180), ("V02", 365)),
}

FIXED_HORIZON_CONFIG: dict[str, Any] = {
    "data_path": DEFAULT_DUMMY_DATA_PATH,
    "sheet_name": 0,
    "score_column": DEFAULT_SCORE_COLUMN,
    "target_horizon_days": DEFAULT_TARGET_HORIZON_DAYS,
    "window_tolerance_days": DEFAULT_WINDOW_TOLERANCE_DAYS,
    "progression_threshold": DEFAULT_PROGRESSION_THRESHOLD,
    "target_column": "progression_label",
    "n_splits": DEFAULT_N_SPLITS,
    "modeling_results_dir": MODELING_RESULTS_DIR,
    "log_path": PIPELINE_LOG_PATH,
    "log_level": DEFAULT_LOG_LEVEL,
    "logistic_regression": DEFAULT_LOGISTIC_REGRESSION_CONFIG.copy(),
}
