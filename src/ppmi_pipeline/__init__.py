"""Public package API for the PPMI computational methodology framework."""

from .adapter import harmonize_paper2_schema, harmonize_schema
from .cohort import CohortBuilder, CohortDefinition
from .config import SOFTWARE_VERSION
from .data_utils import (
    create_fixed_horizon_dataset,
    create_fixed_horizon_feature_dataset,
)
from .etl import PPMIDataLoader
from .main import create_progression_label, load_config, run_pipeline
from .modeling import ExecutionHarnessModel, PrognosticModel
from .paper2_config import load_paper2_config
from .paper2_experiment import Paper2ExperimentRunner
from .paper2_validation import NestedPatientValidationRunner
from .phenotypes import EndpointDefinition, MedicationState
from .task_spec import OutcomeType, TaskSpecification
from .validation import ValidationFramework

__version__ = SOFTWARE_VERSION

__all__ = [
    "PPMIDataLoader",
    "CohortBuilder",
    "CohortDefinition",
    "EndpointDefinition",
    "ExecutionHarnessModel",
    "PrognosticModel",
    "MedicationState",
    "NestedPatientValidationRunner",
    "OutcomeType",
    "Paper2ExperimentRunner",
    "TaskSpecification",
    "ValidationFramework",
    "create_fixed_horizon_dataset",
    "create_fixed_horizon_feature_dataset",
    "create_progression_label",
    "harmonize_paper2_schema",
    "harmonize_schema",
    "load_paper2_config",
    "load_config",
    "run_pipeline",
]
