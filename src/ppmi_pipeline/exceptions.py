"""Project-specific exceptions with clear methodology-pipeline diagnostics."""


class PPMIPipelineError(Exception):
    """Base exception for expected PPMI pipeline failures."""


class ConfigurationError(PPMIPipelineError, ValueError):
    """Raised when an experiment configuration is missing or invalid."""


class DataFileNotFoundError(PPMIPipelineError, FileNotFoundError):
    """Raised when a required clinical input or modelling report is absent."""


class MissingColumnError(PPMIPipelineError, ValueError):
    """Raised when a required clinical or modelling column is absent."""


class CohortDefinitionError(PPMIPipelineError, ValueError):
    """Raised when a Paper 2 cohort contract is invalid or cannot be applied."""


class PhenotypeDefinitionError(PPMIPipelineError, ValueError):
    """Raised when an endpoint or medication-state contract is invalid."""


class PatientLeakageError(PPMIPipelineError, AssertionError):
    """Raised when one patient appears in training and evaluation partitions."""
