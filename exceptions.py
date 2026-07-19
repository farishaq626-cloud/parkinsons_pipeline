"""Project-specific exceptions with clear clinical-pipeline error messages."""


class PPMIPipelineError(Exception):
    """Base exception for expected PPMI pipeline failures."""


class ConfigurationError(PPMIPipelineError, ValueError):
    """Raised when an experiment configuration is missing or invalid."""


class DataFileNotFoundError(PPMIPipelineError, FileNotFoundError):
    """Raised when a required clinical input or modelling report is absent."""


class MissingColumnError(PPMIPipelineError, ValueError):
    """Raised when a required clinical or modelling column is absent."""
