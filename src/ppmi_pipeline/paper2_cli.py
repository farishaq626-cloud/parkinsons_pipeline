"""Command-line entry point for YAML-defined Paper 2 experiments."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from .exceptions import PPMIPipelineError
from .logging_config import configure_logging
from .paper2_config import load_paper2_config
from .paper2_experiment import Paper2ExperimentRunner

LOGGER = logging.getLogger("ppmi_pipeline.paper2_cli")


def build_parser() -> argparse.ArgumentParser:
    """Build and return the Paper 2 command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Run leakage-resistant, YAML-defined Paper 2 methodology experiments."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to a YAML file under configs/paper2/.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Execute a Paper 2 experiment from a YAML configuration.

    Args:
        argv: Optional argument sequence for testing. Defaults to process args.

    Returns:
        Process exit code: zero on success and two for a controlled failure.
    """
    arguments = build_parser().parse_args(argv)
    configure_logging("paper2_pipeline.log", arguments.log_level)
    try:
        config = load_paper2_config(arguments.config)
        result = Paper2ExperimentRunner(config).run_from_configured_path()
    except (PPMIPipelineError, TypeError, ValueError) as error:
        LOGGER.error("Paper 2 execution failed: %s", error)
        return 2
    LOGGER.info("Paper 2 artifacts written to %s", result.run_directory)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
