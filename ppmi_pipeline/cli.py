"""Installed command-line entry point for PPMI-Pipeline."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from ppmi_pipeline import __version__
from ppmi_pipeline.main import load_config, run_pipeline


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser for reproducible execution overrides."""
    parser = argparse.ArgumentParser(
        prog="ppmi-pipeline",
        description=(
            "Run the auditable fixed-horizon PPMI methodology pipeline. "
            "Omitted options retain the pinned defaults from config.py."
        ),
    )
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("--data-path", type=Path, help="PPMI CSV or Excel input path.")
    parser.add_argument(
        "--sheet-name",
        help="Excel worksheet name or zero-based worksheet index.",
    )
    parser.add_argument(
        "--score-column",
        help="Normalized score column used for fixed-horizon construction.",
    )
    parser.add_argument("--target-horizon-days", type=int)
    parser.add_argument("--window-tolerance-days", type=int)
    parser.add_argument("--progression-threshold", type=float)
    parser.add_argument("--n-splits", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Execute the methodology pipeline with optional command-line overrides."""
    arguments = build_parser().parse_args(argv)
    config = load_config()
    for field in (
        "data_path",
        "score_column",
        "target_horizon_days",
        "window_tolerance_days",
        "progression_threshold",
        "n_splits",
    ):
        value = getattr(arguments, field)
        if value is not None:
            config[field] = value

    if arguments.sheet_name is not None:
        config["sheet_name"] = (
            int(arguments.sheet_name)
            if arguments.sheet_name.isdigit()
            else arguments.sheet_name
        )
    run_pipeline(config)


if __name__ == "__main__":
    main()
