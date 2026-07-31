"""Resolve and execute a provenance-checked fixed-horizon reference run."""

from __future__ import annotations

import argparse
import copy
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ppmi_pipeline.config import DEFAULT_RESULTS_DIR, FIXED_HORIZON_CONFIG
from ppmi_pipeline.etl import PPMIDataLoader
from ppmi_pipeline.exceptions import DataFileNotFoundError
from ppmi_pipeline.main import run_pipeline
from ppmi_pipeline.modeling import ExecutionHarnessModel

RUN_LOG_PATTERN = re.compile(r"config:\s*\n(\{.*?\})\s*\nmetrics:", re.DOTALL)
PROVENANCE_FILENAME = "real_run_provenance.json"


def resolve_latest_real_configuration(
    data_path: str | Path | None = None,
    sheet_name: str | int | None = None,
) -> tuple[dict[str, Any], Path | None]:
    """Resolve a validated non-synthetic source from explicit input or run logs.

    The resolver prioritises an explicit non-synthetic path. Otherwise it scans
    historical run logs from newest to oldest, selects the first existing input
    that is not under ``tests/`` and not named as a dummy fixture, and validates
    the canonical ETL header schema before returning it.

    Args:
        data_path: Optional explicit clinical input file. It must not point to a
            synthetic test fixture.
        sheet_name: Optional Excel worksheet override for ``data_path``.

    Returns:
        A fixed-horizon configuration copy and the historical run log that
        identified it. The log path is ``None`` for an explicit input path.

    Raises:
        DataFileNotFoundError: If no validated non-synthetic source is found.
        ValueError: If an explicit path is synthetic or fails schema validation.
    """
    if data_path is not None:
        candidate = Path(data_path).expanduser().resolve()
        if _is_synthetic_path(candidate):
            raise ValueError(
                "Synthetic fixtures cannot be used for real-data poster assets."
            )
        config = _build_real_config(candidate, sheet_name)
        return config, None

    run_logs = sorted(
        DEFAULT_RESULTS_DIR.glob("*_run_log.txt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for run_log in run_logs:
        metadata = _read_logged_config(run_log)
        logged_path = metadata.get("data_path")
        if not isinstance(logged_path, str):
            continue
        candidate = Path(logged_path).expanduser()
        if _is_synthetic_path(candidate) or not candidate.exists():
            continue
        try:
            config = _build_real_config(candidate, metadata.get("sheet_name"))
        except (FileNotFoundError, ValueError):
            continue
        return config, run_log

    raise DataFileNotFoundError(
        "No validated non-synthetic input could be resolved. Supply --data-path "
        "with an approved PPMI .csv or .xlsx export."
    )


def run_real_fixed_horizon_pipeline(
    data_path: str | Path | None = None,
    sheet_name: str | int | None = None,
) -> tuple[ExecutionHarnessModel, dict[str, Any]]:
    """Run the canonical pipeline and persist non-synthetic run provenance.

    Args:
        data_path: Optional explicit real-data input file.
        sheet_name: Optional Excel worksheet override.

    Returns:
        The fitted execution harness and a provenance record for its reports.

    Raises:
        DataFileNotFoundError: If no valid non-synthetic input can be resolved.
        ValueError: If data or model assumptions cannot support the analysis.
    """
    real_config, source_run_log = resolve_latest_real_configuration(
        data_path=data_path, sheet_name=sheet_name
    )
    model = run_pipeline(real_config)
    provenance = {
        "input_classification": "non_synthetic",
        "input_file_name": Path(real_config["data_path"]).name,
        "sheet_name": real_config["sheet_name"],
        "score_column": real_config["score_column"],
        "target_horizon_days": real_config["target_horizon_days"],
        "window_tolerance_days": real_config["window_tolerance_days"],
        "progression_threshold": real_config["progression_threshold"],
        "n_patients": int(model.dataset["patient_id"].nunique()),
        "n_folds": int(len(model.fold_metrics_)),
        "resolved_from_run_log": (
            str(source_run_log.relative_to(DEFAULT_RESULTS_DIR.parent))
            if source_run_log is not None
            else None
        ),
        "generated_at_utc": datetime.now(UTC).isoformat(),
    }
    provenance_path = Path(real_config["modeling_results_dir"]) / PROVENANCE_FILENAME
    provenance_path.write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    return model, provenance


def load_real_run_provenance(
    results_directory: str | Path,
) -> dict[str, Any]:
    """Load and validate provenance before plotting a real-data report.

    Args:
        results_directory: Directory containing canonical model reports.

    Returns:
        Validated real-data provenance information.

    Raises:
        DataFileNotFoundError: If real-run provenance has not been recorded.
        ValueError: If the report provenance is not non-synthetic.
    """
    path = Path(results_directory) / PROVENANCE_FILENAME
    if not path.exists():
        raise DataFileNotFoundError(
            f"Real-data provenance was not found: {path}. Run "
            "python -m scripts.poster_assets.real_run before plotting poster results."
        )
    provenance = json.loads(path.read_text(encoding="utf-8"))
    if provenance.get("input_classification") != "non_synthetic":
        raise ValueError(
            "Poster figures require reports generated from non-synthetic data."
        )
    return provenance


def _build_real_config(
    candidate: Path,
    sheet_name: str | int | None,
) -> dict[str, Any]:
    """Copy canonical settings and validate a non-synthetic input header."""
    candidate = candidate.resolve()
    if not candidate.exists():
        raise DataFileNotFoundError(
            f"Resolved clinical input was not found: {candidate}"
        )
    if _is_synthetic_path(candidate):
        raise ValueError(
            "Synthetic fixtures cannot be used for real-data poster assets."
        )
    selected_sheet = 0 if sheet_name is None else sheet_name
    PPMIDataLoader().validate_file_schema(candidate, sheet_name=selected_sheet)
    config = copy.deepcopy(FIXED_HORIZON_CONFIG)
    config["data_path"] = candidate
    config["sheet_name"] = selected_sheet
    return config


def _read_logged_config(run_log: Path) -> dict[str, Any]:
    """Extract the JSON configuration object embedded in a historical run log."""
    match = RUN_LOG_PATTERN.search(run_log.read_text(encoding="utf-8"))
    if match is None:
        return {}
    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError:
        return {}


def _is_synthetic_path(path: Path) -> bool:
    """Return whether a path points to the repository's synthetic test fixture."""
    parts = {part.lower() for part in path.parts}
    return (
        "tests" in parts
        or "dummy" in path.name.lower()
        or "synthetic" in path.name.lower()
    )


def main() -> None:
    """Resolve a non-synthetic input and execute the canonical pipeline."""
    parser = argparse.ArgumentParser(
        description="Run the canonical fixed-horizon pipeline on a real-data source."
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        help="Optional approved non-synthetic PPMI input file.",
    )
    parser.add_argument(
        "--sheet-name",
        help="Optional Excel worksheet name or index.",
    )
    arguments = parser.parse_args()
    sheet_name: str | int | None = arguments.sheet_name
    if sheet_name is not None and sheet_name.isdigit():
        sheet_name = int(sheet_name)
    _, provenance = run_real_fixed_horizon_pipeline(
        data_path=arguments.data_path,
        sheet_name=sheet_name,
    )
    print("Generated canonical non-synthetic model reports.")
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
