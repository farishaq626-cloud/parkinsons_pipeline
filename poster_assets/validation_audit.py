"""Audit patient-level isolation in the canonical GroupKFold validation."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from adapter import harmonize_schema
from data_utils import create_fixed_horizon_dataset
from etl import PPMIDataLoader
from main import create_progression_label
from poster_assets.real_run import resolve_latest_real_configuration
from validation import ValidationFramework

DEFAULT_OUTPUT_DIRECTORY = Path("results") / "poster_assets"


def run_validation_audit(config: dict[str, Any]) -> tuple[pd.DataFrame, str]:
    """Verify zero patient overlap in every canonical GroupKFold partition.

    Args:
        config: Active fixed-horizon configuration from ``config.py``.

    Returns:
        A fold-level audit table and a poster-ready integrity statement.

    Raises:
        AssertionError: If a patient is found in both train and test data.
        KeyError: If an expected configuration value is absent.
    """
    etl_data = PPMIDataLoader().load(
        config["data_path"], sheet_name=config["sheet_name"]
    )
    harmonized_data = harmonize_schema(etl_data, score_column=config["score_column"])
    fixed_horizon_data, _ = create_fixed_horizon_dataset(
        harmonized_data,
        target_horizon_days=config["target_horizon_days"],
        window_tolerance=config["window_tolerance_days"],
    )
    labelled_data = create_progression_label(
        fixed_horizon_data,
        threshold=config["progression_threshold"],
        target_column=config["target_column"],
    )
    framework = ValidationFramework(
        labelled_data,
        target=config["target_column"],
        n_splits=config["n_splits"],
    )

    audit_rows: list[dict[str, int]] = []
    for fold_number, (train_df, test_df) in enumerate(framework.get_splits(), start=1):
        train_patients = set(train_df["patient_id"])
        test_patients = set(test_df["patient_id"])
        overlap = train_patients.intersection(test_patients)
        ValidationFramework.verify_no_overlap(train_df, test_df)
        audit_rows.append(
            {
                "fold": fold_number,
                "train_patients": len(train_patients),
                "test_patients": len(test_patients),
                "overlapping_patients": len(overlap),
            }
        )

    audit_table = pd.DataFrame(audit_rows)
    if audit_table["overlapping_patients"].ne(0).any():
        raise AssertionError("Patient leakage detected in the validation audit.")
    integrity_statement = (
        f"Integrity statement: Across {len(audit_table)} patient-isolated GroupKFold "
        f"folds ({framework.dataset['patient_id'].nunique()} unique patients), no "
        "patient_id appeared in both the training and testing partition "
        "(0 overlapping patient IDs in every fold)."
    )
    return audit_table, integrity_statement


def save_validation_audit(
    audit_table: pd.DataFrame,
    integrity_statement: str,
    output_directory: str | Path = DEFAULT_OUTPUT_DIRECTORY,
) -> tuple[Path, Path]:
    """Save fold-level validation evidence and a poster-ready statement.

    Args:
        audit_table: Fold-level patient-overlap audit table.
        integrity_statement: Validated text statement describing the audit.
        output_directory: Directory for CSV and text outputs.

    Returns:
        Paths to the CSV audit and plain-text integrity statement.
    """
    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)
    audit_path = destination / "validation_audit.csv"
    statement_path = destination / "integrity_statement.txt"
    audit_table.to_csv(audit_path, index=False)
    statement_path.write_text(integrity_statement + "\n", encoding="utf-8")
    return audit_path, statement_path


def plot_validation_audit(
    audit_table: pd.DataFrame,
    output_directory: str | Path = DEFAULT_OUTPUT_DIRECTORY,
    dpi: int = 300,
) -> tuple[Path, Path]:
    """Create a centred, poster-ready patient-isolation audit panel.

    Args:
        audit_table: Fold-level patient-overlap audit table.
        output_directory: Directory for PNG and PDF output files.
        dpi: Raster output resolution. Must be at least 300.

    Returns:
        Paths to the PNG and PDF integrity-panel outputs, in that order.

    Raises:
        ValueError: If audit fields are missing, an overlap is found, or DPI is
            below the poster-quality threshold.
    """
    required_columns = {
        "fold",
        "train_patients",
        "test_patients",
        "overlapping_patients",
    }
    missing_columns = sorted(required_columns.difference(audit_table.columns))
    if missing_columns:
        raise ValueError("Validation audit is missing: " + ", ".join(missing_columns))
    if audit_table["overlapping_patients"].ne(0).any():
        raise ValueError(
            "Cannot plot a passing integrity panel when overlap is non-zero."
        )
    if dpi < 300:
        raise ValueError("dpi must be at least 300 for poster-quality output.")

    figure, axis = plt.subplots(figsize=(8.6, 5.0), constrained_layout=True)
    axis.set_axis_off()
    table = axis.table(
        cellText=audit_table.astype(int).values,
        colLabels=[
            "Fold",
            "Training patients",
            "Testing patients",
            "Patient overlap",
        ],
        cellLoc="center",
        colLoc="center",
        loc="center",
        colWidths=[0.14, 0.29, 0.28, 0.22],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.0, 1.65)
    for (row_index, _), cell in table.get_celld().items():
        cell.PAD = 0.10
        cell.set_edgecolor("#D0D0D0")
        if row_index == 0:
            cell.set_facecolor("#DCEEF7")
            cell.set_text_props(weight="bold")
        else:
            cell.set_facecolor("#FFFFFF")

    axis.set_title(
        "Patient-Isolated GroupKFold Validation Audit",
        fontsize=18,
        fontweight="bold",
        pad=18,
    )
    figure.text(
        0.5,
        0.08,
        "Zero patient overlap detected in every held-out validation fold.",
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
        color="#166534",
    )
    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)
    png_path = destination / "validation_audit.png"
    pdf_path = destination / "validation_audit.pdf"
    figure.savefig(
        png_path, dpi=dpi, bbox_inches="tight", pad_inches=0.16, facecolor="white"
    )
    figure.savefig(
        pdf_path, dpi=dpi, bbox_inches="tight", pad_inches=0.16, facecolor="white"
    )
    plt.close(figure)
    return png_path, pdf_path


def main() -> None:
    """Run and save the patient-isolation validation audit."""
    parser = argparse.ArgumentParser(
        description="Verify patient-level GroupKFold isolation for the active pipeline."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
        help="Directory for validation_audit.csv and integrity_statement.txt.",
    )
    arguments = parser.parse_args()
    real_config, _ = resolve_latest_real_configuration()
    audit_table, integrity_statement = run_validation_audit(real_config)
    audit_path, statement_path = save_validation_audit(
        audit_table, integrity_statement, arguments.output_dir
    )
    png_path, pdf_path = plot_validation_audit(audit_table, arguments.output_dir)
    print(integrity_statement)
    print(
        "Saved validation audit to "
        f"{audit_path}, {statement_path}, {png_path}, and {pdf_path}."
    )


if __name__ == "__main__":
    main()
