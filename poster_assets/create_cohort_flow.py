"""Create a fixed-horizon cohort-flow diagram from the active PPMI pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from adapter import harmonize_schema
from data_utils import create_fixed_horizon_dataset
from etl import PPMIDataLoader
from poster_assets.real_run import resolve_latest_real_configuration

DEFAULT_OUTPUT_DIRECTORY = Path("results") / "poster_assets"


def collect_cohort_counts(config: dict[str, Any]) -> dict[str, int]:
    """Run the canonical preparation steps and return auditable cohort counts.

    Counts are derived from the same ETL, schema-adaptation, and fixed-horizon
    functions used by ``main.py``. The first count represents validated visit
    records after ETL normalisation, not unvalidated source rows.

    Args:
        config: Fixed-horizon configuration with data path, score, and horizon
            entries used by the canonical pipeline.

    Returns:
        Counts for validated visits, baseline eligibility, final retention, and
        exclusion criteria.

    Raises:
        KeyError: If a required fixed-horizon configuration entry is missing.
        ValueError: If the configured data cannot produce a usable cohort.
    """
    required_keys = {
        "data_path",
        "sheet_name",
        "score_column",
        "target_horizon_days",
        "window_tolerance_days",
    }
    missing_keys = sorted(required_keys.difference(config))
    if missing_keys:
        raise KeyError(
            "Cohort-flow configuration is missing required entries: "
            + ", ".join(missing_keys)
        )

    etl_data = PPMIDataLoader().load(
        config["data_path"], sheet_name=config["sheet_name"]
    )
    harmonized_data = harmonize_schema(etl_data, score_column=config["score_column"])
    _, fixed_horizon_summary = create_fixed_horizon_dataset(
        harmonized_data,
        target_horizon_days=config["target_horizon_days"],
        window_tolerance=config["window_tolerance_days"],
    )
    baseline_patients = int(
        harmonized_data.loc[
            harmonized_data["EVENT_ID"]
            .astype("string")
            .str.strip()
            .str.upper()
            .eq("BL"),
            "PATNO",
        ].nunique()
    )
    return {
        "validated_visit_records": int(len(harmonized_data)),
        "unique_patients": int(harmonized_data["PATNO"].nunique()),
        "baseline_patients": baseline_patients,
        "usable_baseline_patients": fixed_horizon_summary["usable_baseline_patients"],
        "retained_patients": fixed_horizon_summary["retained_patients"],
        "excluded_invalid_baseline": fixed_horizon_summary["excluded_invalid_baseline"],
        "excluded_missing_follow_up": fixed_horizon_summary[
            "excluded_missing_follow_up"
        ],
    }


def create_cohort_flow_diagram(
    cohort_counts: dict[str, int],
    target_horizon_days: int,
    window_tolerance_days: int,
    output_directory: str | Path = DEFAULT_OUTPUT_DIRECTORY,
    dpi: int = 300,
) -> tuple[Path, Path]:
    """Create a publication-ready fixed-horizon cohort-flow diagram.

    Args:
        cohort_counts: Counts returned by :func:`collect_cohort_counts`.
        target_horizon_days: Prespecified number of days after baseline.
        window_tolerance_days: Allowed deviation from the target horizon.
        output_directory: Directory for PNG and PDF output files.
        dpi: Raster output resolution. Must be at least 300.

    Returns:
        Paths to the saved PNG and PDF figures, in that order.

    Raises:
        ValueError: If required counts are unavailable or DPI is below 300.
    """
    required_counts = {
        "validated_visit_records",
        "unique_patients",
        "baseline_patients",
        "usable_baseline_patients",
        "retained_patients",
        "excluded_invalid_baseline",
        "excluded_missing_follow_up",
    }
    missing_counts = sorted(required_counts.difference(cohort_counts))
    if missing_counts:
        raise ValueError("Missing cohort counts: " + ", ".join(missing_counts))
    if dpi < 300:
        raise ValueError("dpi must be at least 300 for poster-quality output.")

    figure, axis = plt.subplots(figsize=(11.2, 8.6), constrained_layout=True)
    axis.set_axis_off()
    axis.set_xlim(0, 11)
    axis.set_ylim(0, 10)

    main_boxes = [
        (
            1.4,
            8.35,
            "Validated PPMI clinical visit records\n"
            f"n = {cohort_counts['validated_visit_records']:,} records; "
            f"{cohort_counts['unique_patients']:,} patients",
        ),
        (
            1.4,
            6.35,
            f"Patients with a BL event\nn = {cohort_counts['baseline_patients']:,}",
        ),
        (
            1.4,
            4.35,
            "Usable baseline patients\n"
            f"n = {cohort_counts['usable_baseline_patients']:,}",
        ),
        (
            1.4,
            2.35,
            "Final fixed-horizon modelling cohort\n"
            f"n = {cohort_counts['retained_patients']:,} patients",
        ),
    ]
    for x_position, y_position, label in main_boxes:
        _draw_box(axis, x_position, y_position, label, facecolor="#E6F2F8")

    _draw_arrow(axis, (3.3, 7.82), (3.3, 6.88), "Restrict to EVENT_ID = BL")
    _draw_arrow(
        axis,
        (3.3, 5.82),
        (3.3, 4.88),
        "Require PATNO, VISIT_DATE, and SCORE",
    )
    _draw_arrow(
        axis,
        (3.3, 3.82),
        (3.3, 2.88),
        "Select closest follow-up within "
        f"{target_horizon_days} $\\pm$ {window_tolerance_days} days",
    )

    _draw_box(
        axis,
        6.4,
        4.35,
        "Excluded: invalid baseline\n"
        f"n = {cohort_counts['excluded_invalid_baseline']:,}",
        facecolor="#FCE8E6",
    )
    _draw_box(
        axis,
        6.4,
        2.35,
        "Excluded: no usable follow-up\n"
        f"within horizon window\nn = {cohort_counts['excluded_missing_follow_up']:,}",
        facecolor="#FCE8E6",
    )
    _draw_side_arrow(axis, (5.38, 4.35), (6.22, 4.35))
    _draw_side_arrow(axis, (5.38, 2.35), (6.22, 2.35))

    axis.text(
        5.5,
        9.55,
        "Fixed-Horizon PPMI Cohort Flow",
        ha="center",
        va="center",
        fontsize=20,
        fontweight="bold",
    )
    axis.text(
        5.5,
        0.62,
        "Counts are generated from the canonical ETL, schema-harmonisation, "
        "and fixed-horizon dataset-construction workflow.",
        ha="center",
        va="center",
        fontsize=11,
        color="#444444",
    )

    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)
    png_path = destination / "cohort_flow_diagram.png"
    pdf_path = destination / "cohort_flow_diagram.pdf"
    figure.savefig(
        png_path, dpi=dpi, bbox_inches="tight", pad_inches=0.16, facecolor="white"
    )
    figure.savefig(
        pdf_path, dpi=dpi, bbox_inches="tight", pad_inches=0.16, facecolor="white"
    )
    plt.close(figure)
    return png_path, pdf_path


def _draw_box(
    axis: plt.Axes,
    x_position: float,
    y_position: float,
    label: str,
    facecolor: str,
) -> None:
    """Draw one labelled cohort-flow box."""
    box = FancyBboxPatch(
        (x_position, y_position - 0.45),
        3.8,
        0.9,
        boxstyle="round,pad=0.10,rounding_size=0.09",
        facecolor=facecolor,
        edgecolor="#3A3A3A",
        linewidth=1.0,
    )
    axis.add_patch(box)
    axis.text(
        x_position + 1.9, y_position, label, ha="center", va="center", fontsize=12
    )


def _draw_arrow(
    axis: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    label: str,
) -> None:
    """Draw a vertical transition arrow and its inclusion criterion."""
    axis.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=1.2,
            shrinkA=2,
            shrinkB=2,
        )
    )
    axis.text(
        start[0] + 0.25,
        (start[1] + end[1]) / 2,
        label,
        ha="left",
        va="center",
        fontsize=10,
    )


def _draw_side_arrow(
    axis: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
) -> None:
    """Draw an exclusion arrow from the main flow to a side box."""
    axis.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=1.2,
            color="#A33A2B",
            shrinkA=2,
            shrinkB=2,
        )
    )


def main() -> None:
    """Generate a cohort-flow figure using the active fixed-horizon configuration."""
    parser = argparse.ArgumentParser(
        description="Create a fixed-horizon PPMI cohort-flow diagram."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
        help="Directory for cohort_flow_diagram.png and .pdf.",
    )
    arguments = parser.parse_args()

    real_config, _ = resolve_latest_real_configuration()
    cohort_counts = collect_cohort_counts(real_config)
    png_path, pdf_path = create_cohort_flow_diagram(
        cohort_counts,
        target_horizon_days=real_config["target_horizon_days"],
        window_tolerance_days=real_config["window_tolerance_days"],
        output_directory=arguments.output_dir,
    )
    print(f"Saved cohort-flow diagram to {png_path} and {pdf_path}.")


if __name__ == "__main__":
    main()
