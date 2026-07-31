"""Publication-ready specification curves for Paper 2 master summaries."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch, Rectangle

from .exceptions import DataFileNotFoundError, MissingColumnError
from .paper2_config import PAPER2_FEATURE_BLOCKS

LOGGER = logging.getLogger("ppmi_pipeline.plot_specification")

DEFAULT_METRICS = ("r2", "auroc", "brier_score")
METRIC_LABELS = {
    "r2": r"$R^2$",
    "auroc": "AUROC",
    "auprc": "AUPRC",
    "brier_score": "Brier score",
    "mae": "MAE",
    "rmse": "RMSE",
}
PLOT_REQUIRED_COLUMNS = {
    "specification_key",
    "batch_experiment",
    "metric",
    "estimate",
    "ci_lower",
    "ci_upper",
    "horizon_months",
    "tolerance_days",
    "imputation_strategy",
    "validation_strategy",
    "model_family",
}


class SpecificationCurvePlotter:
    """Render metric-specific two-panel specification curves."""

    def __init__(self, master_summary: pd.DataFrame, source_path: Path | None = None):
        """Initialize and validate a master summary.

        Args:
            master_summary: Metric-long output from ``run_all_experiments``.
            source_path: Optional source CSV used for provenance hashing.

        Raises:
            TypeError: If the summary is not a DataFrame.
            MissingColumnError: If plotting fields are missing.
        """
        if not isinstance(master_summary, pd.DataFrame):
            raise TypeError("master_summary must be a pandas DataFrame.")
        missing = sorted(PLOT_REQUIRED_COLUMNS.difference(master_summary.columns))
        if missing:
            raise MissingColumnError(
                "Paper 2 master summary is missing plotting columns: "
                + ", ".join(missing)
            )
        missing_blocks = [
            f"feature_block__{block}"
            for block in PAPER2_FEATURE_BLOCKS
            if f"feature_block__{block}" not in master_summary.columns
        ]
        if missing_blocks:
            raise MissingColumnError(
                "Paper 2 master summary is missing feature-choice columns: "
                + ", ".join(missing_blocks)
            )
        self.master_summary = master_summary.copy()
        self.source_path = source_path

    @classmethod
    def from_csv(cls, path: str | Path) -> SpecificationCurvePlotter:
        """Load a master summary from CSV.

        Args:
            path: Master-summary CSV path.

        Returns:
            Validated plotter instance.

        Raises:
            DataFileNotFoundError: If the CSV is missing.
        """
        source = Path(path).resolve()
        if not source.exists():
            raise DataFileNotFoundError(
                f"Paper 2 master summary was not found: {source}"
            )
        return cls(pd.read_csv(source), source_path=source)

    def plot_metric(
        self,
        metric: str,
        output_dir: str | Path = "outputs/figures",
    ) -> dict[str, Path]:
        """Plot one metric and its aligned analytical-choice grid.

        Args:
            metric: Metric name present in the long master summary.
            output_dir: Destination for PDF, PNG, and metadata JSON.

        Returns:
            Mapping containing the three generated artifact paths.

        Raises:
            ValueError: If the metric has no finite estimates or duplicate rows.
        """
        data = self.master_summary.loc[self.master_summary["metric"].eq(metric)].copy()
        for column in ("estimate", "ci_lower", "ci_upper"):
            data[column] = pd.to_numeric(data[column], errors="coerce")
        data = data.loc[data["estimate"].notna()].copy()
        if data.empty:
            raise ValueError(
                f"No finite estimates are available for metric {metric!r}."
            )
        if data["specification_key"].duplicated().any():
            duplicates = sorted(
                data.loc[
                    data["specification_key"].duplicated(), "specification_key"
                ].unique()
            )
            raise ValueError(
                f"Metric {metric!r} contains duplicate specifications: {duplicates}"
            )
        data = data.sort_values(
            ["estimate", "specification_key"], kind="stable"
        ).reset_index(drop=True)
        data["plot_index"] = np.arange(1, len(data) + 1)
        x = np.arange(len(data))
        width = max(12.0, min(30.0, 0.32 * len(data) + 6.0))
        sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
        figure = plt.figure(figsize=(width, 10.5), constrained_layout=True)
        grid = figure.add_gridspec(2, 1, height_ratios=(2.2, 3.4), hspace=0.04)
        performance_axis = figure.add_subplot(grid[0])
        choice_axis = figure.add_subplot(grid[1], sharex=performance_axis)

        experiments = list(dict.fromkeys(data["batch_experiment"].astype(str)))
        palette = sns.color_palette("colorblind", n_colors=max(1, len(experiments)))
        colors = dict(zip(experiments, palette, strict=True))
        for index, row in data.iterrows():
            lower = row["estimate"] - row["ci_lower"]
            upper = row["ci_upper"] - row["estimate"]
            error = None
            if np.isfinite(lower) and np.isfinite(upper) and lower >= 0 and upper >= 0:
                error = np.array([[lower], [upper]])
            performance_axis.errorbar(
                index,
                row["estimate"],
                yerr=error,
                fmt="o",
                color=colors[str(row["batch_experiment"])],
                ecolor="#4d4d4d",
                elinewidth=1.0,
                capsize=2.5,
                markersize=5.5,
                zorder=3,
            )
        performance_axis.set_ylabel(
            f"{METRIC_LABELS.get(metric, metric)} with patient-bootstrap CI"
        )
        performance_axis.set_title(
            f"Paper 2 specification curve: {METRIC_LABELS.get(metric, metric)}",
            fontweight="bold",
            pad=10,
        )
        performance_axis.tick_params(axis="x", labelbottom=False)
        performance_axis.grid(axis="x", visible=False)
        if metric == "r2":
            performance_axis.axhline(0, color="#777777", linewidth=0.9, linestyle="--")
        elif metric == "auroc":
            performance_axis.axhline(
                0.5, color="#777777", linewidth=0.9, linestyle="--"
            )
        performance_axis.legend(
            handles=[
                Patch(facecolor=colors[name], edgecolor="none", label=name)
                for name in experiments
            ],
            title="Experiment",
            loc="best",
            frameon=True,
        )

        choices = _choice_rows(data)
        self._draw_choice_grid(choice_axis, choices, x)
        choice_axis.set_xticks(x)
        choice_axis.set_xticklabels(
            [f"S{value:03d}" for value in data["plot_index"]],
            rotation=90,
            fontsize=7,
        )
        choice_axis.set_xlabel("Specifications ordered by estimate (low to high)")
        figure.text(
            0.995,
            0.005,
            "All confidence intervals use patient-level cluster bootstrap resampling.",
            ha="right",
            va="bottom",
            fontsize=8,
            color="#555555",
        )

        destination = Path(output_dir).resolve()
        destination.mkdir(parents=True, exist_ok=True)
        stem = destination / f"specification_curve_{metric}"
        pdf_path = stem.with_suffix(".pdf")
        png_path = stem.with_suffix(".png")
        metadata_path = stem.with_suffix(".json")
        _atomic_figure_write(figure, pdf_path, "pdf", dpi=300)
        _atomic_figure_write(figure, png_path, "png", dpi=300)
        plt.close(figure)
        metadata = {
            "artifact_type": "paper2_specification_curve",
            "created_at_utc": datetime.now(UTC).isoformat(),
            "metric": metric,
            "specification_count": int(len(data)),
            "ordering": "estimate_ascending",
            "patient_level_bootstrap_intervals": True,
            "source_path": str(self.source_path) if self.source_path else None,
            "source_sha256": (
                _file_sha256(self.source_path) if self.source_path else None
            ),
            "ordered_specification_keys": data["specification_key"].tolist(),
            "files": {
                "pdf": {"path": str(pdf_path), "sha256": _file_sha256(pdf_path)},
                "png": {"path": str(png_path), "sha256": _file_sha256(png_path)},
            },
        }
        _atomic_text_write(
            metadata_path,
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        )
        return {"pdf": pdf_path, "png": png_path, "metadata": metadata_path}

    def plot_all(
        self,
        output_dir: str | Path = "outputs/figures",
        metrics: Sequence[str] = DEFAULT_METRICS,
    ) -> dict[str, dict[str, Path]]:
        """Generate every available requested metric-specific curve.

        Args:
            output_dir: Destination directory.
            metrics: Ordered metric names to attempt.

        Returns:
            Nested mapping from metric to generated artifacts.

        Raises:
            ValueError: If none of the requested metrics are available.
        """
        available = set(self.master_summary["metric"].dropna().astype(str))
        outputs: dict[str, dict[str, Path]] = {}
        for metric in metrics:
            if metric not in available:
                LOGGER.info("Skipping unavailable specification metric: %s", metric)
                continue
            outputs[metric] = self.plot_metric(metric, output_dir)
        if not outputs:
            raise ValueError(
                "None of the requested specification metrics are available."
            )
        return outputs

    @staticmethod
    def _draw_choice_grid(
        axis: plt.Axes,
        choices: list[tuple[str, list[str], bool]],
        x: np.ndarray,
    ) -> None:
        """Draw categorical and binary analytical choices as aligned tiles."""
        categorical_palette = sns.color_palette("colorblind", n_colors=10)
        for row_index, (_, values, binary) in enumerate(choices):
            categories = list(dict.fromkeys(values))
            category_colors = {
                category: categorical_palette[index % len(categorical_palette)]
                for index, category in enumerate(categories)
            }
            for column_index, value in enumerate(values):
                if binary:
                    color = "#2166ac" if value == "1" else "#eeeeee"
                    label = "•" if value == "1" else ""
                else:
                    color = category_colors[value]
                    label = _short_label(value)
                axis.add_patch(
                    Rectangle(
                        (column_index - 0.5, row_index - 0.5),
                        1,
                        1,
                        facecolor=color,
                        edgecolor="white",
                        linewidth=0.6,
                    )
                )
                if label:
                    axis.text(
                        column_index,
                        row_index,
                        label,
                        ha="center",
                        va="center",
                        fontsize=6.5 if not binary else 8,
                        color="white" if binary or _is_dark(color) else "black",
                    )
        axis.set_yticks(np.arange(len(choices)))
        axis.set_yticklabels([label for label, _, _ in choices], fontsize=8)
        axis.set_ylim(len(choices) - 0.5, -0.5)
        axis.set_xlim(-0.5, len(x) - 0.5)
        axis.grid(False)
        axis.set_title(
            "Aligned analytical choices",
            loc="left",
            fontweight="bold",
            fontsize=11,
        )


def _choice_rows(data: pd.DataFrame) -> list[tuple[str, list[str], bool]]:
    """Convert master-summary fields into ordered display rows."""
    rows: list[tuple[str, list[str], bool]] = [
        (
            "Horizon",
            [f"{int(value)}m" for value in data["horizon_months"]],
            False,
        ),
        (
            "Tolerance",
            [f"+/-{int(value)}d" for value in data["tolerance_days"]],
            False,
        ),
        ("Imputation", data["imputation_strategy"].astype(str).tolist(), False),
        ("Validation", data["validation_strategy"].astype(str).tolist(), False),
        ("Model", data["model_family"].astype(str).tolist(), False),
    ]
    for block in PAPER2_FEATURE_BLOCKS:
        values = (
            pd.to_numeric(data[f"feature_block__{block}"], errors="coerce")
            .fillna(0)
            .astype(int)
            .astype(str)
            .tolist()
        )
        rows.append((_feature_block_label(block), values, True))
    return rows


def _feature_block_label(block: str) -> str:
    """Return a concise publication label for one feature domain."""
    labels = {
        "demographics_disease_history": "Demographics/history",
        "baseline_motor": "Baseline motor",
        "cognition_neuropsychology": "Cognition",
        "olfaction_sleep_autonomic": "Olfaction/sleep/autonomic",
        "biofluid_biomarkers": "Biofluid",
        "imaging_dat_spect": "DAT-SPECT",
        "genetic_variables": "Genetics",
    }
    return labels[block]


def _short_label(value: str, maximum: int = 12) -> str:
    """Shorten categorical tile labels without altering stored metadata."""
    replacements = {
        "patient_isolated": "patient-CV",
        "most_frequent": "mode",
        "random_forest": "RF",
        "hist_gradient_boosting": "HGB",
    }
    label = replacements.get(value, value)
    return label if len(label) <= maximum else label[: maximum - 1] + "â€¦"


def _is_dark(color: Any) -> bool:
    """Return whether a Matplotlib-compatible color needs light text."""
    red, green, blue = matplotlib.colors.to_rgb(color)
    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    return luminance < 0.55


def build_parser() -> argparse.ArgumentParser:
    """Build the specification-plot command-line parser.

    Returns:
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(
        description="Generate publication-ready Paper 2 specification curves."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("outputs/paper2_master_summary.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/figures"))
    parser.add_argument("--metrics", nargs="+", default=list(DEFAULT_METRICS))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Generate specification curves from the master summary.

    Args:
        argv: Optional command-line argument sequence.

    Returns:
        Zero after successful figure generation.
    """
    arguments = build_parser().parse_args(argv)
    SpecificationCurvePlotter.from_csv(arguments.input).plot_all(
        arguments.output_dir,
        metrics=arguments.metrics,
    )
    return 0


def _atomic_figure_write(
    figure: plt.Figure,
    path: Path,
    output_format: str,
    dpi: int,
) -> None:
    """Save one figure atomically with explicit format and print resolution."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    figure.savefig(
        temporary,
        format=output_format,
        dpi=dpi,
        bbox_inches="tight",
        facecolor="white",
    )
    temporary.replace(path)


def _atomic_text_write(path: Path, content: str) -> None:
    """Write one metadata artifact atomically."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Calculate the SHA-256 hash of a source or generated artifact."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
