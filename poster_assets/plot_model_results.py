"""Create poster-ready diagnostics from canonical fixed-horizon reports."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from config import MODELING_RESULTS_DIR
from exceptions import DataFileNotFoundError, MissingColumnError
from poster_assets.real_run import load_real_run_provenance

DEFAULT_OUTPUT_DIRECTORY = Path("results") / "poster_assets"
POSTER_DPI = 300
POSTER_FONT_SCALE = 1.3
COLOURBLIND_BLUE = "#0072B2"
COLOURBLIND_ORANGE = "#D55E00"


def load_model_reports(
    results_directory: str | Path = MODELING_RESULTS_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the three reports emitted by ``ExecutionHarnessModel.save_reports``.

    Args:
        results_directory: Directory containing ``fold_metrics.csv``,
            ``feature_stability.csv``, and ``fold_coefficients.csv``.

    Returns:
        Fold metrics, feature-stability, and fold-coefficient DataFrames.

    Raises:
        FileNotFoundError: If a required model report is absent.
        ValueError: If a report does not contain its expected fields.
    """
    directory = Path(results_directory)
    load_real_run_provenance(directory)
    report_names = {
        "fold_metrics": "fold_metrics.csv",
        "feature_stability": "feature_stability.csv",
        "fold_coefficients": "fold_coefficients.csv",
    }
    reports: dict[str, pd.DataFrame] = {}
    for report_key, filename in report_names.items():
        report_path = directory / filename
        if not report_path.exists():
            raise DataFileNotFoundError(
                f"Required modelling report was not found: {report_path}. "
                "Run python main.py before generating poster figures."
            )
        reports[report_key] = pd.read_csv(report_path)

    _require_columns(
        reports["fold_metrics"],
        {"fold", "precision", "f1_score", "auc_roc"},
        "fold_metrics.csv",
    )
    _require_columns(
        reports["feature_stability"],
        {
            "feature",
            "mean_normalized_importance",
            "importance_standard_deviation",
            "stability_rate",
        },
        "feature_stability.csv",
    )
    _require_columns(
        reports["fold_coefficients"],
        {"fold", "feature"},
        "fold_coefficients.csv",
    )
    return (
        reports["fold_metrics"],
        reports["feature_stability"],
        reports["fold_coefficients"],
    )


def plot_fold_performance(
    fold_metrics: pd.DataFrame,
    output_directory: str | Path = DEFAULT_OUTPUT_DIRECTORY,
    dpi: int = POSTER_DPI,
) -> tuple[Path, Path]:
    """Plot AUC-ROC, F1, and precision distributions across validation folds.

    Args:
        fold_metrics: ``fold_metrics.csv`` content from the canonical model.
        output_directory: Directory for PNG and PDF output files.
        dpi: Raster output resolution. Must be at least 300.

    Returns:
        Paths to PNG and PDF output files, in that order.
    """
    _require_columns(
        fold_metrics,
        {"fold", "precision", "f1_score", "auc_roc"},
        "fold metrics",
    )
    _validate_dpi(dpi)
    plot_data = fold_metrics.melt(
        id_vars="fold",
        value_vars=["auc_roc", "f1_score", "precision"],
        var_name="metric",
        value_name="score",
    ).dropna(subset=["score"])
    plot_data["metric"] = plot_data["metric"].map(
        {"auc_roc": "AUC-ROC", "f1_score": "F1-score", "precision": "Precision"}
    )

    sns.set_theme(style="whitegrid", context="paper", font_scale=POSTER_FONT_SCALE)
    figure, axis = plt.subplots(figsize=(8, 5.5), constrained_layout=True)
    metric_order = ["AUC-ROC", "F1-score", "Precision"]
    sns.boxplot(
        data=plot_data,
        x="metric",
        y="score",
        order=metric_order,
        color="#B8D8E8",
        width=0.5,
        fliersize=0,
        ax=axis,
    )
    sns.stripplot(
        data=plot_data,
        x="metric",
        y="score",
        order=metric_order,
        color=COLOURBLIND_BLUE,
        jitter=0.08,
        size=6,
        ax=axis,
    )
    axis.set_xlabel("")
    axis.set_ylabel("Held-out patient-fold score", fontweight="bold", fontsize=15)
    axis.set_ylim(0.0, 1.05)
    axis.set_title(
        "Performance Across Patient-Isolated Folds", fontweight="bold", fontsize=17
    )
    axis.tick_params(axis="both", labelsize=13)
    axis.spines[["top", "right"]].set_visible(False)
    return _save_figure(figure, output_directory, "fold_performance", dpi)


def plot_feature_stability(
    feature_stability: pd.DataFrame,
    output_directory: str | Path = DEFAULT_OUTPUT_DIRECTORY,
    dpi: int = POSTER_DPI,
) -> tuple[Path, Path]:
    """Plot mean normalized importance against cross-fold feature stability.

    Args:
        feature_stability: ``feature_stability.csv`` content from the model.
        output_directory: Directory for PNG and PDF output files.
        dpi: Raster output resolution. Must be at least 300.

    Returns:
        Paths to PNG and PDF output files, in that order.
    """
    _require_columns(
        feature_stability,
        {
            "feature",
            "mean_normalized_importance",
            "importance_standard_deviation",
            "stability_rate",
        },
        "feature stability",
    )
    _validate_dpi(dpi)
    plot_data = feature_stability.copy()
    importance_std = plot_data["importance_standard_deviation"].fillna(0.0).to_numpy()
    inverse_std = 1.0 / (importance_std + 1e-9)
    sizes = np.full(len(plot_data), 150.0)
    if len(plot_data) > 1 and np.ptp(inverse_std) > 0:
        sizes = 80.0 + 270.0 * (inverse_std - inverse_std.min()) / np.ptp(inverse_std)

    sns.set_theme(style="whitegrid", context="paper", font_scale=POSTER_FONT_SCALE)
    figure, axis = plt.subplots(figsize=(8, 5.5), constrained_layout=True)
    axis.scatter(
        plot_data["mean_normalized_importance"],
        plot_data["stability_rate"],
        s=sizes,
        color=COLOURBLIND_BLUE,
        alpha=0.82,
        edgecolors="white",
        linewidths=0.8,
    )
    top_features = plot_data.assign(
        _rank=plot_data["mean_normalized_importance"] * plot_data["stability_rate"]
    ).nlargest(5, "_rank")
    for position, row in enumerate(top_features.itertuples(index=False)):
        offset = 8 if position % 2 == 0 else -12
        axis.annotate(
            row.feature,
            (row.mean_normalized_importance, row.stability_rate),
            xytext=(6, offset),
            textcoords="offset points",
            fontsize=11,
        )
    axis.set_xlabel("Mean normalized importance", fontweight="bold", fontsize=15)
    axis.set_ylabel(
        "Stability score (non-zero coefficient rate)", fontweight="bold", fontsize=15
    )
    axis.set_ylim(-0.03, 1.05)
    axis.set_title(
        "Baseline-Score Coefficient Consistency", fontweight="bold", fontsize=17
    )
    axis.tick_params(axis="both", labelsize=13)
    axis.spines[["top", "right"]].set_visible(False)
    return _save_figure(figure, output_directory, "feature_stability", dpi)


def plot_coefficient_heatmap(
    fold_coefficients: pd.DataFrame,
    feature_stability: pd.DataFrame,
    output_directory: str | Path = DEFAULT_OUTPUT_DIRECTORY,
    dpi: int = POSTER_DPI,
) -> tuple[Path, Path]:
    """Create a coefficient-direction heatmap, clustering features when possible.

    Args:
        fold_coefficients: ``fold_coefficients.csv`` content from the model.
        feature_stability: ``feature_stability.csv`` content used for row labels.
        output_directory: Directory for PNG and PDF output files.
        dpi: Raster output resolution. Must be at least 300.

    Returns:
        Paths to PNG and PDF output files, in that order.
    """
    _require_columns(fold_coefficients, {"fold", "feature"}, "fold coefficients")
    _require_columns(
        feature_stability, {"feature", "stability_rate"}, "feature stability"
    )
    _validate_dpi(dpi)
    coefficient_column = _coefficient_column(fold_coefficients)
    matrix = fold_coefficients.pivot_table(
        index="feature", columns="fold", values=coefficient_column, aggfunc="mean"
    ).sort_index(axis="columns")
    if matrix.empty:
        raise ValueError("No coefficient values are available for heatmap plotting.")
    stability = feature_stability.set_index("feature")["stability_rate"]
    matrix.index = [
        f"{feature} (stability={stability.get(feature, float('nan')):.2f})"
        for feature in matrix.index
    ]
    maximum = float(np.nanmax(np.abs(matrix.to_numpy())))
    maximum = maximum if maximum > 0 else 1.0
    sns.set_theme(style="white", context="paper", font_scale=POSTER_FONT_SCALE)

    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)
    png_path = destination / "coefficient_heatmap_clustered.png"
    pdf_path = destination / "coefficient_heatmap_clustered.pdf"
    if matrix.shape[0] >= 2 and matrix.shape[1] >= 2:
        cluster_grid = sns.clustermap(
            matrix,
            cmap="coolwarm",
            center=0,
            vmin=-maximum,
            vmax=maximum,
            linewidths=0.4,
            linecolor="white",
            figsize=(8, max(4.5, 0.5 * len(matrix) + 2.2)),
            cbar_kws={"label": "ElasticNet coefficient"},
        )
        cluster_grid.ax_heatmap.set_xlabel(
            "Validation fold", fontweight="bold", fontsize=15
        )
        cluster_grid.ax_heatmap.set_ylabel(
            "Feature (cross-fold stability)", fontweight="bold", fontsize=15
        )
        cluster_grid.fig.suptitle(
            "ElasticNet Coefficient Direction Across Folds",
            y=1.02,
            fontweight="bold",
            fontsize=17,
        )
        cluster_grid.fig.savefig(
            png_path,
            dpi=dpi,
            bbox_inches="tight",
            pad_inches=0.14,
            facecolor="white",
        )
        cluster_grid.fig.savefig(
            pdf_path,
            dpi=dpi,
            bbox_inches="tight",
            pad_inches=0.14,
            facecolor="white",
        )
        plt.close(cluster_grid.fig)
    else:
        figure, axis = plt.subplots(
            figsize=(8, max(3.5, 0.6 * len(matrix) + 2.2)), constrained_layout=True
        )
        sns.heatmap(
            matrix,
            cmap="coolwarm",
            center=0,
            vmin=-maximum,
            vmax=maximum,
            linewidths=0.4,
            linecolor="white",
            cbar_kws={"label": "ElasticNet coefficient"},
            ax=axis,
        )
        axis.set_xlabel("Validation fold", fontweight="bold", fontsize=15)
        axis.set_ylabel(
            "Feature (cross-fold stability)", fontweight="bold", fontsize=15
        )
        axis.set_title(
            "ElasticNet Coefficient Direction Across Folds",
            fontweight="bold",
            fontsize=17,
        )
        figure.savefig(
            png_path,
            dpi=dpi,
            bbox_inches="tight",
            pad_inches=0.14,
            facecolor="white",
        )
        figure.savefig(
            pdf_path,
            dpi=dpi,
            bbox_inches="tight",
            pad_inches=0.14,
            facecolor="white",
        )
        plt.close(figure)
    return png_path, pdf_path


def _require_columns(data: pd.DataFrame, required: set[str], label: str) -> None:
    """Raise a descriptive error when a report schema is incomplete."""
    missing = sorted(required.difference(data.columns))
    if missing:
        raise MissingColumnError(
            f"{label} does not match the canonical report schema; missing: "
            + ", ".join(missing)
        )


def _coefficient_column(fold_coefficients: pd.DataFrame) -> str:
    """Return the available linear-coefficient field from the model report."""
    for column in ("coefficient", "feature_importance"):
        if column in fold_coefficients.columns:
            return column
    raise MissingColumnError(
        "fold_coefficients.csv must include 'coefficient' or 'feature_importance'."
    )


def _validate_dpi(dpi: int) -> None:
    """Validate publication-resolution output settings."""
    if isinstance(dpi, bool) or not isinstance(dpi, int) or dpi < 300:
        raise ValueError("dpi must be an integer of at least 300.")


def _save_figure(
    figure: plt.Figure,
    output_directory: str | Path,
    stem: str,
    dpi: int,
) -> tuple[Path, Path]:
    """Save a Matplotlib figure as poster-ready PNG and PDF files."""
    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)
    png_path = destination / f"{stem}.png"
    pdf_path = destination / f"{stem}.pdf"
    figure.savefig(
        png_path,
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0.14,
        facecolor="white",
    )
    figure.savefig(
        pdf_path,
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0.14,
        facecolor="white",
    )
    plt.close(figure)
    return png_path, pdf_path


def main() -> None:
    """Create all poster-ready model-result figures from canonical reports."""
    parser = argparse.ArgumentParser(
        description="Create poster figures from fixed-horizon model reports."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=MODELING_RESULTS_DIR,
        help="Directory containing canonical modelling CSV reports.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
        help="Directory for poster-ready PNG and PDF figures.",
    )
    arguments = parser.parse_args()
    fold_metrics, feature_stability, fold_coefficients = load_model_reports(
        arguments.results_dir
    )
    outputs = [
        plot_fold_performance(fold_metrics, arguments.output_dir),
        plot_feature_stability(feature_stability, arguments.output_dir),
        plot_coefficient_heatmap(
            fold_coefficients, feature_stability, arguments.output_dir
        ),
    ]
    for png_path, pdf_path in outputs:
        print(f"Saved {png_path} and {pdf_path}.")


if __name__ == "__main__":
    main()
