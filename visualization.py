"""Publication-quality visualisation of prognostic model stability reports."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import DEFAULT_FIGURE_DPI, MODELING_RESULTS_DIR
from exceptions import DataFileNotFoundError, MissingColumnError

class ResultVisualizer:
    """Load model reports and create publication-quality interpretability plots.

    The class expects the CSV outputs written by
    ``PrognosticModel.save_reports``. The model's native column names are
    converted internally to the plot-facing names ``Importance_Mean``,
    ``Importance_Std``, and ``Stability_Score``.

    Args:
        results_dir: Directory containing ``feature_stability.csv`` and
            ``fold_coefficients.csv``. Defaults to ``results/modeling``.
        dpi: Resolution for raster figure output. Must be at least 300.

    Raises:
        FileNotFoundError: If either required modelling report is absent.
        ValueError: If report columns do not match the modelling output schema
            or if an invalid DPI value is supplied.
    """

    stability_filename = "feature_stability.csv"
    coefficient_filename = "fold_coefficients.csv"

    def __init__(
        self,
        results_dir: str | Path = MODELING_RESULTS_DIR,
        dpi: int = DEFAULT_FIGURE_DPI,
    ) -> None:
        if isinstance(dpi, bool) or not isinstance(dpi, int) or dpi < 300:
            raise ValueError("dpi must be an integer of at least 300.")

        self.results_dir = Path(results_dir)
        self.dpi = dpi
        self.feature_stability = self._load_csv(self.stability_filename)
        self.fold_coefficients = self._load_csv(self.coefficient_filename)
        self._stability_plot_data = self._prepare_stability_plot_data()
        self._validate_coefficient_schema()

    def plot_stability_importance(self) -> tuple[Path, Path]:
        """Plot mean importance against cross-fold feature stability.

        Point size is inversely proportional to importance standard deviation.
        The five features with the strongest joint ranking for stability and
        mean importance are annotated.

        Returns:
            Paths to the saved PNG and PDF figures, in that order.
        """
        data = self._stability_plot_data.copy()
        point_sizes = self._inverse_standard_deviation_sizes(data["Importance_Std"])
        figure, axis = plt.subplots(figsize=(8, 6), constrained_layout=True)
        scatter = axis.scatter(
            data["Importance_Mean"],
            data["Stability_Score"],
            s=point_sizes,
            c="#0072B2",
            alpha=0.8,
            edgecolors="white",
            linewidths=0.8,
        )
        del scatter

        top_features = data.assign(
            _ranking=data["Importance_Mean"] * data["Stability_Score"]
        ).sort_values(
            ["_ranking", "Stability_Score", "Importance_Mean"],
            ascending=False,
            kind="stable",
        ).head(5)
        for position, (_, feature) in enumerate(top_features.iterrows()):
            vertical_offset = 8 if position % 2 == 0 else -12
            axis.annotate(
                feature["feature"],
                (feature["Importance_Mean"], feature["Stability_Score"]),
                xytext=(6, vertical_offset),
                textcoords="offset points",
                fontsize=8,
                color="#1A1A1A",
            )

        axis.set_xlabel("Importance Mean", fontweight="bold")
        axis.set_ylabel("Stability Score", fontweight="bold")
        axis.set_title("Feature Importance and Cross-Fold Stability", fontweight="bold")
        axis.grid(axis="both", color="#D9D9D9", linewidth=0.6, alpha=0.7)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
        return self._save_figure(figure, "stability_importance")

    def plot_coef_heatmap(self) -> tuple[Path, Path]:
        """Plot per-fold coefficients as a diverging publication-ready heatmap.

        Returns:
            Paths to the saved PNG and PDF figures, in that order.
        """
        coefficient_column = self._coefficient_column()
        heatmap_data = self.fold_coefficients.pivot_table(
            index="feature",
            columns="fold",
            values=coefficient_column,
            aggfunc="mean",
        )
        ordered_features = self._stability_plot_data.sort_values(
            ["Stability_Score", "Importance_Mean"],
            ascending=False,
            kind="stable",
        )["feature"]
        heatmap_data = heatmap_data.reindex(
            [feature for feature in ordered_features if feature in heatmap_data.index]
        )
        heatmap_data = heatmap_data.sort_index(axis="columns")
        if heatmap_data.empty:
            raise ValueError("No feature coefficients are available for heatmap plotting.")

        max_abs_value = float(np.nanmax(np.abs(heatmap_data.to_numpy())))
        if max_abs_value == 0 or np.isnan(max_abs_value):
            max_abs_value = 1.0
        figure_height = max(4.5, 0.42 * len(heatmap_data.index) + 1.8)
        figure, axis = plt.subplots(
            figsize=(8, figure_height),
            constrained_layout=True,
        )
        image = axis.imshow(
            heatmap_data.to_numpy(),
            cmap="coolwarm",
            vmin=-max_abs_value,
            vmax=max_abs_value,
            aspect="auto",
        )
        colorbar = figure.colorbar(image, ax=axis, shrink=0.86, pad=0.02)
        colorbar.set_label("Coefficient", fontweight="bold")
        axis.set_xticks(range(len(heatmap_data.columns)), heatmap_data.columns)
        axis.set_yticks(range(len(heatmap_data.index)), heatmap_data.index)
        axis.set_xlabel("Validation Fold", fontweight="bold")
        axis.set_ylabel("Feature", fontweight="bold")
        axis.set_title("Feature Coefficients Across Patient-Isolated Folds", fontweight="bold")
        axis.tick_params(axis="x", bottom=False, top=False)
        axis.tick_params(axis="y", left=False)

        if heatmap_data.shape[0] <= 15 and heatmap_data.shape[1] <= 10:
            for row_index, row in enumerate(heatmap_data.to_numpy()):
                for column_index, value in enumerate(row):
                    if not np.isnan(value):
                        text_colour = "white" if abs(value) > max_abs_value * 0.55 else "black"
                        axis.text(
                            column_index,
                            row_index,
                            f"{value:.2f}",
                            ha="center",
                            va="center",
                            fontsize=7,
                            color=text_colour,
                        )
        return self._save_figure(figure, "coefficient_heatmap")

    def _load_csv(self, filename: str) -> pd.DataFrame:
        """Load a required report CSV with a clear missing-file message."""
        path = self.results_dir / filename
        if not path.exists():
            raise DataFileNotFoundError(
                f"Required modelling report was not found: {path}. "
                "Run PrognosticModel.train_and_evaluate(..., output_dir='results/modeling') "
                "before creating visualisations."
            )
        return pd.read_csv(path)

    def _prepare_stability_plot_data(self) -> pd.DataFrame:
        """Map modelling-report fields to the stability-importance plot schema."""
        source_columns = {
            "feature": "feature",
            "Importance_Mean": "mean_normalized_importance",
            "Importance_Std": "importance_standard_deviation",
            "Stability_Score": "stability_rate",
        }
        missing = [
            source for source in source_columns.values() if source not in self.feature_stability.columns
        ]
        if missing:
            raise MissingColumnError(
                "feature_stability.csv does not match the expected modelling schema; "
                "missing columns: " + ", ".join(missing)
            )

        plot_data = self.feature_stability.rename(
            columns={source: target for target, source in source_columns.items()}
        )[list(source_columns)]
        plot_data["Importance_Mean"] = pd.to_numeric(
            plot_data["Importance_Mean"], errors="coerce"
        )
        plot_data["Importance_Std"] = pd.to_numeric(
            plot_data["Importance_Std"], errors="coerce"
        ).fillna(0.0)
        plot_data["Stability_Score"] = pd.to_numeric(
            plot_data["Stability_Score"], errors="coerce"
        )
        if plot_data[["Importance_Mean", "Stability_Score"]].isna().any().any():
            raise ValueError(
                "feature_stability.csv contains non-numeric importance or stability values."
            )
        if (plot_data["Importance_Std"] < 0).any():
            raise ValueError("Importance standard deviation cannot be negative.")
        return plot_data

    def _validate_coefficient_schema(self) -> None:
        """Validate the fields required to construct a fold-coefficient heatmap."""
        required = {"fold", "feature"}
        missing = sorted(required.difference(self.fold_coefficients.columns))
        if missing:
            raise MissingColumnError(
                "fold_coefficients.csv does not match the expected modelling schema; "
                "missing columns: " + ", ".join(missing)
            )
        self._coefficient_column()

    def _coefficient_column(self) -> str:
        """Return the available linear-coefficient or tree-importance field."""
        for column in ("coefficient", "feature_importance"):
            if column in self.fold_coefficients.columns:
                return column
        raise MissingColumnError(
            "fold_coefficients.csv must contain 'coefficient' or 'feature_importance'."
        )

    @staticmethod
    def _inverse_standard_deviation_sizes(importance_std: pd.Series) -> np.ndarray:
        """Scale inverse standard deviation to readable scatter-point sizes."""
        inverse_std = 1.0 / (importance_std.to_numpy(dtype=float) + 1e-9)
        if np.ptp(inverse_std) == 0:
            return np.full_like(inverse_std, 130.0)
        scaled = (inverse_std - inverse_std.min()) / np.ptp(inverse_std)
        return 70.0 + scaled * 260.0

    def _save_figure(self, figure: plt.Figure, stem: str) -> tuple[Path, Path]:
        """Save one figure in both PNG and PDF formats, then close it."""
        self.results_dir.mkdir(parents=True, exist_ok=True)
        png_path = self.results_dir / f"{stem}.png"
        pdf_path = self.results_dir / f"{stem}.pdf"
        figure.savefig(png_path, dpi=self.dpi, bbox_inches="tight", facecolor="white")
        figure.savefig(pdf_path, dpi=self.dpi, bbox_inches="tight", facecolor="white")
        plt.close(figure)
        return png_path, pdf_path
