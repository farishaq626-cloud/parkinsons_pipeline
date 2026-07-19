"""Tests for publication-quality modelling result visualisations."""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from visualization import ResultVisualizer


class ResultVisualizerTests(unittest.TestCase):
    """Verify report loading and dual-format figure generation."""

    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.results_dir = Path(self.temporary_directory.name)
        pd.DataFrame(
            {
                "feature": ["age", "baseline_moca", "upsit"],
                "mean_normalized_importance": [0.4, 0.35, 0.25],
                "importance_standard_deviation": [0.05, 0.10, 0.02],
                "mean_coefficient": [0.5, -0.3, 0.2],
                "nonzero_folds": [3, 2, 3],
                "stability_rate": [1.0, 0.67, 1.0],
            }
        ).to_csv(self.results_dir / "feature_stability.csv", index=False)
        pd.DataFrame(
            {
                "fold": [1, 1, 1, 2, 2, 2, 3, 3, 3],
                "feature": ["age", "baseline_moca", "upsit"] * 3,
                "coefficient": [0.4, -0.2, 0.1, 0.5, -0.3, 0.2, 0.6, -0.4, 0.3],
            }
        ).to_csv(self.results_dir / "fold_coefficients.csv", index=False)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_creates_stability_and_coefficient_figures(self) -> None:
        visualizer = ResultVisualizer(self.results_dir)

        stability_png, stability_pdf = visualizer.plot_stability_importance()
        heatmap_png, heatmap_pdf = visualizer.plot_coef_heatmap()

        for output_path in (stability_png, stability_pdf, heatmap_png, heatmap_pdf):
            self.assertTrue(output_path.exists())
            self.assertGreater(output_path.stat().st_size, 0)

    def test_rejects_missing_modelling_reports(self) -> None:
        with tempfile.TemporaryDirectory() as empty_directory:
            with self.assertRaisesRegex(FileNotFoundError, "Required modelling report"):
                ResultVisualizer(empty_directory)


if __name__ == "__main__":
    unittest.main()
