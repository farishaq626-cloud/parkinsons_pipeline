"""Tests for the interpretable baseline prognostic model."""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from modeling import PrognosticModel


class PrognosticModelTests(unittest.TestCase):
    """Verify patient-isolated model reports and coefficient exports."""

    def setUp(self) -> None:
        patient_ids = list(range(100, 112))
        labels = [0, 1] * 6
        self.dataset = pd.DataFrame(
            {
                "PATNO": patient_ids,
                "Baseline_Score": [20 + label * 5 for label in labels],
                "baseline_age": [55 + index for index in range(len(patient_ids))],
                "Target_Score": [21 + label * 5 for label in labels],
                "Delta_Score": labels,
                "progression_label": labels,
            }
        )

    def test_train_and_evaluate_returns_fold_reports_and_coefficients(self) -> None:
        model = PrognosticModel.train_and_evaluate(
            self.dataset,
            target="progression_label",
            n_splits=3,
            random_state=7,
        )

        self.assertEqual(len(model.fold_metrics_), 3)
        self.assertEqual(set(model.fold_metrics_.columns), {
            "fold", "train_patients", "test_patients", "precision", "recall", "f1_score", "auc_roc"
        })
        self.assertEqual(set(model.fold_coefficients_["fold"]), {1, 2, 3})
        self.assertNotIn("Target_Score", set(model.fold_coefficients_["feature"]))
        self.assertNotIn("Delta_Score", set(model.fold_coefficients_["feature"]))
        self.assertIn("stability_rate", model.feature_stability_.columns)

    def test_save_reports_writes_all_fold_outputs(self) -> None:
        model = PrognosticModel.train_and_evaluate(
            self.dataset,
            target="progression_label",
            n_splits=3,
        )

        with tempfile.TemporaryDirectory() as directory:
            model.save_reports(directory)
            output_directory = Path(directory)
            self.assertTrue((output_directory / "fold_metrics.csv").exists())
            self.assertTrue((output_directory / "fold_coefficients.csv").exists())
            self.assertTrue((output_directory / "feature_stability.csv").exists())


if __name__ == "__main__":
    unittest.main()
