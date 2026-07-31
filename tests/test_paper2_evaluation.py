"""Tests for TRIPOD+AI metrics and patient-level uncertainty."""

import unittest

import pandas as pd

from ppmi_pipeline.evaluation import (
    classification_metrics,
    patient_bootstrap_confidence_intervals,
    regression_metrics,
    summarize_feature_stability,
)
from ppmi_pipeline.task_spec import OutcomeType


class Paper2EvaluationTests(unittest.TestCase):
    """Verify discrimination, calibration, error, CI, and stability outputs."""

    def test_metric_families_and_patient_bootstrap_are_reported(self) -> None:
        regression = regression_metrics(
            pd.Series([1.0, 2.0, 3.0]),
            pd.Series([1.0, 2.5, 2.5]),
        )
        self.assertEqual(
            set(regression),
            {"mae", "rmse", "r2", "residual_mean", "residual_standard_deviation"},
        )
        classification = classification_metrics(
            pd.Series([0, 0, 1, 1]),
            pd.Series([0.1, 0.3, 0.7, 0.9]),
            pd.Series([0, 0, 1, 1]),
        )
        self.assertTrue(
            {
                "auroc",
                "auprc",
                "brier_score",
                "calibration_intercept",
                "calibration_slope",
            }.issubset(classification)
        )

        predictions = pd.DataFrame(
            {
                "patient_id": list(range(8)),
                "y_true": [0, 1] * 4,
                "y_probability": [0.1, 0.8, 0.2, 0.7, 0.3, 0.9, 0.4, 0.6],
                "y_pred": [0, 1] * 4,
            }
        )
        intervals = patient_bootstrap_confidence_intervals(
            predictions,
            OutcomeType.BINARY,
            n_resamples=20,
            random_state=3,
        )
        self.assertIn("auroc", set(intervals["metric"]))
        self.assertTrue(intervals["valid_resamples"].gt(0).all())

    def test_stability_reports_sign_inclusion_and_rank(self) -> None:
        records = pd.DataFrame(
            {
                "fit_id": ["a", "b", "a", "b"],
                "feature": ["x", "x", "y", "y"],
                "importance": [1.0, 2.0, 0.0, 1.0],
                "normalized_importance": [1.0, 0.67, 0.0, 0.33],
                "signed_value": [1.0, 2.0, 0.0, -1.0],
                "rank": [1.0, 1.0, 2.0, 2.0],
            }
        )
        stability = summarize_feature_stability(records).set_index("feature")

        self.assertEqual(stability.loc["x", "sign_consistency"], 1.0)
        self.assertEqual(stability.loc["y", "inclusion_frequency"], 0.5)
        self.assertEqual(stability.loc["x", "rank_stability"], 1.0)


if __name__ == "__main__":
    unittest.main()
