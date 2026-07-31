"""Tests for repeated nested patient-isolated validation."""

import unittest

import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from ppmi_pipeline.paper2_validation import NestedPatientValidationRunner
from ppmi_pipeline.task_spec import OutcomeType


class NestedPatientValidationTests(unittest.TestCase):
    """Verify outer and inner isolation plus auditable OOF persistence."""

    def test_repeated_nested_folds_have_zero_patient_overlap(self) -> None:
        patients = list(range(24))
        labels = [value % 2 for value in patients]
        dataset = pd.DataFrame(
            {
                "PATNO": patients,
                "Baseline_Score": [15 + 4 * label for label in labels],
                "age": [55 + value for value in patients],
                "progression_label": labels,
            }
        )
        result = NestedPatientValidationRunner(
            outcome_type=OutcomeType.BINARY,
            target_column="progression_label",
            feature_columns=["Baseline_Score", "age"],
            model_family="elasticnet",
            parameter_grid={"C": [1.0], "l1_ratio": [0.5]},
            outer_splits=3,
            outer_repeats=2,
            inner_splits=2,
            threshold_selection="fixed",
            random_state=7,
            configuration_id="isolation-test",
        ).run(dataset)

        self.assertEqual(len(result.fold_metrics), 6)
        self.assertEqual(len(result.oof_predictions), 48)
        for fit_id, assignments in result.fold_assignments.groupby("fit_id"):
            train = set(
                assignments.loc[assignments["partition"].eq("train"), "patient_id"]
            )
            test = set(
                assignments.loc[assignments["partition"].eq("test"), "patient_id"]
            )
            self.assertTrue(train.isdisjoint(test), msg=fit_id)
        importance_sums = result.feature_records.groupby("fit_id")[
            "normalized_importance"
        ].sum()
        self.assertTrue(importance_sums.between(0.999, 1.001).all())

    def test_continuous_nested_runner_reports_error_and_residual_metrics(self) -> None:
        patients = list(range(18))
        dataset = pd.DataFrame(
            {
                "PATNO": patients,
                "Baseline_Score": [10.0 + patient / 2 for patient in patients],
                "age": [50.0 + patient for patient in patients],
                "Delta_Score": [
                    (-1.0) ** patient + patient / 10 for patient in patients
                ],
            }
        )
        result = NestedPatientValidationRunner(
            outcome_type=OutcomeType.CONTINUOUS,
            target_column="Delta_Score",
            feature_columns=["Baseline_Score", "age"],
            model_family="elasticnet",
            parameter_grid={"alpha": [0.01], "l1_ratio": [0.5]},
            outer_splits=3,
            outer_repeats=2,
            inner_splits=2,
            random_state=5,
            configuration_id="continuous-test",
        ).run(dataset)

        self.assertEqual(len(result.oof_predictions), 36)
        self.assertTrue(
            {
                "mae",
                "rmse",
                "r2",
                "residual_mean",
                "residual_standard_deviation",
            }.issubset(result.fold_metrics.columns)
        )
        self.assertTrue(result.oof_predictions["y_probability"].isna().all())

    def test_random_forest_regression_model_is_constructed(self) -> None:
        runner = NestedPatientValidationRunner(
            outcome_type=OutcomeType.CONTINUOUS,
            target_column="Delta_Score",
            feature_columns=["Baseline_Score"],
            model_family="random_forest",
            parameter_grid={"n_estimators": [10]},
            outer_splits=2,
            outer_repeats=2,
            inner_splits=2,
            random_state=5,
            configuration_id="random-forest-regression-test",
        )

        model = runner._build_model(random_state=5)

        self.assertIsInstance(model, RandomForestRegressor)
        self.assertEqual(model.n_jobs, 1)


if __name__ == "__main__":
    unittest.main()
