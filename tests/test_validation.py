"""Tests for patient-isolated GroupKFold validation."""

import unittest

import pandas as pd

from validation import ValidationFramework


class ValidationFrameworkTests(unittest.TestCase):
    """Verify GroupKFold uses the fixed-horizon patient identifier safely."""

    def setUp(self) -> None:
        self.dataset = pd.DataFrame(
            {
                "PATNO": [101, 102, 103, 104, 105, 106],
                "Baseline_Score": [20, 21, 22, 23, 24, 25],
                "Target_Score": [19, 23, 20, 24, 27, 24],
                "Delta_Score": [-1, 2, -2, 1, 3, -1],
            }
        )

    def test_group_kfold_isolates_patients(self) -> None:
        framework = ValidationFramework(
            self.dataset,
            target="Delta_Score",
            n_splits=3,
        )

        splits = framework.get_splits()

        self.assertEqual(len(splits), 3)
        for train_df, test_df in splits:
            self.assertIn("patient_id", train_df.columns)
            self.assertIn("patient_id", test_df.columns)
            ValidationFramework.verify_no_overlap(train_df, test_df)
            self.assertTrue(set(train_df["PATNO"]).isdisjoint(test_df["PATNO"]))

    def test_overlap_check_raises_for_shared_patient(self) -> None:
        train_df = pd.DataFrame({"patient_id": [1, 2]})
        test_df = pd.DataFrame({"patient_id": [2, 3]})

        with self.assertRaisesRegex(AssertionError, "Patient leakage"):
            ValidationFramework.verify_no_overlap(train_df, test_df)


if __name__ == "__main__":
    unittest.main()
