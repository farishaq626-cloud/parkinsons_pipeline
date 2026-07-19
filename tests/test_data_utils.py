"""Tests for fixed-horizon longitudinal dataset construction."""

import unittest

import pandas as pd

from data_utils import create_fixed_horizon_dataset


class FixedHorizonDatasetTests(unittest.TestCase):
    """Verify alignment, target selection, and follow-up drop-off reporting."""

    def test_selects_closest_follow_up_and_tracks_missing_patients(self) -> None:
        df = pd.DataFrame(
            {
                "PATNO": [1, 1, 1, 2, 2, 3],
                "EVENT_ID": ["BL", "V03", "V04", "BL", "V03", "BL"],
                "SCORE": [10, 14, 15, 20, 25, 30],
                "VISIT_DATE": [
                    "2020-01-01",
                    "2020-12-20",
                    "2021-01-20",
                    "2020-01-01",
                    "2021-06-01",
                    "2020-01-01",
                ],
            }
        )

        result, summary = create_fixed_horizon_dataset(
            df,
            target_horizon_days=365,
            window_tolerance=30,
        )

        expected = pd.DataFrame(
            {
                "PATNO": [1],
                "Baseline_Score": [10],
                "Target_Score": [14],
                "Delta_Score": [4],
            }
        )
        pd.testing.assert_frame_equal(result, expected)
        self.assertEqual(summary["baseline_patients"], 3)
        self.assertEqual(summary["retained_patients"], 1)
        self.assertEqual(summary["excluded_missing_follow_up"], 2)

    def test_rejects_unexpected_data_structure(self) -> None:
        df = pd.DataFrame({"PATNO": [1], "EVENT_ID": ["BL"]})

        with self.assertRaisesRegex(ValueError, "missing required columns"):
            create_fixed_horizon_dataset(df, target_horizon_days=365, window_tolerance=90)


if __name__ == "__main__":
    unittest.main()
