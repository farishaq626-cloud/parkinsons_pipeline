"""Unit tests for PPMI schema validation and baseline-delta construction."""

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from etl import PPMIDataLoader  # noqa: E402
from feature_engine import PPMIFeatureEngine  # noqa: E402


def ppmi_records():
    return pd.DataFrame(
        {
            "PATNO": [101, 101, 202, 202],
            "EVENT_ID": ["BL", "V02", "BL", "V02"],
            "visit_date": ["2020-01-01", "2021-01-01", "2020-02-01", "2021-02-01"],
            "moca": [28, 26, 27, 28],
            "updrs3_score": [12, 16, 15, 14],
            "age": [60, 61, 65, 66],
            "SEX": [1, 1, 2, 2],
            "EDUCYRS": [16, 16, 14, 14],
            "duration": [1.0, 2.0, 3.0, 4.0],
            "upsit": [30, 29, 25, 24],
        }
    )


class TestPPMIPipeline(unittest.TestCase):
    def test_schema_validation_accepts_required_columns(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "ppmi.csv"
            ppmi_records().to_csv(path, index=False)
            PPMIDataLoader().validate_schema(path)

    def test_schema_validation_reports_missing_endpoint(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "invalid_ppmi.csv"
            ppmi_records().drop(columns="moca").to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "moca"):
                PPMIDataLoader().validate_schema(path)

    def test_baseline_delta_uses_only_post_baseline_visits(self):
        frame = ppmi_records()
        frame["visit_date"] = pd.to_datetime(frame["visit_date"])
        dataset, features, target = PPMIFeatureEngine().build_endpoint_dataset(
            frame, "moca", ["age", "SEX", "EDUCYRS", "duration", "upsit"]
        )
        self.assertEqual(len(dataset), 2)
        self.assertEqual(target, "moca_delta")
        self.assertIn("moca_baseline", features)
        self.assertTrue((dataset["time_from_baseline_days"] > 0).all())
        self.assertEqual(dict(zip(dataset["PATNO"], dataset[target])), {101: -2, 202: 1})


if __name__ == "__main__":
    unittest.main()
