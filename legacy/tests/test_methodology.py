"""Tests for patient-grouped evaluation on the synthetic PPMI fixture."""

import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from analytics_engine import PPMIAnalyticsEngine  # noqa: E402
from etl import PPMIDataLoader  # noqa: E402
from generate_dummy_data import generate_dummy_data  # noqa: E402


class TestPatientGroupedSplit(unittest.TestCase):
    def test_train_and_test_patients_do_not_overlap(self):
        fixture_path = PROJECT_ROOT / "tests" / "dummy_ppmi.csv"
        if not fixture_path.exists():
            generate_dummy_data(fixture_path)
        data = PPMIDataLoader().load(fixture_path)
        model_config = {"test_size": 0.2, "random_state": 42}
        with tempfile.TemporaryDirectory() as output_directory:
            engine = PPMIAnalyticsEngine(model_config, output_directory, "test")
            train_idx, test_idx = engine.split_by_patient(
                data[["moca"]], data["updrs3_score"], data["PATNO"]
            )

        train_patients = set(data.iloc[train_idx]["PATNO"])
        test_patients = set(data.iloc[test_idx]["PATNO"])
        self.assertTrue(train_patients)
        self.assertTrue(test_patients)
        self.assertFalse(train_patients.intersection(test_patients))
