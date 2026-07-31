"""Tests for baseline-only Paper 2 feature construction."""

import unittest

import pandas as pd

from ppmi_pipeline.adapter import harmonize_paper2_schema
from ppmi_pipeline.data_utils import create_fixed_horizon_feature_dataset
from ppmi_pipeline.paper2_config import PAPER2_FEATURE_BLOCKS
from ppmi_pipeline.task_spec import OutcomeType, TaskSpecification


def _feature_blocks() -> dict[str, list[str]]:
    """Return a complete seven-domain feature declaration for tests."""
    blocks = {name: [] for name in PAPER2_FEATURE_BLOCKS}
    blocks["demographics_disease_history"] = ["age"]
    blocks["cognition_neuropsychology"] = ["moca"]
    return blocks


class Paper2FeatureTests(unittest.TestCase):
    """Verify adapters and fixed-horizon construction exclude future features."""

    def test_follow_up_covariates_cannot_enter_predictor_matrix(self) -> None:
        data = pd.DataFrame(
            {
                "PATNO": [1, 1, 2, 2],
                "EVENT_ID": ["BL", "V01", "BL", "V01"],
                "VISIT_DATE": ["2020-01-01", "2020-12-31"] * 2,
                "SCORE": [20, 28, 30, 31],
                "age": [60, 99, 70, 101],
                "moca": [27, 5, 24, 4],
            }
        )
        task = TaskSpecification(
            name="test_12m_binary",
            endpoint="mds_updrs_iii",
            outcome_type=OutcomeType.BINARY,
            horizon_months=12,
            tolerance_days=5,
            progression_threshold=5,
            medication_state="OFF",
        )

        result = create_fixed_horizon_feature_dataset(
            data,
            task,
            _feature_blocks(),
        )

        self.assertEqual(result.dataset["age"].tolist(), [60, 70])
        self.assertEqual(result.dataset["moca"].tolist(), [27, 24])
        self.assertEqual(result.dataset["progression_label"].tolist(), [1, 0])
        self.assertTrue(
            result.feature_manifest["temporal_scope"].eq("baseline_only").all()
        )

    def test_adapter_retains_only_declared_core_and_feature_fields(self) -> None:
        source = pd.DataFrame(
            {
                "subject": [1],
                "visit": ["BL"],
                "date": ["2020-01-01"],
                "group": ["PD"],
                "motor": [20],
                "med_state": ["OFF"],
                "age": [60],
                "moca": [27],
                "future_target": [999],
            }
        )
        harmonized = harmonize_paper2_schema(
            source,
            {
                "PATNO": "subject",
                "EVENT_ID": "visit",
                "VISIT_DATE": "date",
                "DIAGNOSTIC_GROUP": "group",
                "SCORE": "motor",
                "MEDICATION_STATE": "med_state",
            },
            _feature_blocks(),
        )

        self.assertNotIn("future_target", harmonized.columns)
        self.assertTrue(
            {"PATNO", "EVENT_ID", "VISIT_DATE", "SCORE", "age", "moca"}.issubset(
                harmonized.columns
            )
        )

    def test_adapter_can_derive_documented_off_medication_state(self) -> None:
        source = pd.DataFrame(
            {
                "PATNO": [1],
                "EVENT_ID": ["BL"],
                "visit_date": ["2020-01-01"],
                "COHORT": [1],
                "updrs3_score": [20],
                "age": [60],
                "moca": [27],
            }
        )
        harmonized = harmonize_paper2_schema(
            source,
            {
                "PATNO": "PATNO",
                "EVENT_ID": "EVENT_ID",
                "VISIT_DATE": "visit_date",
                "DIAGNOSTIC_GROUP": "COHORT",
                "SCORE": "updrs3_score",
                "MEDICATION_STATE": "absent_source_column",
            },
            _feature_blocks(),
            column_constants={"MEDICATION_STATE": "OFF"},
        )

        self.assertEqual(harmonized["MEDICATION_STATE"].tolist(), ["OFF"])
        self.assertNotIn("absent_source_column", harmonized.columns)


if __name__ == "__main__":
    unittest.main()
