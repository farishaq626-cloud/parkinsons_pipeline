"""End-to-end test for Paper 2 orchestration and provenance artifacts."""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from ppmi_pipeline.paper2_config import PAPER2_FEATURE_BLOCKS, ResolvedPaper2Config
from ppmi_pipeline.paper2_experiment import Paper2ExperimentRunner


class Paper2ExperimentTests(unittest.TestCase):
    """Verify one compact YAML-equivalent experiment persists all evidence."""

    def test_runner_persists_predictions_assignments_and_provenance(self) -> None:
        rows = []
        for patient in range(16):
            label = patient % 2
            baseline_score = 15 + label * 4
            for event, date, score in (
                ("BL", "2020-01-01", baseline_score),
                ("V01", "2020-12-31", baseline_score + (6 if label else 1)),
            ):
                rows.append(
                    {
                        "subject": patient,
                        "visit": event,
                        "date": date,
                        "group": "PD",
                        "raw_score": score,
                        "medication": "OFF",
                        "age": 55 + patient,
                    }
                )
        data = pd.DataFrame(rows)
        blocks = {name: [] for name in PAPER2_FEATURE_BLOCKS}
        blocks["demographics_disease_history"] = ["age"]

        with tempfile.TemporaryDirectory() as directory:
            values = {
                "study": {
                    "name": "paper2_test",
                    "database_version": "synthetic-fixture-v1",
                    "random_seed": 11,
                },
                "data": {
                    "path": None,
                    "sheet_name": 0,
                    "column_map": {
                        "PATNO": "subject",
                        "EVENT_ID": "visit",
                        "VISIT_DATE": "date",
                        "DIAGNOSTIC_GROUP": "group",
                        "SCORE": "raw_score",
                        "MEDICATION_STATE": "medication",
                    },
                },
                "cohort": {
                    "eligible_groups": ["PD"],
                    "baseline_event_ids": ["BL"],
                },
                "endpoint": {
                    "name": "mds_updrs_iii",
                    "score_column": "SCORE",
                    "required_medication_state": "OFF",
                    "medication_state_column": "MEDICATION_STATE",
                    "minimum_score": 0,
                    "maximum_score": 132,
                },
                "task_grid": {
                    "horizons_months": [12],
                    "tolerance_days": [5],
                    "outcome_types": ["binary"],
                    "progression_threshold": 5.0,
                    "feature_sets": [["demographics_disease_history"]],
                },
                "feature_blocks": blocks,
                "validation": {
                    "split_strategy": "patient_isolated",
                    "methodological_negative_control": False,
                    "outer_splits": 2,
                    "outer_repeats": 2,
                    "inner_splits": 2,
                    "imputation_strategies": ["median"],
                    "threshold_selection": "fixed",
                    "fixed_threshold": 0.5,
                    "bootstrap_resamples": 10,
                    "confidence_level": 0.95,
                },
                "models": {
                    "elasticnet": {
                        "binary_parameter_grid": {"C": [1.0], "l1_ratio": [0.5]}
                    }
                },
                "output": {"directory": directory},
            }
            config = ResolvedPaper2Config(
                path=Path(__file__).resolve(),
                values=values,
                sha256="a" * 64,
            )
            result = Paper2ExperimentRunner(config).run(data)

            self.assertEqual(
                result.provenance["database_version"], "synthetic-fixture-v1"
            )
            self.assertEqual(len(result.oof_predictions), 32)
            expected = {
                "provenance.json",
                "resolved_config.yaml",
                "cohort_flow.csv",
                "fold_metrics.csv",
                "metric_confidence_intervals.csv",
                "oof_predictions.csv",
                "fold_assignments.csv",
                "feature_records.csv",
                "feature_stability.csv",
                "feature_manifest.csv",
                "specification_manifest.csv",
                "tuning_results.csv",
            }
            self.assertTrue(
                expected.issubset(
                    {path.name for path in result.run_directory.iterdir()}
                )
            )


if __name__ == "__main__":
    unittest.main()
