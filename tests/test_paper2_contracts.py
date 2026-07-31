"""Tests for Paper 2 cohort, phenotype, task, and YAML contracts."""

import copy
import os
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from ppmi_pipeline.cohort import CohortBuilder
from ppmi_pipeline.exceptions import ConfigurationError, PhenotypeDefinitionError
from ppmi_pipeline.paper2_config import expand_experiment_grid, load_paper2_config
from ppmi_pipeline.phenotypes import (
    EndpointDefinition,
    MedicationState,
    PhenotypeBuilder,
)
from ppmi_pipeline.task_spec import OutcomeType, TaskSpecification


class Paper2ContractTests(unittest.TestCase):
    """Verify strict eligibility, medication state, and configuration rules."""

    def test_cohort_is_pd_only_and_tracks_baseline_exclusions(self) -> None:
        data = pd.DataFrame(
            {
                "PATNO": [1, 1, 2, 2, 3],
                "DIAGNOSTIC_GROUP": ["PD", "PD", "HC", "HC", "PD"],
                "EVENT_ID": ["BL", "V01", "BL", "V01", "V01"],
            }
        )

        result = CohortBuilder().select(data)

        self.assertEqual(set(result.data["PATNO"]), {1})
        baseline_stage = result.flow.loc[
            result.flow["stage"].eq("baseline_eligible")
        ].iloc[0]
        self.assertEqual(int(baseline_stage["patients_excluded_at_stage"]), 1)

    def test_updrs_phenotype_requires_one_explicit_medication_state(self) -> None:
        data = pd.DataFrame(
            {
                "PATNO": [1, 1, 2],
                "SCORE": [20, 22, 30],
                "MEDICATION_STATE": ["OFF", "ON", "OFF medication"],
            }
        )
        result = PhenotypeBuilder(
            EndpointDefinition(
                name="mds_updrs_iii",
                score_column="SCORE",
                required_medication_state=MedicationState.OFF,
                minimum_score=0,
                maximum_score=132,
            )
        ).apply(data)

        self.assertEqual(set(result.data["MEDICATION_STATE"]), {"OFF"})
        self.assertEqual(len(result.data), 2)

        with self.assertRaisesRegex(PhenotypeDefinitionError, "explicitly require"):
            EndpointDefinition(
                name="MDS-UPDRS III",
                score_column="SCORE",
                required_medication_state=MedicationState.NOT_APPLICABLE,
            )

    def test_task_contract_encodes_fixed_horizon_and_binary_threshold(self) -> None:
        task = TaskSpecification(
            name="updrs_24m_binary",
            endpoint="mds_updrs_iii",
            outcome_type=OutcomeType.BINARY,
            horizon_months=24,
            tolerance_days=90,
            progression_threshold=5.0,
            medication_state="OFF",
        )

        self.assertEqual(task.horizon_days, 730)
        self.assertEqual(task.target_column, "progression_label")
        self.assertEqual(len(task.fingerprint()), 64)

    def test_yaml_grid_is_deterministic_and_forbids_row_splitting(self) -> None:
        source_directory = Path(__file__).parents[1] / "configs" / "paper2"
        path = source_directory / "primary_analysis.yaml"
        environment = {
            "PPMI_DATA_PATH": str(source_directory / "restricted-input.xlsx"),
            "PPMI_DATABASE_VERSION": "test-snapshot-v1",
            "PPMI_SHEET_NAME": "DATA",
        }
        with patch.dict(os.environ, environment, clear=False):
            first = load_paper2_config(path)
            second = load_paper2_config(path)
        specifications = expand_experiment_grid(first)

        self.assertEqual(first.sha256, second.sha256)
        self.assertEqual(
            first.values["study"]["database_version"],
            "test-snapshot-v1",
        )
        self.assertEqual(len(specifications), 4)
        binary = next(
            item
            for item in specifications
            if item.task.outcome_type is OutcomeType.BINARY
        )
        continuous = next(
            item
            for item in specifications
            if item.task.outcome_type is OutcomeType.CONTINUOUS
        )
        self.assertIn("C", binary.model_parameters)
        self.assertIn("alpha", continuous.model_parameters)

        unsafe = copy.deepcopy(first.values)
        unsafe["validation"]["split_strategy"] = "row_level"
        with self.assertRaisesRegex(ConfigurationError, "Row-level splitting"):
            expand_experiment_grid(unsafe)

    def test_public_yaml_requires_local_environment_values(self) -> None:
        path = Path(__file__).parents[1] / "configs" / "paper2" / "base.yaml"

        with (
            patch.dict(os.environ, {}, clear=True),
            self.assertRaisesRegex(ConfigurationError, "PPMI_DATABASE_VERSION"),
        ):
            load_paper2_config(path)


if __name__ == "__main__":
    unittest.main()
