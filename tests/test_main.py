"""Tests for canonical fixed-horizon pipeline target construction."""

import unittest

import pandas as pd

from exceptions import ConfigurationError
from main import create_progression_label, load_config


class MainPipelineTests(unittest.TestCase):
    """Verify canonical configuration and progression-label safeguards."""

    def test_progression_labels_follow_declared_threshold(self) -> None:
        """Create binary labels from fixed-horizon motor-score change."""
        dataset = pd.DataFrame({"Delta_Score": [2.0, 5.0, 8.0]})

        labelled = create_progression_label(
            dataset,
            threshold=5.0,
            target_column="progression_label",
        )

        self.assertEqual(labelled["progression_label"].tolist(), [0, 1, 1])

    def test_progression_label_rejects_single_class_cohort(self) -> None:
        """Reject a threshold that makes binary model fitting impossible."""
        dataset = pd.DataFrame({"Delta_Score": [1.0, 2.0, 3.0]})

        with self.assertRaises(ConfigurationError):
            create_progression_label(
                dataset,
                threshold=10.0,
                target_column="progression_label",
            )

    def test_load_config_returns_fixed_horizon_settings(self) -> None:
        """Load the canonical configuration without JSON experiment overrides."""
        config = load_config()

        self.assertIn("target_horizon_days", config)
        self.assertIn("progression_threshold", config)
        self.assertIn("logistic_regression", config)
