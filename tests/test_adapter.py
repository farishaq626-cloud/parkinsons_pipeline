"""Tests for the ETL-to-fixed-horizon schema adapter."""

import unittest

import pandas as pd

from adapter import harmonize_schema
from exceptions import MissingColumnError


class SchemaAdapterTests(unittest.TestCase):
    """Verify that normalised PPMI fields map to the fixed-horizon contract."""

    def test_harmonize_schema_maps_selected_score_and_date(self) -> None:
        """Map canonical ETL fields to SCORE and VISIT_DATE without mutation."""
        source = pd.DataFrame(
            {
                "PATNO": [101],
                "EVENT_ID": ["BL"],
                "updrs3_score": [12],
                "visit_date": ["2020-01-01"],
                "moca": [28],
            }
        )

        result = harmonize_schema(source)

        self.assertEqual(
            list(result.columns), ["PATNO", "EVENT_ID", "SCORE", "VISIT_DATE"]
        )
        self.assertEqual(result.loc[0, "SCORE"], 12)
        self.assertEqual(result.loc[0, "VISIT_DATE"], "2020-01-01")
        self.assertIn("updrs3_score", source.columns)

    def test_harmonize_schema_rejects_missing_score_column(self) -> None:
        """Raise a descriptive error when the configured clinical score is absent."""
        source = pd.DataFrame(
            {
                "PATNO": [101],
                "EVENT_ID": ["BL"],
                "visit_date": ["2020-01-01"],
            }
        )

        with self.assertRaises(MissingColumnError):
            harmonize_schema(source)
