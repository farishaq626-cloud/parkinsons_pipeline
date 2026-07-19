"""PPMI-specific data loading, pre-load schema checks, and normalisation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import pandas as pd

from exceptions import DataFileNotFoundError, MissingColumnError


LOGGER = logging.getLogger("ppmi_pipeline.etl")
PPMI_REQUIRED_COLUMNS = {"PATNO", "EVENT_ID", "visit_date", "moca", "updrs3_score"}


class PPMIDataLoader:
    """Load curated PPMI exports after validating their canonical schema."""

    def __init__(self, required_columns: Iterable[str] = PPMI_REQUIRED_COLUMNS) -> None:
        """Initialise a schema-aware PPMI input loader.

        Args:
            required_columns: Canonical columns required in every input export.

        Returns:
            None.
        """
        self.required_columns = set(required_columns)

    def validate_schema(self, df: pd.DataFrame) -> None:
        """Validate the required canonical columns in a PPMI table.

        Args:
            df: Loaded PPMI table to validate.

        Returns:
            None.

        Raises:
            TypeError: If ``df`` is not a pandas DataFrame.
            MissingColumnError: If required canonical columns are absent.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("PPMI schema validation requires a pandas DataFrame.")
        missing = sorted(self.required_columns.difference(df.columns))
        if missing:
            raise MissingColumnError(
                "Unsupported PPMI input: required columns are missing: "
                + ", ".join(missing)
            )

    def validate_file_schema(
        self,
        file_path: str | Path,
        sheet_name: str | int = 0,
    ) -> None:
        """Check mandatory PPMI headers before loading the full dataset.

        Args:
            file_path: CSV or Excel export path.
            sheet_name: Excel worksheet name or position.

        Returns:
            None.

        Raises:
            DataFileNotFoundError: If ``file_path`` does not exist.
            ValueError: If the input extension is unsupported.
            MissingColumnError: If mandatory PPMI columns are absent.
        """
        path = Path(file_path)
        if not path.exists():
            raise DataFileNotFoundError(f"PPMI input file was not found: {path}")
        if path.suffix.lower() in {".xlsx", ".xls"}:
            headers = pd.read_excel(path, sheet_name=sheet_name, nrows=0).columns
        elif path.suffix.lower() == ".csv":
            headers = pd.read_csv(path, nrows=0).columns
        else:
            raise ValueError("PPMI input must be an .xlsx, .xls, or .csv file.")
        self.validate_schema(pd.DataFrame(columns=headers))

    def load(self, file_path: str | Path, sheet_name: str | int = 0) -> pd.DataFrame:
        """Load, validate, and normalise a PPMI Excel or CSV export.

        Args:
            file_path: CSV or Excel export path.
            sheet_name: Excel worksheet name or position.

        Returns:
            A normalised, chronologically sorted PPMI DataFrame.

        Raises:
            DataFileNotFoundError: If the input file is absent.
            MissingColumnError: If required PPMI fields are absent.
            ValueError: If the file type or content is invalid.
        """
        self.validate_file_schema(file_path, sheet_name=sheet_name)
        path = Path(file_path)
        if path.suffix.lower() in {".xlsx", ".xls"}:
            df = pd.read_excel(path, sheet_name=sheet_name)
        else:
            df = pd.read_csv(path)
        self.validate_schema(df)
        LOGGER.info("Loaded %d PPMI rows from %s", len(df), path)
        return self._normalise(df)

    @staticmethod
    def _normalise(df: pd.DataFrame) -> pd.DataFrame:
        """Coerce PPMI fields to analysis-ready types and sort visits.

        Args:
            df: Validated raw PPMI DataFrame.

        Returns:
            Normalised PPMI data with invalid identity/date rows removed.

        Raises:
            MissingColumnError: If fields required for normalisation are absent.
        """
        required = {"PATNO", "EVENT_ID", "visit_date"}
        missing = sorted(required.difference(df.columns))
        if missing:
            raise MissingColumnError(
                "Cannot normalise PPMI data; missing columns: " + ", ".join(missing)
            )
        clean = df.copy()
        clean["PATNO"] = pd.to_numeric(clean["PATNO"], errors="coerce")
        clean["EVENT_ID"] = clean["EVENT_ID"].astype("string").str.strip()
        clean["visit_date"] = pd.to_datetime(
            clean["visit_date"], format="mixed", errors="coerce"
        )
        for column in (
            "moca",
            "updrs3_score",
            "age",
            "SEX",
            "EDUCYRS",
            "duration",
            "upsit",
        ):
            if column in clean.columns:
                clean[column] = pd.to_numeric(clean[column], errors="coerce")
        clean = clean.dropna(subset=["PATNO", "EVENT_ID", "visit_date"])
        clean["PATNO"] = clean["PATNO"].astype("int64")
        return clean.sort_values(["PATNO", "visit_date", "EVENT_ID"]).reset_index(
            drop=True
        )
