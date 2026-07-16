"""PPMI-specific data loading, pre-load schema checks, and normalisation."""

from pathlib import Path
from typing import Iterable

import pandas as pd


PPMI_REQUIRED_COLUMNS = {
    "PATNO",
    "EVENT_ID",
    "visit_date",
    "moca",
    "updrs3_score",
}


class PPMIDataLoader:
    """Load one curated PPMI export after validating its header schema."""

    def __init__(self, required_columns: Iterable[str] = PPMI_REQUIRED_COLUMNS):
        self.required_columns = set(required_columns)

    def validate_schema(self, df: pd.DataFrame) -> None:
        """Validate the required canonical columns in a PPMI table."""
        missing = sorted(self.required_columns.difference(df.columns))
        if missing:
            raise ValueError(
                "Unsupported PPMI input: required columns are missing: "
                + ", ".join(missing)
            )

    def validate_file_schema(
        self,
        file_path: str | Path,
        sheet_name: str | int = 0,
    ) -> None:
        """Check mandatory PPMI headers before loading the full dataset.

        The curated public export uses ``moca`` and ``updrs3_score`` for the
        MoCA and UPDRS-III endpoints respectively.
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"PPMI input file was not found: {path}")

        if path.suffix.lower() in {".xlsx", ".xls"}:
            headers = pd.read_excel(
                path,
                sheet_name=sheet_name,
                nrows=0,
            ).columns
        elif path.suffix.lower() == ".csv":
            headers = pd.read_csv(path, nrows=0).columns
        else:
            raise ValueError("PPMI input must be an .xlsx, .xls, or .csv file.")

        self.validate_schema(pd.DataFrame(columns=headers))

    def load(
        self,
        file_path: str | Path,
        sheet_name: str | int = 0,
    ) -> pd.DataFrame:
        """Load, validate, and normalise a PPMI Excel or CSV export."""
        self.validate_file_schema(file_path, sheet_name=sheet_name)

        path = Path(file_path)
        if path.suffix.lower() in {".xlsx", ".xls"}:
            df = pd.read_excel(path, sheet_name=sheet_name)
        else:
            df = pd.read_csv(path)

        self.validate_schema(df)
        return self._normalise(df)

    @staticmethod
    def _normalise(df: pd.DataFrame) -> pd.DataFrame:
        """Coerce PPMI fields to analysis-ready types and sort visits."""
        clean = df.copy()

        clean["PATNO"] = pd.to_numeric(clean["PATNO"], errors="coerce")
        clean["EVENT_ID"] = clean["EVENT_ID"].astype("string").str.strip()
        clean["visit_date"] = pd.to_datetime(
            clean["visit_date"],
            format="mixed",
            errors="coerce",
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

        return clean.sort_values(
            ["PATNO", "visit_date", "EVENT_ID"]
        ).reset_index(drop=True)