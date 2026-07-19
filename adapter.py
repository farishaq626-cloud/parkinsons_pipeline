"""Schema adapter connecting normalised PPMI data to fixed-horizon modelling."""

from __future__ import annotations

import pandas as pd

from config import DEFAULT_PROGNOSTIC_SCORE_COLUMN
from exceptions import MissingColumnError


def harmonize_schema(
    df: pd.DataFrame,
    score_column: str = DEFAULT_PROGNOSTIC_SCORE_COLUMN,
) -> pd.DataFrame:
    """Map normalised PPMI fields to the fixed-horizon input schema.

    Args:
        df: Normalised PPMI data from ``PPMIDataLoader.load``.
        score_column: Normalised clinical score column used as the prognostic
            outcome, such as ``updrs3_score``.

    Returns:
        A copy with exactly ``PATNO``, ``EVENT_ID``, ``SCORE``, and
        ``VISIT_DATE`` columns required by ``create_fixed_horizon_dataset``.

    Raises:
        TypeError: If ``df`` is not a pandas DataFrame.
        MissingColumnError: If a required PPMI identity, visit, date, or score
            column is unavailable.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Schema harmonisation requires a pandas DataFrame.")
    source_columns = {"PATNO", "EVENT_ID", "visit_date", score_column}
    missing = sorted(source_columns.difference(df.columns))
    if missing:
        raise MissingColumnError(
            "Cannot harmonize PPMI schema; missing columns: " + ", ".join(missing)
        )
    return (
        df[["PATNO", "EVENT_ID", score_column, "visit_date"]]
        .rename(columns={score_column: "SCORE", "visit_date": "VISIT_DATE"})
        .copy()
    )
