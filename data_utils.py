"""Utilities for constructing fixed-horizon PPMI modelling datasets."""

from __future__ import annotations

import pandas as pd
from pandas.api.types import is_datetime64_any_dtype


REQUIRED_FIXED_HORIZON_COLUMNS = {"PATNO", "EVENT_ID", "SCORE", "VISIT_DATE"}


def create_fixed_horizon_dataset(
    df: pd.DataFrame,
    target_horizon_days: int,
    window_tolerance: int,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Build a patient-level baseline-to-fixed-horizon outcome dataset.

    The function identifies each patient's earliest usable ``BL`` visit, then
    selects the usable follow-up observation closest to ``target_horizon_days``
    within the inclusive tolerance window. A patient can contribute at most one
    target observation, preventing repeated visits from appearing as separate
    modelling rows.

    Args:
        df: Long-format clinical data containing ``PATNO``, ``EVENT_ID``,
            ``SCORE``, and ``VISIT_DATE`` columns.
        target_horizon_days: Number of days after baseline to target. Must be
            a positive integer.
        window_tolerance: Number of days on either side of the target horizon
            accepted as a valid follow-up visit. Must be a non-negative integer.

    Returns:
        A tuple containing:

        * A patient-level DataFrame with ``PATNO``, ``Baseline_Score``,
          ``Target_Score``, and ``Delta_Score`` columns.
        * A summary dictionary with baseline, retained, and follow-up drop-off
          counts. ``excluded_missing_follow_up`` counts patients with a usable
          baseline but no usable follow-up observation in the requested window.

    Raises:
        TypeError: If ``df`` is not a pandas DataFrame.
        ValueError: If required columns are absent, horizon parameters are
            invalid, or no usable baseline visits are available.
    """
    _validate_fixed_horizon_inputs(df, target_horizon_days, window_tolerance)

    working = df.copy()
    working["VISIT_DATE"] = pd.to_datetime(working["VISIT_DATE"], errors="coerce")
    assert is_datetime64_any_dtype(working["VISIT_DATE"]), (
        "VISIT_DATE must be converted to a datetime-compatible dtype."
    )
    working["EVENT_ID"] = working["EVENT_ID"].astype("string").str.strip().str.upper()
    working["SCORE"] = pd.to_numeric(working["SCORE"], errors="coerce")

    baseline_rows = working.loc[working["EVENT_ID"].eq("BL") & working["PATNO"].notna()]
    baseline_patient_count = int(baseline_rows["PATNO"].nunique())
    usable_baseline = (
        baseline_rows.dropna(subset=["VISIT_DATE", "SCORE"])
        .sort_values(["PATNO", "VISIT_DATE"], kind="stable")
        .drop_duplicates("PATNO", keep="first")
        [["PATNO", "VISIT_DATE", "SCORE"]]
        .rename(
            columns={
                "VISIT_DATE": "_baseline_date",
                "SCORE": "Baseline_Score",
            }
        )
    )
    if usable_baseline.empty:
        raise ValueError(
            "No usable baseline visits found. Baseline rows require EVENT_ID 'BL' "
            "and non-missing PATNO, VISIT_DATE, and SCORE values."
        )

    follow_up_rows = working.dropna(subset=["PATNO", "VISIT_DATE", "SCORE"])
    aligned = follow_up_rows.merge(
        usable_baseline,
        on="PATNO",
        how="inner",
        validate="many_to_one",
    )
    aligned["_days_from_baseline"] = (
        aligned["VISIT_DATE"] - aligned["_baseline_date"]
    ).dt.days
    lower_bound = target_horizon_days - window_tolerance
    upper_bound = target_horizon_days + window_tolerance
    candidates = aligned.loc[
        aligned["_days_from_baseline"].gt(0)
        & aligned["_days_from_baseline"].between(lower_bound, upper_bound, inclusive="both")
    ].copy()
    candidates["_distance_from_horizon"] = (
        candidates["_days_from_baseline"] - target_horizon_days
    ).abs()

    selected_targets = (
        candidates.sort_values(
            ["PATNO", "_distance_from_horizon", "VISIT_DATE", "EVENT_ID"],
            kind="stable",
        )
        .drop_duplicates("PATNO", keep="first")
        [["PATNO", "Baseline_Score", "SCORE"]]
        .rename(columns={"SCORE": "Target_Score"})
    )
    selected_targets["Delta_Score"] = (
        selected_targets["Target_Score"] - selected_targets["Baseline_Score"]
    )
    result = selected_targets[
        ["PATNO", "Baseline_Score", "Target_Score", "Delta_Score"]
    ].sort_values("PATNO", kind="stable").reset_index(drop=True)

    usable_baseline_count = int(usable_baseline["PATNO"].nunique())
    retained_patient_count = int(result["PATNO"].nunique())
    summary = {
        "baseline_patients": baseline_patient_count,
        "usable_baseline_patients": usable_baseline_count,
        "retained_patients": retained_patient_count,
        "excluded_invalid_baseline": baseline_patient_count - usable_baseline_count,
        "excluded_missing_follow_up": usable_baseline_count - retained_patient_count,
    }
    print(
        "Fixed-horizon dataset: "
        f"{retained_patient_count} retained / {usable_baseline_count} usable baseline patients. "
        "Excluded for missing usable follow-up within "
        f"{target_horizon_days} +/- {window_tolerance} days: "
        f"{summary['excluded_missing_follow_up']}."
    )
    return result, summary


def _validate_fixed_horizon_inputs(
    df: pd.DataFrame,
    target_horizon_days: int,
    window_tolerance: int,
) -> None:
    """Validate the input DataFrame and fixed-horizon parameters."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")
    missing_columns = sorted(REQUIRED_FIXED_HORIZON_COLUMNS.difference(df.columns))
    if missing_columns:
        raise ValueError(
            "Unexpected clinical data structure; missing required columns: "
            + ", ".join(missing_columns)
        )
    if df["PATNO"].isna().all():
        raise ValueError("PATNO is present but contains no patient identifiers.")
    if isinstance(target_horizon_days, bool) or not isinstance(target_horizon_days, int):
        raise ValueError("target_horizon_days must be a positive integer.")
    if target_horizon_days <= 0:
        raise ValueError("target_horizon_days must be a positive integer.")
    if isinstance(window_tolerance, bool) or not isinstance(window_tolerance, int):
        raise ValueError("window_tolerance must be a non-negative integer.")
    if window_tolerance < 0:
        raise ValueError("window_tolerance must be a non-negative integer.")
