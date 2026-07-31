"""Utilities for constructing auditable fixed-horizon PPMI datasets."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import pandas as pd
from pandas.api.types import is_datetime64_any_dtype

from .config import DEFAULT_TARGET_HORIZON_DAYS, DEFAULT_WINDOW_TOLERANCE_DAYS
from .exceptions import MissingColumnError
from .paper2_config import PAPER2_FEATURE_BLOCKS
from .task_spec import OutcomeType, TaskSpecification

REQUIRED_FIXED_HORIZON_COLUMNS = {"PATNO", "EVENT_ID", "SCORE", "VISIT_DATE"}
LOGGER = logging.getLogger("ppmi_pipeline.data_utils")


@dataclass(slots=True)
class FixedHorizonFeatureResult:
    """Hold a Paper 2 dataset and its auditable construction artifacts."""

    dataset: pd.DataFrame
    cohort_flow: pd.DataFrame
    feature_manifest: pd.DataFrame
    summary: dict[str, int]


def create_fixed_horizon_dataset(
    df: pd.DataFrame,
    target_horizon_days: int = DEFAULT_TARGET_HORIZON_DAYS,
    window_tolerance: int = DEFAULT_WINDOW_TOLERANCE_DAYS,
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
    working["VISIT_DATE"] = pd.to_datetime(
        working["VISIT_DATE"], format="mixed", errors="coerce"
    )
    if not is_datetime64_any_dtype(working["VISIT_DATE"]):
        raise ValueError("VISIT_DATE must be converted to a datetime-compatible dtype.")
    working["EVENT_ID"] = working["EVENT_ID"].astype("string").str.strip().str.upper()
    working["SCORE"] = pd.to_numeric(working["SCORE"], errors="coerce")

    baseline_rows = working.loc[working["EVENT_ID"].eq("BL") & working["PATNO"].notna()]
    baseline_patient_count = int(baseline_rows["PATNO"].nunique())
    usable_baseline = (
        baseline_rows.dropna(subset=["VISIT_DATE", "SCORE"])
        .sort_values(["PATNO", "VISIT_DATE"], kind="stable")
        .drop_duplicates("PATNO", keep="first")[["PATNO", "VISIT_DATE", "SCORE"]]
        .rename(
            columns={
                "VISIT_DATE": "_baseline_date",
                "SCORE": "Baseline_Score",
            }
        )
    )
    if usable_baseline.empty:
        raise MissingColumnError(
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
        & aligned["_days_from_baseline"].between(
            lower_bound, upper_bound, inclusive="both"
        )
    ].copy()
    candidates["_distance_from_horizon"] = (
        candidates["_days_from_baseline"] - target_horizon_days
    ).abs()

    selected_targets = (
        candidates.sort_values(
            ["PATNO", "_distance_from_horizon", "VISIT_DATE", "EVENT_ID"],
            kind="stable",
        )
        .drop_duplicates("PATNO", keep="first")[["PATNO", "Baseline_Score", "SCORE"]]
        .rename(columns={"SCORE": "Target_Score"})
    )
    selected_targets["Delta_Score"] = (
        selected_targets["Target_Score"] - selected_targets["Baseline_Score"]
    )
    result = (
        selected_targets[["PATNO", "Baseline_Score", "Target_Score", "Delta_Score"]]
        .sort_values("PATNO", kind="stable")
        .reset_index(drop=True)
    )

    usable_baseline_count = int(usable_baseline["PATNO"].nunique())
    retained_patient_count = int(result["PATNO"].nunique())
    summary = {
        "baseline_patients": baseline_patient_count,
        "usable_baseline_patients": usable_baseline_count,
        "retained_patients": retained_patient_count,
        "excluded_invalid_baseline": baseline_patient_count - usable_baseline_count,
        "excluded_missing_follow_up": usable_baseline_count - retained_patient_count,
    }
    LOGGER.info(
        "Fixed-horizon dataset: %d retained / %d usable baseline patients. "
        "Excluded for missing usable follow-up within %d +/- %d days: %d.",
        retained_patient_count,
        usable_baseline_count,
        target_horizon_days,
        window_tolerance,
        summary["excluded_missing_follow_up"],
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
    if isinstance(target_horizon_days, bool) or not isinstance(
        target_horizon_days, int
    ):
        raise ValueError("target_horizon_days must be a positive integer.")
    if target_horizon_days <= 0:
        raise ValueError("target_horizon_days must be a positive integer.")
    if isinstance(window_tolerance, bool) or not isinstance(window_tolerance, int):
        raise ValueError("window_tolerance must be a non-negative integer.")
    if window_tolerance < 0:
        raise ValueError("window_tolerance must be a non-negative integer.")


def create_fixed_horizon_feature_dataset(
    df: pd.DataFrame,
    task: TaskSpecification,
    feature_blocks: Mapping[str, Sequence[str]],
    selected_blocks: Sequence[str] | None = None,
) -> FixedHorizonFeatureResult:
    """Build a leakage-resistant Paper 2 dataset with baseline covariates.

    Every predictor is copied exclusively from the selected baseline row. No
    follow-up or intermediate observation contributes predictor values. The
    follow-up row supplies only the target score and elapsed follow-up time.

    Args:
        df: Canonical long-format data containing ``PATNO``, ``EVENT_ID``,
            ``VISIT_DATE``, and ``SCORE`` plus declared feature columns.
        task: Fixed-horizon task contract.
        feature_blocks: Seven domain-specific baseline feature lists.
        selected_blocks: Optional subset of domains for a modality-ablation run.

    Returns:
        Patient-level dataset, construction flow, feature provenance, and
        attrition summary.

    Raises:
        TypeError: If inputs are not the expected types.
        ValueError: If required fields, feature columns, or usable visits are
            absent.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Paper 2 dataset construction requires a DataFrame.")
    if not isinstance(task, TaskSpecification):
        raise TypeError("task must be a TaskSpecification.")
    blocks = tuple(selected_blocks or PAPER2_FEATURE_BLOCKS)
    unknown_blocks = sorted(set(blocks).difference(feature_blocks))
    if unknown_blocks:
        raise ValueError(
            "Selected Paper 2 feature blocks are undefined: "
            + ", ".join(unknown_blocks)
        )
    feature_columns: list[str] = []
    for block in blocks:
        for column in feature_blocks[block]:
            if column not in feature_columns:
                feature_columns.append(str(column))

    required = {"PATNO", "EVENT_ID", "VISIT_DATE", "SCORE", *feature_columns}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(
            "Cannot construct Paper 2 fixed-horizon dataset; missing columns: "
            + ", ".join(missing)
        )

    working = df.copy()
    working["VISIT_DATE"] = pd.to_datetime(
        working["VISIT_DATE"], format="mixed", errors="coerce"
    )
    working["EVENT_ID"] = working["EVENT_ID"].astype("string").str.strip().str.upper()
    working["SCORE"] = pd.to_numeric(working["SCORE"], errors="coerce")
    baseline_mask = working["EVENT_ID"].isin(task.baseline_event_ids)
    baseline_records = working.loc[baseline_mask & working["PATNO"].notna()].copy()
    baseline_patients = int(baseline_records["PATNO"].nunique())

    baseline_columns = ["PATNO", "VISIT_DATE", "SCORE", *feature_columns]
    usable_baselines = (
        baseline_records.dropna(subset=["PATNO", "VISIT_DATE", "SCORE"])
        .sort_values(["PATNO", "VISIT_DATE"], kind="stable")
        .drop_duplicates("PATNO", keep="first")[baseline_columns]
        .rename(
            columns={
                "VISIT_DATE": "Baseline_Date",
                "SCORE": "Baseline_Score",
            }
        )
    )
    if usable_baselines.empty:
        raise ValueError(
            "No usable baseline rows satisfy the Paper 2 task specification."
        )

    # Follow-up rows intentionally expose no covariates. This makes it
    # structurally impossible for post-baseline feature values to enter the
    # modelling matrix during the merge.
    follow_up = working[["PATNO", "EVENT_ID", "VISIT_DATE", "SCORE"]].dropna(
        subset=["PATNO", "VISIT_DATE", "SCORE"]
    )
    candidates = follow_up.merge(
        usable_baselines,
        on="PATNO",
        how="inner",
        validate="many_to_one",
    )
    candidates["Follow_Up_Days"] = (
        candidates["VISIT_DATE"] - candidates["Baseline_Date"]
    ).dt.days
    lower = task.horizon_days - task.tolerance_days
    upper = task.horizon_days + task.tolerance_days
    candidates = candidates.loc[
        candidates["Follow_Up_Days"].gt(0)
        & candidates["Follow_Up_Days"].between(lower, upper, inclusive="both")
    ].copy()
    candidates["_horizon_distance"] = (
        candidates["Follow_Up_Days"] - task.horizon_days
    ).abs()
    selected = (
        candidates.sort_values(
            ["PATNO", "_horizon_distance", "VISIT_DATE", "EVENT_ID"],
            kind="stable",
        )
        .drop_duplicates("PATNO", keep="first")
        .rename(columns={"SCORE": "Target_Score", "VISIT_DATE": "Target_Date"})
    )
    output_columns = [
        "PATNO",
        "Baseline_Date",
        "Target_Date",
        "Follow_Up_Days",
        "Baseline_Score",
        *feature_columns,
        "Target_Score",
    ]
    result = selected[output_columns].copy()
    result["Delta_Score"] = result["Target_Score"] - result["Baseline_Score"]
    result["task_id"] = task.fingerprint()[:16]
    result["horizon_months"] = task.horizon_months
    result["tolerance_days"] = task.tolerance_days
    if task.outcome_type is OutcomeType.BINARY:
        threshold = float(task.progression_threshold)
        result["progression_label"] = result["Delta_Score"].ge(threshold).astype(int)
    result = result.sort_values("PATNO", kind="stable").reset_index(drop=True)

    usable_baseline_patients = int(usable_baselines["PATNO"].nunique())
    retained_patients = int(result["PATNO"].nunique())
    summary = {
        "baseline_patients": baseline_patients,
        "usable_baseline_patients": usable_baseline_patients,
        "retained_patients": retained_patients,
        "excluded_invalid_baseline": baseline_patients - usable_baseline_patients,
        "excluded_missing_follow_up": usable_baseline_patients - retained_patients,
    }
    flow = pd.DataFrame(
        [
            {
                "stage": "eligible_baseline_event",
                "patients_retained": baseline_patients,
                "patients_excluded_at_stage": 0,
                "exclusion_reason": "Eligible baseline event present",
            },
            {
                "stage": "usable_baseline",
                "patients_retained": usable_baseline_patients,
                "patients_excluded_at_stage": (
                    baseline_patients - usable_baseline_patients
                ),
                "exclusion_reason": "Missing baseline score or date",
            },
            {
                "stage": "fixed_horizon_follow_up",
                "patients_retained": retained_patients,
                "patients_excluded_at_stage": (
                    usable_baseline_patients - retained_patients
                ),
                "exclusion_reason": (
                    f"No eligible follow-up within {task.horizon_days} +/- "
                    f"{task.tolerance_days} days"
                ),
            },
        ]
    )
    manifest_rows = [
        {
            "feature": "Baseline_Score",
            "feature_block": "baseline_motor",
            "temporal_scope": "baseline_only",
            "source": "SCORE",
        }
    ]
    for block in blocks:
        manifest_rows.extend(
            {
                "feature": str(column),
                "feature_block": block,
                "temporal_scope": "baseline_only",
                "source": str(column),
            }
            for column in feature_blocks[block]
        )
    return FixedHorizonFeatureResult(
        dataset=result,
        cohort_flow=flow,
        feature_manifest=pd.DataFrame(manifest_rows).drop_duplicates("feature"),
        summary=summary,
    )
