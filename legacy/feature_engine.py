"""Longitudinal, baseline-delta feature construction for PPMI."""

from __future__ import annotations

import logging
from collections.abc import Sequence

import pandas as pd

from exceptions import MissingColumnError


LOGGER = logging.getLogger("ppmi_pipeline.feature_engine")


class PPMIFeatureEngine:
    """Construct leakage-aware baseline and endpoint-delta modelling data."""

    def __init__(
        self,
        patient_col: str = "PATNO",
        visit_col: str = "EVENT_ID",
        date_col: str = "visit_date",
        baseline_events: Sequence[str] = ("BL", "V01"),
    ) -> None:
        """Initialise feature-engineering column and baseline definitions.

        Args:
            patient_col: Unique patient identifier column.
            visit_col: Visit/event identifier column.
            date_col: Chronological clinical-visit date column.
            baseline_events: Event IDs accepted as baseline observations.

        Returns:
            None.
        """
        self.patient_col = patient_col
        self.visit_col = visit_col
        self.date_col = date_col
        self.baseline_events = tuple(baseline_events)

    def build_endpoint_dataset(
        self,
        df: pd.DataFrame,
        endpoint: str,
        baseline_features: Sequence[str],
    ) -> tuple[pd.DataFrame, list[str], str]:
        """Build post-baseline endpoint-delta rows and leakage-safe features.

        Args:
            df: Normalised long-format PPMI clinical data.
            endpoint: Clinical endpoint for which to construct a delta target.
            baseline_features: Predictors retained from each patient's baseline.

        Returns:
            Aligned endpoint data, feature names, and endpoint-delta target name.

        Raises:
            MissingColumnError: If required columns are absent.
            ValueError: If usable baseline or post-baseline data is unavailable.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Endpoint feature construction requires a pandas DataFrame.")
        required = {
            self.patient_col,
            self.visit_col,
            self.date_col,
            endpoint,
            *baseline_features,
        }
        missing = sorted(required.difference(df.columns))
        if missing:
            raise MissingColumnError(
                f"Cannot build {endpoint} model; missing columns: {', '.join(missing)}"
            )
        ordered = df.sort_values([self.patient_col, self.date_col, self.visit_col]).copy()
        baseline = ordered[ordered[self.visit_col].isin(self.baseline_events)].copy()
        baseline = baseline.dropna(subset=[endpoint]).drop_duplicates(
            self.patient_col, keep="first"
        )
        if baseline.empty:
            raise ValueError(
                f"No usable baseline rows found; expected EVENT_ID in {self.baseline_events}."
            )
        baseline_columns = [self.patient_col, self.date_col, endpoint, *baseline_features]
        baseline = baseline[baseline_columns].rename(
            columns={
                self.date_col: "baseline_date",
                endpoint: f"{endpoint}_baseline",
                **{
                    feature: f"{feature}_baseline"
                    for feature in baseline_features
                },
            }
        )
        aligned = ordered.merge(
            baseline, on=self.patient_col, how="inner", validate="many_to_one"
        )
        aligned["time_from_baseline_days"] = (
            aligned[self.date_col] - aligned["baseline_date"]
        ).dt.days
        target_column = f"{endpoint}_delta"
        aligned[target_column] = aligned[endpoint] - aligned[f"{endpoint}_baseline"]
        aligned = aligned.loc[
            aligned["time_from_baseline_days"].gt(0)
            & aligned[target_column].notna()
        ].copy()
        if aligned.empty:
            raise ValueError(
                f"No post-baseline {endpoint} observations are available for modelling."
            )
        feature_columns = [
            f"{endpoint}_baseline",
            *[f"{feature}_baseline" for feature in baseline_features],
            "time_from_baseline_days",
        ]
        LOGGER.info(
            "Built %s endpoint dataset with %d observations across %d patients",
            endpoint,
            len(aligned),
            aligned[self.patient_col].nunique(),
        )
        return aligned, feature_columns, target_column
