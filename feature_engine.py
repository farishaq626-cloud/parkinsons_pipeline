"""Longitudinal, baseline-delta feature construction for PPMI."""

from collections.abc import Sequence

import pandas as pd


class PPMIFeatureEngine:
    def __init__(self, patient_col="PATNO", visit_col="EVENT_ID", date_col="visit_date", baseline_events=("BL", "V01")):
        self.patient_col = patient_col
        self.visit_col = visit_col
        self.date_col = date_col
        self.baseline_events = tuple(baseline_events)

    def build_endpoint_dataset(self, df: pd.DataFrame, endpoint: str, baseline_features: Sequence[str]):
        """Build post-baseline endpoint-delta rows and leakage-safe features."""
        required = {self.patient_col, self.visit_col, self.date_col, endpoint, *baseline_features}
        missing = sorted(required.difference(df.columns))
        if missing:
            raise ValueError(f"Cannot build {endpoint} model; missing columns: {', '.join(missing)}")
        ordered = df.sort_values([self.patient_col, self.date_col, self.visit_col]).copy()
        baseline = ordered[ordered[self.visit_col].isin(self.baseline_events)].copy()
        baseline = baseline.dropna(subset=[endpoint]).drop_duplicates(self.patient_col, keep="first")
        if baseline.empty:
            raise ValueError(f"No usable baseline rows found; expected EVENT_ID in {self.baseline_events}.")
        baseline_columns = [self.patient_col, self.date_col, endpoint, *baseline_features]
        baseline = baseline[baseline_columns].rename(columns={
            self.date_col: "baseline_date",
            endpoint: f"{endpoint}_baseline",
            **{feature: f"{feature}_baseline" for feature in baseline_features},
        })
        aligned = ordered.merge(baseline, on=self.patient_col, how="inner", validate="many_to_one")
        aligned["time_from_baseline_days"] = (aligned[self.date_col] - aligned["baseline_date"]).dt.days
        target_column = f"{endpoint}_delta"
        aligned[target_column] = aligned[endpoint] - aligned[f"{endpoint}_baseline"]
        aligned = aligned[(aligned["time_from_baseline_days"] > 0) & aligned[target_column].notna()].copy()
        if aligned.empty:
            raise ValueError(f"No post-baseline {endpoint} observations are available for modelling.")
        feature_columns = [f"{endpoint}_baseline", *[f"{feature}_baseline" for feature in baseline_features], "time_from_baseline_days"]
        return aligned, feature_columns, target_column
