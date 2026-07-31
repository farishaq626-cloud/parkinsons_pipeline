"""Strict cohort eligibility and machine-readable attrition tracking."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .exceptions import CohortDefinitionError, MissingColumnError


@dataclass(frozen=True, slots=True)
class CohortDefinition:
    """Define diagnostic and baseline eligibility for a Paper 2 cohort.

    Args:
        patient_column: Participant identifier column.
        diagnostic_group_column: Column containing cohort labels.
        eligible_groups: Diagnostic labels considered Parkinson's disease.
        event_column: Visit/event identifier column.
        baseline_event_ids: Visit identifiers eligible to establish baseline.
    """

    patient_column: str = "PATNO"
    diagnostic_group_column: str = "DIAGNOSTIC_GROUP"
    eligible_groups: tuple[str, ...] = ("PD",)
    event_column: str = "EVENT_ID"
    baseline_event_ids: tuple[str, ...] = ("BL",)

    def __post_init__(self) -> None:
        """Normalize labels and reject ambiguous empty contracts."""
        groups = tuple(str(value).strip().upper() for value in self.eligible_groups)
        events = tuple(str(value).strip().upper() for value in self.baseline_event_ids)
        if not groups or any(not value for value in groups):
            raise CohortDefinitionError("eligible_groups must not be empty.")
        if not events or any(not value for value in events):
            raise CohortDefinitionError("baseline_event_ids must not be empty.")
        object.__setattr__(self, "eligible_groups", groups)
        object.__setattr__(self, "baseline_event_ids", events)


@dataclass(slots=True)
class CohortSelectionResult:
    """Hold the eligible longitudinal records and their attrition ledger."""

    data: pd.DataFrame
    flow: pd.DataFrame

    def save_flow(self, path: str | Path) -> Path:
        """Save cohort attrition as a machine-readable CSV file.

        Args:
            path: Output CSV path.

        Returns:
            Resolved output path.
        """
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        self.flow.to_csv(destination, index=False)
        return destination


class CohortBuilder:
    """Apply strict Parkinson's disease and baseline eligibility rules."""

    def __init__(self, definition: CohortDefinition | None = None) -> None:
        """Initialize the cohort builder.

        Args:
            definition: Cohort contract. Defaults to a strict PD-only contract.
        """
        self.definition = definition or CohortDefinition()

    def select(self, df: pd.DataFrame) -> CohortSelectionResult:
        """Filter longitudinal records and track every attrition stage.

        Args:
            df: Canonical long-format clinical table.

        Returns:
            Eligible records and a stage-level cohort-flow table.

        Raises:
            TypeError: If ``df`` is not a DataFrame.
            MissingColumnError: If identity, diagnosis, or event fields are absent.
            CohortDefinitionError: If no eligible PD baseline remains.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Cohort selection requires a pandas DataFrame.")
        contract = self.definition
        required = {
            contract.patient_column,
            contract.diagnostic_group_column,
            contract.event_column,
        }
        missing = sorted(required.difference(df.columns))
        if missing:
            raise MissingColumnError(
                "Cannot apply cohort definition; missing columns: " + ", ".join(missing)
            )

        stages: list[dict[str, int | str]] = []
        current = df.copy()
        self._record_stage(stages, "input", current, "All input records", None)

        previous = current
        current = current.loc[current[contract.patient_column].notna()].copy()
        self._record_stage(
            stages,
            "valid_patient_identifier",
            current,
            "Missing patient identifier",
            previous,
        )

        previous = current
        group = (
            current[contract.diagnostic_group_column]
            .astype("string")
            .str.strip()
            .str.upper()
        )
        current = current.loc[group.isin(contract.eligible_groups)].copy()
        self._record_stage(
            stages,
            "eligible_diagnostic_group",
            current,
            "Control or unrelated diagnostic group",
            previous,
        )

        normalized_event = (
            current[contract.event_column].astype("string").str.strip().str.upper()
        )
        baseline_patients = set(
            current.loc[
                normalized_event.isin(contract.baseline_event_ids),
                contract.patient_column,
            ]
        )
        previous = current
        current = current.loc[
            current[contract.patient_column].isin(baseline_patients)
        ].copy()
        self._record_stage(
            stages,
            "baseline_eligible",
            current,
            "No eligible baseline event",
            previous,
        )

        if current.empty or not baseline_patients:
            raise CohortDefinitionError(
                "No Parkinson's disease participants with an eligible baseline "
                "remained after cohort filtering."
            )
        return CohortSelectionResult(
            data=current.reset_index(drop=True),
            flow=pd.DataFrame(stages),
        )

    def _record_stage(
        self,
        stages: list[dict[str, int | str]],
        stage: str,
        current: pd.DataFrame,
        exclusion_reason: str,
        previous: pd.DataFrame | None,
    ) -> None:
        """Append one stage to the cohort-flow ledger."""
        patient_column = self.definition.patient_column
        records = len(current)
        patients = int(current[patient_column].nunique(dropna=True))
        previous_records = records if previous is None else len(previous)
        previous_patients = (
            patients
            if previous is None
            else int(previous[patient_column].nunique(dropna=True))
        )
        stages.append(
            {
                "stage": stage,
                "records_retained": records,
                "patients_retained": patients,
                "records_excluded_at_stage": previous_records - records,
                "patients_excluded_at_stage": previous_patients - patients,
                "exclusion_reason": exclusion_reason,
            }
        )
