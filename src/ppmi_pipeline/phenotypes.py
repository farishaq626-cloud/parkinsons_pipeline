"""Endpoint and medication-state contracts for longitudinal PPMI tasks."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import pandas as pd

from .exceptions import MissingColumnError, PhenotypeDefinitionError


class MedicationState(StrEnum):
    """Canonical medication states supported by the phenotype layer."""

    ON = "ON"
    OFF = "OFF"
    NOT_APPLICABLE = "NOT_APPLICABLE"


MEDICATION_STATE_ALIASES: dict[str, MedicationState] = {
    "ON": MedicationState.ON,
    "ON MEDICATION": MedicationState.ON,
    "ON_MEDICATION": MedicationState.ON,
    "1": MedicationState.ON,
    "OFF": MedicationState.OFF,
    "OFF MEDICATION": MedicationState.OFF,
    "OFF_MEDICATION": MedicationState.OFF,
    "0": MedicationState.OFF,
}


@dataclass(frozen=True, slots=True)
class EndpointDefinition:
    """Define one endpoint and its required medication-state context.

    Args:
        name: Endpoint identifier.
        score_column: Source score field in the canonical long table.
        required_medication_state: Required ON or OFF state. Use
            ``NOT_APPLICABLE`` only for endpoints unaffected by medication state.
        medication_state_column: Column containing medication-state labels.
        minimum_score: Optional inclusive lower score bound.
        maximum_score: Optional inclusive upper score bound.
    """

    name: str
    score_column: str
    required_medication_state: MedicationState
    medication_state_column: str = "MEDICATION_STATE"
    minimum_score: float | None = None
    maximum_score: float | None = None

    def __post_init__(self) -> None:
        """Reject mixed-state MDS-UPDRS III definitions."""
        try:
            state = MedicationState(self.required_medication_state)
        except ValueError as error:
            raise PhenotypeDefinitionError(
                "required_medication_state must be ON, OFF, or NOT_APPLICABLE."
            ) from error
        object.__setattr__(self, "required_medication_state", state)
        if "updrs" in self.name.lower() and state is MedicationState.NOT_APPLICABLE:
            raise PhenotypeDefinitionError(
                "MDS-UPDRS endpoints must explicitly require ON or OFF medication "
                "state; mixed-state scoring is not permitted."
            )
        if (
            self.minimum_score is not None
            and self.maximum_score is not None
            and self.minimum_score >= self.maximum_score
        ):
            raise PhenotypeDefinitionError(
                "minimum_score must be smaller than maximum_score."
            )


@dataclass(slots=True)
class PhenotypeResult:
    """Hold phenotype-compatible records and an exclusion summary."""

    data: pd.DataFrame
    flow: pd.DataFrame


class PhenotypeBuilder:
    """Apply score and medication-state requirements without silent mixing."""

    def __init__(self, definition: EndpointDefinition) -> None:
        """Initialize the phenotype builder.

        Args:
            definition: Endpoint and medication-state contract.
        """
        self.definition = definition

    def apply(self, df: pd.DataFrame) -> PhenotypeResult:
        """Construct a canonical endpoint score under an explicit state.

        Args:
            df: Canonical long-format cohort records.

        Returns:
            Compatible records with ``SCORE`` and normalized
            ``MEDICATION_STATE`` fields plus an exclusion ledger.

        Raises:
            TypeError: If ``df`` is not a DataFrame.
            MissingColumnError: If required endpoint fields are missing.
            PhenotypeDefinitionError: If medication labels are unrecognized or
                no usable records remain.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Phenotype construction requires a pandas DataFrame.")
        definition = self.definition
        required = {definition.score_column}
        if definition.required_medication_state is not MedicationState.NOT_APPLICABLE:
            required.add(definition.medication_state_column)
        missing = sorted(required.difference(df.columns))
        if missing:
            raise MissingColumnError(
                "Cannot construct endpoint phenotype; missing columns: "
                + ", ".join(missing)
            )

        working = df.copy()
        initial_records = len(working)
        patient_column = "PATNO" if "PATNO" in working.columns else None
        initial_patients = (
            int(working[patient_column].nunique(dropna=True))
            if patient_column is not None
            else None
        )
        working["SCORE"] = pd.to_numeric(
            working[definition.score_column], errors="coerce"
        )
        valid_score = working["SCORE"].notna()
        if definition.minimum_score is not None:
            valid_score &= working["SCORE"].ge(definition.minimum_score)
        if definition.maximum_score is not None:
            valid_score &= working["SCORE"].le(definition.maximum_score)
        score_filtered = working.loc[valid_score].copy()

        flow = [
            {
                "stage": "endpoint_input",
                "records_retained": initial_records,
                "patients_retained": initial_patients,
                "records_excluded_at_stage": 0,
                "patients_excluded_at_stage": 0
                if initial_patients is not None
                else None,
                "exclusion_reason": "All cohort-eligible records",
            },
            {
                "stage": "valid_endpoint_score",
                "records_retained": len(score_filtered),
                "patients_retained": (
                    int(score_filtered[patient_column].nunique(dropna=True))
                    if patient_column is not None
                    else None
                ),
                "records_excluded_at_stage": initial_records - len(score_filtered),
                "patients_excluded_at_stage": (
                    initial_patients
                    - int(score_filtered[patient_column].nunique(dropna=True))
                    if patient_column is not None and initial_patients is not None
                    else None
                ),
                "exclusion_reason": "Missing or out-of-range endpoint score",
            },
        ]

        if definition.required_medication_state is MedicationState.NOT_APPLICABLE:
            score_filtered["MEDICATION_STATE"] = MedicationState.NOT_APPLICABLE.value
            final = score_filtered
        else:
            normalized = score_filtered[definition.medication_state_column].map(
                normalize_medication_state
            )
            recognized = normalized.notna()
            if not recognized.all():
                unknown = sorted(
                    {
                        str(value)
                        for value in score_filtered.loc[
                            ~recognized, definition.medication_state_column
                        ].dropna()
                    }
                )
                detail = ", ".join(unknown) if unknown else "missing values"
                raise PhenotypeDefinitionError(
                    "Unrecognized medication-state labels: " + detail
                )
            score_filtered["MEDICATION_STATE"] = normalized.astype("string")
            required_value = definition.required_medication_state.value
            final = score_filtered.loc[
                score_filtered["MEDICATION_STATE"].eq(required_value)
            ].copy()
            flow.append(
                {
                    "stage": f"medication_state_{required_value.lower()}",
                    "records_retained": len(final),
                    "patients_retained": (
                        int(final[patient_column].nunique(dropna=True))
                        if patient_column is not None
                        else None
                    ),
                    "records_excluded_at_stage": len(score_filtered) - len(final),
                    "patients_excluded_at_stage": (
                        int(score_filtered[patient_column].nunique(dropna=True))
                        - int(final[patient_column].nunique(dropna=True))
                        if patient_column is not None
                        else None
                    ),
                    "exclusion_reason": (
                        f"Endpoint requires {required_value} medication state"
                    ),
                }
            )

        if final.empty:
            raise PhenotypeDefinitionError(
                f"No records satisfy endpoint '{definition.name}' and its "
                "medication-state contract."
            )
        return PhenotypeResult(
            data=final.reset_index(drop=True),
            flow=pd.DataFrame(flow),
        )


def normalize_medication_state(value: object) -> str | None:
    """Normalize one medication-state label.

    Args:
        value: Raw medication-state value.

    Returns:
        ``ON`` or ``OFF`` when recognized; otherwise ``None``.
    """
    if pd.isna(value):
        return None
    normalized = str(value).strip().upper()
    state = MEDICATION_STATE_ALIASES.get(normalized)
    return None if state is None else state.value
