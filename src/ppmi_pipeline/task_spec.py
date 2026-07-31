"""Configuration contracts for Paper 2 fixed-horizon prediction tasks."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any, Self

from .exceptions import ConfigurationError

SUPPORTED_HORIZONS = {12: 365, 24: 730, 36: 1095}


class OutcomeType(StrEnum):
    """Supported Paper 2 outcome families."""

    CONTINUOUS = "continuous"
    BINARY = "binary"


@dataclass(frozen=True, slots=True)
class TaskSpecification:
    """Describe one auditable fixed-horizon prediction task.

    Args:
        name: Human-readable task name.
        endpoint: Endpoint identifier, for example ``mds_updrs_iii``.
        outcome_type: Continuous score change or binary progression.
        horizon_months: Prediction horizon in months. Paper 2 supports 12, 24,
            and 36 months.
        tolerance_days: Inclusive visit tolerance on either side of the target
            horizon.
        baseline_event_ids: Event identifiers eligible to define baseline.
        progression_threshold: Minimum score increase for a positive binary
            outcome. Required only for binary tasks.
        medication_state: Endpoint medication state retained for the task.

    Raises:
        ConfigurationError: If the task contract is internally inconsistent.
    """

    name: str
    endpoint: str
    outcome_type: OutcomeType
    horizon_months: int
    tolerance_days: int
    baseline_event_ids: tuple[str, ...] = ("BL",)
    progression_threshold: float | None = None
    medication_state: str | None = None

    def __post_init__(self) -> None:
        """Validate and normalize immutable task fields."""
        if not self.name.strip():
            raise ConfigurationError("Task name must be a non-empty string.")
        if not self.endpoint.strip():
            raise ConfigurationError("Task endpoint must be a non-empty string.")
        if self.horizon_months not in SUPPORTED_HORIZONS:
            raise ConfigurationError(
                "horizon_months must be one of 12, 24, or 36 for Paper 2."
            )
        if (
            isinstance(self.tolerance_days, bool)
            or not isinstance(self.tolerance_days, int)
            or self.tolerance_days < 0
        ):
            raise ConfigurationError("tolerance_days must be a non-negative integer.")
        normalized_events = tuple(
            str(event).strip().upper() for event in self.baseline_event_ids
        )
        if not normalized_events or any(not event for event in normalized_events):
            raise ConfigurationError("baseline_event_ids must not be empty.")
        object.__setattr__(self, "baseline_event_ids", normalized_events)

        try:
            outcome_type = OutcomeType(self.outcome_type)
        except ValueError as error:
            raise ConfigurationError(
                "outcome_type must be 'continuous' or 'binary'."
            ) from error
        object.__setattr__(self, "outcome_type", outcome_type)

        if outcome_type is OutcomeType.BINARY:
            threshold = self.progression_threshold
            if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
                raise ConfigurationError(
                    "Binary tasks require a numeric progression_threshold."
                )
        elif self.progression_threshold is not None:
            raise ConfigurationError(
                "Continuous tasks must not define progression_threshold."
            )

    @property
    def horizon_days(self) -> int:
        """Return the fixed day representation of the configured horizon."""
        return SUPPORTED_HORIZONS[self.horizon_months]

    @property
    def target_column(self) -> str:
        """Return the modelling target emitted by the dataset builder."""
        if self.outcome_type is OutcomeType.BINARY:
            return "progression_label"
        return "Delta_Score"

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> Self:
        """Construct a task from a validated configuration mapping.

        Args:
            values: Mapping containing task fields.

        Returns:
            A validated task specification.

        Raises:
            ConfigurationError: If required fields are missing or invalid.
        """
        required = {"name", "endpoint", "outcome_type", "horizon_months"}
        missing = sorted(required.difference(values))
        if missing:
            raise ConfigurationError(
                "Task specification is missing: " + ", ".join(missing)
            )
        try:
            return cls(
                name=str(values["name"]),
                endpoint=str(values["endpoint"]),
                outcome_type=OutcomeType(str(values["outcome_type"]).lower()),
                horizon_months=int(values["horizon_months"]),
                tolerance_days=int(values.get("tolerance_days", 90)),
                baseline_event_ids=tuple(values.get("baseline_event_ids", ("BL",))),
                progression_threshold=values.get("progression_threshold"),
                medication_state=values.get("medication_state"),
            )
        except (TypeError, ValueError) as error:
            raise ConfigurationError(
                f"Invalid task specification for {values.get('name', '<unnamed>')!r}."
            ) from error

    def to_dict(self) -> dict[str, Any]:
        """Serialize the task to JSON-compatible values."""
        payload = asdict(self)
        payload["outcome_type"] = self.outcome_type.value
        payload["baseline_event_ids"] = list(self.baseline_event_ids)
        payload["horizon_days"] = self.horizon_days
        payload["target_column"] = self.target_column
        return payload

    def fingerprint(self) -> str:
        """Return a stable SHA-256 fingerprint for this task contract."""
        serialized = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(serialized).hexdigest()
