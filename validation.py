"""Patient-isolated cross-validation utilities for fixed-horizon PPMI datasets."""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import GroupKFold

from config import DEFAULT_N_SPLITS
from exceptions import MissingColumnError


class ValidationFramework:
    """Create leakage-safe patient-level cross-validation folds.

    ``create_fixed_horizon_dataset`` emits ``PATNO`` as its patient identifier.
    This framework creates a checked ``patient_id`` alias from that column so
    all grouping and overlap checks use one explicit identifier name.

    Args:
        dataset: Modelling DataFrame returned by ``create_fixed_horizon_dataset``.
            It must include ``PATNO`` or ``patient_id`` and the target column.
        target: Name of the outcome column to model, for example ``Delta_Score``.
        n_splits: Number of patient-level GroupKFold partitions.

    Raises:
        TypeError: If ``dataset`` is not a pandas DataFrame.
        ValueError: If the target or patient identifier is unavailable, contains
            missing values, or there are too few unique patients for the folds.
    """

    patient_id_column = "patient_id"
    fixed_horizon_patient_column = "PATNO"

    def __init__(
        self,
        dataset: pd.DataFrame,
        target: str,
        n_splits: int = DEFAULT_N_SPLITS,
    ) -> None:
        self.target = target
        self.n_splits = n_splits
        self.dataset = self._prepare_dataset(dataset)
        self._validate_configuration()

    def get_splits(self) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
        """Return GroupKFold splits with absolute patient-level isolation.

        Returns:
            A list of ``(train_df, test_df)`` DataFrame pairs. Every returned
            DataFrame includes ``patient_id`` so the isolation rule remains
            visible and can be verified before model fitting.
        """
        features = self.dataset.drop(columns=[self.target])
        outcome = self.dataset[self.target]
        groups = self.dataset[self.patient_id_column]
        splitter = GroupKFold(n_splits=self.n_splits)

        splits: list[tuple[pd.DataFrame, pd.DataFrame]] = []
        for train_index, test_index in splitter.split(features, outcome, groups):
            train_df = self.dataset.iloc[train_index].copy()
            test_df = self.dataset.iloc[test_index].copy()
            self.verify_no_overlap(train_df, test_df)
            splits.append((train_df, test_df))
        return splits

    @staticmethod
    def verify_no_overlap(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
        """Assert that training and testing partitions share no patient IDs.

        Args:
            train_df: Training partition containing a ``patient_id`` column.
            test_df: Testing partition containing a ``patient_id`` column.

        Raises:
            ValueError: If either partition lacks the required identifier.
            AssertionError: If a patient appears in both partitions.
        """
        required = {ValidationFramework.patient_id_column}
        missing_train = sorted(required.difference(train_df.columns))
        missing_test = sorted(required.difference(test_df.columns))
        if missing_train or missing_test:
            missing = sorted(set(missing_train).union(missing_test))
            raise MissingColumnError(
                "Cannot verify patient overlap; missing required column: "
                + ", ".join(missing)
            )

        overlapping_patients = set(train_df["patient_id"]).intersection(
            test_df["patient_id"]
        )
        assert not overlapping_patients, (
            "Patient leakage detected between training and testing partitions: "
            f"{sorted(overlapping_patients)}"
        )

    def _prepare_dataset(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """Copy the dataset and create a validated patient_id alias if needed."""
        if not isinstance(dataset, pd.DataFrame):
            raise TypeError("dataset must be a pandas DataFrame.")

        prepared = dataset.copy()
        has_patient_id = self.patient_id_column in prepared.columns
        has_patno = self.fixed_horizon_patient_column in prepared.columns
        if not has_patient_id and not has_patno:
            raise MissingColumnError(
                "dataset must contain 'patient_id' or the fixed-horizon identifier 'PATNO'."
            )
        if not has_patient_id:
            prepared[self.patient_id_column] = prepared[self.fixed_horizon_patient_column]
        elif has_patno and not prepared[self.patient_id_column].equals(
            prepared[self.fixed_horizon_patient_column]
        ):
            raise ValueError("patient_id and PATNO must match when both are present.")

        return prepared

    def _validate_configuration(self) -> None:
        """Validate the target, patient groups, and GroupKFold configuration."""
        if self.target not in self.dataset.columns:
            raise MissingColumnError(
                f"Target column '{self.target}' is not present in dataset."
            )
        if self.dataset[self.patient_id_column].isna().any():
            raise ValueError("patient_id contains missing values and cannot be grouped safely.")
        if self.dataset[self.target].isna().any():
            raise ValueError(f"Target column '{self.target}' contains missing values.")
        if isinstance(self.n_splits, bool) or not isinstance(self.n_splits, int):
            raise ValueError("n_splits must be an integer greater than one.")
        if self.n_splits < 2:
            raise ValueError("n_splits must be an integer greater than one.")

        patient_count = int(self.dataset[self.patient_id_column].nunique())
        if patient_count < self.n_splits:
            raise ValueError(
                f"n_splits={self.n_splits} requires at least {self.n_splits} unique "
                f"patients, but dataset contains {patient_count}."
            )
