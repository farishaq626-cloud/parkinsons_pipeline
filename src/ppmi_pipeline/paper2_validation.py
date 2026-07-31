"""Nested, repeated, patient-isolated model evaluation for Paper 2."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.model_selection import (
    GridSearchCV,
    GroupKFold,
    StratifiedGroupKFold,
    cross_val_predict,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .evaluation import classification_metrics, regression_metrics
from .exceptions import MissingColumnError, PatientLeakageError
from .task_spec import OutcomeType

LOGGER = logging.getLogger("ppmi_pipeline.paper2_validation")


@dataclass(slots=True)
class NestedValidationResult:
    """Hold all auditable artifacts from one nested validation run."""

    fold_metrics: pd.DataFrame
    oof_predictions: pd.DataFrame
    fold_assignments: pd.DataFrame
    feature_records: pd.DataFrame
    tuning_results: pd.DataFrame

    def save(self, output_dir: str | Path) -> dict[str, Path]:
        """Persist nested-validation artifacts as CSV files.

        Args:
            output_dir: Local-only result directory.

        Returns:
            Mapping from artifact name to written path.
        """
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        artifacts = {
            "fold_metrics": destination / "fold_metrics.csv",
            "oof_predictions": destination / "oof_predictions.csv",
            "fold_assignments": destination / "fold_assignments.csv",
            "feature_records": destination / "feature_records.csv",
            "tuning_results": destination / "tuning_results.csv",
        }
        self.fold_metrics.to_csv(artifacts["fold_metrics"], index=False)
        self.oof_predictions.to_csv(artifacts["oof_predictions"], index=False)
        self.fold_assignments.to_csv(artifacts["fold_assignments"], index=False)
        self.feature_records.to_csv(artifacts["feature_records"], index=False)
        self.tuning_results.to_csv(artifacts["tuning_results"], index=False)
        return artifacts


class NestedPatientValidationRunner:
    """Run repeated nested validation with participant-level isolation.

    Hyperparameter tuning, imputation, scaling, and binary threshold selection
    occur exclusively within outer-training data. Outer test folds are used once
    for evaluation and never influence configuration or threshold choices.
    """

    def __init__(
        self,
        outcome_type: OutcomeType | str,
        target_column: str,
        feature_columns: list[str],
        model_family: str = "elasticnet",
        parameter_grid: dict[str, list[Any]] | None = None,
        patient_column: str = "PATNO",
        outer_splits: int = 5,
        outer_repeats: int = 5,
        inner_splits: int = 5,
        imputation_strategy: str = "median",
        threshold_selection: str = "youden",
        fixed_threshold: float = 0.5,
        random_state: int = 42,
        configuration_id: str = "paper2",
    ) -> None:
        """Initialize a nested validation run.

        Args:
            outcome_type: Continuous or binary outcome family.
            target_column: Modelling target field.
            feature_columns: Explicit baseline-only predictor list.
            model_family: ``elasticnet``, ``random_forest``, or
                ``hist_gradient_boosting``.
            parameter_grid: Hyperparameter grid without or with ``model__``
                prefixes.
            patient_column: Participant identifier field.
            outer_splits: Number of outer evaluation folds.
            outer_repeats: Number of independently shuffled outer repetitions.
            inner_splits: Number of inner tuning folds.
            imputation_strategy: Fold-local imputation strategy.
            threshold_selection: ``youden``, ``f1``, or ``fixed``.
            fixed_threshold: Probability cutoff for fixed thresholding.
            random_state: Base seed for all reproducible operations.
            configuration_id: Experiment-grid identifier persisted in outputs.

        Raises:
            ValueError: If validation or modelling settings are invalid.
        """
        self.outcome_type = OutcomeType(outcome_type)
        self.target_column = target_column
        self.feature_columns = list(dict.fromkeys(feature_columns))
        self.model_family = model_family
        self.parameter_grid = parameter_grid or {}
        self.patient_column = patient_column
        self.outer_splits = outer_splits
        self.outer_repeats = outer_repeats
        self.inner_splits = inner_splits
        self.imputation_strategy = imputation_strategy
        self.threshold_selection = threshold_selection
        self.fixed_threshold = fixed_threshold
        self.random_state = random_state
        self.configuration_id = configuration_id
        self._validate_settings()

    def run(self, dataset: pd.DataFrame) -> NestedValidationResult:
        """Execute nested patient-isolated evaluation.

        Args:
            dataset: One or more rows per patient with target and explicitly
                selected baseline-only predictors.

        Returns:
            Fold metrics, out-of-fold predictions, assignments, feature scores,
            and inner-tuning results.

        Raises:
            TypeError: If ``dataset`` is not a DataFrame.
            MissingColumnError: If required fields are absent.
            PatientLeakageError: If any outer or inner partition shares a patient.
            ValueError: If target or feature values cannot support evaluation.
        """
        prepared = self._prepare_dataset(dataset)
        x = prepared[self.feature_columns]
        y = prepared[self.target_column]
        groups = prepared[self.patient_column]
        metric_rows: list[dict[str, Any]] = []
        prediction_frames: list[pd.DataFrame] = []
        assignment_frames: list[pd.DataFrame] = []
        feature_frames: list[pd.DataFrame] = []
        tuning_rows: list[dict[str, Any]] = []

        for repeat in range(1, self.outer_repeats + 1):
            seed = self.random_state + repeat - 1
            outer_splitter = self._make_splitter(
                n_splits=self.outer_splits,
                random_state=seed,
            )
            for fold, (train_index, test_index) in enumerate(
                outer_splitter.split(x, y, groups), start=1
            ):
                train_df = prepared.iloc[train_index].copy()
                test_df = prepared.iloc[test_index].copy()
                self.verify_no_patient_overlap(train_df, test_df)
                fit_id = f"{self.configuration_id}-r{repeat}-f{fold}"
                assignment_frames.append(
                    self._fold_assignments(train_df, test_df, repeat, fold, fit_id)
                )

                estimator = self._build_pipeline(seed)
                inner_splitter = self._make_splitter(
                    n_splits=self.inner_splits,
                    random_state=seed + 10_000 + fold,
                )
                train_x = train_df[self.feature_columns]
                train_y = train_df[self.target_column]
                train_groups = train_df[self.patient_column]
                inner_cv = self._audited_splits(
                    train_x,
                    train_y,
                    train_groups,
                    inner_splitter,
                )
                search = GridSearchCV(
                    estimator=estimator,
                    param_grid=self._prefixed_parameter_grid(),
                    scoring=(
                        "roc_auc"
                        if self.outcome_type is OutcomeType.BINARY
                        else "neg_mean_absolute_error"
                    ),
                    cv=inner_cv,
                    refit=True,
                    error_score="raise",
                    n_jobs=-1,
                    pre_dispatch="2*n_jobs",
                )
                search.fit(train_x, train_y, groups=train_groups)
                tuning_rows.append(
                    {
                        "configuration_id": self.configuration_id,
                        "fit_id": fit_id,
                        "repeat": repeat,
                        "outer_fold": fold,
                        "best_score": float(search.best_score_),
                        "best_parameters": _json_parameters(search.best_params_),
                    }
                )

                test_x = test_df[self.feature_columns]
                test_y = test_df[self.target_column]
                threshold = float("nan")
                probabilities: np.ndarray | None = None
                if self.outcome_type is OutcomeType.BINARY:
                    threshold = self._select_threshold(
                        clone(search.best_estimator_),
                        train_x,
                        train_y,
                        train_groups,
                        seed + 20_000 + fold,
                    )
                    probabilities = search.best_estimator_.predict_proba(test_x)[:, 1]
                    predictions = probabilities >= threshold
                    metrics = classification_metrics(
                        test_y,
                        pd.Series(probabilities),
                        pd.Series(predictions.astype(int)),
                    )
                else:
                    predictions = search.best_estimator_.predict(test_x)
                    metrics = regression_metrics(test_y, pd.Series(predictions))

                metric_rows.append(
                    {
                        "configuration_id": self.configuration_id,
                        "fit_id": fit_id,
                        "repeat": repeat,
                        "outer_fold": fold,
                        "model_family": self.model_family,
                        "train_patients": int(train_groups.nunique()),
                        "test_patients": int(test_df[self.patient_column].nunique()),
                        "threshold": threshold,
                        **metrics,
                    }
                )
                prediction_frames.append(
                    self._prediction_frame(
                        test_df,
                        predictions,
                        probabilities,
                        repeat,
                        fold,
                        fit_id,
                        threshold,
                    )
                )
                feature_frames.append(
                    self._feature_records(
                        search.best_estimator_,
                        test_x,
                        test_y,
                        fit_id,
                        repeat,
                        fold,
                        seed,
                    )
                )

        return NestedValidationResult(
            fold_metrics=pd.DataFrame(metric_rows),
            oof_predictions=pd.concat(prediction_frames, ignore_index=True),
            fold_assignments=pd.concat(assignment_frames, ignore_index=True),
            feature_records=pd.concat(feature_frames, ignore_index=True),
            tuning_results=pd.DataFrame(tuning_rows),
        )

    @staticmethod
    def verify_no_patient_overlap(
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        patient_column: str = "PATNO",
    ) -> None:
        """Raise when a participant appears in both partitions.

        Args:
            train_df: Training partition.
            test_df: Evaluation partition.
            patient_column: Participant identifier field.

        Raises:
            MissingColumnError: If the participant field is missing.
            PatientLeakageError: If any participant is shared.
        """
        missing = [
            label
            for label, frame in (("train", train_df), ("test", test_df))
            if patient_column not in frame.columns
        ]
        if missing:
            raise MissingColumnError(
                f"Cannot audit patient overlap; {patient_column!r} is missing from "
                + " and ".join(missing)
                + "."
            )
        overlap = set(train_df[patient_column]).intersection(test_df[patient_column])
        if overlap:
            raise PatientLeakageError(
                "Patient leakage detected between training and evaluation: "
                f"{sorted(overlap)}"
            )

    def _prepare_dataset(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """Validate target, grouping, and baseline-only feature fields."""
        if not isinstance(dataset, pd.DataFrame):
            raise TypeError("Nested validation requires a pandas DataFrame.")
        required = {
            self.patient_column,
            self.target_column,
            *self.feature_columns,
        }
        missing = sorted(required.difference(dataset.columns))
        if missing:
            raise MissingColumnError(
                "Nested validation dataset is missing columns: " + ", ".join(missing)
            )
        if dataset[self.patient_column].isna().any():
            raise ValueError("Patient identifiers must be complete.")
        if dataset[self.target_column].isna().any():
            raise ValueError("The modelling target must not contain missing values.")
        if dataset[self.patient_column].duplicated().any():
            raise ValueError(
                "Nested validation expects one fixed-horizon row per patient."
            )
        prepared = dataset.copy()
        for column in self.feature_columns:
            prepared[column] = pd.to_numeric(prepared[column], errors="coerce")
        all_missing = [
            column for column in self.feature_columns if prepared[column].isna().all()
        ]
        if all_missing:
            raise ValueError(
                "Features contain no usable baseline values: " + ", ".join(all_missing)
            )
        if self.outcome_type is OutcomeType.BINARY:
            classes = set(prepared[self.target_column].unique())
            if classes != {0, 1}:
                raise ValueError("Binary targets must contain both classes 0 and 1.")
        patient_count = int(prepared[self.patient_column].nunique())
        largest_outer_test = int(np.ceil(patient_count / self.outer_splits))
        if patient_count < self.outer_splits:
            raise ValueError("The number of patients must be at least outer_splits.")
        if patient_count - largest_outer_test < self.inner_splits:
            raise ValueError(
                "Each outer-training partition must contain enough patients for "
                "inner_splits."
            )
        if self.outcome_type is OutcomeType.BINARY:
            class_counts = prepared.groupby(self.target_column)[
                self.patient_column
            ].nunique()
            for label, count in class_counts.items():
                largest_class_test = int(np.ceil(count / self.outer_splits))
                if (
                    count < self.outer_splits
                    or count - largest_class_test < self.inner_splits
                ):
                    raise ValueError(
                        "Binary nested validation requires enough patients from "
                        f"class {label} for every outer training and test fold."
                    )
        return prepared

    def _make_splitter(
        self,
        n_splits: int,
        random_state: int,
    ) -> GroupKFold | StratifiedGroupKFold:
        """Create a shuffled patient-grouped splitter for one validation level."""
        if self.outcome_type is OutcomeType.BINARY:
            return StratifiedGroupKFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=random_state,
            )
        return GroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state,
        )

    def _build_pipeline(self, random_state: int) -> Pipeline:
        """Build fold-local imputation, scaling, and model steps."""
        imputer_arguments: dict[str, Any] = {
            "strategy": self.imputation_strategy,
            "keep_empty_features": True,
        }
        if self.imputation_strategy == "constant":
            imputer_arguments["fill_value"] = 0.0
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(**imputer_arguments)),
                ("scaler", StandardScaler()),
                ("model", self._build_model(random_state)),
            ]
        )

    def _build_model(self, random_state: int) -> BaseEstimator:
        """Instantiate a supported classification or regression model."""
        if self.model_family == "elasticnet":
            if self.outcome_type is OutcomeType.BINARY:
                return LogisticRegression(
                    solver="saga",
                    l1_ratio=0.5,
                    C=1.0,
                    max_iter=10_000,
                    random_state=random_state,
                )
            return ElasticNet(max_iter=10_000, random_state=random_state)
        if self.model_family == "random_forest":
            if self.outcome_type is OutcomeType.BINARY:
                return RandomForestClassifier(
                    n_estimators=300,
                    min_samples_leaf=2,
                    random_state=random_state,
                    n_jobs=1,
                )
            return RandomForestRegressor(
                n_estimators=300,
                min_samples_leaf=2,
                random_state=random_state,
                n_jobs=1,
            )
        if self.model_family == "hist_gradient_boosting":
            if self.outcome_type is OutcomeType.BINARY:
                return HistGradientBoostingClassifier(random_state=random_state)
            return HistGradientBoostingRegressor(random_state=random_state)
        raise ValueError(f"Unsupported model family: {self.model_family}")

    def _prefixed_parameter_grid(self) -> dict[str, list[Any]]:
        """Prefix model parameters for use inside the scikit-learn pipeline."""
        return {
            key if key.startswith("model__") else f"model__{key}": values
            for key, values in self.parameter_grid.items()
        }

    def _select_threshold(
        self,
        estimator: Pipeline,
        train_x: pd.DataFrame,
        train_y: pd.Series,
        train_groups: pd.Series,
        random_state: int,
    ) -> float:
        """Select a binary threshold using inner-training predictions only."""
        if self.threshold_selection == "fixed":
            return self.fixed_threshold
        inner = self._make_splitter(self.inner_splits, random_state)
        inner_cv = self._audited_splits(
            train_x,
            train_y,
            train_groups,
            inner,
        )
        probabilities = cross_val_predict(
            estimator,
            train_x,
            train_y,
            groups=train_groups,
            cv=inner_cv,
            method="predict_proba",
            n_jobs=-1,
            pre_dispatch="2*n_jobs",
        )[:, 1]
        candidate_thresholds = np.unique(
            np.concatenate(([0.01], probabilities, [0.99]))
        )
        scores = []
        for threshold in candidate_thresholds:
            predicted = probabilities >= threshold
            if self.threshold_selection == "f1":
                score = classification_metrics(
                    train_y,
                    pd.Series(probabilities),
                    pd.Series(predicted.astype(int)),
                )["f1_score"]
            else:
                positives = train_y.eq(1).to_numpy()
                negatives = ~positives
                sensitivity = float(predicted[positives].mean())
                specificity = float((~predicted[negatives]).mean())
                score = sensitivity + specificity - 1
            scores.append(score)
        best_index = int(np.nanargmax(scores))
        return float(candidate_thresholds[best_index])

    def _audited_splits(
        self,
        x: pd.DataFrame,
        y: pd.Series,
        groups: pd.Series,
        splitter: GroupKFold | StratifiedGroupKFold,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Materialize and verify every patient-grouped inner split."""
        splits = list(splitter.split(x, y, groups))
        for train_index, test_index in splits:
            train_groups = set(groups.iloc[train_index])
            test_groups = set(groups.iloc[test_index])
            overlap = train_groups.intersection(test_groups)
            if overlap:
                raise PatientLeakageError(
                    "Patient leakage detected inside nested validation: "
                    f"{sorted(overlap)}"
                )
        return splits

    def _prediction_frame(
        self,
        test_df: pd.DataFrame,
        predictions: np.ndarray,
        probabilities: np.ndarray | None,
        repeat: int,
        fold: int,
        fit_id: str,
        threshold: float,
    ) -> pd.DataFrame:
        """Create one participant-level out-of-fold prediction table."""
        frame = pd.DataFrame(
            {
                "configuration_id": self.configuration_id,
                "fit_id": fit_id,
                "repeat": repeat,
                "outer_fold": fold,
                "patient_id": test_df[self.patient_column].to_numpy(),
                "y_true": test_df[self.target_column].to_numpy(),
                "y_pred": np.asarray(predictions, dtype=float),
                "threshold": threshold,
            }
        )
        frame["y_probability"] = (
            np.nan if probabilities is None else np.asarray(probabilities, dtype=float)
        )
        frame["residual"] = frame["y_true"] - frame["y_pred"]
        return frame

    def _fold_assignments(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        repeat: int,
        fold: int,
        fit_id: str,
    ) -> pd.DataFrame:
        """Persist complete train/test patient assignments for one outer fold."""
        rows = []
        for partition, frame in (("train", train_df), ("test", test_df)):
            rows.extend(
                {
                    "configuration_id": self.configuration_id,
                    "fit_id": fit_id,
                    "repeat": repeat,
                    "outer_fold": fold,
                    "partition": partition,
                    "patient_id": patient_id,
                }
                for patient_id in frame[self.patient_column].drop_duplicates()
            )
        return pd.DataFrame(rows)

    def _feature_records(
        self,
        estimator: Pipeline,
        test_x: pd.DataFrame,
        test_y: pd.Series,
        fit_id: str,
        repeat: int,
        fold: int,
        random_state: int,
    ) -> pd.DataFrame:
        """Extract signed coefficients or model-agnostic test importance."""
        model = estimator.named_steps["model"]
        importance_type = "permutation"
        if hasattr(model, "coef_"):
            signed_values = np.asarray(model.coef_).reshape(-1)
            importance = np.abs(signed_values)
            importance_type = "coefficient"
        elif hasattr(model, "feature_importances_"):
            importance = np.asarray(model.feature_importances_, dtype=float)
            signed_values = importance.copy()
            importance_type = "native_importance"
        else:
            scoring = (
                "roc_auc"
                if self.outcome_type is OutcomeType.BINARY
                else "neg_mean_absolute_error"
            )
            report = permutation_importance(
                estimator,
                test_x,
                test_y,
                scoring=scoring,
                n_repeats=5,
                random_state=random_state,
            )
            signed_values = np.asarray(report.importances_mean, dtype=float)
            importance = np.abs(signed_values)
        frame = pd.DataFrame(
            {
                "configuration_id": self.configuration_id,
                "fit_id": fit_id,
                "repeat": repeat,
                "outer_fold": fold,
                "model_family": self.model_family,
                "importance_type": importance_type,
                "feature": self.feature_columns,
                "signed_value": signed_values,
                "importance": importance,
            }
        )
        frame["rank"] = frame["importance"].rank(method="average", ascending=False)
        total_importance = float(frame["importance"].sum())
        frame["normalized_importance"] = (
            frame["importance"] / total_importance if total_importance > 0 else 0.0
        )
        return frame

    def _validate_settings(self) -> None:
        """Validate safety-critical nested-validation settings."""
        if not self.feature_columns:
            raise ValueError("At least one baseline-only feature is required.")
        for name, value in (
            ("outer_splits", self.outer_splits),
            ("outer_repeats", self.outer_repeats),
            ("inner_splits", self.inner_splits),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 2:
                raise ValueError(f"{name} must be an integer of at least 2.")
        if self.imputation_strategy not in {"median", "most_frequent", "constant"}:
            raise ValueError("Unsupported fold-local imputation strategy.")
        if self.threshold_selection not in {"youden", "f1", "fixed"}:
            raise ValueError("threshold_selection must be youden, f1, or fixed.")
        if not 0 < self.fixed_threshold < 1:
            raise ValueError("fixed_threshold must be between zero and one.")
        if self.model_family not in {
            "elasticnet",
            "random_forest",
            "hist_gradient_boosting",
        }:
            raise ValueError(f"Unsupported model family: {self.model_family}")


def _json_parameters(parameters: dict[str, Any]) -> str:
    """Serialize best-model parameters in deterministic key order."""
    import json

    return json.dumps(parameters, sort_keys=True, default=str)
