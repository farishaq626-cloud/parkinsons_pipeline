"""Interpretable baseline prognostic modelling for fixed-horizon PPMI data."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import DEFAULT_LOGISTIC_REGRESSION_CONFIG, DEFAULT_N_SPLITS
from validation import ValidationFramework


LOGGER = logging.getLogger("ppmi_pipeline.modeling")


class PrognosticModel:
    """Train an interpretable, patient-isolated baseline prognosis model.

    The baseline model is ElasticNet-regularised logistic regression. It uses
    only baseline-compatible numeric predictors and excludes identifiers and
    fixed-horizon outcome columns that could leak target information.

    Args:
        dataset: Patient-level modelling dataset. A ``PATNO`` column from
            ``create_fixed_horizon_dataset`` is accepted and is handled by the
            validation framework as ``patient_id``.
        target: Binary prognosis-label column to predict.
        n_splits: Number of patient-grouped GroupKFold partitions.
        c: Inverse regularisation strength for logistic regression.
        l1_ratio: ElasticNet mixing parameter, where 0 is L2 and 1 is L1.
        random_state: Seed used by the ``saga`` solver.

    Attributes:
        fold_metrics_: Per-fold Precision, Recall, F1, and AUC-ROC results.
        fold_coefficients_: Per-fold linear coefficients and normalized
            importance values.
        feature_stability_: Cross-fold summary of feature importance and
            coefficient stability.
    """

    non_feature_columns = {
        "PATNO",
        "patient_id",
        "Target_Score",
        "Delta_Score",
        "progression_label",
    }

    def __init__(
        self,
        dataset: pd.DataFrame,
        target: str = "progression_label",
        n_splits: int = DEFAULT_N_SPLITS,
        c: float = DEFAULT_LOGISTIC_REGRESSION_CONFIG["c"],
        l1_ratio: float = DEFAULT_LOGISTIC_REGRESSION_CONFIG["l1_ratio"],
        random_state: int = DEFAULT_LOGISTIC_REGRESSION_CONFIG["random_state"],
        validation_framework: ValidationFramework | None = None,
    ) -> None:
        self.target = target
        self.c = c
        self.l1_ratio = l1_ratio
        self.random_state = random_state
        self.validation = validation_framework or ValidationFramework(
            dataset, target=target, n_splits=n_splits
        )
        if self.validation.target != target:
            raise ValueError(
                "validation_framework target must match the PrognosticModel target."
            )
        self.dataset = self.validation.dataset
        self.feature_names = self._select_feature_names()
        self.fold_metrics_ = pd.DataFrame()
        self.fold_coefficients_ = pd.DataFrame()
        self.feature_stability_ = pd.DataFrame()
        self.fold_models_: list[Pipeline] = []
        self._validate_model_configuration()

    @classmethod
    def train_and_evaluate(
        cls,
        dataset: pd.DataFrame,
        target: str = "progression_label",
        n_splits: int = DEFAULT_N_SPLITS,
        c: float = DEFAULT_LOGISTIC_REGRESSION_CONFIG["c"],
        l1_ratio: float = DEFAULT_LOGISTIC_REGRESSION_CONFIG["l1_ratio"],
        random_state: int = DEFAULT_LOGISTIC_REGRESSION_CONFIG["random_state"],
        output_dir: str | Path | None = None,
        validation_framework: ValidationFramework | None = None,
    ) -> PrognosticModel:
        """Train and evaluate the baseline model across patient-isolated folds.

        Args:
            dataset: Patient-level modelling dataset.
            target: Binary prognosis-label column to predict.
            n_splits: Number of patient-grouped GroupKFold partitions.
            c: Inverse regularisation strength for logistic regression.
            l1_ratio: ElasticNet mixing parameter, where 0 is L2 and 1 is L1.
            random_state: Seed used by the ``saga`` solver.
            output_dir: Optional directory for CSV copies of fold metrics,
                fold coefficients, and feature-stability summaries.

        Returns:
            A fitted ``PrognosticModel`` instance containing standard reports
            in ``fold_metrics_``, ``fold_coefficients_``, and
            ``feature_stability_``.
        """
        model = cls(
            dataset=dataset,
            target=target,
            n_splits=n_splits,
            c=c,
            l1_ratio=l1_ratio,
            random_state=random_state,
            validation_framework=validation_framework,
        )
        model._fit_folds()
        if output_dir is not None:
            model.save_reports(output_dir)
        return model

    def save_reports(self, output_dir: str | Path) -> None:
        """Save fold metrics, coefficients, and stability summaries as CSV files.

        Args:
            output_dir: Directory in which the modelling reports will be saved.
        """
        if self.fold_metrics_.empty:
            raise RuntimeError("No fold results are available. Run train_and_evaluate first.")

        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        self.fold_metrics_.to_csv(destination / "fold_metrics.csv", index=False)
        self.fold_coefficients_.to_csv(destination / "fold_coefficients.csv", index=False)
        self.feature_stability_.to_csv(destination / "feature_stability.csv", index=False)
        LOGGER.info("Saved prognostic model reports to %s", destination)

    def _fit_folds(self) -> None:
        """Fit a separate ElasticNet logistic model for each validation fold."""
        metric_rows: list[dict[str, float | int]] = []
        coefficient_frames: list[pd.DataFrame] = []

        for fold_number, (train_df, test_df) in enumerate(
            self.validation.get_splits(), start=1
        ):
            train_x = train_df[self.feature_names]
            test_x = test_df[self.feature_names]
            train_y = train_df[self.target]
            test_y = test_df[self.target]
            if train_y.nunique() < 2:
                raise ValueError(
                    f"Fold {fold_number} training data contains only one target class. "
                    "Reduce n_splits or provide more balanced patient data."
                )

            estimator = self._build_estimator()
            estimator.fit(train_x, train_y)
            self.fold_models_.append(estimator)

            predictions = estimator.predict(test_x)
            probabilities = estimator.predict_proba(test_x)
            classifier = estimator.named_steps["classifier"]
            positive_class = classifier.classes_[1]
            positive_class_index = list(classifier.classes_).index(positive_class)
            auc_roc = float("nan")
            if test_y.nunique() == 2:
                auc_roc = float(roc_auc_score(test_y, probabilities[:, positive_class_index]))

            metric_rows.append(
                {
                    "fold": fold_number,
                    "train_patients": int(train_df["patient_id"].nunique()),
                    "test_patients": int(test_df["patient_id"].nunique()),
                    "precision": float(
                        precision_score(
                            test_y,
                            predictions,
                            pos_label=positive_class,
                            zero_division=0,
                        )
                    ),
                    "recall": float(
                        recall_score(
                            test_y,
                            predictions,
                            pos_label=positive_class,
                            zero_division=0,
                        )
                    ),
                    "f1_score": float(
                        f1_score(
                            test_y,
                            predictions,
                            pos_label=positive_class,
                            zero_division=0,
                        )
                    ),
                    "auc_roc": auc_roc,
                }
            )
            coefficient_frames.append(
                self._extract_feature_importance(estimator, fold_number)
            )

        self.fold_metrics_ = pd.DataFrame(metric_rows)
        self.fold_coefficients_ = pd.concat(coefficient_frames, ignore_index=True)
        self.feature_stability_ = self._summarize_feature_stability()
        LOGGER.info("Completed %d patient-isolated prognostic folds", len(metric_rows))

    def _build_estimator(self) -> Pipeline:
        """Build the standardised ElasticNet logistic-regression pipeline."""
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        solver="saga",
                        l1_ratio=self.l1_ratio,
                        C=self.c,
                        max_iter=DEFAULT_LOGISTIC_REGRESSION_CONFIG["max_iter"],
                        random_state=self.random_state,
                    ),
                ),
            ]
        )

    def _extract_feature_importance(
        self,
        estimator: Pipeline,
        fold_number: int,
    ) -> pd.DataFrame:
        """Extract normalized coefficient or tree-importance values per fold."""
        classifier = estimator.named_steps["classifier"]
        if hasattr(classifier, "coef_"):
            raw_scores = classifier.coef_.ravel()
            score_name = "coefficient"
        elif hasattr(classifier, "feature_importances_"):
            raw_scores = classifier.feature_importances_
            score_name = "feature_importance"
        else:
            raise TypeError(
                "Feature importance is unavailable for the configured classifier."
            )

        importance = pd.DataFrame(
            {
                "fold": fold_number,
                "feature": self.feature_names,
                score_name: raw_scores,
            }
        )
        importance["absolute_importance"] = importance[score_name].abs()
        total_importance = float(importance["absolute_importance"].sum())
        if total_importance == 0:
            importance["normalized_importance"] = 0.0
        else:
            importance["normalized_importance"] = (
                importance["absolute_importance"] / total_importance
            )
        return importance

    def _summarize_feature_stability(self) -> pd.DataFrame:
        """Summarize feature magnitude and non-zero coefficient stability."""
        coefficient_column = (
            "coefficient"
            if "coefficient" in self.fold_coefficients_.columns
            else "feature_importance"
        )
        summary = (
            self.fold_coefficients_
            .assign(_nonzero=lambda frame: frame[coefficient_column].ne(0))
            .groupby("feature", as_index=False)
            .agg(
                mean_normalized_importance=("normalized_importance", "mean"),
                importance_standard_deviation=("normalized_importance", "std"),
                mean_coefficient=(coefficient_column, "mean"),
                nonzero_folds=("_nonzero", "sum"),
            )
        )
        summary["stability_rate"] = summary["nonzero_folds"] / self.validation.n_splits
        return summary.sort_values(
            "mean_normalized_importance", ascending=False, kind="stable"
        ).reset_index(drop=True)

    def _select_feature_names(self) -> list[str]:
        """Select numeric baseline-compatible predictors and exclude leakage fields."""
        excluded_columns = self.non_feature_columns.union({self.target})
        candidate_columns = [
            column for column in self.dataset.columns if column not in excluded_columns
        ]
        return self.dataset[candidate_columns].select_dtypes(include="number").columns.tolist()

    def _validate_model_configuration(self) -> None:
        """Validate binary-target and ElasticNet modelling assumptions."""
        if not self.feature_names:
            raise ValueError(
                "No numeric baseline-compatible features are available for modelling."
            )
        if self.dataset[self.target].nunique() != 2:
            raise ValueError(
                f"Target '{self.target}' must contain exactly two classes for logistic regression."
            )
        if not isinstance(self.c, (int, float)) or self.c <= 0:
            raise ValueError("c must be a positive number.")
        if not isinstance(self.l1_ratio, (int, float)) or not 0 <= self.l1_ratio <= 1:
            raise ValueError("l1_ratio must be a number between 0 and 1.")
