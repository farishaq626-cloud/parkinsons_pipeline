"""Patient-grouped PPMI endpoint modelling and publication diagnostics."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import matplotlib
import pandas as pd
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline

from exceptions import MissingColumnError


matplotlib.use("Agg")
import matplotlib.pyplot as plt


LOGGER = logging.getLogger("ppmi_pipeline.analytics")
LEGACY_ANALYTICS_MODEL_CONFIG: dict[str, Any] = {
    "n_estimators": 100,
    "max_depth": None,
    "min_samples_leaf": 2,
    "test_size": 0.2,
    "n_jobs": -1,
    "random_state": 42,
    "shap_sample_size": 500,
}
LEGACY_FIGURE_DPI = 300


@dataclass(frozen=True)
class EndpointResult:
    """Immutable performance and diagnostic outputs for one clinical endpoint.

    Attributes:
        endpoint: Clinical endpoint modelled.
        n_observations: Number of aligned patient-visit observations.
        n_patients: Number of unique patients contributing observations.
        mae: Mean absolute prediction error.
        rmse: Root mean squared prediction error.
        r2: Coefficient of determination on the held-out partition.
        residual_plot_path: Saved residual-distribution figure.
        shap_plot_path: Saved SHAP feature-importance figure.
        model: Fitted preprocessing and regression pipeline.
    """

    endpoint: str
    n_observations: int
    n_patients: int
    mae: float
    rmse: float
    r2: float
    residual_plot_path: Path
    shap_plot_path: Path
    model: Pipeline


class PPMIAnalyticsEngine:
    """Train patient-isolated Random Forest endpoint models and diagnostics."""

    def __init__(
        self,
        model_config: dict[str, Any],
        output_dir: str | Path,
        run_id: str,
    ) -> None:
        """Initialise analytics outputs and merge model defaults.

        Args:
            model_config: Random Forest and diagnostic configuration overrides.
            output_dir: Directory for metrics and diagnostic figures.
            run_id: Unique execution identifier used in output filenames.

        Returns:
            None.
        """
        self.model_config = LEGACY_ANALYTICS_MODEL_CONFIG.copy()
        self.model_config.update(model_config)
        self.output_dir = Path(output_dir)
        self.run_id = run_id
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def split_by_patient(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        groups: pd.Series,
    ) -> tuple[Any, Any]:
        """Return deterministic train/test indices with no patient overlap.

        Args:
            features: Predictor matrix.
            target: Endpoint-delta outcome vector.
            groups: Patient identifier vector.

        Returns:
            Training and testing positional index arrays.

        Raises:
            ValueError: If fewer than two unique patients are available.
        """
        if groups.nunique() < 2:
            raise ValueError("At least two PPMI patients are required for splitting.")
        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=self.model_config["test_size"],
            random_state=self.model_config["random_state"],
        )
        return next(splitter.split(features, target, groups=groups))

    def train_and_evaluate(
        self,
        df: pd.DataFrame,
        feature_columns: Sequence[str],
        target_column: str,
        endpoint: str,
        group_column: str = "PATNO",
    ) -> EndpointResult:
        """Fit and evaluate a patient-grouped endpoint regression model.

        Args:
            df: Aligned endpoint modelling data.
            feature_columns: Baseline-compatible predictor names.
            target_column: Endpoint-delta outcome column.
            endpoint: Endpoint name used in reporting.
            group_column: Patient identifier column.

        Returns:
            Held-out performance metrics, diagnostics, and fitted model.

        Raises:
            MissingColumnError: If a requested feature, target, or group is absent.
            ValueError: If too few patients are available for grouped evaluation.
        """
        required = set(feature_columns) | {target_column, group_column}
        missing = sorted(required.difference(df.columns))
        if missing:
            raise MissingColumnError("PPMI model schema is missing: " + ", ".join(missing))
        features = df[list(feature_columns)]
        target = df[target_column]
        groups = df[group_column]
        train_idx, test_idx = self.split_by_patient(features, target, groups)
        model = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=self.model_config["n_estimators"],
                        max_depth=self.model_config["max_depth"],
                        min_samples_leaf=self.model_config["min_samples_leaf"],
                        n_jobs=self.model_config["n_jobs"],
                        random_state=self.model_config["random_state"],
                    ),
                ),
            ]
        )
        x_train, x_test = features.iloc[train_idx], features.iloc[test_idx]
        y_train, y_test = target.iloc[train_idx], target.iloc[test_idx]
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        residuals = y_test.to_numpy() - predictions
        residual_path = self._save_residual_plot(residuals, endpoint)
        shap_path = self._save_shap_summary(model, x_test, feature_columns, endpoint)
        result = EndpointResult(
            endpoint=endpoint,
            n_observations=len(df),
            n_patients=int(groups.nunique()),
            mae=float(mean_absolute_error(y_test, predictions)),
            rmse=float(mean_squared_error(y_test, predictions) ** 0.5),
            r2=float(r2_score(y_test, predictions)),
            residual_plot_path=residual_path,
            shap_plot_path=shap_path,
            model=model,
        )
        LOGGER.info("Completed patient-isolated %s endpoint modelling", endpoint)
        return result

    def _save_residual_plot(self, residuals: Any, endpoint: str) -> Path:
        """Save a residual-distribution diagnostic plot.

        Args:
            residuals: Held-out observed-minus-predicted residual values.
            endpoint: Endpoint name used in the figure title and filename.

        Returns:
            Path to the saved PNG figure.
        """
        path = self.output_dir / f"{self.run_id}_{endpoint}_residual_distribution.png"
        figure, axis = plt.subplots(figsize=(7, 5))
        axis.hist(residuals, bins=30, color="#4C78A8", edgecolor="white")
        axis.axvline(0, color="#D62728", linestyle="--", linewidth=1.5, label="Zero error")
        axis.set(
            title=f"{endpoint}: residual distribution",
            xlabel="Observed delta - predicted delta",
            ylabel="Count",
        )
        axis.legend()
        figure.tight_layout()
        figure.savefig(path, dpi=LEGACY_FIGURE_DPI)
        plt.close(figure)
        return path

    def _save_shap_summary(
        self,
        model: Pipeline,
        x_test: pd.DataFrame,
        feature_columns: Sequence[str],
        endpoint: str,
    ) -> Path:
        """Save a SHAP feature-importance summary for one held-out endpoint.

        Args:
            model: Fitted imputation and Random Forest pipeline.
            x_test: Held-out predictor matrix.
            feature_columns: Predictor names in model order.
            endpoint: Endpoint name used in the figure title and filename.

        Returns:
            Path to the saved PNG figure.

        Raises:
            ValueError: If no held-out observations are available for SHAP.
        """
        if x_test.empty:
            raise ValueError("Cannot create SHAP summary without held-out observations.")
        path = self.output_dir / f"{self.run_id}_{endpoint}_shap_summary.png"
        sample_size = min(len(x_test), self.model_config["shap_sample_size"])
        sample = x_test.sample(n=sample_size, random_state=self.model_config["random_state"])
        imputed = model.named_steps["imputer"].transform(sample)
        shap_data = pd.DataFrame(imputed, columns=feature_columns, index=sample.index)
        explainer = shap.TreeExplainer(model.named_steps["model"])
        shap_values = explainer.shap_values(shap_data)
        shap.summary_plot(shap_values, shap_data, show=False, plot_size=(8, 5))
        plt.title(f"{endpoint}: SHAP feature importance")
        plt.tight_layout()
        plt.savefig(path, dpi=LEGACY_FIGURE_DPI, bbox_inches="tight")
        plt.close()
        return path
