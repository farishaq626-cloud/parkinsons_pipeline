"""Patient-grouped PPMI endpoint modelling and publication diagnostics."""

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline


@dataclass(frozen=True)
class EndpointResult:
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
    def __init__(self, model_config: dict, output_dir: str | Path, run_id: str):
        self.model_config = model_config
        self.output_dir = Path(output_dir)
        self.run_id = run_id
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def train_and_evaluate(self, df, feature_columns, target_column, endpoint, group_column="PATNO"):
        required = set(feature_columns) | {target_column, group_column}
        missing = sorted(required.difference(df.columns))
        if missing:
            raise ValueError(f"PPMI model schema is missing: {', '.join(missing)}")
        X, y, groups = df[feature_columns], df[target_column], df[group_column]
        if groups.nunique() < 2:
            raise ValueError("At least two PPMI patients are required for a grouped train/test split.")

        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=self.model_config["test_size"],
            random_state=self.model_config["random_state"],
        )
        train_idx, test_idx = next(splitter.split(X, y, groups=groups))
        model = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestRegressor(
                n_estimators=self.model_config["n_estimators"],
                max_depth=self.model_config["max_depth"],
                min_samples_leaf=self.model_config["min_samples_leaf"],
                n_jobs=self.model_config["n_jobs"],
                random_state=self.model_config["random_state"],
            )),
        ])
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        residuals = y_test.to_numpy() - predictions

        residual_path = self._save_residual_plot(residuals, endpoint)
        shap_path = self._save_shap_summary(model, X_test, feature_columns, endpoint)
        return EndpointResult(
            endpoint=endpoint,
            n_observations=len(df),
            n_patients=groups.nunique(),
            mae=mean_absolute_error(y_test, predictions),
            rmse=mean_squared_error(y_test, predictions) ** 0.5,
            r2=r2_score(y_test, predictions),
            residual_plot_path=residual_path,
            shap_plot_path=shap_path,
            model=model,
        )

    def _save_residual_plot(self, residuals, endpoint: str) -> Path:
        path = self.output_dir / f"{self.run_id}_{endpoint}_residual_distribution.png"
        figure, axis = plt.subplots(figsize=(7, 5))
        axis.hist(residuals, bins=30, color="#4C78A8", edgecolor="white")
        axis.axvline(0, color="#D62728", linestyle="--", linewidth=1.5, label="Zero error")
        axis.set(title=f"{endpoint}: residual distribution", xlabel="Observed delta − predicted delta", ylabel="Count")
        axis.legend()
        figure.tight_layout()
        figure.savefig(path, dpi=300)
        plt.close(figure)
        return path

    def _save_shap_summary(self, model: Pipeline, X_test: pd.DataFrame, feature_columns, endpoint: str) -> Path:
        path = self.output_dir / f"{self.run_id}_{endpoint}_shap_summary.png"
        sample_size = min(len(X_test), self.model_config["shap_sample_size"])
        sample = X_test.sample(n=sample_size, random_state=self.model_config["random_state"])
        imputed = model.named_steps["imputer"].transform(sample)
        shap_data = pd.DataFrame(imputed, columns=feature_columns, index=sample.index)
        explainer = shap.TreeExplainer(model.named_steps["model"])
        shap_values = explainer.shap_values(shap_data)
        shap.summary_plot(shap_values, shap_data, show=False, plot_size=(8, 5))
        plt.title(f"{endpoint}: SHAP feature importance")
        plt.tight_layout()
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()
        return path
