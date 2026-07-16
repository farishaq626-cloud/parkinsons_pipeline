"""Single, reproducible entry point for longitudinal PPMI endpoint modelling."""

import json
import os
from datetime import datetime
from pathlib import Path

import pandas as pd

from analytics_engine import PPMIAnalyticsEngine
from etl import PPMIDataLoader
from feature_engine import PPMIFeatureEngine


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.json"


def load_config(config_path: str | Path | None = None) -> dict:
    path = Path(config_path or os.environ.get("PPMI_CONFIG_PATH", DEFAULT_CONFIG_PATH))
    with path.open(encoding="utf-8") as config_file:
        config = json.load(config_file)
    data_path = Path(config["data_path"])
    config["data_path"] = data_path if data_path.is_absolute() else path.parent / data_path
    results_dir = Path(config.get("results_dir", "results"))
    config["results_dir"] = results_dir if results_dir.is_absolute() else path.parent / results_dir
    return config


def _write_run_log(path: Path, timestamp: datetime, config: dict, results) -> None:
    log_config = {key: str(value) if isinstance(value, Path) else value for key, value in config.items()}
    lines = [
        f"timestamp: {timestamp.isoformat(timespec='seconds')}",
        f"data_file: {config['data_path']}",
        "config:",
        json.dumps(log_config, indent=2, default=str),
        "metrics:",
    ]
    lines.extend(
        f"{result.endpoint}: R2={result.r2:.6f}, MAE={result.mae:.6f}, RMSE={result.rmse:.6f}"
        for result in results
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_pipeline(config_path: str | Path | None = None):
    config = load_config(config_path)
    started_at = datetime.now().astimezone()
    run_id = started_at.strftime("run_%Y_%m_%d_%H%M%S")
    results_dir = config["results_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)

    ppmi = PPMIDataLoader().load(config["data_path"], sheet_name=config["sheet_name"])
    feature_engine = PPMIFeatureEngine(
        patient_col=config["patient_column"], baseline_events=config["baseline_events"]
    )
    analytics_engine = PPMIAnalyticsEngine(config["model"], results_dir, run_id)
    results = []
    for endpoint in config["endpoints"]:
        endpoint_df, feature_columns, target_column = feature_engine.build_endpoint_dataset(
            ppmi, endpoint, config["baseline_features"]
        )
        result = analytics_engine.train_and_evaluate(
            endpoint_df, feature_columns, target_column, endpoint, config["patient_column"]
        )
        results.append(result)
        print(
            f"{endpoint}: {result.n_observations} observations / {result.n_patients} patients | "
            f"MAE={result.mae:.3f} | RMSE={result.rmse:.3f} | R²={result.r2:.3f}"
        )

    metrics_path = results_dir / f"{run_id}_model_metrics.csv"
    pd.DataFrame([
        {
            "endpoint": result.endpoint,
            "n_observations": result.n_observations,
            "n_patients": result.n_patients,
            "r2": result.r2,
            "mae": result.mae,
            "rmse": result.rmse,
            "residual_plot": str(result.residual_plot_path),
            "shap_summary_plot": str(result.shap_plot_path),
        }
        for result in results
    ]).to_csv(metrics_path, index=False)
    log_path = results_dir / f"{run_id}_run_log.txt"
    _write_run_log(log_path, started_at, config, results)
    print(f"Metrics: {metrics_path}")
    print(f"Run log: {log_path}")
    return results


if __name__ == "__main__":
    run_pipeline()
