"""TRIPOD+AI-oriented metrics and stability analysis for Paper 2."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

from .exceptions import MissingColumnError
from .task_spec import OutcomeType


def regression_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    """Calculate continuous-outcome performance and residual diagnostics.

    Args:
        y_true: Observed continuous outcomes.
        y_pred: Predicted continuous outcomes.

    Returns:
        MAE, RMSE, R-squared, residual mean, and residual standard deviation.

    Raises:
        ValueError: If values are empty, non-finite, or different lengths.
    """
    observed, predicted = _validated_numeric_pair(y_true, y_pred)
    residuals = observed - predicted
    return {
        "mae": float(mean_absolute_error(observed, predicted)),
        "rmse": float(mean_squared_error(observed, predicted) ** 0.5),
        "r2": float(r2_score(observed, predicted)),
        "residual_mean": float(np.mean(residuals)),
        "residual_standard_deviation": float(np.std(residuals, ddof=1)),
    }


def classification_metrics(
    y_true: pd.Series,
    probabilities: pd.Series,
    predictions: pd.Series,
) -> dict[str, float]:
    """Calculate discrimination, calibration, and thresholded metrics.

    Args:
        y_true: Observed binary outcomes.
        probabilities: Predicted positive-class probabilities.
        predictions: Thresholded binary predictions selected without test data.

    Returns:
        AUROC, AUPRC, Brier score, calibration intercept and slope, precision,
        recall, and F1 score.

    Raises:
        ValueError: If arrays differ in length or contain invalid values.
    """
    observed, probability = _validated_numeric_pair(y_true, probabilities)
    _, predicted = _validated_numeric_pair(y_true, predictions)
    classes = set(np.unique(observed))
    if not classes.issubset({0.0, 1.0}):
        raise ValueError("Classification outcomes must be encoded as 0 and 1.")
    if np.any((probability < 0) | (probability > 1)):
        raise ValueError("Classification probabilities must be between 0 and 1.")
    auc_roc = float("nan")
    auc_pr = float("nan")
    calibration_intercept = float("nan")
    calibration_slope = float("nan")
    if len(classes) == 2:
        auc_roc = float(roc_auc_score(observed, probability))
        auc_pr = float(average_precision_score(observed, probability))
        calibration_intercept, calibration_slope = calibration_parameters(
            observed, probability
        )
    return {
        "auroc": auc_roc,
        "auprc": auc_pr,
        "brier_score": float(brier_score_loss(observed, probability)),
        "calibration_intercept": calibration_intercept,
        "calibration_slope": calibration_slope,
        "precision": float(precision_score(observed, predicted, zero_division=0)),
        "recall": float(recall_score(observed, predicted, zero_division=0)),
        "f1_score": float(f1_score(observed, predicted, zero_division=0)),
    }


def calibration_parameters(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> tuple[float, float]:
    """Estimate logistic calibration intercept and slope.

    Args:
        y_true: Binary observed outcomes.
        probabilities: Predicted positive-class probabilities.

    Returns:
        Calibration intercept and slope. ``NaN`` values are returned when the
        estimate is undefined.
    """
    clipped = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1 - 1e-6)
    logits = np.log(clipped / (1 - clipped))
    if np.std(logits) == 0 or len(np.unique(y_true)) < 2:
        return float("nan"), float("nan")
    calibrator = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2_000)
    calibrator.fit(logits.reshape(-1, 1), y_true)
    return float(calibrator.intercept_[0]), float(calibrator.coef_[0, 0])


def patient_bootstrap_confidence_intervals(
    predictions: pd.DataFrame,
    outcome_type: OutcomeType | str,
    patient_column: str = "patient_id",
    n_resamples: int = 1_000,
    confidence_level: float = 0.95,
    random_state: int = 42,
) -> pd.DataFrame:
    """Calculate cluster-bootstrap confidence intervals at patient level.

    Repeated-fold predictions for a sampled patient remain together, so the
    bootstrap unit is the participant rather than the prediction row.

    Args:
        predictions: Out-of-fold prediction table.
        outcome_type: Continuous or binary outcome family.
        patient_column: Participant identifier field.
        n_resamples: Number of patient-level bootstrap samples.
        confidence_level: Central interval coverage between zero and one.
        random_state: Reproducible random seed.

    Returns:
        Metric estimates, lower and upper limits, and valid resample counts.

    Raises:
        MissingColumnError: If prediction fields are absent.
        ValueError: If bootstrap settings or patient identifiers are invalid.
    """
    outcome = OutcomeType(outcome_type)
    required = {patient_column, "y_true", "y_pred"}
    if outcome is OutcomeType.BINARY:
        required.add("y_probability")
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise MissingColumnError(
            "Cannot bootstrap prediction metrics; missing columns: "
            + ", ".join(missing)
        )
    if predictions[patient_column].isna().any():
        raise ValueError("Patient identifiers must be complete for bootstrap CIs.")
    if (
        isinstance(n_resamples, bool)
        or not isinstance(n_resamples, int)
        or n_resamples < 1
    ):
        raise ValueError("n_resamples must be a positive integer.")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between zero and one.")

    metric_function = _metric_function(outcome)
    point_estimates = metric_function(predictions)
    patient_ids = predictions[patient_column].drop_duplicates().to_numpy()
    grouped = {
        patient_id: group
        for patient_id, group in predictions.groupby(patient_column, sort=False)
    }
    rng = np.random.default_rng(random_state)
    samples: dict[str, list[float]] = {metric: [] for metric in point_estimates}
    for _ in range(n_resamples):
        sampled_ids = rng.choice(patient_ids, size=len(patient_ids), replace=True)
        sample = pd.concat(
            [grouped[patient_id] for patient_id in sampled_ids],
            ignore_index=True,
        )
        metrics = metric_function(sample)
        for metric, value in metrics.items():
            if np.isfinite(value):
                samples[metric].append(value)

    alpha = (1 - confidence_level) / 2
    rows = []
    for metric, estimate in point_estimates.items():
        values = samples[metric]
        lower = float("nan")
        upper = float("nan")
        if values:
            lower, upper = np.quantile(values, [alpha, 1 - alpha]).tolist()
        rows.append(
            {
                "metric": metric,
                "estimate": estimate,
                "ci_lower": lower,
                "ci_upper": upper,
                "confidence_level": confidence_level,
                "valid_resamples": len(values),
            }
        )
    return pd.DataFrame(rows)


def summarize_feature_stability(feature_records: pd.DataFrame) -> pd.DataFrame:
    """Summarize coefficient direction, inclusion, and rank stability.

    Args:
        feature_records: Per-fit feature scores containing ``feature``,
            ``importance``, ``signed_value``, ``rank``, and ``fit_id``.

    Returns:
        One row per feature with sign consistency, inclusion frequency, mean
        rank, rank standard deviation, and normalized rank stability.

    Raises:
        MissingColumnError: If required feature-report fields are absent.
    """
    required = {
        "feature",
        "importance",
        "signed_value",
        "rank",
        "fit_id",
    }
    missing = sorted(required.difference(feature_records.columns))
    if missing:
        raise MissingColumnError(
            "Cannot summarize feature stability; missing columns: " + ", ".join(missing)
        )
    if feature_records.empty:
        return pd.DataFrame(
            columns=[
                "feature",
                "mean_importance",
                "inclusion_frequency",
                "sign_consistency",
                "mean_rank",
                "rank_standard_deviation",
                "rank_stability",
                "n_fits",
            ]
        )
    total_fits = int(feature_records["fit_id"].nunique())
    maximum_rank = max(float(feature_records["rank"].max()), 1.0)

    rows = []
    for feature, group in feature_records.groupby("feature", sort=True):
        importance_column = (
            "normalized_importance"
            if "normalized_importance" in group.columns
            else "importance"
        )
        included = group["importance"].gt(0)
        signed = group.loc[included, "signed_value"]
        sign_consistency = float("nan")
        if not signed.empty:
            positive = int(signed.gt(0).sum())
            negative = int(signed.lt(0).sum())
            sign_consistency = max(positive, negative) / len(signed)
        rank_sd = float(group["rank"].std(ddof=0))
        rows.append(
            {
                "feature": feature,
                "mean_importance": float(group[importance_column].mean()),
                "inclusion_frequency": float(included.sum() / total_fits),
                "sign_consistency": sign_consistency,
                "mean_rank": float(group["rank"].mean()),
                "rank_standard_deviation": rank_sd,
                "rank_stability": float(max(0.0, 1.0 - rank_sd / maximum_rank)),
                "n_fits": int(group["fit_id"].nunique()),
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values(["mean_rank", "feature"], kind="stable")
        .reset_index(drop=True)
    )


def _metric_function(
    outcome_type: OutcomeType,
) -> Callable[[pd.DataFrame], dict[str, float]]:
    """Return a DataFrame-facing metric function for bootstrap evaluation."""
    if outcome_type is OutcomeType.BINARY:
        return lambda frame: classification_metrics(
            frame["y_true"], frame["y_probability"], frame["y_pred"]
        )
    return lambda frame: regression_metrics(frame["y_true"], frame["y_pred"])


def _validated_numeric_pair(
    first: pd.Series | np.ndarray,
    second: pd.Series | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate two equal-length finite numeric vectors."""
    left = np.asarray(first, dtype=float)
    right = np.asarray(second, dtype=float)
    if left.ndim != 1 or right.ndim != 1 or len(left) != len(right) or len(left) == 0:
        raise ValueError("Metric inputs must be non-empty one-dimensional pairs.")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("Metric inputs must contain only finite values.")
    return left, right
