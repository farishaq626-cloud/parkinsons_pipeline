"""Synthetic longitudinal PPMI-like data and dry-run configuration generation."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .paper2_config import load_paper2_config

LOGGER = logging.getLogger("ppmi_pipeline.synthetic_generator")

PAPER2_BATCH_CONFIGS = (
    "primary_analysis.yaml",
    "horizon_sensitivity.yaml",
    "missingness_sensitivity.yaml",
    "modality_ablation.yaml",
)


@dataclass(frozen=True, slots=True)
class SyntheticPPMIConfig:
    """Control generation of a non-clinical longitudinal fixture.

    Args:
        n_patients: Total synthetic participants before PD-only filtering.
        missingness_rate: Independent missingness probability for predictors.
        control_fraction: Fraction labelled as synthetic healthy controls.
        follow_up_dropout_rate: Per-visit probability of omitting a follow-up.
        temporal_gaps_days: Nominal follow-up days after baseline.
        temporal_jitter_days: Maximum absolute visit-date jitter in days.
        random_seed: Seed controlling every stochastic operation.
    """

    n_patients: int = 120
    missingness_rate: float = 0.1
    control_fraction: float = 0.1
    follow_up_dropout_rate: float = 0.05
    temporal_gaps_days: tuple[int, ...] = (365, 730, 1095)
    temporal_jitter_days: int = 30
    random_seed: int = 42

    def __post_init__(self) -> None:
        """Reject settings that cannot support a stable patient-level dry run.

        Raises:
            ValueError: If counts, probabilities, gaps, or the seed are invalid.
        """
        if (
            isinstance(self.n_patients, bool)
            or not isinstance(self.n_patients, int)
            or self.n_patients < 20
        ):
            raise ValueError("n_patients must be an integer of at least 20.")
        for name, value in (
            ("missingness_rate", self.missingness_rate),
            ("control_fraction", self.control_fraction),
            ("follow_up_dropout_rate", self.follow_up_dropout_rate),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not 0 <= value < 1
            ):
                raise ValueError(f"{name} must be in the interval [0, 1).")
        gaps = tuple(int(value) for value in self.temporal_gaps_days)
        if not gaps or any(value <= 0 for value in gaps):
            raise ValueError("temporal_gaps_days must contain positive values.")
        if tuple(sorted(gaps)) != gaps or len(set(gaps)) != len(gaps):
            raise ValueError("temporal_gaps_days must be unique and increasing.")
        object.__setattr__(self, "temporal_gaps_days", gaps)
        if (
            isinstance(self.temporal_jitter_days, bool)
            or not isinstance(self.temporal_jitter_days, int)
            or self.temporal_jitter_days < 0
        ):
            raise ValueError("temporal_jitter_days must be non-negative.")
        if isinstance(self.random_seed, bool) or not isinstance(self.random_seed, int):
            raise ValueError("random_seed must be an integer.")

    def fingerprint(self) -> str:
        """Return a stable SHA-256 hash of the generator settings.

        Returns:
            Hexadecimal configuration hash.
        """
        payload = json.dumps(
            asdict(self), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


@dataclass(slots=True)
class SyntheticPPMIResult:
    """Hold generated flat and modality-specific synthetic tables."""

    flat: pd.DataFrame
    clinical: pd.DataFrame
    biofluid: pd.DataFrame
    imaging: pd.DataFrame
    genetics: pd.DataFrame
    metadata: dict[str, Any]

    def write(self, output_dir: str | Path) -> dict[str, Path]:
        """Serialize all synthetic tables and their provenance.

        Args:
            output_dir: Destination directory for CSV and JSON artifacts.

        Returns:
            Mapping of artifact names to resolved paths.
        """
        destination = Path(output_dir).resolve()
        destination.mkdir(parents=True, exist_ok=True)
        paths = {
            "flat": destination / "synthetic_ppmi_flat.csv",
            "clinical": destination / "synthetic_ppmi_clinical.csv",
            "biofluid": destination / "synthetic_ppmi_biofluid.csv",
            "imaging": destination / "synthetic_ppmi_imaging.csv",
            "genetics": destination / "synthetic_ppmi_genetics.csv",
            "metadata": destination / "synthetic_generation_metadata.json",
        }
        for name, table in (
            ("flat", self.flat),
            ("clinical", self.clinical),
            ("biofluid", self.biofluid),
            ("imaging", self.imaging),
            ("genetics", self.genetics),
        ):
            _atomic_csv_write(table, paths[name])
        metadata = {
            **self.metadata,
            "files": {
                name: {
                    "path": str(path),
                    "sha256": _file_sha256(path),
                    "rows": int(getattr(self, name).shape[0]),
                }
                for name, path in paths.items()
                if name != "metadata"
            },
        }
        _atomic_text_write(
            paths["metadata"],
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        )
        return paths


def generate_synthetic_ppmi(
    config: SyntheticPPMIConfig | None = None,
) -> SyntheticPPMIResult:
    """Generate a longitudinal synthetic fixture matching Paper 2 contracts.

    The data are entirely simulated and contain no source PPMI records. Core
    patient, visit, date, diagnostic-group, medication-state, and endpoint fields
    remain complete. Configured missingness is applied only to candidate
    predictors so fold-local imputation paths can be exercised safely.

    Args:
        config: Synthetic generation settings.

    Returns:
        Flat integrated table, domain tables, and generation provenance.
    """
    settings = config or SyntheticPPMIConfig()
    rng = np.random.default_rng(settings.random_seed)
    patient_ids = np.arange(100_000, 100_000 + settings.n_patients)
    controls = rng.random(settings.n_patients) < settings.control_fraction
    latent = rng.normal(0, 1, settings.n_patients)
    age = np.clip(rng.normal(64, 8, settings.n_patients), 35, 85)
    sex = rng.integers(0, 2, settings.n_patients)
    education = np.clip(np.rint(rng.normal(15, 3, settings.n_patients)), 6, 24)
    duration = np.clip(rng.gamma(2.0, 1.2, settings.n_patients), 0.1, 12)
    baseline_motor = np.clip(20 + 5 * latent + 0.12 * (age - 64), 1, 60)
    baseline_motor = np.where(controls, baseline_motor * 0.35, baseline_motor)
    annual_change = np.clip(2.5 + 1.7 * latent + 0.025 * (age - 64), -1, 7)
    annual_change = np.where(controls, annual_change * 0.25, annual_change)
    moca = np.clip(27 - 0.8 * latent - 0.05 * (age - 64), 12, 30)
    upsit = np.clip(24 - 3.5 * latent - 2.5 * (~controls), 1, 40)
    csf_alpha = np.clip(
        1800 - 140 * latent + rng.normal(0, 120, settings.n_patients), 500, 3500
    )
    csf_abeta = np.clip(
        900 - 70 * latent + rng.normal(0, 80, settings.n_patients), 250, 1800
    )
    dat_spect = np.clip(2.0 - 0.28 * latent - 0.45 * (~controls), 0.25, 3.5)
    gba_variant = (rng.random(settings.n_patients) < 0.08).astype(float)
    lrrk2_variant = (rng.random(settings.n_patients) < 0.05).astype(float)
    apoe_e4_count = rng.choice(
        [0.0, 1.0, 2.0], settings.n_patients, p=[0.72, 0.25, 0.03]
    )

    patient_features = pd.DataFrame(
        {
            "PATNO": patient_ids,
            "diagnostic_group": np.where(controls, "HC", "PD"),
            "age": age,
            "SEX": sex.astype(float),
            "EDUCYRS": education.astype(float),
            "duration": duration,
            "moca": moca,
            "upsit": upsit,
            "csf_alpha_synuclein": csf_alpha,
            "csf_abeta42": csf_abeta,
            "dat_spect_sbr": dat_spect,
            "gba_variant": gba_variant,
            "lrrk2_variant": lrrk2_variant,
            "apoe_e4_count": apoe_e4_count,
        }
    )
    predictor_columns = [
        column
        for column in patient_features.columns
        if column not in {"PATNO", "diagnostic_group"}
    ]
    missing_mask = (
        rng.random((settings.n_patients, len(predictor_columns)))
        < settings.missingness_rate
    )
    patient_features.loc[:, predictor_columns] = patient_features[
        predictor_columns
    ].mask(missing_mask)

    baseline_dates = pd.Timestamp("2020-01-01") + pd.to_timedelta(
        rng.integers(0, 180, settings.n_patients), unit="D"
    )
    longitudinal_rows: list[dict[str, Any]] = []
    visit_plan = [
        ("BL", 0),
        *(
            (f"V{index:02d}", gap)
            for index, gap in enumerate(settings.temporal_gaps_days, start=1)
        ),
    ]
    for patient_index, patient_id in enumerate(patient_ids):
        features = patient_features.iloc[patient_index].to_dict()
        for event_id, nominal_gap in visit_plan:
            if nominal_gap and rng.random() < settings.follow_up_dropout_rate:
                continue
            jitter = (
                0
                if nominal_gap == 0
                else int(
                    rng.integers(
                        -settings.temporal_jitter_days,
                        settings.temporal_jitter_days + 1,
                    )
                )
            )
            elapsed_days = max(1, nominal_gap + jitter) if nominal_gap else 0
            years = elapsed_days / 365.25
            score = np.clip(
                baseline_motor[patient_index]
                + annual_change[patient_index] * years
                + (0 if nominal_gap == 0 else rng.normal(0, 1.5)),
                0,
                132,
            )
            longitudinal_rows.append(
                {
                    **features,
                    "PATNO": patient_id,
                    "EVENT_ID": event_id,
                    "visit_date": (
                        baseline_dates[patient_index] + pd.Timedelta(days=elapsed_days)
                    )
                    .date()
                    .isoformat(),
                    "updrs3_score": round(float(score), 3),
                    "medication_state": "OFF",
                }
            )
    flat = (
        pd.DataFrame(longitudinal_rows)
        .sort_values(["PATNO", "visit_date"], kind="stable")
        .reset_index(drop=True)
    )
    clinical_columns = [
        "PATNO",
        "EVENT_ID",
        "visit_date",
        "diagnostic_group",
        "updrs3_score",
        "medication_state",
        "age",
        "SEX",
        "EDUCYRS",
        "duration",
        "moca",
        "upsit",
    ]
    biofluid_columns = [
        "PATNO",
        "EVENT_ID",
        "visit_date",
        "csf_alpha_synuclein",
        "csf_abeta42",
    ]
    imaging_columns = [
        "PATNO",
        "EVENT_ID",
        "visit_date",
        "dat_spect_sbr",
    ]
    genetics_columns = [
        "PATNO",
        "gba_variant",
        "lrrk2_variant",
        "apoe_e4_count",
    ]
    metadata = {
        "artifact_type": "synthetic_non_clinical_fixture",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "generator_config": asdict(settings),
        "generator_config_sha256": settings.fingerprint(),
        "patients": int(flat["PATNO"].nunique()),
        "records": int(len(flat)),
        "contains_real_ppmi_data": False,
    }
    return SyntheticPPMIResult(
        flat=flat,
        clinical=flat[clinical_columns].copy(),
        biofluid=flat[biofluid_columns].copy(),
        imaging=flat[imaging_columns].copy(),
        genetics=flat[genetics_columns].drop_duplicates("PATNO").copy(),
        metadata=metadata,
    )


def write_dry_run_configurations(
    data_path: str | Path,
    template_dir: str | Path,
    output_dir: str | Path,
    random_seed: int = 42,
) -> dict[str, Path]:
    """Write fully resolved, lightweight YAML configurations for dry runs.

    Args:
        data_path: Generated flat synthetic CSV.
        template_dir: Directory containing the four Paper 2 YAML templates.
        output_dir: Destination for resolved dry-run YAML files and run outputs.
        random_seed: Reproducible validation seed.

    Returns:
        Mapping from experiment name to validated YAML path.

    Raises:
        FileNotFoundError: If a source YAML template is missing.
        ValueError: If a source YAML document is not a mapping.
    """
    source = Path(template_dir).resolve()
    destination = Path(output_dir).resolve()
    config_destination = destination / "configs"
    run_destination = destination / "runs"
    config_destination.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    config_hashes: dict[str, str] = {}
    for filename in PAPER2_BATCH_CONFIGS:
        values = _resolve_yaml_template(source / filename, visited=set())
        values["study"]["name"] = f"{Path(filename).stem}_synthetic_dry_run"
        values["study"]["database_version"] = "synthetic-ppmi-fixture-v1"
        values["study"]["random_seed"] = random_seed
        values["data"]["path"] = str(Path(data_path).resolve())
        values["data"]["sheet_name"] = 0
        values["data"]["column_map"] = {
            "PATNO": "PATNO",
            "EVENT_ID": "EVENT_ID",
            "VISIT_DATE": "visit_date",
            "DIAGNOSTIC_GROUP": "diagnostic_group",
            "SCORE": "updrs3_score",
            "MEDICATION_STATE": "medication_state",
        }
        values["data"].pop("column_constants", None)
        values["cohort"]["eligible_groups"] = ["PD"]
        values["feature_blocks"] = {
            "demographics_disease_history": ["age", "SEX", "EDUCYRS", "duration"],
            "baseline_motor": [],
            "cognition_neuropsychology": ["moca"],
            "olfaction_sleep_autonomic": ["upsit"],
            "biofluid_biomarkers": ["csf_alpha_synuclein", "csf_abeta42"],
            "imaging_dat_spect": ["dat_spect_sbr"],
            "genetic_variables": [
                "gba_variant",
                "lrrk2_variant",
                "apoe_e4_count",
            ],
        }
        values["validation"].update(
            {
                "outer_splits": 2,
                "outer_repeats": 2,
                "inner_splits": 2,
                "threshold_selection": "fixed",
                "bootstrap_resamples": 50,
            }
        )
        values["models"] = {
            "elasticnet": {
                "binary_parameter_grid": {"C": [1.0], "l1_ratio": [0.5]},
                "continuous_parameter_grid": {
                    "alpha": [0.01],
                    "l1_ratio": [0.5],
                },
            }
        }
        values["output"]["directory"] = str(run_destination / Path(filename).stem)
        config_path = config_destination / filename
        _atomic_text_write(config_path, yaml.safe_dump(values, sort_keys=False))
        resolved = load_paper2_config(config_path)
        experiment_name = Path(filename).stem
        written[experiment_name] = config_path
        config_hashes[experiment_name] = resolved.sha256
    config_metadata = {
        "artifact_type": "paper2_synthetic_dry_run_configurations",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "data_path": str(Path(data_path).resolve()),
        "data_sha256": _file_sha256(Path(data_path).resolve()),
        "random_seed": random_seed,
        "config_sha256": config_hashes,
        "contains_real_ppmi_data": False,
    }
    _atomic_text_write(
        destination / "dry_run_configuration_metadata.json",
        json.dumps(config_metadata, indent=2, sort_keys=True) + "\n",
    )
    return written


def build_parser() -> argparse.ArgumentParser:
    """Build the synthetic-data command-line parser.

    Returns:
        Configured parser.
    """
    parser = argparse.ArgumentParser(
        description="Generate synthetic PPMI-like Paper 2 dry-run artifacts."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/synthetic"))
    parser.add_argument("--patients", type=int, default=120)
    parser.add_argument("--missingness-rate", type=float, default=0.1)
    parser.add_argument("--control-fraction", type=float, default=0.1)
    parser.add_argument("--dropout-rate", type=float, default=0.05)
    parser.add_argument("--temporal-jitter-days", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--template-dir", type=Path, default=Path("configs/paper2"))
    parser.add_argument(
        "--write-batch-configs",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Generate and serialize synthetic dry-run inputs.

    Args:
        argv: Optional command-line arguments.

    Returns:
        Zero after successful generation.
    """
    arguments = build_parser().parse_args(argv)
    settings = SyntheticPPMIConfig(
        n_patients=arguments.patients,
        missingness_rate=arguments.missingness_rate,
        control_fraction=arguments.control_fraction,
        follow_up_dropout_rate=arguments.dropout_rate,
        temporal_jitter_days=arguments.temporal_jitter_days,
        random_seed=arguments.seed,
    )
    paths = generate_synthetic_ppmi(settings).write(arguments.output_dir)
    if arguments.write_batch_configs:
        write_dry_run_configurations(
            paths["flat"],
            arguments.template_dir,
            Path(arguments.output_dir) / "batch",
            random_seed=arguments.seed,
        )
    LOGGER.info("Synthetic Paper 2 artifacts written to %s", arguments.output_dir)
    return 0


def _read_yaml_mapping(path: Path) -> dict[str, Any]:
    """Read one YAML mapping without applying experiment validation."""
    if not path.exists():
        raise FileNotFoundError(f"Paper 2 YAML template was not found: {path}")
    values = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(values, dict):
        raise ValueError(f"YAML template must contain a mapping: {path}")
    return values


def _resolve_yaml_template(path: Path, visited: set[Path]) -> dict[str, Any]:
    """Resolve the complete relative ``extends`` chain for one YAML template."""
    resolved_path = path.resolve()
    if resolved_path in visited:
        raise ValueError(f"Circular synthetic YAML inheritance at {resolved_path}.")
    visited.add(resolved_path)
    values = _read_yaml_mapping(resolved_path)
    base_reference = values.pop("extends", None)
    if base_reference is None:
        return values
    base = _resolve_yaml_template(
        (resolved_path.parent / str(base_reference)).resolve(),
        visited,
    )
    return _deep_merge(base, values)


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge YAML mappings while replacing lists and scalars."""
    merged: dict[str, Any] = json.loads(json.dumps(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _atomic_csv_write(table: pd.DataFrame, path: Path) -> None:
    """Write a CSV through a same-directory temporary file."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    table.to_csv(temporary, index=False)
    temporary.replace(path)


def _atomic_text_write(path: Path, content: str) -> None:
    """Write text through a same-directory temporary file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 hash of one generated artifact."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
