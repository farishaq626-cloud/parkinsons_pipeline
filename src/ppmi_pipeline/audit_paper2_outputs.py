"""Audit persisted Paper 2 outputs without exposing participant identifiers."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .paper2_config import load_paper2_config


def audit_paper2_outputs(
    metadata_path: str | Path = "outputs/paper2_master_metadata.json",
    output_path: str | Path = "outputs/paper2_integrity_audit.json",
) -> dict[str, Any]:
    """Verify provenance, patient isolation, and OOF coverage for a batch.

    Args:
        metadata_path: Master batch metadata JSON produced by the runner.
        output_path: Aggregate-only JSON destination for the audit report.

    Returns:
        Aggregate integrity report containing no participant identifiers.

    Raises:
        FileNotFoundError: If a declared input or run artifact is absent.
        ValueError: If hashes, schemas, assignments, or OOF coverage disagree.
    """
    metadata_file = Path(metadata_path).resolve()
    if not metadata_file.is_file():
        raise FileNotFoundError(f"Master metadata file was not found: {metadata_file}")
    metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    master_path = Path(metadata["master_summary_path"]).resolve()
    if not master_path.is_file():
        raise FileNotFoundError(f"Master summary file was not found: {master_path}")
    if _sha256(master_path) != metadata["master_summary_sha256"]:
        raise ValueError("Master summary SHA-256 does not match its metadata.")
    master = pd.read_csv(master_path)
    forbidden = {"PATNO", "patient_id"}.intersection(master.columns)
    if forbidden:
        raise ValueError(
            "Master summary contains participant-level identifier columns."
        )

    config_directory = Path(metadata["config_directory"]).resolve()
    run_reports: dict[str, dict[str, Any]] = {}
    source_hashes: set[str] = set()
    total_fits = 0
    total_oof_rows = 0
    total_specifications = 0

    for filename in metadata["config_order"]:
        experiment = Path(filename).stem
        config = load_paper2_config(config_directory / filename)
        expected_hash = metadata["config_sha256"][experiment]
        if config.sha256 != expected_hash:
            raise ValueError(
                f"Resolved YAML hash mismatch for experiment {experiment}."
            )
        run_directory = Path(metadata["run_directories"][experiment]).resolve()
        report = _audit_run(
            experiment,
            run_directory,
            expected_hash,
            metadata["database_versions"][experiment],
            int(metadata["random_seeds"][experiment]),
        )
        run_reports[experiment] = report
        source_hashes.add(report["source_file_sha256"])
        total_fits += int(report["outer_fits_audited"])
        total_oof_rows += int(report["oof_rows_audited"])
        total_specifications += int(report["specifications_audited"])

    if len(source_hashes) != 1:
        raise ValueError("Experiment runs do not share one source-file SHA-256.")
    master_experiments = set(master["batch_experiment"].astype(str))
    if master_experiments != set(run_reports):
        raise ValueError("Master summary experiment names do not match run metadata.")
    master_specifications = int(master["specification_key"].nunique())
    if master_specifications != total_specifications:
        raise ValueError(
            "Master specification count does not match persisted run manifests."
        )

    report = {
        "artifact_type": "paper2_aggregate_integrity_audit",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "status": "PASS",
        "contains_patient_level_data": False,
        "master_summary_path": str(master_path),
        "master_summary_sha256": metadata["master_summary_sha256"],
        "source_file_sha256": next(iter(source_hashes)),
        "database_versions": metadata["database_versions"],
        "config_sha256": metadata["config_sha256"],
        "random_seeds": metadata["random_seeds"],
        "master_metric_rows": int(len(master)),
        "specifications_audited": total_specifications,
        "outer_fits_audited": total_fits,
        "oof_rows_audited": total_oof_rows,
        "patient_overlap_count": 0,
        "oof_assignment_mismatch_count": 0,
        "runs": run_reports,
    }
    destination = Path(output_path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(destination)
    return report


def _audit_run(
    experiment: str,
    run_directory: Path,
    expected_config_hash: str,
    expected_database_version: str,
    expected_seed: int,
) -> dict[str, Any]:
    """Audit one complete experiment run and return aggregate counts."""
    required_files = {
        "provenance": run_directory / "provenance.json",
        "resolved_config": run_directory / "resolved_config.yaml",
        "assignments": run_directory / "fold_assignments.csv",
        "oof": run_directory / "oof_predictions.csv",
        "manifest": run_directory / "specification_manifest.csv",
    }
    missing_files = [
        str(path) for path in required_files.values() if not path.is_file()
    ]
    if missing_files:
        raise FileNotFoundError(
            f"Experiment {experiment} is missing required run artifacts."
        )
    provenance = json.loads(required_files["provenance"].read_text(encoding="utf-8"))
    resolved = yaml.safe_load(
        required_files["resolved_config"].read_text(encoding="utf-8")
    )
    if provenance.get("resolved_config_sha256") != expected_config_hash:
        raise ValueError(f"Provenance hash mismatch for experiment {experiment}.")
    if provenance.get("database_version") != expected_database_version:
        raise ValueError(f"Database version mismatch for experiment {experiment}.")
    if int(provenance.get("random_seed")) != expected_seed:
        raise ValueError(f"Random-seed mismatch for experiment {experiment}.")

    assignments = pd.read_csv(required_files["assignments"])
    oof = pd.read_csv(required_files["oof"])
    manifest = pd.read_csv(required_files["manifest"])
    _require_columns(
        assignments,
        {
            "configuration_id",
            "fit_id",
            "repeat",
            "outer_fold",
            "partition",
            "patient_id",
        },
        f"{experiment} fold assignments",
    )
    _require_columns(
        oof,
        {"configuration_id", "fit_id", "repeat", "outer_fold", "patient_id"},
        f"{experiment} OOF predictions",
    )
    _require_columns(manifest, {"configuration_id"}, f"{experiment} manifest")
    if assignments["patient_id"].isna().any() or oof["patient_id"].isna().any():
        raise ValueError(f"Experiment {experiment} contains missing patient IDs.")

    specification_ids = set(manifest["configuration_id"].astype(str))
    if specification_ids != set(assignments["configuration_id"].astype(str)):
        raise ValueError(f"Assignment specifications disagree for {experiment}.")
    if specification_ids != set(oof["configuration_id"].astype(str)):
        raise ValueError(f"OOF specifications disagree for {experiment}.")
    if int(provenance["specification_count"]) != len(specification_ids):
        raise ValueError(f"Provenance specification count disagrees for {experiment}.")

    fits = 0
    for _fit_id, group in assignments.groupby("fit_id", sort=False):
        train = set(group.loc[group["partition"].eq("train"), "patient_id"])
        test = set(group.loc[group["partition"].eq("test"), "patient_id"])
        if not train or not test:
            raise ValueError(f"Experiment {experiment} has an empty fold partition.")
        if train.intersection(test):
            raise ValueError(f"Patient overlap detected in experiment {experiment}.")
        fits += 1

    test_assignments = assignments.loc[assignments["partition"].eq("test")].copy()
    test_key = ["configuration_id", "repeat", "patient_id"]
    if test_assignments.duplicated(test_key).any():
        raise ValueError(
            f"A patient has multiple outer test folds in one repeat for {experiment}."
        )
    oof_key = ["configuration_id", "repeat", "patient_id"]
    if oof.duplicated(oof_key).any():
        raise ValueError(f"Duplicate OOF prediction detected for {experiment}.")
    assignment_keys = [
        "configuration_id",
        "fit_id",
        "repeat",
        "outer_fold",
        "patient_id",
    ]
    expected = test_assignments[assignment_keys].sort_values(assignment_keys)
    observed = oof[assignment_keys].sort_values(assignment_keys)
    if not expected.reset_index(drop=True).equals(observed.reset_index(drop=True)):
        raise ValueError(f"OOF rows do not equal test assignments for {experiment}.")

    outer_splits = int(resolved["validation"]["outer_splits"])
    outer_repeats = int(resolved["validation"]["outer_repeats"])
    expected_fits = len(specification_ids) * outer_splits * outer_repeats
    if fits != expected_fits:
        raise ValueError(f"Outer-fit count disagrees for experiment {experiment}.")
    source_hash = str(provenance.get("source_file_sha256", ""))
    if len(source_hash) != 64:
        raise ValueError(f"Source-file hash is missing or invalid for {experiment}.")

    return {
        "status": "PASS",
        "run_directory": str(run_directory),
        "resolved_config_sha256": expected_config_hash,
        "source_file_sha256": source_hash,
        "specifications_audited": len(specification_ids),
        "outer_fits_audited": fits,
        "outer_splits": outer_splits,
        "outer_repeats": outer_repeats,
        "oof_rows_audited": int(len(oof)),
        "patient_overlap_count": 0,
        "oof_assignment_mismatch_count": 0,
    }


def _require_columns(table: pd.DataFrame, required: set[str], label: str) -> None:
    """Raise a schema error without exposing table values."""
    missing = sorted(required.difference(table.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {', '.join(missing)}")


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return a file SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Audit Paper 2 provenance, patient isolation, and OOF coverage."
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("outputs/paper2_master_metadata.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/paper2_integrity_audit.json"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the aggregate integrity audit."""
    arguments = build_parser().parse_args(argv)
    report = audit_paper2_outputs(arguments.metadata, arguments.output)
    print(
        "Paper 2 integrity audit PASS: "
        f"{report['specifications_audited']} specifications, "
        f"{report['outer_fits_audited']} outer fits, zero patient overlap."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
