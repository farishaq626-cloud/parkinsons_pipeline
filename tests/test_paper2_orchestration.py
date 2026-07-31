"""Tests for Paper 2 synthetic, batch, and specification-plot automation."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from ppmi_pipeline.paper2_config import (
    PAPER2_FEATURE_BLOCKS,
    ResolvedPaper2Config,
    expand_experiment_grid,
    load_paper2_config,
)
from ppmi_pipeline.paper2_experiment import (
    Paper2ExperimentResult,
    Paper2ExperimentRunner,
)
from ppmi_pipeline.plot_specification import SpecificationCurvePlotter
from ppmi_pipeline.run_all_experiments import (
    build_experiment_summary,
    run_all_experiments,
)
from ppmi_pipeline.synthetic_generator import (
    PAPER2_BATCH_CONFIGS,
    SyntheticPPMIConfig,
    generate_synthetic_ppmi,
    write_dry_run_configurations,
)


class Paper2OrchestrationTests(unittest.TestCase):
    """Verify synthetic execution, batch aggregation, and figure generation."""

    def test_synthetic_fixture_supports_a_real_end_to_end_dry_run(self) -> None:
        settings = SyntheticPPMIConfig(
            n_patients=60,
            missingness_rate=0.1,
            control_fraction=0.0,
            follow_up_dropout_rate=0.0,
            temporal_jitter_days=15,
            random_seed=17,
        )
        generated = generate_synthetic_ppmi(settings)
        required = {
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
            "csf_alpha_synuclein",
            "csf_abeta42",
            "dat_spect_sbr",
            "gba_variant",
            "lrrk2_variant",
            "apoe_e4_count",
        }
        self.assertTrue(required.issubset(generated.flat.columns))
        self.assertFalse(generated.metadata["contains_real_ppmi_data"])
        self.assertFalse(
            generated.flat[
                [
                    "PATNO",
                    "EVENT_ID",
                    "visit_date",
                    "diagnostic_group",
                    "updrs3_score",
                    "medication_state",
                ]
            ]
            .isna()
            .any()
            .any()
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = generated.write(root / "synthetic")
            configs = write_dry_run_configurations(
                paths["flat"],
                Path(__file__).parents[1] / "configs" / "paper2",
                root / "batch",
                random_seed=17,
            )
            self.assertEqual(
                set(configs), {Path(name).stem for name in PAPER2_BATCH_CONFIGS}
            )
            expected_counts = {
                "primary_analysis": 2,
                "horizon_sensitivity": 18,
                "missingness_sensitivity": 6,
                "modality_ablation": 14,
            }
            actual_counts = {
                name: len(expand_experiment_grid(load_paper2_config(path)))
                for name, path in configs.items()
            }
            self.assertEqual(actual_counts, expected_counts)
            primary = load_paper2_config(configs["primary_analysis"])
            result = Paper2ExperimentRunner(primary).run_from_configured_path()

            self.assertFalse(result.oof_predictions.empty)
            self.assertEqual(
                result.fold_assignments.groupby("fit_id")
                .apply(
                    _patient_overlap_count,
                    include_groups=False,
                )
                .max(),
                0,
            )
            metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
            self.assertFalse(metadata["contains_real_ppmi_data"])
            self.assertEqual(len(metadata["files"]["flat"]["sha256"]), 64)

    def test_batch_runner_executes_configs_and_writes_master_summary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_names = ("first.yaml", "second.yaml")
            for name in config_names:
                (root / name).write_text("study: test\n", encoding="utf-8")
            configs = [
                _fake_config(root / config_names[0], "a" * 64),
                _fake_config(root / config_names[1], "b" * 64),
            ]
            results = [
                _fake_result(root / "run-a", "config-a", "binary", "auroc", 0.72),
                _fake_result(root / "run-b", "config-b", "continuous", "r2", 0.31),
            ]
            with (
                patch(
                    "ppmi_pipeline.run_all_experiments.load_paper2_config",
                    side_effect=configs,
                ) as load_mock,
                patch(
                    "ppmi_pipeline.run_all_experiments.Paper2ExperimentRunner"
                ) as runner_mock,
            ):
                runner_mock.return_value.run_from_configured_path.side_effect = results
                batch = run_all_experiments(
                    root,
                    root / "outputs" / "paper2_master_summary.csv",
                    config_names=config_names,
                )

            self.assertEqual(load_mock.call_count, 2)
            self.assertEqual(runner_mock.call_count, 2)
            self.assertEqual(len(batch.master_summary), 2)
            self.assertTrue(batch.master_summary_path.exists())
            self.assertTrue(batch.metadata_path.exists())
            self.assertEqual(
                set(batch.master_summary["metric"]),
                {"auroc", "r2"},
            )
            metadata = json.loads(batch.metadata_path.read_text(encoding="utf-8"))
            self.assertFalse(metadata["contains_patient_level_data"])
            self.assertEqual(len(metadata["master_summary_sha256"]), 64)

    def test_batch_resume_reuses_hash_matched_completed_runs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_names = ("first.yaml", "second.yaml")
            for name in config_names:
                (root / name).write_text("study: test\n", encoding="utf-8")
            configs = [
                _fake_config(root / config_names[0], "a" * 64),
                _fake_config(root / config_names[1], "b" * 64),
            ]
            results = [
                _fake_result(root / "run-a", "config-a", "binary", "auroc", 0.72),
                _fake_result(root / "run-b", "config-b", "continuous", "r2", 0.31),
            ]
            with (
                patch(
                    "ppmi_pipeline.run_all_experiments.load_paper2_config",
                    side_effect=configs,
                ),
                patch(
                    "ppmi_pipeline.run_all_experiments._load_latest_completed_run",
                    side_effect=results,
                ) as reuse_mock,
                patch(
                    "ppmi_pipeline.run_all_experiments.Paper2ExperimentRunner"
                ) as runner_mock,
            ):
                batch = run_all_experiments(
                    root,
                    root / "outputs" / "paper2_master_summary.csv",
                    config_names=config_names,
                    reuse_completed=True,
                )

            self.assertEqual(reuse_mock.call_count, 2)
            runner_mock.assert_not_called()
            self.assertEqual(len(batch.master_summary), 2)

    def test_specification_plot_writes_valid_pdf_png_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = _fake_config(root / "plot.yaml", "c" * 64)
            first = build_experiment_summary(
                "primary",
                config,
                _fake_result(root / "run-1", "one", "binary", "auroc", 0.65),
            )
            second = build_experiment_summary(
                "sensitivity",
                config,
                _fake_result(root / "run-2", "two", "binary", "auroc", 0.78),
            )
            summary = pd.concat([first, second], ignore_index=True)
            source = root / "paper2_master_summary.csv"
            summary.to_csv(source, index=False)

            outputs = SpecificationCurvePlotter.from_csv(source).plot_metric(
                "auroc", root / "figures"
            )

            self.assertEqual(outputs["pdf"].read_bytes()[:4], b"%PDF")
            self.assertEqual(outputs["png"].read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
            metadata = json.loads(outputs["metadata"].read_text(encoding="utf-8"))
            self.assertEqual(metadata["specification_count"], 2)
            self.assertEqual(metadata["ordering"], "estimate_ascending")


def _fake_config(path: Path, sha256: str) -> ResolvedPaper2Config:
    """Create a minimal resolved configuration for orchestration tests."""
    return ResolvedPaper2Config(
        path=path,
        sha256=sha256,
        values={
            "study": {
                "database_version": "synthetic-fixture-v1",
                "random_seed": 42,
            },
            "validation": {
                "split_strategy": "patient_isolated",
                "outer_splits": 2,
                "outer_repeats": 2,
                "inner_splits": 2,
            },
        },
    )


def _fake_result(
    run_directory: Path,
    configuration_id: str,
    outcome_type: str,
    metric: str,
    estimate: float,
) -> Paper2ExperimentResult:
    """Create aggregate-only experiment outputs for batch and plot tests."""
    run_directory.mkdir(parents=True, exist_ok=True)
    task = {
        "name": configuration_id,
        "endpoint": "mds_updrs_iii",
        "outcome_type": outcome_type,
        "horizon_months": 24,
        "tolerance_days": 90,
        "progression_threshold": 5.0 if outcome_type == "binary" else None,
        "medication_state": "OFF",
    }
    blocks = list(PAPER2_FEATURE_BLOCKS)
    return Paper2ExperimentResult(
        run_directory=run_directory,
        provenance={
            "software_version": "2.1.8",
            "source_file_sha256": "d" * 64,
        },
        cohort_flow=pd.DataFrame(
            {
                "configuration_id": [configuration_id],
                "flow_component": ["fixed_horizon"],
                "stage": ["fixed_horizon_follow_up"],
                "patients_retained": [50],
            }
        ),
        fold_metrics=pd.DataFrame(),
        metric_confidence_intervals=pd.DataFrame(
            {
                "configuration_id": [configuration_id],
                "metric": [metric],
                "estimate": [estimate],
                "ci_lower": [estimate - 0.05],
                "ci_upper": [estimate + 0.05],
                "confidence_level": [0.95],
                "valid_resamples": [100],
            }
        ),
        oof_predictions=pd.DataFrame(),
        fold_assignments=pd.DataFrame(),
        feature_records=pd.DataFrame(),
        feature_stability=pd.DataFrame(
            {
                "configuration_id": [configuration_id],
                "feature": ["Baseline_Score"],
                "mean_importance": [1.0],
                "inclusion_frequency": [1.0],
                "sign_consistency": [1.0],
                "rank_stability": [1.0],
            }
        ),
        feature_manifest=pd.DataFrame(),
        specification_manifest=pd.DataFrame(
            {
                "configuration_id": [configuration_id],
                "task_fingerprint": ["e" * 64],
                "task": [json.dumps(task)],
                "feature_blocks": [json.dumps(blocks)],
                "imputation_strategy": ["median"],
                "model_family": ["elasticnet"],
                "model_parameters": [json.dumps({"C": [1.0]})],
            }
        ),
        tuning_results=pd.DataFrame(),
    )


def _patient_overlap_count(assignments: pd.DataFrame) -> int:
    """Count shared patients in one persisted outer-fold assignment."""
    train = set(assignments.loc[assignments["partition"].eq("train"), "patient_id"])
    test = set(assignments.loc[assignments["partition"].eq("test"), "patient_id"])
    return len(train.intersection(test))


if __name__ == "__main__":
    unittest.main()
