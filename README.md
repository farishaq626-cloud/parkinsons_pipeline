# PPMI-Pipeline

## Abstract

Parkinson’s disease exhibits substantial clinical heterogeneity, complicating efforts to forecast individual progression from routine longitudinal assessments. Transparent and reproducible prognostic workflows are therefore needed to transform heterogeneous clinical exports into clinically interpretable modelling datasets while minimising information leakage.

We developed a modular prognostic pipeline for Parkinson’s Progression Markers Initiative (PPMI) clinical data. The pipeline validates and normalises canonical patient, visit, date, cognitive, and motor fields during ingestion. Longitudinal observations are then mapped to fixed-horizon outcomes: the earliest usable baseline visit (`BL`) is identified for each patient, and the follow-up score closest to a prespecified horizon within an allowable visit window is selected. This produces patient-level baseline score, target score, and delta-score records while explicitly reporting attrition caused by unavailable follow-up data. Default settings define a 365-day horizon with a ±90-day tolerance, although these parameters are centrally configurable.

For binary prognostic targets, ElasticNet-regularised logistic regression is evaluated using `GroupKFold` cross-validation. Patient identifiers are mapped to a dedicated `patient_id` grouping field, and every fold is checked to ensure that no patient contributes to both training and testing partitions. The framework reports precision, recall, F1-score, and AUC-ROC per fold, while retaining model coefficients for each split. Feature stability is summarised using mean normalised importance, importance variability, coefficient direction, and the proportion of folds in which a feature remains non-zero. Publication-ready visualisations display the relationship between importance and cross-fold stability and show coefficient patterns across patient-isolated folds.

This methodology prioritises interpretability and methodological validity over predictive discrimination alone. By providing explicit longitudinal-to-fixed-horizon mapping, auditable follow-up attrition, and strict patient-level isolation, the pipeline establishes a reproducible basis for future patient stratification studies and clinically responsible evaluation of candidate prognostic markers.

PPMI-Pipeline is a reproducible Python workflow for fixed-horizon prognostic modelling with Parkinson's Progression Markers Initiative (PPMI) clinical data. It converts longitudinal clinical observations into patient-level progression targets, applies patient-isolated validation, and reports clinically interpretable feature stability.

## Project Enhancements (July 2026)

- **Centralized Configuration:** `config.py` is the single source of truth for runtime paths, fixed-horizon parameters, validation settings, model defaults, and figure settings.
- **Robust Logging:** `logging_config.py` configures structured logging to both the console and `pipeline.log`, supporting traceable pipeline execution and diagnostics.
- **Custom Exception Handling:** `exceptions.py` provides descriptive, project-specific errors for missing files, columns, and invalid configuration.
- **Orchestration:** `main.py` is the single entry point for the fixed-horizon workflow and coordinates ETL, schema harmonisation, target construction, validation, modelling, reporting, and visualisation.

## Pipeline Overview

The codebase is organised as modular, type-hinted components with explicit schema validation, error handling, and reproducible defaults. `etl.py` validates and normalises PPMI exports; `adapter.py` harmonises the configured clinical score and visit date fields; and `data_utils.py` converts long-format observations into patient-level baseline, target, and delta records while reporting follow-up attrition.

`validation.py` applies `GroupKFold` with explicit `patient_id` isolation, `modeling.py` fits ElasticNet logistic-regression baselines and records coefficient stability, and `visualization.py` generates publication-quality stability and coefficient figures. Earlier Random Forest endpoint-delta components are preserved in `legacy/` and are not executed by the canonical pipeline. Together, the active modules support inspectable, reproducible clinical research rather than diagnostic use.

## Statement of need

Longitudinal Parkinson's disease studies require analysis that respects the fact that multiple visits belong to the same participant. Common row-level splits can place observations from one person in both training and test sets, leading to overly optimistic performance estimates. PPMI-Pipeline provides a compact, inspectable implementation that:

* identifies an earliest usable BL baseline and a fixed-horizon follow-up visit;
* constructs auditable baseline, target, delta, and binary progression-label data;
* uses `patient_id`-grouped `GroupKFold` splitting to prevent participant leakage;
* performs median imputation and standardisation inside each training fold; and
* records reproducible fold metrics, coefficients, stability reports, and figures.

The repository is intended for clinical research and method development. It is not a diagnostic device and must not be used as a substitute for clinical judgment.

## Installation

Python 3.10 or later is recommended.

```bash
git clone https://github.com/farishaq626-cloud/parkinsons_pipeline.git
cd parkinsons_pipeline
python -m venv .venv
```

Activate the virtual environment, then install dependencies:

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

Download the appropriate PPMI curated export through the PPMI data portal and place it locally. PPMI data are governed by their own access and data-use agreement and are not redistributed with this repository.

## Quick start

1. Confirm your PPMI export contains the required columns: `PATNO`, `EVENT_ID`, `visit_date`, `moca`, and `updrs3_score`.
2. Update `FIXED_HORIZON_CONFIG` in `config.py` to set `data_path`, `score_column`, the fixed horizon, its tolerance, and the progression threshold.
3. Run the canonical fixed-horizon pipeline:

   ```bash
   python main.py
   ```

## Running with Dummy Data

To regenerate the synthetic test fixture, run:

```bash
python tests/generate_dummy_data.py
```

The repository includes `tests/dummy_ppmi.csv`, containing 50 synthetic
patients observed at BL, V01, and V02. The default `config.py` configuration
uses this fixture; run the complete workflow with:

```bash
python main.py
```

## Methodology

The pipeline first performs a header-only schema validation before loading the full PPMI export. It normalises identifiers and visit dates, then adapts the configured clinical score into the fixed-horizon schema. For each participant, it selects the earliest usable BL record and the valid follow-up observation closest to the configured horizon within the configured tolerance window. The resulting patient-level data contain baseline score, target score, and score change; follow-up attrition is reported explicitly.

Baseline-compatible numeric predictors are median-imputed and standardised within each training fold before ElasticNet logistic regression. A binary progression label is defined from the configured score-change threshold. `GroupKFold` groups strictly by patient, and each fold asserts that training and testing partitions share no participants.

The reported metrics are mean absolute error (MAE), root mean squared error (RMSE), and coefficient of determination (R²). For clinical interpretability, the pipeline saves a residual histogram and a TreeSHAP summary plot for each endpoint.

The preceding Random Forest/TreeSHAP sentence describes the quarantined endpoint-delta workflow in `legacy/`, not the canonical pipeline. The active fixed-horizon workflow reports precision, recall, F1-score, and AUC-ROC; it saves per-fold coefficients, feature-stability summaries, an importance-versus-stability plot, and a cross-fold coefficient heatmap.

## Testing

Run the lightweight schema and longitudinal-alignment tests with:

```bash
python -m unittest discover -s tests -v
```

## Citation

See [CITATION.cff](CITATION.cff) for software citation metadata.

## Documentation

The accompanying manuscript draft, bibliography, and manuscript-specific figures
are located in [manuscript/](manuscript/). The draft source is
[manuscript/paper.md](manuscript/paper.md).

## PPMI Compliance

This project utilizes data from the Parkinson's Progression Markers Initiative (PPMI) database.

- **Data Download Date:** 2026-07-15
- **PPMI Database RRID:** SCR_006431

**Disclaimer:**
As the PPMI database is always evolving, it is possible that the code may not work if the database has changed since the date the code was created.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).
