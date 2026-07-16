# PPMI-Pipeline

PPMI-Pipeline is a reproducible Python workflow for modelling longitudinal clinical progression in the Parkinson's Progression Markers Initiative (PPMI) curated data export. It models post-baseline change in the Montreal Cognitive Assessment (MoCA) and MDS-UPDRS Part III motor score, with patient-level data splitting, clinical error metrics, residual diagnostics, and SHAP-based feature attribution.

## Statement of need

Longitudinal Parkinson's disease studies require analysis that respects the fact that multiple visits belong to the same participant. Common row-level splits can place observations from one person in both training and test sets, leading to overly optimistic performance estimates. PPMI-Pipeline provides a compact, inspectable implementation that:

* identifies a configurable BL/V01 baseline for each participant;
* constructs MoCA and UPDRS-III change-from-baseline endpoints;
* uses `PATNO`-grouped train/test splitting to prevent participant leakage;
* performs median imputation inside the training pipeline; and
* records reproducible metrics and interpretability artifacts for each run.

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

1. Copy `config.json` and set `data_path` to the local PPMI `.xlsx`, `.xls`, or `.csv` export. The default configuration expects the worksheet named `20260511`.
2. Confirm the export contains `PATNO`, `EVENT_ID`, `visit_date`, `moca`, and `updrs3_score`.
3. Run the primary entry point:

```bash
python main.py
```

Each run creates timestamped files in `results/`:

* a metrics CSV containing R², MAE, and RMSE for each endpoint;
* a text run log containing the timestamp, configuration, input path, and performance metrics;
* residual-distribution histograms; and
* SHAP summary plots for feature attribution.

To use a different experiment configuration without changing the default file:

```bash
PPMI_CONFIG_PATH=path/to/experiment_config.json python main.py
```

On Windows PowerShell, use:

```powershell
$env:PPMI_CONFIG_PATH = 'path\to\experiment_config.json'
python main.py
```

## Running with Dummy Data

To test the pipeline without PPMI data, run:

```bash
python tests/generate_dummy_data.py
```

The repository includes `tests/dummy_ppmi.csv`, containing 50 synthetic
patients observed at BL, V01, and V02. To run the pipeline with sample data,
use:

```bash
python main.py --config config.example.json
```

## Methodology

The pipeline first performs a header-only schema validation before loading the full PPMI export. It normalizes identifiers and visit dates, then sorts records by participant and date. For each endpoint, it selects the first usable BL or V01 value per participant as baseline, joins that baseline to later visits, and defines the outcome as the later endpoint value minus its baseline value.

Predictors consist of the endpoint's baseline value, baseline age, sex, education, disease duration, UPSIT score, and continuous days since baseline. Missing predictors are median-imputed within a scikit-learn pipeline fitted only to training data. A `RandomForestRegressor` is evaluated with `GroupShuffleSplit`, grouping by `PATNO`, so no participant contributes records to both the training and test partitions.

The reported metrics are mean absolute error (MAE), root mean squared error (RMSE), and coefficient of determination (R²). For clinical interpretability, the pipeline saves a residual histogram and a TreeSHAP summary plot for each endpoint.

## Testing

Run the lightweight schema and longitudinal-alignment tests with:

```bash
python -m unittest discover -s tests -v
```

## Citation

See [CITATION.cff](CITATION.cff) for software citation metadata.

## PPMI Compliance

This project utilizes data from the Parkinson's Progression Markers Initiative (PPMI) database.

- **Data Download Date:** 2026-07-15
- **PPMI Database RRID:** SCR_006431

**Disclaimer:**
As the PPMI database is always evolving, it is possible that the code may not work if the database has changed since the date the code was created.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).
