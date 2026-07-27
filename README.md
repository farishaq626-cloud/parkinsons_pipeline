# PPMI-Pipeline

## Poster Artifacts

This repository serves as the computational artifact for the poster:
"PPMI-Pipeline: A Reproducible Python Framework for Auditable Fixed-Horizon
Longitudinal Modelling".

The current software release is **v2.1.7**. The project archive is
available at [Zenodo (DOI 10.5281/zenodo.21225810)](https://doi.org/10.5281/zenodo.21225810).

### Public Repository Scope

This public repository contains **Technical Artifacts for Reproducibility**:
software, configuration, synthetic tests, validation procedures, and technical
diagnostic outputs. It does **not** publish clinical research conclusions,
patient-level data, identifiable cohort analyses, or unpublished manuscripts.

When generated locally, cohort-flow, validation-audit, and model-performance
figures are written to `results/poster_assets/`. This directory is intentionally
untracked and must be reviewed for clinical interpretation before any release.

## Abstract

PPMI-Pipeline is an open-source Python framework for auditable data engineering
and computational-method validation using longitudinal Parkinson’s Progression
Markers Initiative (PPMI) exports as a reference implementation. It replaces
one-off preprocessing scripts with explicit contracts for ingestion, schema
harmonization, fixed-horizon construction, patient-isolated validation, and
artifact generation.

The ETL layer validates canonical fields before loading and normalizes patient,
visit, score, and date values. A schema adapter then maps the selected score
endpoint into a stable fixed-horizon contract. For each patient, the earliest
usable baseline is paired with the eligible follow-up nearest a configured
horizon, while retained and excluded patient counts are reported explicitly.

`GroupKFold` partitions by patient identifier, and an explicit overlap check
guards every train/test split. The included single-feature ElasticNet logistic
regression is an execution harness: it verifies preprocessing, fitting,
reporting, and figure generation across the complete pipeline. It is not an
optimized clinical predictor and must not be interpreted as a diagnostic or
decision-support model.

The framework prioritizes inspectable software behavior, deterministic data
transformation, automated tests, configuration control, and run provenance.
Synthetic fixtures allow public verification without redistribution of
controlled PPMI data.

## Project Enhancements (July 2026)

- **Centralized Configuration:** `config.py` is the single source of truth for runtime paths, fixed-horizon parameters, validation settings, model defaults, and figure settings.
- **Robust Logging:** `logging_config.py` configures structured logging to both the console and `pipeline.log`, supporting traceable pipeline execution and diagnostics.
- **Custom Exception Handling:** `exceptions.py` provides descriptive, project-specific errors for missing files, columns, and invalid configuration.
- **Orchestration:** `main.py` is the single entry point for the fixed-horizon workflow and coordinates ETL, schema harmonisation, target construction, validation, modelling, reporting, and visualisation.

## Pipeline Overview

The codebase is organised as modular, type-hinted components with explicit schema validation, error handling, and reproducible defaults. `etl.py` validates and normalises PPMI exports; `adapter.py` harmonises the configured clinical score and visit date fields; and `data_utils.py` converts long-format observations into patient-level baseline, target, and delta records while reporting follow-up attrition.

`validation.py` applies `GroupKFold` with explicit `patient_id` isolation,
`modeling.py` runs a deliberately minimal ElasticNet execution harness, and
`visualization.py` produces diagnostic figures from its fold reports. Earlier
Random Forest endpoint-delta components are preserved in `legacy/` and are not
executed by the canonical pipeline. Together, the active modules implement an
inspectable software and computational-methodology framework rather than a
clinical prediction product.

## Statement of need

Longitudinal research software must preserve patient identity, temporal
ordering, schema assumptions, and cohort decisions across every processing
stage. One-off scripts commonly hide these contracts and can accidentally
place records from the same participant in both training and test data.
PPMI-Pipeline provides a compact, inspectable implementation that:

* identifies an earliest usable BL baseline and a fixed-horizon follow-up visit;
* constructs auditable baseline, target, delta, and binary progression-label data;
* uses `patient_id`-grouped `GroupKFold` splitting to prevent participant leakage;
* performs median imputation and standardisation inside each training fold; and
* records reproducible fold diagnostics, coefficients, audit reports, and figures.

The repository is intended for software verification, data-engineering
research, and computational-method development. The included model is an
execution harness, not a diagnostic device or substitute for clinical
judgment.

## Installation

Python 3.12 through 3.14 is supported, matching the interpreter constraint in
`pyproject.toml`.

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

pip install -r requirements-lock.txt
pip install -e . --no-deps
```

The editable installation exposes the namespaced `ppmi_pipeline` API and the
`ppmi-pipeline` console command while retaining backward-compatible root
modules and `python main.py`. `requirements-lock.txt` records the complete
dependency graph used to validate v2.1.7; `requirements.txt` retains the
smaller set of direct runtime pins for dependency review.

Contributors can install the pinned formatting and linting tool with:

```bash
pip install -e ".[dev]"
ruff format --check .
ruff check .
```

Download the appropriate PPMI curated export through the PPMI data portal and place it locally. PPMI data are governed by their own access and data-use agreement and are not redistributed with this repository.

## Quick start

1. Confirm your PPMI export contains the required columns: `PATNO`, `EVENT_ID`, `visit_date`, `moca`, and `updrs3_score`.
2. Run the installed command with an explicit input path:

   ```bash
   ppmi-pipeline --data-path path/to/ppmi_export.csv
   ```

   Optional flags include `--sheet-name`, `--score-column`,
   `--target-horizon-days`, `--window-tolerance-days`,
   `--progression-threshold`, and `--n-splits`.

3. For backward-compatible source execution, update
   `FIXED_HORIZON_CONFIG` in `config.py`, then run:

   ```bash
   python main.py
   ```

## Running with Dummy Data

To regenerate the synthetic test fixture, run:

```bash
python tests/generate_dummy_data.py
```

The repository includes `tests/dummy_ppmi.csv`, containing 50 synthetic
patients observed at BL, V01, and V02. An identical fixture is packaged under
`ppmi_pipeline/sample_data/` so installed and source executions share the same
safe default. Run the complete workflow with:

```bash
python main.py
```

## Methodology

The pipeline first performs a header-only schema validation before loading the full PPMI export. It normalises identifiers and visit dates, then adapts the configured clinical score into the fixed-horizon schema. For each participant, it selects the earliest usable BL record and the valid follow-up observation closest to the configured horizon within the configured tolerance window. The resulting patient-level data contain baseline score, target score, and score change; follow-up attrition is reported explicitly.

Baseline-compatible numeric fields are median-imputed and standardized within
each training fold before the ElasticNet execution harness. A binary test
target is defined from the configured score-change threshold. `GroupKFold`
groups strictly by patient, and every fold raises an explicit error if training
and testing partitions share a participant.

The active workflow reports fold-level diagnostic metrics and saves
coefficients, a consistency summary, and visualization artifacts. These outputs
demonstrate end-to-end execution and expose pipeline behavior; they are not
claims of optimized clinical performance. The quarantined endpoint-delta
workflow in `legacy/` is not executed by `main.py`.

## Testing

Run the lightweight schema and longitudinal-alignment tests with:

```bash
python -m unittest discover -s tests -v
```

## Citation

See [CITATION.cff](CITATION.cff) for software citation metadata. The project
archive is available from
[Zenodo](https://doi.org/10.5281/zenodo.21225810).

## Documentation

The accompanying manuscript draft, bibliography, and manuscript-specific figures
are located in [manuscript/](manuscript/). The draft source is
[manuscript/paper.md](manuscript/paper.md).

## PPMI Compliance

The following acknowledgement follows the
[September 2024 PPMI Publication Policy](https://www.ppmi-info.org/sites/default/files/docs/ppmi-publication-policy.pdf)
and uses the
[March 2026 funding-partner list](https://www.ppmi-info.org/sites/default/files/docs/PPMI%20Funding%20Partners.pdf):

> Data used in the preparation of this article was obtained on 2026-07-15 from
> the Parkinson’s Progression Markers Initiative (PPMI) database
> (www.ppmi-info.org/access-data-specimens/download-data), RRID:SCR_006431. For
> up-to-date information on the study, visit www.ppmi-info.org.

> PPMI – a public-private partnership – is funded by the Michael J. Fox
> Foundation for Parkinson’s Research, and funding partners; including AbbVie,
> Alamar Biosciences, Aligning Science Across Parkinson’s (ASAP), Arrowhead
> Pharma, Arvinas, AskBio, BIAL, BioArctic, Biohaven, BlueRock Therapeutics,
> Bristol Myers Squibb, Calico Labs, Capsida Biotherapeutics, Critical Path
> Institute, DaCapo Brainscience, Denali, Edmond J. Safra Foundation, Eli Lilly,
> Gain Therapeutics, GE Healthcare, Genentech, GSK, Insitro, Johnson & Johnson
> Innovative Medicine, Lundbeck, Merck, Neumora, Neuron23, Novartis, Olink,
> Regeneron, Roche, Sanofi, Tenvie, UCB, Vanqua Bio, Voyager Therapeutics, The
> Weston Family Foundation.

This is an external PPMI work, so PPMI group authorship is not claimed. Posters
and abstracts using PPMI data must be uploaded through the PPMI website for DPC
administrative compliance review and the DPC must be notified after acceptance
or presentation at `PPMI.Publications@indd.org`.

**Reproducibility disclaimer:** “As the PPMI database is always evolving, it is
possible that the code may not work if the database has changed since the date
the code was created.”

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).
