# PPMI-Pipeline

[![Version](https://img.shields.io/badge/version-2.1.8-blue.svg)](https://github.com/farishaq626-cloud/parkinsons_pipeline)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-31%20passing-brightgreen.svg)](tests/)

PPMI-Pipeline is an open-source Python framework for auditable fixed-horizon
data engineering and patient-isolated validation with longitudinal clinical
data. The Parkinson’s Progression Markers Initiative (PPMI) schema is its
reference implementation, but the repository publishes software methodology—not
patient data, clinical conclusions, or a diagnostic product.

The current release is **v2.1.8**. The project archive is available from
[Zenodo](https://doi.org/10.5281/zenodo.21225810).

## Public repository scope and data compliance

This repository contains technical artifacts for reproducibility: source code,
release-safe configuration templates, synthetic fixtures, tests, documentation,
and validation utilities.

The following must remain local and require the applicable PPMI DCI/DUA and
publication approvals before any authorized disclosure:

- raw or transformed PPMI data;
- patient-level fold assignments and out-of-fold predictions;
- cohort or feature outputs derived from restricted data;
- run metadata, audit logs, and figures produced from restricted analyses; and
- manuscripts, posters, or interpretations presenting clinical findings.

The repository ignores `data/`, `results/`, `outputs/`, logs, tabular data
formats, workbooks, and generated figures. Always review `git status` before a
commit or release.

## Architecture

```text
src/ppmi_pipeline/     Installable pipeline and Paper 2 methodology package
configs/paper2/        Release-safe YAML experiment specifications
scripts/               Source-checkout entry points and poster utilities
tests/                 Synthetic unit and integration tests
poster/                LaTeX poster source and Gemini theme files
manuscript/            Historical manuscript materials
legacy/                Quarantined superseded workflow
data/                  Local restricted inputs (ignored)
results/               Local run artifacts (ignored)
outputs/               Local aggregate outputs and figures (ignored)
```

The active fixed-horizon workflow consists of:

1. schema validation and normalization;
2. explicit schema harmonization;
3. baseline and fixed-horizon outcome construction;
4. patient-isolated outer and inner validation;
5. fold-local imputation and scaling;
6. interpretable baseline and sensitivity models; and
7. provenance, OOF, stability, audit, and figure artifacts.

The historical endpoint-delta workflow in `legacy/` is not imported or executed
by the active package.

## Installation

Python 3.12–3.14 is supported.

```bash
git clone https://github.com/farishaq626-cloud/parkinsons_pipeline.git
cd parkinsons_pipeline
python -m venv .venv
```

Activate the environment and install the locked dependencies plus the package:

```powershell
# Windows PowerShell
.venv\Scripts\Activate.ps1
python -m pip install -r requirements-lock.txt
python -m pip install -e . --no-deps
```

```bash
# macOS/Linux
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python -m pip install -e . --no-deps
```

For development checks:

```bash
python -m pip install -e ".[dev]"
ruff format --check .
ruff check .
python -m unittest discover -s tests -v
```

## Quick start with synthetic data

The installed package includes a non-clinical synthetic fixture. Run the
canonical fixed-horizon workflow with:

```bash
ppmi-pipeline
```

Equivalent source-checkout execution is available through:

```bash
python scripts/run_pipeline.py
```

To regenerate `tests/dummy_ppmi.csv`:

```bash
python tests/generate_dummy_data.py
```

## Controlled PPMI execution

Download PPMI data only through authorized channels. Do not place controlled
data outside an ignored local directory.

The Paper 2 YAML files resolve three required environment variables at runtime.
This prevents user-specific paths and controlled snapshot metadata from being
stored in public configuration files.

```powershell
# Windows PowerShell example
$env:PPMI_DATA_PATH = "C:\path\to\authorized\ppmi_export.xlsx"
$env:PPMI_DATABASE_VERSION = "your-authorized-snapshot-identifier"
$env:PPMI_SHEET_NAME = "your-data-sheet"
```

```bash
# macOS/Linux example
export PPMI_DATA_PATH="/path/to/authorized/ppmi_export.xlsx"
export PPMI_DATABASE_VERSION="your-authorized-snapshot-identifier"
export PPMI_SHEET_NAME="your-data-sheet"
```

Run one configuration:

```bash
ppmi-paper2 --config configs/paper2/primary_analysis.yaml
```

Run all four experiment families:

```bash
ppmi-paper2-batch --config-dir configs/paper2 --output outputs/paper2_master_summary.csv
```

Use `--resume` to reuse only complete runs whose stored provenance hash matches
the currently resolved YAML:

```bash
ppmi-paper2-batch --config-dir configs/paper2 --resume
```

The batch covers primary analysis, horizon/tolerance sensitivity,
missingness sensitivity, and feature-domain ablation. It records resolved
configuration hashes, source hashes, seeds, fold assignments, OOF predictions,
bootstrap confidence intervals, and aggregate feature-stability summaries.

## Synthetic Paper 2 dry run

Generate mock flat and modality tables plus lightweight experiment configs:

```bash
ppmi-paper2-synthetic --output-dir outputs/synthetic --write-batch-configs
```

Then execute the generated configurations:

```bash
ppmi-paper2-batch --config-dir outputs/synthetic/batch/configs \
  --output outputs/synthetic/batch/paper2_master_summary.csv
```

Synthetic fixtures contain no PPMI records and are suitable for public CI.

## Integrity audit and specification curves

Audit configuration hashes, provenance, train/test isolation, one-test-fold
coverage per repeat, and exact OOF/test-assignment agreement:

```bash
ppmi-paper2-audit \
  --metadata outputs/paper2_master_metadata.json \
  --output outputs/paper2_integrity_audit.json
```

Generate publication-quality aggregate specification curves:

```bash
ppmi-paper2-plot \
  --input outputs/paper2_master_summary.csv \
  --output-dir outputs/figures
```

These commands operate locally. Their outputs remain ignored and must not be
published merely because they are aggregate or visually de-identified.

## Validation design

- Patient IDs define every outer and inner split.
- An explicit runtime check rejects any train/test patient overlap.
- Hyperparameters and binary thresholds are selected using training data only.
- Imputation and scaling are fitted independently inside each training fold.
- OOF records preserve configuration, repeat, fold, and fit identifiers.
- Patient-level bootstrap intervals quantify metric uncertainty.
- The public test suite uses synthetic data and requires no PPMI access.

The supplied models are methodology baselines and execution harnesses. They are
not clinical diagnostic or treatment-decision systems.

## Testing and CI

```bash
ruff format --check .
ruff check .
python -m unittest discover -s tests -v
```

GitHub Actions runs the same formatting, linting, and synthetic test checks.

## Poster and manuscript materials

- Poster source: [`poster/poster.tex`](poster/poster.tex)
- Poster build: `python scripts/build_poster.py`
- Historical manuscript: [`manuscript/paper.md`](manuscript/paper.md)

Generated poster PDFs and data-derived figures remain local and ignored.

## Citation

See [`CITATION.cff`](CITATION.cff). The canonical repository is
[farishaq626-cloud/parkinsons_pipeline](https://github.com/farishaq626-cloud/parkinsons_pipeline).

## PPMI acknowledgement and publication policy

The following acknowledgement follows the September 2024 PPMI Publication
Policy and the applicable funding-partner list:

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
> Regeneron, Roche, Sanofi, Tenvie, UCB, Vanqua Bio, Voyager Therapeutics, and
> The Weston Family Foundation.

This is an external PPMI work; PPMI group authorship is not claimed. Authors
must independently verify the current PPMI acknowledgement, funding-partner
list, DPC submission process, and post-acceptance notification requirements
before disseminating PPMI-derived work.

**Reproducibility disclaimer:** “As the PPMI database is always evolving, it is
possible that the code may not work if the database has changed since the date
the code was created.”

## License

PPMI-Pipeline is distributed under the [MIT License](LICENSE). PPMI data remain
subject to separate access and data-use terms.
