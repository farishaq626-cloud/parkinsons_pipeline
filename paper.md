---
title: 'PPMI-Pipeline: Reproducible Longitudinal Clinical Analysis for Parkinson''s Disease Research'
tags:
  - Python
  - Parkinson's disease
  - clinical research
  - longitudinal data
  - reproducibility
authors:
  - name: Faris Haq
    affiliation: 1
affiliations:
  - name: Independent Researcher
    index: 1
date: 16 July 2026
bibliography: paper.bib
---

# Summary

Parkinson's disease research frequently relies on longitudinal clinical data:
repeated assessments of the same participant that describe cognition, motor
function, symptoms, and treatment over time. Preparing these data for analysis
is labor-intensive. It commonly requires researchers to identify eligible
visits, harmonize variable types, determine a participant-specific baseline,
handle incomplete measurements, and avoid evaluating models on visits from
participants already represented in the training data. These steps are often
implemented as one-off scripts, which makes them difficult to inspect,
reproduce, or reuse.

PPMI-Pipeline is a Python package-style research workflow for automating these
steps for curated exports from the Parkinson's Progression Markers Initiative
(PPMI) [@ppmi_dataset]. The pipeline ingests a PPMI Excel or CSV export,
validates its required clinical schema before loading the full file, normalizes
patient and visit fields, constructs longitudinal change-from-baseline
outcomes, and evaluates models using patient-grouped train/test splits. Its
primary entry point, `main.py`, produces analyses for change in Montreal
Cognitive Assessment (MoCA) score and MDS-UPDRS Part III motor score.

The software is configured through a version-controlled JSON file and produces
timestamped results for each run. These artifacts include a metric table,
machine-readable configuration and input-path log, residual-distribution
figures, and SHAP feature-attribution plots. The source code is available in
the project repository [@ppmi_pipeline].

# Statement of Need

Clinical research pipelines must preserve the relationship between repeated
visits and the people from whom they were collected. A conventional random
row-level split may assign one participant's baseline record to a training set
and a later record to a test set. This contaminates the evaluation with
participant-specific information and can overstate expected performance on new
participants. Researchers also need transparent handling of missing predictors,
explicit endpoint definitions, and preserved records of exactly how a result was
created.

PPMI-Pipeline addresses these practical needs with a small, explicit workflow
for clinical progression modelling. It uses `PATNO` as the participant grouping
variable for `GroupShuffleSplit`, keeping all records for a participant in one
partition. Baseline observations are selected from configurable `BL` and `V01`
events; all later observations are linked to the participant's first valid
baseline. The resulting targets are the later MoCA or UPDRS-III values minus
their baseline values. This design makes the prediction target and temporal
reference clear rather than implicitly mixing visits.

The package also minimizes data leakage in missing-data handling. Median
imputation is included in the scikit-learn modelling pipeline, so the imputer is
fitted only on the training partition. Reproducibility metadata, model metrics,
and graphical diagnostics are written automatically to a timestamped results
directory. Together, these choices turn repeated manual preparation into a
reviewable workflow suitable for method development, exploratory clinical
research, and teaching.

# State of the Field

The PPMI study provides a large, multi-modal resource for studying Parkinson's
disease progression and biomarkers [@ppmi_dataset]. Its broad clinical and
biological coverage enables a range of analyses, but it also makes reproducible
longitudinal cohort preparation essential. General scientific Python libraries
provide robust foundations for data management and machine learning, including
pandas for tabular data and scikit-learn for model construction and evaluation
[@mckinney2010; @pedregosa2011]. However, they do not prescribe a
PPMI-specific baseline definition, endpoint construction, patient-level split,
or study-oriented results record.

PPMI-Pipeline occupies this applied layer. Rather than proposing a new disease
model or replacing established general-purpose libraries, it encodes a
transparent clinical analysis protocol around them. The workflow gives a
researcher a defined starting point for two commonly used longitudinal clinical
endpoints while retaining editable configuration for baseline event labels,
predictors, model parameters, and input location. It therefore complements
general data-science libraries by reducing repeated study-specific engineering.

# Software Design

The software is intentionally modular. `etl.py` implements PPMI input loading,
header-only schema validation, type normalization, and sorting by patient and
visit date. Required columns include `PATNO`, `EVENT_ID`, `visit_date`, `moca`,
and `updrs3_score`. Schema validation occurs before the full file is loaded so
that incompatible exports produce a clear error message early in execution.

`feature_engine.py` transforms the patient-visit table into endpoint-specific
modelling tables. For each participant, it selects the first valid configured
baseline record and joins its baseline values to later visits. Predictors include
the endpoint's baseline value, baseline clinical covariates (age, sex,
education, disease duration, and UPSIT score in the default configuration), and
continuous days since baseline. The endpoint delta remains separate from these
predictors, avoiding use of the later observed outcome as an input feature.

`analytics_engine.py` creates a `RandomForestRegressor` inside a scikit-learn
pipeline with median imputation. It uses `GroupShuffleSplit` on `PATNO`, then
reports mean absolute error (MAE), root mean squared error (RMSE), and the
coefficient of determination (R²) on held-out participants. The module also
saves residual histograms and TreeSHAP summary plots. These diagnostics support
inspection of systematic prediction error and ranking of the clinical features
that influence the fitted model [@lundberg2017].

`main.py` is the single execution entry point. It reads `config.json`, creates a
timestamped run identifier, executes both configured endpoints, and writes a
CSV metric table and text log. The log captures the timestamp, resolved input
path, configuration, MAE, RMSE, and R² for each endpoint. A synthetic
50-participant fixture generator and unit tests are included so that the
workflow can be exercised without access to controlled clinical data.

# Research Impact

PPMI-Pipeline supports more reproducible clinical Parkinson's disease research
by making cohort preparation and model-evaluation choices explicit and
persistently recorded. The automated workflow reduces transcription and
bookkeeping errors that can occur when data cleaning, feature construction, and
result reporting are performed manually in notebooks or spreadsheets. It also
makes it easier for collaborators and reviewers to identify the baseline
definition, predictors, random seed, and patient-level split used for a result.

The package is useful for exploratory progression modelling, sensitivity
analyses across endpoint definitions or predictor sets, and training researchers
in longitudinal machine-learning practice. The included SHAP outputs provide a
clinically interpretable starting point for discussing which baseline features
the model relies on. The software is not intended for diagnostic or treatment
decisions; results require appropriate clinical, statistical, and external
validation before any translational use.

# AI Disclosure

Generative AI assistance was used to help draft and edit software documentation
and this manuscript. The author reviewed, edited, and takes responsibility for
the final content, software design, analysis claims, and citations.

# Acknowledgements

The author thanks the Parkinson's Progression Markers Initiative (PPMI) for
making curated clinical research data available to qualified investigators. PPMI
is a public-private partnership funded by the Michael J. Fox Foundation for
Parkinson's Research and funding partners. This paper describes software for
working with PPMI exports; it does not redistribute PPMI data.
