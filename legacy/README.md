# Legacy endpoint-delta workflow

This directory preserves the earlier Random Forest endpoint-delta workflow,
including its configuration and tests. It is not imported or executed by the
canonical fixed-horizon ElasticNet pipeline.

The current research workflow begins at the repository-root `main.py` and uses
`etl.py`, `adapter.py`, `data_utils.py`, `validation.py`, `modeling.py`, and
`visualization.py`.

The preserved Random Forest/SHAP analysis has an optional dependency on
`shap`. Install it separately only when running code in this directory.
