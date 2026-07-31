# Paper 2 configuration

The four experiment YAML files inherit from `base.yaml`. Restricted data paths
and snapshot identifiers are never stored in these public files.

Set the following variables in the local execution environment:

- `PPMI_DATA_PATH`: absolute path to the authorized CSV/XLSX export;
- `PPMI_DATABASE_VERSION`: exact local snapshot identifier; and
- `PPMI_SHEET_NAME`: worksheet name for Excel input (ignored for CSV input).

Configuration loading fails explicitly when a required variable is absent. The
resolved values contribute to the run’s SHA-256 configuration fingerprint.

All generated artifacts under `results/` and `outputs/` are local-only and must
be reviewed under the applicable PPMI DCI/DUA and publication policy.
