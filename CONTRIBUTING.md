# Contributing to PPMI-Pipeline

Contributions to the public software methodology are welcome through GitHub
issues and pull requests.

## Development workflow

1. Fork the repository and create a focused feature branch.
2. Install the package with `python -m pip install -e ".[dev]"`.
3. Add or update synthetic tests for every behavior change.
4. Run `ruff format --check .`, `ruff check .`, and
   `python -m unittest discover -s tests -v`.
5. Submit a pull request describing the software change and validation evidence.

## Data safety

Never attach, commit, paste, or upload PPMI records, participant identifiers,
restricted metadata, local paths, OOF predictions, fold assignments, analysis
logs, data-derived figures, or unpublished clinical findings. Reproduce bugs
with the synthetic fixtures under `tests/` or with newly generated mock data.

Security or data-exposure concerns should not be posted with sensitive details
in a public issue. Contact the repository owner privately first.
