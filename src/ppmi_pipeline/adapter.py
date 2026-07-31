"""Schema adapter connecting normalized PPMI data to the methodology contract."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import pandas as pd

from .config import DEFAULT_SCORE_COLUMN
from .exceptions import ConfigurationError, MissingColumnError
from .paper2_config import PAPER2_FEATURE_BLOCKS


def harmonize_schema(
    df: pd.DataFrame,
    score_column: str = DEFAULT_SCORE_COLUMN,
) -> pd.DataFrame:
    """Map normalised PPMI fields to the fixed-horizon input schema.

    Args:
        df: Normalised PPMI data from ``PPMIDataLoader.load``.
        score_column: Normalized score column used for fixed-horizon target
            construction, such as ``updrs3_score``.

    Returns:
        A copy with exactly ``PATNO``, ``EVENT_ID``, ``SCORE``, and
        ``VISIT_DATE`` columns required by ``create_fixed_horizon_dataset``.

    Raises:
        TypeError: If ``df`` is not a pandas DataFrame.
        MissingColumnError: If a required PPMI identity, visit, date, or score
            column is unavailable.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Schema harmonisation requires a pandas DataFrame.")
    source_columns = {"PATNO", "EVENT_ID", "visit_date", score_column}
    missing = sorted(source_columns.difference(df.columns))
    if missing:
        raise MissingColumnError(
            "Cannot harmonize PPMI schema; missing columns: " + ", ".join(missing)
        )
    return (
        df[["PATNO", "EVENT_ID", score_column, "visit_date"]]
        .rename(columns={score_column: "SCORE", "visit_date": "VISIT_DATE"})
        .copy()
    )


def harmonize_paper2_schema(
    df: pd.DataFrame,
    column_map: Mapping[str, str],
    feature_blocks: Mapping[str, Sequence[str]],
    column_constants: Mapping[str, object] | None = None,
) -> pd.DataFrame:
    """Create a canonical Paper 2 table while retaining declared covariates.

    ``column_map`` maps canonical names to source columns. Only explicitly
    declared core fields and feature-block columns are retained, preventing
    accidental use of undocumented or post-baseline variables.

    Args:
        df: Source longitudinal table.
        column_map: Canonical-to-source mapping. Required canonical fields are
            ``PATNO``, ``EVENT_ID``, ``VISIT_DATE``, ``DIAGNOSTIC_GROUP``, and
            ``SCORE``. ``MEDICATION_STATE`` is required by medication-sensitive
            phenotypes rather than by this adapter.
        feature_blocks: Seven domain-specific feature column lists.
        column_constants: Optional canonical fields assigned from an explicit
            constant. Constants take precedence over source mappings. This is
            used, for example, when an export provides a documented OFF-state
            endpoint column but no separate medication-state field.

    Returns:
        Canonical core fields plus the explicitly declared feature columns.

    Raises:
        TypeError: If ``df`` is not a DataFrame.
        ConfigurationError: If mappings or feature blocks are ambiguous.
        MissingColumnError: If a declared source column is unavailable.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Paper 2 schema harmonisation requires a DataFrame.")
    constants = dict(column_constants or {})
    unsupported_constants = sorted(set(constants).difference({"MEDICATION_STATE"}))
    if unsupported_constants:
        raise ConfigurationError(
            "Paper 2 column_constants contains unsupported canonical fields: "
            + ", ".join(unsupported_constants)
        )
    required_canonical = {
        "PATNO",
        "EVENT_ID",
        "VISIT_DATE",
        "DIAGNOSTIC_GROUP",
        "SCORE",
    }
    available_canonical = set(column_map).union(constants)
    missing_mapping = sorted(required_canonical.difference(available_canonical))
    if missing_mapping:
        raise ConfigurationError(
            "Paper 2 column_map is missing canonical fields: "
            + ", ".join(missing_mapping)
        )
    unknown_blocks = sorted(set(feature_blocks).difference(PAPER2_FEATURE_BLOCKS))
    missing_blocks = sorted(set(PAPER2_FEATURE_BLOCKS).difference(feature_blocks))
    if unknown_blocks or missing_blocks:
        details = []
        if unknown_blocks:
            details.append("unknown=" + ",".join(unknown_blocks))
        if missing_blocks:
            details.append("missing=" + ",".join(missing_blocks))
        raise ConfigurationError(
            "Paper 2 feature-block contract is invalid: " + "; ".join(details)
        )

    effective_map = {
        canonical: source
        for canonical, source in column_map.items()
        if canonical not in constants
    }
    source_core = list(effective_map.values())
    if len(source_core) != len(set(source_core)):
        raise ConfigurationError(
            "Paper 2 column_map cannot map one source column to multiple "
            "canonical fields."
        )
    feature_columns = resolve_feature_columns(feature_blocks)
    declared_columns = list(dict.fromkeys([*source_core, *feature_columns]))
    missing_source = sorted(set(declared_columns).difference(df.columns))
    if missing_source:
        raise MissingColumnError(
            "Paper 2 source data are missing declared columns: "
            + ", ".join(missing_source)
        )

    source_to_canonical = {
        source: canonical for canonical, source in effective_map.items()
    }
    harmonized = df[declared_columns].rename(columns=source_to_canonical).copy()
    for canonical, value in constants.items():
        harmonized[canonical] = value
    if harmonized.columns.duplicated().any():
        duplicates = sorted(set(harmonized.columns[harmonized.columns.duplicated()]))
        raise ConfigurationError(
            "Schema harmonisation produced duplicate columns: " + ", ".join(duplicates)
        )
    return harmonized


def resolve_feature_columns(
    feature_blocks: Mapping[str, Sequence[str]],
    selected_blocks: Sequence[str] | None = None,
) -> list[str]:
    """Return ordered unique feature columns from selected domain blocks.

    Args:
        feature_blocks: Mapping of Paper 2 domain names to source columns.
        selected_blocks: Optional ordered subset of blocks. Defaults to all
            seven blocks in the canonical Paper 2 order.

    Returns:
        Ordered unique feature column names.

    Raises:
        ConfigurationError: If an unknown feature block is selected.
    """
    blocks = tuple(selected_blocks or PAPER2_FEATURE_BLOCKS)
    unknown = sorted(set(blocks).difference(feature_blocks))
    if unknown:
        raise ConfigurationError(
            "Selected feature blocks are undefined: " + ", ".join(unknown)
        )
    columns: list[str] = []
    for block in blocks:
        for column in feature_blocks[block]:
            if column not in columns:
                columns.append(str(column))
    return columns
