"""SQL connector (spec §6): Postgres/MySQL/SQLite → canonical schema.

The user supplies either a SQL query or a table name, plus a ColumnMapping when
the source columns aren't already canonical. Everything lands in the shared
validator, so duplicate/monotonicity rules are enforced identically to CSV.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from cadence.connectors.base import SchemaValidationError, validate_canonical_df
from cadence.connectors.csv_connector import ColumnMapping


class SQLConnector:
    """Reads rows via SQLAlchemy and emits a validated canonical frame."""

    def __init__(
        self,
        connection_string: str,
        mapping: ColumnMapping | None = None,
        engine_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.connection_string = connection_string
        self.mapping = mapping or ColumnMapping()
        self.engine_kwargs = engine_kwargs or {}

    def _read(self, query: str | None = None, table: str | None = None) -> pd.DataFrame:
        try:
            from sqlalchemy import create_engine, text
        except ModuleNotFoundError as exc:  # sqlalchemy is a core dep; guard anyway
            raise SchemaValidationError("sqlalchemy is required for the SQL connector") from exc

        if not query and not table:
            raise SchemaValidationError("SQL connector needs either 'query' or 'table'")
        if query and table:
            raise SchemaValidationError("pass 'query' or 'table', not both")

        engine = create_engine(self.connection_string, **self.engine_kwargs)
        sql = query if query else f'SELECT * FROM "{table}"'
        try:
            with engine.connect() as conn:
                return pd.read_sql(text(sql), conn)
        except Exception as exc:
            raise SchemaValidationError(f"SQL read failed: {type(exc).__name__}: {exc}") from exc

    def load(
        self,
        query: str | None = None,
        table: str | None = None,
    ):
        from cadence.connectors.csv_connector import CSVConnector

        raw = self._read(query=query, table=table)
        canonical = CSVConnector(self.mapping).canonicalize(raw)
        return validate_canonical_df(canonical, source_type="sql")
