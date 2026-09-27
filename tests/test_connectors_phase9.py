"""Phase 9 connector tests: SQL (SQLite via SQLAlchemy) and REST (httpx MockTransport)."""

from __future__ import annotations

import httpx
import pandas as pd
import pytest

from cadence.connectors.api_connector import RESTConnector
from cadence.connectors.base import SchemaValidationError
from cadence.connectors.csv_connector import ColumnMapping
from cadence.connectors.sql_connector import SQLConnector


@pytest.fixture
def sqlite_url(tmp_path):
    """Two series x 60 days — enough history for the graph's rolling backtest."""
    from sqlalchemy import create_engine, text

    url = f"sqlite:///{tmp_path}/shop.db"
    engine = create_engine(url)
    rng = pd.date_range("2024-01-01", periods=60, freq="D")
    rows = [
        {
            "sku": sku,
            "day": str(day.date()),
            "units": float(10 + i) + (2.0 if sku == "a" else 0.0),
            "promo": int(i % 7 == 0),
        }
        for sku in ("a", "b")
        for i, day in enumerate(rng)
    ]
    with engine.begin() as conn:
        conn.execute(
            text("CREATE TABLE sales (sku TEXT, day TIMESTAMP, units REAL, promo INTEGER)")
        )
        conn.execute(text("INSERT INTO sales VALUES (:sku, :day, :units, :promo)"), rows)
    return url


SKU_MAPPING = ColumnMapping(unique_id="sku", ds="day", y="units", covariates={"promo": "promo"})


class TestSQLConnector:
    def test_table_load_with_mapping(self, sqlite_url):
        frame = SQLConnector(sqlite_url, SKU_MAPPING).load(table="sales")
        assert frame.source_meta.source_type == "sql"
        assert frame.source_meta.row_count == 120  # 2 series x 60 days
        assert list(frame.df.columns) == ["unique_id", "ds", "y", "promo"]

    def test_canonical_columns_pass_without_mapping(self, sqlite_url):
        """A table already in canonical shape needs no mapping (§6)."""
        from sqlalchemy import create_engine, text

        engine = create_engine(sqlite_url)
        with engine.begin() as conn:
            conn.execute(text("CREATE TABLE canonical_t (unique_id TEXT, ds TIMESTAMP, y REAL)"))
            conn.execute(text("INSERT INTO canonical_t VALUES ('c1', '2024-01-01', 1.0)"))
        frame = SQLConnector(sqlite_url).load(table="canonical_t")
        assert frame.source_meta.row_count == 1

    def test_noncanonical_without_mapping_raises(self, sqlite_url):
        """§6: the user supplies a mapping; we never guess column roles."""
        with pytest.raises(SchemaValidationError, match="missing source column"):
            SQLConnector(sqlite_url).load(table="sales")

    def test_query_with_mapping_and_covariates(self, sqlite_url):
        frame = SQLConnector(sqlite_url, SKU_MAPPING).load(
            query="SELECT sku, day, units, promo FROM sales WHERE sku = 'a'"
        )
        assert frame.source_meta.series_count == 1
        assert set(frame.df["promo"]) == {0, 1}

    def test_duplicate_rejection_via_shared_validator(self, sqlite_url):
        from sqlalchemy import create_engine, text

        engine = create_engine(sqlite_url)
        with engine.begin() as conn:
            conn.execute(text("INSERT INTO sales VALUES ('a', '2024-01-01', 99.0, 0)"))
        with pytest.raises(SchemaValidationError, match="duplicate"):
            SQLConnector(sqlite_url, SKU_MAPPING).load(table="sales")

    def test_query_and_table_both_given_raises(self, sqlite_url):
        with pytest.raises(SchemaValidationError, match="not both"):
            SQLConnector(sqlite_url).load(query="SELECT 1", table="sales")

    def test_neither_query_nor_table_raises(self, sqlite_url):
        with pytest.raises(SchemaValidationError, match="either"):
            SQLConnector(sqlite_url).load()


class TestRESTConnector:
    @staticmethod
    def _app_records(pages: dict[int, list[dict]]):
        """Build a MockTransport serving page-numbered records."""

        def handler(request: httpx.Request) -> httpx.Response:
            page = int(request.url.params.get("page", "1"))
            return httpx.Response(
                200, json={"data": pages.get(page, []), "has_more": page in pages}
            )

        return httpx.MockTransport(handler)

    def test_pagination_and_mapping(self):
        pages = {
            1: [{"sensor": "s1", "ts": "2024-01-01", "temp": 20.0}],
            2: [{"sensor": "s1", "ts": "2024-01-02", "temp": 21.0}],
        }
        connector = RESTConnector(
            url="https://api.example.com/reads",
            mapping=ColumnMapping(unique_id="sensor", ds="ts", y="temp"),
            records_path="data",
            page_param="page",
            size_param="size",
            transport=self._app_records(pages),
        )
        frame = connector.load()
        assert frame.source_meta.source_type == "api"
        assert frame.source_meta.row_count == 2
        assert list(frame.df.columns) == ["unique_id", "ds", "y"]

    def test_next_url_pagination(self):
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            if calls["n"] == 1:
                return httpx.Response(
                    200,
                    json={
                        "items": [{"dev": "d1", "at": "2024-01-01", "v": 1.0}],
                        "next": "/reads?cursor=2",
                    },
                )
            return httpx.Response(
                200,
                json={
                    "items": [{"dev": "d1", "at": "2024-01-02", "v": 2.0}],
                    "next": None,
                },
            )

        connector = RESTConnector(
            url="https://api.example.com/reads",
            mapping=ColumnMapping(unique_id="dev", ds="at", y="v"),
            records_path="items",
            next_path="next",
            transport=httpx.MockTransport(handler),
        )
        frame = connector.load()
        assert frame.source_meta.row_count == 2
        assert calls["n"] == 2

    def test_records_path_missing_raises(self):
        connector = RESTConnector(
            url="https://api.example.com/x",
            records_path="nope.deep",
            transport=httpx.MockTransport(lambda request: httpx.Response(200, json={"data": []})),
        )
        with pytest.raises(SchemaValidationError, match="records_path"):
            connector.load()

    def test_http_error_raises(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, json={"detail": "boom"})

        connector = RESTConnector(
            url="https://api.example.com/x", transport=httpx.MockTransport(handler)
        )
        with pytest.raises(SchemaValidationError, match="500"):
            connector.load()

    def test_empty_records_raises(self):
        connector = RESTConnector(
            url="https://api.example.com/x",
            records_path="data",
            transport=httpx.MockTransport(lambda request: httpx.Response(200, json={"data": []})),
        )
        with pytest.raises(SchemaValidationError, match="no records"):
            connector.load()


class TestGraphDispatch:
    """The §7.6 ingest node dispatches by source_config shape (path/sql/url)."""

    def test_graph_runs_sql_source(self, sqlite_url):
        import asyncio

        from cadence.config.default_config import CadenceConfig
        from cadence.graph.cadence_graph import build_cadence_graph

        app = build_cadence_graph(CadenceConfig())
        state = asyncio.run(
            app.ainvoke(
                {
                    "source_config": {
                        "connection_string": sqlite_url,
                        "table": "sales",
                        "column_mapping": {
                            "unique_id": "sku",
                            "ds": "day",
                            "y": "units",
                            "covariates": {"promo": "promo"},
                        },
                    },
                    "errors": [],
                }
            )
        )
        assert state["source_meta"]["source_type"] == "sql"
        assert set(state["diagnostics"]) == {"a", "b"}
        assert state["errors"] == []

    def test_graph_rejects_unknown_source_shape(self):
        import asyncio

        from cadence.config.default_config import CadenceConfig
        from cadence.connectors.base import SchemaValidationError
        from cadence.graph.cadence_graph import build_cadence_graph

        app = build_cadence_graph(CadenceConfig())
        with pytest.raises(SchemaValidationError, match="source_config"):
            asyncio.run(app.ainvoke({"source_config": {}, "errors": []}))
