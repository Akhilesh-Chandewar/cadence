"""Demo every §6 ingest source through the full graph: CSV, SQL (SQLite), REST.

The REST stage serves records from a tiny local HTTP server (stdlib), so the
whole demo is network-free. Each run goes through the complete §7.6 graph —
ingest → diagnose → plan → forecast → report — printing the per-source result.

Usage:
    uv run python scripts/sources_demo.py
"""

from __future__ import annotations

import json
import threading
from datetime import date, timedelta
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from cadence.config.default_config import CadenceConfig
from cadence.graph.cadence_graph import run_pipeline


# ----------------------------------------------------------- tiny REST server
class _Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # http.server dispatch is case-sensitive
        page = int(self.path.split("page=")[1].split("&")[0]) if "page=" in self.path else 1
        # 30 consecutive daily reads split across two pages
        all_records = [
            {
                "sensor": "api_sensor",
                "ts": (date(2024, 1, 1) + timedelta(days=i)).isoformat(),
                "temp": 20.0 + i % 5,
            }
            for i in range(30)
        ]
        records = all_records[(page - 1) * 15 : page * 15]
        body = json.dumps({"data": records}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args) -> None:  # silence request logging
        pass


def main() -> None:
    cfg = CadenceConfig()  # LLM off: deterministic, fast, no key needed
    cfg.forecast.horizon = 4  # short demo horizon: 28-point series backtest cleanly

    # ------------------------------------------------------------- 1. CSV
    print("=== CSV (path) ===")
    state = run_pipeline({"path": "data/sample/synthetic_seasonal.csv"}, cfg)
    print(
        f"  rows={state['source_meta']['row_count']}  "
        f"series={list(state['diagnostics'])}  errors={len(state['errors'])}"
    )

    # ------------------------------------------------------------- 2. SQL
    print("=== SQL (SQLite via SQLAlchemy) ===")
    from sqlalchemy import create_engine, text

    db_path = Path("/tmp/cadence_demo.db")
    engine = create_engine(f"sqlite:///{db_path}")
    rng = [f"2024-01-{d:02d}" for d in range(1, 29)]
    rows = [{"sku": "sql_sku", "day": day, "units": float(10 + i % 5)} for i, day in enumerate(rng)]
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS sales"))
        conn.execute(text("CREATE TABLE sales (sku TEXT, day TIMESTAMP, units REAL)"))
        conn.execute(text("INSERT INTO sales VALUES (:sku, :day, :units)"), rows)
    state = run_pipeline(
        {
            "connection_string": f"sqlite:///{db_path}",
            "table": "sales",
            "column_mapping": {"unique_id": "sku", "ds": "day", "y": "units"},
        },
        cfg,
    )
    print(
        f"  rows={state['source_meta']['row_count']}  "
        f"series={list(state['diagnostics'])}  errors={len(state['errors'])}"
    )

    # ------------------------------------------------------------ 3. REST
    print("=== REST (page-number pagination, local server) ===")
    server = HTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    port = server.server_address[1]
    try:
        state = run_pipeline(
            {
                "url": f"http://127.0.0.1:{port}/reads",
                "records_path": "data",
                "page_param": "page",
                "size_param": "size",
                "column_mapping": {"unique_id": "sensor", "ds": "ts", "y": "temp"},
            },
            cfg,
        )
        print(
            f"  rows={state['source_meta']['row_count']}  "
            f"series={list(state['diagnostics'])}  errors={len(state['errors'])}"
        )
    finally:
        server.shutdown()

    print("\nall three §6 sources flowed through the same graph ✅")


if __name__ == "__main__":
    main()
