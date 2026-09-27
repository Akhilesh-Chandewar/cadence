"""Phase 8 tests: ReportAgent rendering + FastAPI layer."""

from __future__ import annotations

import pytest

from cadence.agents.report_agent import ReportAgent


@pytest.fixture
def report() -> dict:
    return {
        "air_passengers": {
            "series": "air_passengers",
            "length": 144,
            "freq": "MS",
            "preprocessing": {
                "missing": "interpolate_linear",
                "outliers": "keep_and_flag",
                "transform": "log",
                "reason": "right-skewed positive series → log",
            },
            "decision": "ensemble",
            "selected": ["AutoARIMA", "AutoETS"],
            "weights": {"AutoARIMA": 0.6, "AutoETS": 0.4},
            "scores": [
                {"model": "AutoARIMA", "mase_mean": 0.87, "smape_mean": 0.068, "wql": 0.0007},
                {"model": "AutoETS", "mase_mean": 1.02, "smape_mean": 0.075, "wql": None},
            ],
            "forecast": [
                {"unique_id": "air_passengers", "ds": "1961-01-01", "yhat": 447.1},
                {"unique_id": "air_passengers", "ds": "1961-02-01", "yhat": 428.0},
            ],
            "intervals": [
                {
                    "unique_id": "air_passengers",
                    "ds": "1961-01-01",
                    "yhat-lo-95": 412.0,
                    "yhat-hi-95": 483.0,
                },
                {
                    "unique_id": "air_passengers",
                    "ds": "1961-02-01",
                    "yhat-lo-95": 390.0,
                    "yhat-hi-95": 468.0,
                },
            ],
        }
    }


class TestMarkdown:
    def test_full_report_renders(self, report):
        md = ReportAgent().render_markdown(
            report,
            {
                "row_count": 144,
                "series_count": 1,
                "source_type": "csv",
                "ingested_at": "2026-09-27T10:00:00",
            },
            [{"stage": "forecast", "unique_id": "other", "error": "ValueError: boom"}],
        )
        assert "# Cadence forecast report" in md
        assert "**Run**: 144 rows, 1 series" in md
        assert "transform=log" in md and "right-skewed positive series → log" in md
        assert "| AutoARIMA | 0.870 |" in md
        assert "no single model was a clear, consistent winner" in md
        assert "AutoARIMA 0.60" in md
        assert "| 1961-01-01 | 447.10 | 412.00 | 483.00 |" in md
        assert "`[forecast]` other: ValueError: boom" in md

    def test_best_decision_wording(self, report):
        report["air_passengers"]["decision"] = "best"
        report["air_passengers"]["selected"] = ["AutoARIMA"]
        md = ReportAgent().render_markdown(report)
        assert "won the backtest" in md

    def test_missing_intervals_render_dashes(self, report):
        report["air_passengers"]["intervals"] = None
        md = ReportAgent().render_markdown(report)
        assert "| 1961-01-01 | 447.10 | — | — |" in md


class TestHTML:
    def test_tables_lists_and_escaping(self, report):
        md = ReportAgent().render_markdown(report)
        html = ReportAgent().render_html(md)
        assert "<table>" in html and "<tr><th>model</th>" in html
        assert "<li>" in html and "<strong>Decision</strong>" in html
        # user-influenced text is escaped (reason comes from an LLM in production)
        report["air_passengers"]["preprocessing"]["reason"] = "<script>alert(1)</script>"
        md2 = ReportAgent().render_markdown(report)
        html2 = ReportAgent().render_html(md2)
        assert "<script>" not in html2
        assert "&lt;script&gt;" in html2


class TestFastAPI:
    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient

        from cadence.api.main import app

        return TestClient(app)

    def test_forecast_json(self, client, tmp_path):
        import shutil

        shutil.copy("data/sample/synthetic_seasonal.csv", tmp_path / "s.csv")
        resp = client.post(
            "/forecast",
            json={"source_config": {"path": str(tmp_path / "s.csv")}, "horizon": 6},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["source_meta"]["row_count"] == 196
        assert "synthetic_seasonal" in body["report"]
        assert body["errors"] == []

    def test_forecast_markdown_and_html_formats(self, client, tmp_path):
        import shutil

        shutil.copy("data/sample/synthetic_seasonal.csv", tmp_path / "s.csv")
        cfg = {"source_config": {"path": str(tmp_path / "s.csv")}, "horizon": 6}
        md = client.post("/forecast", json={**cfg, "format": "markdown"}).json()["markdown"]
        assert "# Cadence forecast report" in md
        html = client.post("/forecast", json={**cfg, "format": "html"}).json()["html"]
        assert "<table>" in html

    def test_diagnostics_endpoint(self, client, tmp_path):
        import shutil

        shutil.copy("data/sample/synthetic_seasonal.csv", tmp_path / "s.csv")
        resp = client.get(
            "/diagnostics/synthetic_seasonal", params={"path": str(tmp_path / "s.csv")}
        )
        assert resp.status_code == 200
        diag = resp.json()["diagnostics"]
        assert diag["seasonality"]["period"] == 7

    def test_diagnostics_unknown_id_404(self, client, tmp_path):
        import shutil

        shutil.copy("data/sample/synthetic_seasonal.csv", tmp_path / "s.csv")
        resp = client.get("/diagnostics/nope", params={"path": str(tmp_path / "s.csv")})
        assert resp.status_code == 404
        assert "known" in resp.json()["detail"]

    def test_forecast_missing_file_400(self, client, tmp_path):
        resp = client.post("/forecast", json={"source_config": {"path": str(tmp_path / "x.csv")}})
        assert resp.status_code == 400
