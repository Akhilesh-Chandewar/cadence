# Cadence — a time series analysis harness

Ingest → diagnose → plan → forecast → report. See `cadence-project-spec.md` for the full
architecture, data contracts, and phased build order.

## Quickstart (uv)

```bash
uv sync --group dev       # core deps + pytest/ruff — enough through Phase 2
uv run pytest             # network-free test suite (fixtures are committed)
uv run ruff format . && uv run ruff check .
```

Heavier tiers are opt-in (spec §10):

```bash
uv sync --all-groups      # + llm, ml, dl dependency groups
uv sync --extra chronos   # foundation-model tier, lightest first
```

## LLM providers (§7.7)

The DiagnosticAgent/PlannerAgent LLM calls run through one LiteLLM-backed client —
**any** litellm-supported provider works, and the whole harness runs fine with the
LLM disabled (deterministic fallbacks, the default).

To enable it: `uv sync --group llm`, copy `.env.example` to `.env`, set the key for
your provider, then verify the whole chain:

```bash
uv run python scripts/llm_smoke_test.py --provider anthropic --model claude-sonnet-4-5
```

Regenerate the committed sample fixtures (deterministic, seed 42) with:

```bash
uv run python scripts/generate_fixtures.py
```

End-to-end demos:

```bash
# forecast: CSV → rolling backtest → forecast + 95% intervals
uv run python scripts/forecast_demo.py                    # AirPassengers, h=12

# diagnostics: CSV → per-series stats + preprocessing plan (deterministic path)
uv run python scripts/diagnose_demo.py                    # seasonal fixture

# planning: CSV → diagnostics → per-series model shortlist (§7.3 rule table)
uv run python scripts/plan_demo.py data/sample/air_passengers.csv

# model zoo: all synced tiers compared through one §9 scorer
uv run python scripts/zoo_demo.py data/sample/air_passengers.csv 12
```

Tier 4 zero-shot forecasting needs the chronos extra (spec §8):

```bash
uv sync --extra chronos   # + chronos-forecasting (transformers capped <5 for torch 2.4)
CADENCE_TEST_CHRONOS=1 uv run pytest tests/test_foundation.py::TestRealWeights -q  # downloads weights
```

Full §7.4 chain (diagnose → plan → backtest → best-or-ensemble → forecast):

```bash
uv run python scripts/forecast_agent_demo.py data/sample/air_passengers.csv 12
```

Full graph (ingest → diagnose → plan → forecast → report, §7.6):

```bash
uv run python scripts/pipeline_demo.py data/sample/air_passengers.csv 12
```

HTTP API (§12 Phase 8) — serves the graph with JSON/Markdown/HTML report formats:

```bash
uv run uvicorn cadence.api.main:app --reload
curl -X POST localhost:8000/forecast -H 'content-type: application/json' \
  -d '{"source_config": {"path": "data/sample/air_passengers.csv"}, "horizon": 12, "format": "markdown"}'
```

Ingest sources (§6): CSV/Parquet via `path`, SQL via `connection_string` + `query`/`table`
(+ optional `column_mapping`), REST via `url` (+ `records_path`, `page_param`/`size_param`
or `next_path` pagination) — all validated into the same canonical schema.

## Live pipeline UI (Phase 10)

Streamlit dashboard where **every graph stage renders its checkpoint as it completes**:
ingest metrics + source preview, per-series diagnostics (trend / seasonality /
stationarity badges, preprocessing with the §7.7 LLM-vs-deterministic reason), the
§7.3 shortlist, the §7.4 decision with ensemble weights + score table + forecast
chart, the rendered report, and a raw checkpoint inspector.

```bash
uv sync --group ui
uv run streamlit run cadence/ui/app.py
```

Toggle "§7.7 LLM arbitration" in the sidebar to run with the LLM enabled (needs a
provider key in `.env`). The same event feed is available programmatically via
`cadence.graph.cadence_graph.stream_pipeline` and as SSE at `GET /pipeline/stream`
when the FastAPI app is running.

## Status

| Phase | Scope | Status |
|---|---|---|
| 0 | Scaffolding: uv project, canonical schema, CSV connector, round-trip tests, fixtures | ✅ |
| 1 | Classical MVP (statsforecast + rolling backtest + MASE/sMAPE) | ✅ |
| 2 | DiagnosticAgent (ADF/KPSS/STL/outliers) | ✅ |
| 3 | PlannerAgent (rule table + optional LLM) | ✅ |
| 4 | Model zoo: mlforecast, neuralforecast | ✅ |
| 5 | Foundation models (Chronos first) | ✅ |
| 6 | Ensembling | ✅ |
| 7 | LangGraph wiring | ✅ |
| 8 | ReportAgent + FastAPI | ✅ |
| 9 | SQL/API connectors | ✅ |
| 10 | Live pipeline UI (Streamlit; replaces the optional chat/Telegram v2 item) | ✅ |

> Note: `scripts/` and `cadence/ui/` are small additions to the spec §11 layout; everything
> else follows the spec.
