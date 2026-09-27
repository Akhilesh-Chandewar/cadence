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

# model zoo: tiers 1–3 compared through one §9 scorer (ml/dl groups optional)
uv run python scripts/zoo_demo.py data/sample/air_passengers.csv 12
```

## Status

| Phase | Scope | Status |
|---|---|---|
| 0 | Scaffolding: uv project, canonical schema, CSV connector, round-trip tests, fixtures | ✅ |
| 1 | Classical MVP (statsforecast + rolling backtest + MASE/sMAPE) | ✅ |
| 2 | DiagnosticAgent (ADF/KPSS/STL/outliers) | ✅ |
| 3 | PlannerAgent (rule table + optional LLM) | ✅ |
| 4 | Model zoo: mlforecast, neuralforecast | ✅ |
| 5 | Foundation models (Chronos first) | ⬜ |
| 6 | Ensembling | ⬜ |
| 7 | LangGraph wiring | ⬜ |
| 8 | ReportAgent + FastAPI | ⬜ |
| 9 | SQL/API connectors | ⬜ |

> Note: `scripts/` is a small addition to the spec §11 layout (fixture generation); everything
> else follows the spec.
