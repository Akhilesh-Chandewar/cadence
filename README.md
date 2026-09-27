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

Regenerate the committed sample fixtures (deterministic, seed 42) with:

```bash
uv run python scripts/generate_fixtures.py
```

## Status

| Phase | Scope | Status |
|---|---|---|
| 0 | Scaffolding: uv project, canonical schema, CSV connector, round-trip tests, fixtures | ✅ |
| 1 | Classical MVP (statsforecast + rolling backtest + MASE/sMAPE) | ⬜ |
| 2 | DiagnosticAgent (ADF/KPSS/STL/outliers) | ⬜ |
| 3 | PlannerAgent (rule table + optional LLM) | ⬜ |
| 4 | Model zoo: mlforecast, neuralforecast | ⬜ |
| 5 | Foundation models (Chronos first) | ⬜ |
| 6 | Ensembling | ⬜ |
| 7 | LangGraph wiring | ⬜ |
| 8 | ReportAgent + FastAPI | ⬜ |
| 9 | SQL/API connectors | ⬜ |

> Note: `scripts/` is a small addition to the spec §11 layout (fixture generation); everything
> else follows the spec.
