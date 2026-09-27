# Cadence — a time series analysis harness

**Ingest → Diagnose → Plan → Forecast → Report.** Cadence is an agentic control loop (not a
fixed script) that takes any time-stamped data source, converts it into forecast-ready shape,
diagnoses its characteristics, picks and fits the right class of models for *that* series,
backtests honestly, and returns a validated forecast with confidence intervals — plus a
plain-language report explaining every decision it made along the way.

Built against the full spec in [`cadence-project-spec.md`](cadence-project-spec.md)
(TSci-inspired five-agent architecture, §7). **All phases 0–10 are implemented.**

---

## How it works

```
 CSV / SQL / REST ──▶ IngestAgent ──▶ DiagnosticAgent ──▶ PlannerAgent
                      (connectors,      (ADF/KPSS, STL,     (§7.3 rule table,
                       schema mapping,   outliers, trend,     LLM arbitrates
                       validation)       intermittency)       borderline cases)
                            │                 │                    │
                            ▼                 ▼                    ▼
                     canonical df       cleaned series +      candidate models
                    (unique_id,ds,y)    preprocessing plan        shortlist
                                                                   │
                                                                   ▼
                                                           ForecastAgent
                                                     (rolling backtest, MASE/sMAPE/WQL,
                                                      best-vs-ensemble, refit + intervals)
                                                                   │
                                                                   ▼
                                                            ReportAgent
                                                    (forecast + bands + why, in Markdown/HTML)
```

Everything is wired into a **LangGraph** state machine (`CadenceState`): per-series channels
merge, errors **accumulate** (one bad series never halts a 5,000-series batch), and the ingest
node retries transient I/O failures. The §7.7 LLM (Groq by default, any LiteLLM provider) is
*optional at every stage* — with no key, deterministic fallbacks produce the same artifacts.

### The four model tiers (§8) — all backtested, none assumed

| Tier | Models | Library |
|---|---|---|
| 1 · Classical | AutoARIMA, AutoETS, AutoTheta, Croston, TSB | `statsforecast` |
| 2 · ML | LightGBM over lag/rolling/calendar features | `mlforecast` |
| 3 · Deep learning | N-HiTS, TFT | `neuralforecast` |
| 4 · Foundation (zero-shot) | Chronos-Bolt | `chronos-forecasting` |

The ForecastAgent (§7.4) backtests every shortlisted candidate with **rolling-origin
cross-validation** (never a random split), scores with **MASE + sMAPE + WQL**, and declares a
winner only when it is *clear* (≥5% mean-MASE margin) **and** *consistent* (top in ≥70% of
windows) — otherwise it ensembles the top-k with **inverse-error weights**. Final forecasts are
back-transformed to the original series space (log/difference inversions included).

---

## Quickstart

Requires [uv](https://docs.astral.sh/uv/). Python 3.11 is pinned via `.python-version`
(uv fetches it automatically).

```bash
git clone https://github.com/Akhilesh-Chandewar/cadence && cd cadence

uv sync --group dev      # core deps + pytest/ruff — enough for phases 0–3
uv run pytest            # 209 tests, network-free (fixtures are committed)
```

### Run your first forecast

```bash
# full graph on the classic AirPassengers series, 12-step horizon
uv run python scripts/pipeline_demo.py data/sample/air_passengers.csv 12
```

---

## Manual testing guide (everything you can click/run)

### 1 · Test suite

```bash
uv run pytest -q                                # whole suite, no network needed
uv run pytest -q --group dev                    # (after `uv sync --group dev` only: some tiers skip)
CADENCE_TEST_CHRONOS=1 uv run pytest tests/test_foundation.py::TestRealWeights -q
                                                # downloads real Chronos-Bolt weights (~200MB, one-time)
```

### 2 · Demos (each stage of the harness, end to end)

| Command | What it shows |
|---|---|
| `uv run python scripts/pipeline_demo.py data/sample/air_passengers.csv 12` | **The whole graph**: ingest → diagnose → plan → forecast → report, printing each stage's checkpoint |
| `uv run python scripts/forecast_agent_demo.py data/sample/air_passengers.csv 12` | Diagnose → plan → §7.4 backtest + best/ensemble decision with weights, score table, final forecast |
| `uv run python scripts/diagnose_demo.py data/sample/synthetic_seasonal.csv` | Per-series diagnostics: trend, seasonality (period + strength), stationarity verdicts, preprocessing plan |
| `uv run python scripts/plan_demo.py data/sample/synthetic_stationary.csv` | §7.3 rule table: which models get shortlisted and why, borderline detection |
| `uv run python scripts/forecast_demo.py data/sample/air_passengers.csv 12` | Phase 1 minimal path: rolling backtest → winner → refit + 95% intervals |
| `uv run python scripts/zoo_demo.py data/sample/air_passengers.csv 12` | All synced tiers compared through one §9 scorer |
| `uv run python scripts/sources_demo.py` | All three ingest sources (CSV, SQL/SQLite, REST against a local server) through the same graph |
| `uv run python scripts/generate_fixtures.py` | Regenerate the deterministic seed-42 fixtures in `data/sample/` |

Fixtures: `air_passengers` (monthly, strong seasonality), `synthetic_stationary`,
`synthetic_trending`, `synthetic_seasonal` (period 7) — each with known-answer roles used by
the tests.

### 3 · Live UI (Streamlit) — watch every stage checkpoint render

```bash
uv sync --group ui
uv run streamlit run cadence/ui/app.py
```

Then open http://localhost:8501, pick a CSV path + horizon, toggle **§7.7 LLM arbitration**,
and press **▶ Run pipeline**. Each graph node renders its checkpoint as it completes:

- **ingest** — row/series metrics + source preview
- **diagnose** — trend/seasonality/stationarity badges, preprocessing plan with the
  LLM-vs-deterministic decision reason
- **plan** — the shortlist with ready/phase markers and rule hits
- **forecast** — decision, ensemble weight cards, §9 score table, forecast chart with bands
- **report** — the rendered Markdown report

Plus a raw checkpoint inspector and the accumulated-errors panel. The same event feed is
available programmatically via `stream_pipeline()` and as SSE (below).

### 4 · HTTP API (FastAPI)

```bash
uv run uvicorn cadence.api.main:app --reload     # docs at http://localhost:8000/docs

# JSON report
curl -s -X POST localhost:8000/forecast -H 'content-type: application/json' \
  -d '{"source_config": {"path": "data/sample/air_passengers.csv"}, "horizon": 12}' | jq

# Markdown or HTML report
curl -s -X POST localhost:8000/forecast -H 'content-type: application/json' \
  -d '{"source_config": {"path": "data/sample/air_passengers.csv"}, "horizon": 12, "format": "markdown"}'

# server-sent events: one checkpoint per graph stage, as it completes
curl -N "localhost:8000/pipeline/stream?path=data/sample/air_passengers.csv&horizon=12"

# diagnostics for one series
curl -s "localhost:8000/diagnostics/air_passengers?path=data/sample/air_passengers.csv"
```

### 5 · LLM providers (§7.7) — optional

The LLM arbitrates preprocessing choices (DiagnosticAgent) and borderline shortlist calls
(PlannerAgent). It is **off by default**; everything works without it.

```bash
uv sync --group llm
cp .env.example .env            # then set your provider's key inside
uv run python scripts/llm_smoke_test.py          # one real structured LLM call
```

Supported: OpenAI, Anthropic, Gemini, Groq, OpenRouter, Mistral, DeepSeek, Together, xAI,
and keyless local models via Ollama. Default provider is **Groq** (`openai/gpt-oss-120b`).
`.env` is gitignored — never commit keys.

---

## Ingest sources (§6)

All connectors emit the same canonical schema and pass the same validation:

```python
{"path": "data.csv"}  # CSV / Parquet
{"path": "data.parquet", "column_mapping": {...}}  # renamed columns + covariates
{"connection_string": "postgresql://...", "query": "SELECT ...", "column_mapping": {...}}
{"connection_string": "sqlite:///shop.db", "table": "sales"}
{
    "url": "https://api.example.com/reads",
    "records_path": "data",
    "page_param": "page",
    "size_param": "size",
    "column_mapping": {...},
}
{"url": "...", "next_path": "next"}  # cursor/next-URL pagination
```

## Data contract (§5)

Every series, regardless of source, becomes:

```
unique_id: str   # which series this row belongs to
ds: datetime     # tz-aware, normalized to UTC
y: float         # the target
```

Validation is strict: duplicate `(unique_id, ds)` pairs are **rejected, not dropped**;
`ds` must be monotonic per series; non-canonical columns become covariates (known-future vs
past-only is tracked for later tiers).

## Configuration

All thresholds live in `cadence/config/default_config.py` as a Pydantic model: LLM
(`provider/model/temperature`, hard off-switch), backtest (windows/horizon), planner
(short/long-history thresholds, ambiguity window, scale), ML/DL/Chronos hyperparameters, and
forecast (winner margin, consistency ratio, ensemble top-k).

## Repository layout

```
cadence/
├── connectors/        # CSV/Parquet, SQL, REST → canonical schema (§6)
├── agents/            # ingest, diagnostic, planner, forecast, report (§7)
│   └── diagnostic_stats.py   # pure stats functions (tested against fixtures)
├── llm/               # §7.7: LLMClient protocol, LiteLLM impl, disabled no-op
├── models/            # tier zoo: classical, ml, deep_learning, foundation (§8)
├── graph/             # LangGraph wiring + stream_pipeline (§7.6, Phase 10)
├── eval/              # backtest harness + MASE/sMAPE/WQL (§9)
├── api/               # FastAPI: /forecast, /diagnostics, /pipeline/stream
├── ui/                # Streamlit live pipeline dashboard (Phase 10)
├── config/            # Pydantic settings
├── state.py           # CadenceState contract (§7.6)
data/sample/           # committed seed-42 fixtures
scripts/               # demos (see the manual testing guide above)
tests/                 # 209 network-free tests
```

## Status

| Phase | Scope | Status |
|---|---|---|
| 0 | Scaffolding: uv project, canonical schema, CSV connector, round-trip tests, fixtures | ✅ |
| 1 | Classical MVP (statsforecast + rolling backtest + MASE/sMAPE) | ✅ |
| 2 | DiagnosticAgent (ADF/KPSS/STL/outliers) | ✅ |
| 3 | PlannerAgent (rule table + optional LLM) | ✅ |
| 4 | Model zoo: mlforecast, neuralforecast | ✅ |
| 5 | Foundation models (Chronos-Bolt) | ✅ |
| 6 | Ensembling (inverse-error weights, WQL) | ✅ |
| 7 | LangGraph wiring (CadenceState, retries, error accumulation) | ✅ |
| 8 | ReportAgent (Markdown/HTML) + FastAPI | ✅ |
| 9 | SQL/API connectors | ✅ |
| 10 | Live pipeline UI (Streamlit — replaces the optional v2 chat/Telegram item) | ✅ |

> Layout notes: `scripts/`, `cadence/ui/` and `cadence/state.py` are small additions to the
> spec §11 tree; everything else follows the spec.

## Design notes

- **Backtest honesty (§9):** rolling-origin CV everywhere; classical models batch through one
  statsforecast call, tier 2–4 models refit per window; a model is never scored on data it was
  fit on.
- **LLM discipline (§7.7):** one injectable `LLMClient` protocol, structured output only
  (Pydantic-validated), exponential-backoff retries, deterministic fallback on any failure,
  and a hard off-switch. Hallucinated model names are filtered against the catalog.
- **Dependency hygiene (§10):** core install is torch-free; heavy tiers live in optional
  groups (`ml`, `dl`, `llm`) and extras (`chronos`), with actionable `uv sync` hints when a
  group is missing. `torch` is pinned to the CPU wheel index.
