# Cadence — A Time Series Analysis Harness

**Working name:** Cadence
**Alternates if you want options:** TimeLoom, Chronoforge, Prognos, Wayfinder-TS

*Why "Cadence": a cadence is a recurring rhythm — the natural word for something whose whole job is to read the rhythm of any data stream (hourly, daily, monthly, irregular) and continue it forward. Short, easy to say out loud, not already a heavily-used OSS project name in this space.*

This document is the build spec for a coding agent. It defines the architecture, the exact data contracts between components, the model zoo, the tech stack with package names, the file layout, and a phased build order. Treat section 12 (Build Phases) as the literal task list.

---

## 1. Problem Statement

Build a harness — not a single model, not a single notebook — that takes **any data source** (CSV, a SQL table, a REST API, a streaming feed) containing time-stamped values, automatically converts it into forecast-ready data, diagnoses its characteristics, selects and fits the right class of time-series model(s) for that specific data, and produces a validated next-step (or next-N-step) forecast with a confidence interval and a plain-language explanation of *why* that model was chosen.

The "harness" framing matters: the pipeline must not be a fixed hardcoded script (source → ARIMA → done). It must be an agentic control loop that inspects the data at runtime and adapts each stage's decision — preprocessing strategy, model tier, ensemble weights — to what the data actually looks like. A monthly sales series with 40 points and an hourly IoT sensor stream with 200,000 points must NOT go through the same code path unmodified.

## 2. Goals

- Ingest from arbitrary sources into one canonical schema
- Diagnose each series automatically (trend, seasonality, stationarity, missingness, outliers, frequency)
- Select from four tiers of forecasting models based on that diagnosis, not a fixed default
- Backtest properly (rolling-window, not random split) before ever showing a forecast
- Ensemble when no single model wins
- Produce a forecast + confidence interval + a human-readable report explaining the reasoning
- Work on a single series or thousands of related series (SKUs, sensors, tickers) without code changes

## 3. Non-Goals (v1)

- Not a general anomaly-detection or classification product (forecasting only, though the diagnostics layer will surface anomalies as a side effect)
- Not building new foundation models from scratch — v1 consumes existing pretrained ones (Chronos, Moirai, TimesFM) as zero-shot tools, not training targets
- Not a hosted multi-tenant SaaS in v1 — single-user/local-first, API-first so a UI or Telegram bot can sit on top later

---

## 4. Architecture Overview

```
                        ┌─────────────────────────────────────────┐
                        │              CADENCE HARNESS              │
                        └─────────────────────────────────────────┘

 [Any Data Source]                                                    [Output]
 CSV / SQL / API /  ──▶  IngestAgent  ──▶  DiagnosticAgent  ──▶  PlannerAgent
 Kafka / Sheets           (connectors,        (Curator role:         (model tier
                          schema mapping,      stats + LLM-guided     selection,
                          validation)          preprocessing plan)    hyperparams)
                                │                    │                    │
                                ▼                    ▼                    ▼
                          canonical df         cleaned series      candidate model
                       (unique_id, ds, y)     + diagnostics.json      shortlist
                                                                          │
                                                                          ▼
                                                                  ForecastAgent
                                                            (fit, rolling backtest,
                                                             score, ensemble)
                                                                          │
                                                                          ▼
                                                                   ReportAgent
                                                         (forecast + intervals +
                                                          plain-English reasoning)
```

This mirrors the four/five-agent shape used by **TimeSeriesScientist (TSci)** — the closest published prior art for this exact idea (see References, §13). TSci uses PreprocessAgent → AnalysisAgent → ValidationAgent → ForecastAgent → ReportAgent, orchestrated with LangGraph and a shared mutable state object passed node to node. Cadence adopts the same shared-state pattern but folds "Analysis" and "Preprocess" into one `DiagnosticAgent` and folds "Validation" (model selection) into a `PlannerAgent`, since for a harness-first (vs. research-first) build, tighter agent boundaries are easier to test independently.

**Read `github.com/Y-Research-SBU/TimeSeriesScientist` before scaffolding anything — it is a working reference implementation of this exact pattern, not just a paper.**

---

## 5. Canonical Data Schema

Every connector's only job is to emit rows in this shape. Nothing downstream ever looks at a source-specific format again.

```python
# canonical long-format schema (this is the Nixtla-ecosystem convention —
# adopting it means StatsForecast / MLForecast / NeuralForecast work with zero glue code)

unique_id: str  # which series this row belongs to (SKU id, sensor id, "series_1"...)
ds: datetime  # timestamp, tz-aware, ISO 8601 on the wire
y: float  # the target value

# optional columns, present only if the source has them:
# any column NOT in {unique_id, ds, y} is treated as an exogenous/covariate column.
# Cadence must track, per covariate, whether it is "known-future" (e.g. a holiday
# calendar, a planned promotion) or "past-only" (e.g. another sensor reading) —
# this distinction changes which models can use it.
```

Validation rules the IngestAgent must enforce before anything proceeds:
- `unique_id` non-null, `ds` parseable and monotonic per `unique_id`, `y` numeric
- Inferred frequency per `unique_id` (pandas `infer_freq` as a first pass, LLM fallback when ambiguous — e.g. business-day vs. calendar-day data)
- Duplicate `(unique_id, ds)` pairs rejected with a clear error, not silently dropped

## 6. Connector Layer

One connector per source type, each doing exactly one job: map source rows → canonical schema.

| Connector | Library | Notes |
|---|---|---|
| CSV / Parquet | `pandas` | column-mapping config (which column is `ds`, which is `y`) |
| SQL (Postgres/MySQL/SQLite) | `SQLAlchemy` | user supplies a query or table+column mapping |
| REST API | `httpx` | pagination handling, response → row mapping |
| Google Sheets | `gspread` | optional, only if requested |
| Streaming (Kafka) | `confluent-kafka` or `aiokafka` | v2 — batches into windows before hitting the rest of the pipeline; do not build this in v1 |

Each connector returns a `pandas.DataFrame` in the canonical schema plus a small `SourceMeta` object (source type, ingestion timestamp, row count) that gets attached to the run's audit trail.

---

## 7. Agent Specifications

### 7.1 IngestAgent
**Input:** raw source config (path/connection string/URL + column mapping)
**Output:** canonical DataFrame + `SourceMeta`
**Logic:** deterministic, no LLM call needed here — this is pure data engineering, not reasoning. Keep it that way; don't spend LLM calls on plumbing.

### 7.2 DiagnosticAgent (the "Curator")
**Input:** canonical DataFrame (one or more `unique_id`s)
**Output:** `diagnostics.json` per series + a cleaned DataFrame

Computes, per series:
- Length, frequency, date range, % missing
- Trend presence (simple linear-fit slope significance, or STL decomposition)
- Seasonality strength + period (STL decomposition; ACF peak detection)
- Stationarity (ADF test *and* KPSS test — they test opposite null hypotheses, run both, don't rely on one)
- Outliers (rolling-window IQR, flagged not silently removed)
- Intermittency (for demand data: is this series mostly zeros? — changes model choice entirely, e.g. Croston's method territory)

Then makes an LLM call with these stats as structured input (not the raw series — keep tokens down) to decide: missing-value strategy (interpolate / forward-fill / model-based), outlier treatment (clip / keep-and-flag / remove), and whether to log-transform or difference the series. This is the step that makes the harness adaptive rather than a fixed script — the same code path runs on every series, but the *decision* it reaches is data-dependent.

```python
# diagnostics.json shape
{
    "unique_id": "sku_1042",
    "length": 480,
    "freq": "D",
    "pct_missing": 0.02,
    "trend": {"present": true, "direction": "up"},
    "seasonality": {"present": true, "period": 7, "strength": 0.62},
    "stationarity": {"adf_pvalue": 0.31, "kpss_pvalue": 0.02, "verdict": "non_stationary"},
    "outlier_count": 4,
    "intermittent": false,
    "recommended_preprocessing": {
        "missing_strategy": "interpolate_linear",
        "outlier_strategy": "clip_at_p99",
        "transform": "log",
    },
}
```

### 7.3 PlannerAgent
**Input:** `diagnostics.json` + run config (horizon, how many series, latency budget)
**Output:** a shortlist of 2–5 candidate models across tiers, with initial hyperparameters

Decision logic (this is where "proper TSA models" gets decided per-series, not globally):

| Data shape | Tier(s) to shortlist |
|---|---|
| Single series, short history (<~200 points), clear seasonality | Classical: AutoARIMA, AutoETS, Theta |
| Intermittent/sparse demand | Classical: Croston, TSB (both in StatsForecast) |
| Thousands of related series, tabular covariates available | ML: LightGBM/XGBoost via MLForecast with lag + calendar features |
| Long history, complex nonlinear patterns, enough data to justify it | Deep learning: N-BEATS, N-HiTS, TFT via NeuralForecast |
| New series with little/no history, or "give me a fast reasonable baseline now" | Foundation model zero-shot: Chronos-Bolt first choice (fastest, most mature); Moirai if multivariate/covariates matter; TimesFM as a second opinion |
| Any ambiguous case | Shortlist one model from two tiers and let the backtest in §7.4 decide |

The LLM's job here is *not* to invent hyperparameters from scratch — it's to pick which of the above rules apply when diagnostics are borderline (e.g., "is 180 points 'short' or not, given a period-7 seasonality?"), and to justify the choice in the report.

### 7.4 ForecastAgent
**Input:** cleaned series + candidate model shortlist
**Output:** fitted models, backtest scores, final forecast + intervals

1. **Rolling-window backtest first, always.** Never fit on 100% of history and trust the in-sample fit. Use expanding-window or sliding-window cross-validation (`statsforecast`'s and `darts`' built-in `cross_validation` utilities do this correctly out of the box — do not hand-roll a random `train_test_split`, it's wrong for time series).
2. Score each candidate on held-out windows using **MASE** (scale-free, comparable across series) and **sMAPE** as the primary pair; add **WQL** (weighted quantile loss) for any model producing probabilistic/quantile output.
3. If no single model is a clear, consistent winner across backtest windows — which the M-competition results show is the common case — ensemble: start with a simple weighted average (weight ∝ inverse backtest error) before reaching for a learned stacker.
4. Refit the winning model(s)/ensemble on full history, produce the final `horizon`-step forecast with prediction intervals.

### 7.5 ReportAgent
**Input:** everything above
**Output:** a structured report (JSON for machine consumption + a rendered Markdown/HTML summary for humans)

Must state, in plain language: what the data looked like, what preprocessing was applied and why, which models were tried, which won the backtest and by how much, and the final forecast with confidence bands. This is the "white-box, not black-box" requirement — the whole point of the harness over a bare model call is that every decision is inspectable.

### 7.6 Shared State & Orchestration

Use **LangGraph** for the control flow — a single `CadenceState` object (a `TypedDict` or Pydantic model) passed through each node, each agent reading what it needs and appending its own output:

```python
class CadenceState(TypedDict):
    source_meta: dict
    raw_df: "pd.DataFrame"
    diagnostics: dict  # per unique_id
    cleaned_df: "pd.DataFrame"
    candidate_models: dict  # per unique_id: list of model configs
    backtest_scores: dict
    forecasts: dict  # per unique_id: point + interval forecast
    report: dict
    errors: list  # accumulate, don't crash the whole run on one bad series
```

Build in retry/backoff for LLM rate limits, and make sure one bad series in a 5,000-series batch logs to `errors` and continues rather than halting the whole run.

### 7.7 LLM Access — One Client, Everywhere (both LLM-using agents)

All LLM calls in the harness (DiagnosticAgent §7.2, PlannerAgent §7.3) go through **one injectable client interface** in `llm/client.py` — never a provider SDK import inside an agent:

```python
class LLMClient(Protocol):
    async def complete(
        self, messages: list[dict], response_schema: type[BaseModel]
    ) -> BaseModel: ...
```

- **Default implementation:** LiteLLM, so OpenAI / Anthropic / a local model is a config-string swap (`llm.provider` + `llm.model` in `default_config.py`), not a code change. LangGraph is the orchestrator; the LLM choice stays swappable.
- **Disabled mode is mandatory:** with `llm.enabled = false`, every agent that would call the LLM must still work — DiagnosticAgent falls back to deterministic defaults (interpolate_linear / clip_at_p99 / no transform), PlannerAgent relies purely on the §7.3 rule table. This is required for cost control and for running the test suite with no keys and no network.
- **Structured output only:** both agents return Pydantic models (the `recommended_preprocessing` block, the model shortlist) validated before anything enters `CadenceState` — malformed LLM output is retried, then logged to `errors`, never propagated as a raw string.

---

## 8. Model Zoo — Full Detail

### Tier 1 — Classical statistical (fast, cheap, strong baseline, no training data needed beyond the series itself)
- AutoARIMA, AutoETS, AutoTheta, TBATS, Croston, TSB — all via **`statsforecast`** (Nixtla). This library is built for exactly this: fitting hundreds/thousands of classical models per second via Numba-compiled implementations.

### Tier 2 — Machine learning (scales across many related series, uses covariates well)
- LightGBM, XGBoost, linear models over lag + rolling-window + calendar features — via **`mlforecast`** (Nixtla). Same `unique_id/ds/y` input, no manual feature engineering needed for the lag features (the library generates them).

### Tier 3 — Deep learning (worth it with long history + enough related series to learn cross-series patterns)
- N-BEATS, N-HiTS, TFT, DeepAR, PatchTST — via **`neuralforecast`** (Nixtla) or **`darts`** (Unit8). Darts is the broader all-in-one toolkit if you'd rather have one dependency cover tiers 1–3 plus backtesting utilities; Nixtla's three separate libraries are faster and more purpose-built per tier. **Recommendation: start with the Nixtla trio for tiers 1–3 since it's one consistent API (`.fit()` / `.predict()` / `.cross_validation()`) across all three, and it's what the harness's PlannerAgent shortlisting logic above assumes.**

### Tier 4 — Foundation models (zero-shot, no training step, good cold-start baseline or fallback)

| Model | Org | Paper | Params | Best for |
|---|---|---|---|---|
| Chronos-Bolt / Chronos-2 | Amazon | Ansari et al., 2024 | 9M–710M | Fastest, most production-mature, best default zero-shot pick |
| Moirai / Moirai-2 | Salesforce | Woo et al., 2024 | 14M–311M | Multivariate series, irregular timestamps, arbitrary covariates |
| TimesFM 2.5 | Google DeepMind | Das et al., 2024 | 200M–500M | Long-horizon forecasts, very large pretraining corpus |
| Lag-Llama | ServiceNow/Mila et al. | Rasul et al., 2023 | — | Probabilistic univariate forecasts via a LLaMA-style decoder |

Install each as its own uv extra, never together by default (uni2ts/timesfm pull JAX, which can fight the torch-based tiers during dependency resolution): `uv sync --extra chronos` (chronos-forecasting, Hugging Face weights, `amazon-science/chronos-forecasting`), `--extra moirai` (uni2ts, `SalesforceAIResearch/uni2ts`), `--extra timesfm` (`google-research/timesfm`). See §10 for the layout.

**How tier 4 fits the harness:** it's the PlannerAgent's answer for "new series, no history yet" or "just give me a fast baseline before the expensive tiers finish backtesting" — call it in parallel with tiers 1–3 and let the ForecastAgent's backtest decide if it's actually the winner for this series, don't assume it always is (the agricultural-commodities literature has already shown newer foundation-model versions sometimes underperform older ones or classical baselines on specific domains — treat every tier as a hypothesis to be backtested, not a foregone conclusion).

---

## 9. Evaluation & Backtesting Detail

- **Never random-split.** Time series cross-validation must respect order: expanding-window (grow the training set, fixed-size test windows moving forward) or sliding-window (fixed train size, slide forward).
- **Primary metrics:** MASE (Mean Absolute Scaled Error — scale-free, comparable across very different series), sMAPE (symmetric, bounded, human-interpretable as a percentage).
- **Probabilistic metric:** WQL (Weighted Quantile Loss) for any model emitting quantiles/intervals, not just point forecasts.
- **Minimum backtest windows:** at least 3 rolling windows per series before trusting a model ranking; fewer than that and the ranking is noise.

---

## 10. Tech Stack Summary

**Package management is uv, not pip.** Dependencies live in `pyproject.toml` with a committed `uv.lock`; there is no `requirements.txt`, and `pip install` is never run by hand. Heavy tiers (PyTorch/JAX-based) live in **optional groups/extras** so the default install stays light and CI stays fast. Pin `requires-python = "3.11"` (`.python-version`) — several libraries below lag on 3.13, and this avoids per-machine drift.

| Layer | Choice | Goes in |
|---|---|---|
| Data handling | pandas, pyarrow | core dependencies |
| DB connector | SQLAlchemy | core dependencies |
| API connector | httpx | core dependencies |
| Diagnostics/stats | statsmodels | core dependencies |
| Classical models | statsforecast | core dependencies |
| ML models | mlforecast + lightgbm | `[dependency-groups]` → `ml` |
| Deep learning models | neuralforecast | `[dependency-groups]` → `dl` |
| Foundation models | chronos-forecasting (first), uni2ts, timesfm | `[project.optional-dependencies]` → `chronos`, `moirai`, `timesfm` (one extra per model — uni2ts/timesfm pull JAX and can conflict with torch-based tiers during resolution) |
| Alt all-in-one toolkit (optional) | darts | `[project.optional-dependencies]` → `darts` (`darts[torch]`) |
| Agent orchestration | LangGraph | core dependencies |
| LLM client | LiteLLM (provider-agnostic, see §7.7) | `[dependency-groups]` → `llm` |
| API layer | FastAPI + uvicorn | core dependencies |
| Config/validation | Pydantic **v2** — pin the major version explicitly, don't rely on whatever FastAPI bundles | core dependencies |
| Testing | pytest + pytest-asyncio (async graph nodes, FastAPI endpoints) | `[dependency-groups]` → `dev` |
| Lint/format | ruff (one tool, both jobs) | `[dependency-groups]` → `dev` |

Working commands (always via `uv run`, never bare `pytest`/`python`):

```bash
uv sync                          # core only — enough through Phase 2
uv sync --all-groups             # dev + llm + ml + dl
uv sync --extra chronos          # tier 4, lightest first
uv run pytest
```

---

## 11. Repository Structure

```
cadence/
├── connectors/
│   ├── csv_connector.py
│   ├── sql_connector.py
│   ├── api_connector.py
│   └── base.py                  # shared canonical-schema validation
├── agents/
│   ├── ingest_agent.py
│   ├── diagnostic_agent.py
│   ├── planner_agent.py
│   ├── forecast_agent.py
│   └── report_agent.py
├── llm/
│   └── client.py                 # provider-agnostic LLMClient protocol + LiteLLM impl + disabled no-op (§7.7)
├── models/
│   ├── classical.py              # statsforecast wrappers
│   ├── ml.py                     # mlforecast wrappers
│   ├── deep_learning.py          # neuralforecast wrappers
│   └── foundation.py             # chronos / moirai / timesfm wrappers
├── graph/
│   └── cadence_graph.py          # LangGraph state machine definition
├── eval/
│   ├── backtest.py                # rolling-window CV utilities
│   └── metrics.py                 # MASE, sMAPE, WQL
├── api/
│   └── main.py                    # FastAPI app exposing /forecast, /diagnostics
├── config/
│   └── default_config.py
├── data/
│   └── sample/                    # committed fixtures: air_passengers.csv + synthetic stationary/trending/seasonal CSVs (see §12)
├── tests/
│   └── conftest.py                # shared fixtures: canonical-df builders, sample-data loaders
├── pyproject.toml                 # uv-managed: deps + dependency groups + extras (§10)
├── uv.lock                        # committed
├── .python-version                # 3.11
└── README.md
```

---

## 12. Build Phases (task order for the coding agent)

1. **Phase 0 — Scaffolding:** repo structure above, uv project init (`pyproject.toml` + committed `uv.lock` + `.python-version`, dependency groups and extras per §10), ruff + pytest configured, canonical schema + Pydantic v2 validation model, one CSV connector, one round-trip test (CSV in → validated canonical DataFrame out), and **commit the sample fixtures** under `data/sample/`: the classic AirPassengers series plus 2–3 small synthetic CSVs (stationary / trending / seasonal, ~200 points each). These double as Phase 2's known-answer test inputs and keep the test suite network-free.
2. **Phase 1 — Classical MVP, no agents yet:** wire `statsforecast` directly (AutoARIMA + AutoETS) on the canonical schema, with `cross_validation()` backtesting and MASE/sMAPE scoring. Get one series forecasting end-to-end before touching any LLM logic.
3. **Phase 2 — DiagnosticAgent:** implement the stats computation (ADF/KPSS/STL/outliers) as pure functions first, test them against known synthetic series (stationary vs trending vs seasonal) before wrapping the LLM call around them.
4. **Phase 3 — PlannerAgent:** implement the rule table in §7.3 as deterministic code first; add the LLM call only for the borderline-case arbitration, and make it optional/overridable via config per §7.7 so the harness still works with the LLM call disabled (important for cost control and for testing).
5. **Phase 4 — Expand model zoo:** add `mlforecast` (Tier 2) and `neuralforecast` (Tier 3) behind the same interface the classical models use — all should expose `.fit(df) -> .predict(h) -> forecast_df`.
6. **Phase 5 — Foundation model tier:** add Chronos-Bolt as the first zero-shot option (lightest weight, fastest to integrate), then Moirai/TimesFM if the project needs multivariate or long-horizon cases.
7. **Phase 6 — Ensembling + full ForecastAgent:** implement weighted-average ensembling based on backtest scores.
8. **Phase 7 — LangGraph wiring:** assemble Phases 0–6 into the actual agent graph with `CadenceState`, error accumulation, and retries.
9. **Phase 8 — ReportAgent + API layer:** FastAPI endpoints (`POST /forecast`, `GET /diagnostics/{unique_id}`), Markdown/HTML report rendering.
10. **Phase 9 — Additional connectors:** SQL, REST API connectors once the core pipeline is proven on CSV.
11. **Phase 10 (optional, v2):** streaming connector, a thin chat/Telegram interface on top of the API layer, multi-tenant scheduling.

Build and test each phase against a small, known public dataset (e.g., the classic `AirPassengers` series for a quick single-series sanity check, plus a multi-series retail dataset like M5 or the ETT benchmark subset for the "thousands of series" path) before moving to the next phase.

---

## 13. References (Research Papers)

These are the papers the harness's design choices are drawn from — worth having the coding agent read the abstracts/READMEs of the first two before writing the agent graph.

1. **Zhao, H., Zhang, X., Wei, J., Xu, Y., He, Y., Sun, S., You, C. (2025).** *TimeSeriesScientist: A General-Purpose AI Agent for Time Series Analysis.* arXiv:2510.01538. Code: `github.com/Y-Research-SBU/TimeSeriesScientist` — **direct architectural precedent for this project; read the repo, not just the paper.**
2. **Ansari, A.F. et al. (2024).** *Chronos: Learning the Language of Time Series.* arXiv:2403.07815, TMLR 2024. Code: `github.com/amazon-science/chronos-forecasting`.
3. **Woo, G., Liu, C., Kumar, A., Xiong, C., Savarese, S., Sahoo, D. (2024).** *Unified Training of Universal Time Series Forecasting Transformers* (Moirai). arXiv:2402.02592, ICML 2024. Code: `github.com/SalesforceAIResearch/uni2ts`.
4. **Das, A., Kong, W., Sen, R., Zhou, Y. (2024).** *A decoder-only foundation model for time-series forecasting* (TimesFM). arXiv:2310.10688, ICML 2024. Code: `github.com/google-research/timesfm`.
5. **Rasul, K. et al. (2023).** *Lag-Llama: Towards Foundation Models for Probabilistic Time Series Forecasting.* arXiv:2310.08278.
6. **"From Prompts to Agents: A Comprehensive Survey of LLM-Driven Time Series Analysis" (2025).** Reviews 150+ studies; taxonomy of agent architectures across perception, planning, tool use, memory, and reflection — useful for validating that §7's agent split covers the field's standard components. Repo: `github.com/CoderPowerBeyond/Agent-Prompt-TS-Survey`.
7. **"A Survey of Reasoning and Agentic Systems in Time Series with Large Language Models" (2025).** arXiv:2509.11575 — complementary taxonomy, additional benchmark references.

---

## 14. Open Design Decisions (flag these back to the user, don't guess silently)

- Single-user local tool vs. eventually multi-tenant service — affects whether the API layer needs auth/tenancy now or can be added later
- Which LLM provider/model for the agent reasoning calls — **the interface is now settled (§7.7: one `LLMClient` protocol, LiteLLM default, hard off-switch); still open is which provider/model to default to** (cost vs. capability tradeoff), and whether the foundation-model tier needs GPU access at all — CPU-only deployments should still work with Tiers 1–2 and Chronos-Bolt's smaller checkpoints
- Expected scale: tens of series (interactive, can afford deep learning + foundation models per series) vs. thousands (needs the ML tier to carry most of the load, deep learning/foundation models reserved for a sampled subset or as a fallback)
