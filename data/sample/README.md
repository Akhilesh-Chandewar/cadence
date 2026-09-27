# Sample fixtures (spec §12, Phase 0)

Committed, deterministic (seed 42), network-free — the test suite only ever reads these
files. Regenerate with `uv run python scripts/generate_fixtures.py`.

| File | unique_id | Shape | Known-answer role |
|---|---|---|---|
| `air_passengers.csv` | `air_passengers` | 144 rows, monthly, 1949–1960 | Classic Box & Jenkins series; single-series sanity check + strong multiplicative seasonality |
| `synthetic_stationary.csv` | `synthetic_stationary` | 200 rows, daily | White noise around a constant mean — Phase 2 tests: stationary verdict |
| `synthetic_trending.csv` | `synthetic_trending` | 200 rows, daily | Slope 0.15/day + noise — Phase 2 tests: trend present |
| `synthetic_seasonal.csv` | `synthetic_seasonal` | 196 rows, daily | Weekly sine (period 7) + mild trend — Phase 2 tests: seasonality period 7, strong |

All files are canonical-schema CSVs (`unique_id, ds, y`), tz-aware UTC timestamps.
