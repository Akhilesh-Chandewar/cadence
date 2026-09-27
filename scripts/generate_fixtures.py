"""Generate the committed fixtures under data/sample/ (spec §12, Phase 0).

Deterministic (seed 42) and network-free: AirPassengers is embedded, the synthetic
series are seeded. Re-run any time with:

    uv run python scripts/generate_fixtures.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "sample"

# The classic Box & Jenkins international airline passengers series,
# monthly totals (thousands), Jan 1949 – Dec 1960 (144 rows).
AIR_PASSENGERS = [
    112,
    118,
    132,
    129,
    121,
    135,
    148,
    148,
    136,
    119,
    104,
    118,
    115,
    126,
    141,
    135,
    125,
    149,
    170,
    170,
    158,
    133,
    114,
    140,
    145,
    150,
    178,
    163,
    172,
    178,
    199,
    199,
    184,
    162,
    146,
    166,
    171,
    180,
    193,
    181,
    183,
    218,
    230,
    242,
    209,
    191,
    172,
    194,
    196,
    196,
    236,
    235,
    229,
    243,
    264,
    272,
    237,
    211,
    180,
    201,
    204,
    188,
    235,
    227,
    234,
    264,
    302,
    293,
    259,
    229,
    203,
    229,
    242,
    233,
    267,
    269,
    270,
    315,
    364,
    347,
    312,
    274,
    237,
    278,
    284,
    277,
    317,
    313,
    318,
    374,
    413,
    405,
    355,
    306,
    271,
    306,
    315,
    301,
    356,
    348,
    355,
    422,
    465,
    467,
    404,
    347,
    305,
    336,
    340,
    318,
    362,
    348,
    363,
    435,
    491,
    505,
    404,
    359,
    310,
    337,
    360,
    342,
    406,
    396,
    420,
    472,
    548,
    559,
    463,
    407,
    362,
    405,
    417,
    391,
    419,
    461,
    472,
    535,
    622,
    606,
    508,
    461,
    390,
    432,
]


def air_passengers_df() -> pd.DataFrame:
    ds = pd.date_range("1949-01-01", periods=144, freq="MS", tz="UTC")
    return pd.DataFrame(
        {"unique_id": "air_passengers", "ds": ds, "y": [float(v) for v in AIR_PASSENGERS]}
    )


def synthetic_stationary_df(n: int = 200) -> pd.DataFrame:
    """White noise around a fixed mean — ADF should reject a unit root."""
    rng = np.random.default_rng(SEED)
    ds = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    y = 10.0 + rng.normal(0.0, 1.0, size=n)
    return pd.DataFrame({"unique_id": "synthetic_stationary", "ds": ds, "y": y})


def synthetic_trending_df(n: int = 200) -> pd.DataFrame:
    """Deterministic upward drift + noise — trend test must fire; ADF should NOT reject."""
    rng = np.random.default_rng(SEED + 1)
    ds = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    y = 0.15 * np.arange(n, dtype=float) + rng.normal(0.0, 2.0, size=n)
    return pd.DataFrame({"unique_id": "synthetic_trending", "ds": ds, "y": y})


def synthetic_seasonal_df(n: int = 196) -> pd.DataFrame:
    """Strong weekly seasonality (period 7), mild trend — STL/ACF must find period 7.

    196 = 28 full weeks; seasonality amplitude (3.0) is large vs noise (0.5).
    """
    rng = np.random.default_rng(SEED + 2)
    ds = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    t = np.arange(n, dtype=float)
    y = 50.0 + 0.02 * t + 3.0 * np.sin(2 * np.pi * t / 7.0) + rng.normal(0.0, 0.5, size=n)
    return pd.DataFrame({"unique_id": "synthetic_seasonal", "ds": ds, "y": y})


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fixtures = {
        "air_passengers.csv": air_passengers_df(),
        "synthetic_stationary.csv": synthetic_stationary_df(),
        "synthetic_trending.csv": synthetic_trending_df(),
        "synthetic_seasonal.csv": synthetic_seasonal_df(),
    }
    for name, df in fixtures.items():
        path = OUT_DIR / name
        df.to_csv(path, index=False)
        print(f"wrote {path.relative_to(OUT_DIR.parent.parent)}  ({len(df)} rows)")


if __name__ == "__main__":
    main()
