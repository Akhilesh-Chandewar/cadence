"""Shared fixtures: canonical-df builders and data/sample/ loaders (spec §11)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "data" / "sample"


def make_canonical_df(
    uids: list[str] | None = None,
    n: int = 30,
    freq: str = "D",
    tz: str | None = "UTC",
) -> pd.DataFrame:
    """Build a small valid canonical DataFrame (optionally multi-series)."""
    uids = uids or ["series_1"]
    frames = []
    for i, uid in enumerate(uids):
        ds = pd.date_range("2024-01-01", periods=n, freq=freq, tz=tz) + pd.Timedelta(days=i)
        y = [float(j + i) for j in range(n)]
        frames.append(pd.DataFrame({"unique_id": uid, "ds": ds, "y": y}))
    return pd.concat(frames, ignore_index=True)


@pytest.fixture
def canonical_df() -> pd.DataFrame:
    return make_canonical_df()


@pytest.fixture
def multi_series_df() -> pd.DataFrame:
    return make_canonical_df(uids=["a", "b", "c"])


@pytest.fixture
def sample_dir() -> Path:
    return SAMPLE_DIR


@pytest.fixture(scope="session")
def air_passengers_csv() -> Path:
    return SAMPLE_DIR / "air_passengers.csv"


@pytest.fixture(scope="session")
def synthetic_stationary_csv() -> Path:
    return SAMPLE_DIR / "synthetic_stationary.csv"


@pytest.fixture(scope="session")
def synthetic_trending_csv() -> Path:
    return SAMPLE_DIR / "synthetic_trending.csv"


@pytest.fixture(scope="session")
def synthetic_seasonal_csv() -> Path:
    return SAMPLE_DIR / "synthetic_seasonal.csv"
