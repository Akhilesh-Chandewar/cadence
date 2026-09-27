"""Classical model wrappers over statsforecast (spec §8 Tier 1, Phase 1).

Every wrapper model exposes the same shape the rest of the zoo will follow:
input is the canonical (unique_id, ds, y) frame, output is a statsforecast
prediction frame with per-model point columns plus `-lo-{level}` / `-hi-{level}`
interval columns.

Note on timestamps: statsforecast's numba paths want tz-naive datetimes, so the
wrapper standardizes to naive UTC internally (the canonical schema is already
UTC-normalized upstream in connectors/base.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, AutoETS, AutoTheta, CrostonOptimized

# Phase 1 registry (§7.3 shortlist names → statsforecast classes).
# Croston is included now because intermittent demand (§7.3 row 2) needs no
# season_length; TSB arrives with the intermittent path.
# Note: statsforecast 2.x split `Croston` into variants — "Croston" maps to the
# optimized variant, which is what the old class aliased.
MODEL_REGISTRY: dict[str, type] = {
    "AutoARIMA": AutoARIMA,
    "AutoETS": AutoETS,
    "AutoTheta": AutoTheta,
    "Croston": CrostonOptimized,
}

DEFAULT_CLASSICAL_MODELS: tuple[str, ...] = ("AutoARIMA", "AutoETS")

# frequency → seasonal period for season_length (m) selection
_SEASONALITY_BY_FREQ: dict[str, int] = {
    "h": 24,
    "D": 7,
    "B": 5,
    "W": 52,
    "MS": 12,
    "M": 12,
    "QS": 4,
    "Q": 4,
    "YS": 1,
    "Y": 1,
}


def infer_frequency(df: pd.DataFrame) -> str:
    """Infer a pandas frequency string from the frame (first series wins, Phase 1)."""
    first = df["unique_id"].iloc[0]
    ds = df.loc[df["unique_id"] == first, "ds"]
    freq = pd.infer_freq(pd.DatetimeIndex(ds.sort_values()))
    if freq is None:
        raise ValueError("could not infer frequency; pass freq explicitly (e.g. 'D', 'MS')")
    return freq


def normalize_frequency(freq: str) -> str:
    """Map pandas offsets like 'ME'/'W-SUN' onto the base keys used for seasonality."""
    base = freq.split("-")[0].upper()
    aliases = {"ME": "M", "YE": "Y", "W": "W", "SME": "QS", "SE": "Q"}
    return aliases.get(base, base if base in _SEASONALITY_BY_FREQ else base.lower())


def infer_season_length(df: pd.DataFrame, freq: str | None = None) -> int:
    """Seasonal period m for the inferred (or given) frequency; 1 when unknown."""
    freq = normalize_frequency(freq or infer_frequency(df))
    return _SEASONALITY_BY_FREQ.get(freq, 1)


@dataclass
class ClassicalModelConfig:
    models: tuple[str, ...] = DEFAULT_CLASSICAL_MODELS
    freq: str | None = None  # inferred from data when None
    season_length: int | None = None  # inferred from freq when None
    level: tuple[int, ...] = (95,)

    def __post_init__(self) -> None:
        unknown = [m for m in self.models if m not in MODEL_REGISTRY]
        if unknown:
            raise ValueError(f"unknown model(s) {unknown}; registry: {sorted(MODEL_REGISTRY)}")


@dataclass
class ClassicalForecaster:
    """Tier-1 entry point: canonical df in, statsforecast predictions out.

    Stateless by design — `fit` only stages the data; each forecast/cross_validation
    call hands the full history to StatsForecast, which fits on demand.
    """

    config: ClassicalModelConfig = field(default_factory=ClassicalModelConfig)
    _df: pd.DataFrame | None = None
    _sf: StatsForecast | None = None

    def fit(self, df: pd.DataFrame) -> ClassicalForecaster:
        missing = {"unique_id", "ds", "y"} - set(df.columns)
        if missing:
            raise ValueError(f"canonical schema violation: missing column(s) {sorted(missing)}")

        work = df[["unique_id", "ds", "y"]].copy()
        # tz-aware → naive UTC for statsforecast's numba paths
        if isinstance(work["ds"].dtype, pd.DatetimeTZDtype):
            work["ds"] = work["ds"].dt.tz_convert("UTC").dt.tz_localize(None)

        freq = self.config.freq or infer_frequency(work)
        season_length = self.config.season_length or infer_season_length(work, freq)

        model_objs: list[Any] = []
        for name in self.config.models:
            cls = MODEL_REGISTRY[name]
            if name == "Croston":
                model_objs.append(cls(alias=name))  # no season_length parameter
            else:
                model_objs.append(cls(season_length=season_length, alias=name))

        self._df = work
        self._sf = StatsForecast(models=model_objs, freq=freq, n_jobs=-1)
        return self

    def _require_fitted(self) -> StatsForecast:
        if self._sf is None or self._df is None:
            raise RuntimeError("call fit(df) before forecasting")
        return self._sf

    def forecast(self, h: int) -> pd.DataFrame:
        """Fit on full history, forecast h steps with prediction intervals (wide format)."""
        sf = self._require_fitted()
        return sf.forecast(h=h, df=self._df, level=list(self.config.level))

    def cross_validation(
        self,
        h: int,
        n_windows: int = 3,
        step_size: int | None = None,
        input_size: int | None = None,
    ) -> pd.DataFrame:
        """Rolling-origin CV (spec §9: never random-split).

        Returns long frame: unique_id, ds, cutoff, y, <model>, <model>-lo-{l}, <model>-hi-{l}.
        """
        sf = self._require_fitted()
        return sf.cross_validation(
            h=h,
            df=self._df,
            n_windows=n_windows,
            step_size=step_size if step_size is not None else h,
            input_size=input_size,
            level=list(self.config.level),
        )
