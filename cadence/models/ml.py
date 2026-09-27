"""Tier 2 — MLForecast + LightGBM (spec §8): tabular ML over generated lag features.

Same Forecaster contract as the classical tier: canonical df in, wide prediction
frame out with a `<name>` point column and optional `<name>-lo/hi-95` interval
columns (conformal prediction intervals, since LightGBM has none natively).
Imports mlforecast/lightgbm lazily via require_group so the core install stays
light (spec §10).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from cadence.config.default_config import MLModelConfig
from cadence.models.base import require_group

# pandas offset → mlforecast date_features set (calendar features the §7.3 row
# promises: "lag + calendar features")
_DATE_FEATURES_BY_FREQ: dict[str, list[str]] = {
    "D": ["dayofweek", "month"],
    "B": ["dayofweek", "month"],
    "W": ["month"],
    "MS": ["month", "quarter"],
    "M": ["month", "quarter"],
    "QS": ["quarter", "year"],
    "Q": ["quarter", "year"],
    "h": ["dayofweek", "hour"],
}


@dataclass
class MLForecaster:
    """Tier-2 forecaster: LightGBM over MLForecast-generated lag/rolling/calendar features."""

    config: MLModelConfig = field(default_factory=MLModelConfig)
    _mf: object | None = None
    _df: pd.DataFrame | None = None
    _freq: str = "D"
    _intervals_calibrated: bool = False

    def fit(self, df: pd.DataFrame) -> MLForecaster:
        require_group("mlforecast", "ml")
        require_group("lightgbm", "ml")
        from lightgbm import LGBMRegressor
        from mlforecast import MLForecast
        from mlforecast.conformal_prediction import PredictionIntervals
        from mlforecast.lag_transforms import RollingMean

        missing = {"unique_id", "ds", "y"} - set(df.columns)
        if missing:
            raise ValueError(f"canonical schema violation: missing column(s) {sorted(missing)}")

        work = df[["unique_id", "ds", "y"]].copy()
        if isinstance(work["ds"].dtype, pd.DatetimeTZDtype):
            work["ds"] = work["ds"].dt.tz_convert("UTC").dt.tz_localize(None)

        from cadence.models.classical import infer_frequency

        self._freq = infer_frequency(work)
        date_features = self.config.date_features or _DATE_FEATURES_BY_FREQ.get(self._freq, [])
        lag_transforms = {
            lag: [RollingMean(window_size=window)]
            for lag, window in self.config.rolling_mean_windows.items()
        }

        model = LGBMRegressor(
            num_leaves=self.config.num_leaves,
            learning_rate=self.config.learning_rate,
            n_estimators=self.config.n_estimators,
            verbose=-1,
            random_state=0,
        )
        self._mf = MLForecast(
            models={"MLForecast-LightGBM": model},
            freq=self._freq,
            lags=self.config.lags,
            lag_transforms=lag_transforms,
            date_features=date_features,
        )
        self._df = work
        # conformal intervals must be calibrated at fit time (mlforecast contract):
        # predict(h) then emits lo/hi columns for any h <= interval_horizon
        try:
            self._mf.fit(
                work,
                prediction_intervals=PredictionIntervals(
                    n_windows=2, h=self.config.interval_horizon
                ),
            )
        except ValueError:
            # too little data to calibrate — fall back to point forecasts only
            self._mf.fit(work)
            self._intervals_calibrated = False
        else:
            self._intervals_calibrated = True
        return self

    def _require_fitted(self):
        if self._mf is None or self._df is None:
            raise RuntimeError("call fit(df) before predicting")
        return self._mf

    def predict(self, h: int, level: tuple[int, ...] = (95,)) -> pd.DataFrame:
        mf = self._require_fitted()
        if level and h > self.config.interval_horizon:
            raise ValueError(
                f"predict(h={h}) exceeds the interval calibration horizon "
                f"({self.config.interval_horizon}); refit with a larger "
                "MLModelConfig.interval_horizon, or pass level=() for point forecasts only"
            )
        if level and not self._intervals_calibrated:
            import warnings

            warnings.warn(
                "data too small to calibrate conformal intervals — returning point forecasts only",
                stacklevel=2,
            )
            return mf.predict(h=h)
        return mf.predict(h=h, level=list(level) if level else None)

    # zoo-wide alias
    def forecast(self, h: int, level: tuple[int, ...] = (95,)) -> pd.DataFrame:
        return self.predict(h=h, level=level)

    def cross_validation(
        self,
        h: int,
        n_windows: int = 3,
        step_size: int | None = None,
    ) -> pd.DataFrame:
        mf = self._require_fitted()
        # note: mlforecast CV frames carry point columns only (no conformal intervals)
        return mf.cross_validation(
            df=self._df,
            h=h,
            n_windows=n_windows,
            step_size=step_size if step_size is not None else h,
        )
