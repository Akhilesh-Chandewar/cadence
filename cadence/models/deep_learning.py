"""Tier 3 — NeuralForecast deep learning (spec §8): N-HiTS / TFT via neuralforecast.

Same Forecaster contract as tiers 1–2. neuralforecast's predict() is horizon-fixed
(models are constructed with h), so fit() takes the horizon and predict()/forecast()
read it back — callers get the same `.fit(df) → .predict(h)` shape regardless.
Imports torch/neuralforecast lazily via require_group (spec §10).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from cadence.config.default_config import DLModelConfig
from cadence.models.base import require_group

_SUPPORTED_MODELS = ("NHITS", "TFT")


@dataclass
class DLForecaster:
    """Tier-3 forecaster: N-HiTS/TFT over the canonical schema (CPU-friendly defaults)."""

    config: DLModelConfig = field(default_factory=DLModelConfig)
    horizon: int = 12
    _nf: object | None = None
    _df: pd.DataFrame | None = None
    _freq: str = "D"

    def _build_models(self, h: int, freq: str) -> list:
        from neuralforecast.models import NHITS, TFT

        unknown = [m for m in self.config.models if m not in _SUPPORTED_MODELS]
        if unknown:
            raise ValueError(f"unsupported DL model(s) {unknown}; supported: {_SUPPORTED_MODELS}")

        kwargs = dict(
            h=h,
            input_size=self.config.input_size,
            max_steps=self.config.max_steps,
            scaler_type=self.config.scaler_type,
            enable_progress_bar=False,
        )
        # seasonality-aware encoder length: one cycle of the data's frequency
        from cadence.models.classical import infer_season_length

        season = infer_season_length(None, freq)
        kwargs["input_size"] = max(self.config.input_size, 2 * season)

        models: list = []
        for name in self.config.models:
            if name == "NHITS":
                models.append(NHITS(**kwargs))
            elif name == "TFT":
                models.append(TFT(**kwargs))
        return models

    def fit(self, df: pd.DataFrame, h: int | None = None) -> DLForecaster:
        require_group("neuralforecast", "dl")
        import torch
        from neuralforecast import NeuralForecast

        torch.set_num_threads(max(1, torch.get_num_threads() // 2))

        missing = {"unique_id", "ds", "y"} - set(df.columns)
        if missing:
            raise ValueError(f"canonical schema violation: missing column(s) {sorted(missing)}")

        if h is not None:
            self.horizon = h

        work = df[["unique_id", "ds", "y"]].copy()
        if isinstance(work["ds"].dtype, pd.DatetimeTZDtype):
            work["ds"] = work["ds"].dt.tz_convert("UTC").dt.tz_localize(None)

        from cadence.models.classical import infer_frequency

        self._freq = infer_frequency(work)
        models = self._build_models(self.horizon, self._freq)
        self._nf = NeuralForecast(models=models, freq=self._freq, local_scaler_type="robust")
        self._df = work
        self._nf.fit(work)
        return self

    def _require_fitted(self):
        if self._nf is None or self._df is None:
            raise RuntimeError("call fit(df) before predicting")
        return self._nf

    def predict(self) -> pd.DataFrame:
        nf = self._require_fitted()
        return nf.predict()

    # zoo-wide alias — horizon is fixed at fit time for neural models
    def forecast(self, h: int | None = None) -> pd.DataFrame:
        if h is not None and h != self.horizon:
            raise ValueError(
                f"neuralforecast models are horizon-fixed (h={self.horizon}); "
                "pass h to fit() instead"
            )
        return self.predict()

    def cross_validation(
        self, h: int, n_windows: int = 3, step_size: int | None = None
    ) -> pd.DataFrame:
        nf = self._require_fitted()
        # neuralforecast requires CV horizon >= the models' fitted horizon
        h = max(h, self.horizon)
        # refit=True: retrain per cutoff — §9 forbids scoring the full-history fit
        return nf.cross_validation(
            df=self._df,
            h=h,
            n_windows=n_windows,
            step_size=step_size if step_size is not None else h,
            refit=True,
        )
