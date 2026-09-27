"""Tier 4 — Chronos-Bolt zero-shot foundation model (spec §8).

Zero-shot: no per-series fitting at all — `fit()` stages data and loads the
pretrained pipeline once. Same Forecaster contract as tiers 1–3: canonical df in,
wide prediction frame out (`Chronos-Bolt`, `Chronos-Bolt-lo-95`, `Chronos-Bolt-hi-95`),
and a rolling-origin CV frame the shared §9 scorer consumes.

Imports chronos/torch lazily via require_group — installed with `uv sync --extra chronos`
(one extra per foundation model, spec §10; never co-installed with the JAX-based tiers).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from cadence.config.default_config import ChronosConfig
from cadence.models.base import require_group

MODEL_NAME = "Chronos-Bolt"
# expanding-window CV needs a minimal context to be meaningful at all
_MIN_CV_CONTEXT = 16


@dataclass
class ChronosForecaster:
    """Tier-4 zero-shot forecaster (spec §8: 'a hypothesis to be backtested')."""

    config: ChronosConfig = field(default_factory=ChronosConfig)
    _pipeline: object | None = None
    _df: pd.DataFrame | None = None
    _freq: str = "D"

    # ------------------------------------------------------------------ fit
    def fit(self, df: pd.DataFrame) -> ChronosForecaster:
        require_group("chronos", "extra:chronos")
        import torch
        from chronos import BaseChronosPipeline

        missing = {"unique_id", "ds", "y"} - set(df.columns)
        if missing:
            raise ValueError(f"canonical schema violation: missing column(s) {sorted(missing)}")

        work = df[["unique_id", "ds", "y"]].copy()
        if isinstance(work["ds"].dtype, pd.DatetimeTZDtype):
            work["ds"] = work["ds"].dt.tz_convert("UTC").dt.tz_localize(None)

        from cadence.models.classical import infer_frequency

        self._freq = infer_frequency(work)
        self._pipeline = BaseChronosPipeline.from_pretrained(
            self.config.model_id,
            torch_dtype=getattr(torch, self.config.torch_dtype),
            device_map=self.config.device,
        )
        self._df = work
        return self

    def _require_staged(self) -> None:
        if self._pipeline is None or self._df is None:
            raise RuntimeError("call fit(df) before predicting")

    # ------------------------------------------------- pipeline interaction
    def _pipeline_quantiles(
        self, context: np.ndarray, h: int, quantile_levels: list[float]
    ) -> tuple[np.ndarray, np.ndarray]:
        """One zero-shot call. Returns (quantiles (h, n_levels), mean (h,)).

        Isolated so tests can stub the model without weights or network.
        """
        import torch

        tensor = torch.tensor(context, dtype=torch.float32)
        quantiles, mean = self._pipeline.predict_quantiles(
            [tensor], prediction_length=h, quantile_levels=quantile_levels
        )
        return (
            quantiles[0].cpu().numpy(),
            mean[0].cpu().numpy(),
        )

    @staticmethod
    def _levels_for(level: tuple[int, ...]) -> list[float]:
        """Interval level L% → quantile pair ((1-L/100)/2, 1-(1-L/100)/2)."""
        levels: list[float] = []
        for lv in level:
            alpha = (100 - lv) / 200
            levels.extend([alpha, 1 - alpha])
        return sorted(set(levels))

    # --------------------------------------------------------------- predict
    def _forecast_frame(
        self, h: int, levels: tuple[int, ...], context: pd.DataFrame
    ) -> pd.DataFrame:
        """Zero-shot forecast per series from `context`, wide-format rows."""
        q_levels = self._levels_for(levels)
        parts: list[pd.DataFrame] = []
        for uid, grp in context.groupby("unique_id", sort=False):
            y = grp["y"].to_numpy(float)
            if len(y) < 2:
                raise ValueError(f"series {uid!r} too short for zero-shot forecasting")

            if q_levels:
                q, mean = self._pipeline_quantiles(y, h, [0.5, *q_levels])
                # q columns follow the requested order we passed: 0.5 then pairs
            else:
                _, mean = self._pipeline_quantiles(y, h, [0.5])
                q = None

            ds = grp["ds"].reset_index(drop=True)
            future = pd.date_range(ds.iloc[-1], periods=h + 1, freq=self._freq)[1:]
            frame = pd.DataFrame({"unique_id": uid, "ds": future, MODEL_NAME: mean.astype(float)})
            if q is not None:
                for suffix, idx in self._level_column_map(levels, [0.5, *q_levels]):
                    frame[f"{MODEL_NAME}-{suffix}"] = q[:, idx].astype(float)
            parts.append(frame)
        return pd.concat(parts, ignore_index=True)

    @staticmethod
    def _level_column_map(
        levels: tuple[int, ...], ordered_levels: list[float]
    ) -> list[tuple[int, int]]:
        """Map (lo/hi suffix, column index) pairs for the requested interval levels."""
        out = []
        for lv in levels:
            alpha = (100 - lv) / 200
            lo_idx = ordered_levels.index(alpha)
            hi_idx = ordered_levels.index(1 - alpha)
            out.append((f"lo-{lv}", lo_idx))
            out.append((f"hi-{lv}", hi_idx))
        return out

    def predict(self, h: int, level: tuple[int, ...] = (95,)) -> pd.DataFrame:
        self._require_staged()
        return self._forecast_frame(h, level, self._df)

    # zoo-wide alias
    def forecast(self, h: int, level: tuple[int, ...] = (95,)) -> pd.DataFrame:
        return self.predict(h=h, level=level)

    # -------------------------------------------------------- cross validation
    def cross_validation(
        self,
        h: int,
        n_windows: int = 3,
        step_size: int | None = None,
    ) -> pd.DataFrame:
        """Rolling-origin CV for a zero-shot model: refitting is free (no training),
        so every window forecasts from its own expanding context (§9-honest)."""
        self._require_staged()
        step = step_size if step_size is not None else h
        rows: list[pd.DataFrame] = []

        for uid, grp in self._df.groupby("unique_id", sort=False):
            grp = grp.sort_values("ds").reset_index(drop=True)
            n = len(grp)
            first_start = n - h - (n_windows - 1) * step
            if first_start < _MIN_CV_CONTEXT:
                raise ValueError(
                    f"series {uid!r} too short for {n_windows} windows of h={h} "
                    f"step={step}: needs >= {_MIN_CV_CONTEXT} training points in the "
                    f"first window, has {first_start}"
                )
            for w in range(n_windows):
                test_start = first_start + w * step
                train = grp.iloc[:test_start]
                test = grp.iloc[test_start : test_start + h]

                q_levels = [0.5]
                q, mean = self._pipeline_quantiles(train["y"].to_numpy(float), len(test), q_levels)
                rows.append(
                    pd.DataFrame(
                        {
                            "unique_id": uid,
                            "ds": test["ds"].to_numpy(),
                            "cutoff": train["ds"].iloc[-1],
                            "y": test["y"].to_numpy(float),
                            MODEL_NAME: mean.astype(float),
                        }
                    )
                )
        return pd.concat(rows, ignore_index=True)
