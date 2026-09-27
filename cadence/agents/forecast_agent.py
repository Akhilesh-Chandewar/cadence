"""ForecastAgent (spec §7.4): shortlist → rolling backtest → best-or-ensemble → forecast.

Per series:
1. Backtest every implemented candidate with the §9 rolling-window harness.
2. Decide: a *clear* winner (beats #2 on mean MASE by ≥ winner_margin) that is also
   *consistent* (top model in ≥ consistency_ratio of that series' windows) wins
   outright; otherwise ensemble the top-k by inverse mean error (§7.4 step 3).
3. Produce the final horizon-step forecast with prediction intervals. Every
   forecaster is fit on the full history before predicting (CV frames come from
   internal rolling-origin refits, so full-history fit + predict is correct).

`DefaultModelFactory` maps catalog names to any tier's wrapper, so the agent is
tier-agnostic; candidates whose dependency group isn't synced are skipped with a
reason (not an error), while genuine failures accumulate per §7.6.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from cadence.agents.decisions import Transform
from cadence.agents.planner_agent import ModelShortlist
from cadence.config.default_config import CadenceConfig
from cadence.eval.backtest import BacktestResult, rolling_backtest
from cadence.models.base import MissingDependencyError
from cadence.models.classical import MODEL_REGISTRY, ClassicalForecaster, ClassicalModelConfig

_INTERVAL_SUFFIXES = ("-lo-95", "-hi-95")


class DefaultModelFactory:
    """Catalog name → configured Forecaster for the run's config."""

    def __init__(self, config: CadenceConfig) -> None:
        self.config = config

    def build(self, name: str):
        if name in MODEL_REGISTRY:
            return ClassicalForecaster(ClassicalModelConfig(models=(name,)))
        if name == "MLForecast-LightGBM":
            from cadence.models.ml import MLForecaster

            return MLForecaster(self.config.ml)
        if name in {"N-HiTS", "TFT"}:
            from cadence.models.deep_learning import DLForecaster

            dl_cfg = self.config.dl.model_copy(
                update={"models": ["NHITS" if name == "N-HiTS" else "TFT"]}
            )
            return DLForecaster(config=dl_cfg, horizon=self.config.forecast.horizon)
        if name == "Chronos-Bolt":
            from cadence.models.foundation import ChronosForecaster

            return ChronosForecaster(self.config.chronos)
        raise ValueError(f"unknown model name {name!r} (not in the catalog)")


@dataclass
class FittedModel:
    name: str
    forecaster: object
    aggregate: pd.DataFrame  # single-row aggregate for this model (mase_mean, ...)
    consistency: float  # share of that series' windows where this model ranked first
    wql: float | None = None  # §9 probabilistic score, when the CV frame exposes intervals
    weight: float | None = None  # set when the model joins the ensemble
    model_col: str | None = None  # point column when the forecaster emits several (classical)


@dataclass
class ForecastOutput:
    unique_id: str
    decision: str  # "best" or "ensemble"
    selected: list[str]
    weights: dict[str, float]
    point: pd.DataFrame  # unique_id, ds, yhat
    intervals: pd.DataFrame | None  # unique_id, ds, yhat-lo-95, yhat-hi-95
    scores: pd.DataFrame  # per-model backtest aggregate (audit trail, §7.5)
    cv_frame: pd.DataFrame | None


@dataclass
class AgentResult:
    outputs: list[ForecastOutput] = field(default_factory=list)
    errors: list[dict] = field(default_factory=list)


class ForecastAgent:
    def __init__(
        self,
        config: CadenceConfig | None = None,
        factory: DefaultModelFactory | None = None,
    ) -> None:
        self.config = config or CadenceConfig()
        self.factory = factory or DefaultModelFactory(self.config)

    # ---------------------------------------------------------------- scoring
    def _consistency(self, result: BacktestResult, model: str) -> float:
        """Share of windows (per series) where `model` has the lowest window MASE."""
        per_window = pd.DataFrame([s.__dict__ for s in result.per_window])
        win_counts = total = 0
        for (_, _window), grp in per_window.groupby(["unique_id", "window"]):
            total += 1
            if grp.loc[grp["mase"].idxmin(), "model"] == model:
                win_counts += 1
        return win_counts / total if total else 0.0

    @staticmethod
    def _wql_from_cv(cv: pd.DataFrame, model: str) -> float | None:
        """§9 WQL for models whose CV frame carries interval columns (classical does)."""
        from cadence.eval.metrics import wql_from_interval

        lo_col, hi_col = f"{model}-lo-95", f"{model}-hi-95"
        if lo_col not in cv.columns or hi_col not in cv.columns:
            return None
        vals: list[float] = []
        for _, grp in cv.groupby("unique_id", sort=False):
            y = grp["y"].to_numpy(float)
            if np.abs(y).sum() == 0:
                continue
            vals.append(wql_from_interval(y, grp[lo_col].to_numpy(), grp[hi_col].to_numpy()))
        return float(np.mean(vals)) if vals else None

    # ---------------------------------------------------------------- decision
    def _decide(self, fitted: list[FittedModel]) -> tuple[str, list[str], dict[str, float]]:
        """§7.4 steps 2–3: clear+consistent winner, else inverse-error ensemble."""
        f = self.config.forecast
        ranked = sorted(fitted, key=lambda m: m.aggregate["mase_mean"].iloc[0])

        if len(ranked) == 1:
            return "best", [ranked[0].name], {}

        best, second = ranked[0], ranked[1]
        best_score = float(best.aggregate["mase_mean"].iloc[0])
        second_score = float(second.aggregate["mase_mean"].iloc[0])
        clear = second_score <= 0 or (second_score - best_score) / second_score >= f.winner_margin
        if clear and best.consistency >= f.consistency_ratio:
            return "best", [best.name], {}

        members = ranked[: f.ensemble_top_k]
        inv = np.array([1.0 / max(1e-12, float(m.aggregate["mase_mean"].iloc[0])) for m in members])
        weights = inv / inv.sum()
        for m, w in zip(members, weights, strict=True):
            m.weight = float(w)
        return "ensemble", [m.name for m in members], {m.name: m.weight for m in members}

    # ----------------------------------------------------------------- predict
    def _point_frame(
        self, forecaster: object, h: int, model_col: str | None = None
    ) -> pd.DataFrame:
        raw = forecaster.forecast(h=h)
        col = model_col or next(c for c in raw.columns if c not in {"unique_id", "ds"})
        return raw[["unique_id", "ds", col]].rename(columns={col: "yhat"})

    def _interval_frame(
        self, forecaster: object, h: int, model_col: str | None = None
    ) -> pd.DataFrame | None:
        raw = forecaster.forecast(h=h)
        col = model_col or next(c for c in raw.columns if c not in {"unique_id", "ds"})
        if f"{col}{_INTERVAL_SUFFIXES[0]}" not in raw.columns:
            return None
        return raw[
            ["unique_id", "ds", f"{col}{_INTERVAL_SUFFIXES[0]}", f"{col}{_INTERVAL_SUFFIXES[1]}"]
        ].rename(
            columns={
                f"{col}{_INTERVAL_SUFFIXES[0]}": "yhat-lo-95",
                f"{col}{_INTERVAL_SUFFIXES[1]}": "yhat-hi-95",
            }
        )

    def _combine(
        self, fitted: list[FittedModel], h: int
    ) -> tuple[pd.DataFrame, pd.DataFrame | None]:
        """Weighted-average member forecasts on the aligned (unique_id, ds) grid."""
        point_frames, interval_frames = [], []
        for m in fitted:
            point_frames.append(self._point_frame(m.forecaster, h, m.model_col).assign(_w=m.weight))
            interval = self._interval_frame(m.forecaster, h, m.model_col)
            if interval is not None:
                interval_frames.append(interval.assign(_w=m.weight))

        merged = point_frames[0]
        for pf in point_frames[1:]:
            merged = merged.merge(pf, on=["unique_id", "ds"], suffixes=("_l", "_r"))
        weight_cols = [c for c in merged.columns if c == "_w" or c.startswith("_w_")]
        yhat_cols = [c for c in merged.columns if c.startswith("yhat")]
        weight_arr = np.column_stack([merged[c].to_numpy() for c in weight_cols])
        yhat_arr = np.column_stack([merged[c].to_numpy(float) for c in yhat_cols])
        point = merged[["unique_id", "ds"]].copy()
        point["yhat"] = (yhat_arr * weight_arr).sum(axis=1)

        intervals = None
        if len(interval_frames) == len(fitted):
            merged_i = interval_frames[0]
            for inf in interval_frames[1:]:
                merged_i = merged_i.merge(inf, on=["unique_id", "ds"], suffixes=("_l", "_r"))
            weight_cols_i = [c for c in merged_i.columns if c == "_w" or c.startswith("_w_")]
            lo_cols = [c for c in merged_i.columns if c.startswith("yhat-lo")]
            hi_cols = [c for c in merged_i.columns if c.startswith("yhat-hi")]
            w_arr = np.column_stack([merged_i[c].to_numpy() for c in weight_cols_i])
            intervals = merged_i[["unique_id", "ds"]].copy()
            # weighted average of member quantile curves (approximation of a pooled
            # predictive; exact pooling needs the full quantile grid — Phase 8 note)
            intervals["yhat-lo-95"] = (
                np.column_stack([merged_i[c].to_numpy(float) for c in lo_cols]) * w_arr
            ).sum(axis=1)
            intervals["yhat-hi-95"] = (
                np.column_stack([merged_i[c].to_numpy(float) for c in hi_cols]) * w_arr
            ).sum(axis=1)
        return point, intervals

    # -------------------------------------------------------------- per series
    def run(
        self,
        df: pd.DataFrame,
        shortlists: list[ModelShortlist],
        transforms: dict[str, Transform] | None = None,
    ) -> AgentResult:
        """§7.4 pipeline per series. Returns outputs + accumulated errors.

        `transforms` maps unique_id → the §7.2 transform applied upstream (pass
        diag_result decisions). Backtest scores rank models in the cleaned data's
        space (same for all candidates, so ranking is fair); FINAL FORECASTS ARE
        BACK-TRANSFORMED to the original series space (exp for log; cumulative
        integration anchored at y_raw for difference) — a forecast in log space is
        not a forecast the user asked for.
        """
        result = AgentResult()
        h = self.config.forecast.horizon
        # classical models backtest together in one statsforecast call; non-classical
        # candidates are CV'd individually. Fit happens on the FULL df (predict uses
        # full history; CV frames come from rolling-origin refits).
        for shortlist in shortlists:
            uid = shortlist.unique_id
            series_df = df[df["unique_id"] == uid]
            if series_df.empty:
                result.errors.append(
                    {
                        "stage": "forecast",
                        "unique_id": uid,
                        "error": "no rows in input for this unique_id",
                    }
                )
                continue

            names = [c.name for c in shortlist.candidates if c.is_available]
            if not names:
                result.errors.append(
                    {
                        "stage": "forecast",
                        "unique_id": uid,
                        "error": "no implemented candidates in shortlist",
                    }
                )
                continue

            classical = [n for n in names if n in MODEL_REGISTRY]
            other = [n for n in names if n not in MODEL_REGISTRY]
            fitted: list[FittedModel] = []
            skipped: list[str] = []
            cv_frames: dict[str, pd.DataFrame] = {}

            try:
                if classical:
                    bt = rolling_backtest(
                        series_df,
                        horizon=h,
                        n_windows=self.config.forecast.n_windows,
                        models=tuple(classical),
                    )
                    for name in classical:
                        agg = bt.aggregate[bt.aggregate["model"] == name]
                        fitted.append(
                            FittedModel(
                                name=name,
                                forecaster=None,
                                aggregate=agg,
                                consistency=self._consistency(bt, name),
                                wql=self._wql_from_cv(bt.cv_frame, name),
                                model_col=name,
                            )
                        )
                    cv_frames.update({name: bt.cv_frame for name in classical})
                    # forecaster for classical members: fit once on full history
                    classical_fc = ClassicalForecaster(
                        ClassicalModelConfig(models=tuple(classical))
                    ).fit(series_df)
                    for m in fitted:
                        if m.name in MODEL_REGISTRY:
                            m.forecaster = classical_fc
            except Exception as exc:  # noqa: BLE001 — §7.6: classical block failed
                result.errors.append(
                    {
                        "stage": "forecast",
                        "unique_id": uid,
                        "error": f"classical tier: {type(exc).__name__}: {exc}",
                    }
                )
                fitted = [m for m in fitted if m.name not in classical]

            for name in other:
                try:
                    fc = self.factory.build(name)
                    fc.fit(series_df)
                    from cadence.eval.backtest import _score_frame
                    from cadence.models.classical import infer_season_length

                    cv = fc.cross_validation(h=h, n_windows=self.config.forecast.n_windows)
                    scores = _score_frame(cv, [name], infer_season_length(series_df), series_df)
                    rows = [s.__dict__ for s in scores]
                    agg = (
                        pd.DataFrame(rows)
                        .groupby("model", as_index=False)
                        .agg(
                            mase_mean=("mase", "mean"),
                            mase_max=("mase", "max"),
                            smape_mean=("smape", "mean"),
                            smape_max=("smape", "max"),
                            windows=("window", "nunique"),
                        )
                    )
                    fitted.append(
                        FittedModel(
                            name=name,
                            forecaster=fc,
                            aggregate=agg,
                            consistency=self._consistency_from_rows(rows, name),
                            wql=self._wql_from_cv(cv, name),
                        )
                    )
                    cv_frames[name] = cv
                except MissingDependencyError:
                    skipped.append(name)
                except Exception as exc:  # noqa: BLE001 — §7.6
                    result.errors.append(
                        {
                            "stage": "forecast",
                            "unique_id": uid,
                            "error": f"{name}: {type(exc).__name__}: {exc}",
                        }
                    )

            if not fitted:
                result.errors.append(
                    {
                        "stage": "forecast",
                        "unique_id": uid,
                        "error": "no candidates produced scores"
                        + (f"; skipped (deps): {skipped}" if skipped else ""),
                    }
                )
                continue

            decision, selected, weights = self._decide(fitted)
            members = [m for m in fitted if m.name in selected]
            try:
                if decision == "best":
                    point = self._point_frame(members[0].forecaster, h, members[0].model_col)
                    intervals = self._interval_frame(members[0].forecaster, h, members[0].model_col)
                else:
                    point, intervals = self._combine(members, h)
                if transforms and transforms.get(uid, Transform.NONE) != Transform.NONE:
                    point, intervals = self._backtransform(
                        point, intervals, transforms[uid], series_df
                    )
            except Exception as exc:  # noqa: BLE001 — §7.6
                result.errors.append(
                    {
                        "stage": "forecast",
                        "unique_id": uid,
                        "error": f"final predict: {type(exc).__name__}: {exc}",
                    }
                )
                continue

            scores = pd.concat([m.aggregate.assign(wql=m.wql) for m in fitted], ignore_index=True)
            result.outputs.append(
                ForecastOutput(
                    unique_id=uid,
                    decision=decision,
                    selected=selected,
                    weights=weights,
                    point=point,
                    intervals=intervals,
                    scores=scores,
                    cv_frame=pd.concat(cv_frames.values(), ignore_index=True)
                    if cv_frames
                    else None,
                )
            )
        return result

    @staticmethod
    def _backtransform(
        point: pd.DataFrame,
        intervals: pd.DataFrame | None,
        transform: Transform,
        series_df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame | None]:
        """Invert the §7.2 transform so forecasts land in the original series space.

        log → exp; difference → cumulative integration anchored at the last raw
        value (y_raw, preserved by apply_preprocessing). Interval bounds are
        transformed the same way (monotone maps preserve ordering/coverage
        approximately; exact quantile pooling is a Phase 8 concern).
        """
        point = point.copy()
        value_cols = ["yhat"]
        if intervals is not None:
            intervals = intervals.copy()
            value_cols += ["yhat-lo-95", "yhat-hi-95"]

        if transform == Transform.LOG:
            for col in value_cols:
                frame = intervals if col != "yhat" else point
                frame[col] = np.exp(frame[col].to_numpy(float))
            return point, intervals

        if transform == Transform.DIFFERENCE:
            if "y_raw" not in series_df.columns:
                raise ValueError("difference inversion needs y_raw (from preprocessing)")
            anchor = float(series_df.sort_values("ds")["y_raw"].iloc[-1])
            for col in value_cols:
                frame = intervals if col != "yhat" else point
                diffs = frame[col].to_numpy(float)
                frame[col] = anchor + np.cumsum(diffs)
            return point, intervals

        return point, intervals

    def _consistency_from_rows(self, rows: list[dict], model: str) -> float:
        """Consistency from already-computed per-window rows (single-model CV path)."""
        per_window = pd.DataFrame(rows)
        win_counts = total = 0
        for _, grp in per_window.groupby(["unique_id", "window"]):
            total += 1
            if grp.loc[grp["mase"].idxmin(), "model"] == model:
                win_counts += 1
        return win_counts / total if total else 0.0
