"""ForecastAgent (§7.4) tests: metrics, decision rule, weights, end-to-end paths."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cadence.agents.forecast_agent import DefaultModelFactory, FittedModel, ForecastAgent
from cadence.agents.planner_agent import CandidateModel, ModelShortlist, Tier
from cadence.config.default_config import CadenceConfig, ForecastConfig
from cadence.eval.metrics import pinball_loss, wql, wql_from_interval
from cadence.models.base import MissingDependencyError
from tests.conftest import make_canonical_df


# --------------------------------------------------------------- §9 metrics
class TestWQL:
    def test_pinball_zero_when_perfect(self):
        y = np.array([1.0, 2.0, 3.0])
        assert pinball_loss(y, y, q=0.5) == 0.0

    def test_pinball_asymmetry_across_quantiles(self):
        y = np.array([1.0])
        # over-forecasting by 1 hurts the median most, high quantiles least
        assert pinball_loss(y, np.array([2.0]), q=0.5) == pytest.approx(0.5)
        assert pinball_loss(y, np.array([2.0]), q=0.1) == pytest.approx(0.9)
        assert pinball_loss(y, np.array([2.0]), q=0.9) == pytest.approx(0.1)

    def test_invalid_q_raises(self):
        with pytest.raises(ValueError, match="q must be"):
            pinball_loss(np.ones(3), np.ones(3), q=1.5)

    def test_wql_scale_free(self):
        rng = np.random.default_rng(3)
        y = rng.random(20) + 1
        small = wql_from_interval(y, y * 0.9, y * 1.1)
        big = wql_from_interval(y * 1000, y * 900, y * 1100)
        assert small == pytest.approx(big, rel=1e-9)

    def test_wql_all_zero_raises(self):
        with pytest.raises(ValueError, match="scale is zero"):
            wql(np.zeros(5), np.ones(5), q=0.5)


# ------------------------------------------------------ §7.4 decision helpers
def make_fitted(name: str, mase: float, consistency: float = 0.5) -> FittedModel:
    agg = pd.DataFrame(
        [
            {
                "unique_id": "s",
                "model": name,
                "mase_mean": mase,
                "mase_max": mase + 0.1,
                "smape_mean": 0.1,
                "smape_max": 0.2,
                "windows": 3,
            }
        ]
    )
    return FittedModel(name=name, forecaster=None, aggregate=agg, consistency=consistency)


class TestDecisionRule:
    def test_single_candidate_wins_outright(self):
        decision, selected, weights = ForecastAgent(CadenceConfig())._decide(
            [make_fitted("A", 0.9)]
        )
        assert decision == "best" and selected == ["A"] and weights == {}

    def test_clear_and_consistent_best(self):
        # 20% margin >= 5%, consistency 0.9 >= 0.7 → outright winner
        fitted = [make_fitted("A", 0.80, consistency=0.9), make_fitted("B", 1.00)]
        decision, selected, _ = ForecastAgent(CadenceConfig())._decide(fitted)
        assert decision == "best" and selected == ["A"]

    def test_clear_but_inconsistent_ensembles(self):
        fitted = [make_fitted("A", 0.80, consistency=0.3), make_fitted("B", 1.00)]
        decision, selected, _ = ForecastAgent(CadenceConfig())._decide(fitted)
        assert decision == "ensemble"
        assert {"A", "B"} <= set(selected)

    def test_close_race_ensembles_despite_consistency(self):
        fitted = [make_fitted("A", 0.80, consistency=0.9), make_fitted("B", 0.81)]
        decision, _, _ = ForecastAgent(CadenceConfig())._decide(fitted)
        assert decision == "ensemble"  # 1.25% < 5% margin

    def test_inverse_error_weights_sum_to_one(self):
        fitted = [make_fitted("A", 0.5), make_fitted("B", 1.0), make_fitted("C", 2.0)]
        decision, selected, weights = ForecastAgent(CadenceConfig())._decide(fitted)
        assert decision == "ensemble" and selected == ["A", "B", "C"]
        assert weights["A"] == pytest.approx(2.0 / 3.5)
        assert weights["C"] == pytest.approx(0.5 / 3.5)
        assert sum(weights.values()) == pytest.approx(1.0)

    def test_top_k_limits_members(self):
        cfg = CadenceConfig(forecast=ForecastConfig(ensemble_top_k=2))
        fitted = [make_fitted("A", 0.5), make_fitted("B", 1.0), make_fitted("C", 2.0)]
        decision, selected, weights = ForecastAgent(cfg)._decide(fitted)
        assert decision == "ensemble" and selected == ["A", "B"]
        assert "C" not in weights


# ------------------------------------------------------- stub-tier end-to-end
class DriftForecaster:
    """Deterministic stub tier: last value + slope*h, wide frame + interval columns."""

    def __init__(self, slope: float, name: str) -> None:
        self.slope = slope
        self.name = name
        self._train: pd.DataFrame | None = None

    def fit(self, df: pd.DataFrame):
        self._train = df.copy()
        return self

    def _last_rows(self, grp: pd.DataFrame) -> tuple[pd.Timestamp, float]:
        grp = grp.sort_values("ds")
        return grp["ds"].iloc[-1], float(grp["y"].iloc[-1])

    def forecast(self, h: int, level=(95,)):
        rows = []
        for uid, grp in self._train.groupby("unique_id"):
            last_ds, last_y = self._last_rows(grp)
            ds = pd.date_range(last_ds, periods=h + 1, freq="D")[1:]
            yhat = last_y + self.slope * np.arange(1, h + 1)
            rows.append(
                pd.DataFrame(
                    {
                        "unique_id": uid,
                        "ds": ds,
                        self.name: yhat,
                        f"{self.name}-lo-95": yhat - 2.0,
                        f"{self.name}-hi-95": yhat + 2.0,
                    }
                )
            )
        return pd.concat(rows, ignore_index=True)

    def cross_validation(self, h: int, n_windows: int = 3, step_size: int | None = None):
        step = step_size or h
        rows = []
        for uid, grp in self._train.groupby("unique_id"):
            grp = grp.sort_values("ds").reset_index(drop=True)
            n = len(grp)
            for w in range(n_windows):
                start = n - h - w * step
                train, test = grp.iloc[:start], grp.iloc[start : start + h]
                yhat = float(train["y"].iloc[-1]) + self.slope * np.arange(1, len(test) + 1)
                rows.append(
                    pd.DataFrame(
                        {
                            "unique_id": uid,
                            "ds": test["ds"].to_numpy(),
                            "cutoff": train["ds"].iloc[-1],
                            "y": test["y"].to_numpy(float),
                            self.name: yhat,
                        }
                    )
                )
        return pd.concat(rows, ignore_index=True)


class StubFactory(DefaultModelFactory):
    def __init__(self, config: CadenceConfig, slope: float, names: tuple[str, ...]) -> None:
        super().__init__(config)
        self.slope = slope
        self.names = names

    def build(self, name: str):
        if name in self.names:
            return DriftForecaster(self.slope, name)
        return super().build(name)


def stub_shortlist(uid: str, names: tuple[str, ...] = ("StubModel",)) -> ModelShortlist:
    return ModelShortlist(
        unique_id=uid,
        candidates=[
            CandidateModel(name=n, tier=Tier.ML, reason="stub", implemented_in_phase=4)
            for n in names
        ],
    )


class TestEndToEndStub:
    def _agent(self, names=("StubModel",)) -> ForecastAgent:
        cfg = CadenceConfig(forecast=ForecastConfig(horizon=6))
        return ForecastAgent(cfg, factory=StubFactory(cfg, slope=0.5, names=names))

    def test_best_decision_produces_point_and_intervals(self):
        df = make_canonical_df(uids=["s"], n=60, freq="D")
        result = self._agent().run(df, [stub_shortlist("s")])

        assert result.errors == []
        out = result.outputs[0]
        assert out.decision == "best" and out.selected == ["StubModel"]
        assert len(out.point) == 6
        assert out.intervals is not None
        assert (out.intervals["yhat-lo-95"] <= out.intervals["yhat-hi-95"]).all()
        assert set(out.scores["model"]) == {"StubModel"}
        assert (out.scores["mase_mean"] > 0).all()
        assert out.scores["smape_mean"].between(0, 2).all()

    def test_identical_candidates_ensemble_equally(self):
        df = make_canonical_df(uids=["s"], n=60, freq="D")
        names = ("StubA", "StubB")
        result = self._agent(names).run(df, [stub_shortlist("s", names)])

        out = result.outputs[0]
        assert out.decision == "ensemble"  # identical MASE → margin 0
        assert len(out.selected) == 2
        assert sum(out.weights.values()) == pytest.approx(1.0)
        assert len(out.point) == 6
        assert out.intervals is not None  # both members expose intervals

    def test_unknown_uid_accumulates_error(self):
        df = make_canonical_df(uids=["s"], n=60, freq="D")
        result = self._agent().run(df, [stub_shortlist("nope")])
        assert result.outputs == []
        assert result.errors[0]["unique_id"] == "nope"

    def test_missing_dependency_skipped_not_fatal(self, monkeypatch):
        df = make_canonical_df(uids=["s"], n=60, freq="D")
        cfg = CadenceConfig(forecast=ForecastConfig(horizon=6))
        agent = ForecastAgent(cfg, factory=StubFactory(cfg, 0.5, ("StubModel",)))
        shortlist = stub_shortlist("s", ("StubModel", "Gone"))
        original = StubFactory.build

        def build(self, name):
            if name == "Gone":
                raise MissingDependencyError("not installed")
            return original(self, name)

        monkeypatch.setattr(StubFactory, "build", build)
        result = agent.run(df, [shortlist])

        assert result.errors == []  # skipped ≠ error
        assert result.outputs[0].selected == ["StubModel"]

    def test_failing_candidate_accumulates_error_not_fatal(self, monkeypatch):
        df = make_canonical_df(uids=["s"], n=60, freq="D")
        cfg = CadenceConfig(forecast=ForecastConfig(horizon=6))
        agent = ForecastAgent(cfg, factory=StubFactory(cfg, 0.5, ("StubModel",)))
        shortlist = stub_shortlist("s", ("StubModel", "Broken"))
        original = StubFactory.build

        def build(self, name):
            if name == "Broken":
                raise ValueError("boom")
            return original(self, name)

        monkeypatch.setattr(StubFactory, "build", build)
        result = agent.run(df, [shortlist])

        assert len(result.outputs) == 1  # healthy candidate still forecasts
        assert any("Broken" in e["error"] for e in result.errors)  # §7.6 audit


# --------------------------------------------------- real classical end-to-end
class TestRealClassical:
    def test_air_passengers_end_to_end(self):
        from cadence.connectors.csv_connector import CSVConnector

        df = CSVConnector().load("data/sample/air_passengers.csv").df
        cfg = CadenceConfig(forecast=ForecastConfig(horizon=12))
        shortlist = ModelShortlist(
            unique_id="air_passengers",
            candidates=[
                CandidateModel(
                    name="AutoARIMA", tier=Tier.CLASSICAL, reason="r", implemented_in_phase=1
                ),
                CandidateModel(
                    name="AutoETS", tier=Tier.CLASSICAL, reason="r", implemented_in_phase=1
                ),
            ],
        )
        result = ForecastAgent(cfg).run(df, [shortlist])

        assert result.errors == []
        out = result.outputs[0]
        assert out.decision in {"best", "ensemble"}
        assert len(out.point) == 12
        assert out.intervals is not None
        assert (out.intervals["yhat-lo-95"] <= out.intervals["yhat-hi-95"]).all()
        assert set(out.scores["model"]) == {"AutoARIMA", "AutoETS"}
        # §9 probabilistic audit: WQL computed from classical CV interval columns
        assert out.scores["wql"].notna().all()
        assert (out.scores["wql"] > 0).all()
        assert out.cv_frame is not None
