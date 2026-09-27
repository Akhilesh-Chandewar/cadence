"""DiagnosticAgent — the "Curator" (spec §7.2).

Per series: compute the §7.2 stats with the pure functions in diagnostic_stats,
then decide preprocessing. The decision comes from the §7.7 LLM client when
enabled (structured output, retried), otherwise from the deterministic fallback
recommender — the code path is identical either way, only the decision source
changes. One bad series logs to `errors` and never halts the run (§7.6).
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field

import pandas as pd

from cadence.agents.decisions import (
    PreprocessingDecision,
    SeriesDiagnostics,
    Transform,
    apply_preprocessing,
    recommend_preprocessing,
)
from cadence.agents.diagnostic_stats import (
    compute_intermittency,
    compute_missingness,
    compute_outliers,
    compute_seasonality,
    compute_stationarity,
    compute_trend,
)
from cadence.config.default_config import CadenceConfig
from cadence.llm.client import LLMClient
from cadence.llm.litellm_client import make_llm_client
from cadence.models.classical import infer_frequency, infer_season_length


def _json_default(o: object) -> object:
    """JSON serializer for numpy scalars that sneak into the stats payload."""
    item = getattr(o, "item", None)
    return item() if callable(item) else str(o)


_SYSTEM_PROMPT = (
    "You are a time-series preprocessing expert. Given summary statistics for one "
    "series, choose a missing-value strategy, an outlier strategy, and an optional "
    "transform. Prefer conservative choices: keep_and_flag outliers unless they are "
    "clearly corrupting, log-transform only right-skewed strictly-positive series, "
    "difference only strongly non-stationary ones. Respond only with a valid "
    "PreprocessingDecision."
)


@dataclass
class DiagnosticsResult:
    diagnostics: list[SeriesDiagnostics]
    cleaned_df: pd.DataFrame
    errors: list[dict] = field(default_factory=list)

    def by_id(self, unique_id: str) -> SeriesDiagnostics:
        for d in self.diagnostics:
            if d.unique_id == unique_id:
                return d
        raise KeyError(f"no diagnostics for unique_id={unique_id!r}")


def _coerce_decision(decision: PreprocessingDecision, y: pd.Series) -> PreprocessingDecision:
    """Guard the applier's invariants before anything touches the data."""
    if decision.transform == Transform.LOG and bool((y <= 0).any()):
        return decision.model_copy(
            update={
                "transform": Transform.NONE,
                "reason": decision.reason + " [log dropped: non-positive values present]",
            }
        )
    return decision


class DiagnosticAgent:
    def __init__(
        self,
        config: CadenceConfig | None = None,
        llm: LLMClient | None = None,
    ) -> None:
        self.config = config or CadenceConfig()
        # §7.7 factory: disabled config → DisabledLLMClient; enabled config →
        # LiteLLM-backed client (fails fast at construction if no API key).
        self.llm = llm if llm is not None else make_llm_client(self.config.llm)

    async def diagnose(self, df: pd.DataFrame) -> DiagnosticsResult:
        cleaned_parts: list[pd.DataFrame] = []
        diagnostics: list[SeriesDiagnostics] = []
        errors: list[dict] = []

        for uid, grp in df.groupby("unique_id", sort=True):
            try:
                diag, cleaned = await self._diagnose_series(uid, grp)
                diagnostics.append(diag)
                cleaned_parts.append(cleaned)
            except Exception as exc:  # noqa: BLE001 — §7.6: accumulate, don't crash
                errors.append(
                    {
                        "stage": "diagnostic",
                        "unique_id": uid,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )

        cleaned_df = pd.concat(cleaned_parts, ignore_index=True) if cleaned_parts else df.iloc[:0]
        return DiagnosticsResult(diagnostics=diagnostics, cleaned_df=cleaned_df, errors=errors)

    async def _diagnose_series(
        self, uid: str, grp: pd.DataFrame
    ) -> tuple[SeriesDiagnostics, pd.DataFrame]:
        grp = grp.sort_values("ds").reset_index(drop=True)
        y = grp["y"].astype(float)

        # frequency + seasonal period (infer; tolerate irregular series)
        try:
            freq = infer_frequency(grp)
        except Exception:  # noqa: BLE001 — irregular timestamps are diagnostic info, not a crash
            freq = "unknown"
        period = infer_season_length(grp, freq) if freq != "unknown" else 1

        trend = compute_trend(y.to_numpy())
        seasonality = compute_seasonality(y.to_numpy(), period)
        stationarity = compute_stationarity(y.to_numpy())
        outlier_count, _ = compute_outliers(y.to_numpy(), period)
        intermittency = compute_intermittency(y.to_numpy())
        missingness = (
            compute_missingness(grp["ds"], freq) if freq != "unknown" else {"pct_missing": 0.0}
        )

        decision = await self._decide(
            y, freq, trend, seasonality, stationarity, outlier_count, intermittency, missingness
        )
        cleaned = apply_preprocessing(grp, decision)

        diag = SeriesDiagnostics(
            unique_id=str(uid),
            length=len(grp),
            freq=freq,
            pct_missing=missingness["pct_missing"],
            trend=trend,
            seasonality=seasonality,
            stationarity=stationarity,
            outlier_count=outlier_count,
            intermittent=intermittency["intermittent"],
            zero_fraction=intermittency["zero_fraction"],
            recommended_preprocessing=decision,
        )
        return diag, cleaned

    async def _decide(
        self,
        y: pd.Series,
        freq: str,
        trend: dict,
        seasonality: dict,
        stationarity: dict,
        outlier_count: int,
        intermittency: dict,
        missingness: dict,
    ) -> PreprocessingDecision:
        """LLM arbitration when enabled (§7.7), deterministic fallback otherwise."""
        stats_payload = {
            "length": len(y),
            "freq": freq,
            "pct_missing": missingness.get("pct_missing", 0.0),
            "trend": trend,
            "seasonality": seasonality,
            "stationarity": stationarity,
            "outlier_count": outlier_count,
            "intermittent": intermittency["intermittent"],
            "skew": float(y.skew()),
            "has_nonpositive_values": bool((y <= 0).any()),
        }

        payload = {**stats_payload, "y": y.to_numpy()}
        if not (self.config.llm.enabled and self.llm is not None):
            return _coerce_decision(recommend_preprocessing(payload), y)

        decision = await self._llm_decide(stats_payload)
        if decision is None:  # retries exhausted → deterministic fallback per §7.7
            fallback = recommend_preprocessing(
                payload, reason="deterministic fallback (LLM unavailable after retries)"
            )
            return _coerce_decision(fallback, y)
        return _coerce_decision(decision, y)

    async def _llm_decide(self, stats_payload: dict) -> PreprocessingDecision | None:
        for attempt in range(self.config.llm.max_retries):
            try:
                messages = [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": json.dumps(stats_payload, default=_json_default)},
                ]
                result = await self.llm.complete(messages, response_schema=PreprocessingDecision)
                if isinstance(result, PreprocessingDecision):
                    return result
                return PreprocessingDecision.model_validate(result)
            except Exception:  # noqa: BLE001 — retry with backoff, then fall through
                if attempt == self.config.llm.max_retries - 1:
                    break
                await asyncio.sleep(2**attempt)
        return None


# re-exported so the graph wiring (Phase 7) has one import site
__all__ = [
    "DiagnosticAgent",
    "DiagnosticsResult",
    "compute_outliers",
    "compute_stationarity",
    "compute_trend",
]
