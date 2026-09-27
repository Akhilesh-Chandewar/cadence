"""PlannerAgent (spec §7.3): diagnostics + run config → 2–5 candidate models per series.

The §7.3 rule table is deterministic code first (testable, cheap, explainable).
The §7.7 LLM is used *only* to arbitrate borderline cases (a series sitting near a
rule threshold); when the LLM is disabled or fails, the deterministic shortlist
stands — the harness never depends on the LLM for correctness.

ML/DL/foundation candidates are emitted with the phase that implements them so
the ForecastAgent can skip what isn't wired yet.
"""

from __future__ import annotations

import asyncio
import json
from enum import StrEnum

from pydantic import BaseModel, Field

from cadence.agents.diagnostic_agent import SeriesDiagnostics
from cadence.config.default_config import CadenceConfig
from cadence.llm.client import LLMClient
from cadence.models.classical import MODEL_REGISTRY

# name → (tier, phase that implements it). Classical entries come from the registry.
_CATALOG: dict[str, tuple[str, int]] = {
    **{name: ("classical", 1) for name in MODEL_REGISTRY},
    "MLForecast-LightGBM": ("ml", 4),
    "N-HiTS": ("deep_learning", 4),
    "TFT": ("deep_learning", 4),
    "Chronos-Bolt": ("foundation", 5),
    "Moirai": ("foundation", 5),
}


class Tier(StrEnum):
    CLASSICAL = "classical"
    ML = "ml"
    DEEP_LEARNING = "deep_learning"
    FOUNDATION = "foundation"


class CandidateModel(BaseModel):
    name: str
    tier: Tier
    reason: str
    implemented_in_phase: int = 1

    @property
    def is_available(self) -> bool:
        return self.implemented_in_phase <= 4  # phases 0–4 are built; update as phases land


class ModelShortlist(BaseModel):
    unique_id: str
    candidates: list[CandidateModel] = Field(min_length=1)
    rule_hits: list[str] = Field(default_factory=list)
    borderline: bool = False
    arbitration: str = "deterministic"  # or "llm"


class ShortlistDecision(BaseModel):
    """§7.7 structured-output contract for LLM arbitration."""

    candidates: list[CandidateModel] = Field(min_length=1, max_length=5)


_SYSTEM_PROMPT = (
    "You are a forecasting-expert planner. Given per-series diagnostics, a run "
    "config, and a deterministic preliminary shortlist, arbitrate the borderline "
    "case: keep, drop, or add candidates from the provided catalog (2-5 total). "
    "Do not invent models outside the catalog. Respond only with a valid "
    "ShortlistDecision."
)


def _is_borderline(diag: SeriesDiagnostics, cfg: CadenceConfig) -> bool:
    """True when any §7.3 threshold is within `ambiguous_window` of being flipped."""
    w = cfg.planner.ambiguous_window
    length = diag.length

    def near(value: float, threshold: float) -> bool:
        return abs(value - threshold) <= threshold * w

    checks = [
        near(length, cfg.planner.short_history),  # "is 180 points 'short'?" (§7.3 example)
        near(length, cfg.planner.long_history),
        near(diag.seasonality.get("strength", 0.0), 0.6),  # the presence threshold
    ]
    # intermittency threshold proximity (0.5 zero-fraction)
    checks.append(near(diag.zero_fraction, 0.5))
    return any(checks)


def rule_table_shortlist(
    diag: SeriesDiagnostics, cfg: CadenceConfig
) -> tuple[list[CandidateModel], list[str], bool]:
    """The §7.3 table as deterministic code. Returns (candidates, rule_hits, borderline).

    Rows are evaluated in spec order; every matching row contributes candidates.
    """
    candidates: list[CandidateModel] = []
    hits: list[str] = []
    p = cfg.planner

    def add(name: str, reason: str) -> None:
        tier, phase = _CATALOG[name]
        if not any(c.name == name for c in candidates):
            candidates.append(
                CandidateModel(
                    name=name, tier=Tier(tier), reason=reason, implemented_in_phase=phase
                )
            )

    # Row 1 — intermittent/sparse demand → Croston, TSB
    if diag.intermittent:
        hits.append("intermittent")
        add("Croston", "intermittent demand (mostly zeros) → Croston territory")
        add("TSB", "intermittent demand → TSB (dynamic top-off)")

    # Row 2 — single series, short history, clear seasonality → classical trio
    if p.scale == "single" and diag.length < p.short_history and diag.seasonality.get("present"):
        hits.append("short_history_with_seasonality")
        add("AutoARIMA", "short univariate history with seasonality → classical")
        add("AutoETS", "short univariate history with seasonality → classical")
        add("AutoTheta", "short seasonal history: Theta as third classical opinion")

    # Row 3 — thousands of related series with covariates → ML tier
    if p.scale == "many" and p.has_covariates:
        hits.append("many_series_with_covariates")
        add("MLForecast-LightGBM", "many related series + tabular covariates → MLForecast")

    # Row 4 — long history, enough data to justify deep learning
    if diag.length >= p.long_history:
        hits.append("long_history")
        add("N-HiTS", "long history → deep learning tier (N-HiTS)")
        add("TFT", "long history → deep learning tier (TFT, covariate-capable)")

    # Row 5 — fast baseline / cold start → foundation zero-shot
    if p.prefer_fast or diag.length < max(20, p.short_history // 10):
        hits.append("fast_baseline_or_cold_start")
        add("Chronos-Bolt", "zero-shot baseline: fastest, most mature foundation model")

    # Safety net — every series gets at least a classical baseline (§7.3 spirit:
    # let the backtest decide; never emit an empty shortlist)
    if not candidates:
        hits.append("default_classical")
        add("AutoARIMA", "default classical baseline")
        add("AutoETS", "default classical baseline")

    # Ambiguous case → shortlist across two tiers and let the backtest decide (§7.3 row 6)
    borderline = _is_borderline(diag, cfg)
    tiers = {c.tier for c in candidates}
    if borderline and len(tiers) == 1:
        hits.append("ambiguous_two_tiers")
        add("Chronos-Bolt", "borderline case: add a second tier, backtest decides")

    # Spec: shortlist is 2–5 candidates
    if len(candidates) == 1:
        add("AutoETS", "shortlist must have at least two candidates")
    candidates = candidates[:5]
    return candidates, hits, borderline


class PlannerAgent:
    def __init__(self, config: CadenceConfig | None = None, llm: LLMClient | None = None) -> None:
        self.config = config or CadenceConfig()
        self.llm = llm

    async def plan(self, diag: SeriesDiagnostics) -> ModelShortlist:
        candidates, hits, borderline = rule_table_shortlist(diag, self.config)
        arbitration = "deterministic"

        # §7.7: LLM only for borderline arbitration, only when enabled end-to-end
        if (
            borderline
            and self.config.planner.llm_arbitration
            and self.config.llm.enabled
            and self.llm is not None
        ):
            llm_candidates = await self._llm_arbitrate(diag, candidates, hits)
            if llm_candidates is not None:
                candidates = llm_candidates
                arbitration = "llm"

        return ModelShortlist(
            unique_id=diag.unique_id,
            candidates=candidates,
            rule_hits=hits,
            borderline=borderline,
            arbitration=arbitration,
        )

    async def _llm_arbitrate(
        self, diag: SeriesDiagnostics, preliminary: list[CandidateModel], hits: list[str]
    ) -> list[CandidateModel] | None:
        """Structured-output arbitration; None → deterministic shortlist stands."""
        payload = {
            "diagnostics": diag.model_dump(mode="json"),
            "run_config": self.config.planner.model_dump(mode="json"),
            "preliminary_shortlist": [c.model_dump() for c in preliminary],
            "rule_hits": hits,
            "catalog": {name: {"tier": t, "phase": ph} for name, (t, ph) in _CATALOG.items()},
        }
        for attempt in range(self.config.llm.max_retries):
            try:
                messages = [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": json.dumps(payload, default=str)},
                ]
                result = await self.llm.complete(messages, response_schema=ShortlistDecision)
                decision = (
                    result
                    if isinstance(result, ShortlistDecision)
                    else ShortlistDecision.model_validate(result)
                )
                # structured-output-only rule: the LLM may keep/drop/reorder but never
                # invent models outside the catalog; dedup and cap at 5
                seen: set[str] = set()
                valid = [
                    c.model_copy(update={"reason": c.reason or "LLM arbitration"})
                    for c in decision.candidates
                    if c.name in _CATALOG and not (c.name in seen or seen.add(c.name))
                ]
                if valid:
                    return valid[:5]
                return None
            except Exception:  # noqa: BLE001 — retry with backoff, then fall through
                if attempt == self.config.llm.max_retries - 1:
                    break
                await asyncio.sleep(2**attempt)
        return None
