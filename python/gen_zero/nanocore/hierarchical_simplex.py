"""Two-stage hierarchical Simplex ETF routing: scenarios first, then sub-intents.

A flat Simplex ETF over every intent (K = 60 for MASSIVE) packs all labels into one
frame, so inner products shrink to -1/(K-1) and near-synonym intents blur together.
This module splits the decision in two:

    Stage 1: one ETF over the scenarios (K1 = 18 for MASSIVE).
    Stage 2: one small ETF per scenario over that scenario's sub-intents (K2 <= 9).

Geometry
--------
Stage-1 anchors occupy coordinates ``[0, K1 - 1)`` of the hidden state. Every
stage-2 frame occupies the *next* ``K2 - 1`` coordinates, ``[K1 - 1, K1 - 1 + K2 - 1)``.
The two blocks are orthogonal, so the sub-intent signal can never leak into the
scenario logits, and each frame keeps the exact equiangular property
``<v_i, v_j> = -1 / (K - 1)`` with ``||v_i|| = 1``.

Routing
-------
Stage 1 scores scenarios by cosine similarity. When the top scenario wins by a clear
probability margin, only it is expanded. Otherwise the top ``beam`` scenarios are
expanded and the best joint probability ``p(scenario) * p(intent | scenario)`` wins.
Scenarios with a single intent skip stage 2 (probability 1.0).

The module is pure NumPy and deterministic: no randomness, no I/O.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .action_etf_embedding import generate_simplex_etf

__all__ = [
    "MASSIVE_TAXONOMY",
    "HierarchicalRoute",
    "HierarchicalETFReport",
    "HierarchicalSimplexRouter",
]

#: The 18 MASSIVE scenarios (60 intents), the default two-level label tree.
MASSIVE_TAXONOMY: Mapping[str, Tuple[str, ...]] = {
    "alarm": ("alarm_query", "alarm_remove", "alarm_set"),
    "audio": (
        "audio_volume_down",
        "audio_volume_mute",
        "audio_volume_other",
        "audio_volume_up",
    ),
    "calendar": ("calendar_query", "calendar_remove", "calendar_set"),
    "cooking": ("cooking_query", "cooking_recipe"),
    "datetime": ("datetime_convert", "datetime_query"),
    "email": ("email_addcontact", "email_query", "email_querycontact", "email_sendemail"),
    "general": ("general_greet", "general_joke", "general_quirky"),
    "iot": (
        "iot_cleaning",
        "iot_coffee",
        "iot_hue_lightchange",
        "iot_hue_lightdim",
        "iot_hue_lightoff",
        "iot_hue_lighton",
        "iot_hue_lightup",
        "iot_wemo_off",
        "iot_wemo_on",
    ),
    "lists": ("lists_createoradd", "lists_query", "lists_remove"),
    "music": ("music_dislikeness", "music_likeness", "music_query", "music_settings"),
    "news": ("news_query",),
    "play": ("play_audiobook", "play_game", "play_music", "play_podcasts", "play_radio"),
    "qa": ("qa_currency", "qa_definition", "qa_factoid", "qa_maths", "qa_stock"),
    "recommendation": (
        "recommendation_events",
        "recommendation_locations",
        "recommendation_movies",
    ),
    "social": ("social_post", "social_query"),
    "takeaway": ("takeaway_order", "takeaway_query"),
    "transport": ("transport_query", "transport_taxi", "transport_ticket", "transport_traffic"),
    "weather": ("weather_query",),
}


@dataclass(frozen=True)
class HierarchicalRoute:
    """Result of routing one hidden state through both stages."""

    scenario: str
    intent: str
    scenario_prob: float
    intent_prob: float
    joint_prob: float
    stage1_margin: float
    expanded_scenarios: Tuple[str, ...]

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-friendly view of the route."""
        return {
            "scenario": self.scenario,
            "intent": self.intent,
            "scenario_prob": self.scenario_prob,
            "intent_prob": self.intent_prob,
            "joint_prob": self.joint_prob,
            "stage1_margin": self.stage1_margin,
            "expanded_scenarios": list(self.expanded_scenarios),
        }


@dataclass(frozen=True)
class HierarchicalETFReport:
    """Geometry audit: worst deviation from the ideal ETF, per frame and across stages."""

    stage1_max_deviation: float
    stage2_max_deviation: float
    cross_stage_max_overlap: float
    is_valid: bool

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-friendly view of the report."""
        return {
            "stage1_max_deviation": self.stage1_max_deviation,
            "stage2_max_deviation": self.stage2_max_deviation,
            "cross_stage_max_overlap": self.cross_stage_max_overlap,
            "is_valid": self.is_valid,
        }


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    return exp / np.sum(exp)


def _cosine_to_rows(vec: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Cosine of ``vec`` against each unit-norm row. A zero vector scores 0 everywhere."""
    norm = float(np.linalg.norm(vec))
    if norm < 1e-12:
        return np.zeros(rows.shape[0], dtype=np.float64)
    return rows @ (vec / norm)


class HierarchicalSimplexRouter:
    """Route hidden states to ``(scenario, intent)`` with two nested Simplex ETFs."""

    def __init__(
        self,
        taxonomy: Optional[Mapping[str, Sequence[str]]] = None,
        dim: int = 64,
        temperature: float = 0.2,
        margin_threshold: float = 0.25,
        beam: int = 3,
    ) -> None:
        """
        Args:
            taxonomy: Scenario name -> ordered sub-intent names. Defaults to MASSIVE.
            dim: Hidden-state dimension. Must be >= (K1 - 1) + (max K2 - 1).
            temperature: Softmax temperature applied to cosine scores (both stages).
            margin_threshold: Stage-1 probability gap (top1 - top2) at or above which
                only the top scenario is expanded.
            beam: Number of scenarios expanded when the stage-1 margin is small.

        Raises:
            ValueError: On an empty or duplicated taxonomy, or a ``dim`` that is too small.
        """
        tree = dict(MASSIVE_TAXONOMY if taxonomy is None else taxonomy)
        if not tree:
            raise ValueError("taxonomy must contain at least one scenario")
        seen: Dict[str, str] = {}
        for scenario, intents in tree.items():
            if not intents:
                raise ValueError(f"scenario {scenario!r} has no sub-intents")
            if len(set(intents)) != len(intents):
                raise ValueError(f"scenario {scenario!r} repeats a sub-intent")
            for intent in intents:
                if intent in seen:
                    raise ValueError(
                        f"intent {intent!r} appears in both {seen[intent]!r} and {scenario!r}"
                    )
                seen[intent] = scenario

        self.scenarios: Tuple[str, ...] = tuple(tree)
        self.intents: Dict[str, Tuple[str, ...]] = {s: tuple(v) for s, v in tree.items()}
        self.temperature = max(1e-4, float(temperature))
        self.margin_threshold = float(margin_threshold)
        self.beam = max(1, int(beam))

        k1 = len(self.scenarios)
        self._stage1_width = max(k1 - 1, 0)
        self._stage2_width = max(max(len(v) for v in self.intents.values()) - 1, 0)
        needed = self._stage1_width + self._stage2_width
        if dim < max(needed, 1):
            raise ValueError(f"dim={dim} is too small: need >= {needed} for this taxonomy")
        self.dim = int(dim)

        # Stage-1 frame lives in coordinates [0, K1 - 1).
        self._stage1: np.ndarray = (
            generate_simplex_etf(k1, k1 - 1) if k1 > 1 else np.zeros((1, 0), dtype=np.float64)
        )
        # Stage-2 frames live in the next K2 - 1 coordinates, orthogonal to stage 1.
        self._stage2: Dict[str, np.ndarray] = {
            s: (
                generate_simplex_etf(len(v), len(v) - 1)
                if len(v) > 1
                else np.zeros((1, 0), dtype=np.float64)
            )
            for s, v in self.intents.items()
        }

    # ------------------------------------------------------------------ geometry

    def stage1_anchor(self, scenario: str) -> np.ndarray:
        """Return the unit anchor of ``scenario`` embedded in the full hidden space."""
        vec = np.zeros(self.dim, dtype=np.float64)
        vec[: self._stage1_width] = self._stage1[self._scenario_index(scenario)]
        return vec

    def stage2_anchor(self, scenario: str, intent: str) -> np.ndarray:
        """Return the unit anchor of ``intent`` inside ``scenario``, in the full hidden space."""
        names = self._intents_of(scenario)
        if intent not in names:
            raise KeyError(f"intent {intent!r} does not belong to scenario {scenario!r}")
        row = self._stage2[scenario][names.index(intent)]
        vec = np.zeros(self.dim, dtype=np.float64)
        vec[self._stage1_width : self._stage1_width + row.shape[0]] = row
        return vec

    def encode(self, scenario: str, intent: str, intent_weight: float = 1.0) -> np.ndarray:
        """Build a clean hidden state for a known label pair (useful for tests and probes)."""
        return self.stage1_anchor(scenario) + float(intent_weight) * self.stage2_anchor(
            scenario, intent
        )

    def verify(self, atol: float = 1e-9) -> HierarchicalETFReport:
        """Check every frame is equiangular and the two stages are mutually orthogonal."""
        stage1_dev = self._frame_deviation(self._stage1)
        stage2_dev = max(
            (self._frame_deviation(f) for f in self._stage2.values()), default=0.0
        )
        stage1_block = np.stack([self.stage1_anchor(s) for s in self.scenarios])
        overlap = max(
            float(np.max(np.abs(stage1_block @ self.stage2_anchor(s, i))))
            for s, names in self.intents.items()
            for i in names
        )
        return HierarchicalETFReport(
            stage1_max_deviation=stage1_dev,
            stage2_max_deviation=stage2_dev,
            cross_stage_max_overlap=overlap,
            is_valid=stage1_dev <= atol and stage2_dev <= atol and overlap <= atol,
        )

    # ------------------------------------------------------------------ routing

    def scenario_probs(self, hidden: np.ndarray) -> np.ndarray:
        """Stage-1 softmax over scenarios, in ``self.scenarios`` order."""
        vec = self._check_hidden(hidden)
        if len(self.scenarios) == 1:
            return np.ones(1, dtype=np.float64)
        cos = _cosine_to_rows(vec[: self._stage1_width], self._stage1)
        return _softmax(cos / self.temperature)

    def intent_probs(self, hidden: np.ndarray, scenario: str) -> np.ndarray:
        """Stage-2 softmax over the sub-intents of ``scenario``, in taxonomy order."""
        vec = self._check_hidden(hidden)
        self._intents_of(scenario)  # raises KeyError on an unknown scenario
        frame = self._stage2[scenario]
        if frame.shape[0] == 1:
            return np.ones(1, dtype=np.float64)
        start = self._stage1_width
        cos = _cosine_to_rows(vec[start : start + frame.shape[1]], frame)
        return _softmax(cos / self.temperature)

    def route(self, hidden: np.ndarray) -> HierarchicalRoute:
        """Route one hidden state through both stages."""
        p1 = self.scenario_probs(hidden)
        # Stable order: highest probability first, ties broken by taxonomy order.
        order = sorted(range(len(p1)), key=lambda i: (-p1[i], i))
        margin = float(p1[order[0]] - p1[order[1]]) if len(order) > 1 else 1.0
        width = 1 if margin >= self.margin_threshold else self.beam
        expanded = order[:width]

        best: Optional[Tuple[float, int, str, float]] = None
        for idx in expanded:
            scenario = self.scenarios[idx]
            p2 = self.intent_probs(hidden, scenario)
            j = int(np.argmax(p2))
            joint = float(p1[idx] * p2[j])
            if best is None or joint > best[0]:
                best = (joint, idx, self.intents[scenario][j], float(p2[j]))

        assert best is not None  # expanded is never empty
        joint, idx, intent, p_intent = best
        return HierarchicalRoute(
            scenario=self.scenarios[idx],
            intent=intent,
            scenario_prob=float(p1[idx]),
            intent_prob=p_intent,
            joint_prob=joint,
            stage1_margin=margin,
            expanded_scenarios=tuple(self.scenarios[i] for i in expanded),
        )

    def route_batch(self, hidden_batch: np.ndarray) -> List[HierarchicalRoute]:
        """Route each row of a ``(n, dim)`` array."""
        batch = np.asarray(hidden_batch, dtype=np.float64)
        if batch.ndim != 2:
            raise ValueError(f"expected a 2-D batch, got shape {batch.shape}")
        return [self.route(row) for row in batch]

    # ------------------------------------------------------------------ internals

    def _check_hidden(self, hidden: np.ndarray) -> np.ndarray:
        vec = np.asarray(hidden, dtype=np.float64)
        if vec.shape != (self.dim,):
            raise ValueError(f"expected hidden state of shape ({self.dim},), got {vec.shape}")
        if not np.all(np.isfinite(vec)):
            raise ValueError("hidden state contains NaN or inf")
        return vec

    def _scenario_index(self, scenario: str) -> int:
        try:
            return self.scenarios.index(scenario)
        except ValueError:
            raise KeyError(f"unknown scenario {scenario!r}") from None

    def _intents_of(self, scenario: str) -> Tuple[str, ...]:
        if scenario not in self.intents:
            raise KeyError(f"unknown scenario {scenario!r}")
        return self.intents[scenario]

    @staticmethod
    def _frame_deviation(frame: np.ndarray) -> float:
        k = frame.shape[0]
        if k < 2:
            return 0.0
        gram = frame @ frame.T
        norm_dev = float(np.max(np.abs(np.diag(gram) - 1.0)))
        off = gram[~np.eye(k, dtype=bool)]
        return max(norm_dev, float(np.max(np.abs(off + 1.0 / (k - 1)))))

