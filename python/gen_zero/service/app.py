"""Gen-Zero Universal Decision Microservice.

100% Compatible with OpenRouter / TypeSafe Zero 1.13 Decision API protocol:
- POST /api/alpha/decisions
- POST /v1/decisions
- GET  /v1/models
- GET  /health

Supports:
1. 'noul' question: Boolean guardrail probability (0.0 ~ 1.0)
2. 'choice' question: Categorical decision routing across criteria keys
3. 'score' question: Ordinal Likert scale expected continuous score
"""

from typing import Dict, List, Optional, Tuple, Union, Any
import time
import uuid
import logging
import json
import math
from fastapi import FastAPI, Request, HTTPException, status
import os
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi import Security, Depends
from pydantic import BaseModel, Field, field_validator
from gen_zero.client import GenZero
from gen_zero.gateway.llamacpp_adapter import LlamaCppUnavailableError, compute_closed_form_confidence
from gen_zero.service.ports import DEFAULT_SEMANTIC_PORT
from gen_zero.service.semantic_risk import (
    ESCALATE_THRESHOLD,
    HARD_STOP_THRESHOLD,
    get_risk_classifier,
)
from gen_zero.service.semantic_scorer import (
    ROUTE_FRAME,
    candidate_text,
    get_semantic_scorer,
    select_ask_frame,
)

def _load_api_token() -> str:
    """Loads the API key from the GENZERO_API_KEY environment variable only.

    World-writable credential files are never consulted: an empty token means
    every authenticated endpoint answers 401 (fail closed).
    """
    return os.environ.get("GENZERO_API_KEY", "").strip()


security = HTTPBearer(auto_error=False)


def verify_api_token(credentials: Optional[HTTPAuthorizationCredentials] = Security(security)):
    """Validates Authorization: Bearer <token> header against the configured API token."""
    valid_token = _load_api_token()
    if not valid_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": {
                    "message": "API Key not configured on server. Please set GENZERO_API_KEY environment variable.",
                    "type": "authentication_error",
                    "code": "api_key_not_configured"
                }
            }
        )
    if not credentials or credentials.scheme.lower() != "bearer" or credentials.credentials != valid_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": {
                    "message": "Invalid API Key or missing Authorization: Bearer header.",
                    "type": "invalid_request_error",
                    "code": "invalid_api_key"
                }
            }
        )
    return credentials.credentials


app = FastAPI(
    title="Gen-Zero Decision API (Zero 1.13 Compatible)",
    description="Universal non-autoregressive decision model server compatible with OpenRouter and TypeSafe Zero 1.13",
    version="1.13.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global GenZero instance
client = GenZero()
logger = logging.getLogger(__name__)

MODEL_NOT_LOADED_MESSAGE = (
    "Model checkpoint not loaded; pseudo-random inference is disabled per fail-closed policy"
)


def _require_loaded_model(endpoint: str) -> None:
    """Fail closed: refuse to answer from a random-init dual-head model.

    Raises HTTP 503 unless GenZero strictly loaded a dual-head checkpoint
    (``GENZERO_DUAL_HEAD_CHECKPOINT``). Never substitutes hashed or seeded
    pseudo-probabilities for inference.
    """
    if getattr(client, "weights_loaded_from_checkpoint", False):
        return
    logger.error("%s refused: %s", endpoint, MODEL_NOT_LOADED_MESSAGE)
    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail={
            "error": {
                "message": MODEL_NOT_LOADED_MESSAGE,
                "type": "model_unavailable",
                "code": "checkpoint_not_loaded",
            }
        },
    )


class QuestionSpec(BaseModel):
    type: str = Field(..., description="'noul', 'choice', or 'score'")
    instructions: str = Field("", description="Question prompt or query")
    criteria: Union[Dict[str, str], List[str], Any] = Field(..., description="Criteria options or scale")


class DecisionsRequest(BaseModel):
    model: str = Field("typesafe/zero-1.13", description="Target model name")
    state: Optional[Union[str, Dict[str, Any]]] = Field(None, description="Context, environment observation, or task state")
    states: Optional[List[Union[str, Dict[str, Any]]]] = Field(None, description="Batch of contexts/task states for parallel inference")
    questions: Dict[str, QuestionSpec] = Field(..., description="Map of question IDs to QuestionSpec")
    return_trajectory: bool = Field(False, description="Also return a world-model rollout per 'choice'/'noul'/'score' answer (single-state mode only)")


def _reject_bool_horizon(v: Any) -> Any:
    """bool is a subclass of int in Python; a caller passing horizon=True must not silently become horizon=1."""
    if isinstance(v, bool):
        raise ValueError(f"horizon must be an integer, not a bool: {v!r}")
    return v


class SimulateRequest(BaseModel):
    state: Union[str, Dict[str, Any], List[float]] = Field(..., description="Latent vector, dict state or text state")
    actions: List[Any] = Field(..., description="Action sequence to replay")
    horizon: Optional[int] = Field(None, description="Steps to simulate; defaults to len(actions)")

    @field_validator("horizon", mode="before")
    @classmethod
    def _validate_horizon(cls, v):
        return _reject_bool_horizon(v)


class WhatIfRequest(BaseModel):
    state: Union[str, Dict[str, Any], List[float]] = Field(..., description="Latent vector, dict state or text state")
    candidates: List[Any] = Field(..., description="Candidate first actions (1 to 16)")
    horizon: int = Field(5, description="Steps to roll out after each candidate")

    @field_validator("horizon", mode="before")
    @classmethod
    def _validate_horizon(cls, v):
        return _reject_bool_horizon(v)


class AuditActionRequest(BaseModel):
    state: Union[str, Dict[str, Any], List[float]] = Field(..., description="Latent vector, dict state or text state")
    action: Any = Field(..., description="Candidate action to audit")
    horizon: int = Field(5, description="Rollout depth")
    continuation_actions: Optional[List[Any]] = Field(None, description="Action set for greedy rollout after action")
    warn_risk: float = Field(0.5, description="Risk threshold for WarnHazard vs Approved")

    @field_validator("horizon", mode="before")
    @classmethod
    def _validate_horizon(cls, v):
        return _reject_bool_horizon(v)


class DecideStepRequest(BaseModel):
    model: str = Field("typesafe/zero-1.13", description="Target model name")
    state: Union[str, Dict[str, Any], List[Any]] = Field(..., description="Current environment state / observation")
    affordances: List[str] = Field(..., description="Available interaction target candidates")
    action_types: Optional[List[str]] = Field(None, description="Available discrete action verbs")
    goal: Optional[str] = Field(None, description="Task goal or instruction")
    risk_criteria: Optional[str] = Field(None, description="Custom risk definition")
    state_feedback: Optional[Dict[str, Any]] = Field(None, description="Instantaneous register readings")
    max_affordances: int = Field(15, description="Max affordances window after anti-truncation ranking")
    apply_policy_gate: bool = Field(True, description="Whether to evaluate dual-track policy gate")
    risk_profile: Optional[str] = Field(None, description="Domain risk profile: 'read_only', 'standard', or 'critical'")
    return_trajectory: bool = Field(False, description="Also return a world-model rollout of the chosen step")


class SemanticAskRequest(BaseModel):
    context: str = Field(..., description="Instruction or situation text, any language")
    candidates: List[str] = Field(..., description="Distinct candidate actions (>= 2)")
    state: Optional[Union[str, Dict[str, Any]]] = Field(None, description="Optional extra state appended to the context")
    history: Optional[List[str]] = Field(None, description="Actions already taken (multi-step lookahead)")
    return_embedding: bool = Field(False, description="Return the unit-norm prompt state vector")
    frame: Optional[str] = Field(None, description="Frame sentence ending the prompt; auto-selected if omitted")


class ToolSpec(BaseModel):
    name: str
    description: Optional[str] = None


class SemanticRouteRequest(BaseModel):
    intent: str = Field(..., description="Task goal or user intent, any language")
    tools: List[Union[str, ToolSpec]] = Field(..., description="Candidate tools (names or {name, description})")
    top_k: int = Field(3, ge=1, description="Number of tools to keep")
    state: Optional[Union[str, Dict[str, Any]]] = Field(None, description="Optional extra state appended to the intent")


class SemanticRiskRequest(BaseModel):
    text: str = Field(..., description="Request, intent or scenario text to assess, any language")


class ScoreRequest(BaseModel):
    prompt: Optional[str] = Field(None, description="Context, query, or observation prompt")
    candidates: Optional[List[str]] = Field(None, description="List of discrete candidate options to score")
    model: Optional[str] = Field("typesafe/zero-1.13", description="Target model name")
    temperature: Optional[float] = Field(1.0, description="Sampling temperature scaling factor")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Optional metadata or tags")


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "engine": "gen-zero",
        "version": "1.13.0",
        "compatible_with": "typesafe/zero-1.13"
    }


@app.get("/")
def root():
    return {
        "name": "Gen-Zero Decision Server",
        "version": "1.13.0",
        "endpoints": [
            "POST /api/alpha/decisions (OpenRouter compatible, supports state and batch states)",
            "POST /v1/decisions",
            "POST /v1/decide_step (Composite 4-tuple decision)",
            "POST /v1/score (Minimal non-autoregressive scoring)",
            "POST /v1/semantic_ask (Calibrated LM likelihood decision for the Rust bridge)",
            "POST /v1/semantic_route (Calibrated LM likelihood tool ranking for the Rust bridge)",
            "POST /v1/semantic_risk (Few-shot calibrated safety risk for the Rust PolicyGate)",
            "GET  /v1/semantic_health (Semantic scorer identity probe, no auth)",
            "GET  /v1/decisions/stream (Sub-second SSE decision stream)",
            "GET  /v1/models",
            "GET  /health"
        ]
    }


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": "typesafe/zero-1.13",
                "object": "model",
                "created": 1726617600,
                "owned_by": "typesafe",
                "permission": [],
                "root": "typesafe/zero-1.13",
                "parent": None
            },
            {
                "id": "gen-zero",
                "object": "model",
                "created": 1726704000,
                "owned_by": "gen-zero",
                "permission": [],
                "root": "gen-zero",
                "parent": None
            }
        ]
    }


@app.post("/v1/score", dependencies=[Depends(verify_api_token)])
@app.post("/score", dependencies=[Depends(verify_api_token)])
def handle_score(req: ScoreRequest):
    """Minimal non-autoregressive scoring endpoint (Issue #16).

    Evaluates discrete candidates against the prompt with 0 generated tokens in < 15ms.
    Returns choice (argmax), scores array, probabilities dictionary, confidence, and timing.
    """
    t0 = time.perf_counter()

    if not req.prompt or not isinstance(req.prompt, str):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": {
                    "message": "'prompt' must be a non-empty string.",
                    "type": "invalid_request_error",
                    "code": "missing_prompt"
                }
            }
        )

    if not isinstance(req.candidates, list) or len(req.candidates) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": {
                    "message": "'candidates' must be a non-empty list of candidate strings.",
                    "type": "invalid_request_error",
                    "code": "missing_candidates"
                }
            }
        )

    # llama.cpp is a real backend; the in-process reflex path needs loaded weights.
    # A configured-but-unreachable llama-server must surface as 503, never as a fake 200.
    if not os.environ.get("LLAMACPP_BASE_URL"):
        _require_loaded_model("/v1/score")
    try:
        score_res = client.score(
            prompt=req.prompt,
            candidates=req.candidates,
            model=req.model,
            temperature=req.temperature or 1.0,
        )
    except LlamaCppUnavailableError as exc:
        logger.error("/v1/score refused: llama.cpp backend unavailable: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": {
                    "message": f"llama.cpp scoring backend unavailable: {exc}",
                    "type": "backend_unavailable",
                    "code": "llamacpp_unreachable",
                }
            },
        )

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    score_res["timing_ms"] = round(elapsed_ms, 2)
    return score_res


def _state_text(context: str, state: Optional[Union[str, Dict[str, Any]]]) -> str:
    if state is None:
        return context
    extra = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False, sort_keys=True)
    return f"{context}\n{extra}" if context else extra


def _semantic_error(code: int, err_code: str, message: str) -> HTTPException:
    return HTTPException(
        status_code=code,
        detail={"error": {"message": message, "type": "semantic_error", "code": err_code}},
    )


def _load_scorer():
    try:
        return get_semantic_scorer()
    except (FileNotFoundError, RuntimeError, ValueError, OSError) as exc:
        raise _semantic_error(status.HTTP_503_SERVICE_UNAVAILABLE, "semantic_backbone_unavailable", str(exc))


@app.post("/v1/semantic_ask", dependencies=[Depends(verify_api_token)])
def handle_semantic_ask(req: SemanticAskRequest):
    """Rank candidate actions by calibrated LM likelihood given the context."""
    scorer = _load_scorer()
    context = _state_text(req.context, req.state)
    history = req.history or []
    t0 = time.perf_counter()
    try:
        frame, frame_source = select_ask_frame(context, req.candidates, history, req.frame)
        result = scorer.score(context, req.candidates, frame=frame, history=history)
    except ValueError as exc:
        raise _semantic_error(status.HTTP_400_BAD_REQUEST, "invalid_semantic_request", str(exc))
    best = result.chosen_index
    body = {
        "chosen": result.candidates[best].name,
        "chosen_index": best,
        "candidates": [c.as_dict() for c in result.candidates],
        "entropy": result.entropy,
        "scorer": {"id": scorer.scorer_id, "frame": frame, "frame_source": frame_source, "manifold": None},
        "embedding_dim": scorer.hidden_size,
        "prompt_tokens": result.prompt_tokens,
        "timing_ms": round((time.perf_counter() - t0) * 1000.0, 2),
    }
    if req.return_embedding:
        body["embedding"] = result.prompt_state
    return body


def tool_continuation(tool: Union[str, "ToolSpec"]) -> str:
    """What the model scores for one tool: its name and, if given, its description.

    The description carries the meaning when the name does not ("tool_7").
    Each tool is scored as its own continuation after the intent, so the
    prompt never lists the options (a listed set biases the 0.5B backbone
    toward list position).
    """
    if isinstance(tool, str):
        return candidate_text(tool)
    desc = " ".join((tool.description or "").split())
    return candidate_text(tool.name) + (f": {desc}" if desc else "")


@app.post("/v1/semantic_route", dependencies=[Depends(verify_api_token)])
def handle_semantic_route(req: SemanticRouteRequest):
    """Rank candidate tools by calibrated LM likelihood given the intent.

    Score = PMI of "<name>: <description>" after the intent, so a tool whose
    name says nothing is ranked by what its description says it does.
    """
    scorer = _load_scorer()
    names = [t if isinstance(t, str) else t.name for t in req.tools]
    texts = [tool_continuation(t) for t in req.tools]
    t0 = time.perf_counter()
    try:
        result = scorer.score(_state_text(req.intent, req.state), names, frame=ROUTE_FRAME, texts=texts)
    except ValueError as exc:
        raise _semantic_error(status.HTTP_400_BAD_REQUEST, "invalid_semantic_request", str(exc))
    ranked = sorted(result.candidates, key=lambda c: c.probability, reverse=True)
    return {
        "ranked": [c.as_dict() for c in ranked],
        "selected": [c.name for c in ranked[: req.top_k]],
        "entropy": result.entropy,
        "scorer": {"id": scorer.scorer_id, "frame": ROUTE_FRAME, "manifold": None,
                   "scored_text": "name_and_description"},
        "timing_ms": round((time.perf_counter() - t0) * 1000.0, 2),
    }


@app.post("/v1/semantic_risk", dependencies=[Depends(verify_api_token)])
def handle_semantic_risk(req: SemanticRiskRequest):
    """Probability that carrying out ``text`` is destructive, irreversible,
    privilege-escalating or security-bypassing. Any language; no keyword list.

    The Rust PolicyGate maps it to a tier with the two calibrated thresholds
    returned here (see risk_data/README.md for the measured error rates).
    """
    _load_scorer()
    try:
        classifier = get_risk_classifier()
        result = classifier.assess(req.text)
    except ValueError as exc:
        raise _semantic_error(status.HTTP_400_BAD_REQUEST, "invalid_semantic_request", str(exc))
    return {
        **result.as_dict(),
        "thresholds": {"escalate": ESCALATE_THRESHOLD, "hard_stop": HARD_STOP_THRESHOLD},
        "classifier": {"id": classifier.classifier_id, "method": "in_context_pmi_log_odds"},
    }


SEMANTIC_ENDPOINTS = ["/v1/semantic_ask", "/v1/semantic_route", "/v1/semantic_risk"]


@app.get("/v1/semantic_health")
def semantic_health():
    """Identity probe for the Rust bridge. No auth and no model load, so a
    wrong service on the port (404 here) is told apart from a slow start."""
    from gen_zero.service import semantic_scorer

    return {
        "service": "gen-zero-semantic",
        "version": "1.13.0",
        "endpoints": SEMANTIC_ENDPOINTS,
        "backbone_loaded": semantic_scorer._SCORER is not None,
    }


@app.get("/v1/decisions/stream", dependencies=[Depends(verify_api_token)])
@app.get("/decisions/stream", dependencies=[Depends(verify_api_token)])
async def handle_decisions_stream(request: Request, channel: str = "l2_market"):
    """Sub-second SSE decision stream (Issue #18).

    No live L2 market feed is wired into this service. The endpoint used to
    synthesize a sine-wave order book and stream decisions on it as if it were
    real; that is gone. Until a real feed producer exists, this fails closed.
    """
    logger.error("/v1/decisions/stream refused: no live market feed for channel %r", channel)
    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail={
            "error": {
                "message": (
                    f"No live market feed is connected for channel '{channel}'; "
                    "synthetic order-book streaming is disabled per fail-closed policy"
                ),
                "type": "feed_unavailable",
                "code": "market_feed_unavailable",
            }
        },
    )


def _decide_with_optional_trajectory(want_trajectory: bool, **kwargs) -> Dict[str, Any]:
    """client.decide(..., return_trajectory=want_trajectory), single pass.

    decide() itself scopes the trailing-rollout ValueError narrowly (only the bonus
    world-model continuation, not the core decision) and reports it via
    'trajectory_status', so this wrapper never needs to catch anything or re-run the
    decision a second time.
    """
    return client.decide(**kwargs, return_trajectory=want_trajectory)


def _decide_abstain_info(
    res: Dict[str, Any], candidates: Optional[List[str]] = None, is_noul: bool = False
) -> Optional[Dict[str, str]]:
    """Detects a refused/degenerate client.decide()/decide_batch() result.

    Triggers on: action is ABSTAIN or missing (no admissible candidate -- covers the
    all-infeasible-candidate-set and all-expert-invalid-consensus cases), a missing/empty
    probs distribution, or any probability that isn't a finite number. Returns None when
    the result is a genuine, answerable decision; otherwise a dict with 'status' (the
    kernel's own status string when it provided one) and 'message' for the caller.

    Formatting code MUST call this before touching 'probs'/'action' so an internal refusal
    is never repackaged (e.g. via `1.0 - prob_true`, a uniform distribution, or `cands[0]`)
    into what looks like a normal successful answer.
    """
    action = res.get("action")
    kernel_status = res.get("status")
    if action is None or action == "ABSTAIN":
        return {
            "status": kernel_status or "ABSTAIN",
            "message": "Decision kernel abstained: no admissible candidate action.",
        }
    probs = res.get("probs")
    if not probs:
        return {
            "status": kernel_status or "MISSING_PROBS",
            "message": "Decision kernel returned no probability distribution.",
        }
    if not isinstance(probs, dict):
        return {
            "status": kernel_status or "INVALID_PROBS_TYPE",
            "message": f"Decision kernel returned invalid probability structure: {type(probs).__name__}",
        }
    total_p = 0.0
    for cand, p in probs.items():
        try:
            fp = float(p)
            finite = math.isfinite(fp)
        except (TypeError, ValueError, OverflowError):
            finite = False
        if not finite:
            return {
                "status": kernel_status or "NON_FINITE_PROBS",
                "message": f"Decision kernel returned a non-finite probability for candidate {cand!r}.",
            }
        if not (0.0 <= fp <= 1.0):
            return {
                "status": kernel_status or "OUT_OF_RANGE_PROBS",
                "message": f"Probability for candidate {cand!r} is out of range [0, 1]: {fp}",
            }
        total_p += fp

    if not is_noul and abs(total_p - 1.0) > 1e-2:
        return {
            "status": kernel_status or "UNNORMALIZED_PROBS",
            "message": f"Decision kernel probabilities sum to {total_p:.4f}, expected ~1.0.",
        }

    if candidates is not None:
        cand_set = set(candidates)
        prob_set = set(probs.keys())
        if not is_noul and action not in cand_set:
            return {
                "status": kernel_status or "ACTION_NOT_IN_CANDIDATES",
                "message": f"Selected action {action!r} not in admissible candidates: {candidates}",
            }
        missing = [c for c in candidates if c not in prob_set]
        if missing:
            return {
                "status": kernel_status or "MISSING_CANDIDATE_PROBS",
                "message": f"Probabilities missing for candidates: {missing}",
            }
        extra = [k for k in probs.keys() if k not in cand_set]
        if extra:
            return {
                "status": kernel_status or "EXTRA_CANDIDATE_PROBS",
                "message": f"Decision kernel returned unexpected candidates outside admissible set: {extra}",
            }

    if not is_noul and action not in probs:
        return {
            "status": kernel_status or "ACTION_NOT_IN_PROBS",
            "message": f"Selected action {action!r} not found in candidate probabilities.",
        }

    return None


def _decide_noul_probs(res: Dict[str, Any]) -> Tuple[float, float, Dict[str, str]]:
    """Reads the kernel's real true/false probabilities for a noul question; never fabricates them.

    Returns (prob_true, prob_false, {}) when the kernel supplied a finite pair that sums to
    ~1 (noul always decides over exactly ['true', 'false']). Returns (0.0, 0.0, error_info)
    when 'true' or 'false' is missing, non-finite, or the pair doesn't sum to 1 -- callers
    must treat that as an abstain/invalid answer, never as `1.0 - prob_true`: a caller-side
    complement is a guess, not the kernel's own estimate, and would silently paper over a
    kernel bug or genuine gate refusal. Call this before indexing res['probs']['true']
    directly, so a missing key here is reported explicitly instead of raising a KeyError.
    """
    probs = res.get("probs") or {}
    prob_true = probs.get("true")
    prob_false = probs.get("false")
    if prob_true is None or prob_false is None:
        missing = [name for name, val in (("true", prob_true), ("false", prob_false)) if val is None]
        return 0.0, 0.0, {
            "status": res.get("status") or "MISSING_NOUL_PROB",
            "message": f"Decision kernel returned no {' or '.join(missing)!s} probability for this noul question.",
        }
    try:
        prob_true = float(prob_true)
        prob_false = float(prob_false)
    except (TypeError, ValueError):
        return 0.0, 0.0, {
            "status": res.get("status") or "NON_FINITE_PROBS",
            "message": "Decision kernel returned a non-numeric true/false probability.",
        }
    if not (math.isfinite(prob_true) and math.isfinite(prob_false)):
        return 0.0, 0.0, {
            "status": res.get("status") or "NON_FINITE_PROBS",
            "message": "Decision kernel returned a non-finite true/false probability.",
        }
    if not (0.0 <= prob_true <= 1.0 and 0.0 <= prob_false <= 1.0):
        return 0.0, 0.0, {
            "status": res.get("status") or "OUT_OF_RANGE_NOUL_PROBS",
            "message": f"Decision kernel's true/false probabilities are outside [0, 1] "
                       f"(true={prob_true}, false={prob_false}).",
        }
    if abs((prob_true + prob_false) - 1.0) > 1e-2:
        return 0.0, 0.0, {
            "status": res.get("status") or "UNNORMALIZED_NOUL_PROBS",
            "message": f"Decision kernel's true/false probabilities do not sum to 1 "
                       f"(true={prob_true}, false={prob_false}).",
        }
    return prob_true, prob_false, {}


@app.post("/api/alpha/decisions", dependencies=[Depends(verify_api_token)])
@app.post("/v1/decisions", dependencies=[Depends(verify_api_token)])
@app.post("/decisions", dependencies=[Depends(verify_api_token)])
def handle_decisions(req: DecisionsRequest):
    """Answers typed decision questions across state or batch states with 0 generated tokens."""
    _require_loaded_model("/v1/decisions")
    t0 = time.perf_counter()

    if req.state is None and req.states is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": {
                    "message": "Must provide either 'state' or 'states' in DecisionsRequest.",
                    "type": "invalid_request_error",
                    "code": "missing_state"
                }
            }
        )

    # Batch States Mode
    if req.states is not None:
        if req.return_trajectory:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": {
                        "message": "'return_trajectory' is only supported in single-state mode ('state', not 'states').",
                        "type": "invalid_request_error",
                        "code": "unsupported_return_trajectory_batch"
                    }
                }
            )
        if not isinstance(req.states, list) or len(req.states) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": {
                        "message": "'states' must be a non-empty list of state objects or strings.",
                        "type": "invalid_request_error",
                        "code": "invalid_states"
                    }
                }
            )

        states_strs = [s if isinstance(s, str) else str(s) for s in req.states]
        all_answers: List[Dict[str, Any]] = [{} for _ in range(len(req.states))]

        for q_id, q_spec in req.questions.items():
            q_type = q_spec.type.lower()
            instructions = q_spec.instructions or ""
            criteria = q_spec.criteria

            if q_type == "noul":
                cands = ["true", "false"]
                crit_true = criteria.get("true", "") if isinstance(criteria, dict) else ""
                crit_false = criteria.get("false", "") if isinstance(criteria, dict) else ""
                prompts = [
                    f"{s_str}\nQ: {instructions}\nTrue: {crit_true}\nFalse: {crit_false}"
                    for s_str in states_strs
                ]
                batch_res = client.decide_batch(prompts, candidates=cands, mode="reflex")
                for idx, res in enumerate(batch_res):
                    abstain = _decide_abstain_info(res, is_noul=True)
                    if abstain is not None:
                        all_answers[idx][q_id] = {
                            "type": "noul",
                            "status": "ABSTAIN",
                            "kernel_status": abstain["status"],
                            "error": abstain["message"],
                        }
                        continue
                    prob_true, prob_false, noul_error = _decide_noul_probs(res)
                    if noul_error:
                        all_answers[idx][q_id] = {
                            "type": "noul",
                            "status": "ABSTAIN",
                            "kernel_status": noul_error["status"],
                            "error": noul_error["message"],
                        }
                        continue
                    conf = compute_closed_form_confidence({"true": prob_true, "false": prob_false})
                    all_answers[idx][q_id] = {
                        "type": "noul",
                        "noul": round(prob_true, 4),
                        "confidence": round(conf, 4)
                    }

            elif q_type == "choice":
                candidate_descriptions = None
                if isinstance(criteria, dict) and criteria:
                    cands = list(criteria.keys())
                    candidate_descriptions = {str(k): str(v) for k, v in criteria.items()}
                    desc_body = "\n".join(f"{k}: {v}" for k, v in criteria.items())
                    prompts = [
                        f"{s_str}\nQ: {instructions}\n{desc_body}"
                        for s_str in states_strs
                    ]
                elif isinstance(criteria, list) and criteria:
                    cands = [str(c) for c in criteria]
                    candidate_descriptions = {str(c): str(c) for c in criteria}
                    prompts = [
                        f"{s_str}\nQ: {instructions}\nOptions: {', '.join(cands)}"
                        for s_str in states_strs
                    ]
                else:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail={
                            "error": {
                                "message": f"Question '{q_id}' of type 'choice' requires non-empty criteria options.",
                                "type": "invalid_request_error",
                                "code": "invalid_criteria"
                            }
                        }
                    )

                batch_res = client.decide_batch(
                    prompts,
                    candidates=cands,
                    mode="reflex",
                    candidate_descriptions=candidate_descriptions
                )
                for idx, res in enumerate(batch_res):
                    abstain = _decide_abstain_info(res, candidates=cands)
                    if abstain is not None:
                        all_answers[idx][q_id] = {
                            "type": "choice",
                            "status": "ABSTAIN",
                            "kernel_status": abstain["status"],
                            "error": abstain["message"],
                        }
                        continue
                    probs = res["probs"]
                    conf = compute_closed_form_confidence(probs)
                    all_answers[idx][q_id] = {
                        "type": "choice",
                        "choice": res["action"],
                        "confidence": round(conf, 4),
                        "probabilities": {k: round(v, 4) for k, v in probs.items()}
                    }

            elif q_type == "score":
                if isinstance(criteria, list) and criteria:
                    levels = [str(c) for c in criteria]
                elif isinstance(criteria, dict) and criteria:
                    levels = list(criteria.keys())
                else:
                    levels = ["Low", "Medium", "High"]

                scale_str = " -> ".join(levels)
                prompts = [
                    f"{s_str}\nQ: {instructions}\nScale: {scale_str}"
                    for s_str in states_strs
                ]
                batch_res = client.decide_batch(prompts, candidates=levels, mode="reflex")
                for idx, res in enumerate(batch_res):
                    abstain = _decide_abstain_info(res, candidates=levels)
                    if abstain is not None:
                        all_answers[idx][q_id] = {
                            "type": "score",
                            "status": "ABSTAIN",
                            "kernel_status": abstain["status"],
                            "error": abstain["message"],
                        }
                        continue
                    probs = res["probs"]
                    exp_score = sum(i * probs[lvl] for i, lvl in enumerate(levels))
                    conf = compute_closed_form_confidence(probs)
                    all_answers[idx][q_id] = {
                        "type": "score",
                        "score": round(exp_score, 3),
                        "confidence": round(conf, 4),
                        "probabilities": {lvl: round(probs[lvl], 4) for lvl in levels}
                    }

            else:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail={
                        "error": {
                            "message": f"Unsupported decision question type '{q_spec.type}'. Must be 'noul', 'choice', or 'score'.",
                            "type": "invalid_request_error",
                            "code": "unsupported_question_type"
                        }
                    }
                )

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        total_tokens = max(1, sum(len(s) for s in states_strs) // 4)

        return {
            "id": f"batch-decision-{uuid.uuid4().hex[:12]}",
            "created": int(time.time()),
            "model": req.model,
            "batch_size": len(req.states),
            "results": [
                {
                    "state_index": i,
                    "answers": all_answers[i]
                }
                for i in range(len(req.states))
            ],
            "usage": {
                "prompt_tokens": total_tokens,
                "completion_tokens": 0,
                "total_tokens": total_tokens,
                "estimated": True,
                "estimate_method": "len(text) // 4, not a real tokenizer count",
            },
            "timing_ms": round(elapsed_ms, 2)
        }

    # Single State Mode (Backward Compatibility)
    state_str = req.state if isinstance(req.state, str) else str(req.state)
    answers: Dict[str, Any] = {}

    for q_id, q_spec in req.questions.items():
        q_type = q_spec.type.lower()
        instructions = q_spec.instructions or ""
        criteria = q_spec.criteria

        if q_type == "noul":
            cands = ["true", "false"]
            crit_true = ""
            crit_false = ""
            if isinstance(criteria, dict):
                crit_true = criteria.get("true", "")
                crit_false = criteria.get("false", "")

            combined_prompt = f"{state_str}\nQ: {instructions}\nTrue: {crit_true}\nFalse: {crit_false}"
            decide_res = _decide_with_optional_trajectory(
                req.return_trajectory,
                state=combined_prompt,
                candidates=cands,
                mode="reflex",
            )
            abstain = _decide_abstain_info(decide_res, is_noul=True)
            if abstain is not None:
                answers[q_id] = {
                    "type": "noul",
                    "status": "ABSTAIN",
                    "kernel_status": abstain["status"],
                    "error": abstain["message"],
                }
            else:
                prob_true, prob_false, noul_error = _decide_noul_probs(decide_res)
                if noul_error:
                    answers[q_id] = {
                        "type": "noul",
                        "status": "ABSTAIN",
                        "kernel_status": noul_error["status"],
                        "error": noul_error["message"],
                    }
                else:
                    conf = compute_closed_form_confidence({"true": prob_true, "false": prob_false})
                    answers[q_id] = {
                        "type": "noul",
                        "noul": round(prob_true, 4),
                        "confidence": round(conf, 4)
                    }
            if req.return_trajectory:
                answers[q_id]["trajectory"] = decide_res.get("trajectory")
                answers[q_id]["trajectory_status"] = decide_res.get("trajectory_status")

        elif q_type == "choice":
            candidate_descriptions = None
            if isinstance(criteria, dict) and criteria:
                cands = list(criteria.keys())
                candidate_descriptions = {str(k): str(v) for k, v in criteria.items()}
                combined_prompt = f"{state_str}\nQ: {instructions}\n" + "\n".join(f"{k}: {v}" for k, v in criteria.items())
            elif isinstance(criteria, list) and criteria:
                cands = [str(c) for c in criteria]
                candidate_descriptions = {str(c): str(c) for c in criteria}
                combined_prompt = f"{state_str}\nQ: {instructions}\nOptions: {', '.join(cands)}"
            else:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail={
                        "error": {
                            "message": f"Question '{q_id}' of type 'choice' requires non-empty criteria options.",
                            "type": "invalid_request_error",
                            "code": "invalid_criteria"
                        }
                    }
                )

            decide_res = _decide_with_optional_trajectory(
                req.return_trajectory,
                state=combined_prompt,
                candidates=cands,
                mode="reflex",
                candidate_descriptions=candidate_descriptions,
            )
            abstain = _decide_abstain_info(decide_res, candidates=cands)
            if abstain is not None:
                answers[q_id] = {
                    "type": "choice",
                    "status": "ABSTAIN",
                    "kernel_status": abstain["status"],
                    "error": abstain["message"],
                }
            else:
                probs = decide_res["probs"]
                conf = compute_closed_form_confidence(probs)
                answers[q_id] = {
                    "type": "choice",
                    "choice": decide_res["action"],
                    "confidence": round(conf, 4),
                    "probabilities": {k: round(v, 4) for k, v in probs.items()}
                }
            if req.return_trajectory:
                answers[q_id]["trajectory"] = decide_res.get("trajectory")
                answers[q_id]["trajectory_status"] = decide_res.get("trajectory_status")

        elif q_type == "score":
            if isinstance(criteria, list) and criteria:
                levels = [str(c) for c in criteria]
            elif isinstance(criteria, dict) and criteria:
                levels = list(criteria.keys())
            else:
                levels = ["Low", "Medium", "High"]

            combined_prompt = f"{state_str}\nQ: {instructions}\nScale: {' -> '.join(levels)}"
            decide_res = _decide_with_optional_trajectory(
                req.return_trajectory,
                state=combined_prompt,
                candidates=levels,
                mode="reflex",
            )
            abstain = _decide_abstain_info(decide_res, candidates=levels)
            if abstain is not None:
                answers[q_id] = {
                    "type": "score",
                    "status": "ABSTAIN",
                    "kernel_status": abstain["status"],
                    "error": abstain["message"],
                }
            else:
                probs = decide_res["probs"]
                exp_score = sum(idx * probs[lvl] for idx, lvl in enumerate(levels))
                conf = compute_closed_form_confidence(probs)
                answers[q_id] = {
                    "type": "score",
                    "score": round(exp_score, 3),
                    "confidence": round(conf, 4),
                    "probabilities": {lvl: round(probs[lvl], 4) for lvl in levels}
                }
            if req.return_trajectory:
                answers[q_id]["trajectory"] = decide_res.get("trajectory")
                answers[q_id]["trajectory_status"] = decide_res.get("trajectory_status")

        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": {
                        "message": f"Unsupported decision question type '{q_spec.type}'. Must be 'noul', 'choice', or 'score'.",
                        "type": "invalid_request_error",
                        "code": "unsupported_question_type"
                    }
                }
            )

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    # This is a character-length heuristic (len(text) // 4), not a real tokenizer count.
    # Do not report it as measured usage; callers that bill on tokens must not trust it.
    approx_tokens = max(1, len(state_str) // 4)

    return {
        "id": f"decision-{uuid.uuid4().hex[:12]}",
        "created": int(time.time()),
        "model": req.model,
        "answers": answers,
        "usage": {
            "prompt_tokens": approx_tokens,
            "completion_tokens": 0,
            "total_tokens": approx_tokens,
            "estimated": True,
            "estimate_method": "len(text) // 4, not a real tokenizer count"
        },
        "timing_ms": round(elapsed_ms, 2)
    }


@app.post("/api/alpha/decide_step", dependencies=[Depends(verify_api_token)])
@app.post("/v1/decide_step", dependencies=[Depends(verify_api_token)])
@app.post("/decide_step", dependencies=[Depends(verify_api_token)])
def handle_decide_step(req: DecideStepRequest):
    """Answers composite 4-tuple decision (target, action, done, risk) with policy gate in sub-5ms."""
    _require_loaded_model("/v1/decide_step")
    from gen_zero.gate.policy_gate import DomainRiskProfile
    prof = None
    if req.risk_profile is not None:
        r_name = req.risk_profile.strip().lower()
        if r_name == "read_only":
            prof = DomainRiskProfile.read_only()
        elif r_name == "critical":
            prof = DomainRiskProfile.critical()
        elif r_name == "standard":
            prof = DomainRiskProfile.standard()
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": {
                        "message": (
                            f"Unknown risk_profile {req.risk_profile!r}. "
                            "Must be one of: 'read_only', 'standard', 'critical'."
                        ),
                        "type": "invalid_request_error",
                        "code": "invalid_risk_profile"
                    }
                }
            )

    step_result = client.decide_step(
        state=req.state,
        affordances=req.affordances,
        action_types=req.action_types,
        goal=req.goal,
        risk_criteria=req.risk_criteria,
        state_feedback=req.state_feedback,
        max_affordances=req.max_affordances,
        apply_policy_gate=req.apply_policy_gate,
        profile=prof,
        return_trajectory=req.return_trajectory,
    )

    # This is a character-length heuristic (len(text) // 4), not a real tokenizer count.
    # Do not report it as measured usage; callers that bill on tokens must not trust it.
    approx_tokens = max(1, len(str(req.state)) // 4)
    return {
        "id": f"step-{uuid.uuid4().hex[:12]}",
        "created": int(time.time()),
        "model": req.model,
        "step": step_result,
        "usage": {
            "prompt_tokens": approx_tokens,
            "completion_tokens": 0,
            "total_tokens": approx_tokens,
            "estimated": True,
            "estimate_method": "len(text) // 4, not a real tokenizer count"
        },
        "timing_ms": step_result.get("timing_ms", 0.0)
    }


def _run_world_model_endpoint(fn, **kwargs) -> Dict[str, Any]:
    """Bad input -> 400, simulator unavailable (e.g. weights not loaded) -> 503. Never a made-up result."""
    def error(code: int, err_code: str, exc: Exception) -> HTTPException:
        return HTTPException(
            status_code=code,
            detail={"error": {"message": str(exc), "type": "world_model_error", "code": err_code}},
        )

    try:
        return fn(**kwargs)
    except (ValueError, KeyError, TypeError, IndexError) as exc:
        raise error(status.HTTP_400_BAD_REQUEST, "invalid_world_model_request", exc)
    except RuntimeError as exc:
        raise error(status.HTTP_503_SERVICE_UNAVAILABLE, "world_model_unavailable", exc)


@app.post("/v1/simulate", dependencies=[Depends(verify_api_token)])
def handle_simulate(req: SimulateRequest):
    return _run_world_model_endpoint(client.simulate, state=req.state, actions=req.actions, horizon=req.horizon)


@app.post("/v1/what_if", dependencies=[Depends(verify_api_token)])
def handle_what_if(req: WhatIfRequest):
    return _run_world_model_endpoint(client.what_if, state=req.state, candidates=req.candidates, horizon=req.horizon)


@app.post("/v1/audit_action", dependencies=[Depends(verify_api_token)])
def handle_audit_action(req: AuditActionRequest):
    return _run_world_model_endpoint(
        client.audit_action, state=req.state, action=req.action, horizon=req.horizon,
        continuation_actions=req.continuation_actions, warn_risk=req.warn_risk,
    )


def main(argv: Optional[List[str]] = None) -> None:
    """Run the semantic scorer. Default port 8995: 8999 is the MCP SSE port."""
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(prog="gen-zero-semantic", description=main.__doc__)
    parser.add_argument("--host", default=os.environ.get("GENZERO_SEMANTIC_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int,
                        default=int(os.environ.get("GENZERO_SEMANTIC_PORT", str(DEFAULT_SEMANTIC_PORT))))
    args = parser.parse_args(argv)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
