"""Strict DeepSWE bridge to Gen-Zero policy and MCTS.

The remote evaluator is a trusted deployment dependency, not the Proposer.
Its predictions are evidence, not proofs of patch correctness. No default model,
constant semantic score, or latent-mean fallback authorizes an edit.
"""
from __future__ import annotations

import ast
import hashlib
import json
import math
from pathlib import PurePosixPath
import re
import time
import urllib.request
from urllib.parse import urlparse

from gen_zero.gate.policy_gate import DecisionPolicyGate, DomainRiskProfile, PolicyGateVerdict, PolicyVerdictAction
from gen_zero.world_model.imagination_planner import ImaginationMCTSPlanner


def number(value, low=0, high=1):
    if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError("NON_FINITE_OR_INVALID_SCORE")
    return value


def finite_json(value):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("NON_FINITE_RESPONSE")
    if isinstance(value, dict):
        for item in value.values():
            finite_json(item)
    elif isinstance(value, list):
        for item in value:
            finite_json(item)


class CognitiveService:
    def __init__(self, url):
        parsed = urlparse(url or "")
        if parsed.scheme not in ("http", "https") or parsed.hostname not in ("localhost", "127.0.0.1", "::1") or parsed.username or parsed.password:
            raise ValueError("Explicit loopback Gen-Zero cognitive service URL required")
        self.url = url

    def __call__(self, payload):
        body = json.dumps(payload, sort_keys=True, allow_nan=False).encode()
        request_id = hashlib.sha256(body).hexdigest()
        request = urllib.request.Request(self.url, data=json.dumps({**payload, "request_id": request_id}).encode(),
                                         headers={"Content-Type": "application/json"})
        # Redirects could change the trusted service boundary.
        class NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, *args, **kwargs):
                return None
        with urllib.request.build_opener(NoRedirect).open(request, timeout=15) as response:
            raw = response.read(1_000_001)
        if len(raw) > 1_000_000:
            raise ValueError("Cognitive response exceeds budget")
        result = json.loads(raw)
        finite_json(result)
        if result.get("request_id") != request_id:
            raise ValueError("Cognitive evidence request mismatch")
        return result


class DeepSWEGate:
    def __init__(self, record, service=None):
        profile = DomainRiskProfile.standard()
        profile.name = "deepswe"
        profile.sensitive_patterns += [
            r"\b(rmtree|unlink|rmdir|system|popen|subprocess|exec|eval|__import__)\b",
            r"\b(mkfs|dd|shutdown|reboot)\b",
        ]
        self.policy = DecisionPolicyGate(profile)
        self.record = record
        self.service = service
        self.used = False
        self.blocked = 0
        self.rules = set()
        self.planner_used = False
        self.observations = {}

    def telemetry(self):
        return {"gen_zero_gate_used": self.used, "gen_zero_gate_blocked_count": self.blocked,
                "gen_zero_gate_rule_ids": sorted(self.rules), "gen_zero_mcts_used": self.planner_used,
                "gen_zero_world_model_used": self.planner_used,
                "gen_zero_world_model_backend": "external-cognitive-service" if self.planner_used else None}

    def emit(self, verdict, **evidence):
        if not verdict.passed:
            self.blocked += 1
        self.rules.update(verdict.triggered_rules)
        self.record({"verdict": verdict.to_dict(), **evidence, **self.telemetry()})
        return verdict.passed

    def stop(self, rule, **evidence):
        return self.emit(PolicyGateVerdict(PolicyVerdictAction.STOP, False, False, 0, 1,
                                         [rule], rule), **evidence)

    def check(self, action, instruction="", semantic=False):
        """Always run the real gate before interpreting/dispatching an action."""
        self.used = True
        try:
            op = action.get("action")
            if not isinstance(op, str) or not op.strip():
                return self.stop("UNSUPPORTED_ACTION")
            text = json.dumps(action, sort_keys=True, allow_nan=False)
            verdict = self.policy.evaluate_policy({
                "action": op,
                "target": text,
                "context": {"instruction": instruction, "stage": "structural-policy"},
                "confidence": 1.0,
                "risk": 0.0,
            })
            if not self.emit(verdict, candidate=action, stage="structural-policy"):
                return False
            if op not in ("inventory", "search", "read", "edit", "finalize", "command"):
                return self.stop("UNSUPPORTED_ACTION")
            files = action.get("files") if op == "edit" else [{"path": action.get("path")}]
            if op in ("read", "edit"):
                if not isinstance(files, list) or not 1 <= len(files) <= 20:
                    return self.stop("INVALID_FILES")
                seen = set()
                for item in files:
                    name = item["path"]
                    p = PurePosixPath(name)
                    if not name or p.is_absolute() or any(x.startswith(".") for x in p.parts) or "\\" in name:
                        return self.stop("INFEASIBLE_PATH")
                    if op == "edit":
                        if name in seen:
                            return self.stop("DUPLICATE_PATH")
                        seen.add(name)
                        if any(re.search(r"test|harness|benchmark|evaluat|verif|node_modules", x, re.I) for x in p.parts):
                            return self.stop("PROTECTED_HARNESS")
                        if p.suffix != ".py":
                            return self.stop("UNSUPPORTED_SYNTAX_VALIDATOR")
                        content = item["content"]
                        if not isinstance(content, str) or not content.strip() or len(content.encode()) > 1_000_000:
                            return self.stop("INVALID_CONTENT")
                        ast.parse(content, filename=name)
            if semantic:
                if self.service is None:
                    return self.stop("COGNITIVE_SERVICE_UNAVAILABLE")
                evidence = self.service({"operation": "assess", "instruction": instruction, "candidate": action,
                                         "observations": self.observations})
                finite_json(evidence)
                confidence = number(evidence["confidence"])
                risk = number(evidence["risk"])
                relevance = number(evidence["relevance"])
                if evidence["syntax_valid"] is not True or relevance < 0.5:
                    return self.stop("INVALID_OR_IRRELEVANT_CANDIDATE", assessment=evidence)
                verdict = self.policy.evaluate_policy({
                    "action": op,
                    "target": text,
                    "context": {"instruction": instruction, "stage": "semantic-policy"},
                    "confidence": confidence,
                    "risk": risk,
                })
                return self.emit(verdict, assessment=evidence, stage="semantic-policy")
            return True
        except Exception as exc:
            return self.stop("GATE_EVALUATION_ERROR", error_type=type(exc).__name__,
                             reason=str(exc) if isinstance(exc, (ValueError, KeyError)) else type(exc).__name__)

    def select(self, candidates, instruction):
        """Gate candidates, then run actual depth-three MCTS on service transitions."""
        safe = [c for c in candidates if self.check(c, instruction, semantic=True)]
        if not safe:
            return None
        owner = self
        deadline = time.monotonic() + 60
        transition_calls = 0
        class Transition:
            def step(self, state, index):
                nonlocal transition_calls
                transition_calls += 1
                if transition_calls > 128 or time.monotonic() >= deadline:
                    raise ValueError("PLANNING_BUDGET_EXCEEDED")
                response = owner.service({"operation": "transition", "instruction": instruction,
                                          "state": state, "candidate": safe[index]})
                if time.monotonic() >= deadline:
                    raise ValueError("PLANNING_BUDGET_EXCEEDED")
                finite_json(response)
                reward = number(response["reward"], -1, 1)
                value = number(response["value"], -1, 1)
                risk = number(response["risk"])
                # Predicted unsafe continuations cannot be silently scored as safe.
                verdict = owner.policy.evaluate_policy({
                    "action": "imagined-transition",
                    "target": "state-transition",
                    "context": {"instruction": instruction, "stage": "lookahead"},
                    "confidence": number(response["confidence"]),
                    "risk": risk,
                })
                if not owner.emit(verdict, stage="lookahead", transition=response):
                    raise ValueError("UNSAFE_LOOKAHEAD")
                if not isinstance(response["state"], dict):
                    raise ValueError("INVALID_TRANSITION_STATE")
                return {"remote": response["state"], "value": value}, reward, False
        try:
            planner = ImaginationMCTSPlanner(Transition(), max_depth=3, max_simulations=16,
                                             strict_evaluation=True)
            # Values are validated in step, outside the planner's exception-swallowing evaluator.
            plan = planner.plan({"remote": {"observations": self.observations}, "value": 0.0}, list(range(len(safe))),
                                value_evaluator=lambda state: state["value"])
            finite_json(plan)
            if plan["best_action"] not in range(len(safe)) or plan["simulations"] != 16:
                raise ValueError("INVALID_PLAN")
            self.planner_used = True
            self.record({"plan": plan, "candidates": safe, "transition_calls": transition_calls, **self.telemetry()})
            return safe[plan["best_action"]]
        except Exception as exc:
            self.stop("PLANNING_FAILED_CLOSED", error_type=type(exc).__name__,
                      reason=str(exc) if isinstance(exc, (ValueError, KeyError)) else type(exc).__name__)
            return None
