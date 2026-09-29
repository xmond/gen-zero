"""Regression coverage for safety gates and degradation across SDK entrypoints."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from gen_zero.client import GenZero
from gen_zero.planner.engines.cpsat_formal_engine import CpSatFormalEngine


def test_score_preserves_degradation(monkeypatch):
    monkeypatch.delenv("LLAMACPP_BASE_URL", raising=False)
    client = GenZero.__new__(GenZero)
    client.decide = Mock(return_value={
        "action": "a", "probs": {"a": 0.9, "b": 0.1},
        "degraded": True, "degraded_reason": "untrained_weights_fallback",
    })
    result = client.score("prompt", ["a", "b"])
    assert result["degraded"] is True
    assert result["degraded_reason"] == "untrained_weights_fallback"


@pytest.mark.parametrize("action,pruned,expected", [
    ("safe", True, "safe"), ("ABSTAIN", True, "ABSTAIN"),
    ("ABSTAIN", False, "ABSTAIN"), ("safe", False, "unsafe"),
])
def test_visual_direct_readout_respects_gate(action, pruned, expected):
    client = GenZero.__new__(GenZero)
    client.weights_loaded_from_checkpoint = True
    client.vision_engine = SimpleNamespace(
        prefill_visual_context=Mock(return_value={}),
        score_candidates_direct=Mock(return_value={
            "best_action": "unsafe", "probs": {"unsafe": .9, "safe": .1},
        }),
    )
    client.decide = Mock(return_value={"action": action, "pruned_by_cpsat": pruned})
    result = client.decide_visual("image", ["unsafe", "safe"], mode="direct_readout")
    assert result["action"] == expected


@pytest.mark.parametrize("mode", ["fast", "reflex"])
def test_batch_combines_hard_rules_and_model_validity(mode):
    class Model:
        scalar = True
        forward = True

        def __call__(self, examples, **kwargs):
            assert len(examples) == 4
            return (torch.tensor([[100., 1.]] * 4),
                    torch.tensor([[True, True], [True, True], [True, True], [True, False]]),
                    torch.zeros(4))

    client = GenZero.__new__(GenZero)
    client.model = Model()
    client.weights_loaded_from_checkpoint = True
    client.modality_router = SimpleNamespace(ingest=lambda raw_input: SimpleNamespace(normalized_state=raw_input))
    client.cp_sat_solver = CpSatFormalEngine()
    client.cp_sat_solver.registered_rules.append(lambda state, action: state != "deny_all")
    client.decide = Mock(side_effect=AssertionError("must exercise batched forward"))
    results = client.decide_batch(["unauthorized", "authorized", "deny_all", "unauthorized"],
                                  ["execute", "cancel"], mode=mode)
    assert [r["action"] for r in results] == ["cancel", "execute", "ABSTAIN", "ABSTAIN"]
    assert results[0]["probs"] == {"execute": 0., "cancel": 1.}
    assert results[0]["pruned_by_cpsat"] is True
    assert results[1]["pruned_by_cpsat"] is False
    assert results[2]["confidence"] == results[3]["confidence"] == 0.


def test_visual_registered_hard_rule_blocks_direct_favorite():
    client = GenZero()
    client.cp_sat_solver.register_hard_rule(lambda state, action: action != "unsafe")
    client.vision_engine = SimpleNamespace(
        prefill_visual_context=Mock(return_value={"last_hidden_state": [0.] * 128}),
        score_candidates_direct=Mock(return_value={
            "best_action": "unsafe", "probs": {"unsafe": .99, "safe": .01},
        }),
    )
    # T3-M01: MCTS no longer bypasses trajectory rollout for a single surviving
    # candidate, so it now genuinely simulates repeated "safe" steps through the
    # generic heuristic world model. That model treats an unrelated synthetic visual
    # dict as a snake-style grid state and deterministically "collides" after 3 such
    # steps -- an artifact of the fallback heuristic, not of the cp_sat gate this test
    # exercises. Stub a trivially safe transition so the test isolates the behavior it
    # actually targets (the registered hard rule routes the decision away from
    # "unsafe" to the surviving "safe" candidate) from that unrelated heuristic quirk.
    client._adaptive_transition_tuple = lambda s, a: (s, 0.0, False)
    result = client.decide_visual("image", ["unsafe", "safe"], mode="direct_readout")
    assert result["action"] == "safe"
    assert result["pruned_by_cpsat"] is True


def test_score_preserves_provenance_and_abstention(monkeypatch):
    monkeypatch.delenv("LLAMACPP_BASE_URL", raising=False)
    client = GenZero.__new__(GenZero)
    client.decide = Mock(return_value={
        "action": "ABSTAIN", "probs": {"a": 0.0},
        "degraded": True, "degraded_reason": "untrained_weights_fallback",
        "scorer": "untrained_weights_fallback", "weights_loaded_from_checkpoint": False,
        "status": "INFEASIBLE_ABSTAIN", "pruned_by_cpsat": True,
    })
    result = client.score("prompt", ["a"])
    assert result["choice"] == "ABSTAIN"
    assert result["confidence"] == 0.0
    for key in ("scorer", "weights_loaded_from_checkpoint", "status", "pruned_by_cpsat"):
        assert result[key] == client.decide.return_value[key]


@pytest.mark.parametrize("failure", ["eval", "forward", "nonfinite"])
def test_batch_fallback_retains_both_degradation_causes(failure):
    class Model:
        scalar = True
        forward = True

        def eval(self):
            if failure == "eval":
                raise RuntimeError("eval failed")

        def __call__(self, *args, **kwargs):
            if failure == "forward":
                raise RuntimeError("forward failed")
            return torch.tensor([[float("nan")]]), torch.tensor([[True]]), torch.zeros(1)

    client = GenZero.__new__(GenZero)
    client.model = Model()
    client.weights_loaded_from_checkpoint = True
    client.modality_router = SimpleNamespace(ingest=lambda raw_input: SimpleNamespace(normalized_state=raw_input))
    client.cp_sat_solver = CpSatFormalEngine()
    client.decide = Mock(return_value={
        "action": "a", "degraded": True, "degraded_reason": "existing_cause",
    })
    result = client.decide_batch(["state"], ["a"], mode="fast")[0]
    assert result["degraded"] is True
    assert "existing_cause" in result["degraded_reason"]
    assert ("batch_nonfinite_logits" if failure == "nonfinite" else "batch_torch_exception:RuntimeError") in result["degraded_reason"]


def test_visual_direct_readout_rejects_non_candidate_favorite():
    client = GenZero.__new__(GenZero)
    client.weights_loaded_from_checkpoint = True
    client.vision_engine = SimpleNamespace(
        prefill_visual_context=Mock(return_value={}),
        score_candidates_direct=Mock(return_value={"best_action": "unknown", "probs": {"safe": 1.0}}),
    )
    client.decide = Mock(return_value={"action": "safe", "pruned_by_cpsat": False})
    assert client.decide_visual("image", ["safe"], mode="direct_readout")["action"] == "safe"


@pytest.mark.parametrize("failure", ["eval", "nonfinite"])
def test_reflex_failure_is_visible_in_metadata(failure):
    class Model:
        scalar = True
        forward = True

        def eval(self):
            if failure == "eval":
                raise RuntimeError("eval failed")

        def __call__(self, *args, **kwargs):
            return torch.tensor([[float("nan")]]), torch.tensor([[True]]), torch.zeros(1)

    client = GenZero.__new__(GenZero)
    client.model = Model()
    client.weights_loaded_from_checkpoint = True
    _, _, _, metadata = client._execute_expert_distribution("reflex", "state", ["a"], None)
    assert metadata["degraded"] is True
    assert metadata["degraded_reason"] == "torch_reflex_exception:" + (
        "RuntimeError" if failure == "eval" else "ValueError"
    )


def test_planner_transition_failure_is_not_silently_scored():
    client = GenZero.__new__(GenZero)
    client.weights_loaded_from_checkpoint = True
    transition = Mock(side_effect=ValueError("broken transition"))
    with pytest.raises(RuntimeError, match="Planner transition failed") as error:
        client._execute_expert_distribution("astar", "state", ["a", "b"], transition)
    assert isinstance(error.value.__cause__, ValueError)


def test_scorer_sync_failure_is_logged(caplog):
    client = GenZero.__new__(GenZero)
    model = SimpleNamespace(export_to_scorer_weights=Mock(side_effect=ValueError("bad weights")))
    assert client.sync_model_to_scorer(model=model) is False
    assert "Model-to-scorer synchronization failed (ValueError: bad weights)" in caplog.text
