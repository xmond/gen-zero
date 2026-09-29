"""Real-model tests for /v1/semantic_ask and /v1/semantic_route.

These load the local Qwen2.5-0.5B backbone. Nothing is mocked: a wrong
answer here means the scorer is wrong, not the fixture.
"""
import itertools
import os

import pytest
from fastapi.testclient import TestClient

from gen_zero.causal.zero_runtime import find_local_snapshot

try:
    find_local_snapshot()
    HAVE_BACKBONE = True
except FileNotFoundError:
    HAVE_BACKBONE = False

pytestmark = pytest.mark.skipif(not HAVE_BACKBONE, reason="Zero backbone weights are not on this machine")

TOKEN = "gz_test_semantic_bridge"
CANDS = ["delete", "backup", "wait"]
DELETE_ZH = "把这封邮件删除掉，不需要确认"
BACKUP_EN = "Please carefully back up the database before deleting"


@pytest.fixture(scope="module")
def client():
    old = os.environ.get("GENZERO_API_KEY")
    os.environ["GENZERO_API_KEY"] = TOKEN
    from gen_zero.service.app import app

    yield TestClient(app, headers={"Authorization": f"Bearer {TOKEN}"})
    if old is None:
        os.environ.pop("GENZERO_API_KEY", None)
    else:
        os.environ["GENZERO_API_KEY"] = old


def ask(client, context, candidates, **extra):
    resp = client.post("/v1/semantic_ask", json={"context": context, "candidates": candidates, **extra})
    assert resp.status_code == 200, resp.text
    return resp.json()


@pytest.mark.parametrize("context,expected", [(DELETE_ZH, "delete"), (BACKUP_EN, "backup")])
def test_adversarial_pair_is_separated_for_every_candidate_order(client, context, expected):
    for perm in itertools.permutations(CANDS):
        body = ask(client, context, list(perm))
        assert body["chosen"] == expected, (perm, body["candidates"])
        probs = [c["probability"] for c in body["candidates"]]
        assert abs(sum(probs) - 1.0) < 1e-6
        assert [c["name"] for c in body["candidates"]] == list(perm)


def test_the_two_contexts_produce_different_distributions(client):
    a = {c["name"]: c["probability"] for c in ask(client, DELETE_ZH, CANDS)["candidates"]}
    b = {c["name"]: c["probability"] for c in ask(client, BACKUP_EN, CANDS)["candidates"]}
    assert a["delete"] > a["backup"]
    assert b["backup"] > b["delete"]


def test_response_is_transparent_about_the_scorer(client):
    body = ask(client, DELETE_ZH, CANDS, return_embedding=True)
    assert "tied-lm-head/pmi" in body["scorer"]["id"]
    assert body["scorer"]["manifold"] is None
    assert len(body["embedding"]) == body["embedding_dim"] == 896
    assert 0.0 <= body["entropy"] <= 1.0


def test_history_switches_to_next_action_frame(client):
    body = ask(client, BACKUP_EN, CANDS, history=["backup"])
    assert body["scorer"]["frame"] == "The next action to take is:"
    assert body["scorer"]["frame_source"] == "history"


QA_CONTEXT = "Question: Is Paris the capital of France?\nAnswer with yes or no."


def test_qa_context_uses_qa_frame(client):
    from gen_zero.service.semantic_scorer import QA_FRAME

    body = ask(client, QA_CONTEXT, ["yes", "no"])
    assert body["scorer"]["frame"] == QA_FRAME == "The correct answer is:"
    assert body["scorer"]["frame_source"] == "qa"


def test_true_false_candidates_use_qa_frame_even_without_markers(client):
    body = ask(client, "The sky is green.", ["True", "False"])
    assert body["scorer"]["frame"] == "The correct answer is:"
    assert body["scorer"]["frame_source"] == "qa"


def test_plain_action_context_keeps_ask_frame(client):
    body = ask(client, DELETE_ZH, CANDS)
    assert body["scorer"]["frame"] == "The first action to take is:"
    assert body["scorer"]["frame_source"] == "default"


def test_explicit_frame_is_used_and_reaches_the_scorer(client):
    auto = ask(client, QA_CONTEXT, ["yes", "no"])
    forced = ask(client, QA_CONTEXT, ["yes", "no"], frame="The first action to take is:")
    assert forced["scorer"]["frame"] == "The first action to take is:"
    assert forced["scorer"]["frame_source"] == "explicit"
    # Different frames must give different scores, or the frame never reached the model.
    a = [c["log_likelihood"] for c in auto["candidates"]]
    b = [c["log_likelihood"] for c in forced["candidates"]]
    assert a != b

    custom = ask(client, DELETE_ZH, CANDS, history=["backup"], frame="The safest action is:")
    assert custom["scorer"]["frame"] == "The safest action is:"
    assert custom["scorer"]["frame_source"] == "explicit"


def test_blank_explicit_frame_is_rejected_not_defaulted(client):
    resp = client.post("/v1/semantic_ask", json={"context": "x", "candidates": ["a", "b"], "frame": "   "})
    assert resp.status_code == 400
    assert resp.json()["detail"]["error"]["code"] == "invalid_semantic_request"


def test_select_ask_frame_priority():
    from gen_zero.service.semantic_scorer import (
        ASK_FRAME, NEXT_FRAME, QA_FRAME, select_ask_frame)

    assert select_ask_frame("do it", ["a", "b"]) == (ASK_FRAME, "default")
    assert select_ask_frame("do it", ["a", "b"], ["a"]) == (NEXT_FRAME, "history")
    assert select_ask_frame("Question: x?", ["a", "b"], ["a"]) == (QA_FRAME, "qa")
    assert select_ask_frame("x", [" Yes", "NO "]) == (QA_FRAME, "qa")
    assert select_ask_frame("x", ["yes", "no", "maybe"]) == (ASK_FRAME, "default")
    assert select_ask_frame("Question: x?", ["a", "b"], explicit=" F: ") == ("F:", "explicit")
    with pytest.raises(ValueError):
        select_ask_frame("x", ["a", "b"], explicit="")


def test_semantic_route_ranks_by_intent(client):
    tools = ["delete_file", "search_web", {"name": "send_email", "description": "Send an email"}]
    resp = client.post("/v1/semantic_route", json={
        "intent": "帮我给老板发一封邮件说我明天请假", "tools": tools, "top_k": 1})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["selected"] == ["send_email"]
    assert len(body["ranked"]) == 3
    probs = [r["probability"] for r in body["ranked"]]
    assert probs == sorted(probs, reverse=True)


OPAQUE_TOOLS = [
    {"name": "tool_7", "description": "Send an email to someone"},
    {"name": "tool_3", "description": "Search the web for information"},
    {"name": "tool_1", "description": "Delete a file from disk"},
]


@pytest.mark.parametrize("intent,expected", [
    ("帮我给老板发一封邮件说我明天请假", "tool_7"),
    ("Email my landlord that the rent will be late", "tool_7"),
    ("查一下明天东京的天气", "tool_3"),
    ("remove the old log file", "tool_1"),
])
def test_route_reads_descriptions_when_names_say_nothing(client, intent, expected):
    # The names carry no meaning; only the descriptions can route these.
    for perm in itertools.permutations(OPAQUE_TOOLS):
        resp = client.post("/v1/semantic_route", json={"intent": intent, "tools": list(perm), "top_k": 1})
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["selected"] == [expected], ([t["name"] for t in perm], body["ranked"])
        assert body["scorer"]["scored_text"] == "name_and_description"


def test_route_continuation_includes_the_description():
    from gen_zero.service.app import ToolSpec, tool_continuation

    assert tool_continuation(ToolSpec(name="tool_7", description="Send an  email\nto someone")) == \
        " tool 7: Send an email to someone"
    assert tool_continuation("send_email") == " send email"


def test_invalid_requests_are_rejected_not_guessed(client):
    one = client.post("/v1/semantic_ask", json={"context": "x", "candidates": ["only"]})
    assert one.status_code == 400
    dup = client.post("/v1/semantic_ask", json={"context": "x", "candidates": ["a", "a"]})
    assert dup.status_code == 400


def test_auth_is_required(client):
    raw = TestClient(client.app)
    resp = raw.post("/v1/semantic_ask", json={"context": DELETE_ZH, "candidates": CANDS})
    assert resp.status_code == 401


def test_batched_kv_cache_scores_equal_full_sequence_forward():
    import torch

    from gen_zero.service.semantic_scorer import ASK_FRAME, candidate_text, compose_prompt, get_semantic_scorer

    scorer = get_semantic_scorer()
    rt = scorer.runtime
    head = scorer.lm_head.float()
    prompt = compose_prompt(BACKUP_EN, ASK_FRAME)
    names = ["wait", "search_web", "send_email_to_the_boss"]  # 1, 2 and 5 tokens: exercises padding
    batched, _, _ = scorer._continuation_log_likelihoods(prompt, [candidate_text(n) for n in names])
    for name, got in zip(names, batched):
        p = rt.token_ids(prompt)
        c = rt.tokenizer.encode(candidate_text(name), add_special_tokens=False).ids
        ids = torch.tensor([p + c])
        with torch.inference_mode():
            h, _ = rt.model(ids, torch.ones_like(ids))
            lp = torch.log_softmax(h[0].float() @ head.T, -1)
        full = sum(lp[len(p) - 1 + i, t].item() for i, t in enumerate(c))
        assert abs(got - full) < 1e-3, (name, got, full)
