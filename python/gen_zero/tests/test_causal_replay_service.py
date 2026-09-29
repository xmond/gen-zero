"""Observed transition ingestion through the live HTTP service path."""
import os

import numpy as np
from fastapi.testclient import TestClient

from gen_zero.service.app import app, client


def test_observed_steps_are_sampleable(monkeypatch):
    monkeypatch.setenv("GENZERO_API_KEY", "replay-test-token")
    api = TestClient(app)
    headers = {"Authorization": "Bearer replay-test-token"}
    before = client.causal_replay_buffer.total_transitions
    dim = client.causal_replay_buffer.latent_dim
    for i in range(3):
        state = np.full(dim, float(i)).tolist()
        next_state = np.full(dim, float(i + 1)).tolist()
        response = api.post("/v1/replay/transitions", headers=headers, json={
            "state": state, "action": f"observed_{i}", "reward": float(i),
            "next_state": next_state, "done": False,
        })
        assert response.status_code == 200, response.text
    status = api.get("/v1/replay/status", headers=headers)
    assert status.status_code == 200, status.text
    assert status.json()["total_transitions"] == before + 3
    sample = client.causal_replay_buffer.sample(batch_size=3)
    assert sample["batch_size"] == 3
    assert sample["states"].shape == (3, dim)
    assert set(sample["actions"]) == {"observed_0", "observed_1", "observed_2"}


def test_invalid_step_fails_closed(monkeypatch):
    monkeypatch.setenv("GENZERO_API_KEY", "replay-test-token")
    api = TestClient(app)
    response = api.post("/v1/replay/transitions",
                        headers={"Authorization": "Bearer replay-test-token"},
                        json={"state": [1], "action": "a", "reward": 1,
                              "next_state": [2], "done": False})
    assert response.status_code == 422
