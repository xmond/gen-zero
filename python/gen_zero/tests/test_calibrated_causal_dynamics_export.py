"""Deployment contract tests; numeric geometry is not semantic-production evidence."""
from pathlib import Path
import time

import numpy as np
import pytest

from gen_zero.causal.continuous_causal_reasoning_expert import (
    ContinuousCausalReasoningExpert,
)
from gen_zero.causal.dynamics_calibrator import CausalDynamicsCalibrator


PRODUCTION = (Path(__file__).parents[3] / "benchmarks" / "results_v6" /
              "calibrated_causal_dynamics.npz")
ENCODER_ID = "qwen3.5-9b-zca64-v1"


def _numeric_export(path):
    rng = np.random.default_rng(412)
    n, dim = 96, 64
    x = rng.normal(size=(n, dim)).astype(np.float32)
    sign = np.where(x[:, :1] >= 0, 1.0, -1.0).astype(np.float32)
    positive = np.zeros((n, dim), dtype=np.float32)
    negative = np.zeros((n, dim), dtype=np.float32)
    positive[:, :1] = sign
    negative[:, 1:2] = sign
    # Orthogonal branch geometry is exactly realizable by a rank-one input map.
    # This is only a mathematical objective test and is never called production data.
    candidates = np.stack((positive, negative), axis=1).astype(np.float32)
    calibrator = CausalDynamicsCalibrator(dim, 1, dim, seed=9)
    history = calibrator.fit_candidate_geometry(
        x, candidates, np.zeros(n, dtype=np.int64), sample_ids=np.arange(n),
        source="numeric-test-only", split="train", encoder_id="numeric-zca64",
        epochs=40, lr=.01)
    assert history[-1] < history[0]
    calibrator.adapter.save(path)


def test_npz_loads_into_expert_and_step_meets_deployment_contract(tmp_path):
    artifact = tmp_path / "numeric-test-only.npz"
    _numeric_export(artifact)
    expert = ContinuousCausalReasoningExpert.from_file(
        artifact, encoder_id="numeric-zca64")
    cert = expert.dynamics.certify()
    assert cert["rho"] <= .55
    assert cert["sigma"] <= .95
    assert cert["working_set_bytes"] <= 5.5 * 1024

    x = np.linspace(-1, 1, 64, dtype=np.float32)
    h = np.zeros(64, dtype=np.float32)
    expert.bind_feature(x)
    # Long enough to leave CPU idle-frequency states before wall-clock sampling.
    for _ in range(100_000):
        h = expert.step_bound(h)
    samples = []
    for _ in range(7):
        start = time.perf_counter_ns()
        for _ in range(20_000):
            h = expert.step_bound(h)
        samples.append((time.perf_counter_ns() - start) / 20_000 / 1_000)
    median_us = float(np.median(samples))
    print(f"step latency samples_us={samples}, median_us={median_us:.3f}")
    assert median_us <= 15.0, f"median single-step latency {median_us:.3f} us"


def test_checked_in_production_export_when_present():
    if not PRODUCTION.exists():
        pytest.skip("no provenance-attested production calibration features in repository")
    expert = ContinuousCausalReasoningExpert.from_file(PRODUCTION, encoder_id=ENCODER_ID)
    assert expert.dim == 64
    assert expert.dynamics.certify()["working_set_bytes"] <= 5632
