"""Adversarial evaluator checks; all inputs and predictors here are test fixtures."""
from types import SimpleNamespace

import numpy as np
import pytest

from gen_zero.evaluate.dual_calibrator import DualCalibrator, IsotonicCalibrator
from gen_zero.evaluate.layer_scaling_benchmark import compute_linear_separability
from gen_zero.evaluate.log_filter_benchmark import (
    LogBenchmarkItem, TwoStageLogFilterEvaluator, create_multilingual_log_corpus,
)
from gen_zero.evaluate.web_agent_benchmark import WebAgentBenchmarkSuite






@pytest.mark.parametrize("probs,labels", [([], []), ([float("nan")], [1]), ([0.4], []), ([1.5], [1])])
def test_invalid_calibration_cannot_look_perfect(probs, labels):
    with pytest.raises(ValueError):
        DualCalibrator.compute_10bin_ece(np.array(probs), np.array(labels))


def test_equal_confidence_calibration_is_order_independent():
    a, b = IsotonicCalibrator(), IsotonicCalibrator()
    a.fit(np.array([0.5, 0.5]), np.array([0, 1]))
    b.fit(np.array([0.5, 0.5]), np.array([1, 0]))
    assert a.predict(np.array([0.5])) == pytest.approx([0.5])
    assert b.predict(np.array([0.5])) == pytest.approx([0.5])


def test_linear_probe_cannot_score_memorized_one_hot_ids_as_generalization():
    # Training can memorize arbitrary labels on identity vectors, but held-out
    # identity dimensions carry no learned label information.
    labels = np.array([0, 1] * 20)
    score = compute_linear_separability(np.eye(40), labels, num_classes=2)
    assert score == pytest.approx(0.5)


def test_generated_logs_remain_synthetic_with_checkpoint_client():
    client = SimpleNamespace(weights_loaded_from_checkpoint=True)
    evaluator = TwoStageLogFilterEvaluator(client=client)
    corpus = create_multilingual_log_corpus()
    assert all(item.is_synthetic for item in corpus)
    result = evaluator.evaluate(corpus, "deadlock", coarse_regex="NEVER_MATCH_FIXTURE")
    assert result.is_synthetic is True
    assert result.precision == 0.0
    assert result.metrics_valid is False


def test_web_candidate_count_is_not_executed_steps(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = WebAgentBenchmarkSuite.evaluate(repeat_trials=1)
    assert result["is_synthetic"] is True
    assert result["decide_and_fill"]["average_steps_per_task"] is None
    assert result["average_synthesized_actions_per_task"] > 0


@pytest.mark.parametrize("count", [0, -1, True])
def test_empty_web_evaluation_is_rejected(count):
    with pytest.raises(ValueError):
        WebAgentBenchmarkSuite.evaluate(repeat_trials=count)
