import hashlib
import numpy as np
import pytest

from gen_zero.causal.gepa_daemon import GepaEvolutionDaemon
from gen_zero.causal.gepa_evolution_loop import DecisionTrace, CounterfactualAdversarialSynthesizer


def trace(i):
    return DecisionTrace(str(i), 'numeric', np.array([1., 0., 0.]),
                         np.array([[0., 1., 0.], [1., 0., 0.]]),
                         0, np.array([.51, .49]), .02, 1)


def test_streaming_recalibration_and_audit(tmp_path):
    daemon = GepaEvolutionDaemon(np.eye(3)[:2], np.eye(3) * .7, np.eye(3), reorthogonalize_every=2)
    daemon.run([[trace(0)], [trace(1)], [trace(2)]])
    assert daemon.n_patched == 3 and daemon.n_recalibrated == 1
    assert daemon.orthogonality_error() < 1e-8
    report = daemon.export_audit_log(tmp_path / 'audit.json')
    assert report['contractive_certified'] and report['lyapunov_spectral_radius'] < 1
    digest = hashlib.sha256()
    for array in (daemon.basis, daemon.codebook, daemon.A_operator):
        digest.update(np.ascontiguousarray(array).tobytes())
    assert report['sha256'] == digest.hexdigest()
    assert (tmp_path / 'audit.json').exists()


def test_rejects_excessive_perturbation_and_unstable_initial_operator():
    with pytest.raises(ValueError, match='epsilon'):
        GepaEvolutionDaemon(np.eye(2), np.eye(2) * .5, np.eye(2),
                            synthesizer=CounterfactualAdversarialSynthesizer(epsilon=.06))
    with pytest.raises(ValueError, match='contractive'):
        GepaEvolutionDaemon(np.eye(2), np.eye(2), np.eye(2))
