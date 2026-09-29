"""R5 regressions using real loopback HTTP and the production compiler entrypoints."""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import pytest

from gen_zero.gateway.arbiter_bridge import CloudGPUArbiterBridge
from gen_zero.gate.constraint_compiler import ConstraintLinearProjectionCompiler


@pytest.fixture
def endpoint():
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            assert request['candidates'] == ['A', 'HOLD']
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps(self.server.payload).encode())

        def log_message(self, *args):
            pass

    server = HTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, kwargs={'poll_interval': 0.01})
    thread.start()
    try:
        yield server, f'http://127.0.0.1:{server.server_port}/arbitrate'
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


def arbitrate(bridge, mode='sync'):
    return bridge.arbitrate('state', ['A', 'HOLD'], 'A', 0.2, {'A': 0.2}, mode=mode)


@pytest.mark.parametrize('payload', [
    {}, [], None, {'best_action': 'A'}, {'probs': {'A': 1.0}},
    {'best_action': 'OTHER', 'probs': {'A': 1.0}},
    *[{'best_action': 'A', 'probs': value} for value in
      ({}, [], {'HOLD': 1.0}, {'A': None}, {'A': True}, {'A': '1'},
       {'A': float('nan')}, {'A': float('inf')}, {'A': -0.1},
       {'A': 1.1}, {'A': 0.0}, {'A': 1.0, 'OTHER': 0.1})],
    {'best_action': 'A', 'probs': {'A': 1.0}, 'status': 'failed'},
    {'best_action': 'A', 'probs': {'A': 1.0}, 'backend_reachable': False},
    {'best_action': 'A', 'probs': {'A': 1.0}, 'confidence': float('nan')},
])
def test_malformed_remote_fails_closed(endpoint, payload, caplog):
    server, url = endpoint
    server.payload = payload
    bridge = CloudGPUArbiterBridge(remote_endpoint=url)
    verdict = arbitrate(bridge)
    assert verdict.backend_reachable is False
    assert verdict.metadata['status'] == 'failed'
    assert verdict.action is None
    assert verdict.confidence == 0.0
    assert all(value == 0.0 for value in verdict.probs.values())
    assert 'protocol invalid' in caplog.text


def test_valid_remote_and_async_failure_accounting(endpoint):
    server, url = endpoint
    bridge = CloudGPUArbiterBridge(remote_endpoint=url)
    server.payload = {'best_action': 'HOLD', 'probs': {'A': 0.1, 'HOLD': 0.9}}
    verdict = arbitrate(bridge)
    assert verdict.backend_reachable is True
    assert verdict.action == 'HOLD'
    assert verdict.confidence == 0.9
    assert verdict.metadata['status'] == 'ok'
    server.payload = {}
    arbitrate(bridge, mode='async')
    for thread in bridge._async_tasks:
        thread.join(timeout=5)
        assert not thread.is_alive()
    assert bridge._arbitration_count == 0
    assert bridge._unreachable_count == 1


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), -float('inf')])
@pytest.mark.parametrize('compiled', [False, True])
@pytest.mark.parametrize('metrics', [None, {'x': -1.0}])
def test_nonfinite_latent_rejected_at_all_entries(bad, compiled, metrics):
    compiler = ConstraintLinearProjectionCompiler(latent_dim=2)
    report = compiler.compile_rules(['FORBID A IF x > 0'])
    rule = report.rules[0]
    if not compiled:
        compiler = ConstraintLinearProjectionCompiler(latent_dim=2)
    # Includes invalid data beyond latent_dim: truncation must not hide it.
    for z in ([bad, 0.0], [0.0, 0.0, bad]):
        for candidates in ({'A': 1.0, 'HOLD': 0.0}, {}):
            with pytest.raises(ValueError, match='^z_latent contains NaN or Inf$'):
                compiler.solve_safest_action(candidates, z_latent=z, current_metrics=metrics)
        with pytest.raises(ValueError, match='^z_latent contains NaN or Inf$'):
            compiler.project_latent_propositions(z)
        with pytest.raises(ValueError, match='^z_latent contains NaN or Inf$'):
            compiler.evaluate_condition(rule, z_latent=z, current_metrics=metrics)


def test_finite_latent_still_enforces_forbid():
    compiler = ConstraintLinearProjectionCompiler(latent_dim=2)
    compiler.compile_rules(['FORBID A IF x > 0'])
    verdict = compiler.solve_safest_action(
        {'A': 1.0, 'HOLD': 0.0}, z_latent=np.array([1.0, 0.0]),
        schema_fingerprint=compiler.schema_fingerprint,
    )
    assert verdict.is_safe
    assert verdict.selected_action == 'HOLD'


def test_missing_schema_fingerprint_fails_closed():
    """X-C01: a well-formed z_latent with no (or a stale) schema_fingerprint must
    never be silently reinterpreted against whatever coordinate map the compiler
    currently holds."""
    from gen_zero.gate.constraint_compiler import SchemaMismatchError

    compiler = ConstraintLinearProjectionCompiler(latent_dim=2)
    compiler.compile_rules(['FORBID A IF x > 0'])
    z = np.array([1.0, 0.0])

    with pytest.raises(SchemaMismatchError):
        compiler.solve_safest_action({'A': 1.0, 'HOLD': 0.0}, z_latent=z)
    with pytest.raises(SchemaMismatchError):
        compiler.project_latent_propositions(z)
    with pytest.raises(SchemaMismatchError):
        compiler.evaluate_condition(compiler._rules[0], z_latent=z)

    # A stale fingerprint (schema shifted after adding an alphabetically-earlier
    # variable) must fail closed exactly like a missing one, not silently
    # reinterpret z under the new coordinate map (X-C01 counterexample).
    stale_fingerprint = compiler.schema_fingerprint
    compiler.compile_rules(['FORBID A IF x > 0', 'FORBID B IF a > 0'])
    with pytest.raises(SchemaMismatchError):
        compiler.solve_safest_action({'A': 1.0, 'HOLD': 0.0}, z_latent=z, schema_fingerprint=stale_fingerprint)
