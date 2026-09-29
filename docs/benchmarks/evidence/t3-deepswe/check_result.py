"""Re-evaluate the recorded trial; exit 1 means the requested repair did not pass."""
import hashlib
import json
from pathlib import Path
import sys

root = Path(__file__).parent / 'raw'
job = (root / 'job-name').read_text().strip()
trials = list((root / 'jobs' / job).glob('*/result.json'))
if len(trials) != 1:
    raise RuntimeError(f'Expected exactly one recorded trial, found {len(trials)}')
trial = trials[0].parent
result = json.loads(trials[0].read_text())
reward = json.loads((trial / 'verifier/reward.json').read_text())
patch = trial / 'artifacts/model.patch'
execs = json.loads((root / 'docker-execs.json').read_text())
verifier = [v for v in execs.values() if v['command'].startswith('bash -c (/tests/test.sh)')]
agent_ids = {v['container_id'] for v in execs.values() if '__verifier__' not in v['container_name']}
separate = len(verifier) == 1 and verifier[0]['container_id'] not in agent_ids
agent_exit = int((trial / 'agent/adapter.exit').read_text())
accepted = (agent_exit == 0 and result.get('exception_info') is None
            and reward.get('reward') == 1 and separate and patch.stat().st_size > 0)
report = {
    'accepted': accepted, 'pier_exit': int((root / 'pier-run.exit').read_text()),
    'adapter_exit': agent_exit, 'verifier_exec': verifier,
    'independent_verifier': separate, 'reward': reward,
    'patch_bytes': patch.stat().st_size,
    'patch_sha256': hashlib.sha256(patch.read_bytes()).hexdigest(),
    'exception_type': (result.get('exception_info') or {}).get('exception_type'),
}
print(json.dumps(report, indent=2))
sys.exit(0 if accepted else 1)
