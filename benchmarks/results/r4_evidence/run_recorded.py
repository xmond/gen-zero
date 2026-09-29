"""Run a validation command and retain its unmodified process return code."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

label, *command = sys.argv[1:]
directory = Path(__file__).resolve().parent
start = time.time()
with (directory / f'{label}.log').open('w') as stream:
    process = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
record = {'command': command, 'cwd': os.getcwd(), 'exit_code': process.returncode,
          'elapsed_s': time.time() - start, 'log': str(directory / f'{label}.log'),
          'environment': {k: os.environ.get(k) for k in ['PYTHONPATH', 'OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS']}}
(directory / f'{label}.json').write_text(json.dumps(record, indent=2)+'\n')
print(json.dumps(record), flush=True)
sys.exit(process.returncode)
