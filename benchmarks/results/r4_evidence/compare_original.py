"""Run the HEAD compiler against the same audited synthetic scenarios as the fix.

The original module is executed verbatim from git; no solver behavior is replaced.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import types
from gen_zero.scripts import benchmark_issue_76_constraint_compiler as benchmark

path='python/gen_zero/gate/constraint_compiler.py'
source=subprocess.check_output(['git','show',f'HEAD:{path}'])
module=types.ModuleType('r4_original_compiler')
sys.modules[module.__name__]=module
exec(compile(source,path+'@HEAD','exec'), module.__dict__)
benchmark.ConstraintLinearProjectionCompiler=module.ConstraintLinearProjectionCompiler
report=benchmark.run_benchmark()
report['source']={'git_head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                  'path':path,'sha256':hashlib.sha256(source).hexdigest()}
out=Path(__file__).with_name('constraints_original_audited_report.json')
out.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report['solver_audit'].items() if k!='trials'},indent=2))
sys.exit(0 if report['solver_audit']['passed'] else 1)
