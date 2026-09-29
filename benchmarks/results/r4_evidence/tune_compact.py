import json,random,time
from pathlib import Path
from ortools.sat.python import cp_model
rows=[]
for presolve, probing, symmetry in [(False,0,0),(True,0,0),(True,2,2)]:
    trials=[]
    rng=random.Random(42)
    for i in range(1000):
        n=rng.randrange(1,12)
        coefficients=rng.sample(range(30),n)
        start=time.perf_counter()
        m=cp_model.CpModel(); p=m.Proto()
        for _ in range(n): p.variables.add().domain.extend((0,1))
        p.constraints.add().exactly_one.literals.extend(range(n))
        p.objective.vars.extend(range(n));p.objective.coeffs.extend(-c for c in coefficients)
        s=cp_model.CpSolver();s.parameters.num_search_workers=1
        s.parameters.cp_model_presolve=presolve;s.parameters.cp_model_probing_level=probing
        s.parameters.symmetry_level=symmetry;s.parameters.linearization_level=0
        s.parameters.max_time_in_seconds=max(1e-6,.002-(time.perf_counter()-start))
        status=s.Solve(m); elapsed=(time.perf_counter()-start)*1000
        trials.append({'wall_ms':elapsed,'status':s.StatusName(status)})
    values=sorted(t['wall_ms'] for t in trials)
    r={'presolve':presolve,'probing':probing,'symmetry':symmetry,'misses':sum(v>2 for v in values),
       'p50_ms':values[500],'p99_ms':values[990],'max_ms':values[-1],'trials':trials}
    print(json.dumps({k:v for k,v in r.items() if k!='trials'}),flush=True);rows.append(r)
Path(__file__).with_suffix('.json').write_text(json.dumps(rows,indent=2)+'\n')
