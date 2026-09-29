import hashlib,json
from pathlib import Path
import numpy as np
from cross_model_manifold_alignment import load_features,verify_id_alignment
from generalized_cca_manifold_interference import GeneralizedCCAManifoldInterference
from gen_zero.causal.feature_space import load_source_space
from gen_zero.causal.manifold_anchor_distiller import ManifoldAnchorDistiller
from gen_zero.causal.nanocore_bridge import NanocoreAnchorBridge
out=Path('/tmp/b0928-t4-evidence')
paths=[Path('/ebs/data/extracted_features')/model/'boolq.npz' for model in ('llama70b','qwen72b')]
a,b=[load_features(p) for p in paths]
verify_id_alignment(a,str(paths[0]),b,str(paths[1]))
indices=np.random.default_rng(20260928).permutation(len(a['train_ids']))[:700]
fi,ei=indices[:500],indices[500:]
x,y=a['train_full'],b['train_full']
op=GeneralizedCCAManifoldInterference(max_rank=128,n_shared=128,standardize=True)
fit=op.fit([x[fi],y[fi]])
source_space=load_source_space(paths[0])
space=fit.save_transform(out/'real-gcca.npz',0,source_model=source_space['source_model'],layer=source_space['layer'],norm=source_space['norm'],source_data=x,fit_indices=fi,eval_indices=ei)
anchor=ManifoldAnchorDistiller(128,128).attach_gcca(out/'real-gcca.npz').fit(fit.transform(x[fi]))
anchor.save(out/'real-anchor-unbound.npz')
loaded=ManifoldAnchorDistiller.load(out/'real-anchor-unbound.npz')
expected=anchor.project(fit.transform(x[ei]))
actual=loaded.transform_source(x[ei],source_space)
errors=np.max(np.abs(expected-actual),axis=1)
np.savez(out/'real-paired-transform-evidence.npz',sample_ids=a['train_ids'][ei],expected=expected,reloaded=actual,per_sample_max_abs_error=errors,fit_indices=fi,eval_indices=ei)
assert np.all(errors<1e-12)
try:
 NanocoreAnchorBridge(out/'real-anchor-unbound.npz')
except ValueError as exc:
 rejection=str(exc)
else:
 raise AssertionError('unbound core accepted')
report={'source_paths':list(map(str,paths)),'n_fit':len(fi),'n_eval':len(ei),'paired_ids_verified':True,'gcca_width':fit.n_shared,'anchor_width':128,'per_sample_reload_error_max':float(errors.max()),'distortion':loaded.evaluate_distortion(fit.transform(x[ei])),'unbound_bridge_rejection':rejection,'live_core_executed':False,'claim':'transform contract verification only, not task accuracy or generalization'}
(out/'real-report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
