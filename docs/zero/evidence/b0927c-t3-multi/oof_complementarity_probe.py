"""Train-only OOF probe of 70B/72B complementarity and disagreement. No test labels are read."""
import sys, json
sys.path.insert(0, '/ebs/pj/gen-zero/benchmarks/suites')
import numpy as np
from scipy.special import softmax
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.cross_decomposition import CCA
import grand_challenge_data as gd
import evaluate_dual_70b_72b_ensemble as base

def ent(p):
    return -(p*np.log(np.clip(p,1e-12,1))).sum(1)/np.log(p.shape[1])

out = {}
for ti, task in enumerate(gd.TASKS):
    _, data = base.load(task)
    y = data[0]['train_label'].astype(int); K = len(data[0]['cands'])
    Z = [base.project(d, base.SEED + 100*ti + i)[0] for i, d in enumerate(data)]
    folds = list(StratifiedKFold(5, shuffle=True, random_state=base.SEED).split(Z[0], y))
    P = [np.zeros((len(y), K)) for _ in range(2)]
    for tr, ho in folds:
        for b in range(2):
            raw = base.fit_scores(Z[b][tr], y[tr], Z[b][ho], K)
            tr_raw = base.fit_scores(Z[b][tr], y[tr], Z[b][tr], K)
            P[b][ho] = softmax(raw / max(float(np.std(tr_raw)), 1e-8), axis=1)
    q, l = P; m = (q + l) / 2
    cq, cl, cm = q.argmax(1) == y, l.argmax(1) == y, m.argmax(1) == y
    # entropy decomposition: total = aleatoric (mean entropy) + epistemic (JS / mutual information)
    H_tot = ent(m); H_ale = (ent(q) + ent(l)) / 2; MI = H_tot - H_ale
    err = ~cm
    def auroc(s):
        return float(roc_auc_score(err, s)) if 0 < err.sum() < len(err) else float('nan')
    order = np.argsort(MI)  # keep the least-disagreeing rows
    cov = {}
    for c in (1.0, 0.9, 0.8, 0.7):
        k = int(round(c * len(y))); cov[c] = float(100 * cm[order[:k]].mean())
    order_e = np.argsort(H_tot)
    cov_e = {c: float(100 * cm[order_e[:int(round(c*len(y)))]].mean()) for c in (0.9, 0.8, 0.7)}
    # shared subspace: top canonical correlations on a 64-D slice (n >> 64 keeps CCA well posed)
    zq = (Z[0][:, :64] - Z[0][:, :64].mean(0)) / Z[0][:, :64].std(0)
    zl = (Z[1][:, :64] - Z[1][:, :64].mean(0)) / Z[1][:, :64].std(0)
    n = len(y); half = n // 2
    cca = CCA(n_components=8, max_iter=2000).fit(zq[:half], zl[:half])
    a, b_ = cca.transform(zq[half:], zl[half:])
    rho = [float(np.corrcoef(a[:, j], b_[:, j])[0, 1]) for j in range(8)]
    out[task] = dict(n=int(n), K=K,
        acc_q=float(100*cq.mean()), acc_l=float(100*cl.mean()), acc_mean=float(100*cm.mean()),
        oracle_either=float(100*(cq | cl).mean()), both_wrong=float(100*(~cq & ~cl).mean()),
        disagree_argmax=float(100*(q.argmax(1) != l.argmax(1)).mean()),
        auroc_MI_err=auroc(MI), auroc_Htot_err=auroc(H_tot), auroc_Hale_err=auroc(H_ale),
        acc_at_cov_MI=cov, acc_at_cov_Htot=cov_e,
        heldout_cca_rho_top8=rho)
    r = out[task]
    print(f"{task:22s} n={n:5d} K={K:2d} q={r['acc_q']:.1f} l={r['acc_l']:.1f} avg={r['acc_mean']:.1f} "
          f"oracle={r['oracle_either']:.1f} bothwrong={r['both_wrong']:.1f} dis={r['disagree_argmax']:.1f} "
          f"AUROC MI={r['auroc_MI_err']:.3f} Htot={r['auroc_Htot_err']:.3f} "
          f"acc@80%MI={cov[0.8]:.1f} acc@80%H={cov_e[0.8]:.1f} rho1={rho[0]:.2f} rho8={rho[7]:.2f}", flush=True)
json.dump(out, open('/tmp/b0927c/probe_out.json', 'w'), indent=1)
print("WROTE /tmp/b0927c/probe_out.json")
