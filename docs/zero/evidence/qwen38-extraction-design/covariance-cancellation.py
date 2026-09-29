"""Reproduce the C/N - mu mu^T cancellation in StreamingCovarianceAccumulator. Synthetic gaussian data only."""
import sys, numpy as np
sys.path.insert(0, "python")
from gen_zero.causal.universal_manifold_extractor import StreamingCovarianceAccumulator
rng = np.random.default_rng(0)
for offset in [0.0, 1e3, 1e5, 1e8]:
    x = rng.standard_normal((4096, 64)) + offset
    a = StreamingCovarianceAccumulator(64)
    for b in np.array_split(x, 8): a.update(b)
    xc = x - x.mean(0); exp = xc.T @ xc / len(x)
    shift = x[:8].mean(0); a2 = StreamingCovarianceAccumulator(64)
    for b in np.array_split(x, 8): a2.update(b - shift)
    print(f"offset={offset:g} direct_err={np.max(np.abs(a.covariance()-exp)):.3e} shifted_err={np.max(np.abs(a2.covariance()-exp)):.3e}")
print("diagnostic only: synthetic gaussian data, no Qwen inference")
