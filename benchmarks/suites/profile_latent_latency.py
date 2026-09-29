#!/usr/bin/env python3
"""Profile the latent-thinking symplectic core: FLOPs, params, bandwidth, latency.

One latent thinking step (docs/research/latent_thinking_zero_token_reasoning.md)
is a Stormer-Verlet update on h in R^dim split into canonical (q, p), each dim/2:

    1. p -= (dt/2) * grad_q V(q, context)        # half-step momentum kick
    2. q += dt * Minv * p                        # full coordinate drift
    3. p -= (dt/2) * grad_q V(q, context)        # second half-step kick
    4. kinetic-energy gate: break if mean kinetic energy < early_stop_threshold

V is a 3-layer MLP potential net (dim -> hidden -> hidden -> 1) evaluated on
cat(q, context); grad_q V is computed by autograd, i.e. each step costs TWO
gradient evaluations. This script instantiates that core, then measures per K
latent steps in {1, 2, 4, 8, ...}:

  - parameter count (from the module, plus the closed-form formula),
  - analytical FLOPs (forward GEMMs of the potential net; backward counted at
    ~2x forward) — no thop/fvcore dependency, assumptions reported in JSON,
  - wall-clock latency (perf_counter, CUDA-sync bracketed on GPU), mean/p50/p95
    plus per-step latency inside the K loop,
  - memory bandwidth derived from an analytical bytes-per-step model over the
    measured step latency, with a roofline ratio against the peak device
    bandwidth (read from CUDA device properties when available).

Every printed line and the JSON report go through `sanitize_output()`
(same redaction as run_remote_eval_v4.py): the PR checklist forbids private
filesystem paths in code AND in generated output.

Run `--self-test` for a CPU-only check (dim=64, K in {1, 2}) that also asserts
no absolute path leaks into the report.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    torch = None
    nn = None
    HAS_TORCH = False


# Path bootstrap — mirrors the other suites, no absolute paths anywhere.
SUITE_DIR = Path(__file__).resolve().parent
BENCH_DIR = SUITE_DIR.parent
REPO_ROOT = BENCH_DIR.parent
sys.path.insert(0, str(REPO_ROOT / "python"))

DEFAULT_DIM = 4096
DEFAULT_HIDDEN = 2048
DEFAULT_STEPS = "1,2,4,8"
DEFAULT_BATCH = 1
DEFAULT_WARMUP = 5
DEFAULT_ITERS = 20
DEFAULT_DT = 0.1
EARLY_STOP_THRESHOLD = 1e-3
CPU_ASSUMED_PEAK_GBPS = 51.2       # dual-channel DDR4-3200 assumption (stated)
GPU_FALLBACK_PEAK_GBPS = 1024.0    # used only if device properties lack clocks
FLOP_LINEARITY_TOL = 0.05          # self-test tolerance on K-linearity

_PRINT_LOG: List[str] = []


def sanitize_output(value: Any) -> Any:
    """Redact local paths, credential patterns and configured secret values."""
    if isinstance(value, dict):
        return {sanitize_output(str(k)): sanitize_output(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_output(v) for v in value]
    if not isinstance(value, str):
        return value
    for key, secret in os.environ.items():
        if secret and re.search(r"TOKEN|SECRET|PASSWORD|API_KEY|CREDENTIAL", key, re.I):
            value = value.replace(secret, "[REDACTED]")
    value = re.sub(r"(?<![\w:])/[A-Za-z_][^\s\"'<>]*", "[LOCAL_PATH]", value)
    value = re.sub(r"\b(?:hf_|sk-|ghp_|github_pat_)[A-Za-z0-9_-]+", "[REDACTED]", value)
    return re.sub(r"(?i)\b(bearer\s+|(?:token|password|api[_-]?key|secret)\s*[:=]\s*)[^\s,;]+",
                  r"\1[REDACTED]", value)


def safe_print(*values: Any, **kwargs: Any) -> None:
    line = " ".join(sanitize_output(str(v)) for v in values)
    _PRINT_LOG.append(line)
    print(line, **kwargs)


# ---------------------------------------------------------------------------
# Analytical models
# ---------------------------------------------------------------------------

def potential_param_count(dim: int, hidden: int) -> Dict[str, Any]:
    """Closed-form parameter count of the core (potential net + inverse mass)."""
    q = dim // 2
    layer1 = dim * hidden + hidden
    layer2 = hidden * hidden + hidden
    layer3 = hidden * 1 + 1
    inv_mass = q
    total = layer1 + layer2 + layer3 + inv_mass
    formula = "dim*hidden + hidden + hidden^2 + hidden + hidden + 1 + dim/2"
    return {
        "total": total,
        "formula": formula,
        "terms": {
            "layer1_w_b": layer1, "layer2_w_b": layer2,
            "layer3_w_b": layer3, "inv_mass": inv_mass,
        },
    }


def potential_forward_flops(batch: int, dim: int, hidden: int) -> int:
    """Forward FLOPs of one potential-net evaluation on a batch of states.

    Assumptions (reported verbatim in the JSON output):
      - a GEMM costs 2*B*M*N FLOPs (multiply + accumulate);
      - a bias add costs 1 FLOP per output element;
      - SiLU costs 4 FLOPs per element (sigmoid ~ exp + div, then a multiply);
      - cat / slicing / scaling are elementwise and counted at 1 FLOP/element.
    """
    l1 = 2 * batch * dim * hidden + batch * hidden + 4 * batch * hidden
    l2 = 2 * batch * hidden * hidden + batch * hidden + 4 * batch * hidden
    l3 = 2 * batch * hidden * 1 + batch * 1
    cat_in = batch * dim
    return int(l1 + l2 + l3 + cat_in)


def step_flops(batch: int, dim: int, hidden: int) -> Dict[str, int]:
    """FLOPs of one Stormer-Verlet step: two gradient evaluations plus drift."""
    fwd = potential_forward_flops(batch, dim, hidden)
    bwd = 2 * fwd  # backward is counted at ~2x forward (stated assumption)
    grad_eval = fwd + bwd
    q = dim // 2
    elementwise = 12 * batch * q   # 2 kicks (p update) + drift + kinetic gate
    return {
        "flops_per_forward": int(fwd),
        "flops_per_backward": int(bwd),
        "flops_per_gradient_eval": int(grad_eval),
        "flops_per_step": int(2 * grad_eval + elementwise),
    }


def estimate_bytes_per_step(batch: int, dim: int, hidden: int, elem_size: int) -> Dict[str, Any]:
    """Analytical bytes moved per latent step (weights + activations + grads).

    Per gradient evaluation we count, in elements (x element size):
      - weights: forward read + backward read + backward grad-write = 3x weight
        footprint (stated assumption);
      - activations: each layer's output counted at ~3x its size (write, read
        for the nonlinearity, read by the next GEMM), plus the cat input;
      - backward activation traffic approximated at 2x forward (included in
        the 3x weight factor and the 3x activation factor).
    The step runs two gradient evaluations plus elementwise q/p/grad traffic.
    """
    q = dim // 2
    weights = dim * hidden + hidden * hidden + hidden
    act_per_eval = batch * (dim + 3 * (2 * hidden) + hidden + 3 * hidden + 2 * hidden)
    bytes_per_eval = elem_size * (3 * weights + act_per_eval)
    elementwise = elem_size * batch * (4 * dim + 8 * q)  # q,p reads/writes, grad_q, kinetic
    total = int(2 * bytes_per_eval + elementwise)
    return {
        "estimated_bytes_per_step": total,
        "bytes_per_gradient_eval": int(bytes_per_eval),
        "elementwise_bytes": int(elementwise),
        "elem_size": elem_size,
        "assumptions": [
            "weights counted at 3x footprint (fwd read, bwd read, bwd grad write)",
            "layer activations counted at ~3x output size (write + 2 reads)",
            "backward traffic folded into the 3x factors (approx 2x forward)",
            "elementwise q/p/grad_q/kinetic traffic counted explicitly",
        ],
    }


def peak_bandwidth_gbps(device: str) -> Dict[str, Any]:
    """Peak device bandwidth for the roofline ratio (best effort, stated)."""
    if device == "cuda" and HAS_TORCH and torch.cuda.is_available():
        try:
            props = torch.cuda.get_device_properties(0)
            clock = getattr(props, "memory_clock_rate", None)      # kHz in torch
            bus = getattr(props, "bus_width", None)                # bits
            if clock and bus:
                gbps = float(clock) * 1e3 * (float(bus) / 8.0) * 2.0 / 1e9  # DDR x2
                return {
                    "peak_bandwidth_GBps": gbps,
                    "source": "cuda_device_properties",
                    "assumption": "memory_clock_rate read as kHz, DDR factor 2",
                    "device_name": torch.cuda.get_device_name(0),
                }
        except Exception as exc:  # pragma: no cover - depends on the driver
            safe_print(f"[WARN] device properties unavailable: {type(exc).__name__}")
        return {
            "peak_bandwidth_GBps": GPU_FALLBACK_PEAK_GBPS,
            "source": "stated_constant",
            "assumption": f"fallback assumed peak {GPU_FALLBACK_PEAK_GBPS} GB/s",
        }
    return {
        "peak_bandwidth_GBps": CPU_ASSUMED_PEAK_GBPS,
        "source": "stated_constant",
        "assumption": f"CPU dual-channel DDR4-3200 assumed ({CPU_ASSUMED_PEAK_GBPS} GB/s)",
    }


# ---------------------------------------------------------------------------
# The core under test
# ---------------------------------------------------------------------------

class LatentThinkingCore(nn.Module):
    """Stormer-Verlet latent thinking engine over a 3-layer MLP potential.

    Matches the spec in docs/research/latent_thinking_zero_token_reasoning.md:
    h in R^dim is split into q, p in R^(dim/2); V(q; context) is an MLP on
    cat(q, context); each step does two half-step kicks around one drift, with
    a kinetic-energy early-break gate. When `instrument` is set, every step is
    timed with perf_counter (CUDA-sync bracketed on GPU) into
    `self.last_step_times_ms`.
    """

    def __init__(self, dim: int = DEFAULT_DIM, hidden_dim: int = DEFAULT_HIDDEN,
                 dt: float = DEFAULT_DT, max_steps: int = 8,
                 early_stop_threshold: float = EARLY_STOP_THRESHOLD) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("dim must be even for canonical (q, p) phase space")
        self.dim = dim
        self.q_dim = dim // 2
        self.hidden_dim = hidden_dim
        self.dt = dt
        self.max_steps = max_steps
        self.early_stop_threshold = early_stop_threshold
        self.potential_net = nn.Sequential(
            nn.Linear(self.q_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.inv_mass = nn.Parameter(torch.ones(self.q_dim))
        self.instrument = False
        self.last_step_times_ms: List[float] = []

    def compute_potential(self, q: "torch.Tensor", context: "torch.Tensor") -> "torch.Tensor":
        x_in = torch.cat([q, context], dim=-1)
        return self.potential_net(x_in).sum()

    def forward(self, h: "torch.Tensor", context: "torch.Tensor",
                k: Optional[int] = None) -> Tuple["torch.Tensor", int]:
        """Run k latent steps (k <= max_steps) and return (h_star, steps_taken)."""
        steps = self.max_steps if k is None else k
        if steps > self.max_steps:
            raise ValueError(f"k={steps} exceeds max_steps={self.max_steps}")
        is_cuda = h.is_cuda
        q = h[..., : self.q_dim].clone()
        p = h[..., self.q_dim:].clone()
        ctx = context.detach()
        self.last_step_times_ms = []
        steps_taken = 0
        for _ in range(steps):
            steps_taken += 1
            if is_cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            # 1. half-step momentum kick: p -= (dt/2) * grad_q V(q, ctx)
            q_var = q.detach().requires_grad_(True)
            grad_q = torch.autograd.grad(self.compute_potential(q_var, ctx), q_var)[0]
            p = p - (0.5 * self.dt) * grad_q

            # 2. full coordinate drift: q += dt * Minv * p
            q = q + self.dt * (p * self.inv_mass)

            # 3. second half-step kick
            q_var = q.detach().requires_grad_(True)
            grad_q = torch.autograd.grad(self.compute_potential(q_var, ctx), q_var)[0]
            p = p - (0.5 * self.dt) * grad_q

            # 4. kinetic-energy early-break gate
            kinetic = 0.5 * torch.sum(p * (p * self.inv_mass), dim=-1).mean()

            if is_cuda:
                torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            if self.instrument:
                self.last_step_times_ms.append(elapsed_ms)
            if float(kinetic.detach()) < self.early_stop_threshold:
                break

        return torch.cat([q, p], dim=-1), steps_taken


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------

def _stats(samples_ms: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(samples_ms, dtype=np.float64)
    return {
        "mean_ms": float(arr.mean()),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "min_ms": float(arr.min()),
        "max_ms": float(arr.max()),
        "iters": int(arr.size),
    }


def measure_k(core: "LatentThinkingCore", h: "torch.Tensor", ctx: "torch.Tensor",
              k: int, warmup: int, iters: int, device: str,
              elem_size: int, peak_gbps: float, batch: int, dim: int,
              hidden: int) -> Dict[str, Any]:
    """Measure one K: totals (sync at ends only) then per-step (synced steps)."""
    is_cuda = device == "cuda" and h.is_cuda

    # Pass A: total wall clock, synchronize only at the boundaries so the K
    # steps pipeline exactly as they would in production.
    for _ in range(warmup):
        core(h, ctx, k=k)
    totals: List[float] = []
    for _ in range(iters):
        if is_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _, taken = core(h, ctx, k=k)
        if is_cuda:
            torch.cuda.synchronize()
        totals.append((time.perf_counter() - t0) * 1000.0)

    # Pass B: per-step latency, synchronized at every step boundary. The sync
    # overhead is part of this pass and is reported alongside the clean totals.
    core.instrument = True
    per_step_all: List[List[float]] = []
    steps_taken: List[int] = []
    for _ in range(iters):
        _, taken = core(h, ctx, k=k)
        per_step_all.append(list(core.last_step_times_ms))
        steps_taken.append(taken)
    core.instrument = False
    flat_steps = [t for rep in per_step_all for t in rep]

    fl = step_flops(batch, dim, hidden)
    by = estimate_bytes_per_step(batch, dim, hidden, elem_size)
    mean_total = float(np.mean(totals))
    mean_step = float(np.mean(flat_steps)) if flat_steps else float("nan")
    effective = by["estimated_bytes_per_step"] / (mean_step * 1e-3) / 1e9 if mean_step > 0 else 0.0
    return {
        "k": int(k),
        "steps_taken_mean": float(np.mean(steps_taken)),
        "params": core_param_count(core),
        "flops_per_forward": fl["flops_per_forward"],
        "flops_per_backward": fl["flops_per_backward"],
        "flops_per_step": fl["flops_per_step"],
        "flops_total_per_k": fl["flops_per_step"] * int(k),
        "latency_total": _stats(totals),
        "per_step_latency": _stats(flat_steps),
        "per_step_latencies_last_iter": [round(t, 6) for t in per_step_all[-1]],
        "estimated_bytes_per_step": by["estimated_bytes_per_step"],
        "bytes_model": by,
        "effective_bandwidth_GBps": effective,
        "peak_bandwidth_GBps": peak_gbps,
        "roofline_ratio": effective / peak_gbps if peak_gbps > 0 else 0.0,
        "latency_total_derived_per_step_ms": mean_total / max(1, int(k)),
    }


def core_param_count(core: "LatentThinkingCore") -> int:
    return int(sum(p.numel() for p in core.parameters()))


def resolve_dtype(name: str) -> Any:
    table = {"float32": torch.float32, "float16": torch.float16,
             "bfloat16": torch.bfloat16}
    if name not in table:
        raise ValueError(f"unsupported dtype '{name}' (choose from {sorted(table)})")
    return table[name]


def run_profile(args: argparse.Namespace) -> Dict[str, Any]:
    """Run the full profiling matrix (or the analytic-only fallback)."""
    steps = [int(s) for s in str(args.steps).split(",") if s.strip()]
    device = args.device
    if device == "cuda" and not (HAS_TORCH and torch.cuda.is_available()):
        device = "cpu"
        safe_print("[WARN] CUDA unavailable, falling back to CPU.")
    if not HAS_TORCH:
        return analytic_only_report(args, steps,
                                    error="torch is not installed; latency and "
                                          "bandwidth rows are analytic-only")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dtype = resolve_dtype(args.dtype)
    elem_size = torch.tensor([], dtype=dtype).element_size()
    core = LatentThinkingCore(dim=args.dims, hidden_dim=args.hidden_dim,
                              dt=DEFAULT_DT, max_steps=max(steps)).to(device=device, dtype=dtype)
    core.eval()
    batch, dim, hidden = args.batch_size, args.dims, args.hidden_dim
    h = torch.randn(batch, dim, device=device, dtype=dtype)
    # Context anchors the problem: q-sized, so cat(q, ctx) is exactly dim wide,
    # matching the doc's Linear(q_dim*2, hidden) head (4096 -> hidden -> hidden -> 1).
    ctx = torch.randn(batch, dim // 2, device=device, dtype=dtype)

    pk = peak_bandwidth_gbps(device)
    rows = []
    for k in steps:
        row = measure_k(core, h, ctx, k, args.warmup, args.iters, device,
                        elem_size, pk["peak_bandwidth_GBps"], batch, dim, hidden)
        rows.append(row)
        safe_print(
            f"[K={k:>2}] params={row['params']:,} "
            f"flops/step={row['flops_per_step']:,} "
            f"total={row['latency_total']['mean_ms']:.3f}ms "
            f"(p95 {row['latency_total']['p95_ms']:.3f}) "
            f"step={row['per_step_latency']['mean_ms']:.3f}ms "
            f"bw={row['effective_bandwidth_GBps']:.1f}GB/s "
            f"roofline={row['roofline_ratio']:.3f}"
        )
    report = base_report(args, device)
    report.update({
        "rows": rows,
        "peak_bandwidth": pk,
        "core_params_measured": core_param_count(core),
        "core_params_formula": potential_param_count(dim, hidden),
    })
    report["flops_assumptions"] = [
        "GEMM counted at 2*B*M*N FLOPs; bias at 1 FLOP/element",
        "SiLU counted at 4 FLOPs/element (sigmoid ~ exp + div, then multiply)",
        "backward pass counted at 2x forward FLOPs (stated approximation)",
        "one latent step = TWO gradient evaluations (half-kick, drift, half-kick)",
        "elementwise q/p/grad_q/kinetic ops counted at 12 FLOPs per q element",
        "early-break gate may terminate before K steps; FLOPs are the no-break bound",
        "per-step timings are taken with a CUDA sync at every step boundary and "
        "therefore include sync overhead absent from the clean K-total pass",
    ]
    report["bytes_assumptions"] = rows[0]["bytes_model"]["assumptions"] if rows else []
    return report


def base_report(args: argparse.Namespace, device: str) -> Dict[str, Any]:
    return {
        "suite": "profile_latent_latency",
        "device": device,
        "torch_version": str(torch.__version__) if HAS_TORCH else None,
        "dtype": args.dtype,
        "dims": args.dims,
        "hidden_dim": args.hidden_dim,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "warmup": args.warmup,
        "iters": args.iters,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "error": None,
    }


def analytic_only_report(args: argparse.Namespace, steps: Sequence[int],
                         error: str) -> Dict[str, Any]:
    """Fallback when torch is missing: parameter/FLOPs rows only, no timing."""
    dim, hidden, batch = args.dims, args.hidden_dim, args.batch_size
    rows = []
    for k in steps:
        fl = step_flops(batch, dim, hidden)
        rows.append({
            "k": int(k),
            "params": potential_param_count(dim, hidden)["total"],
            "flops_per_forward": fl["flops_per_forward"],
            "flops_per_step": fl["flops_per_step"],
            "flops_total_per_k": fl["flops_per_step"] * int(k),
            "latency_total": None,
            "per_step_latency": None,
            "estimated_bytes_per_step": estimate_bytes_per_step(
                batch, dim, hidden, 4)["estimated_bytes_per_step"],
            "effective_bandwidth_GBps": None,
            "roofline_ratio": None,
        })
    report = base_report(args, "cpu")
    report.update({"rows": rows, "error": error,
                   "core_params_formula": potential_param_count(dim, hidden)})
    return report


def write_report(report: Dict[str, Any], results_dir: Path, device: str) -> Path:
    name = "latent_latency_profile.json" if device == "cpu" \
        else f"latent_latency_profile_{device}.json"
    path = results_dir / name
    results_dir.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(sanitize_output(report), indent=2) + "\n"
    path.write_text(payload, encoding="utf-8")
    safe_print(f"[WROTE] {path.name} ({len(payload):,} bytes)")
    return path


def print_table(report: Dict[str, Any]) -> None:
    safe_print(f"{'K':>3} {'params':>12} {'flops/step':>16} {'total ms':>10} "
               f"{'p50 ms':>10} {'p95 ms':>10} {'step ms':>10} {'GB/s':>9} {'roofline':>9}")
    for row in report.get("rows", []):
        lat = row.get("latency_total") or {}
        step = row.get("per_step_latency") or {}
        safe_print(
            f"{row['k']:>3} {row.get('params', 0):>12,} "
            f"{row.get('flops_per_step', 0):>16,} "
            f"{lat.get('mean_ms', float('nan')):>10.3f} "
            f"{lat.get('p50_ms', float('nan')):>10.3f} "
            f"{lat.get('p95_ms', float('nan')):>10.3f} "
            f"{step.get('mean_ms', float('nan')):>10.3f} "
            f"{row.get('effective_bandwidth_GBps') or 0.0:>9.1f} "
            f"{row.get('roofline_ratio') or 0.0:>9.3f}"
        )


# ---------------------------------------------------------------------------
# Self test
# ---------------------------------------------------------------------------

_ABS_PATH_RE = re.compile(r"(?<![\w:])/(?:home|Users|root|ebs|mnt|media|data|tmp)/")


def run_self_test() -> int:
    """CPU-only check: report parses, FLOPs linear in K, no path leakage."""
    _PRINT_LOG.clear()
    args = argparse.Namespace(
        dims=64, hidden_dim=32, steps="1,2", batch_size=2, warmup=1, iters=3,
        dtype="float32", device="cpu", seed=7,
    )
    report = run_profile(args)
    failures: List[str] = []

    # (a) output parses
    try:
        payload = json.dumps(sanitize_output(report))
        json.loads(payload)
        safe_print("[SELF-TEST] (a) report parses as JSON: OK")
    except (TypeError, ValueError) as exc:
        failures.append(f"report does not parse: {exc}")

    # (b) FLOPs scale linearly with K within 5%
    rows = {r["k"]: r for r in report.get("rows", [])}
    if 1 in rows and 2 in rows:
        ratio = rows[2]["flops_total_per_k"] / rows[1]["flops_total_per_k"]
        if abs(ratio - 2.0) <= FLOP_LINEARITY_TOL * 2.0:
            safe_print(f"[SELF-TEST] (b) FLOPs linearity ratio K2/K1={ratio:.4f}: OK")
        else:
            failures.append(f"FLOPs linearity ratio {ratio:.4f} outside 5% of 2.0")
    else:
        failures.append(f"missing rows for K=1 and K=2 (got {sorted(rows)})")

    # (c) no private path leaks into the report or any printed line
    serialized = json.dumps(sanitize_output(report)) + "\n".join(_PRINT_LOG)
    home = str(Path.home())
    if home and home in serialized:
        failures.append("report contains str(Path.home())")
    if _ABS_PATH_RE.search(serialized):
        failures.append("report contains an absolute filesystem path")
    if "[LOCAL_PATH]" in serialized:
        # sanitized placeholder is expected if any path slipped through; the
        # checks above already guarantee the raw path is gone.
        safe_print("[SELF-TEST] (c) note: sanitized [LOCAL_PATH] placeholder present")
    if not any(f.startswith("report contains") for f in failures):
        safe_print("[SELF-TEST] (c) no private/absolute path leakage: OK")

    # measured steps_taken must match the requested K (gate must not fire at test scale)
    for r in report.get("rows", []):
        if r.get("steps_taken_mean") is not None and abs(r["steps_taken_mean"] - r["k"]) > 1e-9:
            failures.append(f"K={r['k']}: early-break fired (steps_taken={r['steps_taken_mean']})")

    if failures:
        for f in failures:
            safe_print(f"[SELF-TEST] FAIL: {f}")
        return 1
    safe_print("[SELF-TEST] all checks passed")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dims", type=int, default=DEFAULT_DIM,
                        help=f"latent state dimension (default {DEFAULT_DIM})")
    parser.add_argument("--hidden-dim", type=int, default=DEFAULT_HIDDEN,
                        help=f"potential net hidden width (default {DEFAULT_HIDDEN})")
    parser.add_argument("--steps", type=str, default=DEFAULT_STEPS,
                        help=f"comma list of K values (default \"{DEFAULT_STEPS}\")")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH,
                        help=f"batch of latent states (default {DEFAULT_BATCH})")
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP,
                        help=f"warmup iterations per K (default {DEFAULT_WARMUP})")
    parser.add_argument("--iters", type=int, default=DEFAULT_ITERS,
                        help=f"measured iterations per K (default {DEFAULT_ITERS})")
    parser.add_argument("--dtype", type=str, default="float32",
                        choices=["float32", "float16", "bfloat16"],
                        help="compute dtype (default float32)")
    parser.add_argument("--device", type=str,
                        default="cuda" if HAS_TORCH and torch.cuda.is_available() else "cpu",
                        help="torch device (default: cuda when available, else cpu)")
    parser.add_argument("--results-dir", type=str, default=None,
                        help="output directory override (default: benchmarks/results)")
    parser.add_argument("--seed", type=int, default=0, help="deterministic seed")
    parser.add_argument("--self-test", action="store_true",
                        help="run the CPU-only self test (dim=64, K in {1,2})")
    args = parser.parse_args(argv)
    env_dir = os.environ.get("GEN_ZERO_RESULTS_DIR")
    results_dir = args.results_dir or env_dir or str(REPO_ROOT / "benchmarks" / "results")
    args.results_dir = Path(results_dir)
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        return run_self_test()
    report = run_profile(args)
    write_report(report, args.results_dir, report["device"])
    print_table(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
