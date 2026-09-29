#!/usr/bin/env python3
"""Prototype for Spec 16, Plan 1 (Sequential Layer Streaming Forward).

Simulates a slice of a large decoder-only transformer whose per-layer weights
live on disk (never all resident on GPU at once) and measures REAL CUDA peak
memory for three execution strategies on the same synthetic weights:

  A. full_preload   - every layer's weights resident on GPU simultaneously
                       (what device_map="auto" without offload does).
  B. naive_offload  - one GPU weight buffer, synchronous disk-read -> H2D
                       copy -> compute -> free per layer. This mirrors
                       accelerate's AlignDevicesHook.pre_forward/post_forward
                       (hooks.py:359,402): no prefetch, no second buffer, so
                       copy and compute serialize.
  C. double_buffer  - two GPU weight buffers (ring of 2) + a background
                       reader thread + a dedicated torch.cuda.Stream for H2D
                       copy, so layer k+1's disk read and H2D copy overlap
                       with layer k's compute on the default (compute) stream.

Weights are synthetic (random fp16), because we do not have Qwen3.8-Flash-Next
weights; the point is the memory/bandwidth *mechanism*, not model quality.
Sizes are chosen so the synthetic model's total weight footprint clears the
10 GB budget under test (proving A would violate it) while B and C stay
under it by construction (only 1-2 layers resident at a time).

Every phase does a real forward pass (QKV projection, scaled dot-product
self-attention, MLP with a real matmul chain) and records intermediate
hidden states at 1/3-depth, 2/3-depth and last layer per spec section 2.2.2.
Output is asserted finite (no NaN/Inf) as a correctness sanity check.
"""
from __future__ import annotations

import argparse
import json
import shutil
import signal
import sys
import faulthandler
import threading
import time
from pathlib import Path

import torch

HIDDEN = 5120
INTERMEDIATE = 10240
DTYPE = torch.float16
ELEM_BYTES = 2
N_HEADS = 40
HEAD_DIM = HIDDEN // N_HEADS

SHAPES = [
    ("wqkv", (HIDDEN, 3 * HIDDEN)),
    ("wo", (HIDDEN, HIDDEN)),
    ("w1", (HIDDEN, INTERMEDIATE)),
    ("w2", (INTERMEDIATE, HIDDEN)),
]


def layer_nbytes() -> int:
    return sum(r * c for _, (r, c) in SHAPES) * ELEM_BYTES


def layer_offsets():
    off = 0
    offs = []
    for name, (r, c) in SHAPES:
        nb = r * c * ELEM_BYTES
        offs.append((name, off, nb, (r, c)))
        off += nb
    return offs


def gen_weights(weight_dir: Path, num_layers: int, seed: int = 0) -> int:
    weight_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    for li in range(num_layers):
        path = weight_dir / f"layer_{li:03d}.bin"
        expect = layer_nbytes()
        if path.exists() and path.stat().st_size == expect:
            total += expect
            continue
        g = torch.Generator().manual_seed(seed * 100000 + li)
        init_std = 0.02  # standard small-init scale; with rmsnorm keeps the
        # residual stream well inside fp16 range across dozens of layers
        with open(path, "wb") as f:
            for _, (r, c) in SHAPES:
                t = (torch.randn((r, c), generator=g, dtype=torch.float32) * init_std).to(DTYPE)
                f.write(t.numpy().tobytes())
        total += path.stat().st_size
    return total


def views_from_uint8(buf: torch.Tensor):
    """Bit-cast a contiguous uint8 GPU/CPU buffer into the 4 named weight tensors."""
    out = {}
    for name, off, nb, shape in layer_offsets():
        sub = buf[off:off + nb]
        out[name] = sub.view(DTYPE).view(shape)
    return out


def read_layer_into(path: Path, pinned_u8: torch.Tensor) -> None:
    with open(path, "rb", buffering=0) as f:
        n = f.readinto(memoryview(pinned_u8.numpy()))
    if n != pinned_u8.numel():
        raise IOError(f"short read on {path}: {n} != {pinned_u8.numel()}")


def rmsnorm(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    var = x.float().pow(2).mean(dim=-1, keepdim=True)
    return (x.float() * torch.rsqrt(var + eps)).to(x.dtype)


def block_forward(x: torch.Tensor, w: dict) -> torch.Tensor:
    """One real pre-norm transformer block: self-attention + MLP, both residual.

    Random N(0,1) weights with no normalization make the residual stream
    overflow fp16 within a couple of layers (variance grows by ~HIDDEN per
    matmul). Real transformers always carry a norm layer for exactly this
    reason, so pre-norm here isn't a shortcut, it's the missing component.
    """
    b, s, h = x.shape
    hn = rmsnorm(x)
    qkv = hn @ w["wqkv"]  # [B,S,3H]
    q, k, v = qkv.split(HIDDEN, dim=-1)
    q = q.view(b, s, N_HEADS, HEAD_DIM).transpose(1, 2)
    k = k.view(b, s, N_HEADS, HEAD_DIM).transpose(1, 2)
    v = v.view(b, s, N_HEADS, HEAD_DIM).transpose(1, 2)
    attn = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
    attn = attn.transpose(1, 2).reshape(b, s, h)
    x = x + attn @ w["wo"]
    hn2 = rmsnorm(x)
    mlp = torch.relu(hn2 @ w["w1"]) @ w["w2"]
    x = x + mlp
    return x


def cuda_peak_reset(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)


def cuda_peak_bytes(device: torch.device) -> int:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        return torch.cuda.max_memory_allocated(device)
    return 0


def extract_hidden(x: torch.Tensor, layer_idx: int, num_layers: int, sink: dict):
    third = num_layers // 3
    if layer_idx == third or layer_idx == 2 * third or layer_idx == num_layers - 1:
        pooled = x.mean(dim=1)[0].detach().float().cpu()
        sink[f"layer_{layer_idx}"] = float(pooled.norm().item())


def phase_full_preload(weight_dir: Path, num_layers: int, x0: torch.Tensor, device: torch.device):
    cuda_peak_reset(device)
    t0 = time.perf_counter()
    layers = []
    for li in range(num_layers):
        path = weight_dir / f"layer_{li:03d}.bin"
        pinned = torch.empty(layer_nbytes(), dtype=torch.uint8, pin_memory=(device.type == "cuda"))
        read_layer_into(path, pinned)
        gpu_u8 = pinned.to(device, non_blocking=False)
        layers.append(views_from_uint8(gpu_u8))
    t_load = time.perf_counter() - t0

    x = x0.clone()
    hidden_probe = {}
    t1 = time.perf_counter()
    for li, w in enumerate(layers):
        x = block_forward(x, w)
        extract_hidden(x, li, num_layers, hidden_probe)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t_compute = time.perf_counter() - t1
    peak = cuda_peak_bytes(device)
    del layers
    return {
        "phase": "full_preload",
        "peak_bytes": peak,
        "load_s": t_load,
        "compute_s": t_compute,
        "total_s": t_load + t_compute,
        "hidden_probe": hidden_probe,
        "output_finite": bool(torch.isfinite(x).all().item()),
        "_final_x": x,
    }


def phase_naive_offload(weight_dir: Path, num_layers: int, x0: torch.Tensor, device: torch.device):
    """One buffer: disk read -> H2D copy -> compute -> free, strictly serial.

    Mirrors accelerate.hooks.AlignDevicesHook: pre_forward loads the module's
    weights onto execution_device just before the call, post_forward moves
    them back to 'meta' right after. No second buffer, no async stream.
    """
    cuda_peak_reset(device)
    pinned = torch.empty(layer_nbytes(), dtype=torch.uint8, pin_memory=(device.type == "cuda"))
    x = x0.clone()
    hidden_probe = {}
    t_disk = 0.0
    t_h2d = 0.0
    t_compute = 0.0
    t0 = time.perf_counter()
    for li in range(num_layers):
        path = weight_dir / f"layer_{li:03d}.bin"
        ta = time.perf_counter()
        read_layer_into(path, pinned)
        tb = time.perf_counter()
        gpu_u8 = pinned.to(device, non_blocking=False)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        tc = time.perf_counter()
        w = views_from_uint8(gpu_u8)
        x = block_forward(x, w)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        td = time.perf_counter()
        extract_hidden(x, li, num_layers, hidden_probe)
        del gpu_u8, w
        t_disk += tb - ta
        t_h2d += tc - tb
        t_compute += td - tc
    total = time.perf_counter() - t0
    peak = cuda_peak_bytes(device)
    return {
        "phase": "naive_offload",
        "peak_bytes": peak,
        "disk_s": t_disk,
        "h2d_s": t_h2d,
        "compute_s": t_compute,
        "total_s": total,
        "hidden_probe": hidden_probe,
        "output_finite": bool(torch.isfinite(x).all().item()),
        "_final_x": x,
    }


def phase_double_buffer(weight_dir: Path, num_layers: int, x0: torch.Tensor, device: torch.device):
    """Ring of 2 GPU buffers fed by a 3-stage pipeline across 3 threads:

      reader thread : disk file -> pinned[slot]                (host I/O)
      copier thread : pinned[slot] -async-> gpu_buf[slot]       (H2D on copy_stream)
      main thread   : compute on gpu_buf[slot]                  (default/compute stream)

    Each stage only proceeds once the *downstream-confirmed-complete* signal
    for the buffer it wants to reuse has fired:
      - reader waits `buffer_free[slot]` (set only after the copier has
        host-synced that the H2D copy out of pinned[slot] finished, so the
        disk read can never overwrite bytes still being copied).
      - copier waits `data_ready[slot]` from the reader, and (from a
        buffer's second use onward) makes copy_stream wait on
        `compute_done_evt[slot]` so it never overwrites a GPU buffer whose
        weights a previous compute op is still reading.
      - main waits `copy_ready[slot]`, which the copier only sets after
        `torch.cuda.Event.synchronize()` on its own thread confirms the H2D
        copy is actually done (not just enqueued).

    This is the fix for a real bug found during local testing: an earlier
    queue-based handoff let the reader start overwriting a pinned staging
    buffer before the H2D copy reading it had actually completed, silently
    corrupting weights (caught because this phase's hidden-state probes
    numerically diverged from full_preload/naive_offload on identical
    inputs and weights).
    """
    cuda_peak_reset(device)
    nbytes = layer_nbytes()
    pinned = [
        torch.empty(nbytes, dtype=torch.uint8, pin_memory=(device.type == "cuda"))
        for _ in range(2)
    ]
    gpu_buf = [
        torch.empty(nbytes, dtype=torch.uint8, device=device)
        for _ in range(2)
    ]

    buffer_free = [threading.Event() for _ in range(2)]
    gpu_free = [threading.Event() for _ in range(2)]
    data_ready = [threading.Event() for _ in range(2)]
    copy_ready = [threading.Event() for _ in range(2)]
    for e in buffer_free:
        e.set()  # both pinned staging buffers start out free
    for e in gpu_free:
        e.set()  # both GPU-resident slots start out free (no prior compute)
    stop_flag = threading.Event()

    def reader():
        try:
            for li in range(num_layers):
                if stop_flag.is_set():
                    return
                slot = li % 2
                buffer_free[slot].wait()
                buffer_free[slot].clear()
                path = weight_dir / f"layer_{li:03d}.bin"
                read_layer_into(path, pinned[slot])
                data_ready[slot].set()
        except BaseException:
            import traceback
            traceback.print_exc()
            stop_flag.set()
            raise

    copy_stream = torch.cuda.Stream(device) if device.type == "cuda" else None
    compute_stream = torch.cuda.current_stream(device) if device.type == "cuda" else None
    compute_done_evt = [torch.cuda.Event() if device.type == "cuda" else None for _ in range(2)]
    slot_used_once = [False, False]

    def copier():
        try:
            for li in range(num_layers):
                if stop_flag.is_set():
                    return
                slot = li % 2
                data_ready[slot].wait()
                data_ready[slot].clear()
                gpu_free[slot].wait()  # don't overwrite gpu_buf[slot] until main is done reading it
                gpu_free[slot].clear()
                if device.type == "cuda":
                    if slot_used_once[slot]:
                        copy_stream.wait_event(compute_done_evt[slot])
                    slot_used_once[slot] = True
                    with torch.cuda.stream(copy_stream):
                        gpu_buf[slot].copy_(pinned[slot], non_blocking=True)
                        evt = torch.cuda.Event()
                        evt.record(copy_stream)
                    evt.synchronize()  # blocks this thread only, until H2D truly finished
                else:
                    gpu_buf[slot].copy_(pinned[slot])
                buffer_free[slot].set()
                copy_ready[slot].set()
        except BaseException:
            import traceback
            traceback.print_exc()
            stop_flag.set()
            raise

    reader_thread = threading.Thread(target=reader, daemon=True)
    copier_thread = threading.Thread(target=copier, daemon=True)
    x = x0.clone()
    hidden_probe = {}
    t0 = time.perf_counter()
    reader_thread.start()
    copier_thread.start()

    for li in range(num_layers):
        slot = li % 2
        if not copy_ready[slot].wait(timeout=60):
            raise TimeoutError(
                f"double_buffer: copy_ready[{slot}] never set for layer {li}; "
                f"reader_alive={reader_thread.is_alive()} copier_alive={copier_thread.is_alive()}"
            )
        copy_ready[slot].clear()

        w = views_from_uint8(gpu_buf[slot])
        x = block_forward(x, w)
        if device.type == "cuda":
            compute_done_evt[slot].record(compute_stream)
        gpu_free[slot].set()  # safe for copier to overwrite this slot now
        extract_hidden(x, li, num_layers, hidden_probe)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    stop_flag.set()
    for e in (*buffer_free, *gpu_free, *data_ready, *copy_ready):
        e.set()  # unstick any thread parked on the final iteration's wait
    reader_thread.join(timeout=5)
    copier_thread.join(timeout=5)
    total = time.perf_counter() - t0
    peak = cuda_peak_bytes(device)
    return {
        "phase": "double_buffer",
        "peak_bytes": peak,
        "total_s": total,
        "hidden_probe": hidden_probe,
        "output_finite": bool(torch.isfinite(x).all().item()),
        "_final_x": x,
    }


def isolated_bandwidth_probe(weight_dir: Path, device: torch.device) -> dict:
    """Cold, non-overlapped measurement of disk-read and H2D-copy bandwidth
    for a single layer, run before the pipelined phases so the numbers are
    not contaminated by prior warmup/caching effects.
    """
    path = weight_dir / "layer_000.bin"
    nbytes = layer_nbytes()
    pinned = torch.empty(nbytes, dtype=torch.uint8, pin_memory=(device.type == "cuda"))
    t0 = time.perf_counter()
    read_layer_into(path, pinned)
    t_disk = time.perf_counter() - t0

    gpu = torch.empty(nbytes, dtype=torch.uint8, device=device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t1 = time.perf_counter()
    gpu.copy_(pinned, non_blocking=False)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t_h2d = time.perf_counter() - t1
    del gpu
    return {
        "layer_bytes": nbytes,
        "disk_read_s": t_disk,
        "disk_GBps": (nbytes / 1e9) / t_disk if t_disk > 0 else float("inf"),
        "h2d_copy_s": t_h2d,
        "h2d_GBps": (nbytes / 1e9) / t_h2d if t_h2d > 0 else float("inf"),
    }


def main():
    if hasattr(signal, "SIGUSR1"):
        faulthandler.register(signal.SIGUSR1)
    global HIDDEN, INTERMEDIATE, N_HEADS, HEAD_DIM, SHAPES
    ap = argparse.ArgumentParser()
    ap.add_argument("--weight-dir", type=Path, default=Path("./_streaming_proto_weights"))
    ap.add_argument("--num-layers", type=int, default=32)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--budget-gb", type=float, default=10.0)
    ap.add_argument("--keep-weights", action="store_true")
    ap.add_argument("--skip-full-preload", action="store_true",
                     help="Skip phase A if you already know it will breach VRAM/RAM.")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--hidden", type=int, default=HIDDEN)
    ap.add_argument("--intermediate", type=int, default=INTERMEDIATE)
    ap.add_argument("--heads", type=int, default=N_HEADS)
    args = ap.parse_args()

    HIDDEN, INTERMEDIATE, N_HEADS = args.hidden, args.intermediate, args.heads
    HEAD_DIM = HIDDEN // N_HEADS
    SHAPES = [
        ("wqkv", (HIDDEN, 3 * HIDDEN)),
        ("wo", (HIDDEN, HIDDEN)),
        ("w1", (HIDDEN, INTERMEDIATE)),
        ("w2", (INTERMEDIATE, HIDDEN)),
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[stage] gen_weights start device={device}", flush=True)
    total_weight_bytes = gen_weights(args.weight_dir, args.num_layers)
    print(f"[stage] gen_weights done total={total_weight_bytes/1e9:.3f}GB", flush=True)

    x0 = torch.randn(args.batch, args.seq, HIDDEN, dtype=DTYPE, device=device) * 0.02

    report = {
        "device": str(device),
        "cuda_device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch_version": torch.__version__,
        "num_layers": args.num_layers,
        "hidden": HIDDEN,
        "intermediate": INTERMEDIATE,
        "batch": args.batch,
        "seq": args.seq,
        "dtype": str(DTYPE),
        "per_layer_bytes": layer_nbytes(),
        "total_weight_bytes": total_weight_bytes,
        "total_weight_GB": total_weight_bytes / 1e9,
        "budget_bytes": int(args.budget_gb * 1e9),
    }

    print("[stage] bandwidth_probe", flush=True)
    report["bandwidth_probe"] = isolated_bandwidth_probe(args.weight_dir, device)

    if not args.skip_full_preload:
        print("[stage] full_preload", flush=True)
        report["full_preload"] = phase_full_preload(args.weight_dir, args.num_layers, x0, device)
    print("[stage] naive_offload", flush=True)
    report["naive_offload"] = phase_naive_offload(args.weight_dir, args.num_layers, x0, device)
    print("[stage] double_buffer", flush=True)
    report["double_buffer"] = phase_double_buffer(args.weight_dir, args.num_layers, x0, device)
    print("[stage] all phases done", flush=True)

    budget = report["budget_bytes"]
    db_peak = report["double_buffer"]["peak_bytes"]
    report["assertion_double_buffer_under_budget"] = db_peak < budget

    x_full = report.get("full_preload", {}).pop("_final_x", None)
    x_naive = report["naive_offload"].pop("_final_x")
    x_double = report["double_buffer"].pop("_final_x")
    if x_full is not None:
        report["exact_match_naive_vs_full"] = torch.equal(x_naive, x_full)
        report["exact_match_double_vs_full"] = torch.equal(x_double, x_full)
    else:
        report["exact_match_naive_vs_full"] = None
        report["exact_match_double_vs_full"] = None
    report["exact_match_double_vs_naive"] = torch.equal(x_double, x_naive)

    report["assertion_pass"] = bool(
        report["assertion_double_buffer_under_budget"]
        and report["exact_match_double_vs_naive"]
        and (report["exact_match_double_vs_full"] is not False)
    )

    if not args.keep_weights:
        shutil.rmtree(args.weight_dir, ignore_errors=True)
        report["weight_dir_cleaned_up"] = True
    else:
        report["weight_dir_cleaned_up"] = False

    print(json.dumps(report, indent=2))
    if args.out:
        args.out.write_text(json.dumps(report, indent=2))

    if not report["assertion_pass"]:
        print(f"FAIL: double_buffer peak {db_peak/1e9:.3f} GB >= budget {budget/1e9:.1f} GB", file=sys.stderr)
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
