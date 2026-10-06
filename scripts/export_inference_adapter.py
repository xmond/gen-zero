#!/usr/bin/env python3
"""Strip the ten training-only tensors from a trained tri-teacher LoRA adapter.

Input: a `gen_zero.tri_teacher_lora.v1` safetensors checkpoint as written by
the offline training pipeline in `gen-zero-research` (`tri_teacher_lora_model.py`), e.g.
`adapter_paws_tri.safetensors`
(118 tensors: 96 attention LoRA -- 24 layers x {q,v}_proj x {A,B} -- plus 12
projection-head tensors -- 3 teachers x {0,1}.{weight,bias} -- plus the 10
training-only tensors this script removes: `teacher_mean_<slot>` and
`teacher_std_<slot>` for each of the 3 teachers, and
`task_head.{0,1}.{weight,bias}`).

Two outputs:

1. A pure-inference export: byte-identical values for the 96 LoRA + 12
   projection tensors, `__metadata__` unchanged, the 10 training-only tensors
   gone. Size barely moves -- those 10 tensors are ~0.3 MB of a ~122 MB file
   dominated by the three projection heads' [16384,896]/[8192,896]/[8192,896]
   weight matrices. The point is not a size win, it's that
   crates/gen-zero-model/src/tri_teacher.rs now loads this file too (the 10
   tensors are optional-but-validated-if-present, not required).

2. A lightweight demo/test fixture at `examples/weights/tri_teacher_demo.safetensors`.
   At the real adapter's teacher_dims the projection heads alone are >100 MB,
   so a file under ~20 MB cannot be a slice of the trained weights -- it has
   to use smaller dimensions instead. This is therefore a small, deterministic
   *synthetic* adapter in the same `gen_zero.tri_teacher_lora.v1` format: it
   exercises the real loader and the real decider math, but its numbers are
   not learned and must never be cited as the trained adapter's. A top-level
   `provenance` metadata key says so explicitly.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file, save_file

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SRC = REPO_ROOT / "benchmarks/results/tri_teacher_lora_evidence/adapter_paws_tri.safetensors"
DEFAULT_INFERENCE_OUT = (
    REPO_ROOT / "benchmarks/results/tri_teacher_lora_evidence/adapter_paws_tri_inference.safetensors"
)
DEFAULT_DEMO_OUT = REPO_ROOT / "examples/weights/tri_teacher_demo.safetensors"

ADAPTER_FORMAT = "gen_zero.tri_teacher_lora.v1"
TEACHER_SLOTS = ("405b", "q72b", "llama70b")

TRAINING_TENSOR_NAMES = [f"teacher_mean_{s}" for s in TEACHER_SLOTS] + [
    f"teacher_std_{s}" for s in TEACHER_SLOTS
] + ["task_head.0.weight", "task_head.0.bias", "task_head.1.weight", "task_head.1.bias"]


def _read_metadata(path: Path) -> dict[str, str]:
    """Read only the `__metadata__` map of a safetensors file's header."""
    with open(path, "rb") as f:
        n = int.from_bytes(f.read(8), "little")
        header = json.loads(f.read(n))
    return header.get("__metadata__", {})


def export_inference_adapter(src: Path, dst: Path) -> tuple[list[str], list[str]]:
    """Write `dst` as `src` with the 10 training-only tensors removed.

    Returns (kept_tensor_names, dropped_tensor_names). Raises if `src` is not
    a full training checkpoint (missing any of the 10), so this is never run
    twice by accident on an already-stripped file.
    """
    tensors = load_file(str(src))
    meta = _read_metadata(src)
    if meta.get("format") != ADAPTER_FORMAT:
        raise SystemExit(f"{src}: format {meta.get('format')!r}, expected {ADAPTER_FORMAT!r}")

    missing = [n for n in TRAINING_TENSOR_NAMES if n not in tensors]
    if missing:
        raise SystemExit(
            f"{src}: missing training tensors {missing} -- this does not look like a full "
            "training checkpoint (already an inference export?)"
        )

    dropped = [n for n in TRAINING_TENSOR_NAMES if n in tensors]
    kept = {k: v for k, v in tensors.items() if k not in TRAINING_TENSOR_NAMES}

    lora = [k for k in kept if k.startswith("backbone.")]
    proj = [k for k in kept if k.startswith("proj_")]
    if len(lora) != 96:
        raise SystemExit(f"{src}: expected 96 attention LoRA tensors (48 pairs), got {len(lora)}")
    if len(proj) != 12:
        raise SystemExit(f"{src}: expected 12 projection-head tensors, got {len(proj)}")

    dst.parent.mkdir(parents=True, exist_ok=True)
    save_file(kept, str(dst), metadata=meta)
    return sorted(kept), sorted(dropped)


def _demo_config() -> dict:
    return {
        "model_id_or_path": "synthetic-demo-not-trained",
        "rank": 8,
        "alpha": 16.0,
        "student_dim": 896,
        "teacher_dims": {"405b": 32, "q72b": 24, "llama70b": 24},
        "weights": {"405b": 0.5, "q72b": 0.3, "llama70b": 0.2},
        "lam": 0.0,
        "num_classes": 0,
        "tau": 0.07,
        "target_modules": ["q_proj", "v_proj"],
        "variants": {},
    }


def build_demo_adapter(seed: int, num_layers: int = 24) -> tuple[dict, dict]:
    """A small, deterministic, synthetic `gen_zero.tri_teacher_lora.v1` adapter.

    Dimensions are chosen to stay well under ~10 MB while still exercising
    every shape rule the real loader checks (three distinct teacher widths,
    all 24 layers of Qwen2.5-0.5B, both LoRA targets). Values come from a seeded RNG,
    not training -- this is a demo/test fixture, never a benchmark claim.
    """
    rng = np.random.default_rng(seed)
    config = _demo_config()
    d = config["student_dim"]
    rank = config["rank"]
    kv = 128  # Qwen2.5-0.5B's real num_kv_heads * head_dim; not load-bearing here.

    def f32(*shape: int) -> np.ndarray:
        return rng.standard_normal(shape).astype(np.float32) * 0.02

    tensors: dict[str, np.ndarray] = {}
    for slot, t in zip(TEACHER_SLOTS, (config["teacher_dims"][s] for s in TEACHER_SLOTS)):
        tensors[f"proj_{slot}.0.weight"] = np.ones(d, dtype=np.float32)
        tensors[f"proj_{slot}.0.bias"] = np.zeros(d, dtype=np.float32)
        tensors[f"proj_{slot}.1.weight"] = f32(t, d)
        tensors[f"proj_{slot}.1.bias"] = f32(t)
        tensors[f"teacher_mean_{slot}"] = np.zeros(t, dtype=np.float32)
        tensors[f"teacher_std_{slot}"] = np.ones(t, dtype=np.float32)

    for layer in range(num_layers):
        for target, out in (("q_proj", d), ("v_proj", kv)):
            p = f"backbone.layers.{layer}.self_attn.{target}"
            tensors[f"{p}.lora_A"] = f32(rank, d)
            tensors[f"{p}.lora_B"] = f32(out, rank)

    meta = {
        "format": ADAPTER_FORMAT,
        "adapter_config": json.dumps(config),
        "provenance": (
            "synthetic demo fixture for examples/ and unit tests; NOT a trained adapter; "
            f"values are numpy.random.default_rng(seed={seed}) output, not learned weights"
        ),
    }
    return tensors, meta


def write_demo_adapter(dst: Path, seed: int) -> list[str]:
    tensors, meta = build_demo_adapter(seed)
    dst.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(dst), metadata=meta)
    return sorted(tensors)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC, help="trained adapter to strip")
    ap.add_argument("--out", type=Path, default=DEFAULT_INFERENCE_OUT, help="pure-inference export path")
    ap.add_argument("--demo-out", type=Path, default=DEFAULT_DEMO_OUT, help="lightweight demo fixture path")
    ap.add_argument("--skip-inference-export", action="store_true")
    ap.add_argument("--skip-demo", action="store_true")
    ap.add_argument("--demo-seed", type=int, default=20261002)
    args = ap.parse_args()

    if not args.skip_inference_export:
        kept, dropped = export_inference_adapter(args.src, args.out)
        src_bytes = args.src.stat().st_size
        out_bytes = args.out.stat().st_size
        print(f"[inference-export] {args.src} ({src_bytes / 1e6:.1f} MB) -> {args.out} ({out_bytes / 1e6:.1f} MB)")
        print(f"[inference-export] kept {len(kept)} tensors (96 LoRA + 12 projection)")
        print(f"[inference-export] dropped {len(dropped)} training-only tensors: {dropped}")

    if not args.skip_demo:
        names = write_demo_adapter(args.demo_out, args.demo_seed)
        demo_bytes = args.demo_out.stat().st_size
        print(f"[demo] wrote {args.demo_out} ({demo_bytes / 1e3:.1f} KB, {len(names)} tensors, seed={args.demo_seed})")


if __name__ == "__main__":
    main()
