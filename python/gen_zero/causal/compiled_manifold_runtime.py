"""Online CPU runtime for `causal_codebook.bin`: mmap load, zero-copy views,
one closed-form fixed-point matvec plus a nearest-codebook-vertex readout.

Contract (matches the task spec, and doc 15 section 4.4's "no text, no task
names, no regex in the online source" rule):
  * `load()` mmaps the file read-only and validates magic/version/sha256
    before trusting a single byte of the payload. Corruption is a hard
    `ValueError`; there is no silent fallback.
  * `infer(task_id, x_projected, c_projected)` takes vectors ALREADY projected
    into the task's compact (r, r_cf) space -- this module contains no PCA
    basis application, no ZCA, no tokenization and no per-task literal names.
    `task_id` is an opaque string key into the codebook's own section index,
    exactly as doc 15 section 2 specifies ("task differences show up only in
    the codebook's task_id -> section index").
  * The dynamics readout is the closed-form fixed point
        h* = (I - A)^-1 (B x + W_c c)
    which is mathematically identical to iterating `h_{t+1} = A h_t + B x + W_c c`
    to convergence (proved by `A` being certified contractive at load time; see
    `CounterfactualDriftDynamics.fixed_point` in `counterfactual_drift_dynamics.py`,
    which already computes it the same way). Precomputing the inverse at compile
    time turns 64 Python-loop iterations into one matvec -- this is a precompute,
    not an approximation.
  * Prediction is nearest codebook vertex to h* (no Langevin relaxation, no
    macro-race expert): this is the "readout_only" configuration the existing
    evaluation harness already reports separately from the full pipeline
    (`benchmarks/results_track_c/cpu_dynamics_clean_eval.json`), and it is the
    only configuration that can plausibly hit a microsecond-scale budget.
"""
from __future__ import annotations

import hashlib
import json
import mmap
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from .knowledge_compiler import FORMAT_VERSION, HEADER, MAGIC

__all__ = [
    "CodebookCorruptionError",
    "TaskRuntimeSection",
    "InferenceResult",
    "CompiledManifoldRuntime",
    "CalibratedMoEGateway",
    "RoutedInferenceResult",
]


class CodebookCorruptionError(ValueError):
    """Raised when the mmapped file fails its header, hash, or contractivity check."""


def _view(buf: mmap.mmap, base: int, meta: dict) -> np.ndarray:
    dtype = {"f32": np.float32, "i8": np.int8}[meta["dtype"]]
    shape = tuple(meta["shape"])
    offset = base + meta["offset_in_arrays"]
    count = int(np.prod(shape)) if shape else 1
    arr = np.frombuffer(buf, dtype=dtype, count=count, offset=offset)
    return arr.reshape(shape)


@dataclass(frozen=True)
class TaskRuntimeSection:
    task_id: str
    k: int
    paired: bool
    dim: int
    n_components: int
    cf_components: int
    rho_a: float
    A: np.ndarray
    inv_i_minus_a: np.ndarray
    B: np.ndarray
    W_c: Optional[np.ndarray]
    codebook: np.ndarray
    x_mean: np.ndarray
    x_basis_q: np.ndarray
    x_basis_scale: np.ndarray
    c_mean: Optional[np.ndarray]
    c_basis_q: Optional[np.ndarray]
    c_basis_scale: Optional[np.ndarray]

    def working_set_bytes(self) -> int:
        """Bytes touched by one `infer()` call in the compact space: A-derived
        matrix, B, W_c, codebook, and small scratch vectors. Matches the
        formula in `CounterfactualDriftDynamics.working_set_bytes()` plus the
        codebook (needed here because the readout is nearest-vertex, not the
        macro-race expert)."""
        d, p, pc, k = self.dim, self.n_components, self.cf_components, self.k
        dynamics = 4 * (d * d + d * p + d * pc + 3 * d + p + pc)
        codebook = 4 * (k * d)
        return int(dynamics + codebook)

    def x_basis(self) -> np.ndarray:
        return self.x_basis_q.astype(np.float32) * self.x_basis_scale[None, :]

    def c_basis(self) -> Optional[np.ndarray]:
        if self.c_basis_q is None:
            return None
        return self.c_basis_q.astype(np.float32) * self.c_basis_scale[None, :]


@dataclass(frozen=True)
class InferenceResult:
    task_id: str
    scores: np.ndarray
    prediction: int
    fixed_point: np.ndarray


class CompiledManifoldRuntime:
    """Construct via `CompiledManifoldRuntime.load(path)`."""

    def __init__(self, mmap_obj: mmap.mmap, sections: Dict[str, TaskRuntimeSection],
                 payload_sha256: str, path: str) -> None:
        self._mmap = mmap_obj
        self._sections = sections
        self.payload_sha256 = payload_sha256
        self.path = path
        self._scratch: Dict[str, Dict[str, np.ndarray]] = {
            task_id: {
                "drive": np.zeros(sec.dim, dtype=np.float32),
                "h_star": np.zeros(sec.dim, dtype=np.float32),
                "diff": np.zeros((sec.k, sec.dim), dtype=np.float32),
                "d2": np.zeros(sec.k, dtype=np.float32),
            }
            for task_id, sec in sections.items()
        }

    @property
    def task_ids(self):
        return sorted(self._sections)

    def section(self, task_id: str) -> TaskRuntimeSection:
        try:
            return self._sections[task_id]
        except KeyError:
            raise KeyError(f"unknown task_id {task_id!r}; codebook has {sorted(self._sections)}")

    @classmethod
    def load(cls, path) -> "CompiledManifoldRuntime":
        path = Path(path)
        fh = open(path, "rb")
        mm = mmap.mmap(fh.fileno(), 0, prot=mmap.PROT_READ)
        fh.close()  # the mapping owns its own fd reference; safe to close here

        if len(mm) < HEADER.size:
            raise CodebookCorruptionError("file shorter than header")
        (magic, version, num_tasks, _created_ts, manifest_len,
         arrays_base_offset, payload_len, sha256) = HEADER.unpack_from(mm, 0)
        if magic != MAGIC:
            raise CodebookCorruptionError(f"bad magic {magic!r}")
        if version != FORMAT_VERSION:
            raise CodebookCorruptionError(f"unsupported format_version {version}")
        if len(mm) < HEADER.size + payload_len:
            raise CodebookCorruptionError("file truncated: payload shorter than header claims")
        payload = mm[HEADER.size:HEADER.size + payload_len]
        digest = hashlib.sha256(payload).digest()
        if digest != sha256:
            raise CodebookCorruptionError(
                f"payload sha256 mismatch: file is corrupted or tampered "
                f"(expected {sha256.hex()}, got {digest.hex()})")
        manifest = json.loads(payload[:manifest_len].decode("utf-8"))
        if manifest.get("format_version") != FORMAT_VERSION:
            raise CodebookCorruptionError("manifest format_version mismatch")
        if len(manifest["tasks"]) != num_tasks:
            raise CodebookCorruptionError("manifest task count mismatch with header")

        sections: Dict[str, TaskRuntimeSection] = {}
        for t in manifest["tasks"]:
            arrays = t["arrays"]
            def get(name):
                return _view(mm, arrays_base_offset, arrays[name])
            a = get("A")
            rho = float(np.max(np.abs(np.linalg.eigvals(a.astype(np.float64)))))
            if not rho < 1.0:
                raise CodebookCorruptionError(
                    f"task {t['task_id']!r}: loaded A is not contractive "
                    f"(rho={rho:.6f} >= 1.0); refusing to load a codebook whose "
                    f"Lyapunov certificate does not hold")
            paired = bool(t["paired"])
            sections[t["task_id"]] = TaskRuntimeSection(
                task_id=t["task_id"], k=int(t["k"]), paired=paired, dim=int(t["dim"]),
                n_components=int(t["n_components"]), cf_components=int(t["cf_components"]),
                rho_a=rho, A=a, inv_i_minus_a=get("inv_i_minus_a"), B=get("B"),
                W_c=get("W_c") if paired else None, codebook=get("codebook"),
                x_mean=get("x_mean"), x_basis_q=get("x_basis_q"),
                x_basis_scale=get("x_basis_scale"),
                c_mean=get("c_mean") if paired else None,
                c_basis_q=get("c_basis_q") if paired else None,
                c_basis_scale=get("c_basis_scale") if paired else None,
            )
        return cls(mm, sections, sha256.hex(), str(path))

    def step(self, task_id: str, h: np.ndarray, x_projected: np.ndarray,
              c_projected: Optional[np.ndarray]) -> np.ndarray:
        """One recurrence step `A h + B x (+ W_c c)`. Exposed for the per-step
        latency benchmark; `infer()` does not call this (it uses the
        precomputed closed form)."""
        sec = self.section(task_id)
        out = sec.A @ h + sec.B @ x_projected
        if c_projected is not None:
            if sec.W_c is None:
                raise ValueError(f"task {task_id!r} has no counterfactual drift map")
            out = out + sec.W_c @ c_projected
        return out

    def infer(self, task_id: str, x_projected: np.ndarray,
              c_projected: Optional[np.ndarray] = None) -> InferenceResult:
        sec = self.section(task_id)
        scratch = self._scratch[task_id]
        x_projected = np.asarray(x_projected, dtype=np.float32)
        if x_projected.shape != (sec.n_components,):
            raise ValueError(f"x_projected must have shape ({sec.n_components},)")
        drive = scratch["drive"]
        np.dot(sec.B, x_projected, out=drive)
        if c_projected is not None:
            if sec.W_c is None:
                raise ValueError(f"task {task_id!r} has no counterfactual drift map")
            c_projected = np.asarray(c_projected, dtype=np.float32)
            if c_projected.shape != (sec.cf_components,):
                raise ValueError(f"c_projected must have shape ({sec.cf_components},)")
            drive += sec.W_c @ c_projected
        h_star = scratch["h_star"]
        np.dot(sec.inv_i_minus_a, drive, out=h_star)
        diff = scratch["diff"]
        np.subtract(h_star[None, :], sec.codebook, out=diff)
        d2 = scratch["d2"]
        np.einsum("kd,kd->k", diff, diff, out=d2)
        prediction = int(np.argmin(d2))
        return InferenceResult(task_id=task_id, scores=-d2, prediction=prediction,
                                fixed_point=h_star.copy())

    def infer_routed(self, task_id, x_projected, zero_state, c_projected=None,
                     *, gateway: "CalibratedMoEGateway", covariance=None):
        codebook = self.infer(task_id, x_projected, c_projected)
        weights, fused, chart_weights, tangent = gateway.route(
            codebook.fixed_point, zero_state, covariance=covariance)
        decision_state = fused + gateway.ambient_to_shared @ tangent
        return RoutedInferenceResult(codebook, weights, fused, chart_weights, tangent, decision_state)

@dataclass(frozen=True)
class RoutedInferenceResult:
    """A decision state in a calibrated shared space, with its route evidence."""

    codebook: InferenceResult
    weights: Dict[str, float]
    fused_state: np.ndarray
    tangent_weights: Dict[str, float]
    tangent_state: np.ndarray
    decision_state: np.ndarray


class CalibratedMoEGateway:
    """Explicit linear calibration from two native manifolds into one shared space.

    The matrices must be fitted on calibration data outside this runtime. A
    dimension match alone does not establish that coordinates have the same
    meaning, so the gateway never invents an alignment.
    """

    def __init__(self, router, pool, codebook_to_shared, zero_to_shared,
                 shared_to_ambient, ambient_to_shared, *, chart_prototypes=None, temperature=1.0):
        from .wasserstein_moe_router import MultiTangentManifoldPool, WassersteinOptimalTransportRouter
        if not isinstance(router, WassersteinOptimalTransportRouter):
            raise TypeError("router must be WassersteinOptimalTransportRouter")
        if not isinstance(pool, MultiTangentManifoldPool):
            raise TypeError("pool must be MultiTangentManifoldPool")
        if set(router.names) != {"codebook", "zero"}:
            raise ValueError("router experts must be codebook and zero")
        self.router, self.pool = router, pool
        self.codebook_to_shared = self._matrix(codebook_to_shared, router.dim, "codebook_to_shared")
        self.zero_to_shared = self._matrix(zero_to_shared, router.dim, "zero_to_shared")
        self.shared_to_ambient = self._matrix(shared_to_ambient, pool.ambient_dim, "shared_to_ambient")
        self.ambient_to_shared = self._matrix(ambient_to_shared, router.dim, "ambient_to_shared")
        if self.shared_to_ambient.shape[1] != router.dim:
            raise ValueError("shared_to_ambient input dimension must match router")
        if self.ambient_to_shared.shape[1] != pool.ambient_dim:
            raise ValueError("ambient_to_shared input dimension must match tangent pool")
        if not np.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be positive finite")
        self.temperature = float(temperature)
        if chart_prototypes is None:
            raise ValueError("calibrated chart_prototypes are required")
        if set(chart_prototypes) != set(pool.CHART_NAMES):
            raise ValueError("chart_prototypes must cover all four charts")
        self.chart_prototypes = {name: self._vector(chart_prototypes[name], pool.ambient_dim, name)
                                 for name in pool.CHART_NAMES}
        for name, space in pool.spaces.items():
            if not space.rho < 1.0:
                raise ValueError(f"{name} is not contractive")

    @staticmethod
    def _matrix(value, rows, name):
        m = np.asarray(value, dtype=np.float64)
        if m.ndim != 2 or m.shape[0] != rows or not np.isfinite(m).all():
            raise ValueError(f"{name} must be a finite matrix with {rows} rows")
        return m.copy()

    @staticmethod
    def _vector(value, dim, name):
        v = np.asarray(value, dtype=np.float64)
        if v.shape != (dim,) or not np.isfinite(v).all():
            raise ValueError(f"{name} must be a finite ({dim},) vector")
        return v.copy()

    def route(self, codebook_state, zero_state, *, covariance=None):
        codebook_state = self._vector(codebook_state, self.codebook_to_shared.shape[1], "codebook_state")
        zero_state = self._vector(zero_state, self.zero_to_shared.shape[1], "zero_state")
        states = {"codebook": self.codebook_to_shared @ codebook_state,
                  "zero": self.zero_to_shared @ zero_state}
        # The input's barycenter is the symmetric, continuous observation used
        # for distance routing; neither expert's output is silently privileged.
        observation = 0.5 * (states["codebook"] + states["zero"])
        weights = self.router.route(observation, covariance)
        fused = self.router.fuse(weights, states)
        ambient = self.shared_to_ambient @ fused
        distances = np.array([np.sum((ambient - self.chart_prototypes[name]) ** 2)
                              for name in self.pool.CHART_NAMES])
        logits = -distances / self.temperature
        logits -= logits.max()
        chart_values = np.exp(logits)
        chart_values /= chart_values.sum()
        chart_weights = dict(zip(self.pool.CHART_NAMES, map(float, chart_values)))
        local = self.pool.evolve_all(ambient)
        tangent_state = self.pool.reconstruct_ambient(
            {name: chart_weights[name] * state for name, state in local.items()})
        return weights, fused, chart_weights, tangent_state
