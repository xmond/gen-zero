"""Spawned GGUF workers for ordered, chunked batch CAD inference, with optional NUMA pinning.

Why not multiprocessing.Pool: workers import gen_zero (and with it torch), which loads
libgomp before any Pool initializer can run. libgomp computes its OMP_PLACES from the
affinity mask at load time, so a late sched_setaffinity leaves OpenMP threads floating (or
bound to the wrong socket). Each worker here is started one at a time while the parent's
main thread temporarily carries the worker's cpuset and the OMP_* variables
(``numa_affinity.start_pinned``), so the child inherits both at exec, before any library
loads.

Each worker owns one ``CADEngine``. The engine is built by ``EngineSpec.factory``, a dotted
``module:callable`` resolved inside the worker; the default builds the GGUF engine.

Linux only; this module is deliberately not re-exported from ``gen_zero.causal``.
"""
from __future__ import annotations

import importlib
import logging
import multiprocessing as mp
import os
import queue
import sys
from dataclasses import dataclass
from pathlib import Path

if sys.platform != "linux":  # sched_setaffinity and /proc are Linux-only; fail at import, not mid-run
    raise ImportError("gguf_parallel_pool requires Linux")

from gen_zero.causal.numa_affinity import (
    OMP_PINNED_ENV, NumaTopology, WorkerPlacement, numa_pages, plan_placements, read_topology,
    start_pinned, thread_affinities)

log = logging.getLogger(__name__)

_STOP = None
GGUF_ENGINE_FACTORY = "gen_zero.causal.gguf_parallel_pool:build_gguf_engine"


@dataclass(frozen=True)
class EngineSpec:
    gguf_path: str
    head_path: str | None
    n_ctx: int
    n_threads: int
    use_mmap: bool
    alpha: float = 0.5
    factory: str = GGUF_ENGINE_FACTORY


def build_gguf_engine(spec: EngineSpec):
    from gen_zero.causal.cad_engine import CADEngine

    return CADEngine.from_gguf(spec.gguf_path, head_path=spec.head_path, n_ctx=spec.n_ctx,
                               n_threads=spec.n_threads, use_mmap=spec.use_mmap, alpha=spec.alpha)


def resolve_factory(path: str):
    module, sep, name = path.partition(":")
    if not sep or not module or not name:
        raise ValueError(f"engine factory must look like 'module:callable', got {path!r}")
    return getattr(importlib.import_module(module), name)


def _run_item(engine, mode: str, question: str, context: str) -> dict:
    if mode == "raw":
        cond, prior, tokens = engine.raw_logits(question, context)
        return {"conditional": cond.tolist(), "prior": prior.tolist(), "input_tokens": tokens}
    if mode == "predict":
        return engine.predict(question, context).to_dict()
    return engine.classify(question, context).to_dict()


def _worker_main(worker: int, placement: WorkerPlacement | None, spec: EngineSpec, tasks, results):
    try:
        if placement is not None:
            # libgomp (loaded via torch at import) already bound this main thread to its first
            # place, so the mask is a subset of the plan, never the whole plan.
            got = os.sched_getaffinity(0)
            if not got or not got <= set(placement.cpus):
                raise RuntimeError(f"worker {worker} runs on cpus {sorted(got)}, outside plan {list(placement.cpus)}")
            for key, value in OMP_PINNED_ENV.items():
                if os.environ.get(key) != value:
                    raise RuntimeError(f"worker {worker} missing {key}={value} in its environment")
        engine = resolve_factory(spec.factory)(spec)
    except Exception as exc:  # report, never hang the parent
        results.put(("init", worker, f"{type(exc).__name__}: {exc}"))
        return
    results.put(("ready", worker, os.getpid()))
    while True:
        item = tasks.get()
        if item is _STOP:
            return
        index, chunk = item
        try:
            out = [_run_item(engine, mode, q, c) for mode, q, c in chunk]
            results.put(("ok", index, out))
        except Exception as exc:
            results.put(("error", index, f"{type(exc).__name__}: {exc}"))


class GGUFParallelPool:
    """Each process owns one llama context. numa_pin=True places worker w on node w % nodes.

    threads_per_worker is the number of compute threads; under numa_pin it is also the
    number of physical cores the worker owns exclusively (both HT siblings in its mask).
    use_mmap=False makes each worker read the weights into anonymous memory that is
    first-touched on its own node; with mmap the page cache stays wherever the file was
    first read, so pinning then only localizes the KV cache and activations.
    """

    def __init__(self, gguf_path, *, workers, threads_per_worker, n_ctx=2048, head_path=None,
                 chunk_size=1, numa_pin=False, use_mmap=True, topology: NumaTopology | None = None,
                 alpha: float = 0.5, factory: str = GGUF_ENGINE_FACTORY):
        if workers < 1 or threads_per_worker < 1 or chunk_size < 1 or n_ctx < 1:
            raise ValueError("workers, threads_per_worker, chunk_size and n_ctx must be positive")
        if not Path(gguf_path).is_file():
            raise FileNotFoundError(f"GGUF model missing: {gguf_path}")
        if head_path and not Path(head_path).is_file():
            raise FileNotFoundError(head_path)
        resolve_factory(factory)  # fail in the parent, not as N identical worker init errors
        self.chunk_size = chunk_size
        self.has_head = head_path is not None
        self.numa_pin = numa_pin
        self.alpha = alpha
        self.placements: list[WorkerPlacement | None]
        if numa_pin:
            self.topology = topology or read_topology()
            self.placements = list(plan_placements(self.topology, workers, threads_per_worker))
            if len(self.topology.nodes) == 1:
                log.warning("numa_pin on a single-node host: cores are still exclusive, no socket split")
        else:
            self.topology = None
            self.placements = [None] * workers
        spec = EngineSpec(str(gguf_path), str(head_path) if head_path else None, n_ctx,
                          threads_per_worker, use_mmap, alpha=alpha, factory=factory)
        self.spec = spec
        ctx = mp.get_context("spawn")
        self._tasks = ctx.Queue()
        self._results = ctx.Queue()
        self._procs = []
        self._pids: dict[int, int] = {}
        try:
            for w, placement in enumerate(self.placements):
                self._procs.append(self._start_worker(ctx, w, placement, spec))
            self._await_ready(workers)
        except BaseException:
            self.terminate()
            raise

    def _start_worker(self, ctx, worker, placement, spec):
        proc = ctx.Process(target=_worker_main, name=f"gguf-worker-{worker}",
                           args=(worker, placement, spec, self._tasks, self._results), daemon=True)
        start_pinned(proc, placement)
        return proc

    def _await_ready(self, workers):
        while len(self._pids) < workers:
            try:
                kind, worker, payload = self._results.get(timeout=5.0)
            except queue.Empty:
                dead = [p.name for p in self._procs if p.exitcode is not None]
                if dead:
                    raise RuntimeError(f"GGUF workers exited during startup without reporting: {dead}")
                continue
            if kind == "init":
                raise RuntimeError(f"GGUF worker {worker} initialization failed: {payload}")
            if kind != "ready":
                raise RuntimeError(f"unexpected message {kind!r} from worker {worker} during startup")
            self._pids[worker] = payload

    @property
    def worker_pids(self):
        return [self._pids[w] for w in range(len(self._procs))]

    def _batch(self, items, mode):
        items = list(items)
        if not items:
            raise ValueError("empty inference batch")
        if any(len(i) != 2 or not all(isinstance(s, str) for s in i) for i in items):
            raise TypeError("every item must be a (question, context) pair of strings")
        if mode == "predict" and not self.has_head:
            raise RuntimeError("a head_path is required for predict_batch()")
        chunks = [[(mode, q, c) for q, c in items[i:i + self.chunk_size]]
                  for i in range(0, len(items), self.chunk_size)]
        for index, chunk in enumerate(chunks):
            self._tasks.put((index, chunk))
        done: dict[int, list] = {}
        while len(done) < len(chunks):
            try:
                kind, index, payload = self._results.get(timeout=5.0)
            except queue.Empty:
                dead = [p.name for p in self._procs if not p.is_alive()]
                if dead:
                    raise RuntimeError(f"GGUF workers died mid-batch: {dead}")
                continue
            if kind == "error":
                raise RuntimeError(f"GGUF worker failed on chunk {index}: {payload}")
            if kind != "ok":
                raise RuntimeError(f"unexpected message {kind!r} mid-batch")
            done[index] = payload
        return [result for index in range(len(chunks)) for result in done[index]]

    def raw_logits_batch(self, items):
        """Ordered [{'conditional', 'prior', 'input_tokens'}] for (question, context) pairs."""
        return self._batch(items, "raw")

    def predict_batch(self, items):
        """Ordered calibrated CADResult dicts. Needs head_path."""
        return self._batch(items, "predict")

    def classify_batch(self, items):
        """Ordered CADResult dicts; each carries its own ``calibrated`` flag."""
        return self._batch(items, "classify")

    def pinning_report(self):
        """Per-worker observed affinity of EVERY thread, versus the plan.

        Call after at least one batch so the OpenMP compute threads exist. Raises if any
        thread of a pinned worker can run outside its planned cpuset.
        """
        report = []
        for w, pid in self._pids.items():
            placement = self.placements[w]
            threads = thread_affinities(pid)
            observed = set().union(*threads.values())
            entry = {"worker": w, "pid": pid, "threads": len(threads),
                     "observed_cpus": sorted(observed),
                     "planned_node": None if placement is None else placement.node,
                     "planned_cpus": None if placement is None else list(placement.cpus),
                     "numa_pages": numa_pages(pid)}
            if placement is not None:
                stray = {tid: sorted(c - set(placement.cpus)) for tid, c in threads.items()
                         if c - set(placement.cpus)}
                if stray:
                    raise RuntimeError(f"worker {w} (pid {pid}) has threads outside node {placement.node}: {stray}")
            report.append(entry)
        return report

    def close(self):
        for _ in self._procs:
            self._tasks.put(_STOP)
        for p in self._procs:
            p.join(timeout=60)
        self.terminate()

    def terminate(self):
        for p in self._procs:
            if p.is_alive():
                p.terminate()
        for p in self._procs:
            p.join(timeout=10)
        for q in (getattr(self, "_tasks", None), getattr(self, "_results", None)):
            if q is not None:
                q.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            self.close()
        else:
            self.terminate()
