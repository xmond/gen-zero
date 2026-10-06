"""NUMA topology reader, worker cpuset planner and pinned process start (Linux sysfs only).

The allocation unit is one physical core: a worker that computes with T threads gets
T physical cores from ONE NUMA node, with every hyper-thread sibling of those cores
in its mask. With OMP_PLACES=cores and OMP_PROC_BIND=close libgomp then pins one
compute thread per core and never crosses a socket.

No numactl dependency. Every failure raises: a pin that cannot be honoured is an error,
never a silent "run unpinned".
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

OMP_PINNED_ENV = {"OMP_PROC_BIND": "close", "OMP_PLACES": "cores"}


def parse_cpulist(text: str) -> list[int]:
    """Parse a kernel cpulist such as ``0-3,8,10-11``."""
    cpus: list[int] = []
    for part in text.strip().split(","):
        if not part:
            continue
        lo, _, hi = part.partition("-")
        cpus.extend(range(int(lo), int(hi or lo) + 1))
    return cpus


@dataclass(frozen=True)
class NumaTopology:
    # nodes[node_id] = list of physical cores; each core = sorted tuple of its logical CPUs.
    nodes: tuple[tuple[tuple[int, ...], ...], ...]

    @property
    def logical_cpus(self) -> int:
        return sum(len(core) for node in self.nodes for core in node)

    def node_cpus(self, node: int) -> frozenset[int]:
        return frozenset(cpu for core in self.nodes[node] for cpu in core)

    def node_of(self, cpus: frozenset[int]) -> int:
        """Node containing all cpus; raises if the set straddles nodes."""
        owners = {n for n in range(len(self.nodes)) if cpus & self.node_cpus(n)}
        if len(owners) != 1:
            raise ValueError(f"cpuset {sorted(cpus)} spans NUMA nodes {sorted(owners)}")
        return owners.pop()


def read_topology(sysfs: str | Path = "/sys") -> NumaTopology:
    """Parse ``<sysfs>/devices/system/{node,cpu}`` into a NumaTopology."""
    base = Path(sysfs) / "devices/system"
    node_dirs = sorted(base.glob("node/node[0-9]*"), key=lambda p: int(p.name[4:]))
    if not node_dirs:
        raise RuntimeError(f"no NUMA nodes under {base / 'node'}")
    nodes = []
    seen: set[int] = set()
    for node_dir in node_dirs:
        cpus = parse_cpulist((node_dir / "cpulist").read_text())
        cores: dict[tuple[int, ...], None] = {}
        for cpu in cpus:
            siblings = tuple(sorted(parse_cpulist(
                (base / f"cpu/cpu{cpu}/topology/thread_siblings_list").read_text())))
            if not set(siblings) <= set(cpus):
                raise RuntimeError(f"cpu{cpu} siblings {siblings} leave node {node_dir.name}")
            cores.setdefault(siblings, None)
        if not cores:
            raise RuntimeError(f"{node_dir.name} has no CPUs")
        for core in cores:
            if seen & set(core):
                raise RuntimeError(f"core {core} listed in two nodes")
            seen.update(core)
        nodes.append(tuple(cores))
    return NumaTopology(tuple(nodes))


@dataclass(frozen=True)
class WorkerPlacement:
    worker: int
    node: int
    cpus: tuple[int, ...]  # logical CPUs, both siblings of every assigned core


def plan_placements(topology: NumaTopology, workers: int, cores_per_worker: int) -> list[WorkerPlacement]:
    """Round-robin workers over nodes; disjoint physical cores inside each node.

    Fails closed on oversubscription: this planner never shares a core between workers
    and never lets one worker straddle a node, because the whole point is removing
    remote memory traffic. Pick a smaller workers x cores grid instead.
    """
    if workers < 1 or cores_per_worker < 1:
        raise ValueError("workers and cores_per_worker must be positive")
    n_nodes = len(topology.nodes)
    per_node = [cores_per_worker * (workers // n_nodes + (1 if n < workers % n_nodes else 0))
                for n in range(n_nodes)]
    for node, need in enumerate(per_node):
        have = len(topology.nodes[node])
        if need > have:
            raise ValueError(
                f"NUMA node {node} has {have} physical cores but {workers} workers x "
                f"{cores_per_worker} cores needs {need} on it; shrink the grid")
    cursor = [0] * n_nodes
    placements = []
    for w in range(workers):
        node = w % n_nodes
        cores = topology.nodes[node][cursor[node]:cursor[node] + cores_per_worker]
        cursor[node] += cores_per_worker
        placements.append(WorkerPlacement(w, node, tuple(sorted(c for core in cores for c in core))))
    return placements


def start_pinned(proc, placement: WorkerPlacement | None) -> None:
    """Start a spawn-context Process so it inherits placement.cpus and the OMP_* env at exec.

    libgomp computes OMP_PLACES from the affinity mask when it loads, so a late
    sched_setaffinity inside the child leaves OpenMP threads floating. The parent's main
    thread therefore carries the mask only for the duration of start(); both the mask
    and the environment are restored afterwards, even on failure.
    """
    if placement is None:
        proc.start()
        return
    saved_env = {k: os.environ.get(k) for k in OMP_PINNED_ENV}
    saved_mask = os.sched_getaffinity(0)
    try:
        os.environ.update(OMP_PINNED_ENV)
        os.sched_setaffinity(0, placement.cpus)
        if os.sched_getaffinity(0) != set(placement.cpus):
            raise RuntimeError(f"cannot pin to {placement.cpus}; cpus offline or cgroup-restricted")
        proc.start()
    finally:
        os.sched_setaffinity(0, saved_mask)
        for k, v in saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def thread_affinities(pid: int) -> dict[int, set[int]]:
    """Cpus_allowed_list of EVERY thread of ``pid``, read from /proc (same uid)."""
    threads = {}
    for task in Path(f"/proc/{pid}/task").iterdir():
        status = (task / "status").read_text()
        m = re.search(r"^Cpus_allowed_list:\s*(\S+)$", status, re.M)
        if not m:
            raise RuntimeError(f"no Cpus_allowed_list for pid {pid} tid {task.name}")
        threads[int(task.name)] = set(parse_cpulist(m.group(1)))
    return threads


def numa_pages(pid: int) -> dict[str, dict[str, int]]:
    """Pages per NUMA node from /proc/<pid>/numa_maps, split into anon and file-backed."""
    totals: dict[str, dict[str, int]] = {"anon": {}, "file": {}}
    for line in Path(f"/proc/{pid}/numa_maps").read_text().splitlines():
        kind = "file" if " file=" in line else "anon"
        for node, pages in re.findall(r"\bN(\d+)=(\d+)", line):
            totals[kind][f"N{node}"] = totals[kind].get(f"N{node}", 0) + int(pages)
    return totals
