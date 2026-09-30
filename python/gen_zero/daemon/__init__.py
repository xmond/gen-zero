"""Atomic serving container.

- atomic_container: AtomicModelContainer, which pins one model/scorer snapshot per request.
"""

from .atomic_container import AtomicModelContainer

__all__ = [
    "AtomicModelContainer",
]
