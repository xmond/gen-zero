"""Gen-Zero 24/7 Autonomous Continuous RSI Daemon Package.

Modules:
- atomic_container: AtomicModelContainer for zero-downtime hot reloading.
- curriculum_self_play: CurriculumSelfPlayGenerator for automated frontier task synthesis.
- daemon_engine: GenZeroRSIDaemon coordinating autonomous continuous self-evolution.
"""

from .atomic_container import AtomicModelContainer
from .curriculum_self_play import CurriculumSelfPlayGenerator
from .daemon_engine import GenZeroRSIDaemon

__all__ = [
    "AtomicModelContainer",
    "CurriculumSelfPlayGenerator",
    "GenZeroRSIDaemon",
]
