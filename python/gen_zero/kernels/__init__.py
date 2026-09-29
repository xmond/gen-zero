"""Gen-Zero GPU/Triton kernels with CPU fallbacks."""
from .triton_compact_tree import (
    compact_active,
    compact_active_mask,
    available_backends,
    HAS_TRITON,
)
__all__ = ["compact_active", "compact_active_mask", "available_backends", "HAS_TRITON"]
