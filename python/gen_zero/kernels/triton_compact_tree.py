"""Stream-compaction kernel for dynamic MCTS tree nodes.

This module compacts a boolean mask of *active* MCTS nodes into a dense array
of their indices. It is the filtering primitive used by
:class:`gen_zero.causal.vectorized_latent_mcts.BatchLatentMctsPlanner` to drop
instances that have already expanded a leaf during a rollout descent, so the
vectorized inner loop only gathers the surviving instances each step.

Two backends are provided:

1. **Triton path** (CUDA + ``triton``). Stream compaction is implemented as a
   classic two-pass prefix-sum + scatter:

   * Pass 1 (host-side on the GPU tensor): compute the *exclusive* prefix sum of
     the int-cast mask, ``excum = cumsum(mask) - mask``. ``excum[i]`` is the
     number of active elements strictly before index ``i``, i.e. the write
     position ``i`` would take if it is itself active.
   * Pass 2 (``@triton.jit`` scatter kernel): each program walks a tile of
     ``BLOCK_SIZE`` elements and, for every ``i`` with ``mask[i] == 1``, writes
     ``out[excum[i]] = i`` via ``tl.store`` guarded by the elementwise mask.

   The key correctness property is *determinism without atomics*: because
   ``excum`` is a strict prefix sum, the write positions ``excum[i]`` for active
   ``i`` form a strictly increasing sequence, so every write target is unique.
   The order in which programs execute therefore does not matter: the output is
   always the active indices in ascending order, identical to
   ``np.flatnonzero(mask)``. No ``tl.atomic_*`` call is needed.

2. **CPU path** (NumPy, always available). ``_compact_cpu`` is a thin wrapper
   around :func:`numpy.flatnonzero` over a boolean view, so it returns exactly
   the same indices as the Triton path, byte-for-byte. It accepts either a NumPy
   array or a PyTorch tensor (converted via ``.detach().cpu().numpy()``).

The module imports cleanly even when ``triton`` (and/or ``torch``) is absent:
both imports are guarded, and the ``@triton.jit`` decorator only executes
inside the successful ``try``-branch. The CPU fallback makes the kernel usable
on every machine; the Triton path is the performance path on a CUDA+Triton box.
"""

from __future__ import annotations

from typing import Tuple, Union

import numpy as np

__all__ = ["compact_active", "compact_active_mask", "available_backends", "HAS_TRITON"]

# Guarded torch import: torch is installed here, but the module must not break
# if it is ever dropped.
try:  # pragma: no cover - import guard
    import torch as _torch  # noqa: F401

    HAS_TORCH = True
except Exception:  # pragma: no cover - import guard
    _torch = None
    HAS_TORCH = False

# Guarded triton import: the @triton.jit decorator is only evaluated here, so
# the module imports cleanly when triton is absent (the common case on CPU
# boxes). HAS_TRITON reflects whether *both* triton imported AND we have a
# CUDA-capable torch build; see ``available_backends``.
try:  # pragma: no cover - import guard
    import triton as _triton
    import triton.language as _tl

    HAS_TRITON = True

    @_triton.jit
    def _compact_scatter_kernel(
        mask_ptr,      # *int32* pointer: 0/1 mask values
        excum_ptr,     # *int64* pointer: exclusive prefix sum of mask
        out_ptr,       # *int64* pointer: output indices buffer
        n,
        BLOCK_SIZE: "_tl.constexpr",
    ):
        """Write each active index ``i`` to ``out[excum[i]] = i``.

        Grid is ``(cdiv(n, BLOCK_SIZE),)``. Each program owns a contiguous tile
        ``[pid * BLOCK_SIZE, (pid + 1) * BLOCK_SIZE)`` of the input. Within the
        tile, ``tl.arange`` produces the local offsets; the global indices are
        ``pid * BLOCK_SIZE + offsets``. The elementwise guard
        ``mask == 1`` selects exactly the active lanes, and each selected lane
        stores its global index at ``out[excum[i]]``. Because ``excum`` is a
        strict prefix sum, the write positions are unique and ascending, so no
        atomics are needed and the result is deterministic regardless of the
        order programs run in.
        """
        pid = _tl.program_id(0)
        offsets = _tl.arange(0, BLOCK_SIZE)
        idx = pid * BLOCK_SIZE + offsets
        in_tile = idx < n
        # Load mask and exclusive prefix sum only for in-bounds lanes; out-of
        # bounds lanes are masked off so they never trigger a store.
        m = _tl.load(mask_ptr + idx, mask=in_tile, other=0).to(_tl.int1)
        excum = _tl.load(excum_ptr + idx, mask=in_tile, other=0)
        active = m & in_tile
        _tl.store(out_ptr + excum, idx.to(_tl.int64), mask=active)


except Exception:  # pragma: no cover - import guard
    _triton = None
    _tl = None
    HAS_TRITON = False
    _compact_scatter_kernel = None  # type: ignore[assignment]


# Resolve the public HAS_TRITON to the *runtime-usable* definition: triton
# imported AND torch present AND CUDA available. The bare import-succeed flag
# above still matters (it gates the kernel object), but the public symbol
# mirrors ``available_backends`` so callers can branch on it.
if HAS_TRITON and HAS_TORCH:
    try:
        _HAS_TRITON_RUNTIME = bool(_torch.cuda.is_available())
    except Exception:  # pragma: no cover - defensive
        _HAS_TRITON_RUNTIME = False
else:
    _HAS_TRITON_RUNTIME = False


def available_backends() -> set:
    """Return the set of usable compaction backends on this machine.

    Always includes ``"cpu"`` (NumPy fallback, no optional deps). Includes
    ``"triton"`` exactly when ``triton`` imports AND ``torch`` is available
    AND ``torch.cuda.is_available()`` is true.
    """
    backends = {"cpu"}
    if _HAS_TRITON_RUNTIME:
        backends.add("triton")
    return backends


# Re-export the runtime flag as the public module constant.
HAS_TRITON = _HAS_TRITON_RUNTIME


def _is_numpy(x) -> bool:
    return isinstance(x, np.ndarray)


def _to_numpy(mask) -> np.ndarray:
    """Return a 1-D boolean NumPy view of ``mask`` from numpy or torch input."""
    if isinstance(mask, np.ndarray):
        arr = mask
    elif HAS_TORCH and isinstance(mask, _torch.Tensor):
        arr = mask.detach().cpu().numpy()
    else:
        raise TypeError(
            "mask must be a numpy.ndarray or torch.Tensor, got "
            f"{type(mask).__name__}"
        )
    if arr.ndim != 1:
        raise ValueError(f"mask must be 1-D, got shape {arr.shape}")
    return arr.astype(bool, copy=False)


def _compact_cpu(mask) -> Tuple[np.ndarray, int]:
    """CPU fallback: compact ``mask`` via :func:`np.flatnonzero`.

    Returns ``(indices, count)`` where ``indices`` is a ``int64`` NumPy array of
    the active indices in ascending order and ``count == len(indices)``. Works
    for both NumPy and torch inputs.
    """
    arr = _to_numpy(mask)
    indices = np.flatnonzero(arr).astype(np.int64, copy=False)
    return indices, int(len(indices))


def _compact_triton(mask_tensor) -> Tuple["_torch.Tensor", int]:
    """Triton path: exclusive-prefix-sum + scatter kernel on the GPU.

    ``mask_tensor`` is a 1-D ``torch.Tensor`` (any device triton supports; in
    practice CUDA). Returns ``(out, count)`` where ``out`` is an ``int64``
    tensor of the active indices (ascending) and ``count == out.numel()``.
    """
    if not (HAS_TRITON and HAS_TORCH):
        raise ValueError("triton backend not available")
    n = mask_tensor.numel()
    int_mask = mask_tensor.to(_torch.int64)
    if n == 0:
        return _torch.empty(0, dtype=_torch.int64, device=mask_tensor.device), 0
    excum = _torch.cumsum(int_mask, dim=0) - int_mask  # exclusive prefix sum
    out = _torch.empty((n,), dtype=_torch.int64, device=mask_tensor.device)
    count = int(int_mask.sum().item())

    BLOCK_SIZE = 1024
    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    _compact_scatter_kernel[grid](
        int_mask, excum, out, n, BLOCK_SIZE=BLOCK_SIZE, num_warps=4,
    )
    # The scatter writes the first ``count`` positions; trim the slack tail.
    return out[:count].contiguous(), count


def compact_active(mask, *, backend=None) -> Tuple[Union[np.ndarray, "_torch.Tensor"], int]:
    """Compact ``mask`` into ``(indices, count)``.

    Parameters
    ----------
    mask : numpy.ndarray or torch.Tensor
        1-D boolean array/tensor of shape ``(N,)`` (e.g. ``N == B`` or
        ``N == B * capacity`` over MCTS nodes).
    backend : {"triton", "cpu", None}, optional
        Explicit backend selection. ``None`` auto-selects: use Triton when
        Triton+CUDA+torch are all available *and* the input is a CUDA torch
        tensor, otherwise fall back to CPU.

    Returns
    -------
    (indices, count)
        ``indices`` is a 1-D ``int64`` array (CPU path: NumPy) or tensor
        (Triton path: torch on the mask's device) of the indices where
        ``mask`` is True, in ascending order — identical to
        ``np.flatnonzero(mask)``. ``count == len(indices)`` (Python int).

    Raises
    ------
    ValueError
        If ``backend`` is an unknown string, or ``backend="triton"`` is
        requested when the Triton backend is unavailable.
    """
    valid = {"triton", "cpu", None}
    if backend not in valid:
        raise ValueError(
            f"backend must be one of {sorted(v for v in valid if v is not None)} "
            f"or None, got {backend!r}"
        )

    is_torch = HAS_TORCH and isinstance(mask, _torch.Tensor)
    triton_available = _HAS_TRITON_RUNTIME

    if backend == "cpu":
        return _compact_cpu(mask)
    if backend == "triton":
        if not triton_available:
            raise ValueError("triton backend not available on this machine")
        if not is_torch:
            # Triton path requires a torch tensor; coerce a numpy input.
            mask = _torch.as_tensor(mask)
        return _compact_triton(mask)
    # backend is None: auto-select.
    if triton_available and is_torch and mask.is_cuda:
        return _compact_triton(mask)
    return _compact_cpu(mask)


def compact_active_mask(mask, *, backend=None):
    """Convenience alias returning only the indices (see :func:`compact_active`)."""
    indices, _ = compact_active(mask, backend=backend)
    return indices
