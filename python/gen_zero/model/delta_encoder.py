"""Zero-State First-Order Delta Feature Channel (Velocity & Momentum Sensing).

Implements Milestone 4 of Issue #19:
1. First-order discrete difference computation:
   Delta S_t = S_t - S_{t-1}
2. Parallel concatenated representation:
   S_tilde_t = [S_t, Delta S_t]
3. Zero-State Overhead:
   Eliminates recurrent memory cells (RNN/LSTM/state tracking tensors) while providing
   instantaneous momentum, velocity, and acceleration features to pure Prefill decision graphs.
"""

from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np


class ZeroStateDeltaEncoder:
    """Computes first-order discrete difference features Delta S_t = S_t - S_{t-1} with zero state maintenance overhead."""

    def __init__(
        self,
        clamp_value: float = 10.0,
        normalize: bool = True,
    ):
        self.clamp_value = clamp_value
        self.normalize = normalize

    def compute_vector_delta(
        self,
        curr_vec: Union[np.ndarray, List[float]],
        prev_vec: Optional[Union[np.ndarray, List[float]]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Computes discrete difference and concatenated representation [S_t, Delta S_t].

        Args:
            curr_vec: Current observation vector S_t of shape (D,).
            prev_vec: Previous observation vector S_{t-1} of shape (D,) or None for initial frame.

        Returns:
            Tuple of (delta_vec, concatenated_vec) of shapes (D,) and (2D,).
        """
        curr = np.asarray(curr_vec, dtype=np.float32)
        if prev_vec is None:
            delta = np.zeros_like(curr)
        else:
            prev = np.asarray(prev_vec, dtype=np.float32)
            if prev.shape != curr.shape:
                delta = np.zeros_like(curr)
            else:
                delta = curr - prev

        if self.normalize:
            delta = np.clip(delta, -self.clamp_value, self.clamp_value)

        concat = np.concatenate([curr, delta], axis=-1)
        return delta, concat

    def compute_dict_delta(
        self,
        curr_dict: Dict[str, Any],
        prev_dict: Optional[Dict[str, Any]] = None,
        numeric_keys: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """Extracts first-order difference for numeric fields between two dictionary observations.

        Args:
            curr_dict: Current observation dictionary.
            prev_dict: Previous observation dictionary or None.
            numeric_keys: Optional explicit keys to difference. If None, differences all float/int fields.

        Returns:
            Dictionary containing 'delta_{k}' for each numeric field.
        """
        deltas: Dict[str, float] = {}
        target_keys = numeric_keys if numeric_keys is not None else [
            k for k, v in curr_dict.items() if isinstance(v, (int, float)) and not isinstance(v, bool)
        ]

        for k in target_keys:
            c_val = float(curr_dict.get(k, 0.0))
            if prev_dict is not None and k in prev_dict and isinstance(prev_dict[k], (int, float)):
                p_val = float(prev_dict[k])
                d_val = c_val - p_val
            else:
                d_val = 0.0

            if self.normalize:
                d_val = max(-self.clamp_value, min(self.clamp_value, d_val))

            deltas[f"delta_{k}"] = round(d_val, 4)

        return deltas

    def format_delta_text(self, val: float, delta_val: float, precision: int = 2) -> str:
        """Formats scalar value with its first-order derivative sign annotation."""
        sign = "+" if delta_val >= 0 else ""
        return f"{val:.{precision}f} (Δ{sign}{delta_val:.{precision}f})"
