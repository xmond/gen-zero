"""L2 Order Book Microstructural State Modeling & Feature Encoding.

Implements Milestone 2 of Issue #18:
1. L2OrderBookSnapshot data structure for depth-level bids/asks and CVD.
2. Microstructural Feature Extraction:
   - Book Imbalance: Normalized ratio in [-1.0, 1.0] of top-K levels.
   - Spread in basis points: (best_ask - best_bid) / mid_price * 10000.
   - Depth Gradient Profile: Cumulative volume within 10bps, 25bps, 50bps of mid price.
   - Cumulative Volume Delta (CVD) net pressure.
3. State Text Encoding: Compact, 0-token format optimized for Gen-Zero reflex scoring.
"""

from typing import Dict, List, Tuple, Any, Optional
import dataclasses
import math
import time


@dataclasses.dataclass
class L2OrderBookSnapshot:
    """Represents an L2 order book snapshot at a specific millisecond tick."""
    bids: List[Tuple[float, float]]  # [(price, size), ...] sorted descending by price
    asks: List[Tuple[float, float]]  # [(price, size), ...] sorted ascending by price
    cvd: float = 0.0                 # Cumulative Volume Delta (taker buy vol - taker sell vol)
    last_price: float = 0.0
    timestamp_ms: int = 0
    tick_id: int = 0

    def __post_init__(self):
        if not self.timestamp_ms:
            self.timestamp_ms = int(time.time() * 1000)
        if not self.tick_id:
            self.tick_id = int(self.timestamp_ms % 100000000)


class L2MarketStateEncoder:
    """Extracts microstructural signals and compact state representations from L2 snapshots."""

    def __init__(self, top_k_levels: int = 5):
        self.top_k_levels = top_k_levels

    def is_valid_book(self, snap: L2OrderBookSnapshot) -> bool:
        """Validates that the order book snapshot is non-empty and non-crossed."""
        if not snap.bids or not snap.asks:
            return False
        best_bid = snap.bids[0][0]
        best_ask = snap.asks[0][0]
        if best_bid <= 0 or best_ask <= 0:
            return False
        if best_bid >= best_ask:  # Crossed or locked book
            return False
        return True

    def compute_mid_price(self, snap: L2OrderBookSnapshot) -> float:
        """Calculates the microstructural mid-price."""
        if not self.is_valid_book(snap):
            return snap.last_price if snap.last_price > 0 else 100.0
        return (snap.bids[0][0] + snap.asks[0][0]) / 2.0

    def compute_spread_bps(self, snap: L2OrderBookSnapshot) -> float:
        """Calculates best bid-ask spread in basis points (bps)."""
        if not self.is_valid_book(snap):
            return 0.0
        best_bid = snap.bids[0][0]
        best_ask = snap.asks[0][0]
        mid = (best_bid + best_ask) / 2.0
        if mid <= 1e-6:
            return 0.0
        return round(((best_ask - best_bid) / mid) * 10000.0, 2)

    def compute_book_imbalance(self, snap: L2OrderBookSnapshot, k: Optional[int] = None) -> float:
        """Computes top-k book volume imbalance normalized strictly in [-1.0, 1.0].

        Formula: (bid_vol - ask_vol) / (bid_vol + ask_vol)
        +1.0 indicates pure buy-side pressure, -1.0 indicates pure sell-side pressure.
        """
        if not self.is_valid_book(snap):
            return 0.0

        levels = k or self.top_k_levels
        bid_vol = sum(size for _, size in snap.bids[:levels])
        ask_vol = sum(size for _, size in snap.asks[:levels])

        total_vol = bid_vol + ask_vol
        if total_vol <= 1e-6:
            return 0.0

        imbalance = (bid_vol - ask_vol) / total_vol
        return max(-1.0, min(1.0, round(imbalance, 4)))

    def compute_depth_profile(
        self,
        snap: L2OrderBookSnapshot,
        bps_tiers: Tuple[int, ...] = (10, 25, 50),
    ) -> Dict[str, float]:
        """Calculates cumulative liquidity within specified basis point bands from mid-price."""
        if not self.is_valid_book(snap):
            return {f"bid_{tier}bps": 0.0 for tier in bps_tiers} | {f"ask_{tier}bps": 0.0 for tier in bps_tiers}

        mid = self.compute_mid_price(snap)
        profile: Dict[str, float] = {}

        for tier in bps_tiers:
            band_pct = tier / 10000.0
            bid_cutoff = mid * (1.0 - band_pct)
            ask_cutoff = mid * (1.0 + band_pct)

            tier_bid_vol = sum(size for price, size in snap.bids if price >= bid_cutoff)
            tier_ask_vol = sum(size for price, size in snap.asks if price <= ask_cutoff)

            profile[f"bid_{tier}bps"] = round(tier_bid_vol, 2)
            profile[f"ask_{tier}bps"] = round(tier_ask_vol, 2)

        return profile

    def compute_delta(
        self,
        curr: L2OrderBookSnapshot,
        prev: Optional[L2OrderBookSnapshot] = None,
    ) -> Dict[str, float]:
        """Computes first-order discrete differences Delta S_t = S_t - S_{t-1} between adjacent ticks."""
        curr_mid = self.compute_mid_price(curr)
        curr_spread = self.compute_spread_bps(curr)
        curr_imb = self.compute_book_imbalance(curr)
        curr_cvd = curr.cvd

        if prev is None:
            return {
                "delta_mid_price": 0.0,
                "delta_spread_bps": 0.0,
                "delta_imbalance": 0.0,
                "delta_cvd": 0.0,
            }

        prev_mid = self.compute_mid_price(prev)
        prev_spread = self.compute_spread_bps(prev)
        prev_imb = self.compute_book_imbalance(prev)
        prev_cvd = prev.cvd

        return {
            "delta_mid_price": round(curr_mid - prev_mid, 4),
            "delta_spread_bps": round(curr_spread - prev_spread, 2),
            "delta_imbalance": round(curr_imb - prev_imb, 4),
            "delta_cvd": round(curr_cvd - prev_cvd, 2),
        }

    def encode_to_features(
        self,
        snap: L2OrderBookSnapshot,
        prev_snap: Optional[L2OrderBookSnapshot] = None,
    ) -> Dict[str, Any]:
        """Extracts complete numeric and structured features from the snapshot with first-order deltas."""
        mid = self.compute_mid_price(snap)
        spread_bps = self.compute_spread_bps(snap)
        imbalance = self.compute_book_imbalance(snap)
        depth_profile = self.compute_depth_profile(snap)
        deltas = self.compute_delta(snap, prev_snap)

        return {
            "tick_id": snap.tick_id,
            "timestamp_ms": snap.timestamp_ms,
            "mid_price": mid,
            "spread_bps": spread_bps,
            "imbalance": imbalance,
            "cvd": snap.cvd,
            "depth_profile": depth_profile,
            "is_valid": self.is_valid_book(snap),
            **deltas,
        }

    def encode_to_state_text(self, snap: L2OrderBookSnapshot) -> str:
        """Encodes snapshot into compact, token-efficient state string for non-autoregressive scoring."""
        feats = self.encode_to_features(snap)
        depth = feats["depth_profile"]

        b10 = depth.get("bid_10bps", 0.0)
        a10 = depth.get("ask_10bps", 0.0)

        return (
            f"L2 Microstructure | Mid: {feats['mid_price']:.2f} | Spread: {feats['spread_bps']:.1f}bps | "
            f"Imbalance: {feats['imbalance']:+.3f} | CVD: {feats['cvd']:+.1f} | "
            f"Depth(10bps): Bid={b10:.1f}, Ask={a10:.1f}"
        )

    def encode_to_state_text_with_delta(
        self,
        curr: L2OrderBookSnapshot,
        prev: Optional[L2OrderBookSnapshot] = None,
    ) -> str:
        """Encodes snapshot concatenated with first-order delta channels for momentum sensing."""
        feats = self.encode_to_features(curr, prev)
        depth = feats["depth_profile"]
        b10 = depth.get("bid_10bps", 0.0)
        a10 = depth.get("ask_10bps", 0.0)

        d_mid = feats["delta_mid_price"]
        d_spread = feats["delta_spread_bps"]
        d_imb = feats["delta_imbalance"]
        d_cvd = feats["delta_cvd"]

        return (
            f"L2 Microstructure | Mid: {feats['mid_price']:.2f} (Δ{d_mid:+.2f}) | "
            f"Spread: {feats['spread_bps']:.1f}bps (Δ{d_spread:+.1f}) | "
            f"Imbalance: {feats['imbalance']:+.3f} (Δ{d_imb:+.3f}) | "
            f"CVD: {feats['cvd']:+.1f} (Δ{d_cvd:+.1f}) | "
            f"Depth(10bps): Bid={b10:.1f}, Ask={a10:.1f}"
        )

    def encode_to_feature_vector(
        self,
        curr: L2OrderBookSnapshot,
        prev: Optional[L2OrderBookSnapshot] = None,
    ) -> List[float]:
        """Returns parallel concatenated feature vector [S_t, Delta S_t]."""
        feats = self.encode_to_features(curr, prev)
        # Base vector S_t: [mid, spread, imbalance, cvd]
        base = [feats["mid_price"], feats["spread_bps"], feats["imbalance"], feats["cvd"]]
        # Delta vector Delta S_t: [d_mid, d_spread, d_imb, d_cvd]
        delta = [feats["delta_mid_price"], feats["delta_spread_bps"], feats["delta_imbalance"], feats["delta_cvd"]]
        return base + delta

