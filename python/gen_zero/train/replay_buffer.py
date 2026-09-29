"""Gen-Zero Layer 4: Stability Experience Replay Buffer (E09 Protocol).

Implements the 1:3 hard-to-gold mixing ratio:
- 1 part online-mined hard failure/uncertainty samples
- 3 parts baseline gold anchor samples
Prevents catastrophic representation drift and anchors global geometry.
"""

import random
from typing import List, Dict, Any, Optional, Callable
from ..rollout.hard_miner import MinedSample
from ..model.dual_head import encode_leaf_tokens


class StabilityReplayBuffer:
    """1:3 Stability Replay Buffer preventing catastrophic forgetting."""
    def __init__(
        self,
        capacity: int = 50000,
        hard_ratio: float = 0.25,
        state_preparer: Optional[Callable[[Any], Any]] = None
    ):
        self.capacity = capacity
        self.hard_ratio = hard_ratio
        self.state_preparer = state_preparer
        self.gold_buffer: List[Dict[str, Any]] = []
        self.hard_buffer: List[MinedSample] = []

    def load_gold_samples(self, gold_samples: List[Dict[str, Any]]):
        """Loads baseline gold anchor dataset."""
        self.gold_buffer.extend(gold_samples)
        if len(self.gold_buffer) > self.capacity:
            self.gold_buffer = self.gold_buffer[-self.capacity:]

    def add_mined_samples(self, samples: List[MinedSample]):
        """Adds online-mined hard learning samples."""
        self.hard_buffer.extend(samples)
        if len(self.hard_buffer) > self.capacity:
            self.hard_buffer = self.hard_buffer[-self.capacity:]

    def add_dagger_samples(self, samples: List[Any]):
        """Adds DAgger expert-relabeled recovery samples into the hard buffer."""
        for s in samples:
            if hasattr(s, "to_dict"):
                item = s.to_dict()
            elif isinstance(s, dict):
                item = dict(s)
            else:
                continue
            item["is_dagger_relabeled"] = True
            item["is_hard_sample"] = True
            self.hard_buffer.append(item)
        if len(self.hard_buffer) > self.capacity:
            self.hard_buffer = self.hard_buffer[-self.capacity:]

    def pop_recent_mined_samples(self, count: int):
        """Purges the most recently added mined samples on gate rejection."""
        if count > 0 and self.hard_buffer:
            self.hard_buffer = self.hard_buffer[:-count]

    def sample_batch(self, batch_size: int = 16) -> List[Dict[str, Any]]:
        """Samples a batch strictly maintaining the 1:3 mixing discipline."""
        if not self.gold_buffer and not self.hard_buffer:
            return []

        total_available = len(self.gold_buffer) + len(self.hard_buffer)
        if total_available == 0:
            return []

        actual_batch_size = min(batch_size, total_available)
        n_hard = int(round(actual_batch_size * self.hard_ratio))
        n_gold = actual_batch_size - n_hard

        if len(self.hard_buffer) < n_hard:
            n_hard = len(self.hard_buffer)
            n_gold = min(len(self.gold_buffer), actual_batch_size - n_hard)
        if len(self.gold_buffer) < n_gold:
            n_gold = len(self.gold_buffer)
            n_hard = min(len(self.hard_buffer), actual_batch_size - n_gold)

        if n_hard == 0 and n_gold == 0 and total_available > 0:
            if len(self.hard_buffer) > 0:
                n_hard = 1
            else:
                n_gold = 1

        sampled_hard = random.sample(self.hard_buffer, n_hard) if n_hard > 0 else []
        sampled_gold = random.sample(self.gold_buffer, n_gold) if n_gold > 0 else []

        def _unwrap_and_prepare(raw: Any) -> Any:
            # Unwrap single-key container {"state": ...} to match online ingestion (Probe R10_I02)
            val = raw
            if isinstance(val, dict) and len(val) == 1 and "state" in val:
                val = val["state"]
            if self.state_preparer is not None:
                return self.state_preparer(val)
            return val

        batch = []
        # Convert mined and dagger samples to standard decision example format
        for s in sampled_hard:
            if isinstance(s, dict):
                raw_state = s.get("state", "")
                norm_state = _unwrap_and_prepare(raw_state)
                cands = s.get("candidate_ids", [])
                leaf_toks = s.get("leaf_tokens")
                if not leaf_toks:
                    leaf_toks = [encode_leaf_tokens(norm_state, act) for act in cands]
                item = dict(s)
                item["state"] = norm_state
                item["leaf_tokens"] = leaf_toks
                item.setdefault("is_hard_sample", True)
                batch.append(item)
            elif hasattr(s, "to_dict"):
                item = s.to_dict()
                raw_state = item.get("state", "")
                norm_state = _unwrap_and_prepare(raw_state)
                cands = item.get("candidate_ids", [])
                leaf_toks = item.get("leaf_tokens")
                if not leaf_toks:
                    leaf_toks = [encode_leaf_tokens(norm_state, act) for act in cands]
                item["state"] = norm_state
                item["leaf_tokens"] = leaf_toks
                item.setdefault("is_hard_sample", True)
                batch.append(item)
            else:
                norm_state = _unwrap_and_prepare(s.state_data)
                # Unified token sequence from state_id and candidate_ids if not already present
                leaf_toks = getattr(s, "leaf_tokens", None)
                if not leaf_toks:
                    leaf_toks = [encode_leaf_tokens(norm_state, act) for act in s.candidate_ids]

                batch.append({
                    "id": s.state_id,
                    "type": "choice",
                    "state": norm_state,
                    "candidate_ids": s.candidate_ids,
                    "pi_target": s.pi_target,
                    "soft_target": s.pi_target,
                    "value_target": s.value_target,
                    "leaf_tokens": leaf_toks,
                    "is_hard_sample": True,
                    "is_causal_culprit": getattr(s, "is_causal_culprit", False),
                    "ite": getattr(s, "individual_treatment_effect", 0.0),
                    "best_counterfactual_action": getattr(s, "best_counterfactual_action", None),
                    "environment_id": getattr(s, "environment_id", "env_hard_mined")
                })


        for g in sampled_gold:
            item = dict(g)
            norm_state = _unwrap_and_prepare(item.get("state", ""))
            item["state"] = norm_state
            if "candidate_ids" in item and "leaf_tokens" not in item:
                c_desc = item.get("candidate_descriptions") or {}
                item["leaf_tokens"] = [
                    encode_leaf_tokens(norm_state, c, c_desc.get(c, "") if isinstance(c_desc, dict) else str(c))
                    for c in item["candidate_ids"]
                ]
            item["is_hard_sample"] = False
            item.setdefault("is_causal_culprit", False)
            item.setdefault("ite", 0.0)
            item.setdefault("environment_id", "env_gold_anchor")
            batch.append(item)

        random.shuffle(batch)
        return batch

    def add_synthetic_samples(self, synthetic_samples: List[Dict[str, Any]]):
        """Adds PRM-verified counterfactual synthetic samples to gold/anchor buffer."""
        for s in synthetic_samples:
            item = dict(s)
            item["is_synthetic"] = True
            item.setdefault("environment_id", "env_synthetic_counterfactual")
            self.gold_buffer.append(item)
        if len(self.gold_buffer) > self.capacity:
            self.gold_buffer = self.gold_buffer[-self.capacity:]

    def load_browser_dx_dataset(self, file_path: str) -> int:
        """Loads and converts real browser DX interaction dataset into gold anchor training samples."""
        import json
        import hashlib
        count = 0
        vocab_size = 19999
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                for idx, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    item = json.loads(line)
                    state = item.get("state", "")
                    questions = item.get("questions", {})
                    target_q = questions.get("target_element") or (next(iter(questions.values())) if questions else {})
                    criteria = target_q.get("criteria", {})
                    ground_truth = item.get("ground_truth_ref") or item.get("answer", {}).get("target_element", {}).get("choice")

                    if isinstance(criteria, dict):
                        cands = list(criteria.keys())
                    elif isinstance(criteria, list):
                        cands = criteria
                    else:
                        cands = [ground_truth] if ground_truth else []

                    if not cands or not ground_truth:
                        continue

                    # Soft target distribution with slight label smoothing
                    pi_target = {}
                    eps = 0.05 / max(1, len(cands) - 1)
                    for c in cands:
                        pi_target[c] = (1.0 - 0.05) if c == ground_truth else eps

                    sample = {
                        "id": f"browser_dx_{idx}_{item.get('site', 'web')}_{ground_truth}",
                        "type": "choice",
                        "state": state,
                        "candidate_ids": cands,
                        "candidate_descriptions": criteria if isinstance(criteria, dict) else None,
                        "pi_target": pi_target,
                        "soft_target": pi_target,
                        "value_target": 1.0,
                        "is_hard_sample": False,
                        "environment_id": f"browser_{item.get('site', 'web')}"
                    }
                    self.gold_buffer.append(sample)
                    count += 1
            if len(self.gold_buffer) > self.capacity:
                self.gold_buffer = self.gold_buffer[-self.capacity:]
        except Exception:
            pass
        return count

    def update_retention_feedback(self, retention_rate: float) -> float:
        """Dynamically adjusts hard ratio based on anchor retention rate."""
        from ..adaptive_engine import AdaptiveParameterEngine
        self.hard_ratio = AdaptiveParameterEngine.adjust_replay_hard_ratio(
            current_retention_rate=retention_rate,
            current_hard_ratio=self.hard_ratio
        )
        return self.hard_ratio

    def __len__(self) -> int:
        return len(self.gold_buffer) + len(self.hard_buffer)

    @property
    def stats(self) -> Dict[str, Any]:
        dagger_count = sum(1 for s in self.hard_buffer if (isinstance(s, dict) and s.get("is_dagger_relabeled")) or getattr(s, "is_dagger_relabeled", False))
        return {
            "gold_samples": len(self.gold_buffer),
            "hard_samples": len(self.hard_buffer),
            "dagger_samples": dagger_count,
            "total_samples": len(self),
            "hard_ratio": self.hard_ratio
        }



class IngestionReceipt:
    """Receipt tracking an ingestion batch for ID-based rollback and provenance auditing."""
    def __init__(
        self,
        receipt_id: str,
        domain: str,
        partition: str,
        sample_ids: List[str],
        timestamp: float
    ):
        self.receipt_id = receipt_id
        self.domain = domain
        self.partition = partition
        self.sample_ids = list(sample_ids)
        self.count = len(sample_ids)
        self.timestamp = timestamp

    def to_dict(self) -> Dict[str, Any]:
        return {
            "receipt_id": self.receipt_id,
            "domain": self.domain,
            "partition": self.partition,
            "count": self.count,
            "sample_ids": self.sample_ids,
            "timestamp": self.timestamp
        }


class SampledBatch:
    """Batch container for strictly ratio-enforced training."""
    def __init__(
        self,
        ready: bool,
        samples: List[Dict[str, Any]],
        m: int,
        n_hard: int,
        n_gold: int,
        domain: str
    ):
        self.ready = ready
        self.samples = samples
        self.m = m
        self.n_hard = n_hard
        self.n_gold = n_gold
        self.domain = domain

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self):
        return iter(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ready": self.ready,
            "count": len(self.samples),
            "m": self.m,
            "n_hard": self.n_hard,
            "n_gold": self.n_gold,
            "domain": self.domain
        }


class DomainExperienceBuffer(StabilityReplayBuffer):
    """Specialized experience buffer with domain segregation, physical ownership,
    strict 1:3 without-replacement math, and receipt-based rollback.
    """
    def __init__(
        self,
        domain: str,
        capacity: int = 50000,
        hard_ratio: float = 0.25,
        state_preparer: Optional[Callable[[Any], Any]] = None,
        seed: int = 42
    ):
        super().__init__(capacity=capacity, hard_ratio=hard_ratio, state_preparer=state_preparer)
        self.domain = domain
        self.rng = random.Random(seed)
        self._ingestion_records: Dict[str, IngestionReceipt] = {}

    def append(
        self,
        records: List[Dict[str, Any]],
        *,
        partition: str = "gold",
        ingestion_id: Optional[str] = None
    ) -> IngestionReceipt:
        """Appends validated records to the specified domain partition."""
        import uuid
        import copy
        import time

        rec_id = ingestion_id or f"ingest_{self.domain}_{uuid.uuid4().hex[:8]}"
        added_ids = []

        if partition not in ("gold", "hard"):
            raise ValueError(f"Partition must be 'gold' or 'hard', got '{partition}'")

        target_buf = self.gold_buffer if partition == "gold" else self.hard_buffer

        for i, item in enumerate(records):
            rec = copy.deepcopy(item)
            self._validate_record(rec)
            s_id = str(rec.get("id") or rec.get("sample_id") or f"{self.domain}_{partition}_{uuid.uuid4().hex[:8]}")
            rec["id"] = s_id
            rec["sample_id"] = s_id
            rec["domain"] = self.domain
            rec["partition"] = partition
            rec["ingestion_id"] = rec_id
            target_buf.append(rec)
            added_ids.append(s_id)

        # Enforce capacity
        if len(target_buf) > self.capacity:
            excess = len(target_buf) - self.capacity
            if partition == "gold":
                self.gold_buffer = self.gold_buffer[excess:]
            else:
                self.hard_buffer = self.hard_buffer[excess:]

        receipt = IngestionReceipt(
            receipt_id=rec_id,
            domain=self.domain,
            partition=partition,
            sample_ids=added_ids,
            timestamp=time.time()
        )
        self._ingestion_records[rec_id] = receipt
        return receipt

    def rollback_ingestion(self, receipt: IngestionReceipt) -> int:
        """Rolls back the exact samples associated with the given ingestion receipt."""
        if receipt.domain != self.domain:
            raise ValueError(f"Cannot rollback receipt for domain '{receipt.domain}' in buffer '{self.domain}'")

        ids_to_remove = set(receipt.sample_ids)
        before_count = len(self)

        if receipt.partition == "gold":
            self.gold_buffer = [s for s in self.gold_buffer if (s.get("id") or s.get("sample_id")) not in ids_to_remove]
        else:
            self.hard_buffer = [
                s for s in self.hard_buffer 
                if (getattr(s, "sample_id", None) or getattr(s, "state_id", None) or (s.get("id") if isinstance(s, dict) else None)) not in ids_to_remove
            ]

        if receipt.receipt_id in self._ingestion_records:
            del self._ingestion_records[receipt.receipt_id]

        removed = before_count - len(self)
        return removed

    def sample_training_batch(self, batch_size: int = 16, strict_ratio: bool = True) -> SampledBatch:
        """Samples a training batch maintaining strict 1:3 without-replacement geometry:
        m = min(B / 4, |H_D|, floor(|G_D| / 3)), n_H = m, n_G = 3m.
        """
        if strict_ratio and (batch_size <= 0 or batch_size % 4 != 0):
            raise ValueError(f"Strict ratio requires positive batch_size divisible by 4, got {batch_size}")

        len_h = len(self.hard_buffer)
        len_g = len(self.gold_buffer)

        m = min(batch_size // 4, len_h, len_g // 3)
        if m == 0:
            return SampledBatch(ready=False, samples=[], m=0, n_hard=0, n_gold=0, domain=self.domain)

        n_h = m
        n_g = 3 * m

        sampled_hard_raw = self.rng.sample(self.hard_buffer, n_h)
        sampled_gold_raw = self.rng.sample(self.gold_buffer, n_g)

        # Standardize samples
        batch = []
        for s in sampled_hard_raw:
            if isinstance(s, MinedSample):
                if hasattr(s, "to_dict"):
                    batch.append(s.to_dict())
                else:
                    batch.append({
                        "id": getattr(s, "state_id", "mined"),
                        "sample_id": getattr(s, "state_id", "mined"),
                        "type": "choice",
                        "state": getattr(s, "state_data", ""),
                        "candidate_ids": getattr(s, "candidate_ids", []),
                        "pi_target": getattr(s, "pi_target", {}),
                        "soft_target": getattr(s, "pi_target", {}),
                        "value_target": getattr(s, "value_target", 0.0),
                        "mining_reason": getattr(s, "mining_reason", "uncertainty"),
                        "entropy": getattr(s, "entropy", 0.0),
                        "td_error": getattr(s, "td_error", 0.0),
                        "is_hard_sample": True,
                        "domain": self.domain
                    })
            else:
                batch.append(dict(s))
        for s in sampled_gold_raw:
            batch.append(dict(s))

        self.rng.shuffle(batch)
        return SampledBatch(ready=True, samples=batch, m=m, n_hard=n_h, n_gold=n_g, domain=self.domain)

    def _validate_record(self, record: Dict[str, Any]):
        """Domain-specific record validator hook."""
        pass


class BrowserExperienceBuffer(DomainExperienceBuffer):
    """Specialized Experience Buffer for Browser / DOM interaction tasks."""
    def __init__(self, capacity: int = 50000, seed: int = 42):
        super().__init__(domain="browser", capacity=capacity, hard_ratio=0.25, seed=seed)

    def _validate_record(self, record: Dict[str, Any]):
        if record.get("domain") and record["domain"] != "browser":
            raise ValueError(f"BrowserExperienceBuffer rejected record from domain '{record.get('domain')}'")
        if "state" not in record and "leaf_tokens" not in record:
            raise ValueError("Browser record must contain 'state' or 'leaf_tokens'")


class VisionExperienceBuffer(DomainExperienceBuffer):
    """Specialized Experience Buffer for Vision / Continuous Latent interaction tasks."""
    def __init__(self, capacity: int = 50000, seed: int = 42):
        super().__init__(domain="vision", capacity=capacity, hard_ratio=0.25, seed=seed)

    def _validate_record(self, record: Dict[str, Any]):
        if record.get("domain") and record["domain"] != "vision":
            raise ValueError(f"VisionExperienceBuffer rejected record from domain '{record.get('domain')}'")
        # Ensure numeric latent vectors are preserved and not mangled
        if "state_vec" in record:
            import numpy as np
            vec = record["state_vec"]
            if not isinstance(vec, (list, tuple, np.ndarray)):
                raise ValueError("Vision record 'state_vec' must be a numeric list or numpy array")

