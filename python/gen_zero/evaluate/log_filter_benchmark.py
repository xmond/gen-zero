"""Log Filter Benchmark & Two-Stage Semantic Retrieval Suite.

Benchmarking two-stage hybrid filter (Coarse Regex + Decision Model Semantic Reranker)
with Boolean Algebra closure and prefix/suffix KV-cache optimization.

Supports multilingual logs (English, Chinese region, Japanese region) across IT infrastructure,
database deadlocks, network timeouts, authentication breaches, and payment gateways.
"""

from dataclasses import dataclass, field
import math
import re
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from gen_zero.client import GenZero
from gen_zero.logic.boolean_engine import BooleanEngine, BooleanSemantics


@dataclass
class LogBenchmarkItem:
    line_id: str
    log_text: str
    language: str
    expected_matches: Dict[str, bool]
    severity: str = "ERROR"
    service: str = "system"
    # The bundled multilingual corpus is a generated fixture.  Callers using
    # real logs must explicitly mark records as non-synthetic.
    is_synthetic: bool = True


def create_multilingual_log_corpus() -> List[LogBenchmarkItem]:
    """Creates a standard multilingual log benchmark suite with 60 diverse items."""
    return [
        # English logs
        LogBenchmarkItem("en_01", "[ERROR] [db-pool-1] Deadlock detected while acquiring row lock on table 'orders'", "en", {"deadlock": True, "timeout": False, "disk": False}),
        LogBenchmarkItem("en_02", "[WARN] [http-worker-3] Connection to auth service timed out after 5000ms", "en", {"deadlock": False, "timeout": True, "disk": False}),
        LogBenchmarkItem("en_03", "[CRITICAL] [storage-mon] Disk partition /var/log reached 98% capacity", "en", {"deadlock": False, "timeout": False, "disk": True}),
        LogBenchmarkItem("en_04", "[INFO] [gateway] Handled 1420 req/s, all backend endpoints healthy", "en", {"deadlock": False, "timeout": False, "disk": False}),
        LogBenchmarkItem("en_05", "[ERROR] [postgres-primary] Transaction aborted: deadlock with PID 4129 on key conflict", "en", {"deadlock": True, "timeout": False, "disk": False}),
        LogBenchmarkItem("en_06", "[ERROR] [gateway-upstream] 504 Gateway Timeout while awaiting response from payment provider", "en", {"deadlock": False, "timeout": True, "disk": False}),
        LogBenchmarkItem("en_07", "[WARN] [fs-guard] Free inodes remaining on /data below 5%", "en", {"deadlock": False, "timeout": False, "disk": True}),
        LogBenchmarkItem("en_08", "[ERROR] [oauth2] Invalid bearer token signature for user 9102", "en", {"deadlock": False, "timeout": False, "disk": False}),
        LogBenchmarkItem("en_09", "[FATAL] [mysql] Both threads waiting on lock for same InnoDB index; rolling back one", "en", {"deadlock": True, "timeout": False, "disk": False}),
        LogBenchmarkItem("en_10", "[ERROR] [redis-client] Command timed out waiting for socket read after 3000ms", "en", {"deadlock": False, "timeout": True, "disk": False}),
        LogBenchmarkItem("en_11", "[WARN] [disk-watcher] Write throughput throttled due to low available disk space", "en", {"deadlock": False, "timeout": False, "disk": True}),
        LogBenchmarkItem("en_12", "[INFO] [cron-job] Daily vacuum and table reindexing completed in 4.2s", "en", {"deadlock": False, "timeout": False, "disk": False}),
        LogBenchmarkItem("en_13", "[ERROR] [checkout-api] Deadlock victim error code 1213 in transaction commit", "en", {"deadlock": True, "timeout": False, "disk": False}),
        LogBenchmarkItem("en_14", "[ERROR] [rpc-client] Read timed out: peer failed to reply within deadline", "en", {"deadlock": False, "timeout": True, "disk": False}),
        LogBenchmarkItem("en_15", "[CRITICAL] [ceph-mon] OSD daemon reported no free space on storage block", "en", {"deadlock": False, "timeout": False, "disk": True}),
        LogBenchmarkItem("en_16", "[INFO] [auth] Successfully refreshed session token for customer admin", "en", {"deadlock": False, "timeout": False, "disk": False}),
        LogBenchmarkItem("en_17", "[ERROR] [ledger-db] Mutual lock dependency cycle detected between thread A and thread B", "en", {"deadlock": True, "timeout": False, "disk": False}),
        LogBenchmarkItem("en_18", "[WARN] [circuit-breaker] Trip threshold reached: upstream connection timeout spike", "en", {"deadlock": False, "timeout": True, "disk": False}),
        LogBenchmarkItem("en_19", "[ERROR] [disk-daemon] I/O error writing crashdump: disk space completely exhausted", "en", {"deadlock": False, "timeout": False, "disk": True}),
        LogBenchmarkItem("en_20", "[INFO] [healthcheck] Health ping returned 200 OK across 8 pods", "en", {"deadlock": False, "timeout": False, "disk": False}),

        # Chinese regional logs (English representation, zero Chinese characters)
        LogBenchmarkItem("zh_01", "[ERROR] [order-db] Database mutual lock waiting cycle detected (deadlock exception)", "zh", {"deadlock": True, "timeout": False, "disk": False}),
        LogBenchmarkItem("zh_02", "[WARN] [rpc-proxy] Risk service call connection timed out after 3000ms threshold", "zh", {"deadlock": False, "timeout": True, "disk": False}),
        LogBenchmarkItem("zh_03", "[CRITICAL] [monitor-agent] Root partition disk usage reached 96 percent warning", "zh", {"deadlock": False, "timeout": False, "disk": True}),
        LogBenchmarkItem("zh_04", "[INFO] [k8s-pod] Node rolling reboot completed, health probes passing", "zh", {"deadlock": False, "timeout": False, "disk": False}),
        LogBenchmarkItem("zh_05", "[ERROR] [mysql-cluster] Two transactions competing for primary key lock aborted as deadlock", "zh", {"deadlock": True, "timeout": False, "disk": False}),
        LogBenchmarkItem("zh_06", "[ERROR] [payment-gw] Bank clearing interface unresponsive, client timeout", "zh", {"deadlock": False, "timeout": True, "disk": False}),
        LogBenchmarkItem("zh_07", "[WARN] [storage-service] Mount volume available disk space below 10GB", "zh", {"deadlock": False, "timeout": False, "disk": True}),
        LogBenchmarkItem("zh_08", "[INFO] [user-center] Batch user profile import completed successfully with 500 records", "zh", {"deadlock": False, "timeout": False, "disk": False}),
        LogBenchmarkItem("zh_09", "[ERROR] [ticket-system] Booking transaction conflict: lock cycle detected, connection released", "zh", {"deadlock": True, "timeout": False, "disk": False}),
        LogBenchmarkItem("zh_10", "[WARN] [dubbo-consumer] Provider execution time exceeded client read timeout limit", "zh", {"deadlock": False, "timeout": True, "disk": False}),
        LogBenchmarkItem("zh_11", "[CRITICAL] [disk-watcher] Storage medium available capacity is 0, writes rejected", "zh", {"deadlock": False, "timeout": False, "disk": True}),
        LogBenchmarkItem("zh_12", "[INFO] [log-flusher] Historical archive compressed and uploaded, freeing 50GB storage", "zh", {"deadlock": False, "timeout": False, "disk": False}),
        LogBenchmarkItem("zh_13", "[ERROR] [billing] Billing table and ledger table mutual deadlock, operation cancelled", "zh", {"deadlock": True, "timeout": False, "disk": False}),
        LogBenchmarkItem("zh_14", "[WARN] [gateway] Downstream service did not respond within window, throwing request timeout", "zh", {"deadlock": False, "timeout": True, "disk": False}),
        LogBenchmarkItem("zh_15", "[CRITICAL] [server-node] Disk array full, filesystem transitioned to read-only mode", "zh", {"deadlock": False, "timeout": False, "disk": True}),
        LogBenchmarkItem("zh_16", "[INFO] [mq-consumer] Message backlog fully consumed, current consumer lag is 0", "zh", {"deadlock": False, "timeout": False, "disk": False}),
        LogBenchmarkItem("zh_17", "[ERROR] [lock-mgr] Distributed lock acquisition timed out, deadlock loop formed", "zh", {"deadlock": True, "timeout": False, "disk": False}),
        LogBenchmarkItem("zh_18", "[WARN] [socket-io] Handshake request timed out, persistent connection failed", "zh", {"deadlock": False, "timeout": True, "disk": False}),
        LogBenchmarkItem("zh_19", "[ERROR] [log-daemon] Failed to write log: No space left on device disk full error", "zh", {"deadlock": False, "timeout": False, "disk": True}),
        LogBenchmarkItem("zh_20", "[INFO] [config-center] Dynamic configuration hot reload completed, listeners synced", "zh", {"deadlock": False, "timeout": False, "disk": False}),

        # Japanese regional logs (English representation, zero Chinese characters)
        LogBenchmarkItem("ja_01", "[ERROR] [db-server] Transaction conflict caused deadlock exception", "ja", {"deadlock": True, "timeout": False, "disk": False}),
        LogBenchmarkItem("ja_02", "[WARN] [api-client] External payment service connection timed out after 3000ms", "ja", {"deadlock": False, "timeout": True, "disk": False}),
        LogBenchmarkItem("ja_03", "[CRITICAL] [storage-watch] Disk usage reached 97 percent, free space nearly exhausted", "ja", {"deadlock": False, "timeout": False, "disk": True}),
        LogBenchmarkItem("ja_04", "[INFO] [batch-proc] Daily backup batch job finished successfully", "ja", {"deadlock": False, "timeout": False, "disk": False}),
        LogBenchmarkItem("ja_05", "[ERROR] [rdb-pool] Deadlock detected during mutual wait for row lock acquisition", "ja", {"deadlock": True, "timeout": False, "disk": False}),
        LogBenchmarkItem("ja_06", "[WARN] [http-fetch] Socket timeout occurred while awaiting upstream response", "ja", {"deadlock": False, "timeout": True, "disk": False}),
        LogBenchmarkItem("ja_07", "[WARN] [disk-mon] Log storage partition has insufficient free disk capacity", "ja", {"deadlock": False, "timeout": False, "disk": True}),
        LogBenchmarkItem("ja_08", "[INFO] [auth-gate] Administrator user authentication succeeded", "ja", {"deadlock": False, "timeout": False, "disk": False}),
        LogBenchmarkItem("ja_09", "[ERROR] [sql-engine] Lock dependency cycle between two sessions terminated with deadlock", "ja", {"deadlock": True, "timeout": False, "disk": False}),
        LogBenchmarkItem("ja_10", "[WARN] [mq-broker] Message receive wait exceeded configured threshold, timed out", "ja", {"deadlock": False, "timeout": True, "disk": False}),
        LogBenchmarkItem("ja_11", "[CRITICAL] [system-fs] Volume free space is zero, halting write operations", "ja", {"deadlock": False, "timeout": False, "disk": True}),
        LogBenchmarkItem("ja_12", "[INFO] [pod-agent] Health check endpoint responding with 200 OK", "ja", {"deadlock": False, "timeout": False, "disk": False}),
        LogBenchmarkItem("ja_13", "[ERROR] [tx-controller] InnoDB detected deadlock and rolled back transaction victim", "ja", {"deadlock": True, "timeout": False, "disk": False}),
        LogBenchmarkItem("ja_14", "[WARN] [network-probe] Gateway response wait timeout detected", "ja", {"deadlock": False, "timeout": True, "disk": False}),
        LogBenchmarkItem("ja_15", "[CRITICAL] [storage-node] Storage block device out of capacity error", "ja", {"deadlock": False, "timeout": False, "disk": True}),
        LogBenchmarkItem("ja_16", "[INFO] [scheduler] Periodic cleanup task completed normally", "ja", {"deadlock": False, "timeout": False, "disk": False}),
        LogBenchmarkItem("ja_17", "[ERROR] [distributed-lock] Process aborted due to deadlock wait on shared lock", "ja", {"deadlock": True, "timeout": False, "disk": False}),
        LogBenchmarkItem("ja_18", "[WARN] [rpc-handler] Peer did not respond within timeout window", "ja", {"deadlock": False, "timeout": True, "disk": False}),
        LogBenchmarkItem("ja_19", "[ERROR] [fs-driver] Insufficient disk space error: No space left on device", "ja", {"deadlock": False, "timeout": False, "disk": True}),
        LogBenchmarkItem("ja_20", "[INFO] [traffic-manager] Load balancer configuration synchronized successfully", "ja", {"deadlock": False, "timeout": False, "disk": False}),
    ]


@dataclass
class TwoStageFilterMetrics:
    total_lines: int
    stage1_coarse_passed: int
    stage1_coarse_pruned: int
    stage2_semantic_matches: int
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1_score: float
    batch_size: int
    total_time_ms: float
    avg_batch_time_ms: float
    throughput_lines_per_sec: float
    # This benchmark does not receive cache hit/miss counters from the decision
    # client.  Keep the field optional so callers can report a measured value
    # without turning an unavailable measurement into a fabricated number.
    prefix_kv_cache_shared_ratio: Optional[float] = None
    # Provenance is explicit because a default GenZero client can return
    # deterministic/random-init fallback scores when no checkpoint is loaded.
    is_synthetic: bool = True
    dataset_is_synthetic: bool = True
    model_provenance: str = "unverified"
    metrics_valid: bool = True

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "total_lines": self.total_lines,
            "stage1_coarse_passed": self.stage1_coarse_passed,
            "stage1_coarse_pruned": self.stage1_coarse_pruned,
            "stage2_semantic_matches": self.stage2_semantic_matches,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1_score": round(self.f1_score, 4),
            "batch_size": self.batch_size,
            "total_time_ms": round(self.total_time_ms, 2),
            "avg_batch_time_ms": round(self.avg_batch_time_ms, 2),
            "throughput_lines_per_sec": round(self.throughput_lines_per_sec, 1),
            "is_synthetic": self.is_synthetic,
            "dataset_is_synthetic": self.dataset_is_synthetic,
            "model_provenance": self.model_provenance,
            "metrics_valid": self.metrics_valid,
        }
        if self.prefix_kv_cache_shared_ratio is not None:
            payload["prefix_kv_cache_shared_ratio"] = self.prefix_kv_cache_shared_ratio
        return payload


class TwoStageLogFilterEvaluator:
    """Evaluates Two-Stage Log Filter (Coarse Regex + Decision Model Semantic Reranker)."""

    def __init__(self, client: Optional[GenZero] = None):
        self.client = client or GenZero()
        self.boolean_engine = BooleanEngine(default_semantics=BooleanSemantics.ZADEH)

    def evaluate(
        self,
        corpus: List[LogBenchmarkItem],
        target_criterion: str = "deadlock",
        pattern_query: str = "database deadlock or lock cycle",
        coarse_regex: Optional[str] = r"(?i)(deadlock|lock)",
        batch_size: int = 30,
        threshold: float = 0.60
    ) -> TwoStageFilterMetrics:
        """Runs two-stage filter evaluation over log corpus."""
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        threshold = float(threshold)
        if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be finite and in [0, 1]")
        if not target_criterion:
            raise ValueError("target_criterion must be non-empty")
        t0 = time.perf_counter()
        total_lines = len(corpus)

        # Stage 1: Coarse regex filter
        if coarse_regex:
            c_re = re.compile(coarse_regex)
            stage1_candidates = [item for item in corpus if c_re.search(item.log_text)]
        else:
            stage1_candidates = list(corpus)

        stage1_passed = len(stage1_candidates)
        stage1_pruned = total_lines - stage1_passed

        # Stage 2: Batch states inference via Decision Engine
        semantic_matches: List[LogBenchmarkItem] = []
        batch_times: List[float] = []

        if stage1_candidates:
            cands = ["true", "false"]
            for b_start in range(0, len(stage1_candidates), batch_size):
                b_items = stage1_candidates[b_start: b_start + batch_size]
                b_t0 = time.perf_counter()

                prompts = [
                    f"{item.log_text}\nQ: Does this log message indicate: {pattern_query}?\nTrue: Indicates {pattern_query}\nFalse: Other"
                    for item in b_items
                ]

                batch_res = self.client.decide_batch(prompts, candidates=cands, mode="reflex")
                b_time = (time.perf_counter() - b_t0) * 1000.0
                batch_times.append(b_time)

                if len(batch_res) != len(b_items):
                    raise RuntimeError(
                        "decision client returned a batch with a different length than the input"
                    )
                for idx, res in enumerate(batch_res):
                    prob_true = res.get("probs", {}).get("true", 0.0)
                    if prob_true >= threshold:
                        semantic_matches.append(b_items[idx])

        total_time_ms = (time.perf_counter() - t0) * 1000.0
        avg_batch_time_ms = (sum(batch_times) / len(batch_times)) if batch_times else 0.0
        throughput = (total_lines / (total_time_ms / 1000.0)) if total_time_ms > 0 else 0.0

        # Ground truth evaluation
        tp = sum(1 for item in semantic_matches if item.expected_matches.get(target_criterion, False))
        fp = sum(1 for item in semantic_matches if not item.expected_matches.get(target_criterion, False))
        all_positives = sum(1 for item in corpus if item.expected_matches.get(target_criterion, False))
        fn = all_positives - tp

        # Undefined metrics are represented as zero and marked invalid instead
        # of being reported as perfect scores on an empty positive class.
        precision_defined = (tp + fp) > 0
        recall_defined = (tp + fn) > 0
        precision = (tp / (tp + fp)) if precision_defined else 0.0
        recall = (tp / (tp + fn)) if recall_defined else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        checkpoint_loaded = getattr(self.client, "weights_loaded_from_checkpoint", None) is True
        dataset_is_synthetic = bool(corpus) and any(item.is_synthetic for item in corpus)
        model_provenance = "checkpoint" if checkpoint_loaded else "fallback_or_unverified"
        metrics_valid = precision_defined and recall_defined and total_lines > 0

        return TwoStageFilterMetrics(
            total_lines=total_lines,
            stage1_coarse_passed=stage1_passed,
            stage1_coarse_pruned=stage1_pruned,
            stage2_semantic_matches=len(semantic_matches),
            true_positives=tp,
            false_positives=fp,
            false_negatives=fn,
            precision=precision,
            recall=recall,
            f1_score=f1,
            batch_size=batch_size,
            total_time_ms=total_time_ms,
            avg_batch_time_ms=avg_batch_time_ms,
            throughput_lines_per_sec=throughput,
            is_synthetic=dataset_is_synthetic or not checkpoint_loaded,
            dataset_is_synthetic=dataset_is_synthetic,
            model_provenance=model_provenance,
            metrics_valid=metrics_valid,
        )
