"""Parallel Suffix KV-Cache Log Filter Benchmark Suite (Issue #9).

Evaluates two-stage semantic log filtering:
Stage 1: High-speed regex / keyword coarse filter (optional fast pruning)
Stage 2: Semantic Noul truth reranker via batch states decision inference
         with Boolean Algebra closure and prefix/suffix KV-cache optimization.

Supports multilingual logs (English, Chinese, Japanese) across IT infrastructure,
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

        # Chinese logs
        LogBenchmarkItem("zh_01", "[ERROR] [order-db] 数据库产生相互锁等待，事务自动回滚（死锁异常）", "zh", {"deadlock": True, "timeout": False, "disk": False}),
        LogBenchmarkItem("zh_02", "[WARN] [rpc-proxy] 调用风控微服务连接超时，超过配置上限 3000ms", "zh", {"deadlock": False, "timeout": True, "disk": False}),
        LogBenchmarkItem("zh_03", "[CRITICAL] [monitor-agent] 根分区 / 磁盘使用率达到 96%，触发一级告警", "zh", {"deadlock": False, "timeout": False, "disk": True}),
        LogBenchmarkItem("zh_04", "[INFO] [k8s-pod] 节点平滑重启完毕，健康探针检测通过", "zh", {"deadlock": False, "timeout": False, "disk": False}),
        LogBenchmarkItem("zh_05", "[ERROR] [mysql-cluster] 两个事务竞争同一主键锁资源，判定为死锁被终止", "zh", {"deadlock": True, "timeout": False, "disk": False}),
        LogBenchmarkItem("zh_06", "[ERROR] [payment-gw] 银行清算接口无响应，客户端等待超时", "zh", {"deadlock": False, "timeout": True, "disk": False}),
        LogBenchmarkItem("zh_07", "[WARN] [storage-service] 挂载卷可用空间剩余不足 10GB，请及时清理", "zh", {"deadlock": False, "timeout": False, "disk": True}),
        LogBenchmarkItem("zh_08", "[INFO] [user-center] 批量导入用户信息成功，共计 500 条数据", "zh", {"deadlock": False, "timeout": False, "disk": False}),
        LogBenchmarkItem("zh_09", "[ERROR] [ticket-system] 订票事务冲突，数据库检测到锁环路已释放连接", "zh", {"deadlock": True, "timeout": False, "disk": False}),
        LogBenchmarkItem("zh_10", "[WARN] [dubbo-consumer] 服务端处理耗时过长，触发客户端读超时", "zh", {"deadlock": False, "timeout": True, "disk": False}),
        LogBenchmarkItem("zh_11", "[CRITICAL] [disk-watcher] 存储介质剩余容量为 0，写操作被系统拒绝", "zh", {"deadlock": False, "timeout": False, "disk": True}),
        LogBenchmarkItem("zh_12", "[INFO] [log-flusher] 历史归档日志压缩传输完毕，释放 50GB 存储", "zh", {"deadlock": False, "timeout": False, "disk": False}),
        LogBenchmarkItem("zh_13", "[ERROR] [billing] 扣款表与流水表产生交叉死锁，当前操作已被取消", "zh", {"deadlock": True, "timeout": False, "disk": False}),
        LogBenchmarkItem("zh_14", "[WARN] [gateway] 下游服务未在规定窗口内返回，抛出请求超时异常", "zh", {"deadlock": False, "timeout": True, "disk": False}),
        LogBenchmarkItem("zh_15", "[CRITICAL] [server-node] 磁盘阵列空间写满，系统已自动转入只读模式", "zh", {"deadlock": False, "timeout": False, "disk": True}),
        LogBenchmarkItem("zh_16", "[INFO] [mq-consumer] 消息积压消费完成，当前 lag 为 0", "zh", {"deadlock": False, "timeout": False, "disk": False}),
        LogBenchmarkItem("zh_17", "[ERROR] [lock-mgr] 无法获取分布式悲观锁，互斥等待超时并形成死锁", "zh", {"deadlock": True, "timeout": False, "disk": False}),
        LogBenchmarkItem("zh_18", "[WARN] [socket-io] 握手请求超时未能建立长连接", "zh", {"deadlock": False, "timeout": True, "disk": False}),
        LogBenchmarkItem("zh_19", "[ERROR] [log-daemon] 日志文件无法继续写入：No space left on device 磁盘写满", "zh", {"deadlock": False, "timeout": False, "disk": True}),
        LogBenchmarkItem("zh_20", "[INFO] [config-center] 动态配置热重载成功，监听器已同步更新", "zh", {"deadlock": False, "timeout": False, "disk": False}),

        # Japanese logs
        LogBenchmarkItem("ja_01", "[ERROR] [db-server] トランザクション競合によりデッドロックが発生しました", "ja", {"deadlock": True, "timeout": False, "disk": False}),
        LogBenchmarkItem("ja_02", "[WARN] [api-client] 外部決済サービスへの接続がタイムアウトしました (3000ms)", "ja", {"deadlock": False, "timeout": True, "disk": False}),
        LogBenchmarkItem("ja_03", "[CRITICAL] [storage-watch] ディスク使用率が97%に達し、空き容量が枯渇寸前です", "ja", {"deadlock": False, "timeout": False, "disk": True}),
        LogBenchmarkItem("ja_04", "[INFO] [batch-proc] 日次バックアップバッチが正常に完了しました", "ja", {"deadlock": False, "timeout": False, "disk": False}),
        LogBenchmarkItem("ja_05", "[ERROR] [rdb-pool] 行ロック取得の相互待機によるデッドロックを検知しました", "ja", {"deadlock": True, "timeout": False, "disk": False}),
        LogBenchmarkItem("ja_06", "[WARN] [http-fetch] レスポンス待機中にソケットタイムアウトが発生しました", "ja", {"deadlock": False, "timeout": True, "disk": False}),
        LogBenchmarkItem("ja_07", "[WARN] [disk-mon] ログ保存先パーティションの空きディスク容量が不足しています", "ja", {"deadlock": False, "timeout": False, "disk": True}),
        LogBenchmarkItem("ja_08", "[INFO] [auth-gate] 管理者ユーザーのログインに成功しました", "ja", {"deadlock": False, "timeout": False, "disk": False}),
        LogBenchmarkItem("ja_09", "[ERROR] [sql-engine] 2つのセッション間でロック循環が発生し、デッドロックで終了", "ja", {"deadlock": True, "timeout": False, "disk": False}),
        LogBenchmarkItem("ja_10", "[WARN] [mq-broker] メッセージ受信待機が設定上限時間を超えてタイムアウトしました", "ja", {"deadlock": False, "timeout": True, "disk": False}),
        LogBenchmarkItem("ja_11", "[CRITICAL] [system-fs] ドライブの空き容量がゼロになりました。書き込みを停止します", "ja", {"deadlock": False, "timeout": False, "disk": True}),
        LogBenchmarkItem("ja_12", "[INFO] [pod-agent] ヘルスチェックエンドポイントは正常に応答しています", "ja", {"deadlock": False, "timeout": False, "disk": False}),
        LogBenchmarkItem("ja_13", "[ERROR] [tx-controller] InnoDBがデッドロックを検出し、一方のトランザクションをロールバック", "ja", {"deadlock": True, "timeout": False, "disk": False}),
        LogBenchmarkItem("ja_14", "[WARN] [network-probe] ゲートウェイ応答待機タイムアウトを検知しました", "ja", {"deadlock": False, "timeout": True, "disk": False}),
        LogBenchmarkItem("ja_15", "[CRITICAL] [storage-node] ストレージブロックデバイスの容量が不足しています", "ja", {"deadlock": False, "timeout": False, "disk": True}),
        LogBenchmarkItem("ja_16", "[INFO] [scheduler] 定期クリーンアップタスクが正常終了しました", "ja", {"deadlock": False, "timeout": False, "disk": False}),
        LogBenchmarkItem("ja_17", "[ERROR] [distributed-lock] 共有リソースのデッドロック待機によりプロセスを停止", "ja", {"deadlock": True, "timeout": False, "disk": False}),
        LogBenchmarkItem("ja_18", "[WARN] [rpc-handler] 通信相手からの応答がなく接続タイムアウトとなりました", "ja", {"deadlock": False, "timeout": True, "disk": False}),
        LogBenchmarkItem("ja_19", "[ERROR] [fs-driver] ディスク領域不足エラー (No space left on device)", "ja", {"deadlock": False, "timeout": False, "disk": True}),
        LogBenchmarkItem("ja_20", "[INFO] [traffic-manager] ロードバランサーの接続設定が正常に同期されました", "ja", {"deadlock": False, "timeout": False, "disk": False}),
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
        coarse_regex: Optional[str] = r"(?i)(deadlock|lock|死锁|デッドロック)",
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
