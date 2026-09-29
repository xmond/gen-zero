"""Standardized Taxonomy Tree and Enums for Semantic Code Review Gate.

Implements Module 4 of Issue #22:
- Standardized finite mechanism taxonomies for correctness, security, reliability, compatibility, testGap.
- Sentinels: noMatch and noIssue for explicit anti-hallucination gating.
- ChangeProfile, ReviewAction, and OwnerDomain enums.
"""

from typing import Dict, List
import enum


class RiskDimension(str, enum.Enum):
    CORRECTNESS = "correctness"
    SECURITY = "security"
    RELIABILITY = "reliability"
    COMPATIBILITY = "compatibility"
    TEST_GAP = "testGap"


class ChangeProfile(str, enum.Enum):
    BEHAVIOR = "behavior"
    INTERFACE = "interface"
    INFRA = "infra"
    OBSERVABILITY = "observability"
    REFACTOR = "refactor"
    ROUTINE = "routine"


class ReviewAction(str, enum.Enum):
    REQUEST_CHANGES = "request_changes"
    COMMENT = "comment"
    APPROVE = "approve"


class OwnerDomain(str, enum.Enum):
    SECURITY = "security"
    API = "api"
    RUNTIME = "runtime"
    TESTING = "testing"
    GENERAL = "general"


# Explicit rejection sentinels
NO_MATCH: str = "noMatch"
NO_ISSUE: str = "noIssue"

# Structured Mechanism Taxonomies strictly adhering to Issue #22 table
MECHANISM_TAXONOMY: Dict[str, List[str]] = {
    RiskDimension.CORRECTNESS.value: [
        "condition",       # 条件处理错误
        "state",           # 状态读写保留错误
        "dataFlow",        # 数据流转换传递错误
        "asyncControl",    # 异步乱序/异常处理错误
        "other",
        NO_ISSUE,
    ],
    RiskDimension.SECURITY.value: [
        "authorization",   # 权限/信任边界削弱
        "injection",       # 不可信输入注入
        "exposure",        # 敏感信息泄露
        "unsafeDefault",   # 不安全默认配置
        "other",
        NO_ISSUE,
    ],
    RiskDimension.RELIABILITY.value: [
        "cleanup",         # 资源泄露/副作用未清理
        "concurrency",     # 并发竞争/死锁
        "recovery",        # 容灾/故障恢复失效
        "crash",           # 非预期抛错异常崩溃
        "other",
        NO_ISSUE,
    ],
    RiskDimension.COMPATIBILITY.value: [
        "api",             # 公开接口不兼容
        "behavior",        # 现有调用者观测行为破坏
        "dataFormat",      # 持久化/交换格式不兼容
        "protocol",        # 外部契约协议破坏
        "other",
        NO_ISSUE,
    ],
    RiskDimension.TEST_GAP.value: [
        "branch",          # 重要分支缺乏覆盖
        "failure",         # 失败/取消路径缺乏测试
        "boundary",        # 边界/极值未测
        "integration",     # 模块间交互未测
        "other",
        NO_ISSUE,
    ],
}


def map_dimension_to_owner(dimension: str, mechanism: str) -> OwnerDomain:
    """Maps a risk dimension and specific mechanism to the specialized owner domain."""
    if dimension == RiskDimension.SECURITY.value:
        return OwnerDomain.SECURITY
    elif dimension == RiskDimension.COMPATIBILITY.value or mechanism in ("api", "protocol"):
        return OwnerDomain.API
    elif dimension == RiskDimension.TEST_GAP.value:
        return OwnerDomain.TESTING
    elif dimension in (RiskDimension.RELIABILITY.value, RiskDimension.CORRECTNESS.value):
        return OwnerDomain.RUNTIME
    return OwnerDomain.GENERAL
