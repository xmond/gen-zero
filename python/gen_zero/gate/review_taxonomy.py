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
        "condition",       # condition-handling error
        "state",           # state read/write retention error
        "dataFlow",        # data-flow transformation/propagation error
        "asyncControl",    # async ordering / exception-handling error
        "other",
        NO_ISSUE,
    ],
    RiskDimension.SECURITY.value: [
        "authorization",   # weakened authorization/trust boundary
        "injection",       # untrusted input injection
        "exposure",        # sensitive information exposure
        "unsafeDefault",   # unsafe default configuration
        "other",
        NO_ISSUE,
    ],
    RiskDimension.RELIABILITY.value: [
        "cleanup",         # resource leak / side effect not cleaned up
        "concurrency",     # concurrency race / deadlock
        "recovery",        # failed disaster recovery / failover
        "crash",           # unexpected uncaught exception / crash
        "other",
        NO_ISSUE,
    ],
    RiskDimension.COMPATIBILITY.value: [
        "api",             # breaking change to a public interface
        "behavior",        # observable behavior break for existing callers
        "dataFormat",      # incompatible persistence/exchange format
        "protocol",        # broken external contract/protocol
        "other",
        NO_ISSUE,
    ],
    RiskDimension.TEST_GAP.value: [
        "branch",          # important branch lacks coverage
        "failure",         # failure/cancellation path lacks tests
        "boundary",        # boundary/extreme values untested
        "integration",     # inter-module interaction untested
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
