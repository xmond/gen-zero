"""Gen-Zero Layer 1: Core Perception & Dual-Head Network."""

from .dual_head import (
    GenZeroDualHeadModel,
    DeepSetAttentionHead,
    GlobalStateValueHead,
    AbstainModule,
    compute_normalized_attention_entropy,
    TwoDimensionalCredibilityGate,
    CredibilityVerdict,
)
from .delta_encoder import ZeroStateDeltaEncoder
from .market_state import (
    L2OrderBookSnapshot,
    L2MarketStateEncoder,
)
from .prefix_cache import GenZeroPrefixCacheEngine
from .option_isolation import (
    OptionIsolationEngine,
    build_option_isolation_mask,
    tokenize_option_isolation_sequence
)
from .sanitization import (
    escape_control_tokens,
    unescape_control_tokens,
    is_boundary_forgery_attempt,
    audit_control_tokens,
    sanitize_input_text,
    sanitize_candidates,
    sanitize_state,
    BoundaryForgerySanitizer
)
from .prefix_tree_attention import (
    SINGLE_BRANCH_CONTEXT_LIMIT,
    AGGREGATED_PACKAGE_LIMIT,
    validate_context_quotas,
    build_prefix_tree_attention_mask,
    PrefixTreePackedLayout,
)
from .token_stability import (
    assert_single_token_stability,
    TokenFragmentationError,
    CanonicalLabelMapper,
)
from .invertible_encoder import (
    InvertibleVectorEncoder,
    InvertibleVectorOutput,
    InvertibleAdapter,
)
from .latent_updater import (
    LatentUpdater,
    LatentUpdaterTelemetry,
)

__all__ = [
    "assert_single_token_stability",
    "TokenFragmentationError",
    "CanonicalLabelMapper",
    "GenZeroDualHeadModel",
    "DeepSetAttentionHead",
    "GlobalStateValueHead",
    "AbstainModule",
    "compute_normalized_attention_entropy",
    "TwoDimensionalCredibilityGate",
    "CredibilityVerdict",
    "ZeroStateDeltaEncoder",
    "L2OrderBookSnapshot",
    "L2MarketStateEncoder",
    "GenZeroPrefixCacheEngine",
    "OptionIsolationEngine",
    "build_option_isolation_mask",
    "tokenize_option_isolation_sequence",
    "escape_control_tokens",
    "unescape_control_tokens",
    "is_boundary_forgery_attempt",
    "audit_control_tokens",
    "sanitize_input_text",
    "sanitize_candidates",
    "sanitize_state",
    "BoundaryForgerySanitizer",
    "SINGLE_BRANCH_CONTEXT_LIMIT",
    "AGGREGATED_PACKAGE_LIMIT",
    "validate_context_quotas",
    "build_prefix_tree_attention_mask",
    "PrefixTreePackedLayout",
    "InvertibleVectorEncoder",
    "InvertibleVectorOutput",
    "InvertibleAdapter",
    "LatentUpdater",
    "LatentUpdaterTelemetry",
]


