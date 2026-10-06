# gen-zero-model

Model-facing primitives for Gen-Zero: 0-token prefill choice head,
reflex heads, prompt sanitization, native Qwen2.5 semantic scorer,
and tri-teacher verifier.

## Architecture

- `choice_head`: Action ETF Choice Head. Non-autoregressive, 0-token
  pure-prefill choice head that projects hidden states onto Helmert regular
  simplex ETFs, and assigns vertices canonically by `ActionId`.
- `error`: crate error type (`ModelError`).
- `reflex`: `ReflexPlugin`, a contractive low-rank recurrent operator with
  named linear heads, and its hash-addressed binary archive format.
- `patch`: `ReflexPatch`, a differential patch between two same-shape
  plugins. Apply adds deltas with AVX2 when the CPU has it (scalar otherwise),
  then writes sparse exact fix-ups, and verifies the target SHA-256.
- `sanitize`: prompt and control token sanitization.
- `qwen`: native Candle Qwen2.5 transformer backbone implementation (`QwenConfig`, `QwenModel`).
- `semantic_qwen`: native zero-token semantic scoring and calibrated risk assessment (`QwenSemanticScorer`, `RiskAssessment`).
- `tri_teacher`: stage 2 tri-teacher verifier and projection heads (`TriTeacherProjector`, `TriTeacherPairDecider`).

## Key exports

- `ActionETFChoiceHead`: the 0-token prefill choice head.
- `ModelError`: crate error type.
- `ReflexPlugin`, `ReflexOperator`, `ReflexHead`, `ReflexDecision`: reflex
  plugin inference.
- `ReflexPatch`, `ReflexHeadDelta`, `ExactFixup`: differential patches.
- `contains_raw_control_marker`, `sanitize_control_tokens`: prompt sanitization.
- `QwenModel`, `QwenConfig`, `WeightFormat`: native Qwen2.5 model and configuration.
- `QwenSemanticScorer`, `RiskAssessment`, `ScoreResult`: Candle-based semantic scoring and risk gating.

## Dependencies

- `gen-zero-core`: base types and traits.
