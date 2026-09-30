# gen-zero-model

Model-facing primitives for Gen-Zero: permutation-equivariant set-attention
masking, shared position IDs, a 0-token prefill choice head, and prompt
sanitization.

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
- `mask`: block-causal attention masking semantics. Enforces strict
  mathematical isolation across candidate options: the prefix does causal or
  bidirectional self-attention, and each option attends to the prefix plus
  itself, strictly isolated from other options.
- `sanitize`: prompt and control token sanitization.

## Key exports

- `ActionETFChoiceHead`: the 0-token prefill choice head.
- `ModelError`: crate error type.
- `ReflexPlugin`, `ReflexOperator`, `ReflexHead`, `ReflexDecision`: reflex
  plugin inference.
- `ReflexPatch`, `ReflexHeadDelta`, `ExactFixup`: differential patches.
- `generate_shared_position_ids`, `BlockCausalMask`, `PrefixMode`: block-causal
  masking and shared position ID generation.
- `contains_raw_control_marker`, `sanitize_control_tokens`: prompt sanitization.

## Dependencies

- `gen-zero-core`: base types and traits.
