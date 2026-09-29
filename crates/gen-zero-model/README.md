# gen-zero-model

Model-facing primitives for Gen-Zero: permutation-equivariant set-attention
masking, shared position IDs, a 0-token prefill choice head, and prompt
sanitization.

## Architecture

- `choice_head`: Action ETF Choice Head. Non-autoregressive, 0-token
  pure-prefill choice head that projects hidden states onto Helmert regular
  simplex ETFs, and assigns vertices canonically by `ActionId`.
- `error`: crate error type (`ModelError`).
- `mask`: block-causal attention masking semantics. Enforces strict
  mathematical isolation across candidate options: the prefix does causal or
  bidirectional self-attention, and each option attends to the prefix plus
  itself, strictly isolated from other options.
- `sanitize`: prompt and control token sanitization.

## Key exports

- `ActionETFChoiceHead`: the 0-token prefill choice head.
- `ModelError`: crate error type.
- `generate_shared_position_ids`, `BlockCausalMask`, `PrefixMode`: block-causal
  masking and shared position ID generation.
- `contains_raw_control_marker`, `sanitize_control_tokens`: prompt sanitization.

## Dependencies

- `gen-zero-core`: base types and traits.
