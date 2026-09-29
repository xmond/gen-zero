#!/usr/bin/env bash
# Native Gen-Zero Rust CLI walkthrough. No Python, no PyTorch checkpoints, no
# running server: every subcommand here is a single local process.
set -euo pipefail

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
bin=${GEN_ZERO_BIN:-"$repo/target/release/gen-zero"}

if [[ ! -x "$bin" ]]; then
  echo "gen-zero binary is missing: $bin (run: cargo build --release -p gen-zero-cli)" >&2
  exit 1
fi

state_file=$(mktemp)
trap 'rm -f "$state_file"' EXIT

# A 1024-float latent state (the width every planner subcommand below expects),
# small enough in norm to stay inside the world model's stable ball.
awk 'BEGIN{printf "["; for(i=0;i<1024;i++) printf "%s%.4f", (i?",":""), 0.8*sin(i*0.37); print "]"}' > "$state_file"

echo "== a) what-if: compare candidate first actions on the latent world model =="
"$bin" what-if --state "@$state_file" --candidates proceed,wait

echo
echo "== b) simulate: roll a fixed 5-step action plan forward =="
"$bin" simulate --state "@$state_file" --actions proceed,wait,proceed,wait,proceed --horizon 5

echo
echo "== c) keygen: generate a Gen-Zero connection token =="
"$bin" keygen

echo
echo "== d) serve --mode stdio: MCP initialize handshake over a pipe =="
request='{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"official-example","version":"1"}}}'
serve_json=$(printf '%s\n' "$request" | "$bin" serve --mode stdio 2>/dev/null)
echo "$serve_json"
case "$serve_json" in
  *'"id":1'*'"result"'*) ;;
  *) echo "serve did not return an MCP initialize result" >&2; exit 1 ;;
esac

echo
echo "Native Rust CLI walkthrough complete: what-if, simulate, keygen, and MCP stdio initialize all succeeded."
echo "Every score above comes from the untrained latent prior (no checkpoint mounted): this demonstrates plumbing, not a learned model."
