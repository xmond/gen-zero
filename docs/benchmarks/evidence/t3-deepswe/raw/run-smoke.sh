#!/bin/bash
set -u
export PATH="$HOME/.local/bin:$PATH"
: "${BENCHMARK_ROOT:?Set BENCHMARK_ROOT to the benchmark directory}"
: "${GENZERO_REPO:?Set GENZERO_REPO to the repository checkout}"
export PYTHONPATH="$BENCHMARK_ROOT"
cd "$BENCHMARK_ROOT" || exit 1
evidence="$BENCHMARK_ROOT/t3-evidence"
job=t3-deepswe-smoke-$(date -u +%Y%m%dT%H%M%SZ)
printf '%s\n' "$job" > "$evidence/job-name"
sha256sum gen_zero_deepswe_adapter.py "$GENZERO_REPO/target/release/gen-zero" > "$evidence/sha256.txt"
git -C deep-swe rev-parse HEAD > "$evidence/dataset-head.txt"
pier --version > "$evidence/pier-version.txt" 2>&1
docker version > "$evidence/docker-version.txt" 2>&1
docker events --filter type=container --format '{{json .}}' > "$evidence/docker-events.jsonl" 2> "$evidence/docker-events.stderr" &
events_pid=$!
command=(pier run -p "$BENCHMARK_ROOT/deep-swe/tasks/tomlkit-toml-table-converters" --agent-import-path gen_zero_deepswe_adapter:GenZeroDeepSWEAdapter --ak gen_zero_binary="$GENZERO_REPO/target/release/gen-zero" --ak 'query=def ' --env docker --n-concurrent 1 --n-attempts 1 --max-retries 0 --job-name "$job" --jobs-dir "$BENCHMARK_ROOT/t3-evidence/jobs")
printf '%q ' "${command[@]}" > "$evidence/pier-command.txt"
printf '\n' >> "$evidence/pier-command.txt"
"${command[@]}" > "$evidence/pier-run.log" 2>&1
result=$?
printf '%s\n' "$result" > "$evidence/pier-run.exit"
kill "$events_pid"
wait "$events_pid"
exit "$result"
