#!/usr/bin/env bash
# Run on the dev host. Source provider credentials before invoking this script.
# Provider secrets are inherited by Pier only; never serialized into config/logs.
set -eu
: "${GEN_ZERO_SERVICE_URL:?Set the loopback cognitive service endpoint}"
: "${PROPOSER_URL:?Set the full API endpoint}"
: "${PROPOSER_MODEL:?Set an explicit model}"
: "${PROPOSER_PROTOCOL:?Set anthropic or openai}"
: "${PROPOSER_KEY_ENV:?Set the credential variable NAME}"
bench_dir=${BENCH_DIR:-$HOME/benchmarks}
evidence=${EVIDENCE_DIR:?Set a fresh evidence directory}
if [[ -e "$evidence" ]]; then
    echo "Refusing to reuse evidence directory: $evidence" >&2
    exit 1
fi
mkdir -p "$evidence/audit"
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="$bench_dir${PYTHONPATH:+:$PYTHONPATH}"
cd "$bench_dir"
job=t3-deepswe-r3-$(date -u +%Y%m%dT%H%M%SZ)
printf '%s\n' "$job" > "$evidence/job-name"
sha256sum gen_zero_deepswe_adapter.py deepswe_genzero_gate.py deepswe_audited_docker.py > "$evidence/deployed.sha256"
cp gen_zero_deepswe_adapter.py "$evidence/deployed-adapter.source.txt"
cp deepswe_genzero_gate.py "$evidence/deployed-gate.source.txt"
python -c 'from gen_zero.gate.policy_gate import DecisionPolicyGate; from gen_zero.world_model.imagination_planner import ImaginationMCTSPlanner'
cp deepswe_audited_docker.py "$evidence/deployed-audit.source.txt"
git -C deep-swe rev-parse HEAD > "$evidence/dataset-head.txt"
pier --version > "$evidence/pier-version.txt" 2>&1
docker version > "$evidence/docker-version.txt" 2>&1
command=(pier run -p "$bench_dir/deep-swe/tasks/tomlkit-toml-table-converters"
  --agent-import-path gen_zero_deepswe_adapter:GenZeroDeepSWEAdapter
  --ak "proposer_url=$PROPOSER_URL" --ak "proposer_model=$PROPOSER_MODEL"
  --ak "proposer_protocol=$PROPOSER_PROTOCOL" --ak "proposer_key_env=$PROPOSER_KEY_ENV"
  --ak "gen_zero_service_url=$GEN_ZERO_SERVICE_URL"
  --ak 'validation_command=python -m pytest -q' --ak max_steps=60
  --environment-import-path deepswe_audited_docker:AuditedDockerEnvironment
  --ek "audit_dir=$evidence/audit" --env docker --n-concurrent 1
  --n-attempts 1 --max-retries 0 --job-name "$job" --jobs-dir "$evidence/jobs")
printf '%q ' "${command[@]}" > "$evidence/pier-command.txt"
printf '\n' >> "$evidence/pier-command.txt"
docker events --filter type=container --format '{{json .}}' > "$evidence/docker-events.jsonl" 2> "$evidence/docker-events.stderr" &
events_pid=$!
trap 'kill "$events_pid" 2>/dev/null || true' EXIT
set +e
"${command[@]}" > "$evidence/pier-run.log" 2>&1
result=$?
set -e
printf '%s\n' "$result" > "$evidence/pier-run.exit"
# A zero Pier CLI exit is not equivalent to a successful task. The evidence
# validator separately checks the patch, errors, score and verifier exec status.
exit "$result"
