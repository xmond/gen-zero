#!/bin/bash
set -eu
cd "${GENZERO_EVAL_ROOT:?Set GENZERO_EVAL_ROOT to the evaluation checkout}"
umask 077
if [ ! -f .scorer-token ]; then python3 -c 'import secrets; print(secrets.token_urlsafe(32))' > .scorer-token; fi
for name in service scorer; do
 if [ -f "$name.pid" ]; then
  pid=$(cat "$name.pid")
  if kill -0 "$pid" 2>/dev/null; then
   cwd=$(readlink "/proc/$pid/cwd")
   [ "$cwd" = "$(pwd -P)" ] || exit 1
   kill "$pid"
   for i in {1..50}; do
    if ! kill -0 "$pid" 2>/dev/null || [ "$(ps -p "$pid" -o stat= | cut -c1)" = Z ]; then break; fi
    sleep 0.1
   done
  fi
 fi
done
cp service.log t4-evidence/service-baseline.log
export PYTHONPATH=python ZERO_MODEL_CACHE="$PWD/t4-model" GENZERO_SEMANTIC_THREADS=8
export GENZERO_API_KEY=$(cat .scorer-token)
nohup .venv-t4/bin/python -m gen_zero.cli semantic --host 127.0.0.1 --port 8995 > scorer.log 2>&1 < /dev/null &
echo $! > scorer.pid
unset GENZERO_API_KEY
for i in {1..60}; do
 if curl -fsS http://127.0.0.1:8995/v1/semantic_health > t4-evidence/scorer-health.json 2>/dev/null; then break; fi
 sleep 0.5
done
curl -fsS http://127.0.0.1:8995/v1/semantic_health > t4-evidence/scorer-health.json
export GENZERO_PYTHON_API_KEY=$(cat .scorer-token) GENZERO_BRIDGE_REQUIRED=1
nohup target/release/gen-zero serve --mode sse --port 8080 --host 127.0.0.1 > service.log 2>&1 < /dev/null &
echo $! > service.pid
