#!/bin/bash
set -u
cd "${GENZERO_EVAL_ROOT:?Set GENZERO_EVAL_ROOT to the evaluation checkout}" || exit 1
phase=${1:?phase required}
out=t4-evidence/$phase
mkdir -p "$out"
failed=0
for name in health ready ask route what_if invalid_what_if; do
  url=http://127.0.0.1:8080/v1/decisions
  args=()
  expected=200
  case "$name" in
    health|ready) url=http://127.0.0.1:8080/$name ;;
    *) args=(-X POST -H 'Content-Type: application/json' --data-binary @t4-evidence/$name.json) ;;
  esac
  if [ "$name" = invalid_what_if ]; then expected=400; fi
  cmd=(curl --silent --show-error --fail-with-body --max-time 120 -D "$out/$name.headers" -o "$out/$name.body.json" -w '%{http_code}\n' "${args[@]}" "$url")
  printf '%q ' "${cmd[@]}" > "$out/$name.command"
  printf '\n' >> "$out/$name.command"
  "${cmd[@]}" > "$out/$name.status" 2> "$out/$name.stderr"
  rc=$?
  echo "$rc" > "$out/$name.exit"
  cat "$out/$name.command" "$out/$name.status" "$out/$name.exit" "$out/$name.body.json" "$out/$name.stderr"
  echo
  python3 - "$out/$name.body.json" "$out/$name.status" "$expected" "$name" <<'PY'
import json,sys
body=json.load(open(sys.argv[1])); status=int(open(sys.argv[2]).read()); expected=int(sys.argv[3]); name=sys.argv[4]
assert isinstance(body,dict),body
if name=='invalid_what_if': assert 400<=status<500 and body.get('isError') is True,body
else:
 assert status==expected,(status,body)
 if name not in ('health','ready'):
  assert body.get('isError') is False,body
  if name in ('ask','route'): assert body.get('semantic_scoring') is True,body
  if name=='route': assert body.get('count',0)>0,body
PY
  vrc=$?
  echo "$vrc" > "$out/$name.validation.exit"
  if [ "$vrc" -ne 0 ]; then failed=1; fi
 done
exit "$failed"
