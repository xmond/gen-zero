# PolicyGate and Loop fail-closed verification (2026-09-27)

Workspace: `/ebs/pj/gen-zero-worktree/fix-gate`
Base: `fd1b8bafbbbb23183c60c9a7050cecd23826e82d`

## Commands and observed exits

```bash
pytest python/tests/test_policy_gate_fail_closed_fixes.py python/tests/test_loop_state_machine_verifier.py
```

Exit 0, 73 passed. Full output: `requested-pytest.txt`.

```bash
pytest python/gen_zero/tests/test_issue_8_compound_decision.py
```

Initial compatibility run after the implementation: exit 1, 5 failed and 13 passed.
Full output: `initial-related-failures.txt`. Old tests expected CONFIRM for
operations now required to STOP, or omitted required context. Tests were updated
to the explicit new contract; confidence-tier success/escalation assertions remain.
The whitelist blind-confidence STOP invariant was also preserved in production code.

```bash
pytest python/gen_zero/tests/test_issue_8_compound_decision.py python/tests/test_policy_gate_fail_closed_fixes.py python/tests/test_loop_state_machine_verifier.py
```

Exit 0, 91 passed. Full output: `related-pytest.txt`.

```bash
python -m compileall -q python/gen_zero/gate/policy_gate.py python/gen_zero/runtime/loop_state_machine.py python/tests/test_policy_gate_fail_closed_fixes.py python/tests/test_loop_state_machine_verifier.py python/gen_zero/tests/test_issue_8_compound_decision.py
```

Exit 0, no diagnostic output (quiet bytecode compilation).

## Evidence and limitations

- `python/gen_zero/gate/policy_gate.py`: explicit schema/numeric validation (lines 139–204),
  immutable destructive-operation and inclusive risk-threshold STOP (lines 221–236) before
  configurable sensitive rules. Whitelists no longer grant authorization.
- `python/gen_zero/runtime/loop_state_machine.py`: completion branch at line 267; absent verifier returns
  UNVERIFIED_FINISH; completion reports serialize agent_finished and verified
  separately. Only boolean True from the supplied verifier produces SUCCESS.
  Exceptions escalate; truthy non-booleans cannot certify success.
- Required regression files cover malformed/missing inputs, finite probabilities,
  aliases, profile-independent destructive stops, whitelist bypasses, threshold
  boundaries, verifier absence/rejection/errors, periodic checks and dry runs.
- This is focused Python unit verification and bytecode compilation, not full
  repository or live-environment acceptance. The callback's real independence
  and correctness remain the integrator's responsibility. Regex rules do not
  constitute a semantic proof of arbitrary action safety.
