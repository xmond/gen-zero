"""Fail-closed acceptance of a single Pier trial, independent of CLI exit 0."""
import argparse
import hashlib
import json
from pathlib import Path


def verify(root: Path) -> dict:
    failures = []
    result_paths = list(root.glob("jobs/*/*/result.json"))
    if len(result_paths) != 1:
        raise ValueError("Expected exactly one completed trial result.json")
    result_path = result_paths[0]
    trial = result_path.parent
    result = json.loads(result_path.read_text())
    patch_path = trial / "artifacts/model.patch"
    patch = patch_path.read_bytes()
    if not patch.strip() or not patch.startswith(b"diff --git "):
        failures.append("No nonempty standard Git patch collected")
    agent_patch = trial / "agent/model.patch"
    if not agent_patch.exists() or agent_patch.read_bytes() != patch:
        failures.append("Collected patch does not match agent's finalized patch")
    completion_path = trial / "agent/completion.json"
    if completion_path.exists():
        completion = json.loads(completion_path.read_text())
        if completion.get("patch_sha256") != hashlib.sha256(patch).hexdigest():
            failures.append("Finalized patch hash mismatch")
    else:
        failures.append("Agent never finalized a validated repair")
    if result.get("exception_info") is not None:
        failures.append("Pier trial contains an exception")
    pier_exit = int((root / "pier-run.exit").read_text())
    if pier_exit != 0:
        failures.append(f"Pier process exited {pier_exit}")
    reward = json.loads((trial / "verifier/reward.json").read_text())
    if result.get("verifier_result", {}).get("rewards") != reward:
        failures.append("Verifier reward file differs from the Pier trial result")
    for group in ("f2p", "p2p"):
        total, passed = reward.get(f"{group}_total", 0), reward.get(f"{group}_passed", -1)
        if total <= 0 or passed != total:
            failures.append(f"{group}: {passed}/{total} passed")
    if reward.get("reward") != 1:
        failures.append("Verifier did not award full reward")
    audit = [json.loads(p.read_text()) for p in sorted((root / "audit").glob("*.json"))]
    verifier_exec = [a for a in audit if "__verifier__" in a["session_id"]
                     and "/logs/verifier/test-stdout.txt" in a["command"]]
    if len(verifier_exec) != 1 or verifier_exec[0].get("return_code") != 0:
        failures.append("Missing or unsuccessful independent verifier execution")
    starts = {}
    for line in (root / "docker-events.jsonl").read_text().splitlines():
        event = json.loads(line)
        if event.get("Action") == "start":
            attrs = event.get("Actor", {}).get("Attributes", {})
            project = attrs.get("com.docker.compose.project", "")
            if result["trial_name"].lower() in project:
                starts[project] = event["Actor"]["ID"]
    if len(set(starts.values())) < 2:
        failures.append("Docker events do not prove two distinct containers started")
    log_path = trial / "verifier/test-stdout.txt"
    if not log_path.exists() or not log_path.stat().st_size:
        failures.append("Missing verifier test log")
    return {"accepted": not failures, "failures": failures,
            "trial_result": str(result_path.relative_to(root)),
            "patch_bytes": len(patch), "patch_sha256": hashlib.sha256(patch).hexdigest(),
            "pier_exit": pier_exit, "reward": reward,
            "verifier_exit": verifier_exec[0].get("return_code") if len(verifier_exec) == 1 else None,
            "container_starts": starts,
            "nonzero_sandbox_commands": [{"session": a["session_id"], "command": a["command"],
                                          "exit": a.get("return_code")}
                                         for a in audit if a.get("return_code") not in (None, 0)]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path)
    args = parser.parse_args()
    try:
        report = verify(args.evidence)
    except Exception as exc:
        report = {"accepted": False, "failures": [f"{type(exc).__name__}: {exc}"]}
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["accepted"] else 1)
