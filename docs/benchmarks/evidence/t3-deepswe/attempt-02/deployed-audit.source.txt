"""Pier Docker environment that preserves actual exec results without changing them.

Use --environment-import-path deepswe_audited_docker:AuditedDockerEnvironment.
The separate verifier uses the same class, with its own session_id. No task,
patch, command, score, timeout, or test result is rewritten by this observer.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

from pier.environments.docker.docker import DockerEnvironment


class AuditedDockerEnvironment(DockerEnvironment):
    def __init__(self, *args, audit_dir: str, **kwargs):
        super().__init__(*args, **kwargs)
        self.audit_dir = Path(audit_dir)
        self.audit_dir.mkdir(parents=True, exist_ok=True)
        self.audit_sequence = 0

    async def exec(self, command, cwd=None, env=None, timeout_sec=None, user=None):
        self.audit_sequence += 1
        sequence = self.audit_sequence
        record = {"session_id": self.session_id, "sequence": sequence,
                  "command": command, "cwd": cwd, "user": user,
                  "timeout_sec": timeout_sec,
                  "started_at": datetime.now(timezone.utc).isoformat()}
        path = self.audit_dir / f"{self.session_id}-{sequence:04d}.json"
        # Environment values (including credentials) are deliberately not logged.
        path.write_text(json.dumps(record, indent=2) + "\n")
        try:
            result = await super().exec(command, cwd, env, timeout_sec, user)
        except BaseException as exc:
            record["exception_type"] = type(exc).__name__
            path.write_text(json.dumps(record, indent=2) + "\n")
            raise
        record.update(result.model_dump())
        record["finished_at"] = datetime.now(timezone.utc).isoformat()
        path.write_text(json.dumps(record, indent=2) + "\n")
        return result
