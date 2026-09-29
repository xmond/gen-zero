"""Download official Qwen3.5-0.8B weights from HuggingFace to local directory."""

import os
import sys
from pathlib import Path
from huggingface_hub import snapshot_download

MODEL_ID = "Qwen/Qwen3.5-0.8B"


def main():
    default_dir = os.environ.get("QWEN_MODEL_DIR", "models/Qwen3.5-0.8B")
    target_dir = Path(default_dir).resolve()
    print(f"--> Target download directory: {target_dir}")
    target_dir.mkdir(parents=True, exist_ok=True)

    print(f"--> Initiating snapshot download for {MODEL_ID}...")
    local_path = snapshot_download(
        repo_id=MODEL_ID,
        local_dir=str(target_dir),
        ignore_patterns=["*.msgpack", "*.h5", "*.ot"],
        max_workers=4
    )
    print(f"✓ Model successfully downloaded and ready at: {local_path}")


if __name__ == "__main__":
    main()
