import time
import os
import sys
from huggingface_hub import snapshot_download

print("[DOWNLOAD] Starting download of Qwen/Qwen3.5-9B to D:\\models\\Qwen3.5-9B...", flush=True)
t0 = time.time()
try:
    path = snapshot_download(
        repo_id="Qwen/Qwen3.5-9B",
        local_dir=r"D:\models\Qwen3.5-9B",
        max_workers=8,
    )
    dt = time.time() - t0
    print(f"[DOWNLOAD] Successfully completed in {dt:.1f}s. Model at {path}", flush=True)
except Exception as e:
    print(f"[DOWNLOAD ERROR] {e}", flush=True)
