"""Download the public Qwen2.5-1.5B-Instruct GGUF (Q4_K_M) and verify it byte for byte.

    python -m gen_zero.scripts.setup_qwen15b_models [--dest DIR] [--source hf|modelscope]

The weights are Alibaba's public release, fetched from Hugging Face (huggingface_hub when
installed, otherwise urllib) or ModelScope. Nothing is bundled with this repository. The
expected size and sha256 below come from the Hugging Face LFS metadata of
Qwen/Qwen2.5-1.5B-Instruct-GGUF (revision 91cad51170dc346986eccefdc2dd33a9da36ead9); a
file that does not match is deleted and the script exits non-zero.
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import os
import sys
import urllib.request
from pathlib import Path

log = logging.getLogger("setup_qwen15b_models")

REPO_ID = "Qwen/Qwen2.5-1.5B-Instruct-GGUF"
FILENAME = "qwen2.5-1.5b-instruct-q4_k_m.gguf"
EXPECTED_SIZE = 1117320736
EXPECTED_SHA256 = "6a1a2eb6d15622bf3c96857206351ba97e1af16c30d7a74ee38970e434e9407e"
URLS = {
    "hf": f"{os.environ.get('HF_ENDPOINT', 'https://huggingface.co').rstrip('/')}/{REPO_ID}/resolve/main/{FILENAME}",
    "modelscope": f"https://modelscope.cn/models/{REPO_ID}/resolve/master/{FILENAME}",
}


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(path: Path) -> None:
    """Raise unless ``path`` is exactly the published Q4_K_M file."""
    if not path.is_file():
        raise FileNotFoundError(path)
    size = path.stat().st_size
    if size != EXPECTED_SIZE:
        raise ValueError(f"{path.name}: size {size} != expected {EXPECTED_SIZE}")
    digest = sha256_of(path)
    if digest != EXPECTED_SHA256:
        raise ValueError(f"{path.name}: sha256 {digest} != expected {EXPECTED_SHA256}")


def _download_urllib(url: str, target: Path) -> None:
    part = target.with_suffix(target.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "gen-zero-setup/1"})
    with urllib.request.urlopen(request, timeout=60) as response, part.open("wb") as out:
        while chunk := response.read(8 * 1024 * 1024):
            out.write(chunk)
    part.replace(target)


def _download_hf(dest: Path) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo_id=REPO_ID, filename=FILENAME, local_dir=str(dest)))


def fetch(dest: Path, source: str = "hf") -> Path:
    """Return the verified GGUF path under ``dest``, downloading it when absent or corrupt."""
    dest.mkdir(parents=True, exist_ok=True)
    target = dest / FILENAME
    if target.is_file():
        try:
            verify(target)
            log.info("already present and verified: %s", target)
            return target
        except ValueError as exc:
            log.warning("existing file rejected (%s); re-downloading", exc)
            target.unlink()
    if source == "hf":
        try:
            import huggingface_hub  # noqa: F401
            target = _download_hf(dest)
        except ImportError:
            log.warning("huggingface_hub not installed; downloading %s with urllib", URLS["hf"])
            _download_urllib(URLS["hf"], target)
    elif source == "modelscope":
        _download_urllib(URLS["modelscope"], target)
    else:
        raise ValueError(f"unknown source {source!r}")
    try:
        verify(target)
    except ValueError:
        target.unlink(missing_ok=True)
        raise
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dest", default=os.environ.get("QWEN15B_MODEL_DIR", "models/qwen2.5-1.5b-instruct-gguf"))
    parser.add_argument("--source", choices=sorted(URLS), default="hf")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        path = fetch(Path(args.dest).resolve(), args.source)
    except Exception as exc:  # network, disk or verification failure: report and exit non-zero
        log.error("setup failed: %s: %s", type(exc).__name__, exc)
        return 1
    print(f"OK {path} sha256={EXPECTED_SHA256}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
