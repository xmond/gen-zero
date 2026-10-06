#!/usr/bin/env python3
"""Anti-leakage gate: keep closed-source assets out of the open-source repo.

Rules
  S1  weights / binaries: banned extensions, binaries over 5 MiB, weight or
      archive magic bytes under any name (GGUF, GGML, NumPy, npz, safetensors,
      PyTorch zip, pickle, HDF5, ONNX, TFLite, zip/gzip/xz/zstd/bzip2/7z/rar/tar),
      long base64 payloads in text. A file may pass only if the allowlist pins
      its sha256 AND .gitattributes routes it through git-lfs.
  S2  training / learning system: gradient calls, optimizer updates, the torch
      optimizer package, third-party optimizer libraries, trainer classes, training
      argument classes, autograd / jax / tf gradient APIs, manual weight updates
      from .grad, exec of code that cannot be read statically, and closed data
      generator module names. Python is read through the AST with import and
      assignment alias tracking. Every other file, whatever its extension, goes
      through regex. A Jupyter notebook is read as JSON: its code cells are joined
      and analysed like Python (IPython magic lines are blanked, code that still
      does not parse fails closed), and the regex rules also run on the raw JSON.
      Test files may use the exemptible rules on tiny mock
      tensors ONLY if the file has a whole-line comment saying:
      anti-leakage: allow-mock-tensor
      Optimizer packages, trainer classes and dynamic exec are never exempt.
  S3  proprietary tokens: internal host names, scheduler paths, closed code
      names, credential shapes. Also checked in paths, commit and tag messages.

Text views
  NUL bytes never switch the text rules off. Each file is checked as UTF-8 (or
  its BOM encoding), as a NUL-stripped view (this covers UTF-16/32 and
  NUL-split tokens) and as an NFKC view without Unicode format characters
  (category Cf, including tag characters).

Patterns use a one-char class (for example `x[y]`) and canonical names are
joined at run time, so this file does not match its own rules.

Modes
  default         walk every file in --target-repo except .git
  --git-diff-only staged index blobs, unstaged changes and untracked files
  --history       also every commit, path and blob reachable from any ref
  --pre-push      read git's pre-push stdin and check the pushed commits, the
                  blobs they add or change, and their messages. Never the
                  working tree. A new branch is checked against the refs the
                  remote advertises (git ls-remote), never against local
                  remote-tracking refs; when that fails every commit is scanned.

Exit codes
  0  clean, or --report-only (which logs the violations explicitly)
  1  violations found (--fail-closed stops at the first one)
  2  usage or operational error: contradictory flags, bad path, git failure,
     nothing scanned, unreadable file

Hook
  python3 scripts/anti_leakage_lint.py --target-repo /path/to/open-repo --install-hook
  An existing foreign pre-push hook (for example git-lfs) is moved to
  pre-push.anti-leakage-chained and runs after a clean scan with the same
  arguments and stdin. Only a line holding exactly the hook signature marks a
  hook as ours. `git push --no-verify` skips hooks, so run the gate in
  CI as well.
"""
from __future__ import annotations

import argparse
import ast
import fnmatch
import hashlib
import io
import json
import os
import pickletools
import re
import shlex
import stat
import struct
import subprocess
import sys
import unicodedata
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Set, Tuple

BIN_SIZE_LIMIT = 5 * 1024 * 1024
MAX_TEXT_BYTES = 64 * 1024 * 1024  # larger content is reported as unscannable, never skipped
AST_MAX_BYTES = 4 * 1024 * 1024
ENCODED_PAYLOAD_LIMIT = 1024 * 1024
HEAD_BYTES = 8192
EXEMPT_RE = re.compile(r"^[ \t]*(?:#|//)[ \t]*anti-leakage: allow-mock-tensor[ \t]*$", re.M)
DEFAULT_ALLOWLIST = Path(__file__).resolve().with_name("anti_leakage_allowlist.json")
HOOK_SIGNATURE = "# anti_leakage_lint pre-push hook"
CHAINED_HOOK = "pre-push.anti-leakage-chained"
SHA_RE = re.compile(r"[0-9a-f]{40}|[0-9a-f]{64}")
SKIP_DIRS = {".git"}
PY_EXT = {".py", ".pyw", ".pyi"}

BANNED_EXT = {
    ".safetensors", ".pt", ".pth", ".bin", ".ckpt", ".gguf", ".ggml", ".npz", ".npy", ".onnx",
    ".pkl", ".pickle", ".h5", ".hdf5", ".joblib", ".tflite", ".msgpack",
}

# ---- S2 -------------------------------------------------------------------
# (id, regex, exemptible_in_tests)
S2_REGEX = [
    ("loss-backward", re.compile(r"\bloss\w*\.backwar[d]\s*\("), True),
    ("backward-call", re.compile(r"\bbackwar[d]\s*\("), True),
    ("backward-reflect", re.compile(r"""["']backwar[d]["']"""), True),
    ("optimizer-step", re.compile(r"\b(?=\w*?opt)\w+\.ste[p]\s*\(", re.I), True),
    ("optimizer-step", re.compile(r"\.zero_gra[d]\s*\("), True),
    ("torch-optim", re.compile(r"\btorch\.opti[m]\b|\bfrom\s+torch\s+import\s+(?:(?!from\s+torch\s+import)[^\n])*?\bopti[m]\b"
                               r"|\boptim\.[A-Z]\w*\s*\("), False),
    ("optimizer-library", re.compile(
        r"\bopta[x]\b|\bkeras\.optimizer[s]\b|\boptim(?:izer)?::"
        r"|\b(?:Adam[W]?|SG[D]|Sgd|RmsProp|AdaGrad|Lion)\s*::\s*(?:new|default)\b"
        r"|\bFlux\.(?:trai[n]!|Optimis[e]\b|setu[p]\s*\()|\bOptimisers\.(?:update!?|setu[p])\b|\bupdate!\s*\("), False),
    ("trainer-call", re.compile(r"Traine[r]\s*\("), False),
    ("training-arguments", re.compile(r"\bTrainingArgument[s]\b"), False),
    ("autograd-grad", re.compile(
        r"\bautograd\.gra[d]\s*\(|\bjax\.(?:value_and_)?gra[d]\b|\bGradientTap[e]\b"
        r"|\b(?:Zygote|Flux)\.(?:with)?gradien[t]\b|\bwithgradien[t]\s*\(|\bbackward_ste[p]\s*\("), True),
]
# module / path names of closed data-generation code
S2_NAME_REGEX = [
    ("closed-data-generator", re.compile(r"\bsynthesizer[s]\b|\bbuild_official_clean_split[s]\b|\bteacher_extractio[n]\b")),
]
S2_PATH_REGEX = re.compile(r"(?:^|/)synthesizer[s](?:/|$)|build_official_clean_split[s]|teacher_extractio[n]")

# ---- S3 -------------------------------------------------------------------
S3_REGEX = [
    ("internal-host", re.compile(r"\bai-serve[r]\b", re.I)),
    ("internal-name", re.compile(r"\bsape[x]\b", re.I)),
    ("internal-scheduler", re.compile(r"dev_offload_exec\.s[h]")),
    ("closed-codename", re.compile(r"gen-zero-1b-nativ[e]|s-deq-unified-backbon[e]", re.I)),
    # the open repo is github.com/xmond/gen-zero, so the bare org name is public;
    # only credential shapes and other xmond repos are internal.
    ("xmond-credential", re.compile(r"xmon[d](?:(?!xmon[d])[\w.-])*(?:token|secret|passw(?:or)?d|api[_-]?key)\s*[:=]\s*\S+", re.I)),
    ("xmond-private-repo", re.compile(r"github\.com[/:]xmon[d]/(?!gen-zero(?:\.git)?(?![\w-]))[\w.-]+", re.I)),
    ("credential-private-key", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KE[Y]-----")),
    ("credential-github", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}|\bgithub_pa[t]_[A-Za-z0-9_]{22,}")),
    ("credential-hf-anthropic-aws", re.compile(r"\bhf_[A-Za-z0-9]{30,}|\bsk-an[t]-[A-Za-z0-9_-]{20,}|\bAKI[A]{1}[0-9A-Z]{16}\b")),
]

# Required lowercase substrings per pattern: a pattern runs only when one occurs in the
# lowercased view. Each list must cover every alternative of its pattern; containment is
# case-insensitive, so this is a superset test and never hides a match.
PREFILTER = {
    S2_REGEX[0][1]: ["backwar"], S2_REGEX[1][1]: ["backwar"], S2_REGEX[2][1]: ["backwar"],
    S2_REGEX[3][1]: [".ste"], S2_REGEX[4][1]: ["zero_gra"], S2_REGEX[5][1]: ["optim"],
    S2_REGEX[6][1]: ["opta", "keras.", "optim", "adam", "sgd", "rmsprop", "adagrad", "lion", "flux.",
                     "optimisers.", "update!"],
    S2_REGEX[7][1]: ["traine"], S2_REGEX[8][1]: ["trainingargument"],
    S2_REGEX[9][1]: ["autograd.gra", "jax.", "gradienttap", "zygote.", "flux.", "withgradien", "backward_ste"],
    S2_NAME_REGEX[0][1]: ["synthesizer", "build_official_clean_split", "teacher_extractio"],
    S3_REGEX[0][1]: ["ai-serve"], S3_REGEX[1][1]: ["sape"], S3_REGEX[2][1]: ["dev_offload_exe"],
    S3_REGEX[3][1]: ["gen-zero-1b-nativ", "s-deq-unified-backbon"], S3_REGEX[4][1]: ["xmon"],
    S3_REGEX[5][1]: ["xmon"], S3_REGEX[6][1]: ["-----begin"],
    S3_REGEX[7][1]: ["ghp_", "gho_", "ghu_", "ghs_", "ghr_", "github_pa"],
    S3_REGEX[8][1]: ["hf_", "sk-an", "akia"],
}

def anchors_hit(rx: "re.Pattern[str]", low: str) -> bool:
    anchors = PREFILTER.get(rx)
    return not anchors or any(a in low for a in anchors)


def prefilter_view(text: str) -> str:
    # re.IGNORECASE also folds dotless i and long s to ASCII i / s; str.lower() does not.
    # (Kelvin sign and dotted capital I already lower to k and i.)
    return text.lower().replace("\u0131", "i").replace("\u017f", "s")


ML_IMPORT_RE = re.compile(r"^[ \t]*(?:import|from)[ \t]+(?:torch|functorch|jax|flax|opta[x]|tensorflow|keras|transformers|trl"
                          r"|lightning|pytorch_lightning|accelerate|deepspeed)\b", re.M)
ENCODED_RUN_RE = re.compile(r"[A-Za-z0-9+/=_-]+")
NOT_ENCODED_RE = re.compile(r"[^A-Za-z0-9+/=_-]")
# first opcode of a protocol 0 / 1 pickle; protocol 2-5 start with 0x80
PICKLE_OLD_OPENERS = {bytes([b]) for b in b"(c})]NIFSVLUTXGJKMid"}
PICKLE_GLOBAL_RE = re.compile(r"[A-Za-z_][\w.]* [A-Za-z_][\w.]*")
PICKLE_MIN_OPS = 64
ZIP_TAIL_BYTES = 65536 + 22 + 20  # max EOCD comment + EOCD + zip64 locator


def _j(*parts: str) -> str:
    return ".".join(parts)


# Canonical names are built from parts so the AST string scan of this file stays clean.
T_OPTIM = _j("torch", "optim")
OPT_LIBS = ("".join(("opt", "ax")), _j("keras", "optimizers"), _j("tensorflow", "keras", "optimizers"),
            _j("tf", "keras", "optimizers"))
AUTOGRAD_FUNCS = {
    _j("torch", "autograd", "grad"), _j("torch", "autograd", "backward"), _j("torch", "func", "grad"),
    _j("torch", "func", "grad_and_value"), _j("functorch", "grad"), _j("functorch", "grad_and_value"),
    _j("jax", "grad"), _j("jax", "value_and_grad"), _j("tensorflow", "Gradient" + "Tape"), _j("tf", "Gradient" + "Tape"),
}
TRAINING_ARGS = "".join(("Training", "Arguments"))
EXEC_FUNCS = {"exec", "eval", "compile", "builtins.exec", "builtins.eval", "builtins.compile",
              "__builtins__.exec", "__builtins__.eval", "__builtins__.compile"}
IMPORT_FUNCS = {"__import__", "importlib.import_module", "builtins.__import__", "__builtins__.__import__"}
# any reference to these that is not a direct call hands the function to code we cannot follow
EXEC_REFS = EXEC_FUNCS | {"__import__", "builtins.__import__", "__builtins__.__import__"}
GETATTR_FUNCS = {"getattr", "builtins.getattr"}
CALLER_FUNCS = {"operator.methodcaller", "operator.attrgetter"}


@dataclass(frozen=True)
class Finding:
    rule: str
    path: str
    line: int
    detail: str

    def render(self) -> str:
        loc = f"{self.path}:{self.line}" if self.line else self.path
        return f"[{self.rule}] {loc}: {self.detail}"


@dataclass(frozen=True)
class Content:
    size: int
    head: bytes
    data: Optional[bytes]  # None when size > MAX_TEXT_BYTES
    sha256_of: Callable[[], str]


class FirstViolation(Exception):
    def __init__(self, finding: Finding):
        self.finding = finding


class ScanError(Exception):
    pass


# ---------------------------------------------------------------------------
# git helpers
# ---------------------------------------------------------------------------
def git(repo: Path, *args: str, input_bytes: Optional[bytes] = None) -> bytes:
    try:
        proc = subprocess.run(["git", "-C", str(repo), *args], input=input_bytes,
                              capture_output=True, check=False)
    except OSError as exc:
        raise ScanError(f"cannot run git: {exc}") from exc
    if proc.returncode != 0:
        raise ScanError(f"git {' '.join(args)} failed ({proc.returncode}): {proc.stderr.decode(errors='replace').strip()}")
    return proc.stdout


def git_ok(repo: Path, *args: str) -> bool:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, check=False).returncode == 0


def is_git_repo(repo: Path) -> bool:
    return git_ok(repo, "rev-parse", "--git-dir")


def split_z(raw: bytes) -> List[str]:
    return [p.decode("utf-8", "surrogateescape") for p in raw.split(b"\0") if p]


class CatFile:
    """One long-lived `git cat-file --batch`; streams big blobs instead of buffering them."""

    def __init__(self, repo: Path):
        try:
            self.proc = subprocess.Popen(["git", "-C", str(repo), "cat-file", "--batch"],
                                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except OSError as exc:
            raise ScanError(f"cannot run git cat-file: {exc}") from exc

    def read(self, name: str, missing_ok: bool = False) -> Optional[Tuple[str, Content]]:
        if "\n" in name:
            raise ScanError(f"object name contains a newline: {name!r}")
        if self.proc.stdin is None or self.proc.stdout is None:
            raise ScanError("git cat-file pipes are closed")
        self.proc.stdin.write(name.encode("utf-8", "surrogateescape") + b"\n")
        self.proc.stdin.flush()
        header = self.proc.stdout.readline()
        if not header:
            err = self.proc.stderr.read().decode(errors="replace") if self.proc.stderr else ""
            raise ScanError(f"git cat-file --batch exited while reading {name}: {err.strip()}")
        parts = header.split()
        if parts[-1:] == [b"missing"] or parts[-1:] == [b"ambiguous"]:
            if missing_ok:
                return None
            raise ScanError(f"git object {name} is {parts[-1].decode()}")
        typ, size = parts[1].decode(), int(parts[2])
        sha = hashlib.sha256()
        head = b""
        chunks: List[bytes] = []
        left = size
        while left:
            chunk = self.proc.stdout.read(min(left, 1 << 20))
            if not chunk:
                raise ScanError(f"short read from git cat-file for {name}")
            left -= len(chunk)
            sha.update(chunk)
            if len(head) < HEAD_BYTES:
                head += chunk[:HEAD_BYTES - len(head)]
            if size <= MAX_TEXT_BYTES:
                chunks.append(chunk)
        self.proc.stdout.read(1)  # trailing newline after every object
        digest = sha.hexdigest()
        data = b"".join(chunks) if size <= MAX_TEXT_BYTES else None
        return typ, Content(size, head, data, lambda: digest)

    def get(self, name: str) -> Tuple[str, Content]:
        got = self.read(name)
        if got is None:
            raise ScanError(f"git object {name} is missing")
        return got

    def close(self) -> None:
        if self.proc.stdin:
            self.proc.stdin.close()
        self.proc.wait()

    def __enter__(self) -> "CatFile":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


# ---------------------------------------------------------------------------
# allowlist + lfs
# ---------------------------------------------------------------------------
def load_allowlist(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
        entries = data["files"]
        return {e["path"]: e["sha256"].lower() for e in entries}
    except (ValueError, KeyError, TypeError) as exc:
        raise ScanError(f"malformed allowlist {path}: {exc} (need {{\"files\":[{{\"path\":..,\"sha256\":..}}]}})") from exc


def parse_lfs_patterns(text: str) -> List[str]:
    pats: List[str] = []
    for raw in text.splitlines():
        parts = raw.split()
        if len(parts) >= 2 and not parts[0].startswith("#") and "filter=lfs" in parts[1:]:
            pats.append(parts[0])
    return pats


def load_lfs_patterns(repo: Path) -> List[str]:
    attrs = repo / ".gitattributes"
    return parse_lfs_patterns(attrs.read_text(errors="replace")) if attrs.is_file() else []


def lfs_matches(rel: str, patterns: List[str]) -> bool:
    name = rel.rsplit("/", 1)[-1]
    for pat in patterns:
        p = pat.lstrip("/")
        if fnmatch.fnmatchcase(rel, p) or ("/" not in p and fnmatch.fnmatchcase(name, p)):
            return True
    return False


# ---------------------------------------------------------------------------
# content classification
# ---------------------------------------------------------------------------
def is_test_path(rel: str) -> bool:
    parts = rel.split("/")
    name = parts[-1]
    return (any(p in ("tests", "test") for p in parts[:-1]) or name.startswith("test_")
            or name == "conftest.py" or re.search(r"_tests?\.\w+$", name) is not None)


def zip_magic(head: bytes, data: Optional[bytes]) -> str:
    names: List[str] = []
    if data is not None:
        try:
            names = zipfile.ZipFile(io.BytesIO(data)).namelist()
        except (zipfile.BadZipFile, ValueError, OSError):
            return "zip archive with an unreadable directory (contents unscannable)"
    blob = "\n".join(names) if names else head.decode("latin-1")
    if re.search(r"(?:^|/)data\.pkl\b|(?:^|/)byteorder\b|\.data/(?:version|serialization_id)\b", blob, re.M):
        return "PyTorch zip checkpoint (data.pkl / byteorder entries)"
    if re.search(r"\.npy$", blob, re.M):
        return "NumPy .npz archive (.npy entries)"
    inner = [n for n in names if os.path.splitext(n)[1].lower() in BANNED_EXT]
    if inner:
        return f"zip archive containing {inner[0]}"
    return "zip archive (opaque container, contents not inspected)"


def tar_v7_checksum(head: bytes) -> bool:
    """A V7 tar header has no ustar magic; its checksum field is the only tell."""
    if len(head) < 512:
        return False
    try:
        want = int(head[148:156].strip(b"\0 "), 8)
    except ValueError:
        return False
    return want == sum(head[:148]) + 32 * 8 + sum(head[156:512])


def _old_pickle_proto(blob: bytes) -> Optional[int]:
    """Walk protocol 0/1 opcodes without running them (pickle.loads would execute the payload).

    Returns the protocol of the opcodes seen, or None. A pickle reaches STOP or a long run of valid
    opcodes. No byte cap: an array pickles to one huge string opcode, read in a single step.
    Text dies on the first byte that is not an opcode."""
    count, proto = 0, 0
    try:
        for op, arg, _pos in pickletools.genops(io.BytesIO(blob)):
            count += 1
            proto = max(proto, op.proto)
            if op.name in ("GLOBAL", "INST") and not PICKLE_GLOBAL_RE.fullmatch(str(arg)):
                return None  # "module name" pairs, not prose that happens to start with c or i
            if op.name == "STOP":
                return proto if count >= 3 else None
            if count >= PICKLE_MIN_OPS:
                return proto
    except (ValueError, EOFError, UnicodeDecodeError, OverflowError, struct.error, IndexError, TypeError):
        return None
    return None


def weight_magic(head: bytes, data: Optional[bytes] = None) -> Optional[str]:
    """Name the weight / archive format the bytes carry, whatever the file is called."""
    if head[:4] == b"GGUF":
        return "GGUF"
    if head[:4] in (b"lmgg", b"tjgg", b"fmgg", b"algg"):
        return "legacy GGML"
    if head[:6] == b"\x93NUMPY":
        return "NumPy .npy payload"
    # zipfile finds the archive from the end-of-central-directory record, so any prefix length works
    tail = data[-ZIP_TAIL_BYTES:] if data is not None else b""
    if b"PK\x03\x04" in head or b"PK\x05\x06" in head or b"PK\x05\x06" in tail or b"PK\x06\x07" in tail:
        return zip_magic(head, data)
    if len(head) >= 2 and head[0] == 0x80 and 2 <= head[1] <= 5:
        # pickle.load stops at STOP and ignores whatever follows it
        if b"." in (data if data is not None else head):
            return f"pickle protocol {head[1]} stream"
    old = _old_pickle_proto(data if data is not None else head) if head[:1] in PICKLE_OLD_OPENERS else None
    if old is not None:
        return f"pickle protocol {old} stream"
    for off in (0, 512, 1024, 2048, 4096):
        if head[off:off + 8] == b"\x89HDF\r\n\x1a\n":
            return "HDF5 superblock"
    if head[4:8] == b"TFL3":
        return "TFLite flatbuffer"
    if len(head) >= 3 and head[0] == 0x08 and 1 <= head[1] <= 20 and head[2] in (0x12, 0x1A, 0x22, 0x28, 0x32, 0x3A, 0x42):
        return "ONNX ModelProto header (protobuf heuristic: ir_version varint then a ModelProto field)"
    if len(head) >= 9:
        n = int.from_bytes(head[:8], "little")
        if 2 <= n <= 100 * 1024 * 1024 and head[8:9] == b"{":
            return "safetensors header"
    for magic, name in ((b"\x1f\x8b", "gzip"), (b"\xfd7zXZ\x00", "xz"), (b"\x28\xb5\x2f\xfd", "zstd"),
                        (b"7z\xbc\xaf\x27\x1c", "7z"), (b"Rar!\x1a\x07", "rar")):
        if head.startswith(magic):
            return f"{name} compressed stream (contents unscannable)"
    if head[:3] == b"BZh" and head[4:10] == b"\x31\x41\x59\x26\x53\x59":
        return "bzip2 compressed stream (contents unscannable)"
    if head[257:262] == b"ustar" or tar_v7_checksum(head):
        return "tar archive (contents unscannable)"
    return None


def bom_encoding(data: bytes) -> Optional[str]:
    if data.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig"
    if data.startswith((b"\xff\xfe\x00\x00", b"\x00\x00\xfe\xff")):
        return "utf-32"
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return "utf-16"
    return None


def looks_text(data: bytes) -> bool:
    enc = bom_encoding(data)
    if enc:
        try:
            data.decode(enc)
            return True
        except UnicodeDecodeError:
            return False
    if b"\0" not in data:
        try:
            data.decode("utf-8")
            return True
        except UnicodeDecodeError:
            return False
    for enc in ("utf-16-le", "utf-16-be"):
        try:
            text = data.decode(enc)
        except UnicodeDecodeError:
            continue
        printable = sum(1 for ch in text if ch.isprintable() or ch in "\r\n\t")
        if text and printable / len(text) >= 0.95:
            return True
    return False


def strip_format_chars(norm: str) -> str:
    """Drop every Unicode category Cf character (zero-width, bidi, tag): they can split a token."""
    bad = {c for c in set(norm) if not c.isascii() and unicodedata.category(c) == "Cf"}
    return norm.translate(dict.fromkeys(map(ord, bad))) if bad else norm


def text_views(data: bytes) -> List[Tuple[str, str]]:
    """Every readable rendering of the bytes. A NUL byte never removes a file from the text rules."""
    views: List[Tuple[str, str]] = []
    enc = bom_encoding(data)
    primary = None
    if enc:
        try:
            primary = data.decode(enc)
        except UnicodeDecodeError:
            primary = None
    views.append(("", primary if primary is not None else data.decode("utf-8", "replace")))
    if b"\0" in data:
        views.append((" [NUL-stripped view]", data.replace(b"\0", b"").decode("utf-8", "replace")))
    if looks_text(data):  # binaries decode to noise; NFKC adds nothing there
        for label, text in list(views):
            if not text.isascii():
                norm = strip_format_chars(unicodedata.normalize("NFKC", text))
                if norm != text:
                    views.append((label + " [NFKC view]", norm))
    return views


def encoded_payload(text: str) -> Optional[int]:
    if len(text) < ENCODED_PAYLOAD_LIMIT:
        return None
    longest = max((len(m.group()) for m in ENCODED_RUN_RE.finditer(text)), default=0)
    if longest >= ENCODED_PAYLOAD_LIMIT:
        return longest
    run = 0
    for line in text.splitlines():
        s = line.strip()
        if len(s) >= 16 and NOT_ENCODED_RE.search(s) is None:
            run += len(line)
            if run >= ENCODED_PAYLOAD_LIMIT:
                return run
        else:
            run = 0
    return None


def iter_matches(text: str, rx: "re.Pattern[str]", low: Optional[str] = None) -> Iterator[Tuple[int, str]]:
    if low is not None and not anchors_hit(rx, low):
        return
    # Matches are rare, so counting newlines per match beats indexing a 64 MiB view.
    for m in rx.finditer(text):
        start = text.rfind("\n", 0, m.start()) + 1
        end = text.find("\n", m.end())
        yield text.count("\n", 0, m.start()) + 1, text[start:end if end >= 0 else len(text)]


# ---------------------------------------------------------------------------
# S2 python AST
# ---------------------------------------------------------------------------
Hit = Tuple[str, int, str, bool]


def _has_grad_attr(node: ast.AST) -> bool:
    return any(isinstance(x, ast.Attribute) and x.attr == "grad" for x in ast.walk(node))


class PyAnalyzer:
    """Resolve import / assignment aliases, then map every reference to its canonical name."""

    def __init__(self, tree: ast.AST, line_offset: int = 0, depth: int = 0):
        self.tree = tree
        self.off = line_offset
        self.depth = depth
        self.alias: Dict[str, str] = {}
        self.consts: Dict[str, str] = {}
        self.optimizers: Set[str] = set()
        self.called: Set[int] = set()
        self.dyn_bound: Set[str] = set()
        self.hits: List[Hit] = []

    # -- binding ------------------------------------------------------------
    def run(self) -> List[Hit]:
        nodes = list(ast.walk(self.tree))
        for n in nodes:
            if isinstance(n, ast.Import):
                for a in n.names:
                    if a.asname:
                        self.alias[a.asname] = a.name
                    else:
                        root = a.name.split(".")[0]
                        self.alias[root] = root
            elif isinstance(n, ast.ImportFrom):
                mod = n.module or ""
                for a in n.names:
                    if a.name != "*":
                        self.alias[a.asname or a.name] = f"{mod}.{a.name}" if mod else a.name
        for _ in range(4):  # propagate simple assignments to a fixpoint
            changed = False
            for n in nodes:
                value = getattr(n, "value", None)
                if isinstance(n, ast.Assign):
                    targets = n.targets
                elif isinstance(n, (ast.AnnAssign, ast.NamedExpr)) and value is not None:
                    targets = [n.target]
                else:
                    continue
                if value is None:
                    continue
                for t in targets:
                    changed |= self._bind_target(t, value)
            if not changed:
                break
        self.called = {id(n.func) for n in nodes if isinstance(n, ast.Call)}
        for n in nodes:
            self._check(n)
        return self.hits

    def _bind_target(self, target: ast.AST, value: ast.AST) -> bool:
        if isinstance(target, ast.Name):
            return self._bind(target.id, value)
        seq = (ast.Tuple, ast.List)
        if isinstance(target, seq) and isinstance(value, seq) and len(target.elts) == len(value.elts) \
                and not any(isinstance(e, ast.Starred) for e in target.elts + value.elts):
            changed = False
            for t, v in zip(target.elts, value.elts):
                changed |= self._bind_target(t, v)
            return changed
        return False

    def _bind(self, name: str, value: ast.AST) -> bool:
        before = (self.alias.get(name), self.consts.get(name), name in self.optimizers, name in self.dyn_bound)
        if (isinstance(value, ast.Call) and self._dynamic_reflect(value)) or (
                isinstance(value, ast.Name) and value.id in self.dyn_bound):
            self.dyn_bound.add(name)  # f = getattr(x, unreadable); f() -- the call goes through the name
        s = self.fold(value)
        if s is not None:
            self.consts[name] = s
        if isinstance(value, (ast.Name, ast.Attribute, ast.Subscript)) or (
                isinstance(value, ast.Call) and self._reflective(value)):
            c = self.resolve(value)
            if c != "?" and c != name:
                self.alias[name] = c
        if isinstance(value, ast.Call) and self._is_optimizer(self.resolve(value.func)):
            self.optimizers.add(name)
        return before != (self.alias.get(name), self.consts.get(name), name in self.optimizers,
                          name in self.dyn_bound)

    def _reflective(self, call: ast.Call) -> bool:
        f = self.resolve(call.func)
        return f in GETATTR_FUNCS or f in IMPORT_FUNCS or f.endswith(("__getattribute__", "__getattr__"))

    def _dynamic_reflect(self, call: ast.Call) -> bool:
        """getattr / __getattribute__ whose attribute name does not fold to a constant."""
        if self.resolve(call.func) in GETATTR_FUNCS and len(call.args) >= 2:
            return self.fold(call.args[1]) is None
        if isinstance(call.func, ast.Attribute) and call.func.attr in ("__getattribute__", "__getattr__") and call.args:
            return self.fold(call.args[0]) is None
        return False

    def _dynamic_subscript(self, node: ast.Subscript) -> bool:
        """vars(x)[k] / x.__dict__[k] whose key does not fold to a constant."""
        base = node.value
        is_ns = (isinstance(base, ast.Call) and self.resolve(base.func) == "vars") or (
            isinstance(base, ast.Attribute) and base.attr == "__dict__")
        return is_ns and self.fold(node.slice) is None

    @staticmethod
    def _is_optimizer(c: str) -> bool:
        return any(c == p or c.startswith(p + ".") for p in (T_OPTIM,) + OPT_LIBS)

    # -- folding and resolution ----------------------------------------------
    def fold(self, node: ast.AST, depth: int = 0) -> Optional[str]:
        if depth > 40:
            return None
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.Name):
            return self.consts.get(node.id)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            left, right = self.fold(node.left, depth + 1), self.fold(node.right, depth + 1)
            return left + right if left is not None and right is not None else None
        if isinstance(node, ast.JoinedStr):
            parts = [self.fold(v.value if isinstance(v, ast.FormattedValue) else v, depth + 1) for v in node.values]
            return "".join(parts) if all(p is not None for p in parts) else None  # type: ignore[arg-type]
        if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Slice):
            s, sl = self.fold(node.value, depth + 1), node.slice
            step = None
            if isinstance(sl.step, ast.Constant):
                step = sl.step.value
            elif isinstance(sl.step, ast.UnaryOp) and isinstance(sl.step.op, ast.USub) \
                    and isinstance(sl.step.operand, ast.Constant) and sl.step.operand.value == 1:
                step = -1
            if s is not None and sl.lower is None and sl.upper is None and step == -1:
                return s[::-1]
        if isinstance(node, ast.Call) and not node.keywords:
            if isinstance(node.func, ast.Attribute) and node.func.attr in ("lower", "upper") and not node.args:
                s = self.fold(node.func.value, depth + 1)
                if s is not None:
                    return s.lower() if node.func.attr == "lower" else s.upper()
            if self.resolve(node.func) == "chr" and len(node.args) == 1 and isinstance(node.args[0], ast.Constant) \
                    and type(node.args[0].value) is int:
                try:
                    return chr(node.args[0].value)
                except (ValueError, OverflowError, TypeError):
                    return None
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "join" \
                and len(node.args) == 1 and isinstance(node.args[0], (ast.List, ast.Tuple)):
            sep = self.fold(node.func.value, depth + 1)
            items = [self.fold(e, depth + 1) for e in node.args[0].elts]
            if sep is not None and all(i is not None for i in items):
                return sep.join(items)  # type: ignore[arg-type]
        return None

    def resolve(self, node: ast.AST, depth: int = 0) -> str:
        if depth > 50:
            return "?"
        if isinstance(node, ast.Name):
            return self.alias.get(node.id, node.id)
        if isinstance(node, ast.Attribute):
            return f"{self.resolve(node.value, depth + 1)}.{node.attr}"
        if isinstance(node, ast.Subscript):
            key = self.fold(node.slice, depth + 1)
            base = node.value
            if key is not None and self.resolve(base, depth + 1) == "sys.modules":
                return key
            if key is not None and isinstance(base, ast.Call) and self.resolve(base.func, depth + 1) == "vars" and base.args:
                return f"{self.resolve(base.args[0], depth + 1)}.{key}"
            if key is not None and isinstance(base, ast.Attribute) and base.attr == "__dict__":
                return f"{self.resolve(base.value, depth + 1)}.{key}"
            return "?"
        if isinstance(node, ast.Call):
            f = self.resolve(node.func, depth + 1)
            if f in GETATTR_FUNCS and len(node.args) >= 2:
                name = self.fold(node.args[1])
                if name is not None:
                    return f"{self.resolve(node.args[0], depth + 1)}.{name}"
            if f in IMPORT_FUNCS and node.args:
                name = self.fold(node.args[0])
                if name is not None:
                    return name
            if isinstance(node.func, ast.Attribute) and node.func.attr in ("__getattribute__", "__getattr__") and node.args:
                name = self.fold(node.args[0])
                if name is not None:
                    return f"{self.resolve(node.func.value, depth + 1)}.{name}"
            if isinstance(node.func, ast.Call) and self.resolve(node.func.func, depth + 1) in CALLER_FUNCS and node.func.args:
                name = self.fold(node.func.args[0])
                if name is not None:
                    return f"?.{name}"
        return "?"

    # -- checks ----------------------------------------------------------------
    def _hit(self, rule: str, node: ast.AST, detail: str, exemptible: bool) -> None:
        self.hits.append((rule, getattr(node, "lineno", 0) + self.off, detail, exemptible))

    def _flag_canon(self, c: str, node: ast.AST, detail: str) -> None:
        parts = c.split(".")
        last = parts[-1]
        if c == T_OPTIM or c.startswith(T_OPTIM + ".") or (parts[0] == "optim" and len(parts) > 1 and last[:1].isupper()):
            self._hit("torch-optim", node, detail, False)
        elif self._is_optimizer(c):
            self._hit("optimizer-library", node, detail, False)
        if c in AUTOGRAD_FUNCS:
            self._hit("autograd-grad", node, detail, True)
        if last == "backward":
            self._hit("backward-call", node, detail, True)
        if last == "zero_grad":
            self._hit("optimizer-step", node, detail, True)
        if last == TRAINING_ARGS:
            self._hit("training-arguments", node, detail, False)
        if len(parts) > 1 and parts[0] != "?" and last.endswith("Trainer") and last[:1].isupper():
            self._hit("trainer-call", node, detail, False)

    def _check(self, n: ast.AST) -> None:
        if isinstance(n, (ast.Name, ast.Attribute)) and isinstance(n.ctx, ast.Load) and id(n) not in self.called \
                and self.resolve(n) in EXEC_REFS:
            self._hit("dynamic-code", n, f"{self.resolve(n)} referenced without a direct call "
                      "(aliased, stored or passed on, so its argument cannot be read)", False)
        if isinstance(n, ast.Import):
            for a in n.names:
                self._flag_canon(a.name, n, f"import {a.name}")
        elif isinstance(n, ast.ImportFrom):
            mod = n.module or ""
            for a in n.names:
                c = mod if a.name == "*" else (f"{mod}.{a.name}" if mod else a.name)
                self._flag_canon(c, n, f"from {mod or '.'} import {a.name}")
        elif isinstance(n, ast.Attribute):
            c = self.resolve(n)
            self._flag_canon(c, n, f"use of {c}")
        elif isinstance(n, ast.Name) and n.id in self.alias and self.alias[n.id] != n.id:
            c = self.alias[n.id]
            self._flag_canon(c, n, f"use of {n.id} (alias of {c})")
        elif isinstance(n, ast.Call):
            self._check_call(n)
        elif isinstance(n, ast.AugAssign) and _has_grad_attr(n.value):
            self._hit("manual-gradient-step", n, "in-place update computed from .grad", True)
        elif isinstance(n, ast.Assign) and _has_grad_attr(n.value) and any(
                isinstance(t, (ast.Attribute, ast.Subscript)) for t in n.targets):
            self._hit("manual-gradient-step", n, "parameter assignment computed from .grad", True)
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == "backward":
            self._hit("backward-call", n, "definition of a backward method", True)
        elif isinstance(n, ast.Constant) and isinstance(n.value, str):
            for lineno, line in enumerate(n.value.splitlines() or [n.value]):
                low = prefilter_view(line)
                for rule, rx, ex in S2_REGEX:
                    if anchors_hit(rx, low) and rx.search(line):
                        self.hits.append((rule, getattr(n, "lineno", 0) + self.off + lineno,
                                          f"string literal: {_snip(line)}", ex))

    def _check_call(self, n: ast.Call) -> None:
        f = self.resolve(n.func)
        if isinstance(n.func, ast.Subscript):
            c = self.resolve(n.func)
            if c != "?":
                self._flag_canon(c, n, f"call through namespace lookup {c}")
                if c.rsplit(".", 1)[-1] == "step":
                    self._hit("optimizer-step", n, f"call through namespace lookup {c}", True)
            elif self._dynamic_subscript(n.func):
                self._hit("dynamic-code", n, "call through vars()/__dict__ with a key that cannot be read statically",
                          False)
        func = n.func.value if isinstance(n.func, ast.NamedExpr) else n.func
        if isinstance(func, (ast.List, ast.Tuple, ast.Subscript)):  # [getattr(x, d)][0](...)
            func = next((e for e in ast.walk(func) if isinstance(e, ast.Call) and self._dynamic_reflect(e)), func)
        if (isinstance(func, ast.Call) and self._dynamic_reflect(func)) or (
                isinstance(func, ast.Name) and func.id in self.dyn_bound):
            self._hit("dynamic-code", n, "call of an attribute whose name cannot be read statically", False)
        if f in GETATTR_FUNCS or isinstance(n.func, ast.Call) or (
                isinstance(n.func, ast.Attribute) and n.func.attr in ("__getattribute__", "__getattr__")):
            c = self.resolve(n)
            if c != "?":
                self._flag_canon(c, n, f"reflective access to {c}")
        if f == "setattr" and len(n.args) >= 2:
            attr = self.fold(n.args[1])
            if attr is not None:
                self._flag_canon(attr, n, f"reflective definition of {attr}")
            else:
                self._hit("dynamic-code", n, "setattr with an attribute name that cannot be read statically", False)
        self._check_reflective_name(n, f)
        last = f.rsplit(".", 1)[-1]
        if last == "step" and isinstance(n.func, ast.Attribute):
            recv = n.func.value
            recv_name = recv.id if isinstance(recv, ast.Name) else ""
            if recv_name in self.optimizers or re.search(r"opt", self.resolve(recv), re.I):
                self._hit("optimizer-step", n, f"call to {f}()", True)
        if last.endswith("Trainer") and last[:1].isupper():
            self._hit("trainer-call", n, f"call to {f}()", False)
        if f in EXEC_FUNCS and n.args:
            self._check_exec(n)
        if isinstance(n.func, ast.Attribute) and n.func.attr.endswith("_") and not n.func.attr.startswith("_") \
                and any(_has_grad_attr(a) for a in list(n.args) + [k.value for k in n.keywords]):
            self._hit("manual-gradient-step", n, f"in-place {n.func.attr}() computed from .grad", True)

    def _check_reflective_name(self, n: ast.Call, f: str) -> None:
        """getattr-style access whose name is not readable, on something that may be a loss or optimizer."""
        name_arg: Optional[ast.AST] = None
        recv: Optional[ast.AST] = None
        if f in GETATTR_FUNCS and len(n.args) >= 2:
            recv, name_arg = n.args[0], n.args[1]
        elif isinstance(n.func, ast.Attribute) and n.func.attr in ("__getattribute__", "__getattr__") and n.args:
            recv, name_arg = n.func.value, n.args[0]
        elif isinstance(n.func, ast.Call) and n.func.args and n.args \
                and self.resolve(n.func.func) in CALLER_FUNCS:
            recv, name_arg = n.args[0], n.func.args[0]
            if self.fold(name_arg) is None and (
                    self.resolve(n.func.func).endswith("methodcaller") or id(n) in self.called):
                self._hit("dynamic-code", n, "methodcaller/attrgetter call with a name that cannot be read "
                          "statically", False)
                return
        if recv is None or name_arg is None or self.fold(name_arg) is not None:
            return
        if id(n) in self.called and not isinstance(n.func, ast.Call):
            return  # getattr(x, name)() is reported on the outer call by _check_call
        rname = self.resolve(recv)
        if (isinstance(recv, ast.Name) and recv.id in self.optimizers) or self._is_optimizer(rname) \
                or re.search(r"loss|opt", rname, re.I):
            self._hit("dynamic-code", n, "reflective call with a name that cannot be read statically "
                      "on a loss/optimizer object", False)

    def _check_exec(self, n: ast.Call) -> None:
        code = self.fold(n.args[0])
        if code is None:
            self._hit("dynamic-code", n, "exec/eval/compile of code that cannot be read statically", False)
            return
        if self.depth >= 3:
            self._hit("dynamic-code", n, "exec nested too deep to analyse", False)
            return
        try:
            sub = ast.parse(code)
        except (SyntaxError, ValueError):
            return
        line = getattr(n, "lineno", 1) + self.off - 1
        for rule, hit_line, detail, ex in PyAnalyzer(sub, line, self.depth + 1).run():
            self.hits.append((rule, hit_line, f"inside exec'd string: {detail}", ex))


def notebook_code(text: str) -> str:
    """Join the code cells of a notebook into one Python source. Raises ValueError if it is not a notebook."""
    try:
        nb = json.loads(text)
        cells = nb["cells"] if "cells" in nb else [c for ws in nb["worksheets"] for c in ws["cells"]]
        sources = []
        for cell in cells:
            if cell.get("cell_type") != "code":
                continue
            src = cell.get("source", cell.get("input", ""))
            sources.append(_cell_python("".join(src) if isinstance(src, list) else src))
    except (ValueError, KeyError, TypeError, AttributeError, RecursionError) as exc:
        raise ValueError(f"not a readable notebook: {type(exc).__name__}") from exc
    return "\n".join(sources)


def _line_magic_python(body: str) -> str:
    """%time / %timeit / %prun / %debug ... run the rest of the line as Python: keep it.

    Drop the magic name, then leading words (options such as -n 10) until the rest parses."""
    words = body.split(None, 1)
    rest = words[1].strip() if len(words) > 1 else ""
    while rest:
        try:
            ast.parse(rest)
            return rest
        except (SyntaxError, ValueError, RecursionError, MemoryError):
            parts = rest.split(None, 1)
            rest = parts[1] if len(parts) > 1 else ""
    return "pass"


def _blank_magics(lines: List[str], whole: bool = False) -> str:
    out = []
    for line in lines:
        body = line.lstrip().rstrip("\r\n")
        indent = line[:len(line) - len(line.lstrip())]
        if whole or body[:1] in ("!", "?") or body[:2] == "%%":
            out.append(indent + "pass")
        elif body[:1] == "%":
            out.append(indent + _line_magic_python(body))
        else:
            out.append(line.rstrip("\r\n"))
    return "\n".join(out)


def _cell_python(src: str) -> str:
    """IPython lines are not Python: blank them. A %% cell keeps its body when the body parses."""
    lines = src.splitlines()
    if lines and lines[0].startswith("%%"):
        lines[0] = ""
        body = _blank_magics(lines)
        try:
            ast.parse(body)
            return body
        except (SyntaxError, ValueError, RecursionError, MemoryError):
            return _blank_magics(lines, whole=True)
    return _blank_magics(lines)


def py_findings(text: str) -> List[Hit]:
    """Raises SyntaxError / ValueError when the text is not Python."""
    return PyAnalyzer(ast.parse(text)).run()


# ---------------------------------------------------------------------------
# scanner
# ---------------------------------------------------------------------------
class Scanner:
    def __init__(self, repo: Path, allowlist: dict, lfs_patterns: List[str],
                 fail_closed: bool, verbose: bool):
        self.repo = repo
        self.allowlist = allowlist
        self.lfs = lfs_patterns
        self.fail_closed = fail_closed
        self.verbose = verbose
        self.findings: List[Finding] = []
        self.warnings: List[str] = []
        self.exempted: List[str] = []
        self.notes: List[str] = []
        self.files_scanned = 0
        self.bytes_scanned = 0

    def add(self, rule: str, path: str, line: int, detail: str) -> None:
        f = Finding(rule, path, line, detail)
        self.findings.append(f)
        if self.fail_closed:
            raise FirstViolation(f)

    # -- S1 -----------------------------------------------------------------
    def check_s1(self, rel: str, c: Content) -> str:
        """Return "clean", "allowlisted" (lfs-routed, sha-pinned weight) or "blocked"."""
        ext = os.path.splitext(rel)[1].lower()
        reasons: List[str] = []
        if ext in BANNED_EXT:
            reasons.append(f"weight/cache extension {ext}")
        magic = weight_magic(c.head, c.data)
        if magic:
            reasons.append(f"{magic} content under extension '{ext or '(none)'}'")
        if c.data is None:
            reasons.append(f"content of {c.size} B exceeds the {MAX_TEXT_BYTES} B scan limit")
        elif c.size > BIN_SIZE_LIMIT and not looks_text(c.data):
            reasons.append(f"binary file of {c.size / 1048576:.1f} MiB (limit 5 MiB)")
        if not reasons:
            return "clean"
        pinned = self.allowlist.get(rel)
        if pinned is not None and lfs_matches(rel, self.lfs):
            if c.sha256_of() == pinned:
                self.exempted.append(f"S1 allowlisted+lfs: {rel}")
                return "allowlisted"
            reasons.append("allowlist sha256 does not match file content")
        elif pinned is not None:
            reasons.append("allowlisted but no matching filter=lfs rule in .gitattributes")
        if c.data is None or not looks_text(c.data):
            # The file is blocked already; S2/S3 on binary noise cannot change the verdict.
            reasons.append("S2/S3 text rules not run on this blocked binary")
        self.add("S1-weight-binary", rel, 0, "; ".join(reasons))
        return "blocked"

    # -- S2 / S3 on text ------------------------------------------------------
    def check_text(self, rel: str, size: int, views: List[Tuple[str, str]]) -> None:
        ext = os.path.splitext(rel)[1].lower()
        test_file = is_test_path(rel)
        has_marker = EXEMPT_RE.search(views[0][1]) is not None
        exempt_file = test_file and has_marker
        if has_marker and not test_file:
            self.warnings.append(f"{rel}: exemption marker ignored, not a test file")
        seen: Set[Tuple[str, int]] = set()
        lows = [prefilter_view(text) for _, text in views]

        def emit(rule: str, line: int, detail: str, exemptible: bool = False) -> None:
            if (rule, line) in seen:
                return
            seen.add((rule, line))
            if exemptible and exempt_file:
                self.exempted.append(f"S2 {rule} exempt (test, marker): {rel}:{line}")
                return
            suffix = " (test file lacks the exemption marker)" if exemptible and test_file else ""
            self.add(rule, rel, line, detail + suffix)

        parsed = False
        notebook = ext == ".ipynb"
        if notebook:
            parsed = True  # judged here; the JSON itself is never read as Python
            try:
                code = notebook_code(views[0][1])
                if len(code) > AST_MAX_BYTES:
                    raise ValueError(f"code cells exceed the {AST_MAX_BYTES} B AST limit")
                try:
                    hits = py_findings(code)
                except (SyntaxError, ValueError, RecursionError, MemoryError):
                    hits = py_findings(strip_format_chars(unicodedata.normalize("NFKC", code)))
                for rule, line, detail, ex in hits:
                    emit(f"S2-{rule}", line, detail + " [notebook code cells]", ex)
            except (SyntaxError, ValueError, RecursionError, MemoryError) as exc:
                emit("S2-unanalysable-python", 0, f"notebook code cells cannot be analysed ({exc}); aliases cannot be checked")
        elif size <= AST_MAX_BYTES:
            for label, text in views:
                try:
                    hits = py_findings(text)
                except (SyntaxError, ValueError, RecursionError, MemoryError):
                    continue
                for rule, line, detail, ex in hits:
                    emit(f"S2-{rule}", line, detail + label, ex)
                parsed = True
                break
        if not parsed:
            # Regex cannot follow aliases, so Python the AST could not read fails closed.
            python_shaped = ext in PY_EXT or re.match(r"#![^\n]*python", views[0][1]) is not None
            if python_shaped:
                why = f"exceeds the {AST_MAX_BYTES} B AST limit" if size > AST_MAX_BYTES else "does not parse"
                emit("S2-unanalysable-python", 0, f"Python source {why}; aliases cannot be checked")
            elif size > AST_MAX_BYTES:
                for label, text in views:
                    for line, src in iter_matches(text, ML_IMPORT_RE):
                        emit("S2-unanalysable-python", line,
                             f"ML import in text over the AST limit; aliases cannot be checked: {_snip(src)}{label}")
        if not parsed or notebook:
            for (label, text), low in zip(views, lows):
                for rule, rx, ex in S2_REGEX:
                    for line, src in iter_matches(text, rx, low):
                        emit(f"S2-{rule}", line, _snip(src) + label, ex)
        for (label, text), low in zip(views, lows):
            for rule, rx in S2_NAME_REGEX:
                for line, src in iter_matches(text, rx, low):
                    emit(f"S2-{rule}", line, _snip(src) + label)
            for rule, rx in S3_REGEX:
                for line, src in iter_matches(text, rx, low):
                    emit(f"S3-{rule}", line, _snip(src) + label)
        run = encoded_payload(views[0][1])
        if run is not None:
            emit("S1-encoded-payload", 0, f"base64/hex run of {run} chars (limit {ENCODED_PAYLOAD_LIMIT})")

    # -- one file -------------------------------------------------------------
    def check_path(self, rel: str) -> None:
        if S2_PATH_REGEX.search(rel):
            self.add("S2-closed-data-generator", rel, 0, "path names closed data-generation code")
        for rule, rx in S3_REGEX:
            if rx.search(rel):
                self.add(f"S3-{rule}", rel, 0, "path contains a proprietary token")

    def scan_content(self, rel: str, c: Content, label: str = "") -> None:
        self.files_scanned += 1
        self.bytes_scanned += c.size
        if self.verbose:
            print(f"  scan {rel}{label} ({c.size} B)")
        self.check_path(rel)
        verdict = self.check_s1(rel, c)
        if verdict == "allowlisted" or c.data is None:
            return
        if verdict == "blocked" and not looks_text(c.data):
            return
        self.check_text(rel, c.size, text_views(c.data))

    def scan_metadata(self, where: str, data: bytes) -> None:
        """S3 and closed-name rules on a commit or tag object (message, author, tagger)."""
        for label, text in text_views(data):
            for rule, rx in S3_REGEX + S2_NAME_REGEX:
                for line, src in iter_matches(text, rx):
                    prefix = "S3" if (rule, rx) in S3_REGEX else "S2"
                    self.add(f"{prefix}-{rule}", where, line, _snip(src) + label)

    def scan_worktree_file(self, rel: str) -> None:
        path = self.repo / rel
        try:
            st = os.lstat(path)
        except OSError as exc:
            raise ScanError(f"cannot stat {rel}: {exc}") from exc
        if stat.S_ISLNK(st.st_mode):
            # a link has no bytes to leak, but its name is still checked
            self.scan_content(rel, Content(0, b"", b"", lambda: hashlib.sha256(b"").hexdigest()), " (symlink)")
            return
        if not stat.S_ISREG(st.st_mode):
            return

        def sha_stream() -> str:
            h = hashlib.sha256()
            try:
                with open(path, "rb") as fh:
                    for chunk in iter(lambda: fh.read(1 << 20), b""):
                        h.update(chunk)
            except OSError as exc:
                raise ScanError(f"cannot read {rel}: {exc}") from exc
            return h.hexdigest()

        try:
            with open(path, "rb") as fh:
                if st.st_size <= MAX_TEXT_BYTES:
                    whole = fh.read()
                    data: Optional[bytes] = whole
                    head = whole[:HEAD_BYTES]
                else:
                    data, head = None, fh.read(HEAD_BYTES)
        except OSError as exc:
            raise ScanError(f"cannot read {rel}: {exc}") from exc
        if data is not None:
            blob = data
            self.scan_content(rel, Content(len(blob), head, blob, lambda: hashlib.sha256(blob).hexdigest()))
        else:
            self.scan_content(rel, Content(st.st_size, head, None, sha_stream))


def _snip(line: str) -> str:
    line = line.strip()
    return line if len(line) <= 140 else line[:137] + "..."


# ---------------------------------------------------------------------------
# file enumeration
# ---------------------------------------------------------------------------
def walk_all(repo: Path) -> Iterator[str]:
    for root, dirs, files in os.walk(repo, followlinks=False):
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS)
        for name in sorted(files):
            yield os.path.relpath(os.path.join(root, name), repo)
        for d in list(dirs):  # symlinked dirs are listed as files by name only
            p = os.path.join(root, d)
            if os.path.islink(p):
                dirs.remove(d)
                yield os.path.relpath(p, repo)


def git_visible(repo: Path) -> List[str]:
    return split_z(git(repo, "ls-files", "-z", "--cached", "--others", "--exclude-standard"))


def diff_files(repo: Path) -> Tuple[List[str], List[str]]:
    staged = split_z(git(repo, "diff", "-z", "--name-only", "--relative", "--cached", "--diff-filter=ACMRT"))
    unstaged = split_z(git(repo, "diff", "-z", "--name-only", "--relative", "--diff-filter=ACMRT"))
    untracked = split_z(git(repo, "ls-files", "-z", "--others", "--exclude-standard"))
    return staged, sorted(set(unstaged) | set(untracked))


# ---------------------------------------------------------------------------
# commit / blob scanning (pre-push and history)
# ---------------------------------------------------------------------------
def changed_blobs(repo: Path, commit: str) -> Iterator[Tuple[str, str]]:
    """(path, blob oid) for every file a commit adds or changes, against every parent."""
    raw = git(repo, "diff-tree", "-r", "-z", "--no-commit-id", "--root", "-m", "--no-renames", "--raw", commit)
    toks = raw.split(b"\0")
    i = 0
    while i < len(toks):
        meta = toks[i]
        if not meta:
            i += 1
            continue
        if not meta.startswith(b":") or i + 1 >= len(toks):
            raise ScanError(f"unexpected git diff-tree output for {commit}: {meta[:80]!r}")
        fields = meta[1:].split()
        path = toks[i + 1].decode("utf-8", "surrogateescape")
        i += 2
        if fields[4][:1] == b"D" or fields[1] == b"160000":  # deletion or submodule gitlink
            continue
        yield path, fields[3].decode()


def object_bytes(cat: CatFile, name: str) -> bytes:
    content = cat.get(name)[1]
    if content.data is None:
        raise ScanError(f"git object {name} of {content.size} B exceeds the scan limit")
    return content.data


class HistoryScan:
    """Scan commits: their metadata, the blobs each one introduces, and any other new blob."""

    def __init__(self, sc: Scanner, cat: CatFile):
        self.sc = sc
        self.cat = cat
        self.done: Set[Tuple[str, str]] = set()
        self.blobs_seen: Set[str] = set()
        self.lfs_cache: Dict[str, List[str]] = {}

    def lfs_at(self, commit: str) -> List[str]:
        if commit not in self.lfs_cache:
            got = self.cat.read(f"{commit}:.gitattributes", missing_ok=True)
            text = got[1].data.decode("utf-8", "replace") if got and got[0] == "blob" and got[1].data else ""
            self.lfs_cache[commit] = parse_lfs_patterns(text)
        return self.lfs_cache[commit]

    def blob(self, path: str, oid: str, label: str) -> None:
        self.blobs_seen.add(oid)
        if (path, oid) in self.done:
            return
        self.done.add((path, oid))
        got = self.cat.get(oid)
        if got[0] != "blob":
            raise ScanError(f"{oid} at {path} is a {got[0]}, expected a blob")
        self.sc.scan_content(path, got[1], f" ({label} blob {oid[:12]})")

    def commit(self, commit: str, label: str) -> None:
        self.sc.scan_metadata(f"<{label} commit {commit[:12]}>", object_bytes(self.cat, commit))
        self.sc.lfs = self.lfs_at(commit)
        for path, oid in changed_blobs(self.sc.repo, commit):
            self.blob(path, oid, label)

    def leftover_objects(self, rev_args: List[str], label: str, tip: Optional[str]) -> None:
        """Blobs reachable in the range that no per-commit diff showed (belt and braces)."""
        listing = git(self.sc.repo, "rev-list", "--objects", *rev_args).decode("utf-8", "surrogateescape")
        paths: Dict[str, str] = {}
        for row in listing.splitlines():
            oid, _, path = row.partition(" ")
            if path and oid not in self.blobs_seen:
                paths[oid] = path
        if not paths:
            return
        if tip:
            self.sc.lfs = self.lfs_at(tip)
        check = git(self.sc.repo, "cat-file", "--batch-check=%(objecttype) %(objectname)",
                    input_bytes=("\n".join(paths) + "\n").encode())
        for row in check.decode().splitlines():
            typ, oid = row.split()
            if typ == "blob":
                self.blob(paths[oid], oid, label)
            else:
                self.sc.check_path(paths[oid])


def scan_history(sc: Scanner, cat: CatFile) -> None:
    hs = HistoryScan(sc, cat)
    for commit in git(sc.repo, "rev-list", "--all").decode().split():
        hs.commit(commit, "history")
    for ref in git(sc.repo, "for-each-ref", "--format=%(objectname) %(objecttype)", "refs/tags").decode().splitlines():
        oid, typ = ref.split()
        if typ == "tag":
            sc.scan_metadata(f"<history tag {oid[:12]}>", object_bytes(cat, oid))
    hs.leftover_objects(["--all"], "history", None)


def is_zero(sha: str) -> bool:
    return set(sha) == {"0"}


def parse_pre_push(stdin_text: str) -> List[Tuple[str, str, str, str]]:
    updates = []
    for lineno, line in enumerate(stdin_text.splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 4 or not SHA_RE.fullmatch(fields[1]) or not SHA_RE.fullmatch(fields[3]):
            raise ScanError(f"malformed pre-push stdin line {lineno}: {line!r} "
                            "(need '<local ref> <local sha> <remote ref> <remote sha>')")
        updates.append((fields[0], fields[1], fields[2], fields[3]))
    return updates


def remote_commits(repo: Path, remote: str) -> List[str]:
    """Commits we hold locally that the remote advertises (heads, tags, peeled tags). [] on any failure."""
    if not remote or remote.startswith("-"):
        return []
    try:
        proc = subprocess.run(["git", "-C", str(repo), "ls-remote", "--heads", "--tags", remote],
                              capture_output=True, check=False, timeout=30, stdin=subprocess.DEVNULL,
                              env={**os.environ, "GIT_TERMINAL_PROMPT": "0"})
    except (OSError, subprocess.TimeoutExpired):
        return []
    if proc.returncode != 0:
        return []
    shas = sorted({f.decode() for row in proc.stdout.splitlines() for f in row.split()[:1] if SHA_RE.fullmatch(f.decode(errors="replace"))})
    if not shas:
        return []
    check = git(repo, "cat-file", "--batch-check", input_bytes=("".join(f"{x}^{{commit}}\n" for x in shas)).encode())
    return sorted({row.split()[0] for row in check.decode().splitlines() if row.split()[1:2] == ["commit"]})


def scan_pre_push(sc: Scanner, cat: CatFile, updates: List[Tuple[str, str, str, str]], remote: str) -> None:
    repo = sc.repo
    hs = HistoryScan(sc, cat)
    for local_ref, local_sha, remote_ref, remote_sha in updates:
        if is_zero(local_sha):
            sc.notes.append(f"{remote_ref}: delete, nothing to scan")
            continue
        typ = git(repo, "cat-file", "-t", local_sha).decode().strip()
        if typ == "tag":
            sc.scan_metadata(f"<pushed tag {local_ref}>", object_bytes(cat, local_sha))
        tip = git(repo, "rev-parse", "--verify", "--quiet", f"{local_sha}^{{commit}}").decode().strip()
        if not tip:
            raise ScanError(f"{local_ref} -> {local_sha} is a {typ}, not a commit; refusing to push unscanned objects")
        if not is_zero(remote_sha) and git_ok(repo, "cat-file", "-e", f"{remote_sha}^{{commit}}"):
            rev_args = [f"{remote_sha}..{tip}"]
            how = f"{remote_sha[:12]}..{tip[:12]}"
        elif present := remote_commits(repo, remote):
            rev_args = [tip, "--not", *present]
            how = f"{tip[:12]} not on the remote's advertised refs (ls-remote)"
        else:
            rev_args = [tip]
            how = f"every commit reachable from {tip[:12]} (remote side unknown locally)"
        commits = git(repo, "rev-list", *rev_args).decode().split()
        sc.notes.append(f"{local_ref} -> {remote_ref}: {len(commits)} commit(s), range {how}")
        for commit in commits:
            hs.commit(commit, "pushed")
        hs.leftover_objects(rev_args, "pushed", tip)


# ---------------------------------------------------------------------------
# hook
# ---------------------------------------------------------------------------
def hook_script() -> str:
    cmd = f"{shlex.quote(sys.executable)} {shlex.quote(str(Path(__file__).resolve()))}"
    return f"""#!/bin/sh
{HOOK_SIGNATURE}
# Scans every commit and blob being pushed (git passes them on stdin), not the
# working tree. Exit 1 (violation) or 2 (scan error) blocks the push. A clean
# scan then runs the hook that was here before, with the same args and stdin.
hook_dir=$(dirname "$0")
top=$(git rev-parse --show-toplevel) || exit 2
payload=$(mktemp "${{TMPDIR:-/tmp}}/anti-leakage-pre-push.XXXXXX") || exit 2
trap 'rm -f "$payload"' EXIT
trap 'exit 2' HUP INT TERM
cat > "$payload" || exit 2
{cmd} --target-repo "$top" --pre-push --remote "$1" < "$payload"
rc=$?
if [ "$rc" -ne 0 ]; then
    echo "anti_leakage_lint: push blocked (exit $rc)" >&2
    exit "$rc"
fi
chained="$hook_dir/{CHAINED_HOOK}"
if [ -e "$chained" ]; then
    if [ ! -x "$chained" ]; then
        echo "anti_leakage_lint: $chained is not executable; push blocked" >&2
        exit 2
    fi
    "$chained" "$@" < "$payload"
    exit $?
fi
exit 0
"""


def install_hook(repo: Path) -> List[str]:
    if not is_git_repo(repo):
        raise ScanError(f"{repo} is not a git repository")
    hooks = Path(git(repo, "rev-parse", "--path-format=absolute", "--git-path", "hooks").decode().strip())
    hooks.mkdir(parents=True, exist_ok=True)
    hook, chained = hooks / "pre-push", hooks / CHAINED_HOOK
    log: List[str] = []
    if os.path.lexists(hook):
        # exact whole-line match: a foreign hook that merely mentions the signature is kept and chained
        ours = hook.is_file() and HOOK_SIGNATURE in hook.read_text(errors="replace").splitlines()
        if not ours:
            if os.path.lexists(chained):
                raise ScanError(f"{hook} is foreign and {chained} already exists; refusing to overwrite either")
            os.replace(hook, chained)
            log.append(f"moved existing {hook} to {chained}; it runs after a clean scan with the same args and stdin")
    tmp = hooks / ".pre-push.anti-leakage.tmp"
    tmp.write_text(hook_script())
    tmp.chmod(0o755)
    os.replace(tmp, hook)
    log.append(f"installed {hook}")
    if os.path.lexists(chained):
        log.append(f"chained hook: {chained}")
    return log


# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Anti-leakage gate for the open-source repo (rules S1, S2, S3).")
    ap.add_argument("--target-repo", default=".", help="repository to scan (default: current directory)")
    ap.add_argument("--fail-closed", action="store_true", help="stop at the first violation and exit 1")
    ap.add_argument("--verbose", action="store_true", help="list every scanned file and every exemption")
    ap.add_argument("--git-diff-only", action="store_true",
                    help="scan only staged and unstaged changes plus untracked files (pre-commit mode)")
    ap.add_argument("--respect-gitignore", action="store_true",
                    help="scan only tracked and untracked-not-ignored files. Default is a full walk, "
                         "because rsync-style sync copies ignored files too")
    ap.add_argument("--history", action="store_true",
                    help="also run S1/S2/S3 on every commit, path and blob reachable from any ref")
    ap.add_argument("--pre-push", action="store_true",
                    help="read git pre-push stdin and scan exactly the pushed commits and blobs")
    ap.add_argument("--remote", default="", help="remote name for --pre-push (the hook passes $1)")
    ap.add_argument("--allowlist", default=str(DEFAULT_ALLOWLIST),
                    help="JSON allowlist of pinned weight files (default: next to this script)")
    ap.add_argument("--report-only", action="store_true",
                    help="print violations but exit 0, with an explicit log. Cannot be combined with --fail-closed")
    ap.add_argument("--install-hook", action="store_true",
                    help="install the pre-push hook in the target repo, chaining any existing one, and exit")
    return ap


def usage_error(ap: argparse.ArgumentParser, msg: str) -> int:
    ap.print_usage(sys.stderr)
    print(f"anti_leakage_lint: usage error: {msg}", file=sys.stderr)
    return 2


def run_scan(args: argparse.Namespace, repo: Path, sc: Scanner) -> str:
    in_git = is_git_repo(repo)
    if (args.git_diff_only or args.history or args.pre_push) and not in_git:
        raise ScanError("--git-diff-only, --history and --pre-push need a git repository")
    if args.pre_push:
        mode = f"pre-push (pushed commits and blobs only, remote={args.remote or '(none)'})"
    elif args.git_diff_only:
        mode = "git-diff-only (staged index blobs + working tree changes + untracked)"
    elif args.respect_gitignore and in_git:
        mode = "git-visible (tracked + untracked, gitignored files NOT scanned)"
    else:
        mode = "full walk (every file except .git)"
    print(f"anti_leakage_lint: target={repo} mode={mode}{' +history' if args.history else ''} allowlist={args.allowlist}")
    try:
        if args.pre_push:
            updates = parse_pre_push(sys.stdin.read())
            if not updates:
                sc.notes.append("pre-push stdin carried no ref updates; nothing is being pushed")
            with CatFile(repo) as cat:
                scan_pre_push(sc, cat, updates, args.remote)
            return mode
        if args.git_diff_only:
            staged, rest = diff_files(repo)
            with CatFile(repo) as cat:
                for rel in staged:
                    sc.scan_content(rel, cat.get(f":{rel}")[1], " (staged)")
            for rel in rest:
                if os.path.lexists(repo / rel):
                    sc.scan_worktree_file(rel)
        else:
            files = git_visible(repo) if (args.respect_gitignore and in_git) else list(walk_all(repo))
            for rel in files:
                if os.path.lexists(repo / rel):
                    sc.scan_worktree_file(rel)
        if args.history:
            with CatFile(repo) as cat:
                scan_history(sc, cat)
    except FirstViolation:
        pass
    if sc.files_scanned == 0 and not (args.git_diff_only or args.history):
        raise ScanError("zero files scanned; refusing to report a clean result")
    return mode


def main(argv: Optional[List[str]] = None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)
    if args.report_only and args.fail_closed:
        return usage_error(ap, "--report-only and --fail-closed are mutually exclusive")
    if args.pre_push and (args.git_diff_only or args.history or args.respect_gitignore or args.install_hook):
        return usage_error(ap, "--pre-push cannot be combined with another scan mode or --install-hook")

    repo = Path(args.target_repo).resolve()
    try:
        if not repo.is_dir():
            raise ScanError(f"target repo is not a directory: {repo}")
        if args.install_hook:
            for line in install_hook(repo):
                print(line)
            return 0
        sc = Scanner(repo, load_allowlist(Path(args.allowlist)), load_lfs_patterns(repo),
                     args.fail_closed, args.verbose)
        run_scan(args, repo, sc)
    except ScanError as exc:
        print(f"anti_leakage_lint: ERROR: {exc}", file=sys.stderr)
        return 2

    for note in sc.notes:
        print(f"note: {note}")
    for w in sc.warnings:
        print(f"WARN {w}", file=sys.stderr)
    if args.verbose:
        for e in sc.exempted:
            print(f"  exempt {e}")
    for f in sc.findings:
        print(f.render())
    by_rule: dict = {}
    for f in sc.findings:
        key = f.rule.split("-", 1)[0]
        by_rule[key] = by_rule.get(key, 0) + 1
    print(f"summary: files_scanned={sc.files_scanned} bytes={sc.bytes_scanned} "
          f"violations={len(sc.findings)} by_rule={json.dumps(by_rule, sort_keys=True)} warnings={len(sc.warnings)}"
          + (" (stopped at first violation)" if args.fail_closed and sc.findings else ""))
    if sc.findings:
        if args.report_only:
            print(f"REPORT-ONLY: {len(sc.findings)} violation(s) found; exit forced to 0 by --report-only. "
                  "This is NOT a pass.", file=sys.stderr)
            print("RESULT: VIOLATIONS (report-only)")
            return 0
        print("RESULT: FAIL", file=sys.stderr)
        return 1
    print("RESULT: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
