#!/usr/bin/env python3
"""Per-layer byte budget of a GGUF file from its header only (stdlib, no weights read).

GGUF lists every tensor with an absolute offset, so a first-K-layers slice is a set of byte ranges,
exactly like safetensors. Tensor size = next tensor's offset - this offset (last one: file end).
Usage: python inspect_gguf_layer_bytes.py <file.gguf> [<out.json>]
"""
import collections
import json
import os
import re
import struct
import sys

SCALAR = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}


def rd(f, fmt):
    return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]


def rd_str(f):
    return f.read(rd(f, "<Q")).decode("utf-8", "replace")


def rd_val(f, t):
    if t in SCALAR:
        return rd(f, SCALAR[t])
    if t == 8:
        return rd_str(f)
    if t == 9:
        et, n = rd(f, "<I"), rd(f, "<Q")
        vals = [rd_val(f, et) for _ in range(n)]
        return vals if n <= 16 else f"<array of {n}>"
    raise ValueError(f"bad gguf type {t}")


def main(path, out=None):
    with open(path, "rb") as f:
        assert f.read(4) == b"GGUF"
        version, n_t, n_kv = rd(f, "<I"), rd(f, "<Q"), rd(f, "<Q")
        meta = {}
        for _ in range(n_kv):
            k = rd_str(f)
            meta[k] = rd_val(f, rd(f, "<I"))
        infos = []
        for _ in range(n_t):
            name = rd_str(f)
            nd = rd(f, "<I")
            dims = [rd(f, "<Q") for _ in range(nd)]
            infos.append((name, dims, rd(f, "<I"), rd(f, "<Q")))
        header_end = f.tell()
    align = meta.get("general.alignment", 32)
    data_start = (header_end + align - 1) // align * align
    order = sorted(range(len(infos)), key=lambda i: infos[i][3])
    size = {}
    for j, i in enumerate(order):
        end = infos[order[j + 1]][3] if j + 1 < len(order) else os.path.getsize(path) - data_start
        size[infos[i][0]] = end - infos[i][3]
    arch = meta.get("general.architecture")
    per_block, other = collections.Counter(), collections.Counter()
    for name, sz in size.items():
        m = re.match(r"blk\.(\d+)\.", name)
        (per_block if m else other)[int(m.group(1)) if m else name] += sz
    n_blocks = meta.get(f"{arch}.block_count")
    res = {"file": os.path.basename(path), "gguf_version": version, "arch": arch, "block_count": n_blocks,
           "file_bytes": os.path.getsize(path), "header_bytes": data_start, "n_tensors": n_t,
           "non_block_tensors_bytes": dict(other), "blocks_with_tensors": len(per_block),
           "bytes_per_block": {str(k): per_block[k] for k in sorted(per_block)},
           "truncations": {}}
    for K in (16, 24, 32):
        b = sum(v for k, v in per_block.items() if k < K)
        emb = other.get("token_embd.weight", 0)
        res["truncations"][str(K)] = {"blocks_bytes": b, "plus_token_embd": b + emb,
                                      "fraction_of_file": (b + emb) / res["file_bytes"]}
    print(json.dumps(res, indent=1))
    if out:
        json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
