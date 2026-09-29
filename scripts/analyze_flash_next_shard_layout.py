#!/usr/bin/env python3
"""Exact byte budget of a first-K-layers truncation of Qwen3.8-Flash-Next, from the safetensors headers only.

No weight is downloaded. For each shard we HTTP-Range-read the 8-byte length prefix and the JSON header
(a few hundred KB), which lists every tensor's dtype, shape and byte range. From that we compute:
  * bytes per component (embed, lm_head, each decoder layer, N-gram table, visual, mtp, ...)
  * which shard files a first-K truncation must open, and how many bytes inside them it must read.

Usage: python scripts/analyze_flash_next_shard_layout.py --repo Qwen/Qwen3.8-Flash-Next --out <json>
"""
import argparse
import collections
import concurrent.futures as cf
import json
import re
import struct
import urllib.request

KS = (16, 24, 32)


def http_range(url, a, b):
    req = urllib.request.Request(url, headers={"Range": f"bytes={a}-{b}", "User-Agent": "gen-zero-layout-probe"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def shard_header(repo, shard):
    url = f"https://huggingface.co/{repo}/resolve/main/{shard}"
    n = struct.unpack("<Q", http_range(url, 0, 7))[0]
    hdr = json.loads(http_range(url, 8, 8 + n - 1))
    hdr.pop("__metadata__", None)
    return shard, 8 + n, hdr


def classify(key):
    m = re.search(r"layers\.(\d+)\.(.*)", key)
    if m and ".ple." in key and "ngram_embedding" in key:
        return ("ngram_table", int(m.group(1)))
    if m and key.startswith(("model.language_model.", "model.layers.")):
        return ("layer", int(m.group(1)))
    if "visual" in key:
        return ("visual", None)
    if key.startswith("mtp"):
        return ("mtp", None)
    if "embed_tokens" in key:
        return ("embed_tokens", None)
    if key.startswith("lm_head"):
        return ("lm_head", None)
    return ("other_backbone", None)  # final norm, hyper_connection_mixer, ...


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="Qwen/Qwen3.8-Flash-Next")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    idx = json.loads(http_range(f"https://huggingface.co/{args.repo}/resolve/main/model.safetensors.index.json", 0, 10 ** 7))
    shards = sorted(set(idx["weight_map"].values()))
    with cf.ThreadPoolExecutor(8) as ex:
        heads = list(ex.map(lambda s: shard_header(args.repo, s), shards))
    comp_bytes = collections.Counter()      # (kind, layer) -> bytes
    comp_params = collections.Counter()
    comp_shards = collections.defaultdict(lambda: collections.Counter())  # (kind, layer) -> shard -> bytes
    shard_size = {}
    bytes_per = {"BF16": 2, "F16": 2, "F32": 4, "I64": 8, "F8_E4M3": 1, "U8": 1}
    for shard, base, hdr in heads:
        shard_size[shard] = base + max((v["data_offsets"][1] for v in hdr.values()), default=0)
        for key, meta in hdr.items():
            s, e = meta["data_offsets"]
            k = classify(key)
            comp_bytes[k] += e - s
            comp_params[k] += (e - s) // bytes_per[meta["dtype"]]
            comp_shards[k][shard] += e - s
    total_bytes = sum(comp_bytes.values())
    out = {"repo": args.repo, "n_shards": len(shards), "total_tensor_bytes": total_bytes,
           "total_params": sum(comp_params.values()), "components": {}, "truncations": {}}
    for kind in ("embed_tokens", "lm_head", "ngram_table", "visual", "mtp", "other_backbone"):
        ks = [k for k in comp_bytes if k[0] == kind]
        out["components"][kind] = {"bytes": sum(comp_bytes[k] for k in ks), "params": sum(comp_params[k] for k in ks),
                                   "layers_holding": sorted({k[1] for k in ks if k[1] is not None})}
    layer_keys = sorted(k for k in comp_bytes if k[0] == "layer")
    out["components"]["decoder_layers_excl_ngram"] = {"bytes": sum(comp_bytes[k] for k in layer_keys),
                                                      "params": sum(comp_params[k] for k in layer_keys),
                                                      "bytes_per_layer": {k[1]: comp_bytes[k] for k in layer_keys}}
    for K in KS:
        need = [k for k in comp_bytes if (k[0] == "layer" and k[1] < K) or k[0] in ("embed_tokens", "other_backbone")
                or (k[0] == "ngram_table" and k[1] < K)]
        need_no_ngram = [k for k in need if k[0] != "ngram_table"]
        shards_needed, shards_no_ngram = set(), set()
        for k in need:
            shards_needed |= set(comp_shards[k])
        for k in need_no_ngram:
            shards_no_ngram |= set(comp_shards[k])
        out["truncations"][str(K)] = {
            "bytes_with_ngram_table": sum(comp_bytes[k] for k in need),
            "params_with_ngram_table": sum(comp_params[k] for k in need),
            "bytes_without_ngram_table": sum(comp_bytes[k] for k in need_no_ngram),
            "params_without_ngram_table": sum(comp_params[k] for k in need_no_ngram),
            "shards_needed_with_ngram": len(shards_needed), "shards_needed_without_ngram": len(shards_no_ngram),
            "fraction_of_all_bytes_with_ngram": sum(comp_bytes[k] for k in need) / total_bytes}
    # shard ordering: does shard order follow layer order?
    layer_to_shards = {k[1]: sorted(comp_shards[k]) for k in layer_keys}
    out["layer_shard_spread"] = {str(i): len(v) for i, v in layer_to_shards.items()}
    out["layers_0_to_15_shards"] = len({s for i in range(16) for s in layer_to_shards.get(i, [])})
    json.dump(out, open(args.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "layer_shard_spread"}, indent=1))


if __name__ == "__main__":
    main()
