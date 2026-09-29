"""Canonical source-space identity for raw (pre-GCCA) extracted feature blocks.

A feature store (``<task>.npz`` written by ``benchmarks/suites/gpu_extract_*``)
carries a single ``info_json`` blob describing which encoder produced its rows.
This module is the ONE place that turns that blob into the three-key SOURCE
space identity -- ``{source_model, layer, norm}`` -- that
``ManifoldAnchorDistiller.transform_source`` compares against. It deliberately
does NOT know about GCCA: a raw feature file is written before any GCCA
transform exists, so its space can never carry a ``GCCA_map`` (see
``manifold_anchor_distiller.py`` module docstring and the Defect 1 fix in the
same commit that added this module).

Two entry points:

* ``source_space_from_info(info)`` -- pure derivation from an ``info_json``
  dict, used by writers (``gpu_extract_*.write_npz``) at extraction time.
* ``load_source_space(npz_path_or_NpzFile)`` -- used by readers. Prefers an
  explicit ``space`` key (written by newer extractors); falls back to
  deriving it from ``info_json`` for older cached files, logging a WARNING
  when it does (fail-closed elsewhere in this module, but this one fallback
  is intentionally a derived/degraded path per the "no silent fallback"
  rule -- it always logs).
"""
from __future__ import annotations

import json
import logging
import re
from typing import Dict, Union

import numpy as np

__all__ = ["source_space_from_info", "load_source_space"]

logger = logging.getLogger(__name__)

_SOURCE_SPACE_KEYS = ("source_model", "layer", "norm")

# The LLaMA-70B and Qwen-72B extractors build this with:
#   f"gguf:{model_id}:llama-server:pooling=last:embd_normalize={embd_normalize}"
_ENCODER_RE = re.compile(r"gguf:(?P<model_id>.+):llama-server:pooling=last:embd_normalize=(?P<normalization>-?\d+)")

# The only pooling string every current extractor writes
# (info["pooling"] = "llama-server --pooling last (final post-norm state, last token)").
# Any other string is unknown provenance and must raise, not be guessed at.
_POOLING_TO_LAYER = {
    "llama-server --pooling last (final post-norm state, last token)": "final-post-norm:last-token",
}


def source_space_from_info(info: dict) -> Dict[str, str]:
    """Derive the 3-key SOURCE space ``{source_model, layer, norm}`` from an
    extractor's ``info_json`` dict. Raises ``ValueError`` on any missing or
    unrecognized field -- this function never guesses.
    """
    if not isinstance(info, dict):
        raise ValueError(f"source_space_from_info: info must be a dict, got {type(info)!r}")

    encoder = info.get("encoder")
    if not isinstance(encoder, str) or not encoder.strip():
        raise ValueError("source_space_from_info: info['encoder'] is missing or empty")
    m = _ENCODER_RE.fullmatch(encoder)
    if not m:
        raise ValueError(
            f"source_space_from_info: info['encoder'] = {encoder!r} does not match the "
            "expected 'gguf:<model_id>:llama-server:pooling=last:embd_normalize=<n>' form"
        )
    source_model = m.group("model_id")

    pooling = info.get("pooling")
    if not isinstance(pooling, str) or not pooling.strip():
        raise ValueError("source_space_from_info: info['pooling'] is missing or empty")
    layer = _POOLING_TO_LAYER.get(pooling)
    if layer is None:
        raise ValueError(
            f"source_space_from_info: unrecognized info['pooling'] = {pooling!r}; "
            f"known values: {sorted(_POOLING_TO_LAYER)}"
        )

    embd_normalize = info.get("embd_normalize")
    if embd_normalize is None or isinstance(embd_normalize, bool) or not isinstance(embd_normalize, (int, np.integer)):
        raise ValueError(f"source_space_from_info: info['embd_normalize'] must be an int, got {embd_normalize!r}")
    norm = f"embd_normalize={int(embd_normalize)}"
    if int(m.group("normalization")) != int(embd_normalize):
        raise ValueError(
            "source_space_from_info: encoder normalization disagrees with info['embd_normalize']"
        )

    return {"source_model": source_model, "layer": layer, "norm": norm}


def _validate_space_dict(space) -> Dict[str, str]:
    if not isinstance(space, dict) or set(space) != set(_SOURCE_SPACE_KEYS):
        raise ValueError(
            f"source space must be a dict with exactly keys {_SOURCE_SPACE_KEYS}, got {space!r}"
        )
    if not all(isinstance(space[k], str) and space[k].strip() for k in _SOURCE_SPACE_KEYS):
        raise ValueError(f"source space values must be non-empty strings, got {space!r}")
    return {k: space[k] for k in _SOURCE_SPACE_KEYS}


def load_source_space(npz_path_or_data: Union[str, "np.lib.npyio.NpzFile"]) -> Dict[str, str]:
    """Load the 3-key SOURCE space from a feature store.

    Accepts either a path (opened and closed internally) or an already-open
    ``NpzFile`` (e.g. from a ``with np.load(...) as data:`` block).

    Preference order:
      1. an explicit ``space`` key (JSON dict, exactly the 3 source-space keys)
         -- if ``info_json`` is also present, it must derive the identical
         space, or this raises (the two must never silently disagree).
      2. derived from ``info_json`` -- logs a WARNING, since this is a
         degraded/derived path for feature files written before the ``space``
         key existed.
    Raises ``ValueError`` if neither key is present.
    """
    if isinstance(npz_path_or_data, (str, bytes)) or hasattr(npz_path_or_data, "__fspath__"):
        with np.load(str(npz_path_or_data), allow_pickle=False) as data:
            return _load_source_space_from_open(data)
    return _load_source_space_from_open(npz_path_or_data)


def _load_source_space_from_open(data) -> Dict[str, str]:
    has_space = "space" in data.files if hasattr(data, "files") else "space" in data
    has_info = "info_json" in data.files if hasattr(data, "files") else "info_json" in data

    info = None
    if has_info:
        info = json.loads(str(data["info_json"]))

    if has_space:
        try:
            space = json.loads(str(data["space"]))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"load_source_space: 'space' key is not valid JSON: {exc}") from exc
        space = _validate_space_dict(space)
        if has_info:
            derived = source_space_from_info(info)
            if space != derived:
                raise ValueError(
                    f"load_source_space: stored space {space!r} disagrees with the space "
                    f"derived from info_json {derived!r}"
                )
        return space

    if has_info:
        derived = source_space_from_info(info)
        logger.warning(
            "load_source_space: no explicit 'space' key found; derived %r from info_json. "
            "Regenerate this feature file with a writer that records 'space' directly.",
            derived,
        )
        return derived

    raise ValueError("load_source_space: neither 'space' nor 'info_json' present in feature store")
