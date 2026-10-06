# anti-leakage: allow-mock-tensor
"""Pure-inference CAD stack: cad_engine, verbalizer_extractor, numa_affinity, gguf_parallel_pool.

Real components only. The forward passes run a genuine (tiny, deterministically initialised)
Qwen2 transformer through ``HFVerbalizerExtractor``; NUMA tests parse real sysfs-shaped files
and spawn real pinned processes; the live GGUF tests need the public Qwen2.5-1.5B file and are
selected with ``GENZERO_QWEN15B_GGUF=/path/to/qwen2.5-1.5b-instruct-q4_k_m.gguf``. Without that
variable the live tests are reported as skipped, not passed.
"""
from __future__ import annotations

import ast
import ctypes
import hashlib
import json
import math
import multiprocessing as mp
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "python" / "gen_zero"
NEW_MODULES = [
    PKG / "causal" / "cad_engine.py", PKG / "causal" / "verbalizer_extractor.py",
    PKG / "causal" / "numa_affinity.py", PKG / "causal" / "gguf_parallel_pool.py",
    PKG / "scripts" / "setup_qwen15b_models.py", ROOT / "examples" / "quickstart_qwen15b_cad.py",
]

from gen_zero.causal import cad_engine as cad  # noqa: E402
from gen_zero.causal import numa_affinity as numa  # noqa: E402
from gen_zero.causal import verbalizer_extractor as vx  # noqa: E402
from gen_zero.scripts import setup_qwen15b_models as setup  # noqa: E402

LIVE_GGUF = os.environ.get("GENZERO_QWEN15B_GGUF")
live = pytest.mark.skipif(not LIVE_GGUF, reason="set GENZERO_QWEN15B_GGUF to the Qwen2.5-1.5B Q4_K_M file")


# ------------------------------------------------------------------ real tiny transformer

def build_tiny_hf():
    """Tiny Qwen2 with fixed random weights and a word-level tokenizer that has yes/no/maybe."""
    import torch
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

    words = ["[UNK]", "yes", "no", "maybe", "Yes", "No", "Maybe", "question", ":", "context", "answer",
             "read", "the", "and", "choose", "one", "of", "or", "does", "drug", "work", "it", "did",
             "trial", "is", "open", "closed", "?", ".", "(", ")", "/"]
    tok = Tokenizer(models.WordLevel({w: i for i, w in enumerate(words)}, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="[UNK]")
    torch.manual_seed(0)
    torch.set_num_threads(1)
    config = Qwen2Config(vocab_size=len(words), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256)
    return Qwen2ForCausalLM(config).eval(), tokenizer


def tiny_engine_factory(spec):
    model, tokenizer = build_tiny_hf()
    return cad.CADEngine(vx.HFVerbalizerExtractor(model, tokenizer), alpha=spec.alpha)


def failing_factory(spec):
    raise RuntimeError("factory refuses to build an engine")


@pytest.fixture(scope="module")
def tiny():
    return build_tiny_hf()


@pytest.fixture(scope="module")
def engine(tiny):
    model, tokenizer = tiny
    return cad.CADEngine(vx.HFVerbalizerExtractor(model, tokenizer))


def make_head(alpha=0.5, **kw):
    rng = np.random.default_rng(3)
    return cad.CalibrationHead(alpha=alpha, mean=np.zeros(14), scale=np.ones(14),
                               coef=rng.normal(0, 0.1, (3, 14)), intercept=np.zeros(3), **kw)


Q, C = "does the drug work ?", "the trial is closed and the drug did work ."


# ------------------------------------------------------------------ CAD math

def test_difference_is_cond_minus_alpha_prior():
    cond, prior = np.array([2.0, 1.0, -1.0]), np.array([1.0, 3.0, 0.5])
    np.testing.assert_allclose(cad.difference(cond, prior, 0.5), [1.5, -0.5, -1.25])
    np.testing.assert_allclose(cad.difference(cond, prior, 0.0), cond)


@pytest.mark.parametrize("alpha", [-0.1, 1.01])
def test_difference_rejects_alpha_outside_unit_interval(alpha):
    with pytest.raises(ValueError, match="alpha"):
        cad.difference(np.zeros(3), np.zeros(3), alpha)


def test_difference_fails_closed_on_bad_logits():
    with pytest.raises(ValueError, match="nonfinite"):
        cad.difference(np.array([np.nan, 0, 0]), np.zeros(3), 0.5)
    with pytest.raises(ValueError, match="matched"):
        cad.difference(np.zeros(3), np.zeros(4), 0.5)


def test_causal_features_values():
    cond, prior = np.array([3.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0])
    f = cad.extract_causal_features(cond, prior, 1.0)
    assert f.shape == (cad.N_CAUSAL_FEATURES,) == (14,)
    p = np.exp(cond) / np.exp(cond).sum()
    np.testing.assert_allclose(f[3:6], p)
    np.testing.assert_allclose(f[6:9], np.full(3, 1 / 3))
    h_cond = -(p * np.log(p)).sum()
    assert f[9] == pytest.approx(h_cond, abs=1e-9)
    assert f[10] == pytest.approx(math.log(3), abs=1e-9)
    assert f[11] == pytest.approx(math.log(3) - h_cond, abs=1e-9)
    assert f[12] == pytest.approx((p * np.log(p * 3)).sum(), abs=1e-9)
    assert f[13] == pytest.approx(np.sort(p)[-1] - np.sort(p)[-2], abs=1e-12)
    batch = cad.extract_causal_features(np.stack([cond, cond]), np.stack([prior, prior]), 1.0)
    assert batch.shape == (2, 14)


def test_uncalibrated_probabilities_clip_and_maybe_bias(engine):
    cond, prior = np.array([100.0, -100.0, 0.0]), np.zeros(3)
    r = engine.uncalibrated_from_logits(cond, prior)
    assert r.clipped and r.calibrated is False and r.label == "yes"
    assert max(map(abs, r.contrastive_logits)) <= engine.delta_clip
    assert sum(r.probabilities.values()) == pytest.approx(1.0)
    quiet = engine.uncalibrated_from_logits(np.array([0.5, 0.2, 0.1]), np.zeros(3))
    assert not quiet.clipped and quiet.label == "yes"
    # maybe_bias is the knob that turns a weak yes into maybe
    biased = cad.CADEngine(engine.extractor, maybe_bias=2.0).uncalibrated_from_logits(np.array([0.5, 0.2, 0.1]), np.zeros(3))
    assert biased.label == "maybe" and biased.probabilities["maybe"] > quiet.probabilities["maybe"]


def test_head_standardised_features_are_clipped_and_reported():
    head = make_head(z_clip=2.0)
    feats, probs, clipped = head.probabilities(np.array([40.0, 0.0, 0.0]), np.zeros(3))
    assert clipped and probs.sum() == pytest.approx(1.0)
    _, _, clipped_small = head.probabilities(np.array([0.3, 0.1, 0.0]), np.zeros(3))
    assert not clipped_small


@pytest.mark.parametrize("kw", [dict(alpha=0.05), dict(alpha=1.5), dict(temperature=0.0), dict(maybe_bias=9.0),
                                dict(z_clip=0.5)])
def test_head_rejects_out_of_range_hyperparameters(kw):
    with pytest.raises(ValueError, match="invalid CAD"):
        make_head(**kw)


def test_head_rejects_bad_shapes_and_nonpositive_scale():
    with pytest.raises(ValueError, match="invalid CAD head"):
        cad.CalibrationHead(alpha=0.5, mean=np.zeros(3), scale=np.ones(14), coef=np.zeros((3, 14)), intercept=np.zeros(3))
    with pytest.raises(ValueError, match="invalid CAD head"):
        cad.CalibrationHead(alpha=0.5, mean=np.zeros(14), scale=np.zeros(14), coef=np.zeros((3, 14)), intercept=np.zeros(3))


def test_head_json_is_bound_to_the_gguf_by_sha256(tmp_path):
    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"not a real model, only bytes to hash")
    body = {"format": cad.HEAD_FORMAT, "gguf_sha256": hashlib.sha256(gguf.read_bytes()).hexdigest(),
            "alpha": 0.5, "mean": [0.0] * 14, "scale": [1.0] * 14, "coef": np.zeros((3, 14)).tolist(),
            "intercept": [0.0, 0.0, 0.0], "maybe_bias": 0.25}
    path = tmp_path / "head.json"
    path.write_text(json.dumps(body))
    assert cad.CalibrationHead.from_json(path, gguf).maybe_bias == 0.25
    gguf.write_bytes(b"a different file")
    with pytest.raises(ValueError, match="different GGUF"):
        cad.CalibrationHead.from_json(path, gguf)
    body["format"] = "something.else"
    path.write_text(json.dumps(body))
    with pytest.raises(ValueError, match="format"):
        cad.CalibrationHead.from_json(path, gguf)


def test_prompt_template_contrast_differs_only_by_context():
    t = cad.PromptTemplate()
    full, prior = t.render("Q?", "some context"), t.render_prior("Q?")
    assert "some context" in full and "some context" not in prior
    assert full.startswith(t.prefix) and prior.startswith(t.prefix) and full.endswith(t.suffix)
    with pytest.raises(ValueError):
        t.render("", "ctx")
    with pytest.raises(ValueError):
        t.render("Q?", "  ")


# ------------------------------------------------------------------ verbalizer extractor

def test_candidate_groups_single_token_unique_and_fail_closed():
    table = {"yes": 1, " yes": 11, "Yes": 2, " Yes": 12, "no": 3, "No": 4, "maybe": 5, "Maybe": 6}
    enc = lambda t: [table[t]] if t in table else [90, 91]  # noqa: E731  unknown spellings are multi-token
    groups = vx.candidate_groups(enc)
    assert groups == [[1, 11, 2, 12], [3, 4], [5, 6]]
    assert all(len(set(g)) == len(g) for g in groups) and len({i for g in groups for i in g}) == 8
    with pytest.raises(ValueError, match="no single-token verbalizer"):
        vx.candidate_groups(lambda t: [7, 8])
    with pytest.raises(ValueError, match="distinct"):
        vx.candidate_groups(enc, vx.VerbalizerSpec(("yes", "yes")))
    # one id may never serve two labels: the second label then has no candidate left and raises
    with pytest.raises(ValueError, match="no single-token verbalizer for label 'no'"):
        vx.candidate_groups(lambda t: [1], vx.VerbalizerSpec(("yes", "no")))


def test_reduce_and_gather_match_numpy_logsumexp():
    row = np.arange(20, dtype=np.float32) / 3
    groups = [[1, 4], [7], [9, 10, 11]]
    out = vx.gather_scores(row, [i for g in groups for i in g], vx.group_bounds(groups))
    want = [math.log(sum(math.exp(float(row[i])) for i in g)) for g in groups]
    np.testing.assert_allclose(out, want, rtol=1e-6)
    bad = row.copy()
    bad[7] = np.inf
    with pytest.raises(ValueError, match="nonfinite"):
        vx.gather_scores(bad, [1, 4, 7, 9, 10, 11], vx.group_bounds(groups))


def test_view_logits_is_a_zero_copy_view_of_the_c_buffer():
    n = 152_064
    buf = (ctypes.c_float * n)()
    buf[5] = 1.5
    ptr = ctypes.cast(buf, ctypes.POINTER(ctypes.c_float))
    view = vx.view_logits(ptr, n)
    assert view.shape == (n,) and view.dtype == np.float32 and view[5] == 1.5
    buf[5] = -2.0  # mutate the C memory: a copy would not notice
    assert view[5] == -2.0
    view[6] = 9.0  # and the other direction
    assert buf[6] == 9.0
    with pytest.raises(RuntimeError, match="NULL"):
        vx.view_logits(ctypes.POINTER(ctypes.c_float)(), n)


def test_common_prefix_len():
    assert vx.common_prefix_len([1, 2, 3], [1, 2, 9, 9]) == 2
    assert vx.common_prefix_len([], [1]) == 0


def test_hf_extractor_logits_to_keep_one_equals_last_row_of_full_logits(tiny):
    import torch

    model, tokenizer = tiny
    ex = vx.HFVerbalizerExtractor(model, tokenizer)
    prompt = cad.PromptTemplate().render(Q, C)
    scores, n = ex.score(prompt)
    enc = tokenizer(prompt, return_tensors="pt")
    with torch.inference_mode():
        full = model(**enc).logits
        kept = model(**enc, logits_to_keep=1).logits
    assert kept.shape[1] == 1 and full.shape[1] == n > 1  # the head ran on ONE position
    np.testing.assert_allclose(scores, vx.gather_scores(full[0, -1].numpy(), ex.flat_ids, ex.bounds), rtol=1e-5, atol=1e-6)
    assert len(ex.groups) == 3 and len(ex.flat_ids) == 6  # tiny vocab: yes/no/maybe x {lower, Title}
    with pytest.raises(ValueError, match="n_ctx"):
        vx.HFVerbalizerExtractor(model, tokenizer, n_ctx=10_000)
    short = vx.HFVerbalizerExtractor(model, tokenizer, n_ctx=4)
    with pytest.raises(ValueError, match="input has"):
        short.score(prompt)


# ------------------------------------------------------------------ engine end to end (real forward)

def test_engine_matches_independent_recomputation_from_the_model(tiny, engine):
    import torch

    model, tokenizer = tiny
    t = cad.PromptTemplate()
    rows = []
    for text in (t.render(Q, C), t.render_prior(Q)):
        with torch.inference_mode():
            last = model(**tokenizer(text, return_tensors="pt")).logits[0, -1].double().numpy()
        rows.append([np.logaddexp.reduce(last[g]) for g in engine.extractor.groups])
    want_delta = np.asarray(rows[0]) - engine.alpha * np.asarray(rows[1])
    r = engine.infer_uncalibrated(Q, C)
    np.testing.assert_allclose(r.raw_class_logits, rows[0], rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(r.unconditional_logits, rows[1], rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(r.contrastive_logits, want_delta, rtol=1e-5, atol=1e-6)
    assert r.label == cad.LABELS[int(np.argmax(want_delta))] and r.calibrated is False
    assert r.to_dict()["generated_tokens"] == 0
    assert engine.classify(Q, C) == r  # no head loaded: classify reports the uncalibrated path


def test_predict_without_head_raises_instead_of_downgrading(engine):
    with pytest.raises(RuntimeError, match="CalibrationHead"):
        engine.predict(Q, C)


def test_engine_with_head_is_calibrated_and_uses_head_alpha(tiny):
    model, tokenizer = tiny
    eng = cad.CADEngine(vx.HFVerbalizerExtractor(model, tokenizer), head=make_head(alpha=0.8))
    r = eng.predict(Q, C)
    assert r.calibrated and r.alpha == 0.8 and eng.classify(Q, C) == r
    assert sum(r.probabilities.values()) == pytest.approx(1.0)


def test_engine_rejects_empty_inputs_and_wrong_label_count(engine, tiny):
    with pytest.raises(ValueError):
        engine.infer_uncalibrated("", C)
    with pytest.raises(ValueError):
        engine.infer_uncalibrated(Q, "")
    model, tokenizer = tiny
    two = vx.HFVerbalizerExtractor(model, tokenizer, spec=vx.VerbalizerSpec(("yes", "no")))
    with pytest.raises(ValueError, match="exactly"):
        cad.CADEngine(two)


def test_inference_never_touches_weights_or_gradients(tiny, engine):
    """Physical-state assertion: a forward-only engine leaves every parameter bit-identical."""
    model, _ = tiny
    before = {k: v.clone() for k, v in model.state_dict().items()}
    for _ in range(3):
        engine.infer_uncalibrated(Q, C)
    for k, v in model.state_dict().items():
        assert float((v - before[k]).abs().max()) == 0.0, k
    assert all(p.grad is None for p in model.parameters())


def test_new_modules_contain_no_training_machinery_and_no_internal_paths():
    forbidden_attrs = {"backward", "step", "zero_grad", "requires_grad_", "optim"}
    forbidden_text = ["dev_offload_exec", "ai-" + "server", "/ebs/", "hashlib.sha256(text", "np.load", "pickle", ".safetensors"]
    for path in NEW_MODULES:
        src = path.read_text()
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                assert node.attr not in forbidden_attrs, f"{path.name}:{node.lineno} uses .{node.attr}"
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [a.name for a in node.names] + [getattr(node, "module", "") or ""]
                assert not any("optim" in n or "pickle" in n for n in names), f"{path.name}:{node.lineno}"
        for text in forbidden_text:
            assert text not in src, f"{path.name} contains {text!r}"


def test_no_weight_or_feature_artifacts_ship_with_the_inference_stack():
    banned = (".gguf", ".npz", ".safetensors", ".pt", ".pth", ".bin", ".ckpt")
    for folder in (PKG / "causal", PKG / "scripts", ROOT / "tests"):
        found = [p.name for p in folder.rglob("*") if p.is_file() and p.suffix in banned]
        assert not found, f"{folder}: {found}"


# ------------------------------------------------------------------ NUMA

def fake_sysfs(tmp_path: Path, nodes: list[list[int]], siblings: int = 2) -> Path:
    base = tmp_path / "devices/system"
    for n, cpus in enumerate(nodes):
        d = base / f"node/node{n}"
        d.mkdir(parents=True)
        d.joinpath("cpulist").write_text(",".join(map(str, cpus)) + "\n")
        for i in range(0, len(cpus), siblings):
            group = cpus[i:i + siblings]
            for cpu in group:
                t = base / f"cpu/cpu{cpu}/topology"
                t.mkdir(parents=True)
                t.joinpath("thread_siblings_list").write_text(",".join(map(str, group)) + "\n")
    return tmp_path


DEV_LIKE = [list(range(0, 32)), list(range(32, 64))]


def test_read_topology_two_sockets_and_cpulist_ranges(tmp_path):
    topo = numa.read_topology(fake_sysfs(tmp_path, DEV_LIKE))
    assert [len(n) for n in topo.nodes] == [16, 16] and topo.logical_cpus == 64
    assert topo.nodes[0][0] == (0, 1) and topo.nodes[1][0] == (32, 33)
    assert topo.node_of(frozenset({40, 41})) == 1
    with pytest.raises(ValueError, match="spans NUMA nodes"):
        topo.node_of(frozenset({31, 32}))
    assert numa.parse_cpulist("0-3,8,10-11") == [0, 1, 2, 3, 8, 10, 11]


def test_read_topology_fails_closed(tmp_path):
    with pytest.raises(RuntimeError, match="no NUMA nodes"):
        numa.read_topology(tmp_path)
    root = fake_sysfs(tmp_path / "a", [[0, 1, 2, 3]])
    (root / "devices/system/cpu/cpu0/topology/thread_siblings_list").write_text("0,9\n")
    with pytest.raises(RuntimeError, match="leave node"):
        numa.read_topology(root)


def test_plan_is_round_robin_disjoint_and_node_local(tmp_path):
    topo = numa.read_topology(fake_sysfs(tmp_path, DEV_LIKE))
    plan = numa.plan_placements(topo, workers=4, cores_per_worker=8)
    assert [p.node for p in plan] == [0, 1, 0, 1]
    assert plan[0].cpus == tuple(range(0, 16)) and plan[2].cpus == tuple(range(16, 32))
    assert plan[1].cpus == tuple(range(32, 48)) and plan[3].cpus == tuple(range(48, 64))
    seen: set[int] = set()
    for p in plan:
        assert topo.node_of(frozenset(p.cpus)) == p.node
        assert not seen & set(p.cpus)
        seen |= set(p.cpus)


def test_plan_refuses_oversubscription_and_bad_grid(tmp_path):
    topo = numa.read_topology(fake_sysfs(tmp_path, DEV_LIKE))
    with pytest.raises(ValueError, match="shrink the grid"):
        numa.plan_placements(topo, workers=5, cores_per_worker=16)
    with pytest.raises(ValueError, match="positive"):
        numa.plan_placements(topo, workers=0, cores_per_worker=1)


@pytest.mark.skipif(not Path("/sys/devices/system/node/node0").is_dir(), reason="host exposes no NUMA sysfs")
def test_real_sysfs_topology_covers_online_cpus():
    topo = numa.read_topology()
    online = {c for node in topo.nodes for core in node for c in core}
    assert online and len(online) == topo.logical_cpus
    assert online & os.sched_getaffinity(0)  # at least one CPU this process may use is in the topology


def _report_child(queue):
    queue.put((sorted(os.sched_getaffinity(0)), os.environ.get("OMP_PROC_BIND"), os.environ.get("OMP_PLACES")))


def test_start_pinned_child_inherits_mask_and_omp_env_and_parent_is_restored():
    allowed = sorted(os.sched_getaffinity(0))
    mine = tuple(allowed[:1])
    placement = numa.WorkerPlacement(worker=0, node=0, cpus=mine)
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    proc = ctx.Process(target=_report_child, args=(queue,))
    before_env = {k: os.environ.get(k) for k in numa.OMP_PINNED_ENV}
    numa.start_pinned(proc, placement)
    got = queue.get(timeout=120)
    proc.join(timeout=60)
    assert got == (list(mine), "close", "cores")
    assert sorted(os.sched_getaffinity(0)) == allowed
    assert {k: os.environ.get(k) for k in numa.OMP_PINNED_ENV} == before_env


def test_start_pinned_refuses_an_impossible_cpuset():
    impossible = numa.WorkerPlacement(worker=0, node=0, cpus=(max(os.sched_getaffinity(0)) + 4096,))
    proc = mp.get_context("spawn").Process(target=_report_child, args=(mp.get_context("spawn").Queue(),))
    allowed = os.sched_getaffinity(0)
    with pytest.raises(OSError):
        numa.start_pinned(proc, impossible)
    assert os.sched_getaffinity(0) == allowed and proc.pid is None


# ------------------------------------------------------------------ parallel pool (real engines in real workers)

@pytest.fixture(scope="module")
def placeholder_gguf(tmp_path_factory):
    """The pool checks the path exists; tiny_engine_factory builds its own model and ignores it."""
    p = tmp_path_factory.mktemp("pool") / "unused.gguf"
    p.write_bytes(b"placeholder")
    return p


def test_pool_argument_validation(placeholder_gguf):
    from gen_zero.causal import gguf_parallel_pool as pool

    with pytest.raises(ValueError, match="positive"):
        pool.GGUFParallelPool(placeholder_gguf, workers=0, threads_per_worker=1)
    with pytest.raises(FileNotFoundError):
        pool.GGUFParallelPool(placeholder_gguf.with_name("missing.gguf"), workers=1, threads_per_worker=1)
    with pytest.raises(ValueError, match="module:callable"):
        pool.GGUFParallelPool(placeholder_gguf, workers=1, threads_per_worker=1, factory="no_colon_here")
    assert pool.resolve_factory(pool.GGUF_ENGINE_FACTORY) is pool.build_gguf_engine


def test_pool_reports_worker_init_failure(placeholder_gguf):
    from gen_zero.causal import gguf_parallel_pool as pool

    with pytest.raises(RuntimeError, match="initialization failed.*factory refuses"):
        pool.GGUFParallelPool(placeholder_gguf, workers=1, threads_per_worker=1,
                              factory="test_causal_cad_pure_infer:failing_factory")


def test_pool_batches_are_ordered_equal_to_sequential_and_errors_propagate(placeholder_gguf, tiny):
    from gen_zero.causal import gguf_parallel_pool as pool

    items = [(f"does the drug work {'?' * (i % 3 + 1)}", f"the trial is {w} .")
             for i, w in enumerate(["open", "closed", "open", "closed", "open"])]
    with pool.GGUFParallelPool(placeholder_gguf, workers=2, threads_per_worker=1, chunk_size=2,
                               factory="test_causal_cad_pure_infer:tiny_engine_factory", alpha=0.75) as p:
        assert p.alpha == 0.75 and p.spec.alpha == 0.75
        assert len(set(p.worker_pids)) == 2
        got = p.classify_batch(items)
        raw = p.raw_logits_batch(items[:2])
        with pytest.raises(ValueError, match="empty"):
            p.classify_batch([])
        with pytest.raises(TypeError):
            p.classify_batch([("only one",)])
        with pytest.raises(RuntimeError, match="head_path"):
            p.predict_batch(items)
        with pytest.raises(RuntimeError, match="worker failed"):
            p.classify_batch([("", "context")])
        assert p.classify_batch(items[:1])[0] == got[0]  # the pool survives a failed chunk
    model, tokenizer = tiny
    engine_75 = cad.CADEngine(vx.HFVerbalizerExtractor(model, tokenizer), alpha=0.75)
    want = [engine_75.classify(q, c).to_dict() for q, c in items]
    assert len(got) == len(items)
    for g, w in zip(got, want):
        assert g["label"] == w["label"]
        np.testing.assert_allclose(g["contrastive_logits"], w["contrastive_logits"], rtol=1e-5, atol=1e-6)
    # Mutation proof: contrastive_logits with alpha=0.75 must physically differ from alpha=0.5
    engine_default = cad.CADEngine(vx.HFVerbalizerExtractor(model, tokenizer), alpha=0.5)
    default_want = [engine_default.classify(q, c).to_dict() for q, c in items]
    assert not np.allclose(got[0]["contrastive_logits"], default_want[0]["contrastive_logits"])
    cond, prior, n = engine_75.raw_logits(*items[0])
    np.testing.assert_allclose(raw[0]["conditional"], cond, rtol=1e-5, atol=1e-6)
    assert raw[0]["input_tokens"] == n


def test_pool_numa_pin_places_every_worker_thread_inside_its_plan(placeholder_gguf):
    from gen_zero.causal import gguf_parallel_pool as pool

    try:
        topo = numa.read_topology()
    except (RuntimeError, OSError) as exc:
        pytest.skip(f"no usable NUMA sysfs on this host: {exc}")
    usable = {c for node in topo.nodes for core in node for c in core} & os.sched_getaffinity(0)
    if not usable:
        pytest.skip("no allowed CPU appears in sysfs topology")
    with pool.GGUFParallelPool(placeholder_gguf, workers=1, threads_per_worker=1, numa_pin=True, topology=topo,
                               factory="test_causal_cad_pure_infer:tiny_engine_factory") as p:
        p.classify_batch([(Q, C)])
        report = p.pinning_report()  # raises if any thread can leave the planned cpuset
    plan = set(p.placements[0].cpus)
    assert set(report[0]["observed_cpus"]) <= plan and report[0]["planned_node"] == p.placements[0].node


# ------------------------------------------------------------------ setup script + CLI

def test_setup_verify_rejects_wrong_size_and_wrong_hash(tmp_path):
    assert len(setup.EXPECTED_SHA256) == 64 and setup.EXPECTED_SIZE == 1117320736
    f = tmp_path / setup.FILENAME
    f.write_bytes(b"x" * 10)
    with pytest.raises(ValueError, match="size"):
        setup.verify(f)
    with pytest.raises(FileNotFoundError):
        setup.verify(tmp_path / "absent.gguf")
    assert setup.sha256_of(f) == hashlib.sha256(b"x" * 10).hexdigest()


def test_setup_fetch_deletes_a_corrupt_existing_file_and_never_trusts_it(tmp_path):
    f = tmp_path / setup.FILENAME
    f.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="unknown source"):
        setup.fetch(tmp_path, source="nowhere")
    assert not f.exists()


def test_cli_cad_is_mounted_and_fails_closed(capsys):
    from gen_zero import cli

    assert "cad" in cli.build_cli_parser().format_help()
    rc = cli.main(["--no-color", "cad", "--gguf", "/nonexistent.gguf", "--question", "q?", "--context", "c"])
    assert rc == 1 and "GGUF model not found" in capsys.readouterr().err
    rc = cli.main(["--no-color", "cad", "--gguf", "/nonexistent.gguf"])
    assert rc == 1 and "--question and --context" in capsys.readouterr().err


def test_cli_cad_runs_the_hf_backend_end_to_end(tiny, tmp_path, capsys, placeholder_gguf):
    from gen_zero import cli

    model, tokenizer = tiny
    model.save_pretrained(tmp_path)
    # separate tokenizer dir: with model_type=qwen2 in config.json, AutoTokenizer would rebuild a
    # Qwen2Tokenizer and ignore this tiny word-level vocab
    tokenizer.save_pretrained(tmp_path / "tok")
    rc = cli.main(["--no-color", "cad", "--hf-model", str(tmp_path), "--hf-tokenizer", str(tmp_path / "tok"),
                   "--question", Q, "--context", C, "--json"])
    out = json.loads(capsys.readouterr().out)
    want = cad.CADEngine(vx.HFVerbalizerExtractor(*tiny)).classify(Q, C)
    assert rc == 0 and out["label"] == want.label and out["calibrated"] is False
    np.testing.assert_allclose(out["contrastive_logits"], want.contrastive_logits, rtol=1e-5, atol=1e-6)

    # 1. Both --gguf and --hf-model
    assert cli.main(["--no-color", "cad", "--question", Q, "--context", C,
                     "--gguf", str(placeholder_gguf), "--hf-model", str(tmp_path)]) == 1
    assert "give exactly one of --gguf or --hf-model" in capsys.readouterr().err
    # 2. --hf-model with --workers > 1
    assert cli.main(["--no-color", "cad", "--question", Q, "--context", C,
                     "--hf-model", str(tmp_path), "--workers", "2"]) == 1
    assert "--workers > 1 needs --gguf" in capsys.readouterr().err
    # 3. --hf-model with --numa-pin
    assert cli.main(["--no-color", "cad", "--question", Q, "--context", C,
                     "--hf-model", str(tmp_path), "--numa-pin"]) == 1
    assert "--numa-pin requires --gguf" in capsys.readouterr().err
    # 4. --gguf with --hf-tokenizer (using REAL placeholder_gguf file)
    assert cli.main(["--no-color", "cad", "--question", Q, "--context", C,
                     "--gguf", str(placeholder_gguf), "--hf-tokenizer", str(tmp_path / "tok")]) == 1
    assert "--hf-tokenizer requires --hf-model" in capsys.readouterr().err
    # 5. Neither --gguf nor --hf-model
    assert cli.main(["--no-color", "cad", "--question", Q, "--context", C]) == 1
    assert "give exactly one of --gguf or --hf-model" in capsys.readouterr().err


def test_cli_cad_single_worker_numa_pin_routes_to_pool(placeholder_gguf, monkeypatch, capsys):
    from gen_zero import cli
    from gen_zero.causal.gguf_parallel_pool import GGUFParallelPool

    called = {}
    def mock_init(self, *a, **kw):
        called["workers"] = kw.get("workers")
        called["numa_pin"] = kw.get("numa_pin")
        raise RuntimeError("sentinel_pool_routed")

    monkeypatch.setattr(GGUFParallelPool, "__init__", mock_init)
    rc = cli.main(["--no-color", "cad", "--gguf", str(placeholder_gguf), "--numa-pin",
                   "--workers", "1", "--question", Q, "--context", C])
    assert rc == 1 and "sentinel_pool_routed" in capsys.readouterr().err
    assert called.get("workers") == 1 and called.get("numa_pin") is True


# ------------------------------------------------------------------ live GGUF (opt-in)

@live
def test_live_gguf_engine_prefix_reuse_is_exact_and_result_is_sane():
    eng = cad.CADEngine.from_gguf(LIVE_GGUF, n_ctx=1024, n_threads=2)
    ex = eng.extractor
    assert len(ex.groups) == 3 and all(ex.groups)
    prompt = eng.template.render(Q, C)
    assert ex.verify_prefix_reuse(prompt) < 1e-4
    r = eng.classify(Q, C)
    assert r.label in cad.LABELS and r.calibrated is False and r.input_tokens > 0
    assert sum(r.probabilities.values()) == pytest.approx(1.0)
    assert eng.classify(Q, C) == r  # deterministic
    assert ex.reused_tokens > 0


@live
def test_live_gguf_pool_matches_single_engine(tmp_path):
    from gen_zero.causal import gguf_parallel_pool as pool

    items = [(Q, C), ("is the bridge open ?", "the bridge is closed ."), (Q, "the trial is open .")]
    eng = cad.CADEngine.from_gguf(LIVE_GGUF, n_ctx=1024, n_threads=2)
    want = [eng.classify(q, c) for q, c in items]
    with pool.GGUFParallelPool(LIVE_GGUF, workers=2, threads_per_worker=1, n_ctx=1024) as p:
        got = p.classify_batch(items)
    assert [g["label"] for g in got] == [w.label for w in want]
    for g, w in zip(got, want):
        np.testing.assert_allclose(g["contrastive_logits"], w.contrastive_logits, atol=1e-3)


@live
def test_live_gguf_is_the_published_file():
    setup.verify(Path(LIVE_GGUF))
