#!/usr/bin/env python3
"""Gen-Zero Unified Shell CLI (gen-zero / zero).

Unified command-line interface for the Gen-Zero non-autoregressive cognitive engine.
Provides commands for:
  - ask: Single-forward typed decision micro-cores (Noul, Choice, Score)
  - route: Sub-5ms adaptive tool catalog pruning router
  - imagine: World model counterfactual tree lookahead with CP-SAT safety verification
  - stream: Spatiotemporal streaming with Attention Sinks constant memory
  - grep: Propositional semantic search & Boolean logic filtering
  - compact: Verbatim context compaction with 0.0 fact mutation
  - mcp: Launch Model Context Protocol stdio or SSE server
  - status: Check engine connectivity, model status, and latency
"""

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ANCHOR_ARTIFACT = (
    REPO_ROOT / "benchmarks" / "results" / "manifold" / "anchor_128d_llama70b_boolq_gcca_etf0.npz"
)
DEFAULT_ANCHOR_CORE_MANIFEST = (
    REPO_ROOT / "benchmarks" / "results" / "manifold" / "anchor_128d_llama70b_boolq_gcca_etf0.core_manifest.json"
)

from gen_zero.mcp.server import (
    MCPServer,
    SERVER_NAME,
    SERVER_VERSION,
    execute_zero_ask,
    execute_zero_route,
    execute_zero_imagine,
    execute_zero_stream,
    execute_zero_grep,
    resolve_endpoint,
    resolve_api_token,
)
from gen_zero.compactor import VerbatimContextCompactor


# ANSI Color helpers
def _colors(no_color: bool = False):
    use = not no_color and hasattr(sys.stdout, "isatty") and sys.stdout.isatty()
    return {
        "BOLD": "\033[1m" if use else "",
        "GREEN": "\033[32m" if use else "",
        "CYAN": "\033[36m" if use else "",
        "BLUE": "\033[34m" if use else "",
        "YELLOW": "\033[33m" if use else "",
        "RED": "\033[31m" if use else "",
        "MAGENTA": "\033[35m" if use else "",
        "RESET": "\033[0m" if use else "",
    }


# ---------------------------------------------------------------------------
# Subcommand: ask
# ---------------------------------------------------------------------------
def cmd_ask(args: argparse.Namespace) -> int:
    c = _colors(args.no_color)
    
    if args.json_input:
        try:
            if os.path.exists(args.json_input):
                with open(args.json_input, "r", encoding="utf-8") as f:
                    payload = json.load(f)
            else:
                payload = json.loads(args.json_input)
        except Exception as e:
            sys.stderr.write(f"{c['RED']}Error loading json input: {e}{c['RESET']}\n")
            return 2
        state = payload.get("state", "")
        questions = payload.get("questions", {})
    else:
        if not args.state or not args.question:
            sys.stderr.write(f"{c['RED']}Error: 'state' and 'question' are required unless --json-input is provided.{c['RESET']}\n")
            return 2
        state = args.state
        q_type = args.type.lower()
        q_name = "q1"

        if q_type == "choice":
            choices = args.choices or ["option_a", "option_b"]
            criteria = {ch: ch for ch in choices}
            questions = {
                q_name: {
                    "type": "choice",
                    "instructions": args.question,
                    "criteria": criteria,
                }
            }
        elif q_type == "score":
            criteria = args.criteria or ["Low / None", "Medium", "High / Severe"]
            questions = {
                q_name: {
                    "type": "score",
                    "instructions": args.question,
                    "criteria": criteria,
                }
            }
        else: # noul
            questions = {
                q_name: {
                    "type": "noul",
                    "instructions": args.question,
                    "criteria": {"true": "Yes / Positive match", "false": "No / Negative match"},
                }
            }

    req = {"state": state, "questions": questions}
    res = asyncio.run(execute_zero_ask(req))

    if res.get("isError"):
        err_text = res["content"][0]["text"]
        sys.stderr.write(f"{c['RED']}Error: {err_text}{c['RESET']}\n")
        return 1

    content_str = res["content"][0]["text"]
    data = json.loads(content_str)

    if args.json:
        print(json.dumps(data, indent=2, ensure_ascii=False))
        return 0

    timing = data.get("timing_ms", 0.0)
    model = data.get("model", "typesafe/zero-1.13")
    print(f"{c['BOLD']}{c['CYAN']}[Gen-Zero Decision]{c['RESET']} {model} {c['GREEN']}({timing}ms){c['RESET']}")
    print(f"  {c['BOLD']}State:{c['RESET']} {state}")
    
    answers = data.get("answers", {})
    for k, ans in answers.items():
        ans_type = ans.get("type", "unknown")
        if ans.get("status") == "ABSTAIN":
            # Fail-closed: an abstained answer carries no 'noul'/'choice'/'score' key at
            # all, so falling back to a default (0.0, "none") here would silently repackage
            # a kernel refusal as a confident answer. Show the refusal instead.
            kernel_status = ans.get("kernel_status", "ABSTAIN")
            error = ans.get("error", "")
            print(f"  {c['BOLD']}{k}{c['RESET']} ({ans_type}): {c['RED']}ABSTAIN{c['RESET']} [{kernel_status}] {error}")
            continue
        if ans_type == "noul":
            val = ans["noul"]
            conf = ans.get("confidence", 0.0)
            bar_len = int(val * 20)
            bar = f"{c['GREEN']}{'█' * bar_len}{'░' * (20 - bar_len)}{c['RESET']}"
            print(f"  {c['BOLD']}{k}{c['RESET']} ({ans_type}): {bar} {c['BOLD']}{val:.3f}{c['RESET']} (conf: {conf:.2f})")
        elif ans_type == "choice":
            choice = ans["choice"]
            print(f"  {c['BOLD']}{k}{c['RESET']} ({ans_type}): {c['BOLD']}{c['GREEN']}{choice}{c['RESET']}")
        elif ans_type == "score":
            score = ans["score"]
            print(f"  {c['BOLD']}{k}{c['RESET']} ({ans_type}): {c['BOLD']}{c['YELLOW']}{score}{c['RESET']}")
    return 0


# ---------------------------------------------------------------------------
# Subcommand: route
# ---------------------------------------------------------------------------
def cmd_route(args: argparse.Namespace) -> int:
    c = _colors(args.no_color)
    if not args.tools:
        sys.stderr.write(f"{c['RED']}Error: --tools is required (path to tools.json or inline JSON).{c['RESET']}\n")
        return 2

    try:
        if os.path.exists(args.tools):
            with open(args.tools, "r", encoding="utf-8") as f:
                tools_data = json.load(f)
        else:
            tools_data = json.loads(args.tools)
        if isinstance(tools_data, dict) and "tools" in tools_data:
            tools_data = tools_data["tools"]
        if not isinstance(tools_data, list):
            raise ValueError("Tools data must be a JSON array of tool specifications.")
    except Exception as e:
        sys.stderr.write(f"{c['RED']}Error loading tools catalog: {e}{c['RESET']}\n")
        return 2

    req = {
        "task_goal": args.goal,
        "tools": tools_data,
        "top_k": args.top_k,
        "context": args.context or "",
    }
    res = asyncio.run(execute_zero_route(req))
    if res.get("isError"):
        sys.stderr.write(f"{c['RED']}Error: {res['content'][0]['text']}{c['RESET']}\n")
        return 1

    data = json.loads(res["content"][0]["text"])
    if args.json:
        print(json.dumps(data, indent=2, ensure_ascii=False))
        return 0

    timing = data.get("timing_ms", 0.0)
    total = data.get("total_input_tools", len(tools_data))
    pruned_names = data.get("pruned_tool_names", [])
    pruned_tools = data.get("pruned_tools", [])

    print(f"{c['BOLD']}{c['CYAN']}[Gen-Zero Router]{c['RESET']} Pruned {total} tools down to {len(pruned_tools)} in {c['GREEN']}{timing}ms{c['RESET']}")
    print(f"  {c['BOLD']}Goal:{c['RESET']} {args.goal}")
    for idx, tool in enumerate(pruned_tools, 1):
        name = tool.get("name", "unknown")
        desc = tool.get("description", "").split("\n")[0][:80]
        print(f"  {c['GREEN']}{idx}.{c['RESET']} {c['BOLD']}{name}{c['RESET']} - {desc}")
    return 0


# ---------------------------------------------------------------------------
# Subcommand: imagine
# ---------------------------------------------------------------------------
def cmd_imagine(args: argparse.Namespace) -> int:
    c = _colors(args.no_color)
    if not args.actions:
        sys.stderr.write(f"{c['RED']}Error: --actions <action1> [action2 ...] is required.{c['RESET']}\n")
        return 2

    req = {
        "state": args.state,
        "candidate_actions": args.actions,
        "horizon": args.horizon,
        "enforce_cpsat": not args.no_cpsat,
    }
    res = asyncio.run(execute_zero_imagine(req))
    if res.get("isError"):
        # Check if the result is a typed SAFETY_INTERLOCKED fail-closed verdict
        try:
            content = res.get("content", [])
            if content and content[0].get("type") == "text":
                err_data = json.loads(content[0]["text"])
                if isinstance(err_data, dict) and err_data.get("error") == "SAFETY_INTERLOCKED":
                    if args.json:
                        print(json.dumps(err_data, indent=2, ensure_ascii=False))
                        return 0
                    result_data = err_data.get("result", {})
                    action = result_data.get("selected_action", "HOLD")
                    cpsat_status = err_data.get("cpsat_status", "INTERLOCKED")
                    safety_badge = f"{c['RED']}INTERLOCKED ({cpsat_status}){c['RESET']}"
                    print(f"{c['BOLD']}{c['CYAN']}[Gen-Zero World Model]{c['RESET']} Horizon H={args.horizon} | CP-SAT: {safety_badge}")
                    print(f"  {c['BOLD']}State:{c['RESET']} {args.state}")
                    print(f"  {c['BOLD']}Optimal Action:{c['RESET']} {c['BOLD']}{c['YELLOW']}{action} [SAFETY_INTERLOCKED]{c['RESET']}")
                    msg = err_data.get("message", "")
                    if msg:
                        print(f"  {c['RED']}Notice:{c['RESET']} {msg}")
                    return 0
        except Exception:
            pass

        sys.stderr.write(f"{c['RED']}Error: {res['content'][0]['text']}{c['RESET']}\n")
        return 1

    data = json.loads(res["content"][0]["text"])
    if args.json:
        print(json.dumps(data, indent=2, ensure_ascii=False))
        return 0

    action = data.get("selected_action")
    status = data.get("nanocore_status", {})
    verified = status.get("cpsat_verified", False)
    cpsat_status = status.get("cpsat_status")
    safety_badge = (
        f"{c['GREEN']}VERIFIED (0-1 ILP, {cpsat_status}){c['RESET']}" if verified
        else f"{c['YELLOW']}UNVERIFIED ({cpsat_status}){c['RESET']}"
    )

    print(f"{c['BOLD']}{c['CYAN']}[Gen-Zero World Model]{c['RESET']} Horizon H={args.horizon} | CP-SAT: {safety_badge}")
    print(f"  {c['BOLD']}State:{c['RESET']} {args.state}")
    print(f"  {c['BOLD']}Optimal Action:{c['RESET']} {c['BOLD']}{c['GREEN']}{action}{c['RESET']}")

    trajectory = data.get("imagined_trajectory", [])
    if trajectory:
        print(f"  {c['BOLD']}Imagined Trajectory:{c['RESET']}")
        for step in trajectory:
            t = step.get("t", 0)
            a = step.get("action", "")
            s = step.get("safety", 1.0)
            v = step.get("value", 0.0)
            print(f"    t={t}: {c['CYAN']}{a}{c['RESET']} (safety={s:.2f}, value={v:.2f})")
    return 0


# ---------------------------------------------------------------------------
# Subcommand: stream
# ---------------------------------------------------------------------------
def cmd_stream(args: argparse.Namespace) -> int:
    c = _colors(args.no_color)
    if not args.actions:
        sys.stderr.write(f"{c['RED']}Error: --actions <action1> [action2 ...] is required.{c['RESET']}\n")
        return 2

    req = {
        "observation": args.observation,
        "candidate_actions": args.actions,
        "context_prompt": args.prompt or "",
    }
    res = asyncio.run(execute_zero_stream(req))
    if res.get("isError"):
        sys.stderr.write(f"{c['RED']}Error: {res['content'][0]['text']}{c['RESET']}\n")
        return 1

    data = json.loads(res["content"][0]["text"])
    if args.json:
        print(json.dumps(data, indent=2, ensure_ascii=False))
        return 0

    step = data.get("step", 1)
    act = data.get("selected_action", "unknown")
    shock = data.get("causal_shock_detected", False)
    shock_str = f"{c['RED']}YES (autonomic adaptation){c['RESET']}" if shock else f"{c['GREEN']}NO{c['RESET']}"

    print(f"{c['BOLD']}{c['CYAN']}[Gen-Zero Stream Step {step}]{c['RESET']} Rolling KV Cache O(W)")
    print(f"  {c['BOLD']}Selected Action:{c['RESET']} {c['BOLD']}{c['GREEN']}{act}{c['RESET']}")
    print(f"  {c['BOLD']}Causal Shock:{c['RESET']} {shock_str}")
    if data.get("degraded"):
        print(
            f"  {c['BOLD']}{c['YELLOW']}DEGRADED:{c['RESET']} provenance={data.get('provenance')} "
            f"({', '.join(data.get('degradations', []))}). The latent is a text hash prior; "
            f"the transition model has no language calibration."
        )
    return 0


# ---------------------------------------------------------------------------
# Subcommand: grep
# ---------------------------------------------------------------------------
def cmd_grep(argv: List[str]) -> int:
    """Dispatches to pure-stdlib semantic grep CLI."""
    from importlib.machinery import SourceFileLoader
    from importlib.util import module_from_spec, spec_from_loader
    candidates = [
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bin", "gen-grep"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts", "gen_grep.py"),
    ]
    script_path = next((p for p in candidates if os.path.exists(p)), candidates[0])
    if not os.path.exists(script_path):
        sys.stderr.write(f"Error: gen-grep script not found at {script_path}\n")
        return 2
    try:
        loader = SourceFileLoader("gen_grep", script_path)
        spec = spec_from_loader("gen_grep", loader)
        if not spec:
            raise ImportError(f"Could not create module spec for {script_path}")
        mod = module_from_spec(spec)
        loader.exec_module(mod)
        return mod.main(argv)
    except Exception as e:
        sys.stderr.write(f"Error running gen-grep: {e}\n")
        return 2


# ---------------------------------------------------------------------------
# Subcommand: compact
# ---------------------------------------------------------------------------
def cmd_compact(args: argparse.Namespace) -> int:
    c = _colors(args.no_color)
    input_source = args.file

    try:
        if not input_source or input_source == "-":
            raw_data = sys.stdin.read()
        else:
            with open(input_source, "r", encoding="utf-8") as f:
                raw_data = f.read()
        messages = json.loads(raw_data)
        if isinstance(messages, dict) and "messages" in messages:
            messages = messages["messages"]
        if not isinstance(messages, list):
            raise ValueError("Messages input must be a JSON array of message objects.")
    except Exception as e:
        sys.stderr.write(f"{c['RED']}Error loading conversation messages: {e}{c['RESET']}\n")
        return 2

    compactor = VerbatimContextCompactor(
        head_lines=args.head,
        tail_lines=args.tail,
        truncate_line_threshold=args.threshold,
    )
    compacted, items, summary = compactor.compact_session(messages)

    if args.output:
        try:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(compacted, f, indent=2, ensure_ascii=False)
        except Exception as e:
            sys.stderr.write(f"{c['RED']}Error saving output to {args.output}: {e}{c['RESET']}\n")
            return 2

    if args.json:
        payload = {
            "compacted_messages": compacted,
            "summary": summary.to_dict(),
            "items": [it.to_dict() for it in items],
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0

    print(f"{c['BOLD']}{c['CYAN']}[Gen-Zero Verbatim Compactor]{c['RESET']} {c['GREEN']}(0.0% Fact Mutation Guarantee){c['RESET']}")
    print(f"  {c['BOLD']}Messages:{c['RESET']}        {len(messages)} -> {len(compacted)}")
    print(f"  {c['BOLD']}Original Tokens:{c['RESET']} {summary.original_token_count:,}")
    print(f"  {c['BOLD']}Compacted Tokens:{c['RESET']}{summary.compacted_token_count:,} ({c['GREEN']}-{summary.compression_ratio * 100:.1f}% reduction{c['RESET']})")
    print(f"  {c['BOLD']}Verbatim Kept:{c['RESET']}   {summary.verbatim_count} turns")
    print(f"  {c['BOLD']}Logs Truncated:{c['RESET']}  {summary.truncated_count} turns")
    print(f"  {c['BOLD']}Probes Dropped:{c['RESET']}  {summary.dropped_count} turns (pwd, ls, ping)")
    print(f"  {c['BOLD']}Processing Time:{c['RESET']} {summary.latency_ms:.2f}ms")
    if args.output:
        print(f"  {c['BOLD']}Saved To:{c['RESET']}        {args.output}")
    return 0


# ---------------------------------------------------------------------------
# Subcommand: mcp
# ---------------------------------------------------------------------------
def cmd_mcp(args: argparse.Namespace) -> int:
    from gen_zero.mcp.server import main as mcp_main
    mcp_args = ["--transport", args.transport]
    if args.transport == "sse":
        mcp_args.extend(["--host", args.host, "--port", str(args.port)])
        if args.token:
            mcp_args.extend(["--token", args.token])
    
    orig_argv = sys.argv
    try:
        sys.argv = [sys.argv[0]] + mcp_args
        mcp_main()
    finally:
        sys.argv = orig_argv
    return 0


# ---------------------------------------------------------------------------
# Subcommand: semantic (scorer behind the Rust ask/route/imagine bridge)
# ---------------------------------------------------------------------------
def cmd_semantic(args: argparse.Namespace) -> int:
    from gen_zero.service.app import main as semantic_main

    semantic_main(["--host", args.host, "--port", str(args.port)])
    return 0


# ---------------------------------------------------------------------------
# Subcommand: anchor (manifold anchor bridge: 8192-d -> 128-d nanocore_state)
# ---------------------------------------------------------------------------
def cmd_anchor(args: argparse.Namespace) -> int:
    c = _colors(args.no_color)
    import numpy as np
    from gen_zero.causal.feature_space import load_source_space
    from gen_zero.causal.nanocore_bridge import NanocoreAnchorBridge

    try:
        core_manifest = json.loads(Path(args.core_manifest).read_text())
        bridge = NanocoreAnchorBridge(args.artifact, core_manifest=core_manifest)
        with np.load(args.features, allow_pickle=False) as data:
            if args.block not in data:
                raise KeyError(f"block {args.block!r} not found in {args.features} (available: {list(data.keys())})")
            row = {"values": np.asarray(data[args.block][args.row], dtype=np.float64),
                   "space": load_source_space(data)}
        candidates = [x.strip() for x in args.candidates.split(",") if x.strip()]
        payload = bridge.generate_mcp_ask_payload(row, args.domain_id, candidates)
    except Exception as e:
        sys.stderr.write(f"{c['RED']}Error: {e}{c['RESET']}\n")
        return 1

    if getattr(args, "execute", False):
        from gen_zero.client import GenZero
        try:
            gz = GenZero()
            gz.load_manifold_anchor_artifact(args.artifact, core_manifest=core_manifest)
            gz.register_nanocore(args.domain_id, space_manifest=core_manifest)
            decision = gz.decide_nanocore(row, args.domain_id, candidates)
        except Exception as e:
            sys.stderr.write(f"{c['RED']}Error: {e}{c['RESET']}\n")
            return 1
        if args.json:
            print(json.dumps(decision, indent=2))
        else:
            print(f"{c['BOLD']}{c['CYAN']}[Gen-Zero NanoCore Decision]{c['RESET']}")
            print(f"  {c['BOLD']}Action:{c['RESET']}     {decision['chosen_action']}")
            print(f"  {c['BOLD']}Confidence:{c['RESET']} {decision['confidence']:.4f}")
            print(f"  {c['BOLD']}Entropy:{c['RESET']}    {decision['attention_entropy']:.4f}")
            print(f"  {c['BOLD']}Gate:{c['RESET']}       {decision['gate_status']}")
            print(f"  {c['BOLD']}Domain:{c['RESET']}     {decision['domain']}")
        if decision["gate_status"] != "passed":
            sys.stderr.write(
                f"{c['YELLOW']}Warning: H1 entropy gate status is {decision['gate_status']}; "
                f"the chosen action is not confident.{c['RESET']}\n"
            )
            return 2
        return 0

    if args.json:
        print(json.dumps(payload, indent=2))
        return 0

    state = payload["nanocore_state"]
    print(f"{c['BOLD']}{c['CYAN']}[Gen-Zero Manifold Anchor]{c['RESET']}")
    print(f"  {c['BOLD']}Artifact:{c['RESET']}   {args.artifact}")
    print(f"  {c['BOLD']}Domain:{c['RESET']}     {payload['nanocore_domain']}")
    print(f"  {c['BOLD']}Candidates:{c['RESET']} {', '.join(payload['candidates'])}")
    print(f"  {c['BOLD']}State dim:{c['RESET']}  {len(state)}")
    print(f"  {c['BOLD']}State range:{c['RESET']} [{min(state):.4f}, {max(state):.4f}]")
    return 0


# ---------------------------------------------------------------------------
# Subcommand: manifold-fit (master closed-form head over a .npz feature store)
# ---------------------------------------------------------------------------
def _load_block(data, key: str, path: str):
    if key not in data:
        raise KeyError(f"array {key!r} not found in {path} (available: {list(data.keys())})")
    return data[key]


def cmd_manifold_fit(args: argparse.Namespace) -> int:
    c = _colors(args.no_color)
    import numpy as np
    from gen_zero.manifold import MasterClosedFormSolver, MultiModelGraphLaplacian

    try:
        with np.load(args.features, allow_pickle=False) as data:
            X = np.asarray(_load_block(data, args.block, args.features), dtype=np.float64)
            y = np.asarray(_load_block(data, args.label_key, args.features))
            X_unl = (np.asarray(_load_block(data, args.unlabeled_block, args.features), dtype=np.float64)
                     if args.unlabeled_block else np.zeros((0, X.shape[1])))
        if X.ndim != 2 or y.ndim != 1 or len(y) != len(X):
            raise ValueError(f"need 2-D features and 1-D labels of equal length, got {X.shape} and {y.shape}")
        if not np.issubdtype(y.dtype, np.integer):
            raise ValueError(f"labels must be integers, got dtype {y.dtype}")
        if X_unl.ndim != 2 or X_unl.shape[1] != X.shape[1]:
            raise ValueError(f"unlabeled block must have {X.shape[1]} columns, got {X_unl.shape}")
        if not 0.0 <= args.holdout < 1.0:
            raise ValueError("--holdout must be in [0, 1)")

        perm = np.random.default_rng(args.seed).permutation(len(X))
        n_hold = int(round(args.holdout * len(X)))
        hold_idx, fit_idx = perm[:n_hold], perm[n_hold:]
        classes = np.unique(y[fit_idx])
        if len(classes) < 2:
            raise ValueError("fit split has fewer than 2 classes")

        mean = X[fit_idx].mean(axis=0)
        std = X[fit_idx].std(axis=0)
        constant = std == 0.0
        # Constant columns are all-zero after centring; scale 1 keeps them at zero.
        std = np.where(constant, 1.0, std)
        Zfit = (X[fit_idx] - mean) / std
        Zhold = (X[hold_idx] - mean) / std
        Zunl = (X_unl - mean) / std
        Y = (y[fit_idx][:, None] == classes[None, :]).astype(np.float64)

        W0 = None
        if args.prior:
            prior = MasterClosedFormSolver.load(args.prior)
            W0 = prior.coef_

        M = L = None
        Z = Zfit
        if args.eta > 0.0:
            # Transductive: holdout and unlabeled rows join the graph with zero data weight,
            # so their labels are never seen but their geometry shapes the fit.
            Z = np.vstack([Zfit, Zhold, Zunl])
            L = MultiModelGraphLaplacian().build_sparse_laplacian(
                [Z], k_neighbors=args.k_neighbors, metric=args.metric)
            M = np.concatenate([np.ones(len(Zfit)), np.zeros(len(Zhold) + len(Zunl))])
            Y = np.vstack([Y, np.zeros((len(Z) - len(Zfit), len(classes)))])

        solver = MasterClosedFormSolver()
        solver.fit(Z, Y, W0=W0, M=M, L=L, lambda_reg=args.lambda_reg, eta=args.eta)
        report: Dict[str, Any] = {
            "features": args.features,
            "block": args.block,
            "n_fit": int(len(fit_idx)),
            "n_holdout": int(n_hold),
            "n_unlabeled": int(len(Zunl)),
            "n_features": int(X.shape[1]),
            "n_constant_features": int(constant.sum()),
            "classes": [int(v) for v in classes],
            "lambda_reg": args.lambda_reg,
            "eta": args.eta,
            # transductive: holdout/unlabeled features sit in the graph (labels never do).
            "evaluation": "transductive" if args.eta > 0.0 else "inductive",
            "prior": args.prior,
            "diagnostics": solver.diagnostics_.__dict__,
            "fit_accuracy": float(np.mean(classes[np.argmax(solver.predict(Zfit), axis=1)] == y[fit_idx])),
            "holdout_accuracy": (float(np.mean(classes[np.argmax(solver.predict(Zhold), axis=1)] == y[hold_idx]))
                                 if n_hold else None),
        }
        if args.out:
            solver.save(args.out, mean=mean, std=std, classes=classes)
            report["artifact"] = args.out
    except Exception as e:
        sys.stderr.write(f"{c['RED']}Error: {e}{c['RESET']}\n")
        return 1

    if args.json:
        print(json.dumps(report, indent=2))
        return 0
    d = report["diagnostics"]
    print(f"{c['BOLD']}{c['CYAN']}[Gen-Zero Master Closed-Form Head]{c['RESET']}")
    print(f"  {c['BOLD']}Rows:{c['RESET']}      fit={report['n_fit']} holdout={report['n_holdout']} "
          f"unlabeled={report['n_unlabeled']} dim={report['n_features']}")
    print(f"  {c['BOLD']}Objective:{c['RESET']} {d['objective']:.4f} (data {d['data_term']:.4f}, "
          f"prior {d['prior_term']:.4f}, graph {d['graph_term']:.4f})")
    print(f"  {c['BOLD']}Condition:{c['RESET']} {d['condition_number']:.3e}")
    print(f"  {c['BOLD']}Fit acc:{c['RESET']}   {report['fit_accuracy']:.4f}")
    if report["holdout_accuracy"] is not None:
        print(f"  {c['BOLD']}Holdout:{c['RESET']}   {report['holdout_accuracy']:.4f}")
    return 0


# ---------------------------------------------------------------------------
# Subcommand: candidate-prior (ESZSL closed-form candidate semantic prior, W0 = A E^T)
# ---------------------------------------------------------------------------
def cmd_candidate_prior(args: argparse.Namespace) -> int:
    c = _colors(args.no_color)
    import numpy as np
    from gen_zero.manifold import CandidateSemanticPrior

    try:
        with np.load(args.features, allow_pickle=False) as data:
            for key in ("Z", "Y", "E"):
                if key not in data:
                    raise KeyError(f"{key!r} not found in {args.features} (available: {list(data.keys())})")
            Z, Y, E = data["Z"], data["Y"], data["E"]
        prior = CandidateSemanticPrior()
        W0, A = prior.compute_w0_prior(Z, Y, E, gamma=args.gamma, delta=args.delta)
        out_path = args.out if args.out.endswith(".npz") else f"{args.out}.npz"
        np.savez(args.out, W0=W0, A=A)
    except Exception as e:
        sys.stderr.write(f"{c['RED']}Error: {e}{c['RESET']}\n")
        return 1

    if args.json:
        print(json.dumps({"W0_shape": list(W0.shape), "A_shape": list(A.shape), "out": out_path}, indent=2))
        return 0

    print(f"{c['BOLD']}{c['CYAN']}[Gen-Zero Candidate Semantic Prior]{c['RESET']}")
    print(f"  {c['BOLD']}Features:{c['RESET']} {args.features}")
    print(f"  {c['BOLD']}W0 shape:{c['RESET']} {W0.shape}  (d x K classifier)")
    print(f"  {c['BOLD']}A shape:{c['RESET']}  {A.shape}  (d x q manifold-to-candidate map)")
    print(f"  {c['BOLD']}Saved:{c['RESET']}    {out_path}")
    return 0


# ---------------------------------------------------------------------------
# Subcommand: manifold-fuse (GCCA shared+private / anchor mid-fusion)
# ---------------------------------------------------------------------------
def cmd_manifold_fuse(args: argparse.Namespace) -> int:
    c = _colors(args.no_color)
    import numpy as np
    from gen_zero.manifold import GCCAMidFusion, RelativeAnchorEncoder

    model_cls = GCCAMidFusion if args.mode == "gcca" else RelativeAnchorEncoder

    try:
        with np.load(args.features, allow_pickle=False) as data:
            missing = [k for k in args.view_keys if k not in data]
            if missing:
                raise KeyError(f"view keys {missing} not found in {args.features} (available: {list(data.keys())})")
            views = [np.asarray(data[k], dtype=np.float64) for k in args.view_keys]

        if args.artifact:
            model = model_cls.load(args.artifact)
        elif args.mode == "gcca":
            model = GCCAMidFusion().fit(
                views, shared_dim=args.shared_dim, residual_dim=args.residual_dim, reg=args.reg)
        else:
            model = RelativeAnchorEncoder().fit(views, n_anchors=args.n_anchors, seed=args.seed)

        if args.save_artifact:
            model.save(args.save_artifact)

        fused = model.transform(views)
    except Exception as e:
        sys.stderr.write(f"{c['RED']}Error: {e}{c['RESET']}\n")
        return 1

    if args.output:
        np.save(args.output, fused)

    summary = {
        "mode": args.mode,
        "n_views": len(views),
        "n_samples": fused.shape[0],
        "fused_dim": fused.shape[1],
        "artifact_loaded": bool(args.artifact),
        "artifact_saved": args.save_artifact,
        "output": args.output,
    }
    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print(f"{c['BOLD']}{c['CYAN']}[Gen-Zero Manifold Fuse]{c['RESET']}")
        print(f"  {c['BOLD']}Mode:{c['RESET']}       {summary['mode']}")
        print(f"  {c['BOLD']}Views:{c['RESET']}      {summary['n_views']}")
        print(f"  {c['BOLD']}Samples:{c['RESET']}    {summary['n_samples']}")
        print(f"  {c['BOLD']}Fused dim:{c['RESET']}  {summary['fused_dim']}")
        if args.output:
            print(f"  {c['BOLD']}Saved to:{c['RESET']}  {args.output}")
    return 0


# ---------------------------------------------------------------------------
# Subcommand: status / version
# ---------------------------------------------------------------------------
def _load_cad_items(args: argparse.Namespace) -> List[tuple]:
    if args.input:
        items = []
        for n, line in enumerate(Path(args.input).read_text(encoding="utf-8").splitlines(), 1):
            if line.strip():
                row = json.loads(line)
                if not isinstance(row.get("question"), str) or not isinstance(row.get("context"), str):
                    raise ValueError(f"{args.input}:{n} needs string 'question' and 'context'")
                items.append((row["question"], row["context"]))
        if not items:
            raise ValueError(f"{args.input} has no records")
        return items
    if not args.question or not args.context:
        raise ValueError("give --question and --context, or --input FILE.jsonl")
    return [(args.question, args.context)]


def cmd_cad(args: argparse.Namespace) -> int:
    """Contrastive-decoding yes/no/maybe inference on a local Qwen GGUF (pure forward, no training)."""
    c = _colors(args.no_color)
    try:
        items = _load_cad_items(args)
        if bool(args.gguf) == bool(args.hf_model):
            raise ValueError("give exactly one of --gguf or --hf-model")
        if args.hf_model:
            if args.workers > 1:
                raise ValueError("--workers > 1 needs --gguf (the pool runs GGUF engines)")
            if args.numa_pin:
                raise ValueError("--numa-pin requires --gguf (NUMA pinning is only supported for GGUF)")
            from gen_zero.causal.cad_engine import CADEngine
            engine = CADEngine.from_hf(args.hf_model, tokenizer_path=args.hf_tokenizer, head_path=args.head, alpha=args.alpha)
            results = [engine.classify(q, ctx).to_dict() for q, ctx in items]
        else:
            if args.hf_tokenizer:
                raise ValueError("--hf-tokenizer requires --hf-model")
            if args.workers > 1 or args.numa_pin:
                from gen_zero.causal.gguf_parallel_pool import GGUFParallelPool
                with GGUFParallelPool(args.gguf, workers=args.workers, threads_per_worker=args.threads,
                                      n_ctx=args.n_ctx, head_path=args.head, numa_pin=args.numa_pin,
                                      use_mmap=not args.numa_pin, alpha=args.alpha) as pool:
                    results = pool.classify_batch(items)
            else:
                from gen_zero.causal.cad_engine import CADEngine
                engine = CADEngine.from_gguf(args.gguf, head_path=args.head, n_ctx=args.n_ctx,
                                             n_threads=args.threads, alpha=args.alpha)
                results = [engine.classify(q, ctx).to_dict() for q, ctx in items]
    except Exception as e:
        sys.stderr.write(f"{c['RED']}Error: {type(e).__name__}: {e}{c['RESET']}\n")
        return 1
    if args.json:
        print(json.dumps(results if args.input else results[0], indent=2, ensure_ascii=False))
        return 0
    for (question, _), r in zip(items, results):
        mode = "calibrated" if r["calibrated"] else "UNCALIBRATED (no --head)"
        probs = " ".join(f"{k}={v:.3f}" for k, v in r["probabilities"].items())
        print(f"{c['BOLD']}{r['label']}{c['RESET']}  {probs}  margin={r['margin']:.3f}  [{mode}]  {question[:60]}")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    c = _colors(args.no_color)
    endpoint = resolve_endpoint()
    try:
        token = resolve_api_token()
    except ValueError:
        token = ""

    print(f"{c['BOLD']}{c['CYAN']}[Gen-Zero Architecture Status]{c['RESET']}")
    print(f"  {c['BOLD']}Engine Version:{c['RESET']}   {SERVER_NAME} v{SERVER_VERSION}")
    print(f"  {c['BOLD']}Canonical Tool:{c['RESET']}   zero (Single Polymorphic Entrypoint)")
    print(f"  {c['BOLD']}Decision Endpoint:{c['RESET']} {endpoint}")
    print(f"  {c['BOLD']}Auth Token:{c['RESET']}       {'configured (hidden)' if token else 'not set'}")

    # Ping endpoint
    t0 = time.perf_counter()
    server = MCPServer()
    ping_res = asyncio.run(server.handle_request({"jsonrpc": "2.0", "id": 1, "method": "ping"}))
    elapsed = (time.perf_counter() - t0) * 1000.0

    if ping_res and "result" in ping_res:
        print(f"  {c['BOLD']}MCP Protocol:{c['RESET']}     {c['GREEN']}ONLINE (MCP 2024-11-05, {elapsed:.2f}ms){c['RESET']}")
    else:
        print(f"  {c['BOLD']}MCP Protocol:{c['RESET']}     {c['RED']}OFFLINE{c['RESET']}")
    return 0


# ---------------------------------------------------------------------------
# Subcommand: harness
# ---------------------------------------------------------------------------
def cmd_harness(args: argparse.Namespace) -> int:
    c = _colors(args.no_color)
    action = getattr(args, "harness_action", "setup") or "setup"

    if action == "setup":
        from gen_zero.harness.setup import setup_host_harness
        target_path = getattr(args, "output", None)
        res = setup_host_harness(output_path=target_path)
        if getattr(args, "json", False):
            print(json.dumps(res, indent=2))
        else:
            print(f"{c['BOLD']}{c['CYAN']}[Gen-Zero Host Capability Harness Setup]{c['RESET']}")
            print(f"  {c['BOLD']}Manifest Path:{c['RESET']}      {res['manifest_path']}")
            print(f"  {c['BOLD']}Snapshot Hash:{c['RESET']}      {res['snapshot_hash'][:16]}...")
            print(f"  {c['BOLD']}Host CLI Tools:{c['RESET']}     {res['host_cli_detected']}")
            print(f"  {c['BOLD']}Skills/Subagents:{c['RESET']}   {res['core_skills_registered']}")
            print(f"  {c['BOLD']}Total Capabilities:{c['RESET']} {res['total_capabilities']}")
            print(f"  {c['BOLD']}Elapsed Time:{c['RESET']}       {res['setup_time_ms']:.2f} ms")
            print(f"  {c['GREEN']}✓ Harness successfully configured and bound.{c['RESET']}")
        return 0
    else:
        sys.stderr.write(f"{c['RED']}Unknown harness action: {action}{c['RESET']}\n")
        return 2


# ---------------------------------------------------------------------------
# CLI Parser Definition & Main
# ---------------------------------------------------------------------------
def build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gen-zero",
        description="Gen-Zero Canonical Shell CLI (0-Token Pure-Prefill Decision & Cognitive Engine)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 1. Ask a non-autoregressive decision question
  gen-zero ask "Node memory pressure 92%" "Is it safe to run online migration?"

  # 2. Fast sub-millisecond tool pruning
  gen-zero route "Deploy k8s pod cluster" --tools tools.json --top-k 3

  # 3. Counterfactual imagination tree lookahead with CP-SAT safety
  gen-zero imagine "High checkout error rate" --actions CIRCUIT_BREAK ROLLBACK DRAIN

  # 4. Spatiotemporal streaming with Attention Sinks constant memory
  gen-zero stream "frame_camera_001" --actions ACT_A ACT_B HOLD

  # 5. Propositional semantic search & Boolean filtering
  gen-zero grep --expr '("db error" AND NOT "timeout")' server.log

  # 6. Verbatim context compaction (0.0 fact mutation guarantee)
  gen-zero compact history.json --head 5 --tail 5

  # 7. Launch MCP server
  gen-zero mcp --transport stdio

  # 8. Project hidden reps through the manifold anchor bridge
  gen-zero anchor --features features.npz --domain-id 0 --candidates proceed,abort
"""
    )
    parser.add_argument("--no-color", action="store_true", help="Disable colored ANSI terminal output")
    parser.add_argument("-v", "--version", action="store_true", help="Show Gen-Zero version and exit")

    subparsers = parser.add_subparsers(dest="subcommand", help="Available subcommands")

    # ask
    p_ask = subparsers.add_parser("ask", help="Single-forward typed decision micro-cores (Noul, Choice, Score)")
    p_ask.add_argument("state", nargs="?", default="", help="Environment state description or context")
    p_ask.add_argument("question", nargs="?", default="", help="Decision query or question")
    p_ask.add_argument("--type", choices=["noul", "choice", "score"], default="noul", help="Decision primitive type (default: noul)")
    p_ask.add_argument("--choices", nargs="*", default=[], help="Candidate options (for 'choice' type)")
    p_ask.add_argument("--criteria", nargs="*", default=[], help="Score ratings or evaluation criteria")
    p_ask.add_argument("--json-input", help="File path or raw JSON string defining state and questions")
    p_ask.add_argument("--json", action="store_true", help="Output raw JSON result")

    # route
    p_route = subparsers.add_parser("route", help="Sub-5ms adaptive tool catalog pruning router")
    p_route.add_argument("goal", help="Task goal or instruction to route tools for")
    p_route.add_argument("--tools", required=True, help="File path to tools.json or raw JSON catalog array")
    p_route.add_argument("--top-k", type=int, default=5, help="Maximum number of pruned tools to return (default: 5)")
    p_route.add_argument("--context", default="", help="Optional execution context")
    p_route.add_argument("--json", action="store_true", help="Output raw JSON result")

    # imagine
    p_imagine = subparsers.add_parser("imagine", help="World model counterfactual tree lookahead with CP-SAT safety")
    p_imagine.add_argument("state", help="Current environment state representation")
    p_imagine.add_argument("--actions", nargs="+", required=True, help="List of candidate actions to simulate")
    p_imagine.add_argument("--horizon", type=int, default=4, help="Virtual lookahead depth (default: 4)")
    p_imagine.add_argument("--no-cpsat", action="store_true", help="Disable CP-SAT 0-1 ILP safety verification")
    p_imagine.add_argument("--json", action="store_true", help="Output raw JSON result")

    # stream
    p_stream = subparsers.add_parser("stream", help="Spatiotemporal streaming with Attention Sinks constant memory")
    p_stream.add_argument("observation", help="Incoming observation frame or state string")
    p_stream.add_argument("--actions", nargs="+", required=True, help="Available action candidates")
    p_stream.add_argument("--prompt", default="", help="Optional context prompt")
    p_stream.add_argument("--json", action="store_true", help="Output raw JSON result")

    # grep (special: passes remaining args to grep runner)
    subparsers.add_parser("grep", help="Propositional semantic search & Boolean logic filtering (use 'gen-zero grep --help')")

    # compact
    p_compact = subparsers.add_parser("compact", help="Verbatim context compaction with 0.0 fact mutation guarantee")
    p_compact.add_argument("file", nargs="?", default="-", help="JSON file containing messages (or '-' for stdin)")
    p_compact.add_argument("--head", type=int, default=5, help="Number of head lines to keep for verbose logs (default: 5)")
    p_compact.add_argument("--tail", type=int, default=5, help="Number of tail lines to keep for verbose logs (default: 5)")
    p_compact.add_argument("--threshold", type=int, default=15, help="Line threshold to trigger log truncation (default: 15)")
    p_compact.add_argument("-o", "--output", help="Save compacted messages JSON to file")
    p_compact.add_argument("--json", action="store_true", help="Output complete JSON payload with summary metrics")

    # mcp
    p_mcp = subparsers.add_parser("mcp", help="Launch Gen-Zero Model Context Protocol (MCP) server")
    p_mcp.add_argument("--transport", choices=["stdio", "sse"], default="stdio", help="MCP transport: stdio (default) or sse")
    p_mcp.add_argument("--host", default="0.0.0.0", help="Host address to bind for SSE transport")
    p_mcp.add_argument("--port", type=int, default=8999, help="Port to bind for SSE transport (default: 8999)")
    p_mcp.add_argument("--token", help="Optional Bearer authentication token for SSE")

    # semantic scorer
    from gen_zero.service.ports import DEFAULT_SEMANTIC_PORT
    p_sem = subparsers.add_parser(
        "semantic", help="Run the semantic scorer used by the Rust zero bridge (default port 8995)")
    p_sem.add_argument("--host", default=os.environ.get("GENZERO_SEMANTIC_HOST", "127.0.0.1"),
                       help="Host address to bind (default: 127.0.0.1)")
    p_sem.add_argument("--port", type=int,
                       default=int(os.environ.get("GENZERO_SEMANTIC_PORT", str(DEFAULT_SEMANTIC_PORT))),
                       help="Port to bind (default: 8995; the Rust bridge's GENZERO_PYTHON_ENDPOINT default)")

    # anchor
    p_anchor = subparsers.add_parser(
        "anchor", help="Project high-dim representation to 128-d nanocore_state via manifold anchor bridge")
    p_anchor.add_argument("--core-manifest", default=str(DEFAULT_ANCHOR_CORE_MANIFEST),
                           help="Target core space manifest JSON (default: the manifest sidecar of the "
                                "default --artifact)")
    p_anchor.add_argument("--features", required=True, help="Path to raw features .npz file")
    p_anchor.add_argument("--artifact", default=str(DEFAULT_ANCHOR_ARTIFACT),
                           help="Path to a fitted ManifoldAnchorDistiller .npz artifact")
    p_anchor.add_argument("--block", default="test_full", help="Array key inside the .npz feature store (default: test_full)")
    p_anchor.add_argument("--row", type=int, default=0, help="Row index within the block to project (default: 0)")
    p_anchor.add_argument("--domain-id", type=int, default=0, help="NanoCore domain ID (default: 0)")
    p_anchor.add_argument("--candidates", default="proceed,abort", help="Comma-separated candidate action names")
    p_anchor.add_argument("--execute", action="store_true", help="Execute decision in-process via GenZero.decide_nanocore")
    p_anchor.add_argument("--json", action="store_true", help="Output raw JSON payload")

    # manifold-fit
    p_mfit = subparsers.add_parser(
        "manifold-fit",
        help="Fit the master closed-form head (weighted ridge + prior + graph Laplacian) on a .npz feature store")
    p_mfit.add_argument("--features", required=True, help="Path to .npz feature store")
    p_mfit.add_argument("--block", default="train_full", help="Labeled feature array key (default: train_full)")
    p_mfit.add_argument("--label-key", default="train_label", help="Integer label array key (default: train_label)")
    p_mfit.add_argument("--unlabeled-block", help="Optional unlabeled array key; joins the graph when --eta > 0")
    p_mfit.add_argument("--holdout", type=float, default=0.2, help="Held-out fraction of labeled rows (default: 0.2)")
    p_mfit.add_argument("--seed", type=int, default=0, help="Split seed (default: 0)")
    p_mfit.add_argument("--lambda-reg", type=float, default=100.0, help="Ridge / prior strength (default: 100)")
    p_mfit.add_argument("--eta", type=float, default=0.0, help="Graph Laplacian strength (default: 0, off)")
    p_mfit.add_argument("--k-neighbors", type=int, default=5, help="kNN graph degree (default: 5)")
    p_mfit.add_argument("--metric", choices=["cosine", "euclidean"], default="cosine", help="kNN metric")
    p_mfit.add_argument("--prior", help="Earlier manifold-fit artifact used as W0")
    p_mfit.add_argument("--out", help="Write the fitted head (.npz) here")
    p_mfit.add_argument("--json", action="store_true", help="Output raw JSON report")

    # candidate-prior
    p_cprior = subparsers.add_parser(
        "candidate-prior",
        help="Fit the ESZSL closed-form candidate semantic prior W0 = A E^T from Z/Y/E in an .npz")
    p_cprior.add_argument("--features", required=True,
                           help="Path to .npz containing Z (N x d manifold embeddings), "
                                "Y (N x K one-hot labels), E (K x q candidate embeddings)")
    p_cprior.add_argument("--gamma", type=float, default=10.0, help="Ridge weight on A E^T (default: 10.0)")
    p_cprior.add_argument("--delta", type=float, default=10.0, help="Ridge weight on Z A (default: 10.0)")
    p_cprior.add_argument("--out", required=True, help="Output .npz path to save W0 and A")
    p_cprior.add_argument("--json", action="store_true", help="Output raw JSON summary")

    # manifold-fuse
    p_fuse = subparsers.add_parser(
        "manifold-fuse",
        help="Regularized GCCA shared+private mid-fusion (or relative-anchor encoding) across aligned multi-model feature views")
    p_fuse.add_argument("--mode", choices=["gcca", "anchor"], default="gcca",
                         help="gcca: shared+private mid-fusion; anchor: relative anchor coordinates (default: gcca)")
    p_fuse.add_argument("--features", required=True, help="Path to a .npz file containing one 2D array per aligned view")
    p_fuse.add_argument("--view-keys", nargs="+", required=True,
                         help="Array keys inside --features, one per view, sample-aligned by row")
    p_fuse.add_argument("--artifact", help="Path to a previously fit .npz artifact to load instead of fitting fresh")
    p_fuse.add_argument("--save-artifact", help="Fit on --features and save the fitted model to this path")
    p_fuse.add_argument("--shared-dim", type=int, default=64, help="GCCA shared-space dimension (default: 64)")
    p_fuse.add_argument("--residual-dim", type=int, default=16, help="GCCA per-view private residual dimension (default: 16)")
    p_fuse.add_argument("--reg", type=float, default=1e-3, help="GCCA ridge regularizer (default: 1e-3)")
    p_fuse.add_argument("--n-anchors", type=int, default=128, help="Anchor-mode anchor count (default: 128)")
    p_fuse.add_argument("--seed", type=int, default=42, help="Anchor-mode anchor sampling seed (default: 42)")
    p_fuse.add_argument("-o", "--output", help="Save the fused/encoded output matrix as .npy")
    p_fuse.add_argument("--json", action="store_true", help="Output a JSON summary instead of a human-readable one")

    # harness
    p_harness = subparsers.add_parser("harness", help="Automated host capability discovery and setup")
    harness_subparsers = p_harness.add_subparsers(dest="harness_action", help="Harness actions")
    p_harness_setup = harness_subparsers.add_parser("setup", help="Auto-probe host environment and build capability manifest")
    p_harness_setup.add_argument("-o", "--output", help="Output path for capabilities manifest JSON (default: .gen_zero_capabilities.json)")
    p_harness_setup.add_argument("--json", action="store_true", help="Output raw JSON summary")

    # status
    # cad
    p_cad = subparsers.add_parser("cad", help="Contrastive-decoding yes/no/maybe inference on a local Qwen2.5-1.5B GGUF")
    p_cad.add_argument("--gguf", help="Path to the Qwen2.5-1.5B-Instruct Q4_K_M GGUF (see gen_zero.scripts.setup_qwen15b_models)")
    p_cad.add_argument("--hf-tokenizer", help="Tokenizer directory for --hf-model (default: the model directory)")
    p_cad.add_argument("--hf-model", help="Local transformers causal-LM directory instead of --gguf (single process)")
    p_cad.add_argument("--question", default="", help="Question to answer")
    p_cad.add_argument("--context", default="", help="Context passage the answer must rest on")
    p_cad.add_argument("--input", help="JSONL file of {question, context} records (batch mode)")
    p_cad.add_argument("--head", help="User-supplied calibration head JSON (gen_zero.cad_head.v1); omit for uncalibrated scores")
    p_cad.add_argument("--alpha", type=float, default=0.5, help="Prior weight in delta = cond - alpha*prior (uncalibrated mode; default 0.5)")
    p_cad.add_argument("--workers", type=int, default=1, help="Worker processes (>1 uses the parallel GGUF pool)")
    p_cad.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 2) // 2), help="Compute threads per worker")
    p_cad.add_argument("--n-ctx", type=int, default=2048, help="Context window in tokens")
    p_cad.add_argument("--numa-pin", action="store_true", help="Pin each worker to exclusive cores of one NUMA node (Linux)")
    p_cad.add_argument("--json", action="store_true", help="Output raw JSON result")

    subparsers.add_parser("status", help="Check engine connectivity, model status, and latency")

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    # Quick dispatch for grep subcommand to preserve flags (-e, -a, -v, etc.)
    if argv and argv[0] == "grep":
        return cmd_grep(argv[1:])

    parser = build_cli_parser()
    args, unknown = parser.parse_known_args(argv)

    if args.version:
        print(f"gen-zero {SERVER_VERSION}")
        return 0

    if not args.subcommand:
        parser.print_help()
        return 0

    if args.subcommand == "ask":
        return cmd_ask(args)
    elif args.subcommand == "route":
        return cmd_route(args)
    elif args.subcommand == "imagine":
        return cmd_imagine(args)
    elif args.subcommand == "stream":
        return cmd_stream(args)
    elif args.subcommand == "compact":
        return cmd_compact(args)
    elif args.subcommand == "mcp":
        return cmd_mcp(args)
    elif args.subcommand == "semantic":
        return cmd_semantic(args)
    elif args.subcommand == "anchor":
        return cmd_anchor(args)
    elif args.subcommand == "manifold-fit":
        return cmd_manifold_fit(args)
    elif args.subcommand == "candidate-prior":
        return cmd_candidate_prior(args)
    elif args.subcommand == "manifold-fuse":
        return cmd_manifold_fuse(args)
    elif args.subcommand == "harness":
        return cmd_harness(args)
    elif args.subcommand == "cad":
        return cmd_cad(args)
    elif args.subcommand == "status":
        return cmd_status(args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
