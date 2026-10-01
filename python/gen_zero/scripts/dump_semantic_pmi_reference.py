"""Dump Python fp32 PMI scores for the native Rust scorer's parity test.

Writes ``crates/gen-zero-model/tests/fixtures/python_pmi_reference.json``:
ask and route cases scored by :class:`gen_zero.service.semantic_scorer.SemanticScorer`
(the scorer behind the HTTP bridge). ``qwen_native_parity.rs`` scores the same
cases in process and compares candidate by candidate.

Run from ``python/``: ``PYTHONPATH=. python3 gen_zero/scripts/dump_semantic_pmi_reference.py``
"""
import json
from pathlib import Path

from gen_zero.service.app import ToolSpec, _state_text, tool_continuation
from gen_zero.service.semantic_scorer import ROUTE_FRAME, get_semantic_scorer, select_ask_frame

OUT = Path(__file__).resolve().parents[3] / "crates/gen-zero-model/tests/fixtures/python_pmi_reference.json"

CASES = [
    {"kind": "ask", "context": "The leaves of my tomato plants are dry and drooping.",
     "candidates": ["water the plants", "file the quarterly tax return", "reboot the router"], "history": []},
    {"kind": "ask", "context": "The build is broken after the last merge.",
     "candidates": ["revert_the_merge", "go_home", "read the compiler error output carefully"],
     "history": ["open_ci_logs"]},
    {"kind": "ask", "context": "Question: Is Paris the capital of France?", "candidates": ["yes", "no"], "history": []},
    {"kind": "ask", "context": "服务器磁盘快满了，需要清理空间",
     "candidates": ["删除旧日志文件", "格式化系统盘", "购买新的显示器"], "history": [],
     "state": {"disk": "97%", "host": "web-1"}},
    {"kind": "route", "context": "Find every TODO comment in the repository",
     "tools": [{"name": "grep", "description": "Search file contents for a text pattern"},
               {"name": "tool_7", "description": "Send an email to a colleague"},
               {"name": "compact", "description": None}]},
]


def main() -> None:
    scorer = get_semantic_scorer()
    out = []
    for case in CASES:
        case = dict(case)
        if case["kind"] == "ask":
            context = _state_text(case["context"], case.get("state"))
            frame, _ = select_ask_frame(context, case["candidates"], case["history"], None)
            result = scorer.score(context, case["candidates"], frame=frame, history=case["history"])
        else:
            tools = [ToolSpec(**t) for t in case["tools"]]
            result = scorer.score(case["context"], [t.name for t in tools], frame=ROUTE_FRAME,
                                  texts=[tool_continuation(t) for t in tools])
        case["frame"] = frame if case["kind"] == "ask" else ROUTE_FRAME
        case["python"] = [c.as_dict() for c in result.candidates]
        out.append(case)
    OUT.write_text(json.dumps({"scorer": scorer.scorer_id, "cases": out}, ensure_ascii=False, indent=1) + "\n",
                   encoding="utf-8")
    print(f"wrote {OUT} ({scorer.scorer_id})")


if __name__ == "__main__":
    main()
