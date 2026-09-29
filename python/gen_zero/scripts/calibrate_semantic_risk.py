"""Calibrate and evaluate the semantic risk classifier.

Thresholds come from ``risk_data/calibration.jsonl`` only. ``heldout.jsonl``
is scored afterwards with those thresholds and never feeds back into them.

    PYTHONPATH=python python3 python/gen_zero/scripts/calibrate_semantic_risk.py

Writes ``risk_data/report.json``. Copy the two thresholds into
``semantic_risk.py`` when they change.
"""
from __future__ import annotations

import json
import math
from typing import Dict, List

from gen_zero.service.semantic_risk import DATA_DIR, get_risk_classifier, load_jsonl

MARGIN = 0.02


def auc(pos: List[float], neg: List[float]) -> float:
    pairs = [(a > b) + 0.5 * (a == b) for a in pos for b in neg]
    return sum(pairs) / len(pairs)


def score(rows: List[Dict]) -> List[Dict]:
    clf = get_risk_classifier()
    out = []
    for r in rows:
        a = clf.assess(r["text"])
        out.append({**r, "p_dangerous": a.p_dangerous, "forward_ms": round(a.forward_ms, 1)})
    return out


def confusion(rows: List[Dict], escalate: float, hard: float) -> Dict[str, Dict[str, int]]:
    table: Dict[str, Dict[str, int]] = {"dangerous": {}, "safe": {}}
    for r in rows:
        p = r["p_dangerous"]
        tier = "HardStop" if p >= hard else "Escalate" if p >= escalate else "Proceed"
        key = "dangerous" if r["label"] else "safe"
        table[key][tier] = table[key].get(tier, 0) + 1
    return table


def summary(rows: List[Dict], escalate: float, hard: float) -> Dict:
    pos = [r["p_dangerous"] for r in rows if r["label"]]
    neg = [r["p_dangerous"] for r in rows if not r["label"]]
    return {
        "n": len(rows),
        "auc": auc(pos, neg),
        "min_dangerous": min(pos),
        "max_safe": max(neg),
        "confusion": confusion(rows, escalate, hard),
        "rows": sorted(rows, key=lambda r: r["p_dangerous"]),
    }


def main() -> None:
    calib = score(load_jsonl("calibration.jsonl"))
    pos = [r["p_dangerous"] for r in calib if r["label"]]
    neg = [r["p_dangerous"] for r in calib if not r["label"]]
    escalate = max(0.0, min(pos) - MARGIN)
    hard = min(1.0, max(neg) + MARGIN)
    heldout = score(load_jsonl("heldout.jsonl"))
    report = {
        "classifier": get_risk_classifier().classifier_id,
        "thresholds": {"escalate": escalate, "hard_stop": hard, "margin": MARGIN},
        "calibration": summary(calib, escalate, hard),
        "heldout": summary(heldout, escalate, hard),
    }
    (DATA_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("classifier", "thresholds")}, indent=1))
    for name in ("calibration", "heldout"):
        s = report[name]
        print(name, "auc=%.3f min_dangerous=%.3f max_safe=%.3f" % (s["auc"], s["min_dangerous"], s["max_safe"]),
              json.dumps(s["confusion"]))
        for r in s["rows"]:
            print("   y=%d %.3f %s" % (r["label"], r["p_dangerous"], r["text"]))


if __name__ == "__main__":
    main()
