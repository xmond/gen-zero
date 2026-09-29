#!/usr/bin/env python3
"""
Test fast non-thinking multi-path sampling on Qwen3.8-27B (port 11503)
with reasoning_effort="none" and temperature=0.8.
"""

import json
import re
import time
import urllib.request
from collections import Counter
from typing import Dict, List, Optional

API_URL = "http://127.0.0.1:11503/v1/chat/completions"

def call_qwen_api(prompt: str, temperature: float = 0.8, max_tokens: int = 150, n_paths: int = 3) -> List[str]:
    """Calls local llama-server API with reasoning_effort='none' (THINK OFF)."""
    results = []
    for _ in range(n_paths):
        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "reasoning_effort": "none",
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(API_URL, data=data, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                res = json.loads(resp.read().decode("utf-8"))
                choice = res["choices"][0]["message"]["content"]
                results.append(choice)
        except Exception as e:
            print(f"API call error: {e}")
            results.append("")
    return results

def parse_gsm8k_choice(text: str, candidates: List[str]) -> Optional[str]:
    m = re.search(r"[Ff]inal\s+answer:\s*\(?([A-D])\)?", text)
    if m and m.group(1).upper() in candidates:
        return m.group(1).upper()
    m2 = re.search(r"\(?([A-D])\)?\s*$", text.strip())
    if m2 and m2.group(1).upper() in candidates:
        return m2.group(1).upper()
    for c in ["A", "B", "C", "D"]:
        if f"({c})" in text:
            return c
    return None

def parse_summeval_score(text: str) -> Optional[str]:
    m = re.search(r"[Ff]inal\s+score:\s*([1-5])", text)
    if m:
        return m.group(1)
    m2 = re.search(r"[Ss]core:\s*([1-5])", text)
    if m2:
        return m2.group(1)
    m3 = re.search(r"\b([1-5])\s*/\s*5\b", text)
    if m3:
        return m3.group(1)
    digits = re.findall(r"\b([1-5])\b", text)
    if digits:
        return digits[-1]
    return None

def evaluate_gsm8k(data_path: str, limit: int = 15, n_paths: int = 3) -> Dict:
    print(f"\n=======================================================")
    print(f" EVALUATING GSM8K (THINK OFF | limit={limit} | paths={n_paths} | temp=0.8)")
    print(f"=======================================================")
    correct_single = 0
    correct_consensus = 0
    total = 0
    latencies = []

    with open(data_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f][:limit]

    for idx, rec in enumerate(records):
        ctx = rec["context"]
        cands = rec["candidates"]
        gt = rec["ground_truth"]
        prompt = (
            f"{ctx}\n\n"
            "Solve concisely step by step and conclude on a separate line with 'Final answer: (X)' where X is the option letter."
        )
        t0 = time.perf_counter()
        paths = call_qwen_api(prompt, temperature=0.8, max_tokens=160, n_paths=n_paths)
        elapsed = time.perf_counter() - t0
        latencies.append(elapsed)

        parsed_votes = [parse_gsm8k_choice(p, cands) for p in paths]
        valid_votes = [v for v in parsed_votes if v is not None]

        p0 = parsed_votes[0] if parsed_votes else None
        if p0 == gt:
            correct_single += 1

        if valid_votes:
            consensus = Counter(valid_votes).most_common(1)[0][0]
        else:
            consensus = None

        if consensus == gt:
            correct_consensus += 1
        total += 1

        is_correct = (consensus == gt)
        print(f"[{idx+1:02d}/{limit}] GT: {gt} | Votes: {parsed_votes} | Consensus: {consensus} | {'PASS' if is_correct else 'FAIL'} | {elapsed:.2f}s ({elapsed/n_paths:.2f}s/path)")

    acc_single = (correct_single / total) * 100 if total else 0
    acc_consensus = (correct_consensus / total) * 100 if total else 0
    avg_lat = sum(latencies)/len(latencies)
    print(f"\n>>> GSM8K Single-path Accuracy: {acc_single:.2f}% ({correct_single}/{total})")
    print(f">>> GSM8K Consensus Accuracy:   {acc_consensus:.2f}% ({correct_consensus}/{total})")
    print(f">>> Avg latency per sample:      {avg_lat:.2f}s (single path ~{avg_lat/n_paths:.2f}s)")
    return {
        "task": "gsm8k",
        "single_acc": acc_single,
        "consensus_acc": acc_consensus,
        "samples": total,
        "avg_latency": avg_lat
    }

def evaluate_summeval(data_path: str, limit: int = 15, n_paths: int = 3) -> Dict:
    print(f"\n=======================================================")
    print(f" EVALUATING SUMMEVAL (THINK OFF | limit={limit} | paths={n_paths} | temp=0.8)")
    print(f"=======================================================")
    correct_single = 0
    correct_consensus = 0
    within_one_count = 0
    total = 0
    latencies = []

    with open(data_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f][:limit]

    for idx, rec in enumerate(records):
        ctx = rec["context"]
        cands = rec["candidates"]
        gt = rec["ground_truth"]
        prompt = (
            f"{ctx}\n\n"
            "Evaluate whether the candidate summary contains factual errors or hallucinations compared to the source document.\n"
            "Briefly state the reason, then conclude on a separate line with 'Final score: <score>' where <score> is an integer from 1 (unsupported / hallucination) to 5 (fully supported)."
        )
        t0 = time.perf_counter()
        paths = call_qwen_api(prompt, temperature=0.8, max_tokens=140, n_paths=n_paths)
        elapsed = time.perf_counter() - t0
        latencies.append(elapsed)

        parsed_votes = [parse_summeval_score(p) for p in paths]
        valid_votes = [v for v in parsed_votes if v is not None]

        p0 = parsed_votes[0] if parsed_votes else None
        if p0 == gt:
            correct_single += 1

        if valid_votes:
            consensus = Counter(valid_votes).most_common(1)[0][0]
        else:
            consensus = None

        if consensus == gt:
            correct_consensus += 1
        
        if consensus is not None and abs(int(consensus) - int(gt)) <= 1:
            within_one_count += 1

        total += 1
        is_correct = (consensus == gt)
        print(f"[{idx+1:02d}/{limit}] GT: {gt} | Votes: {parsed_votes} | Consensus: {consensus} | {'PASS' if is_correct else 'FAIL'} | {elapsed:.2f}s ({elapsed/n_paths:.2f}s/path)")

    acc_single = (correct_single / total) * 100 if total else 0
    acc_consensus = (correct_consensus / total) * 100 if total else 0
    within_one_pct = (within_one_count / total) * 100 if total else 0
    avg_lat = sum(latencies)/len(latencies)
    print(f"\n>>> SummEval Exact Accuracy:     {acc_consensus:.2f}% ({correct_consensus}/{total})")
    print(f">>> SummEval Within-1 Accuracy:  {within_one_pct:.2f}% ({within_one_count}/{total})")
    print(f">>> Avg latency per sample:      {avg_lat:.2f}s (single path ~{avg_lat/n_paths:.2f}s)")
    return {
        "task": "summeval",
        "exact_acc": acc_consensus,
        "within_one_acc": within_one_pct,
        "samples": total,
        "avg_latency": avg_lat
    }

if __name__ == "__main__":
    gsm_res = evaluate_gsm8k("D:/workspace/gen-zero-eval/data_13/gsm8k.jsonl", limit=15, n_paths=3)
    sum_res = evaluate_summeval("D:/workspace/gen-zero-eval/data_13/summeval.jsonl", limit=15, n_paths=3)
    print("\n================ FINAL REPORT (THINK OFF) ================")
    print("GSM8K (15 samples):", gsm_res)
    print("SummEval (15 samples):", sum_res)
