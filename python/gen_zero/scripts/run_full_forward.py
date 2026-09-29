"""Execute end-to-end forward pass on real Qwen3.5-0.8B on Windows A100."""

import sys
import os
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

model_dir = os.environ.get("QWEN_MODEL_DIR", "models/Qwen3.5-0.8B")
image_path = os.environ.get("SAMPLE_IMAGE", "data/demo_brick.jpg")

print("=================================================================")
print("     GEN-ZERO REAL FORWARD PASS BENCHMARK (QWEN3.5-0.8B)        ")
print("=================================================================")
print(f"  Target Model Dir:   {model_dir}")
print(f"  Target Image:       {image_path}")
print(f"  Hardware:           {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
print("=================================================================\n")

# 1. Load Model & Tokenizer
t_load_start = time.perf_counter()
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if device == "cuda" else torch.float32

config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
hidden_size = getattr(config, "hidden_size", 1024)
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    config=config,
    dtype=dtype,
    trust_remote_code=True
).to(device).eval()
load_ms = (time.perf_counter() - t_load_start) * 1000.0
print(f"✓ Model & Tokenizer loaded to {device} in {load_ms:.1f} ms (Dynamic hidden_size: {hidden_size})")

# 2. Define Visual / Text Task & Action Candidates
prompt = "Scenario: Breakout Arcade Game. Observation: A ball is moving rapidly toward region 3. Question: Which action should the paddle execute immediately to intercept the ball? Options: A: LEFT, B: STAY, C: RIGHT, D: ABSTAIN. Answer:"
candidates = ["A", "B", "C", "D"]
action_map = {"A": "LEFT (向左移动)", "B": "STAY (保持原位)", "C": "RIGHT (向右移动)", "D": "ABSTAIN (安全放弃)"}

print(f"\n[Prompt Input]:\n{prompt}")
print(f"\n[Candidates to Evaluate]: {candidates} -> {[action_map[c] for c in candidates]}")

# 3. Step 1: 1-Pass Context Prefill & Cache Generation
t_prefill_start = time.perf_counter()
inputs = tokenizer(prompt, return_tensors="pt").to(device)

with torch.inference_mode():
    outputs = model(
        **inputs,
        use_cache=True,
        output_hidden_states=True
    )
    past_kv = outputs.past_key_values
    last_hidden = outputs.hidden_states[-1] # [1, seq_len, hidden_size]
    state_repr = last_hidden[:, -1, :]      # [1, hidden_size]

prefill_ms = (time.perf_counter() - t_prefill_start) * 1000.0
print(f"\n✓ Step 1 (Prefill & Hidden State Extraction):")
print(f"  - Extracted Dense State Representation: Shape {list(state_repr.shape)}, dtype: {state_repr.dtype}")
print(f"  - Prefill Latency: {prefill_ms:.2f} ms")

# 4. Step 2: Non-Autoregressive Direct Candidate Logit Scoring (Fork Cache)
t_scoring_start = time.perf_counter()
cand_logits = {}

with torch.inference_mode():
    for cand in candidates:
        cand_token_id = tokenizer.encode(cand, add_special_tokens=False)[0]
        # Direct single-step forward using prefilled past_key_values
        cand_input_tensor = torch.tensor([[cand_token_id]], device=device)
        cand_out = model(cand_input_tensor, past_key_values=past_kv, use_cache=False)
        cand_logits[cand] = cand_out.logits[0, 0, cand_token_id].item()

# Compute Softmax Probabilities
cand_tensor = torch.tensor([cand_logits[c] for c in candidates], dtype=torch.float32)
probs = torch.softmax(cand_tensor, dim=0).tolist()
prob_dict = {c: round(p, 4) for c, p in zip(candidates, probs)}
scoring_ms = (time.perf_counter() - t_scoring_start) * 1000.0

best_cand = max(prob_dict, key=prob_dict.get)

print(f"\n✓ Step 2 (Non-Autoregressive Direct Candidate Readout):")
print(f"  - Generated Tokens: 0 (Zero Autoregressive Token Generation)")
print(f"  - Candidate Logits: {cand_logits}")
print(f"  - Candidate Probabilities: {prob_dict}")
print(f"  - Best Decision: Option {best_cand} -> {action_map[best_cand]} (Confidence: {prob_dict[best_cand] * 100:.1f}%)")
print(f"  - Scoring Latency (All 4 Candidates): {scoring_ms:.2f} ms")

# 5. Peak GPU Telemetry
peak_vram = torch.cuda.max_memory_allocated() / (1024 * 1024)
print(f"\n=================================================================")
print(f"               REAL FORWARD INFERENCE SUMMARY                    ")
print(f"=================================================================")
print(f"  Total Inference Latency:   {prefill_ms + scoring_ms:.2f} ms")
print(f"  Prefill Latency:           {prefill_ms:.2f} ms")
print(f"  Direct Scoring Latency:    {scoring_ms:.2f} ms")
print(f"  Decisions Per Second:      {1000.0 / (prefill_ms + scoring_ms):.1f} decisions/sec")
print(f"  Peak VRAM Allocated:       {peak_vram:.1f} MB / 81,920 MB ({peak_vram / 819.2:.2f}%)")
print(f"  Autonomous Verdict:        {action_map[best_cand]}")
print(f"=================================================================")
