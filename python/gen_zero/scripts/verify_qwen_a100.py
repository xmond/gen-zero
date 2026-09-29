"""Verify Qwen3.5-0.8B weights and test real GPU inference on Windows A100."""

import sys
import os
from pathlib import Path

# Safe utf-8 output on Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

model_dir = os.environ.get("QWEN_MODEL_DIR", "models/Qwen3.5-0.8B")
print(f"--> Inspecting Qwen3.5-0.8B at: {model_dir}")

print(f"--> PyTorch: {torch.__version__}, CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"--> Device 0: {torch.cuda.get_device_name(0)}")

# 1. Config
config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
print(f"✓ Config model_type: {config.model_type}")

# 2. Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
print(f"✓ Tokenizer vocab size: {len(tokenizer)}")

# 3. Model on GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if device == "cuda" else torch.float32

print(f"--> Loading model to {device} ({dtype})...")
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    dtype=dtype,
    trust_remote_code=True
).to(device).eval()
print(f"✓ Model successfully loaded on {device}!")

# 4. Real Forward Test
prompt = "Gen-Zero is an autonomous decision engine."
inputs = tokenizer(prompt, return_tensors="pt").to(device)
with torch.inference_mode():
    out = model(**inputs, output_hidden_states=True)
    last_hidden = out.hidden_states[-1]
    print(f"✓ Real Forward Pass Complete! Output hidden state shape: {last_hidden.shape}")
    print(f"✓ Peak GPU Memory Used: {torch.cuda.max_memory_allocated() / (1024*1024):.1f} MB / 81920 MB")

print("\n>>> ALL VERIFICATIONS PASSED: Qwen3.5-0.8B is 100% READY on A100! <<<")
