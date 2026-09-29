"""Gen-Zero Browser Pretraining & Sharpness Distillation Pipeline.

Ingests real browser DX interaction datasets (e.g. Google Calendar, GitHub, X/Twitter, YouTube),
trains the dual-head model, and syncs high-sharpness candidate scoring weights to the
production QuantizedCandidateScorer.
"""

import os
import sys
import json
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("GenZeroBrowserPretrain")

def run_browser_pretrain(
    dataset_path: str = "data/browser_dx_pretrain_dataset.jsonl",
    epochs: int = 5,
    batch_size: int = 8
) -> bool:
    from gen_zero.client import GenZero
    from gen_zero.train.replay_buffer import StabilityReplayBuffer
    from gen_zero.train.distiller import GenZeroDistiller

    if not os.path.exists(dataset_path):
        logger.error(f"Dataset path does not exist: {dataset_path}")
        return False

    client = GenZero()
    replay_buffer = StabilityReplayBuffer(capacity=10000)
    
    count = replay_buffer.load_browser_dx_dataset(dataset_path)
    logger.info(f"Loaded {count} browser interaction samples from {dataset_path}")
    if count == 0:
        logger.error("No valid samples loaded.")
        return False

    distiller = GenZeroDistiller(
        model=client.model,
        replay_buffer=replay_buffer,
        lr=5e-4,
        value_weight=0.5
    )

    logger.info(f"Starting browser pretraining distillation ({epochs} epochs)...")
    for epoch in range(epochs):
        total_loss = 0.0
        steps = max(1, count // batch_size)
        for _ in range(steps):
            batch = replay_buffer.sample_batch(batch_size=batch_size)
            res = distiller.train_step(batch)
            total_loss += res.get("loss", 0.0)
        avg_loss = total_loss / steps
        logger.info(f"Epoch {epoch + 1}/{epochs} - Avg Loss: {avg_loss:.4f}")

    # Synchronize trained model to CPU extreme scorer
    client.sync_model_to_scorer()
    logger.info("Synchronized trained model weights to QuantizedCandidateScorer.")

    # Validation on all samples in dataset
    logger.info("Validating decision accuracy and probability sharpness...")
    correct = 0
    total = 0
    sharp_count = 0

    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            gt_ref = item["ground_truth_ref"]
            criteria = item["questions"]["target_element"]["criteria"]
            cands = list(criteria.keys())
            state = item["state"]

            # Route through reflex mode
            res = client.decide(
                state=state,
                candidates=cands,
                mode="reflex",
                candidate_descriptions=criteria
            )
            chosen = res["action"]
            prob = res["probs"].get(gt_ref, 0.0)

            total += 1
            if chosen == gt_ref:
                correct += 1
            if prob >= 0.85:
                sharp_count += 1

    acc = correct / max(1, total) * 100.0
    sharp_pct = sharp_count / max(1, total) * 100.0
    logger.info(f"Validation Results: {correct}/{total} Correct ({acc:.1f}%), {sharp_count}/{total} Sharp >=85% ({sharp_pct:.1f}%)")

    return acc >= 95.0


if __name__ == "__main__":
    success = run_browser_pretrain()
    sys.exit(0 if success else 1)
