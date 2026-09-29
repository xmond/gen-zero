"""Zero-leakage isolation check between the calibration split and the frozen test set.

benchmarks/data/calibration_clean_16.jsonl exists to give the Continuous Causal
Reasoning Expert legitimate few-shot calibration signal. It must never overlap,
by ID or by text content, with the 930 samples in benchmarks/data/manifest.json
that are reserved for evaluation. This test asserts that intersection is empty
and locks the calibration file's SHA-256 so silent edits are caught.
"""
import hashlib
import json
import unittest
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
MANIFEST_PATH = DATA_DIR / "manifest.json"
CALIBRATION_PATH = DATA_DIR / "calibration_clean_16.jsonl"

# Locked at creation time by sha256sum(calibration_clean_16.jsonl).
# If this file is regenerated, update this hash deliberately and explain why
# in the commit message -- an unexpected mismatch here means the calibration
# set changed underneath the tests.
EXPECTED_SHA256 = "bd45f4df430ee7c74440afca44b7880ac033e24013a34345ce4e54f1583efa4a"


def _load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _normalize_text(text: str) -> str:
    return " ".join(text.split()).strip().lower()


class CalibrationSplitIsolationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(MANIFEST_PATH, encoding="utf-8") as f:
            cls.manifest = json.load(f)
        cls.calibration = _load_jsonl(CALIBRATION_PATH)

        cls.test_samples = []
        for task_name, task_info in cls.manifest["tasks"].items():
            task_file = DATA_DIR / task_info["file"]
            cls.test_samples.extend(_load_jsonl(task_file))

    def test_manifest_reports_1700_test_samples(self):
        self.assertEqual(self.manifest["total_samples"], 1700)
        self.assertEqual(len(self.test_samples), 1700)

    def test_calibration_set_is_16_to_32_per_task(self):
        counts: dict[str, int] = {}
        for s in self.calibration:
            counts[s["task"]] = counts.get(s["task"], 0) + 1
        self.assertEqual(set(counts), set(self.manifest["tasks"]))
        for task, n in counts.items():
            self.assertGreaterEqual(n, 16, f"{task} has only {n} calibration samples")
            self.assertLessEqual(n, 32, f"{task} has {n} calibration samples, over the cap")

    def test_zero_id_overlap_with_test_manifest(self):
        test_ids = {s["id"] for s in self.test_samples}
        calibration_ids = {s["id"] for s in self.calibration}
        overlap = test_ids & calibration_ids
        self.assertEqual(overlap, set(), f"ID leakage detected: {sorted(overlap)}")

    def test_zero_content_overlap_with_test_manifest(self):
        test_contexts = {_normalize_text(s["context"]) for s in self.test_samples}
        calibration_contexts = {_normalize_text(s["context"]) for s in self.calibration}
        overlap = test_contexts & calibration_contexts
        self.assertEqual(overlap, set(), f"Text-content leakage detected: {len(overlap)} shared contexts")

    def test_zero_content_overlap_per_task(self):
        test_by_task: dict[str, set[str]] = {}
        for s in self.test_samples:
            test_by_task.setdefault(s["task"], set()).add(_normalize_text(s["context"]))

        for s in self.calibration:
            task = s["task"]
            normalized = _normalize_text(s["context"])
            self.assertNotIn(
                normalized,
                test_by_task.get(task, set()),
                f"Calibration sample {s['id']} duplicates a {task} test context",
            )

    def test_calibration_file_matches_all_test_sample_ids_namespace(self):
        # Calibration IDs use a distinct "-cal-" infix so they can never
        # collide with the "{task}-NNNN" IDs used by the test manifest.
        for s in self.calibration:
            self.assertIn("-cal-", s["id"])
            self.assertNotIn(s["id"], {t["id"] for t in self.test_samples})

    def test_calibration_sha256_locked(self):
        digest = hashlib.sha256(CALIBRATION_PATH.read_bytes()).hexdigest()
        self.assertEqual(
            digest,
            EXPECTED_SHA256,
            "calibration_clean_16.jsonl changed on disk; if intentional, "
            "recompute sha256 and update EXPECTED_SHA256 deliberately.",
        )


if __name__ == "__main__":
    unittest.main()
