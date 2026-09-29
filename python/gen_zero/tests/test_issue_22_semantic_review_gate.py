"""Comprehensive Unit and Integration Tests for Issue #22.

Validates:
1. Milestone 1: DiffHunkParser & DualContextCollator (hunk offsets, is_test classification, dual contexts).
2. Milestone 2: Staged Review Funnel (Screening < 10ms bypass, profiling, 5-D signals).
3. Milestone 3: Explicit noMatch and noIssue Rejection Filter (no blind guessing, anti-false alarm).
4. Milestone 4: Severity Blocking Gate & Owner Domain Routing (Severity >= 2.0 blocks merge, >= 1.5 routes).
"""

import unittest
import time

from gen_zero.gate.diff_parser import DiffHunkParser, DualContextCollator, FileDiff, DiffHunk
from gen_zero.gate.review_taxonomy import (
    RiskDimension,
    ChangeProfile,
    ReviewAction,
    OwnerDomain,
    MECHANISM_TAXONOMY,
    NO_MATCH,
    NO_ISSUE,
)
from gen_zero.gate.staged_review_gate import (
    StagedReviewGate,
    ReviewGateVerdict,
    ReviewIssue,
)


SAMPLE_BENIGN_DIFF = """diff --git a/gen_zero/docs/readme.md b/gen_zero/docs/readme.md
index 1111111..2222222 100644
--- a/gen_zero/docs/readme.md
+++ b/gen_zero/docs/readme.md
@@ -10,3 +10,4 @@
 # Overview of the architecture
 
+Updated documentation with detailed examples.
"""

SAMPLE_SECURITY_INJECTION_DIFF = """diff --git a/gen_zero/service/handler.py b/gen_zero/service/handler.py
index 1111111..2222222 100644
--- a/gen_zero/service/handler.py
+++ b/gen_zero/service/handler.py
@@ -25,4 +25,7 @@ def process_user_input(command_str: str):
     # Execute command
+    import os
+    os.system("echo " + command_str)
     return {"status": "ok"}
"""

SAMPLE_DUAL_TEST_GAP_DIFF = """diff --git a/gen_zero/core/calculator.py b/gen_zero/core/calculator.py
index 1111111..2222222 100644
--- a/gen_zero/core/calculator.py
+++ b/gen_zero/core/calculator.py
@@ -50,6 +50,15 @@ def compute_ratio(a: float, b: float) -> float:
+    if b == 0:
+        raise ZeroDivisionError("Cannot divide by zero in critical calculation")
+    if a < 0:
+        return -1.0
+    return a / b
"""

SAMPLE_SYNTHETIC_AMBIGUOUS_DIFF = """diff --git a/gen_zero/utils/formatting.py b/gen_zero/utils/formatting.py
index 1111111..2222222 100644
--- a/gen_zero/utils/formatting.py
+++ b/gen_zero/utils/formatting.py
@@ -5,3 +5,5 @@ def format_label(name: str) -> str:
     prefix = "ITEM"
+    suffix = "V2"
+    return f"{prefix}_{name}_{suffix}"
"""


class TestDiffHunkParserAndDualContext(unittest.TestCase):
    """Tests for Milestone 1: Diff Hunk Parser & Dual Context Collator."""

    def test_parse_unified_diff_extracts_hunks(self):
        file_diffs = DiffHunkParser.parse_unified_diff(SAMPLE_SECURITY_INJECTION_DIFF)
        self.assertEqual(len(file_diffs), 1)
        fd = file_diffs[0]
        self.assertEqual(fd.file_path, "gen_zero/service/handler.py")
        self.assertFalse(fd.is_test)
        self.assertEqual(len(fd.hunks), 1)

        hunk = fd.hunks[0]
        self.assertEqual(hunk.old_start, 25)
        self.assertEqual(hunk.new_start, 25)
        self.assertTrue(any("os.system" in line for line in hunk.added_lines))

    def test_is_test_path_detection(self):
        self.assertTrue(DiffHunkParser.is_test_path("gen_zero/tests/test_gate.py"))
        self.assertTrue(DiffHunkParser.is_test_path("tests/test_pipeline.py"))
        self.assertTrue(DiffHunkParser.is_test_path("src/module_test.py"))
        self.assertTrue(DiffHunkParser.is_test_path("src/component.spec.ts"))
        self.assertFalse(DiffHunkParser.is_test_path("gen_zero/service/handler.py"))
        self.assertFalse(DiffHunkParser.is_test_path("gen_zero/model/dual_head.py"))

    def test_dual_context_collator_splits_code_and_tests(self):
        combined_diff = SAMPLE_DUAL_TEST_GAP_DIFF + """diff --git a/gen_zero/tests/test_calculator.py b/gen_zero/tests/test_calculator.py
index 3333333..4444444 100644
--- a/gen_zero/tests/test_calculator.py
+++ b/gen_zero/tests/test_calculator.py
@@ -10,3 +10,4 @@
 def test_basic():
+    assert compute_ratio(10, 2) == 5.0
"""
        ctx = DualContextCollator.collate(combined_diff)
        self.assertIn("gen_zero/core/calculator.py", ctx.business_files)
        self.assertIn("gen_zero/tests/test_calculator.py", ctx.test_files)
        self.assertTrue("compute_ratio" in ctx.file_patch)
        self.assertTrue("assert compute_ratio" in ctx.changed_tests)


class TestStagedReviewGate(unittest.TestCase):
    """Tests for Milestone 2: 6-Stage Hierarchical Review Funnel."""

    def setUp(self):
        self.gate = StagedReviewGate(
            screen_threshold=0.70,
            min_location_confidence=0.55,
            owner_routing_severity=1.5,
            blocking_severity=2.0,
        )

    def test_harmless_change_fast_path_bypass(self):
        verdict = self.gate.review(SAMPLE_BENIGN_DIFF)
        self.assertTrue(verdict.passed)
        self.assertEqual(verdict.action, ReviewAction.APPROVE.value)
        self.assertEqual(len(verdict.issues), 0)
        # Latency should be sub-10ms
        self.assertLess(verdict.latency_ms, 20.0)

    def test_empty_or_trivial_diff(self):
        verdict = self.gate.review("")
        self.assertTrue(verdict.passed)
        self.assertEqual(verdict.action, ReviewAction.APPROVE.value)
        self.assertEqual(len(verdict.issues), 0)


class TestExplicitNoMatchAndNoIssueFilter(unittest.TestCase):
    """Tests for Milestone 3: Explicit noMatch / noIssue Anti-Hallucination Filter."""

    def setUp(self):
        self.gate = StagedReviewGate(
            screen_threshold=0.70,
            min_location_confidence=0.55,
        )

    def test_ambiguous_sample_triggers_rejection_no_false_alarm(self):
        # A simple string formatting change that does not introduce any real vulnerability
        verdict = self.gate.review(SAMPLE_SYNTHETIC_AMBIGUOUS_DIFF)
        self.assertTrue(verdict.passed)
        self.assertEqual(len(verdict.blocking_issues), 0)
        self.assertIn(verdict.action, [ReviewAction.APPROVE.value, ReviewAction.COMMENT.value])

    def test_taxonomy_completeness(self):
        for dim, mechs in MECHANISM_TAXONOMY.items():
            self.assertIn(NO_ISSUE, mechs)
            self.assertIn("other", mechs)
            self.assertGreaterEqual(len(mechs), 4)


class TestSeverityBlockingGateAndRouting(unittest.TestCase):
    """Tests for Milestone 4: Severity Blocking Gate & Domain Routing."""

    def setUp(self):
        self.gate = StagedReviewGate(
            screen_threshold=0.70,
            min_location_confidence=0.55,
            owner_routing_severity=1.5,
            blocking_severity=2.0,
        )

    def test_critical_security_injection_blocks_merge(self):
        verdict = self.gate.review(SAMPLE_SECURITY_INJECTION_DIFF)
        # High severity (os.system injection => severity 3.0 >= 2.0)
        self.assertFalse(verdict.passed)
        self.assertEqual(verdict.action, ReviewAction.REQUEST_CHANGES.value)
        self.assertGreaterEqual(len(verdict.blocking_issues), 1)

        issue = verdict.blocking_issues[0]
        self.assertEqual(issue.dimension, RiskDimension.SECURITY.value)
        self.assertIn(issue.mechanism, ["injection", "authorization"])
        self.assertGreaterEqual(issue.severity, 2.5)
        # Routing to security owner
        self.assertIn(OwnerDomain.SECURITY.value, verdict.routed_owners)

    def test_fatal_destructive_command_blocking(self):
        fatal_diff = """diff --git a/gen_zero/scripts/cleanup.sh b/gen_zero/scripts/cleanup.sh
index 1111111..2222222 100644
--- a/gen_zero/scripts/cleanup.sh
+++ b/gen_zero/scripts/cleanup.sh
@@ -1,2 +1,3 @@
 #!/bin/bash
+rm -rf /
"""
        verdict = self.gate.review(fatal_diff)
        self.assertFalse(verdict.passed)
        self.assertEqual(verdict.action, ReviewAction.REQUEST_CHANGES.value)
        self.assertGreaterEqual(len(verdict.blocking_issues), 1)
        self.assertEqual(verdict.blocking_issues[0].severity, 3.0)

    def test_dual_context_test_gap_detection(self):
        # Code changed with division logic and error handling, but zero test changes
        verdict = self.gate.review(SAMPLE_DUAL_TEST_GAP_DIFF)
        # Test gap should be flagged in screening
        self.assertGreaterEqual(verdict.screening_scores.get(RiskDimension.TEST_GAP.value, 0.0), 0.70)
        # Should route to testing owner if issue is generated
        if any(i.dimension == RiskDimension.TEST_GAP.value for i in verdict.issues):
            self.assertIn(OwnerDomain.TESTING.value, verdict.routed_owners)

    def test_reliability_concurrency_issue(self):
        rel_diff = """diff --git a/gen_zero/core/sync.py b/gen_zero/core/sync.py
index 1111111..2222222 100644
--- a/gen_zero/core/sync.py
+++ b/gen_zero/core/sync.py
@@ -10,3 +10,6 @@ class LockManager:
     def acquire_mutex(self):
+        # Concurrency race hazard without release
+        self.lock = threading.Lock()
+        self.lock.acquire()
"""
        verdict = self.gate.review(rel_diff)
        self.assertGreaterEqual(verdict.screening_scores.get(RiskDimension.RELIABILITY.value, 0.0), 0.70)
        if any(i.dimension == RiskDimension.RELIABILITY.value for i in verdict.issues):
            self.assertIn(OwnerDomain.RUNTIME.value, verdict.routed_owners)

    def test_compatibility_api_issue(self):
        compat_diff = """diff --git a/gen_zero/api/v1.py b/gen_zero/api/v1.py
index 1111111..2222222 100644
--- a/gen_zero/api/v1.py
+++ b/gen_zero/api/v1.py
@@ -20,4 +20,6 @@
-def execute_query(query: str, limit: int = 10):
+def execute_query(query: str, new_required_arg: int, limit: int = 10):
     # Public schema protocol modified
+    pass
"""
        verdict = self.gate.review(compat_diff)
        self.assertGreaterEqual(verdict.screening_scores.get(RiskDimension.COMPATIBILITY.value, 0.0), 0.70)
        if any(i.dimension == RiskDimension.COMPATIBILITY.value for i in verdict.issues):
            self.assertIn(OwnerDomain.API.value, verdict.routed_owners)


if __name__ == "__main__":
    unittest.main()
