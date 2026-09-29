"""Comprehensive Test Suite for Issue #9:
Batch States API, Boolean Algebra Engine, Zero-Dependency gen-grep CLI, and Log Filter Benchmark.
"""

import io
import json
import math
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

from gen_zero.client import GenZero
from gen_zero.evaluate.log_filter_benchmark import (
    LogBenchmarkItem,
    TwoStageFilterMetrics,
    TwoStageLogFilterEvaluator,
    create_multilingual_log_corpus,
)
from gen_zero.logic.boolean_engine import (
    AndNode,
    BooleanEngine,
    BooleanNode,
    BooleanSemantics,
    LiteralNode,
    NotNode,
    OrNode,
    p_and,
    p_not,
    p_or,
)
from gen_zero.scripts.gen_grep import (
    build_arg_parser,
    parse_cli_args,
    build_composite_expression,
    main as gen_grep_main,
    parse_expression,
    query_batch_decisions,
    resolve_threshold,
)
from gen_zero.service import app as service_app
from gen_zero.service.app import (
    DecisionsRequest,
    QuestionSpec,
    handle_decisions,
)


class TestMilestone1BatchStatesAPI(unittest.TestCase):
    """Milestone 1: Batch States Decision Inference Protocol."""

    def setUp(self):
        self.client = GenZero()
        # Wire-protocol tests: open the fail-closed checkpoint gate explicitly (503 otherwise).
        gate = patch.object(service_app.client, "weights_loaded_from_checkpoint", True)
        gate.start()
        self.addCleanup(gate.stop)

    def test_01_client_decide_batch_empty_and_single(self):
        # Empty states
        empty_res = self.client.decide_batch([], ["a", "b"])
        self.assertEqual(empty_res, [])

        # Empty candidates
        empty_cands = self.client.decide_batch(["state1", "state2"], [])
        self.assertEqual(len(empty_cands), 2)
        self.assertIsNone(empty_cands[0]["action"])

        # Single state
        single_res = self.client.decide_batch(["User wants to checkout"], ["pay", "cancel"], mode="reflex")
        self.assertEqual(len(single_res), 1)
        self.assertIn(single_res[0]["action"], ["pay", "cancel"])
        self.assertIn("probs", single_res[0])

    def test_02_client_decide_batch_multi_states(self):
        states = [
            "Log line: Error 500 database deadlock on orders table",
            "Log line: HTTP 200 OK service alive and well",
            "Log line: Connection timeout after 3000ms",
            "Log line: Disk space exhausted on /dev/sda1"
        ]
        candidates = ["error", "healthy"]
        results = self.client.decide_batch(states, candidates, mode="reflex")

        self.assertEqual(len(results), 4)
        for i, res in enumerate(results):
            self.assertIn(res["action"], ["error", "healthy", "ABSTAIN"])
            self.assertIn("confidence", res)
            self.assertIn("probs", res)
            self.assertGreaterEqual(res["probs"]["error"], 0.0)
            self.assertGreaterEqual(res["probs"]["healthy"], 0.0)

    def test_03_app_handle_decisions_single_state_backward_compatibility(self):
        req = DecisionsRequest(
            model="typesafe/zero-1.13",
            state="System status: MySQL query took 8500ms and failed.",
            questions={
                "is_error": QuestionSpec(type="noul", instructions="Is there an error?", criteria={"true": "Error occurred", "false": "Normal"}),
                "category": QuestionSpec(type="choice", instructions="Pick component", criteria=["database", "network", "auth"]),
                "urgency": QuestionSpec(type="score", instructions="Score urgency", criteria=["Low", "Med", "High"])
            }
        )
        res = handle_decisions(req)
        self.assertIn("id", res)
        self.assertIn("answers", res)
        self.assertNotIn("results", res)
        self.assertIn("is_error", res["answers"])
        self.assertEqual(res["answers"]["is_error"]["type"], "noul")
        self.assertIn("category", res["answers"])
        self.assertEqual(res["answers"]["category"]["type"], "choice")
        self.assertIn("urgency", res["answers"])
        self.assertEqual(res["answers"]["urgency"]["type"], "score")

    def test_04_app_handle_decisions_batch_states_protocol(self):
        batch_states = [
            "State A: Database deadlock detected in thread 14",
            "State B: Connection timed out while reading socket",
            "State C: All systems operational, CPU 12%"
        ]
        req = DecisionsRequest(
            model="typesafe/zero-1.13",
            states=batch_states,
            questions={
                "has_deadlock": QuestionSpec(type="noul", instructions="Does this indicate deadlock?", criteria={"true": "Deadlock", "false": "No deadlock"}),
                "severity": QuestionSpec(type="score", instructions="Rate severity", criteria=["Minor", "Major", "Critical"])
            }
        )
        res = handle_decisions(req)
        self.assertIn("id", res)
        self.assertEqual(res["batch_size"], 3)
        self.assertIn("results", res)
        self.assertEqual(len(res["results"]), 3)

        for i, item in enumerate(res["results"]):
            self.assertEqual(item["state_index"], i)
            self.assertIn("has_deadlock", item["answers"])
            self.assertIn("severity", item["answers"])
            self.assertGreaterEqual(item["answers"]["has_deadlock"]["noul"], 0.0)
            self.assertLessEqual(item["answers"]["has_deadlock"]["noul"], 1.0)
            self.assertGreaterEqual(item["answers"]["severity"]["score"], 0.0)

    def test_05_app_handle_decisions_validation_errors(self):
        from fastapi import HTTPException

        # Missing both state and states
        with self.assertRaises(HTTPException) as ctx:
            handle_decisions(DecisionsRequest(questions={"q": QuestionSpec(type="noul", criteria={})}))
        self.assertEqual(ctx.exception.status_code, 400)

        # Empty states list
        with self.assertRaises(HTTPException) as ctx:
            handle_decisions(DecisionsRequest(states=[], questions={"q": QuestionSpec(type="noul", criteria={})}))
        self.assertEqual(ctx.exception.status_code, 400)

        # Invalid question type
        with self.assertRaises(HTTPException) as ctx:
            handle_decisions(DecisionsRequest(
                state="test",
                questions={"q": QuestionSpec(type="unsupported_type", criteria={})}
            ))
        self.assertEqual(ctx.exception.status_code, 400)


class TestMilestone2BooleanAlgebraEngine(unittest.TestCase):
    """Milestone 2: Probabilistic Boolean Logic Engine."""

    def setUp(self):
        self.engine = BooleanEngine()

    def test_01_probabilistic_operators(self):
        # NOT
        self.assertAlmostEqual(p_not(0.8), 0.2)
        self.assertAlmostEqual(p_not(0.0), 1.0)
        self.assertAlmostEqual(p_not(1.0), 0.0)
        # NaN / Inf sanitization
        self.assertEqual(p_not(float("nan")), 1.0)

        # AND across semantics
        # Zadeh: min(0.7, 0.4) = 0.4
        self.assertAlmostEqual(p_and(0.7, 0.4, BooleanSemantics.ZADEH), 0.4)
        # Product: 0.7 * 0.4 = 0.28
        self.assertAlmostEqual(p_and(0.7, 0.4, BooleanSemantics.PRODUCT), 0.28)
        # Lukasiewicz: max(0, 0.7 + 0.4 - 1.0) = 0.1
        self.assertAlmostEqual(p_and(0.7, 0.4, BooleanSemantics.LUKASIEWICZ), 0.1)

        # OR across semantics
        # Zadeh: max(0.7, 0.4) = 0.7
        self.assertAlmostEqual(p_or(0.7, 0.4, BooleanSemantics.ZADEH), 0.7)
        # Product: 1 - (1-0.7)*(1-0.4) = 1 - 0.18 = 0.82
        self.assertAlmostEqual(p_or(0.7, 0.4, BooleanSemantics.PRODUCT), 0.82)
        # Lukasiewicz: min(1.0, 0.7 + 0.4) = 1.0
        self.assertAlmostEqual(p_or(0.7, 0.4, BooleanSemantics.LUKASIEWICZ), 1.0)

    def test_02_ast_parsing(self):
        # Simple identifier
        ast1 = self.engine.parse("deadlock")
        self.assertIsInstance(ast1, LiteralNode)
        self.assertEqual(ast1.get_variables(), ["deadlock"])

        # NOT expression
        ast2 = self.engine.parse("NOT timeout")
        self.assertIsInstance(ast2, NotNode)
        self.assertEqual(ast2.get_variables(), ["timeout"])

        # AND expression
        ast3 = self.engine.parse("database AND deadlock")
        self.assertIsInstance(ast3, AndNode)
        self.assertEqual(ast3.get_variables(), ["database", "deadlock"])

        # Compound with precedence and parentheses
        ast4 = self.engine.parse("(deadlock OR timeout) AND NOT disk_full")
        self.assertIsInstance(ast4, AndNode)
        vars4 = ast4.get_variables()
        self.assertIn("deadlock", vars4)
        self.assertIn("timeout", vars4)
        self.assertIn("disk_full", vars4)

        # Quoted strings
        ast5 = self.engine.parse('"database error" && !"connection timeout"')
        self.assertIsInstance(ast5, AndNode)
        self.assertEqual(ast5.get_variables(), ["database error", "connection timeout"])

    def test_03_ast_evaluation(self):
        probs = {
            "deadlock": 0.85,
            "timeout": 0.10,
            "disk_full": 0.05
        }
        # Evaluation of: (deadlock OR timeout) AND NOT disk_full
        expr = "(deadlock OR timeout) AND NOT disk_full"
        res_zadeh = self.engine.evaluate(expr, probs, semantics=BooleanSemantics.ZADEH)
        # deadlock OR timeout = max(0.85, 0.10) = 0.85
        # NOT disk_full = 1 - 0.05 = 0.95
        # AND = min(0.85, 0.95) = 0.85
        self.assertAlmostEqual(res_zadeh, 0.85)

        # Missing variable safely yields 0.0
        res_missing = self.engine.evaluate("deadlock AND unknown_var", probs)
        self.assertAlmostEqual(res_missing, 0.0)

    def test_04_batch_evaluation_and_pattern_builder(self):
        batch_probs = [
            {"A": 0.9, "B": 0.1},
            {"A": 0.8, "B": 0.9},
            {"A": 0.2, "B": 0.3}
        ]
        scores = self.engine.evaluate_batch("A AND B", batch_probs, semantics=BooleanSemantics.ZADEH)
        self.assertEqual(len(scores), 3)
        self.assertAlmostEqual(scores[0], 0.1)
        self.assertAlmostEqual(scores[1], 0.8)
        self.assertAlmostEqual(scores[2], 0.2)

        built = BooleanEngine.build_expression_from_patterns(
            or_patterns=["deadlock", "crash"],
            and_patterns=["production"],
            not_patterns=["timeout"]
        )
        self.assertEqual(built, '("deadlock" OR "crash") AND "production" AND NOT "timeout"')

    def test_05_quote_escaping_and_unescaping(self):
        # Escaping in build_expression_from_patterns
        built = BooleanEngine.build_expression_from_patterns(
            or_patterns=['panic "OOM"'],
            and_patterns=['fatal "crash"']
        )
        self.assertIn(r'\"OOM\"', built)

        # Unescaping in tokenize and parse
        ast = self.engine.parse(r'"error \"kernel\" panic"')
        self.assertEqual(ast.get_variables(), ['error "kernel" panic'])


class TestMilestone3GenGrepCLI(unittest.TestCase):
    """Milestone 3: Zero-Dependency gen-grep CLI."""

    def test_01_cli_arg_parsing(self):
        args = parse_cli_args(["-e", "deadlock", "-a", "db", "-v", "timeout", "--level", "strict", "log.txt"])
        self.assertEqual(args.or_patterns, ["deadlock"])
        self.assertEqual(args.and_patterns, ["db"])
        self.assertEqual(args.not_patterns, ["timeout"])
        self.assertEqual(args.level, "strict")
        self.assertEqual(args.files, ["log.txt"])

    def test_02_threshold_resolution(self):
        self.assertEqual(resolve_threshold("loose", None), 0.50)
        self.assertEqual(resolve_threshold("balanced", None), 0.70)
        self.assertEqual(resolve_threshold("strict", None), 0.85)
        self.assertEqual(resolve_threshold("balanced", 0.92), 0.92)

    def test_03_composite_expression_builder(self):
        ast, pats = build_composite_expression(
            positional_pattern="db_error",
            or_patterns=[],
            and_patterns=["critical"],
            not_patterns=["test"],
            explicit_expr=None
        )
        self.assertIn("db_error", pats)
        self.assertIn("critical", pats)
        self.assertIn("test", pats)

        # Explicit expr overrides
        ast2, pats2 = build_composite_expression(
            positional_pattern=None,
            or_patterns=[],
            and_patterns=[],
            not_patterns=[],
            explicit_expr='"high load" OR "out of memory"'
        )
        self.assertEqual(pats2, ["high load", "out of memory"])

    @patch("urllib.request.urlopen")
    def test_04_query_batch_decisions_mocked_http(self, mock_urlopen):
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "results": [
                {"answers": {"pat_0": {"noul": 0.88}, "pat_1": {"noul": 0.12}}},
                {"answers": {"pat_0": {"noul": 0.05}, "pat_1": {"noul": 0.95}}}
            ]
        }).encode("utf-8")
        mock_response.__enter__.return_value = mock_response
        mock_urlopen.return_value = mock_response

        probs, unresolved = query_batch_decisions(
            endpoint="http://127.0.0.1:8999",
            token="sk-zero-mock",
            states=["line 1", "line 2"],
            patterns=["deadlock", "timeout"]
        )
        self.assertEqual(unresolved, [])
        self.assertEqual(len(probs), 2)
        self.assertAlmostEqual(probs[0]["deadlock"], 0.88)
        self.assertAlmostEqual(probs[0]["timeout"], 0.12)
        self.assertAlmostEqual(probs[1]["deadlock"], 0.05)
        self.assertAlmostEqual(probs[1]["timeout"], 0.95)

    @patch("gen_zero.scripts.gen_grep.query_batch_decisions")
    def test_05_cli_main_match_and_formatting(self, mock_query):
        mock_query.return_value = (
            [
                {"deadlock": 0.90},
                {"deadlock": 0.10}
            ],
            [],
        )
        # Simulate stdin piping with 2 lines
        test_stdin = "ERROR Deadlock in table accounts\nINFO Normal request\n"
        with patch("sys.stdin", io.StringIO(test_stdin)), patch("sys.stdout", new=io.StringIO()) as fake_out:
            code = gen_grep_main(["-e", "deadlock", "--level", "balanced", "-n", "--color", "never"])
            self.assertEqual(code, 0)
            output = fake_out.getvalue()
            self.assertIn("1:ERROR Deadlock in table accounts", output)
            self.assertNotIn("2:INFO Normal request", output)

    @patch("gen_zero.scripts.gen_grep.query_batch_decisions")
    def test_06_cli_unresolved_line_surfaced_not_silent_zero(self, mock_query):
        """B0928 blocker 3: a kernel abstain (no 'noul' key) must show up as an explicit
        [UNRESOLVED] match and a stderr warning, never as a silent probability-0.0 non-match --
        including with --color never, the default for piped/non-tty output."""
        mock_query.return_value = (
            [{"deadlock": None}, {"deadlock": 0.05}],
            [(0, "deadlock", "INFEASIBLE_ABSTAIN")],
        )
        test_stdin = "line one has an issue\nline two is fine\n"
        with patch("sys.stdin", io.StringIO(test_stdin)), \
             patch("sys.stdout", new=io.StringIO()) as fake_out, \
             patch("sys.stderr", new=io.StringIO()) as fake_err:
            code = gen_grep_main(["-e", "deadlock", "--color", "never", "-n"])
            self.assertEqual(code, 0)
            out = fake_out.getvalue()
            err = fake_err.getvalue()
            self.assertIn("[UNRESOLVED]", out)
            self.assertIn("1:", out)
            self.assertNotIn("2:", out)
            self.assertIn("UNRESOLVED", err)
            self.assertIn("INFEASIBLE_ABSTAIN", err)


class TestMilestone4LogFilterBenchmark(unittest.TestCase):
    """Milestone 4: Log Filter Benchmark & Suffix Cache Evaluation."""

    def test_01_multilingual_corpus_composition(self):
        corpus = create_multilingual_log_corpus()
        self.assertEqual(len(corpus), 60)
        languages = {item.language for item in corpus}
        self.assertEqual(languages, {"en", "zh", "ja"})

        en_items = [i for i in corpus if i.language == "en"]
        zh_items = [i for i in corpus if i.language == "zh"]
        ja_items = [i for i in corpus if i.language == "ja"]
        self.assertEqual(len(en_items), 20)
        self.assertEqual(len(zh_items), 20)
        self.assertEqual(len(ja_items), 20)

    def test_02_two_stage_log_filter_evaluator(self):
        evaluator = TwoStageLogFilterEvaluator()
        corpus = create_multilingual_log_corpus()

        metrics = evaluator.evaluate(
            corpus=corpus,
            target_criterion="deadlock",
            pattern_query="database deadlock or lock cycle",
            coarse_regex=r"(?i)(deadlock|lock|死锁|デッドロック|InnoDB)",
            batch_size=30,
            threshold=0.50
        )

        self.assertIsInstance(metrics, TwoStageFilterMetrics)
        self.assertEqual(metrics.total_lines, 60)
        self.assertGreater(metrics.stage1_coarse_passed, 0)
        self.assertGreater(metrics.stage1_coarse_pruned, 0)
        self.assertGreater(metrics.throughput_lines_per_sec, 0.0)
        self.assertGreater(metrics.precision, 0.0)
        self.assertGreater(metrics.recall, 0.0)
        self.assertGreater(metrics.f1_score, 0.0)

        report = metrics.to_dict()
        self.assertIn("throughput_lines_per_sec", report)
        self.assertIsNone(metrics.prefix_kv_cache_shared_ratio)
        self.assertNotIn("prefix_kv_cache_shared_ratio", report)

        metrics.prefix_kv_cache_shared_ratio = 0.5
        self.assertEqual(metrics.to_dict()["prefix_kv_cache_shared_ratio"], 0.5)


if __name__ == "__main__":
    unittest.main()
