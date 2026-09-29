import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.datasets.converters.banking77_converter import convert as banking
from benchmarks.datasets.converters.code_converter import (
    AppsConfig, convert_apps, convert_humaneval, mutants, normalize_output, run_program)
from benchmarks.datasets.converters.common import KEYS, isolated, shuffled

FAST = AppsConfig(solution_timeout=5.0, mutant_timeout=2.0)
UPPER = "s = input()\nprint(s.upper())\n"
ADD = "a, b = map(int, input().split())\nif a < b:\n    print(a + b)\nelse:\n    print(a - b)\n"


def evaluation(tmp_path, contexts=()):
    path = tmp_path / "evaluation.jsonl"
    path.write_text("".join(json.dumps({"context": text}) + "\n" for text in contexts), encoding="utf-8")
    return path


def valid(rows):
    for row in rows:
        assert set(row) == KEYS
        assert all(isinstance(row[key], str) and row[key] for key in ("id", "task", "context", "ground_truth"))
        assert len(row["candidates"]) >= 2
        assert len(row["candidates"]) == len(set(row["candidates"]))
        assert all(candidate.strip() for candidate in row["candidates"])
        assert row["ground_truth"] in row["candidates"]
        assert isinstance(row["metadata"], dict)


def apps(question="Print the input in upper case.", inputs=("one", "two"), outputs=("ONE", "TWO"),
         solutions=(UPPER,), **extra):
    return {"id": extra.pop("id", 7), "question": question, "difficulty": "introductory",
            "input_output": json.dumps({"inputs": list(inputs), "outputs": list(outputs)}),
            "solutions": json.dumps(list(solutions)), **extra}


def convert(sources, tmp_path, contexts=(), config=FAST):
    return convert_apps(sources, split="train", evaluation_path=evaluation(tmp_path, contexts), config=config)


def test_banking77_77_way_labels_and_isolation(tmp_path):
    labels = [f"intent_{i}" for i in range(77)]
    rows = [{"text": "Card declined", "label": 4},
            {"text": "CARD   DECLINED", "label": 4},
            {"text": "transfer delayed", "label": 8}]
    converted = banking(rows, labels, split="train", evaluation_path=evaluation(tmp_path, ["  card declined "]))
    valid(converted)
    assert len(converted) == 1
    assert converted[0]["ground_truth"] == "intent_8"
    assert converted[0]["candidates"] == labels
    assert converted[0]["metadata"]["split"] == "train"


def test_banking77_rejects_test_and_invalid_labels(tmp_path):
    path = evaluation(tmp_path)
    labels = [str(i) for i in range(77)]
    with pytest.raises(ValueError, match="train"):
        banking([], labels, split="test", evaluation_path=path)
    with pytest.raises(ValueError, match="77"):
        banking([], labels[:-1], split="train", evaluation_path=path)
    with pytest.raises(ValueError, match="invalid"):
        banking([{"text": "hello", "label": 77}], labels, split="train", evaluation_path=path)


def test_apps_context_contains_executed_solution_and_stdin(tmp_path):
    rows, stats = convert([apps()], tmp_path)
    valid(rows)
    assert [row["ground_truth"] for row in rows] == ["ONE", "TWO"]
    for row in rows:
        assert UPPER.strip() in row["context"]
        assert row["task"] == "apps_execution_prediction"
        assert row["metadata"]["solution_index"] == 0
    assert "\none\n" in rows[0]["context"] and "\ntwo\n" in rows[1]["context"]
    assert stats["problem.emitted"] == 1 and stats["case.emitted"] == 2 and stats["isolation.kept"] == 2


def test_apps_ground_truth_comes_from_a_verified_solution(tmp_path):
    # Solution 0 prints the wrong thing, solution 1 is Python 2, solution 2 is right.
    wrong, python2 = "print(input())\n", "print input().upper()\n"
    rows, stats = convert([apps(solutions=(wrong, python2, UPPER))], tmp_path)
    assert len(rows) == 2 and all(row["metadata"]["solution_index"] == 2 for row in rows)
    assert all(wrong.strip() not in row["context"] for row in rows)
    assert stats["solution.unparsable"] == 1 and stats["solution_run.mismatch"] == 2
    # A dataset label that no solution reproduces is dropped, not trusted.
    rows, stats = convert([apps(inputs=("2 5", "5 2", "1 1"), outputs=("7", "WRONG", "0"), solutions=(ADD,))], tmp_path)
    assert sorted(row["ground_truth"] for row in rows) == ["0", "7"]
    assert all("WRONG" not in row["candidates"] for row in rows)
    assert stats["case.solution_mismatch"] == 1


def test_apps_single_output_problems_get_mutant_distractors(tmp_path):
    rows, stats = convert([apps(question="Add or subtract.", inputs=("2 5",), outputs=("7",), solutions=(ADD,))], tmp_path)
    valid(rows)
    assert len(rows) == 1
    assert rows[0]["ground_truth"] == "7"
    assert rows[0]["metadata"]["distractors"].get("mutant", 0) == len(rows[0]["candidates"]) - 1
    variants = mutants(ADD, "7", 6)
    assert variants and all(variant != ADD for variant in variants)
    for candidate in rows[0]["candidates"]:
        if candidate != "7":
            # Every distractor is the real output of some mutant on the same stdin.
            assert any(normalize_output(run_program(v, "2 5", timeout=5)[1]) == candidate for v in variants)


def test_apps_malformed_real_data_is_counted_not_crashing(tmp_path):
    sources = [
        apps(id=1, input_output=""),                 # APPS train has 195 of these
        apps(id=2, input_output="{not json"),
        apps(id=3, input_output=json.dumps({"fn_name": "f", "inputs": [[1]], "outputs": [[2]]})),
        apps(id=4, question="  "),
        apps(id=5, solutions=""),
        apps(id=6, solutions=["print input()"]),
        apps(id=8, solutions=("while True:\n    pass\n",)),
    ]
    rows, stats = convert(sources, tmp_path, config=AppsConfig(solution_timeout=1.0))
    assert rows == []
    assert stats["problem.total"] == 7
    for reason in ("io_empty", "io_invalid_json", "fn_call_unsupported", "no_question",
                   "solutions_empty", "no_parsable_solution", "no_verified_solution"):
        assert stats[f"problem.{reason}"] == 1, reason
    assert stats["solution_run.timeout"] == 2
    assert sum(v for k, v in stats.items() if k.startswith("problem.") and k != "problem.total") == 7


def test_apps_blank_stdout_is_filtered_and_list_io_is_lines(tmp_path):
    source = apps(question="Sum per line.", inputs=(["2", "3"], "4\n5\n", "1\n1\n"), outputs=(["5"], "9\n", "   \n"),
                  solutions=("print(int(input()) + int(input()))\n",))
    rows, stats = convert([source], tmp_path)
    valid(rows)
    assert sorted(row["ground_truth"] for row in rows) == ["5", "9"]
    assert stats["case.empty_stdout"] == 1
    assert normalize_output(["a  ", "b", "", ""]) == "a\nb"


def test_apps_huge_integer_literal_is_supported(tmp_path):
    digits = "7" * 9131  # over Python's default 4300-digit limit; APPS train row 3028 has one
    # Real shape: a function-call problem whose JSON holds a 9131-digit integer literal.
    literal = '{"fn_name": "factorial", "inputs": [[1], [3000]], "outputs": [[1], [%s]]}' % digits
    rows, stats = convert([apps(input_output=literal)], tmp_path)
    assert rows == [] and stats["problem.fn_call_unsupported"] == 1
    # A stdin program that converts the huge number must also run in the child interpreter.
    source = apps(question="Echo a number.", inputs=(digits + "\n", "12\n"), outputs=(digits + "\n", "12\n"),
                  solutions=("print(int(input()))\n",))
    rows, stats = convert([source], tmp_path, config=AppsConfig(max_stdin_chars=10000, max_stdout_chars=10000))
    assert {row["ground_truth"] for row in rows} == {digits, "12"}
    assert stats["case.emitted"] == 2


def test_apps_code_dedup_is_case_sensitive_but_eval_exclusion_is_strict(tmp_path):
    lower = apps(id=1, question="Echo.", inputs=("a", "b"), outputs=("a", "b"), solutions=("print(input())\n",))
    upper = apps(id=2, question="Echo.", inputs=("A", "B"), outputs=("A", "B"), solutions=("print(input())\n",))
    rows, stats = convert([lower, upper], tmp_path)
    assert len(rows) == 4 and stats["isolation.duplicate_context"] == 0
    rows, stats = convert([lower, lower], tmp_path)
    assert stats["isolation.duplicate_context"] == 2
    first = convert([lower], tmp_path)[0][0]["context"]
    rows, stats = convert([lower], tmp_path, contexts=[first.upper()])
    assert len(rows) == 1 and stats["isolation.excluded_eval_overlap"] == 1


def test_candidate_order_is_deterministic_and_unbiased():
    candidates = ["truth", "b", "c", "d"]
    positions = {shuffled(candidates, f"id-{i}").index("truth") for i in range(40)}
    assert positions == {0, 1, 2, 3}
    assert shuffled(candidates, "id-3") == shuffled(candidates, "id-3")
    assert sorted(shuffled(candidates, "id-3")) == sorted(candidates)


def test_candidate_order_does_not_encode_which_option_is_the_answer():
    # The converter passes [answer, *distractors]. If the order depended on that input
    # order, anyone who knows the id could undo it and read off the answer.
    for i in range(40):
        orders = {tuple(shuffled([truth, *(c for c in "abcd" if c != truth)], f"id-{i}")) for truth in "abcd"}
        assert len(orders) == 1


def flat(text):
    return " ".join(text.casefold().split())


def no_option_in_question(rows, question):
    for row in rows:
        assert flat(row["ground_truth"]) not in flat(question) or len(flat(row["ground_truth"])) < 2
        assert all(len(flat(c)) < 2 or flat(c) not in flat(question) for c in row["candidates"])


def test_apps_statement_samples_are_dropped_and_counted(tmp_path):
    question = "Print the line in upper case.\n\n-----Example-----\nInput\none\n\nOutput\nONE\n"
    source = apps(question=question, inputs=("one", "two", "three"), outputs=("ONE", "TWO", "THREE"))
    rows, stats = convert([source], tmp_path)
    valid(rows)
    assert sorted(row["ground_truth"] for row in rows) == ["THREE", "TWO"]
    assert all(row["metadata"]["case_index"] != 0 for row in rows)
    assert stats["case.in_question_example"] == 1 and stats["case.emitted"] == 2
    no_option_in_question(rows, question)


def test_apps_stdin_alone_or_stdout_alone_in_question_is_a_sample(tmp_path):
    # Only the input is quoted for case 0, only the output for case 1; case 2 is hidden.
    question = "Upper-case it. For example the word two becomes something; ALPHA is another output."
    source = apps(question=question, inputs=("two", "alpha", "gamma", "delta"), outputs=("TWO", "ALPHA", "GAMMA", "DELTA"))
    rows, stats = convert([source], tmp_path)
    valid(rows)
    assert sorted(row["ground_truth"] for row in rows) == ["DELTA", "GAMMA"]
    assert stats["case.in_question_example"] == 2
    no_option_in_question(rows, question)


def test_apps_sample_match_ignores_case_and_whitespace(tmp_path):
    # The statement lays the sample out on two lines with a tab; stdin is "2 5" on one line.
    question = "Add or subtract.\nSample input:\n2\t 5\n"
    source = apps(question=question, inputs=("2 5", "1 8", "4 6"), outputs=("7", "9", "10"), solutions=(ADD,))
    rows, stats = convert([source], tmp_path)
    assert sorted(row["ground_truth"] for row in rows) == ["10", "9"]
    assert stats["case.in_question_example"] == 1
    # Stdin "one" is caught through the upper-case "ONE" in the statement.
    source = apps(question="Example: Input ONE", inputs=("one", "two", "six"), outputs=("1", "2", "6"),
                  solutions=("print({'one': 1, 'two': 2, 'six': 6}[input()])\n",))
    rows, stats = convert([source], tmp_path)
    assert sorted(row["ground_truth"] for row in rows) == ["2", "6"] and stats["case.in_question_example"] == 1


def test_apps_single_character_io_is_not_a_sample(tmp_path):
    # "a" and "b" occur in any English statement; the length threshold keeps these hidden cases.
    source = apps(question="Echo a line back, a or b.", inputs=("a", "b"), outputs=("a", "b"),
                  solutions=("print(input())\n",))
    rows, stats = convert([source], tmp_path)
    assert sorted(row["ground_truth"] for row in rows) == ["a", "b"]
    assert stats["case.in_question_example"] == 0


def test_apps_problem_with_only_sample_cases_is_counted(tmp_path):
    question = "Upper-case it.\nInput: one\nOutput: ONE\nInput: two\nOutput: TWO"
    rows, stats = convert([apps(question=question)], tmp_path)
    assert rows == []
    assert stats["case.in_question_example"] == 2 and stats["problem.no_usable_case"] == 1


def same_quote_status(rows, question):
    for row in rows:
        assert len({flat(c) in flat(question) for c in row["candidates"]}) == 1, row["candidates"]


def test_apps_distractor_quoted_in_question_is_removed(tmp_path):
    # The a + b -> a - b mutant prints -3 on "2 5"; since the answer is not in the statement,
    # an option that is could be ruled out by search, so it must not be offered.
    question = "Add or subtract. A common wrong answer is -3."
    source = apps(question=question, inputs=("2 5", "1 8", "4 6"), outputs=("7", "9", "10"), solutions=(ADD,))
    rows, stats = convert([source], tmp_path)
    valid(rows)
    assert len(rows) == 3
    assert stats["distractor.question_mismatch"] == 1
    assert all("-3" not in row["candidates"] for row in rows)
    no_option_in_question(rows, question)
    same_quote_status(rows, question)


def test_apps_short_answer_quoted_in_question_gets_only_quoted_distractors(tmp_path):
    # "7" is under the sample threshold, so the "2 5" case stays, but the statement quotes it.
    # Its wrong options 9, 10 and -3 are not quoted, so "pick the quoted option" would win:
    # they are removed and the case has no distractor left. The other answers are not quoted,
    # so the quoted "7" is removed from their options.
    question = "Add or subtract; one result is 7."
    source = apps(question=question, inputs=("2 5", "1 8", "4 6"), outputs=("7", "9", "10"), solutions=(ADD,))
    rows, stats = convert([source], tmp_path)
    valid(rows)
    assert sorted(row["ground_truth"] for row in rows) == ["10", "9"]
    assert all("7" not in row["candidates"] for row in rows)
    assert stats["case.no_distractor"] == 1 and stats["case.in_question_example"] == 0
    same_quote_status(rows, question)
    # With quoted wrong options available, a quoted short answer is kept.
    source = apps(question="Echo a line back, a or b.", inputs=("a", "b"), outputs=("a", "b"),
                  solutions=("print(input())\n",))
    rows, _ = convert([source], tmp_path)
    assert len(rows) == 2
    same_quote_status(rows, "Echo a line back, a or b.")


def test_apps_ground_truth_position_varies(tmp_path):
    source = apps(question="Add or subtract.", inputs=[f"{i} {i + 3}" for i in range(10)],
                  outputs=[str(2 * i + 3) for i in range(10)], solutions=(ADD,))
    rows, _ = convert([source], tmp_path)
    valid(rows)
    assert len({row["candidates"].index(row["ground_truth"]) for row in rows}) > 1


def test_apps_prefers_distractors_with_the_answer_shape(tmp_path):
    # Output line count equals the first stdin number; other cases differ in shape.
    code = "n = int(input())\nfor i in range(n):\n    print(i * 2)\n"
    inputs = ["1", "2", "3", "4", "5"]
    outputs = ["\n".join(str(i * 2) for i in range(int(n))) for n in inputs]
    rows, _ = convert([apps(question="Print evens.", inputs=inputs, outputs=outputs, solutions=(code,))], tmp_path)
    valid(rows)
    shaped = [row for row in rows if "\n" in row["ground_truth"]]  # mutants like i * 3 keep the shape here
    assert len(shaped) == 4
    for row in shaped:
        same = [c for c in row["candidates"] if c.count("\n") == row["ground_truth"].count("\n")]
        assert len(same) >= 3, row["candidates"]  # at least two same-shape mutant outputs exist for n >= 2


def test_isolated_counts_what_it_drops(tmp_path):
    row = {"id": "x", "task": "t", "context": "Print(A)", "candidates": ["1", "2"], "ground_truth": "1", "metadata": {}}
    other = dict(row, id="y", context="print(a)")
    from collections import Counter
    stats = Counter()
    assert len(list(isolated([row, other], evaluation(tmp_path), exact=True, stats=stats))) == 2
    stats = Counter()
    assert len(list(isolated([row, other], evaluation(tmp_path), stats=stats))) == 1
    assert stats == {"kept": 1, "duplicate_context": 1}


def test_apps_rejects_test_and_mismatched_cases(tmp_path):
    path = evaluation(tmp_path)
    with pytest.raises(ValueError, match="train"):
        convert_apps([], split="test", evaluation_path=path)
    with pytest.raises(ValueError, match="invalid"):
        convert_apps([{"question": "q", "input_output": {"inputs": ["x"], "outputs": []}}], split="train", evaluation_path=path)


def test_humaneval_refuses_evaluation_contamination(tmp_path):
    with pytest.raises(ValueError, match="no official train"):
        convert_humaneval([], split="test", evaluation_path=evaluation(tmp_path))
