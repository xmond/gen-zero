import json
from pathlib import Path
import importlib.util
import sys

import pytest

SPEC = importlib.util.spec_from_file_location(
    "clean_evaluation_harness",
    Path(__file__).resolve().parents[1] / "suites" / "clean_evaluation_harness.py",
)
harness = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = harness
SPEC.loader.exec_module(harness)
ContractViolation = harness.ContractViolation
evaluate = harness.evaluate
fit_prototypes = harness.fit_prototypes
load_split = harness.load_split
PrototypeArtifact = harness.PrototypeArtifact


def write_split(path: Path, rows: list[dict]) -> str:
    path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    return __import__("hashlib").sha256(path.read_bytes()).hexdigest()


def test_hash_binding_rejects_tampered_data_and_labels(tmp_path):
    path = tmp_path / "test.json"
    digest = write_split(path, [{"id": "a", "input": "x", "label": "yes"}])
    path.write_text(path.read_text(encoding="utf-8").replace("yes", "no"), encoding="utf-8")
    with pytest.raises(ContractViolation, match="sha256 mismatch"):
        load_split(path, expected_sha256=digest)


def test_calibration_only_and_test_labels_are_not_sent_to_predictor(tmp_path):
    calibration_path = tmp_path / "calibration.json"
    test_path = tmp_path / "test.json"
    calibration = load_split(calibration_path, expected_sha256=write_split(
        calibration_path, [{"id": "c", "input": [0.0], "label": "A"}]))
    test = load_split(test_path, expected_sha256=write_split(
        test_path, [{"id": "t", "input": [1.0], "label": "A"},
                    {"id": "u", "input": [2.0], "label": "B"}]))
    seen = []
    artifact = fit_prototypes(calibration, embed=lambda value: seen.append(value) or value)
    result = evaluate(test, artifact, predict=lambda value: seen.append(value) or "A")
    # Inputs are deep-frozen on load (lists -> tuples), so the predictor sees tuples.
    assert seen == [(0.0,), (1.0,), (2.0,)]
    assert result.total == 2 and result.correct == 1 and result.errors == 1
    assert result.accuracy == pytest.approx(0.5)
    assert result.coverage == 1.0


def test_exception_and_abstention_remain_in_denominator(tmp_path):
    cpath, tpath = tmp_path / "c.json", tmp_path / "t.json"
    calibration = load_split(cpath, expected_sha256=write_split(
        cpath, [{"id": "c", "input": [0], "label": "A"}]))
    test = load_split(tpath, expected_sha256=write_split(
        tpath, [{"id": "1", "input": [1], "label": "A"},
                {"id": "2", "input": [2], "label": "A"},
                {"id": "3", "input": [3], "label": "A"}]))
    artifact = fit_prototypes(calibration, embed=lambda x: x)
    def predictor(value):
        # Inputs are deep-frozen on load (lists -> tuples).
        if value == (2,):
            raise RuntimeError("failed")
        return None if value == (3,) else "A"
    result = evaluate(test, artifact, predict=predictor)
    assert (result.total, result.correct, result.errors) == (3, 1, 2)
    assert result.coverage == pytest.approx(1 / 3)


def test_contaminated_split_rejected_when_test_equals_calibration(tmp_path):
    rows = [{"id": "a", "input": [0.0], "label": "A"},
            {"id": "b", "input": [1.0], "label": "B"}]
    calibration_path = tmp_path / "calibration.json"
    test_path = tmp_path / "test.json"
    # Byte-identical files (and therefore identical sha256): the calibration
    # split leaked verbatim into the test split.
    calibration = load_split(calibration_path, expected_sha256=write_split(calibration_path, rows))
    test = load_split(test_path, expected_sha256=write_split(test_path, rows))
    artifact = fit_prototypes(calibration, embed=lambda x: x)
    with pytest.raises(ContractViolation, match="Data contamination"):
        evaluate(test, artifact, predict=lambda value: "A")


def test_nested_input_and_label_are_deep_frozen(tmp_path):
    path = tmp_path / "nested.json"
    digest = write_split(path, [{"id": "a", "input": {"tokens": [1, 2, 3]}, "label": ["x", "y"]}])
    split = load_split(path, expected_sha256=digest)
    sample = split.samples[0]
    assert isinstance(sample.input, harness.MappingProxyType)
    assert sample.input["tokens"] == (1, 2, 3)
    assert isinstance(sample.label, tuple)
    with pytest.raises(TypeError):
        sample.input["tokens"] = (9, 9, 9)


def test_evaluate_handles_dict_shaped_input_without_deepcopy_crash(tmp_path):
    """copy.deepcopy cannot pickle a MappingProxyType; evaluate() must not use it."""
    calibration_path = tmp_path / "c.json"
    test_path = tmp_path / "t.json"
    calibration = load_split(calibration_path, expected_sha256=write_split(
        calibration_path, [{"id": "c", "input": {"v": [0.0]}, "label": "A"}]))
    test = load_split(test_path, expected_sha256=write_split(
        test_path, [{"id": "t", "input": {"v": [1.0]}, "label": "A"}]))
    artifact = fit_prototypes(calibration, embed=lambda value: value["v"])
    result = evaluate(test, artifact, predict=lambda value: "A")
    assert result.total == 1 and result.correct == 1 and result.errors == 0


def test_reformatted_shuffled_copy_of_calibration_is_contamination(tmp_path):
    """A new sha256 from whitespace, order or ids alone must not hide identical data."""
    rows = [{"id": "a", "input": [0.0], "label": "A"},
            {"id": "b", "input": [1.0], "label": "B"}]
    calibration_path = tmp_path / "calibration.json"
    test_path = tmp_path / "test.json"
    calibration = load_split(calibration_path, expected_sha256=write_split(calibration_path, rows))
    copy_rows = [{"label": "B", "input": [1.0], "id": "x"},
                 {"label": "A", "input": [0.0], "id": "y"}]
    test_path.write_text(json.dumps(copy_rows, indent=2), encoding="utf-8")
    test = load_split(test_path)
    assert test.sha256 != calibration.sha256
    artifact = fit_prototypes(calibration, embed=lambda x: x)
    with pytest.raises(ContractViolation, match="identical content"):
        evaluate(test, artifact, predict=lambda value: "A")


def test_nested_lists_in_labels_and_inputs_cannot_be_mutated(tmp_path):
    path = tmp_path / "nested.json"
    digest = write_split(path, [{"id": "a", "input": {"m": [[1, 2], {"k": [3]}]}, "label": {"y": [1]}}])
    sample = load_split(path, expected_sha256=digest).samples[0]
    assert sample.input["m"] == ((1, 2), harness.MappingProxyType({"k": (3,)}))
    with pytest.raises(AttributeError):
        sample.input["m"][0].append(9)
    with pytest.raises(TypeError):
        sample.input["m"][1]["k"] = (0,)
    with pytest.raises(TypeError):
        sample.label["y"] = (0,)
    with pytest.raises(AttributeError):
        sample.label["y"].append(0)


def test_partial_overlap_with_changed_id_and_label_is_rejected(tmp_path):
    cpath, tpath = tmp_path / "c.json", tmp_path / "t.json"
    write_split(cpath, [{"id": "c", "input": [1], "label": "A"}])
    write_split(tpath, [{"id": "renamed", "input": [1], "label": "B"},
                        {"id": "new", "input": [2], "label": "A"}])
    artifact = fit_prototypes(load_split(cpath), embed=lambda x: x)
    with pytest.raises(ContractViolation, match="overlap"):
        evaluate(load_split(tpath), artifact, predict=lambda x: "A")


def test_duplicate_ids_rejected(tmp_path):
    path = tmp_path / "test.json"
    write_split(path, [{"id": "same", "input": [1], "label": "A"},
                       {"id": "same", "input": [2], "label": "B"}])
    with pytest.raises(ContractViolation, match="duplicate"):
        load_split(path)
