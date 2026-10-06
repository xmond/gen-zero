"""Tests for gen_zero.nanocore.cpu_lookup_engine.

Covers exact-match hit/miss + export round-trips, prototype codebook
classification accuracy, pairwise manifold equivalence judgment, and the
cascaded CPUDecisionEngine including its latency budgets.
"""

import numpy as np
import pytest

from gen_zero.nanocore.action_etf_embedding import generate_simplex_etf
from gen_zero.nanocore.cpu_lookup_engine import (
    CPUDecisionEngine,
    ExactMatchLookupTable,
    PairwiseManifoldLookup,
    SimplexPrototypeCodebook,
    normalize_text,
)

CATEGORIES = ("stop", "continue", "escalate")


# ---------------------------------------------------------------------------
# normalize_text
# ---------------------------------------------------------------------------

class TestNormalizeText:
    def test_whitespace_collapsing_is_invariant(self):
        assert normalize_text("  Hello   WORLD  ") == "Hello WORLD"
        assert normalize_text("Hello\tWORLD\n") == "Hello WORLD"

    def test_case_is_strictly_preserved_for_units(self):
        # "5mW" (milliwatts) must never normalize to "5MW" (megawatts)
        assert normalize_text("power 5mW") != normalize_text("power 5MW")

    def test_operators_and_symbols_are_strictly_preserved(self):
        # "!enabled" must never normalize to "enabled"
        assert normalize_text("!enabled") != normalize_text("enabled")
        # "5′" (5 feet/minutes) must never normalize to "5"
        assert normalize_text("length 5′") != normalize_text("length 5")

    def test_distinct_meanings_stay_distinct(self):
        assert normalize_text("stop now") != normalize_text("continue now")

    def test_decimal_point_is_not_collapsed_into_a_different_number(self):
        # "1.5mg" must never normalize to the same key as "15mg".
        assert normalize_text("1.5mg") != normalize_text("15mg")

    def test_decimal_point_is_preserved_generally(self):
        assert normalize_text("3.14") != normalize_text("314")
        assert "3.14" in normalize_text("3.14")

    def test_negative_sign_prefix_is_preserved(self):
        assert normalize_text("-5") != normalize_text("5")

    def test_word_internal_hyphen_is_preserved(self):
        assert normalize_text("re-sign") != normalize_text("resign")


# ---------------------------------------------------------------------------
# ExactMatchLookupTable
# ---------------------------------------------------------------------------

class TestExactMatchLookupTable:
    def _build_table(self) -> ExactMatchLookupTable:
        table = ExactMatchLookupTable(categories=CATEGORIES)
        table.insert("emergency stop the reactor", label="stop")
        table.insert("keep the process running", label="continue")
        table.insert("page the on-call engineer", label="escalate")
        return table

    def test_hit_exact_text(self):
        table = self._build_table()
        hit, distribution, label = table.lookup("emergency stop the reactor")
        assert hit is True
        assert label == "stop"
        assert distribution is not None
        assert distribution[CATEGORIES.index("stop")] == pytest.approx(1.0)

    def test_hit_is_normalization_invariant(self):
        table = self._build_table()
        hit, _, label = table.lookup("  emergency   stop the reactor  ")
        assert hit is True
        assert label == "stop"

    def test_miss_for_unseen_text(self):
        table = self._build_table()
        hit, distribution, label = table.lookup("completely unrelated input text")
        assert hit is False
        assert distribution is None
        assert label is None

    def test_soft_distribution_entry(self):
        table = ExactMatchLookupTable(categories=CATEGORIES)
        table.insert("ambiguous signal", distribution=[0.5, 0.4, 0.1])
        hit, distribution, label = table.lookup("ambiguous signal")
        assert hit is True
        assert label == "stop"
        assert distribution == pytest.approx([0.5, 0.4, 0.1])

    def test_distribution_is_renormalized(self):
        table = ExactMatchLookupTable(categories=CATEGORIES)
        table.insert("unnormalized", distribution=[5.0, 3.0, 2.0])
        _, distribution, _ = table.lookup("unnormalized")
        assert float(np.sum(distribution)) == pytest.approx(1.0)

    def test_rejects_invalid_label(self):
        table = ExactMatchLookupTable(categories=CATEGORIES)
        with pytest.raises(ValueError):
            table.insert("bad label", label="not_a_category")

    def test_rejects_both_or_neither_label_and_distribution(self):
        table = ExactMatchLookupTable(categories=CATEGORIES)
        with pytest.raises(ValueError):
            table.insert("bad call", label="stop", distribution=[1.0, 0.0, 0.0])
        with pytest.raises(ValueError):
            table.insert("bad call")

    def test_json_export_import_round_trip(self):
        table = self._build_table()
        payload = table.to_json()
        restored = ExactMatchLookupTable.from_json(payload)

        assert len(restored) == len(table)
        for query in ["emergency stop the reactor", "keep the process running", "unseen query"]:
            orig = table.lookup(query)
            new = restored.lookup(query)
            assert orig[0] == new[0]
            assert orig[2] == new[2]
            if orig[1] is not None:
                assert np.allclose(orig[1], new[1])

    def test_binary_export_import_round_trip(self):
        table = self._build_table()
        blob = table.to_bytes()
        restored = ExactMatchLookupTable.from_bytes(blob)

        assert len(restored) == len(table)
        for query in ["emergency stop the reactor", "page the on-call engineer", "unseen query"]:
            orig = table.lookup(query)
            new = restored.lookup(query)
            assert orig[0] == new[0]
            assert orig[2] == new[2]
            if orig[1] is not None:
                assert np.allclose(orig[1], new[1])

    def test_binary_round_trip_rejects_bad_magic(self):
        with pytest.raises(ValueError):
            ExactMatchLookupTable.from_bytes(b"NOPE" + b"\x00" * 20)

    def test_binary_round_trip_rejects_short_blob(self):
        with pytest.raises(ValueError):
            ExactMatchLookupTable.from_bytes(b"GZL0")

    def test_binary_round_trip_rejects_truncated_entries(self):
        table = self._build_table()
        blob = table.to_bytes()
        for cut in (len(blob) - 1, len(blob) // 2, 17):
            with pytest.raises(ValueError):
                ExactMatchLookupTable.from_bytes(blob[:cut])

    def test_binary_round_trip_rejects_trailing_garbage(self):
        table = self._build_table()
        blob = table.to_bytes() + b"\x00"
        with pytest.raises(ValueError):
            ExactMatchLookupTable.from_bytes(blob)

    def test_insert_rejects_negative_distribution(self):
        table = ExactMatchLookupTable(categories=CATEGORIES)
        with pytest.raises(ValueError):
            table.insert("bad probs", distribution=[-1.0, 2.0, 0.0])

    def test_insert_rejects_non_finite_distribution(self):
        table = ExactMatchLookupTable(categories=CATEGORIES)
        with pytest.raises(ValueError):
            table.insert("bad probs", distribution=[float("nan"), 1.0, 0.0])

    def test_lookup_returns_a_copy_not_internal_state(self):
        table = self._build_table()
        _, distribution, _ = table.lookup("emergency stop the reactor")
        distribution[:] = 999.0

        _, second_lookup, _ = table.lookup("emergency stop the reactor")
        assert not np.array_equal(second_lookup, distribution)
        assert second_lookup[CATEGORIES.index("stop")] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# SimplexPrototypeCodebook
# ---------------------------------------------------------------------------

class TestSimplexPrototypeCodebook:
    LABELS = ("alpha", "beta", "gamma", "delta")
    DIM = 32

    def test_default_prototypes_are_unit_norm_and_equiangular(self):
        codebook = SimplexPrototypeCodebook(labels=self.LABELS, dim=self.DIM)
        norms = np.linalg.norm(codebook.prototypes, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-8)

        gram = codebook.prototypes @ codebook.prototypes.T
        off_diag = gram[~np.eye(len(self.LABELS), dtype=bool)]
        expected_ip = -1.0 / (len(self.LABELS) - 1)
        assert np.allclose(off_diag, expected_ip, atol=1e-6)

    def test_classify_recovers_correct_label_under_noise(self):
        rng = np.random.RandomState(42)
        codebook = SimplexPrototypeCodebook(labels=self.LABELS, dim=self.DIM, temperature=0.15)

        for idx, label in enumerate(self.LABELS):
            prototype = codebook.prototypes[idx]
            noisy = prototype + 0.05 * rng.randn(self.DIM)
            predicted_label, probs, metrics = codebook.classify(noisy)
            assert predicted_label == label
            assert probs.shape == (len(self.LABELS),)
            assert metrics["confidence"] > 0.5

    def test_classify_batch_matches_single_classify(self):
        rng = np.random.RandomState(7)
        codebook = SimplexPrototypeCodebook(labels=self.LABELS, dim=self.DIM, temperature=0.2)

        queries = np.stack([
            codebook.prototypes[i] + 0.03 * rng.randn(self.DIM)
            for i in range(len(self.LABELS))
        ])
        batch_labels, batch_probs = codebook.classify_batch(queries)

        for i in range(len(self.LABELS)):
            single_label, single_probs, _ = codebook.classify(queries[i])
            assert batch_labels[i] == single_label
            assert np.allclose(batch_probs[i], single_probs, atol=1e-10)

    def test_explicit_prototypes_are_normalized(self):
        raw = np.array([[3.0, 4.0], [0.0, 5.0]])  # norms 5.0, 5.0
        codebook = SimplexPrototypeCodebook(labels=("a", "b"), dim=2, prototypes=raw)
        norms = np.linalg.norm(codebook.prototypes, axis=1)
        assert np.allclose(norms, 1.0)

    def test_rejects_mismatched_prototype_shape(self):
        with pytest.raises(ValueError):
            SimplexPrototypeCodebook(labels=("a", "b"), dim=4, prototypes=np.zeros((2, 3)))

    def test_zero_vector_abstains_instead_of_faking_a_decision(self):
        codebook = SimplexPrototypeCodebook(labels=self.LABELS, dim=self.DIM)
        label, probs, metrics = codebook.classify(np.zeros(self.DIM))
        assert label is None
        assert metrics["abstain"] is True
        assert metrics["reason"] == "zero_norm"

    def test_nan_vector_abstains_instead_of_faking_a_decision(self):
        codebook = SimplexPrototypeCodebook(labels=self.LABELS, dim=self.DIM)
        vector = np.full(self.DIM, np.nan)
        label, probs, metrics = codebook.classify(vector)
        assert label is None
        assert metrics["abstain"] is True
        assert metrics["reason"] == "non_finite"

    def test_below_confidence_gate_abstains(self):
        codebook = SimplexPrototypeCodebook(
            labels=self.LABELS, dim=self.DIM, temperature=1.0, min_confidence=0.99,
        )
        # A prototype vector itself likely won't clear an unreasonably high bar.
        label, probs, metrics = codebook.classify(codebook.prototypes[0])
        assert label is None
        assert metrics["abstain"] is True
        assert metrics["reason"] in ("low_confidence", "low_margin")

    def test_classify_batch_abstains_per_row_like_classify(self):
        # Regression: classify_batch used to clamp degenerate rows' norms to 1
        # and force an argmax label, disagreeing with classify()'s abstain.
        rng = np.random.RandomState(3)
        codebook = SimplexPrototypeCodebook(labels=self.LABELS, dim=self.DIM, temperature=0.15)
        good = codebook.prototypes[2] + 0.02 * rng.randn(self.DIM)
        batch = np.stack([
            np.zeros(self.DIM),
            good,
            np.full(self.DIM, np.nan),
        ])

        labels, probs = codebook.classify_batch(batch)

        assert labels[0] is None
        assert labels[1] == self.LABELS[2]
        assert labels[2] is None
        assert np.array_equal(probs[0], np.zeros(len(self.LABELS)))
        assert np.array_equal(probs[2], np.zeros(len(self.LABELS)))


# ---------------------------------------------------------------------------
# PairwiseManifoldLookup
# ---------------------------------------------------------------------------

class TestPairwiseManifoldLookup:
    DIM = 16

    def _make_concept_anchors(self, k: int) -> np.ndarray:
        # Well-separated concept anchors via the same ETF construction used
        # elsewhere in the codebase; not tied to any particular benchmark.
        return generate_simplex_etf(k, self.DIM)

    def test_geometric_equivalence_judgment(self):
        rng = np.random.RandomState(11)
        anchors = self._make_concept_anchors(5)

        calibration_pairs = []
        calibration_labels = []
        for i in range(len(anchors)):
            # Equivalent: same concept, small paraphrase-like noise
            u = anchors[i] + 0.02 * rng.randn(self.DIM)
            v = anchors[i] + 0.02 * rng.randn(self.DIM)
            calibration_pairs.append((u, v))
            calibration_labels.append(True)

            # Not equivalent: different, geometrically distant concepts
            j = (i + 1) % len(anchors)
            calibration_pairs.append((anchors[i], anchors[j]))
            calibration_labels.append(False)

        lookup = PairwiseManifoldLookup(dim=self.DIM)
        lookup.fit_centers(calibration_pairs, calibration_labels)

        # Held-out pairs with a fresh noise seed
        rng_holdout = np.random.RandomState(99)
        for i in range(len(anchors)):
            u = anchors[i] + 0.02 * rng_holdout.randn(self.DIM)
            v = anchors[i] + 0.02 * rng_holdout.randn(self.DIM)
            is_equivalent, metrics = lookup.predict_equivalence(u, v)
            assert is_equivalent is True
            assert metrics["margin"] > 0

            j = (i + 2) % len(anchors)
            is_equivalent, metrics = lookup.predict_equivalence(anchors[i], anchors[j])
            assert is_equivalent is False
            assert metrics["margin"] < 0

    def test_predict_before_fit_raises(self):
        lookup = PairwiseManifoldLookup(dim=self.DIM)
        with pytest.raises(RuntimeError):
            lookup.predict_equivalence(np.ones(self.DIM), np.ones(self.DIM))

    def test_fit_requires_both_classes(self):
        lookup = PairwiseManifoldLookup(dim=self.DIM)
        u = np.ones(self.DIM)
        v = np.ones(self.DIM)
        with pytest.raises(ValueError):
            lookup.fit_centers([(u, v)], [True])

    def test_identical_vectors_have_zero_divergence(self):
        rng = np.random.RandomState(3)
        anchors = self._make_concept_anchors(3)
        pairs = [(anchors[0], anchors[0]), (anchors[1], anchors[2])]
        labels = [True, False]
        lookup = PairwiseManifoldLookup(dim=self.DIM)
        lookup.fit_centers(pairs, labels)

        _, metrics = lookup.predict_equivalence(anchors[0], anchors[0])
        assert metrics["divergence_norm"] == pytest.approx(0.0, abs=1e-10)


# ---------------------------------------------------------------------------
# CPUDecisionEngine
# ---------------------------------------------------------------------------

class TestCPUDecisionEngine:
    DIM = 24

    def _build_engine(self) -> CPUDecisionEngine:
        exact_table = ExactMatchLookupTable(categories=CATEGORIES)
        exact_table.insert("emergency stop the reactor", label="stop")

        codebook_labels = ("alpha", "beta", "gamma")
        codebook = SimplexPrototypeCodebook(labels=codebook_labels, dim=self.DIM, temperature=0.15)

        anchors = generate_simplex_etf(4, self.DIM)
        rng = np.random.RandomState(5)
        pairs = [
            (anchors[0] + 0.02 * rng.randn(self.DIM), anchors[0] + 0.02 * rng.randn(self.DIM)),
            (anchors[1], anchors[2]),
        ]
        labels = [True, False]
        manifold = PairwiseManifoldLookup(dim=self.DIM)
        manifold.fit_centers(pairs, labels)

        return CPUDecisionEngine(
            exact_table=exact_table,
            prototype_codebook=codebook,
            manifold_lookup=manifold,
        )

    def test_l0_exact_hit_is_fast_and_correct(self):
        engine = self._build_engine()

        # Warm up
        for _ in range(5):
            engine.decide(text="  emergency   stop the reactor  ")

        results = [
            engine.decide(text="  emergency   stop the reactor  ")
            for _ in range(10)
        ]
        best_us = min(r.elapsed_us for r in results)
        assert results[0].hit_level == "L0_exact_match"
        assert results[0].label == "stop"
        assert best_us < 100.0, f"Expected L0 best latency < 100us, got {best_us}us"

    def test_l1_prototype_fallback_on_exact_miss(self):
        engine = self._build_engine()
        codebook = engine.prototype_codebook
        target_idx = 1  # "beta"
        query_vector = codebook.prototypes[target_idx] + 0.03 * np.random.RandomState(1).randn(self.DIM)

        engine.decide(text="unseen text", vector=query_vector)  # warm up
        result = engine.decide(text="unseen text", vector=query_vector)

        assert result.hit_level == "L1_prototype"
        assert result.label == "beta"
        assert result.elapsed_ms < 5.0

    def test_l1_manifold_fallback_for_pairs(self):
        engine = self._build_engine()
        anchors = generate_simplex_etf(4, self.DIM)
        rng = np.random.RandomState(42)
        u = anchors[0] + 0.02 * rng.randn(self.DIM)
        v = anchors[0] + 0.02 * rng.randn(self.DIM)

        engine.decide(pair=(u, v))  # warm up
        result = engine.decide(pair=(u, v))

        assert result.hit_level == "L1_manifold"
        assert result.label is True
        assert result.elapsed_ms < 5.0

    def test_full_miss_when_nothing_configured_matches(self):
        engine = self._build_engine()
        result = engine.decide(text="totally novel input with no lookup match")
        assert result.hit_level == "miss"
        assert result.label is None

    def test_l0_takes_precedence_over_l1_when_both_available(self):
        engine = self._build_engine()
        query_vector = engine.prototype_codebook.prototypes[0]
        result = engine.decide(text="emergency stop the reactor", vector=query_vector)
        assert result.hit_level == "L0_exact_match"

    def test_decision_result_to_dict_is_json_serializable(self):
        import json

        engine = self._build_engine()
        result = engine.decide(text="emergency stop the reactor")
        serialized = json.dumps(result.to_dict())
        assert "L0_exact_match" in serialized

    def test_l1_abstain_falls_back_to_miss_not_a_fake_decision(self):
        engine = self._build_engine()
        # No L0 hit, and a zero vector can never yield a real L1 prototype hit.
        result = engine.decide(text="totally unseen input", vector=np.zeros(self.DIM))
        assert result.hit_level == "miss"
        assert result.label is None
        # The miss must not be anonymous: why L1 abstained is preserved.
        assert result.details["l1_abstain"]["reason"] == "zero_norm"

    def test_l1_abstain_on_nan_vector_falls_back_to_miss(self):
        engine = self._build_engine()
        result = engine.decide(text="totally unseen input", vector=np.full(self.DIM, np.nan))
        assert result.hit_level == "miss"
        assert result.label is None

    def test_feature_extractor_bridges_l0_miss_to_l1_prototype(self):
        exact_table = ExactMatchLookupTable(categories=CATEGORIES)
        exact_table.insert("emergency stop the reactor", label="stop")

        codebook_labels = ("alpha", "beta", "gamma")
        codebook = SimplexPrototypeCodebook(labels=codebook_labels, dim=self.DIM, temperature=0.15)
        target_vector = codebook.prototypes[1]  # "beta"

        def feature_extractor(text: str) -> np.ndarray:
            # Stand-in for a real text encoder: routes any unseen text to the
            # same target vector so the end-to-end bridge is exercised.
            return target_vector

        engine = CPUDecisionEngine(
            exact_table=exact_table,
            prototype_codebook=codebook,
            feature_extractor=feature_extractor,
        )

        result = engine.decide(text="an input never seen by L0")
        assert result.hit_level == "L1_prototype"
        assert result.label == "beta"

    def test_feature_extractor_is_not_consulted_on_l0_hit(self):
        exact_table = ExactMatchLookupTable(categories=CATEGORIES)
        exact_table.insert("emergency stop the reactor", label="stop")
        codebook = SimplexPrototypeCodebook(labels=("alpha", "beta"), dim=self.DIM)

        calls = []

        def feature_extractor(text: str) -> np.ndarray:
            calls.append(text)
            return codebook.prototypes[0]

        engine = CPUDecisionEngine(
            exact_table=exact_table,
            prototype_codebook=codebook,
            feature_extractor=feature_extractor,
        )
        result = engine.decide(text="emergency stop the reactor")
        assert result.hit_level == "L0_exact_match"
        assert calls == []

    def test_l1_manifold_abstain_falls_back_to_miss(self):
        codebook = SimplexPrototypeCodebook(labels=("a", "b"), dim=self.DIM)
        manifold = PairwiseManifoldLookup(dim=self.DIM)
        manifold.fit_centers(
            pairs=[(codebook.prototypes[0], codebook.prototypes[0]), (codebook.prototypes[0], codebook.prototypes[1])],
            labels=[True, False],
        )
        engine = CPUDecisionEngine(manifold_lookup=manifold)
        # Degenerate zero-vector pair must abstain and fall through to miss
        result = engine.decide(pair=(np.zeros(self.DIM), np.zeros(self.DIM)))
        assert result.hit_level == "miss"
        assert result.label is None
        assert result.details["l1_manifold_abstain"]["abstain"] is True
        assert result.details["l1_manifold_abstain"]["reason"] == "zero_norm"


class TestHardenedRemediationVerifications:
    """Verifications ensuring zero regressions across all Astra & Fable audit findings."""

    def test_numeric_variants_do_not_collide(self):
        from gen_zero.nanocore.cpu_lookup_engine import normalize_text
        pairs = [
            (".5", "5"),
            ("-.5", "5"),
            ("5%", "5"),
            ("–5", "5"),
            ("3.14", "314"),
            ("dose 1.5mg", "dose 15mg"),
            ("أسعار ٥٪", "أسعار ٥"),
        ]
        for a, b in pairs:
            assert normalize_text(a) != normalize_text(b), f"Collision between {a!r} and {b!r}"

    def test_exact_table_collision_detection(self):
        table = ExactMatchLookupTable(categories=("a", "b"))
        table.insert("critical dose 1.5mg", label="a")
        # Colliding key with conflicting label raises ValueError if overwrite=False
        import pytest
        with pytest.raises(ValueError, match="Normalization collision"):
            table.insert("critical dose 1.5mg", label="b", overwrite=False)

    def test_insert_rejects_overflow_sum(self):
        table = ExactMatchLookupTable(categories=("a", "b"))
        import pytest
        with pytest.raises(ValueError, match="finite and positive"):
            table.insert("overflow", distribution=[1e308, 1e308])

    def test_deserialization_rejects_invalid_distributions(self):
        import json
        import pytest
        from gen_zero.nanocore.cpu_lookup_engine import _digest_key
        bad_payload = json.dumps({
            "categories": ["a", "b"],
            "entries": {
                _digest_key("key1"): {
                    "normalized_text": "key1",
                    "categories": ["a", "b"],
                    "distribution": [-1.0, 2.0],
                }
            }
        })
        with pytest.raises(ValueError, match="negative"):
            ExactMatchLookupTable.from_json(bad_payload)

    def test_orthogonal_query_abstains_on_tie(self):
        # Prototypes on x-axis: [1, 0] and [-1, 0]
        prototypes = np.array([[1.0, 0.0], [-1.0, 0.0]])
        codebook = SimplexPrototypeCodebook(labels=("pos", "neg"), dim=2, prototypes=prototypes)
        # Query on y-axis: [0, 1] is exactly orthogonal, dot products are 0, margin is 0 (exact tie)
        label, probs, metrics = codebook.classify(np.array([0.0, 1.0]))
        assert label is None
        assert metrics["abstain"] is True
        assert metrics["reason"] == "tie"

    def test_classify_batch_rejects_1d_input(self):
        codebook = SimplexPrototypeCodebook(labels=("a", "b"), dim=2)
        import pytest
        with pytest.raises(ValueError, match="requires 2D array"):
            codebook.classify_batch(np.array([1.0, 0.0]))

    def test_pairwise_manifold_abstains_on_zero_and_nan(self):
        manifold = PairwiseManifoldLookup(dim=2)
        manifold.fit_centers(
            pairs=[(np.array([1.0, 0.0]), np.array([1.0, 0.0])), (np.array([1.0, 0.0]), np.array([-1.0, 0.0]))],
            labels=[True, False],
        )
        is_equiv, metrics = manifold.predict_equivalence(np.zeros(2), np.zeros(2))
        assert is_equiv is None
        assert metrics["abstain"] is True
        assert metrics["reason"] == "zero_norm"

        is_equiv_nan, metrics_nan = manifold.predict_equivalence(np.array([np.nan, 1.0]), np.array([1.0, 0.0]))
        assert is_equiv_nan is None
        assert metrics_nan["abstain"] is True
        assert metrics_nan["reason"] == "non_finite"

    def test_exact_table_default_overwrite_is_false(self):
        table = ExactMatchLookupTable(categories=("a", "b"))
        table.insert("dose 1.5mg", label="a")
        # Default overwrite=False must reject conflicting label without explicit flag
        with pytest.raises(ValueError, match="Normalization collision"):
            table.insert("dose 1.5mg", label="b")
        # Explicit overwrite=True succeeds
        table.insert("dose 1.5mg", label="b", overwrite=True)
        hit, _, label = table.lookup("dose 1.5mg")
        assert hit is True
        assert label == "b"

    def test_exact_table_lookup_tie_returns_none_label(self):
        table = ExactMatchLookupTable(categories=("a", "b", "c"))
        table.insert("balanced", distribution=[0.45, 0.45, 0.1])
        hit, dist, label = table.lookup("balanced")
        assert hit is True
        assert label is None  # Exact tie between 'a' and 'b' produces None label
        assert dist == pytest.approx([0.45, 0.45, 0.1])

    def test_cpu_decision_engine_l0_tie_falls_through_or_misses(self):
        table = ExactMatchLookupTable(categories=("a", "b"))
        table.insert("ambiguous", distribution=[0.5, 0.5])
        engine = CPUDecisionEngine(exact_table=table)
        res = engine.decide(text="ambiguous")
        assert res.hit_level == "miss"
        assert res.label is None
        assert "l0_tie" in res.details
        assert res.details["l0_tie"]["reason"] == "insufficient_margin"

    def test_exact_table_from_bytes_digest_tampering_rejected(self):
        import hashlib
        table = ExactMatchLookupTable(categories=("a", "b"))
        table.insert("key1", label="a")
        blob = bytearray(table.to_bytes())
        # 1. Tampering without updating sha256 fails with checksum mismatch
        # Header size is 56 bytes. Category block is (4+1) + (4+1) = 10 bytes.
        digest_pos = 56 + 10
        blob[digest_pos] = ord("0") if blob[digest_pos] != ord("0") else ord("1")
        with pytest.raises(ValueError, match="table sha256 checksum mismatch"):
            ExactMatchLookupTable.from_bytes(bytes(blob))

        # 2. Tampering digest WITH recomputed sha256 fails with digest mismatch
        tampered_payload = blob[56:]
        new_sha = hashlib.sha256(blob[:24] + tampered_payload).digest()
        blob[24:56] = new_sha  # table_sha256 offset is 24 (4+4+4+4+8 = 24)
        with pytest.raises(ValueError, match="corrupt blob: entry digest mismatch"):
            ExactMatchLookupTable.from_bytes(bytes(blob))

        # 3. Tampering header fields (e.g. min_margin float64 at offset 16..24) fails with checksum mismatch
        header_tampered = bytearray(table.to_bytes())
        header_tampered[22] ^= 0x10  # tamper byte inside min_margin
        with pytest.raises(ValueError, match="table sha256 checksum mismatch"):
            ExactMatchLookupTable.from_bytes(bytes(header_tampered))

    def test_pairwise_manifold_fit_centers_rejects_nan_and_zero_and_dim_mismatch(self):
        manifold = PairwiseManifoldLookup(dim=2)
        # NaN vector in calibration pair
        with pytest.raises(ValueError, match="contains non-finite values"):
            manifold.fit_centers(
                pairs=[(np.array([np.nan, 1.0]), np.array([1.0, 0.0])), (np.array([1.0, 0.0]), np.array([0.0, 1.0]))],
                labels=[True, False],
            )
        # Zero-norm vector in calibration pair
        with pytest.raises(ValueError, match="contains near-zero or zero-norm vector"):
            manifold.fit_centers(
                pairs=[(np.array([0.0, 0.0]), np.array([1.0, 0.0])), (np.array([1.0, 0.0]), np.array([0.0, 1.0]))],
                labels=[True, False],
            )
        # Dim mismatch in calibration pair
        with pytest.raises(ValueError, match="dimensionality mismatch"):
            manifold.fit_centers(
                pairs=[(np.array([1.0, 0.0, 0.0]), np.array([1.0, 0.0])), (np.array([1.0, 0.0]), np.array([0.0, 1.0]))],
                labels=[True, False],
            )

    def test_pairwise_manifold_predict_equivalence_rejects_dim_mismatch(self):
        manifold = PairwiseManifoldLookup(dim=2)
        manifold.fit_centers(
            pairs=[(np.array([1.0, 0.0]), np.array([1.0, 0.0])), (np.array([1.0, 0.0]), np.array([0.0, 1.0]))],
            labels=[True, False],
        )
        is_equiv, metrics = manifold.predict_equivalence(np.array([1.0, 0.0, 0.0]), np.array([1.0, 0.0]))
        assert is_equiv is None
        assert metrics["abstain"] is True
        assert metrics["reason"] == "dim_mismatch"

    def test_silent_winner_change_rejected_by_overwrite_false(self):
        table = ExactMatchLookupTable(categories=("A", "B"))
        table.insert("sample_x", distribution=[0.500001, 0.499999])
        with pytest.raises(ValueError, match="Normalization collision"):
            table.insert("sample_x", distribution=[0.499999, 0.500001], overwrite=False)

    def test_overflow_vector_norm_in_calibration_rejected(self):
        manifold = PairwiseManifoldLookup(dim=2)
        with pytest.raises(ValueError, match="non-finite norm"):
            manifold.fit_centers(
                pairs=[(np.array([1e308, 0.0]), np.array([1e308, 0.0])), (np.array([1.0, 0.0]), np.array([-1.0, 0.0]))],
                labels=[True, False],
            )

    def test_from_bytes_and_from_json_reject_duplicate_and_unnormalized_keys(self):
        import json
        import struct
        import hashlib
        from gen_zero.nanocore.cpu_lookup_engine import _digest_key

        table = ExactMatchLookupTable(categories=("A", "B"))
        table.insert("norm_key", label="A")

        # Construct duplicate key binary blob GZL0 v2
        cat_block = struct.pack("<I1sI1s", 1, b"A", 1, b"B")
        e_digest = _digest_key("norm_key").encode("ascii")
        e_key = b"norm_key"
        e_dist = np.array([1.0, 0.0], dtype="<f8").tobytes()
        entry_block = struct.pack("<16sI8s16s", e_digest, 8, e_key, e_dist)
        bad_payload = cat_block + entry_block + entry_block
        header_prefix = struct.pack("<4sIIId", b"GZL0", 2, 2, 2, 1e-7)
        bad_sha = hashlib.sha256(header_prefix + bad_payload).digest()
        bad_blob = header_prefix + bad_sha + bad_payload
        with pytest.raises(ValueError, match="duplicate key 'norm_key' detected in binary blob"):
            ExactMatchLookupTable.from_bytes(bad_blob)

        # Raw JSON duplicate key rejection
        raw_dup_json = '{"categories": ["A", "B"], "categories": ["A", "B"], "entries": {}}'
        with pytest.raises(ValueError, match="duplicate member key"):
            ExactMatchLookupTable.from_json(raw_dup_json)

        # Test unnormalized key in JSON
        unnorm_json = json.dumps({
            "categories": ["A", "B"],
            "entries": {
                _digest_key("  unnorm  "): {
                    "normalized_text": "  unnorm  ",
                    "categories": ["A", "B"],
                    "distribution": [1.0, 0.0],
                }
            }
        })
        with pytest.raises(ValueError, match="not canonically normalized"):
            ExactMatchLookupTable.from_json(unnorm_json)

    def test_codebook_rejects_nan_configuration(self):
        with pytest.raises(ValueError, match="min_confidence"):
            SimplexPrototypeCodebook(labels=("a", "b"), dim=2, min_confidence=np.nan)
        with pytest.raises(ValueError, match="min_margin"):
            SimplexPrototypeCodebook(labels=("a", "b"), dim=2, min_margin=np.nan)
        with pytest.raises(ValueError, match="temperature"):
            SimplexPrototypeCodebook(labels=("a", "b"), dim=2, temperature=np.nan)

    def test_codebook_classify_rejects_dim_mismatch(self):
        codebook = SimplexPrototypeCodebook(labels=("a", "b"), dim=2)
        label, probs, metrics = codebook.classify(np.array([1.0, 0.0, 0.0]))
        assert label is None
        assert metrics["abstain"] is True
        assert metrics["reason"] == "dim_mismatch"

    def test_l1_preserves_l0_tie_evidence(self):
        table = ExactMatchLookupTable(categories=("A", "B"))
        table.insert("ambiguous_text", distribution=[0.5, 0.5])
        codebook = SimplexPrototypeCodebook(labels=("A", "B"), dim=2, prototypes=np.eye(2))
        engine = CPUDecisionEngine(exact_table=table, prototype_codebook=codebook)
        res = engine.decide(text="ambiguous_text", vector=np.array([1.0, 0.0]))
        assert res.hit_level == "L1_prototype"
        assert res.label == "A"
        assert "l0_tie" in res.details
        assert res.details["l0_tie"]["reason"] == "insufficient_margin"

    def test_strict_json_serialization_without_nan(self):
        import json
        manifold = PairwiseManifoldLookup(dim=2)
        manifold.fit_centers(
            pairs=[(np.array([1.0, 0.0]), np.array([1.0, 0.0])), (np.array([1.0, 0.0]), np.array([0.0, 1.0]))],
            labels=[True, False],
        )
        engine = CPUDecisionEngine(manifold_lookup=manifold)
        res = engine.decide(pair=(np.array([1.0, 0.0]), np.ones(3)))
        dumped = json.dumps(res.to_dict(), allow_nan=False)
        assert "dim_mismatch" in dumped

    def test_gzl0_v2_preserves_min_margin(self):
        table = ExactMatchLookupTable(categories=("A", "B"), min_margin=0.25)
        table.insert("hello", label="A")
        blob = table.to_bytes()
        loaded = ExactMatchLookupTable.from_bytes(blob)
        assert abs(loaded.min_margin - 0.25) < 1e-9
        hit, dist, label = loaded.lookup("hello")
        assert hit is True
        assert label == "A"

    def test_lookup_min_margin_override_validation(self):
        table = ExactMatchLookupTable(categories=("A", "B"))
        table.insert("hello", label="A")
        with pytest.raises(ValueError, match="min_margin must be a finite non-negative float"):
            table.lookup("hello", min_margin=np.nan)
        with pytest.raises(ValueError, match="min_margin must be a finite non-negative float"):
            table.lookup("hello", min_margin=-0.5)
        with pytest.raises(ValueError, match="min_margin must be a finite non-negative float"):
            table.lookup("hello", min_margin=np.inf)

    def test_l1_manifold_preserves_l0_tie_and_l1_abstain(self):
        table = ExactMatchLookupTable(categories=("A", "B"))
        table.insert("ambiguous_text", distribution=[0.5, 0.5])
        # Prototype codebook that abstains due to low confidence
        codebook = SimplexPrototypeCodebook(
            labels=("A", "B"),
            dim=2,
            prototypes=np.eye(2),
            min_confidence=0.99,  # Force abstain on low confidence
        )
        manifold = PairwiseManifoldLookup(dim=2)
        v1 = np.array([1.0, 0.0])
        v2 = np.array([1.0, 0.0])
        manifold.fit_centers(
            pairs=[(v1, v2), (np.array([1.0, 0.0]), np.array([0.0, 1.0]))],
            labels=[True, False],
        )
        engine = CPUDecisionEngine(
            exact_table=table,
            prototype_codebook=codebook,
            manifold_lookup=manifold,
        )
        # Query with text (L0 tie), vector (L1 prototype abstain), and pair (L1 manifold hit)
        res = engine.decide(
            text="ambiguous_text",
            vector=np.array([0.6, 0.4]),  # softmax confidence < 0.99 -> abstain
            pair=(v1, v2),
        )
        assert res.hit_level == "L1_manifold"
        assert res.label is True
        assert "l0_tie" in res.details
        assert res.details["l0_tie"]["reason"] == "insufficient_margin"
        assert "l1_abstain" in res.details
        assert res.details["l1_abstain"]["abstain"] is True

    def test_roundtrip_preserves_delicate_decision_boundary(self):
        # Astra counterexample: ensure serialized roundtrip never flips decision margin
        weights = [0.9428036791291672, 0.6656574169657534, 0.13339575545169313]
        target_margin = 0.15910966616879713
        table = ExactMatchLookupTable(categories=("a", "b", "c"), min_margin=target_margin)
        table.insert("query_key", distribution=weights)

        # Baseline decision before serialization
        hit, dist, label = table.lookup("query_key")
        assert hit is True
        assert label is None  # Margin is <= min_margin -> abstains on tie

        # Binary roundtrip
        blob = table.to_bytes()
        loaded_bin = ExactMatchLookupTable.from_bytes(blob)
        hit_b, dist_b, label_b = loaded_bin.lookup("query_key")
        assert hit_b is True
        assert label_b is None, "Binary roundtrip must not flip abstention into acceptance!"
        assert np.array_equal(dist, dist_b), "Binary roundtrip must preserve bit-exact floats"

        # JSON roundtrip
        json_str = table.to_json()
        loaded_json = ExactMatchLookupTable.from_json(json_str)
        hit_j, dist_j, label_j = loaded_json.lookup("query_key")
        assert hit_j is True
        assert label_j is None, "JSON roundtrip must not flip abstention into acceptance!"
        assert np.array_equal(dist, dist_j), "JSON roundtrip must preserve bit-exact floats"

    def test_lookup_override_validation_fails_on_miss(self):
        table = ExactMatchLookupTable(categories=("a", "b"))
        with pytest.raises(ValueError, match="min_margin must be a finite non-negative float"):
            table.lookup("non_existent_key", min_margin=np.nan)
        with pytest.raises(ValueError, match="min_margin must be a finite non-negative float"):
            table.lookup("non_existent_key", min_margin=-1.0)
        with pytest.raises(ValueError, match="min_margin must be a finite non-negative float"):
            table.lookup("non_existent_key", min_margin=np.inf)

    def test_from_json_rejects_invalid_types_for_categories_and_margin(self):
        # Non-string categories
        bad_cats_json = '{"categories": [1, 2], "min_margin": 0.1, "entries": {}}'
        with pytest.raises(ValueError, match="categories' must be a list of strings"):
            ExactMatchLookupTable.from_json(bad_cats_json)

        # String min_margin
        bad_margin_str = '{"categories": ["a", "b"], "min_margin": "0.1", "entries": {}}'
        with pytest.raises(ValueError, match="min_margin' must be a finite non-negative number"):
            ExactMatchLookupTable.from_json(bad_margin_str)

        # Boolean min_margin (bool is subclass of int in Python)
        bad_margin_bool = '{"categories": ["a", "b"], "min_margin": true, "entries": {}}'
        with pytest.raises(ValueError, match="min_margin' must be a finite non-negative number"):
            ExactMatchLookupTable.from_json(bad_margin_bool)

    def test_from_json_rejects_non_float_distribution_elements(self):
        import json
        from gen_zero.nanocore.cpu_lookup_engine import _digest_key
        key_digest = _digest_key("key")
        bad_dist_json = json.dumps({
            "categories": ["a", "b"],
            "entries": {
                key_digest: {
                    "normalized_text": "key",
                    "categories": ["a", "b"],
                    "distribution": [0.5, "0.5"],
                }
            }
        })
        with pytest.raises(ValueError, match="must be a real number"):
            ExactMatchLookupTable.from_json(bad_dist_json)

    def test_insert_accepts_numpy_scalars_in_distribution(self):
        table = ExactMatchLookupTable(categories=("a", "b"))
        table.insert("key_np", distribution=[np.float32(0.6), np.float64(0.4)])
        hit, dist, label = table.lookup("key_np")
        assert hit is True
        assert label == "a"







