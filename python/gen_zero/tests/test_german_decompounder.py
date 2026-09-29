"""Tests for the German decompounder and the hyperbolic radius floor."""

import unittest

import numpy as np

from gen_zero.nanocore.german_decompounder import (
    GermanDecompounder,
    decompound,
    project_hyperbolic_floor,
)


class TestGermanDecompounder(unittest.TestCase):
    def setUp(self):
        self.d = GermanDecompounder()

    def test_plain_compound(self):
        self.assertEqual(self.d.split("Haustür"), ["haus", "tür"])
        self.assertEqual(self.d.split("Fahrrad"), ["fahr", "rad"])

    def test_three_parts(self):
        self.assertEqual(
            self.d.split("Donaudampfschiff"), ["donau", "dampf", "schiff"]
        )

    def test_linking_s(self):
        self.assertEqual(self.d.split("Arbeitszeit"), ["arbeit", "zeit"])
        self.assertEqual(self.d.split("Lebensmittel"), ["lebens", "mittel"])

    def test_linking_n_and_en(self):
        self.assertEqual(self.d.split("Straßenbahn"), ["straße", "bahn"])
        self.assertEqual(self.d.split("Krankenschwester"), ["kranken", "schwester"])

    def test_umlaut_plural_stem(self):
        self.assertEqual(self.d.split("Bücherregal"), ["buch", "regal"])
        self.assertEqual(self.d.split("Häuserplatz"), ["haus", "platz"])

    def test_known_root_stays_whole(self):
        self.assertEqual(self.d.split("Haus"), ["haus"])
        self.assertEqual(self.d.split("Handschuh"), ["handschuh"])

    def test_unknown_word_unsplit(self):
        self.assertEqual(self.d.split("Xylophonspieler"), ["xylophonspieler"])

    def test_partial_cover_unsplit(self):
        self.assertEqual(self.d.split("Hausqzxv"), ["hausqzxv"])

    def test_case_and_whitespace(self):
        self.assertEqual(self.d.split("  HAUSTÜR "), ["haus", "tür"])
        self.assertEqual(self.d.split(""), [])

    def test_custom_lexicon(self):
        d = GermanDecompounder(roots=["kaffee", "tasse"])
        self.assertEqual(d.split("Kaffeetasse"), ["kaffee", "tasse"])
        self.assertTrue(d.is_compound("Kaffeetasse"))
        self.assertFalse(d.is_compound("Kaffee"))

    def test_split_text(self):
        self.assertEqual(
            self.d.split_text("Haustür Fahrrad"), ["haus", "tür", "fahr", "rad"]
        )

    def test_module_helper(self):
        self.assertEqual(decompound("Haustür"), ["haus", "tür"])

    def test_bad_input(self):
        with self.assertRaises(TypeError):
            self.d.split(None)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            GermanDecompounder(min_part_len=0)


class TestHyperbolicFloor(unittest.TestCase):
    def test_collapsed_vector_pushed_to_floor(self):
        out = project_hyperbolic_floor(np.array([1e-4, 0.0, 0.0]))
        self.assertAlmostEqual(float(np.linalg.norm(out)), 0.25, places=12)
        np.testing.assert_allclose(out / np.linalg.norm(out), [1.0, 0.0, 0.0])

    def test_direction_preserved(self):
        v = np.array([0.0, -0.01, 0.0])
        out = project_hyperbolic_floor(v)
        np.testing.assert_allclose(out, [0.0, -0.25, 0.0])

    def test_zero_vector_gets_canonical_axis(self):
        out = project_hyperbolic_floor(np.zeros(4))
        np.testing.assert_allclose(out, [0.25, 0.0, 0.0, 0.0])

    def test_mid_radius_unchanged(self):
        v = np.array([0.3, 0.4])
        np.testing.assert_allclose(project_hyperbolic_floor(v), v)

    def test_outside_ball_clipped_inside(self):
        out = project_hyperbolic_floor(np.array([3.0, 4.0]))
        self.assertLess(float(np.linalg.norm(out)), 1.0)
        self.assertAlmostEqual(float(np.linalg.norm(out)), 0.999, places=12)

    def test_batch_and_custom_floor(self):
        x = np.array([[0.0, 0.0], [0.01, 0.0], [0.5, 0.0]])
        out = project_hyperbolic_floor(x, min_radius=0.4)
        self.assertEqual(out.shape, x.shape)
        np.testing.assert_allclose(np.linalg.norm(out, axis=1), [0.4, 0.4, 0.5])

    def test_invalid_input(self):
        with self.assertRaises(ValueError):
            project_hyperbolic_floor(np.ones(3), min_radius=0.0)
        with self.assertRaises(ValueError):
            project_hyperbolic_floor(np.ones(3), min_radius=1.0)
        with self.assertRaises(ValueError):
            project_hyperbolic_floor(np.array([np.nan, 1.0]))
        with self.assertRaises(ValueError):
            project_hyperbolic_floor(np.ones((2, 2, 2)))


if __name__ == "__main__":
    unittest.main()
