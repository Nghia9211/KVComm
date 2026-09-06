import unittest
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("longbench_metrics", Path(__file__).parents[1] / "utils" / "longbench_metrics.py")
metrics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(metrics)
normalize_answer, qa_f1_score, qa_f1_single = metrics.normalize_answer, metrics.qa_f1_score, metrics.qa_f1_single


class LongBenchMetricTests(unittest.TestCase):
    def test_normalization(self):
        self.assertEqual(normalize_answer("The, Quick  Fox!"), "quick fox")

    def test_exact_and_no_overlap(self):
        self.assertEqual(qa_f1_score("Paris", ["Paris"]), 1.0)
        self.assertEqual(qa_f1_score("London", ["Paris"]), 0.0)

    def test_partial_overlap_is_fractional(self):
        score = qa_f1_single("Margaret Way", "Margaret Way and John Smith")
        self.assertGreater(score, 0.0)
        self.assertLess(score, 1.0)

    def test_repeated_tokens_use_counter(self):
        self.assertAlmostEqual(qa_f1_single("red red blue", "red blue blue"), 2 / 3)

    def test_best_reference(self):
        self.assertEqual(qa_f1_score("Hanoi", ["Saigon", "Hanoi"]), 1.0)


if __name__ == "__main__":
    unittest.main()
