import importlib.util
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location("evaluation_config", Path(__file__).parents[1] / "utils" / "evaluation_config.py")
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)


class Evaluator:
    sender_max_tokens = 256
    max_tokens = 64


class TextMASBudgetTests(unittest.TestCase):
    def test_independent_overrides(self):
        self.assertEqual(config.resolve_textmas_budgets(Evaluator(), 512, 32), (512, 32))

    def test_zero_uses_task_defaults(self):
        self.assertEqual(config.resolve_textmas_budgets(Evaluator(), 0, 0), (256, 64))


if __name__ == "__main__":
    unittest.main()
