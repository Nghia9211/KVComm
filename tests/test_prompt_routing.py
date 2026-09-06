import unittest
import ast
from pathlib import Path

from prompts_latent import build_receiver_core, build_sender_core


class FakeEvaluator:
    name = "fixture"
    sender_max_tokens = 256
    primary_metric = "legacy_match"
    prompt_version = "test_v1"

    def validate_task_profile(self):
        return None


def profile(family, task_type, input_mode, answer_format):
    evaluator = FakeEvaluator()
    evaluator.prompt_family = family
    evaluator.task_type = task_type
    evaluator.sender_input_mode = input_mode
    evaluator.answer_format = answer_format
    return evaluator


class PromptRoutingTests(unittest.TestCase):
    def test_all_supported_task_profiles(self):
        expected = {
            "aime2024": ("latentmas", "math", "boxed_integer"),
            "aime2025": ("latentmas", "math", "boxed_integer"),
            "gsm8k": ("latentmas", "math", "boxed_integer"),
            "arc_easy": ("latentmas", "qa", "boxed_choice"),
            "arc_challenge": ("latentmas", "qa", "boxed_choice"),
            "gpqa": ("latentmas", "qa", "boxed_choice"),
            "medqa": ("latentmas", "qa", "boxed_choice"),
            "mbppplus": ("latentmas", "code", "python"),
            "humanevalplus": ("latentmas", "code", "python"),
            "hotpotqa": ("kvcomm", "qa", "short_text"),
            "multifieldqa_en": ("kvcomm", "qa", "short_text"),
            "twowikimqa": ("kvcomm", "qa", "short_text"),
            "musique": ("kvcomm", "qa", "short_text"),
            "qasper": ("kvcomm", "qa", "short_text"),
            "tipsheets": ("kvcomm", "qa", "short_text"),
            "countries": ("kvcomm", "qa", "short_text"),
            "tmath": ("kvcomm", "math", "boxed_integer"),
            "repobench": ("kvcomm", "code", "python"),
            "samsum": ("kvcomm", "summarization", "summary"),
        }
        root = Path(__file__).parents[1] / "dataloader"
        for task, wanted in expected.items():
            tree = ast.parse((root / f"{task}.py").read_text(encoding="utf-8"))
            calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                     and isinstance(node.func, ast.Attribute) and node.func.attr == "configure_task_profile"]
            self.assertTrue(calls, task)
            values = {kw.arg: ast.literal_eval(kw.value) for kw in calls[0].keywords}
            self.assertEqual((values["prompt_family"], values["task_type"], values["answer_format"]), wanted)

    def test_query_aware_qa(self):
        evaluator = profile("kvcomm", "qa", "query_aware_context", "short_text")
        item = {"prompt_A": "CONTEXT_SENTINEL", "prompt_B": "QUESTION_SENTINEL?"}
        sender = build_sender_core(evaluator, item)
        receiver = build_receiver_core(evaluator, item)
        self.assertIn("CONTEXT_SENTINEL", sender)
        self.assertIn("QUESTION_SENTINEL?", sender)
        self.assertIn("QUESTION_SENTINEL?", receiver)
        self.assertNotIn("CONTEXT_SENTINEL", receiver)
        self.assertIn("evidence", sender.lower())

    def test_latentmas_shared_problem(self):
        evaluator = profile("latentmas", "math", "shared_problem", "boxed_integer")
        item = {"prompt_A": "2+2", "prompt_B": "2+2"}
        self.assertIn("Planner Agent", build_sender_core(evaluator, item))
        self.assertIn("boxed", build_receiver_core(evaluator, item))

    def test_native_split(self):
        evaluator = profile("kvcomm", "code", "native_split", "python")
        item = {"prompt_A": "def f():", "prompt_B": "return"}
        self.assertIn("code context", build_sender_core(evaluator, item))
        self.assertNotIn("Evidence Extraction", build_sender_core(evaluator, item))
        self.assertIn("Code Snippet: return", build_receiver_core(evaluator, item))

    def test_unknown_profile_fails(self):
        evaluator = profile("kvcomm", "qa", "shared_problem", "short_text")
        with self.assertRaises(ValueError):
            build_sender_core(evaluator, {"prompt_A": "a", "prompt_B": "b"})


if __name__ == "__main__":
    unittest.main()
