import json
import unittest
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("response_logging", Path(__file__).parents[1] / "utils" / "response_logging.py")
response_logging = importlib.util.module_from_spec(spec)
spec.loader.exec_module(response_logging)
build_response_record, normalize_answers = response_logging.build_response_record, response_logging.normalize_answers


class Evaluator:
    name = "multifieldqa_en"
    prompt_family = "kvcomm"
    prompt_version = "kvcomm_qa_query_aware_v2"
    sender_input_mode = "query_aware_context"


class ResponseLoggingTests(unittest.TestCase):
    def test_answers_are_normalized(self):
        self.assertEqual(normalize_answers({"answers": ["a", "b"]}), ["a", "b"])
        self.assertEqual(normalize_answers({"answer": "a"}), ["a"])

    def test_schema_v2_round_trip(self):
        record = build_response_record(
            idx=0, evaluator=Evaluator(), item={"prompt_A": "ctx", "prompt_B": "q", "answers": ["gold"]},
            method="textmas_two_agent", response="gold", response_a="evidence",
            model_a_prompt="formatted A", model_b_prompt="formatted B",
            item_metrics={"longbench_f1": 1.0}, aggregate_metrics={"longbench_f1": 1.0},
            max_tokens_a=256, max_tokens_b=64, generated_tokens_a=12, generated_tokens_b=1,
            communication_type="text",
        )
        decoded = json.loads(json.dumps(record))
        self.assertEqual(decoded["schema_version"], "v2")
        self.assertEqual(decoded["answers"], ["gold"])
        self.assertEqual(decoded["response_A"], "evidence")


if __name__ == "__main__":
    unittest.main()
