"""
aime2025.py — AIME 2025 competition math evaluator for KVComm + LatentMAS

Data source: HuggingFace "yentinglin/aime_2025" (split="train", 30 samples)

Task format:
  prompt_A = competition problem (A performs latent thinking / deep reasoning)
  prompt_B = same competition problem (B generates step-by-step solution + answer)
  answer   = integer string, e.g. "42"

Evaluation: integer exact match.
  Both pred and gold are parsed as int; any parsing failure = incorrect.

Notes:
  - AIME answers are integers in [0, 999].
  - Models must output final answer in \\boxed{N}.
  - max_tokens = 4096 (competition math needs long reasoning chains).
  - Uses self.aime = True (same flag as aime2024) since prompt logic is identical.

Attributes used by prompts_latent.py:
  self.aime = True
"""

import re
from typing import Dict, Any

from .base_evaluator import BaseEvaluator
from datasets import load_dataset


def _extract_boxed_answer(text: str) -> str:
    boxes = re.findall(r"\\boxed\{([^}]*)\}", text)
    if not boxes:
        return ""
    content = boxes[-1].strip()
    match = re.search(r"[-+]?\d+(?:\.\d+)?", content)
    return match.group(0) if match else content


def _normalize(ans: str) -> str:
    return ans.strip().lower() if ans else ""


class AIME2025Evaluator(BaseEvaluator):
    """
    AIME 2025 competition math evaluator.

    Attributes used by prompts_latent.py for task detection:
        self.aime = True  (shared flag with AIME2024Evaluator — same prompt logic)

    Item format returned by __iter__:
        {
            "prompt_A": str,   # competition problem
            "prompt_B": str,   # same competition problem
            "answer":   str,   # integer string, e.g. "42"
        }
    """

    def __init__(self, n_samples: int = None):
        super().__init__()
        self.max_tokens = 4096       # AIME requires long reasoning chains
        self.truncate_input = False
        self.multiple_answers = False
        self.n_samples = n_samples
        self.aime = True             # task-detection flag (shared with aime2024)
        self.name = "aime2025"
        self.data = self.load_data()

    def load_data(self):
        ds = load_dataset("yentinglin/aime_2025", split="train")

        records = []
        for item in ds:
            problem = item["problem"].strip()
            answer_raw = str(item["answer"]).strip()

            records.append({
                "prompt_A": problem,
                "prompt_B": problem,
                "answer":   answer_raw,
            })

        if self.n_samples is not None:
            import random
            random.seed(self.random_state)
            records = random.sample(records, min(self.n_samples, len(records)))

        return records

    def evaluate_item(self, item: Dict[str, Any], response: str):
        pred_str = _normalize(_extract_boxed_answer(response))
        gold_str = _normalize(str(item["answer"]))

        correct = False
        if pred_str and gold_str:
            try:
                correct = int(pred_str) == int(gold_str)
            except ValueError:
                nums = re.findall(r"[-+]?\d+(?:\.\d+)?", response)
                if nums:
                    try:
                        correct = int(nums[-1]) == int(gold_str)
                    except ValueError:
                        correct = False

        self.f1_total += float(correct)
        self.f1_count += 1

    def get_result(self) -> float:
        return self.f1_total / self.f1_count if self.f1_count > 0 else 0.0
