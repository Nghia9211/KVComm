"""
gsm8k.py — GSM8K math word problem evaluator for KVComm + LatentMAS

Data source: HuggingFace "openai/gsm8k" (split="test", 1319 samples)

Task format:
  prompt_A = math problem (A performs latent thinking about the problem)
  prompt_B = same math problem (B generates step-by-step solution + answer)
  answer   = integer string extracted after "####" in solution

Evaluation: numeric exact match (compare as int/float).
  Both pred and gold are parsed as numbers; parse failure = incorrect.

Notes:
  - GSM8K answers are non-negative integers.
  - Models must output final answer in \\boxed{N}.
  - max_tokens = 1024 (math word problems need reasoning chains).

Attributes used by prompts_latent.py:
  self.gsm8k = True
"""

import re
from typing import Dict, Any

from .base_evaluator import BaseEvaluator
from datasets import load_dataset


def _extract_gold_from_solution(solution: str) -> str:
    """Extract the number after '####' in GSM8K solution string."""
    match = re.search(r"####\s*([-+]?\d+(?:\.\d+)?)", solution)
    return match.group(1).strip() if match else ""


def _extract_boxed_answer(text: str) -> str:
    """Extract the last \\boxed{...} content from text."""
    boxes = re.findall(r"\\boxed\{([^}]*)\}", text)
    if boxes:
        content = boxes[-1].strip()
        match = re.search(r"[-+]?\d+(?:\.\d+)?", content)
        return match.group(0) if match else content
    # Fallback: last number in text
    numbers = re.findall(r"[-+]?\d+(?:\.\d+)?", text)
    return numbers[-1] if numbers else ""


class GSM8KEvaluator(BaseEvaluator):
    """
    GSM8K math word problem evaluator.

    Attributes used by prompts_latent.py for task detection:
        self.gsm8k = True

    Item format returned by __iter__:
        {
            "prompt_A": str,   # math problem (for sender A latent thinking)
            "prompt_B": str,   # same math problem (for receiver B answering)
            "answer":   str,   # integer string, e.g. "72"
        }
    """

    def __init__(self, n_samples: int = None):
        super().__init__()
        self.max_tokens = 1024       # Math word problems: moderate reasoning chain
        self.truncate_input = False  # GSM8K problems are short
        self.multiple_answers = False
        self.n_samples = n_samples
        self.gsm8k = True            # task-detection flag for prompts_latent.py
        self.name = "gsm8k"
        self.data = self.load_data()

    def load_data(self):
        ds = load_dataset("openai/gsm8k", "main", split="test")

        records = []
        for item in ds:
            question = item["question"].strip()
            solution = item["answer"]
            gold = _extract_gold_from_solution(solution)

            records.append({
                "prompt_A": question,   # A thinks about the problem
                "prompt_B": question,   # B also receives the problem and outputs answer
                "answer":   gold,       # integer string
            })

        if self.n_samples is not None:
            import random
            random.seed(self.random_state)
            records = random.sample(records, min(self.n_samples, len(records)))

        return records

    def evaluate_item(self, item: Dict[str, Any], response: str):
        """Numeric exact match (integer comparison)."""
        pred_str = _extract_boxed_answer(response).strip()
        gold_str = str(item["answer"]).strip()

        correct = False
        if pred_str and gold_str:
            try:
                correct = float(pred_str) == float(gold_str)
            except ValueError:
                correct = False

        self.f1_total += float(correct)
        self.f1_count += 1

    def get_result(self) -> float:
        """Returns accuracy (0.0 – 1.0)."""
        return self.f1_total / self.f1_count if self.f1_count > 0 else 0.0
