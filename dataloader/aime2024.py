"""
aime2024.py — AIME 2024 competition math evaluator for KVComm + LatentMAS

Data source: HuggingFace "HuggingFaceH4/aime_2024" (split="train", 30 samples)

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

Attributes used by prompts_latent.py:
  self.aime = True
"""

import re
from typing import Dict, Any

from .base_evaluator import BaseEvaluator
from datasets import load_dataset


def _extract_boxed_answer(text: str) -> str:
    """
    Extract the last \\boxed{...} content from text.
    Returns the numeric string or empty string if not found.
    """
    boxes = re.findall(r"\\boxed\{([^}]*)\}", text)
    if not boxes:
        return ""
    content = boxes[-1].strip()
    # Extract numeric part
    match = re.search(r"[-+]?\d+(?:\.\d+)?", content)
    return match.group(0) if match else content


def _normalize(ans: str) -> str:
    return ans.strip().lower() if ans else ""


class AIME2024Evaluator(BaseEvaluator):
    """
    AIME 2024 competition math evaluator.

    Attributes used by prompts_latent.py for task detection:
        self.aime = True

    Item format returned by __iter__:
        {
            "prompt_A": str,   # competition problem (same as prompt_B)
            "prompt_B": str,   # competition problem
            "answer":   str,   # integer string, e.g. "42"
        }
    """

    def __init__(self, n_samples: int = None):
        super().__init__()
        self.max_tokens = 4096       # AIME requires long reasoning chains
        self.truncate_input = False  # AIME problems are short; no truncation needed
        self.multiple_answers = False
        self.n_samples = n_samples   # None = use all 30 samples
        self.aime = True             # task-detection flag for prompts_latent.py
        self.name = "aime2024"
        self.data = self.load_data()

    def load_data(self):
        """
        Load AIME 2024 from HuggingFace.
        Schema: { "problem": str, "answer": int/str, ... }
        """
        ds = load_dataset("HuggingFaceH4/aime_2024", split="train")

        records = []
        for item in ds:
            problem = item["problem"].strip()
            answer_raw = str(item["answer"]).strip()

            records.append({
                "prompt_A": problem,    # A thinks deeply about the problem
                "prompt_B": problem,    # B also receives the problem and outputs answer
                "answer": answer_raw,   # integer string
            })

        if self.n_samples is not None:
            import random
            random.seed(self.random_state)
            records = random.sample(records, min(self.n_samples, len(records)))

        return records

    def evaluate_item(self, item: Dict[str, Any], response: str):
        """
        Integer exact match.
        Parses both pred and gold as int; any parse failure → incorrect.
        """
        pred_str = _normalize(_extract_boxed_answer(response))
        gold_str = _normalize(str(item["answer"]))

        correct = False
        if pred_str and gold_str:
            try:
                correct = int(pred_str) == int(gold_str)
            except ValueError:
                # If boxed content has no integer, try last number in full text
                nums = re.findall(r"[-+]?\d+(?:\.\d+)?", response)
                if nums:
                    try:
                        correct = int(nums[-1]) == int(gold_str)
                    except ValueError:
                        correct = False

        self.f1_total += float(correct)
        self.f1_count += 1

    def get_result(self) -> float:
        """Returns accuracy (0.0 – 1.0)."""
        return self.f1_total / self.f1_count if self.f1_count > 0 else 0.0
