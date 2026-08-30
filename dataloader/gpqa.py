"""
gpqa.py — GPQA Diamond science MCQ evaluator for KVComm + LatentMAS

Data source: HuggingFace "fingertap/GPQA-Diamond" (split="test", ~198 samples)

Task format:
  prompt_A = question with choices (A performs latent thinking)
  prompt_B = question with choices (B selects correct answer)
  answer   = single lowercase letter: "a", "b", "c", or "d"

Evaluation: exact match on single letter (case-insensitive after normalize).

Notes:
  - GPQA Diamond is a graduate-level science MCQ benchmark.
  - The "question" field in the dataset already includes answer options.
  - The "answer" field is the full text of the correct option; we map it
    back to a letter by matching against the option texts.
  - max_tokens = 1024 (MCQ with reasoning; graduate-level needs longer chains).
  - LatentMAS recommended override (--max_tokens_B):
      N=10 → 4096, N=20 → 2048, N=40 → 1536

Attributes used by prompts_latent.py:
  self.gpqa = True
"""

import re
from typing import Dict, Any

from .base_evaluator import BaseEvaluator
from datasets import load_dataset


_IDX_TO_LETTER = {0: "A", 1: "B", 2: "C", 3: "D"}


def _extract_letter(response: str) -> str:
    """
    Extract single letter answer from model response.
    Priority:
      1. Last \\boxed{X} pattern
      2. Last standalone A/B/C/D in the response
    Returns lowercase letter or empty string.
    """
    boxes = re.findall(r"\\boxed\{([A-Da-d])\}", response)
    if boxes:
        return boxes[-1].strip().lower()
    matches = re.findall(r"\b([A-Da-d])\b", response)
    if matches:
        return matches[-1].strip().lower()
    return ""


class GPQAEvaluator(BaseEvaluator):
    """
    GPQA Diamond science MCQ evaluator.

    Attributes used by prompts_latent.py for task detection:
        self.gpqa = True

    Item format returned by __iter__:
        {
            "prompt_A": str,   # question with options (for A latent thinking)
            "prompt_B": str,   # question with options (for B answering)
            "answer":   str,   # single letter: "a" / "b" / "c" / "d"
        }
    """

    def __init__(self, n_samples: int = None):
        super().__init__()
        self.max_tokens = 1024      # Graduate-level science MCQ: needs 1024+ for reasoning.
                                     # For LatentMAS allow_b_think=True, use --max_tokens_B:
                                     #   N=10 → 4096, N=20 → 2048, N=40 → 1536
        self.truncate_input = True
        self.multiple_answers = False
        self.n_samples = n_samples
        self.gpqa = True             # task-detection flag for prompts_latent.py
        self.name = "gpqa"
        self.data = self.load_data()

    def load_data(self):
        ds = load_dataset("fingertap/GPQA-Diamond", split="test")

        records = []
        for item in ds:
            question = item["question"].strip()
            # "answer" field is the correct letter (A/B/C/D) in this dataset
            raw_answer = str(item["answer"]).strip()
            gold_letter = raw_answer.lower()

            records.append({
                "prompt_A": question,       # A thinks about the question
                "prompt_B": question,       # B answers with choices
                "answer":   gold_letter,    # "a" / "b" / "c" / "d"
            })

        if self.n_samples is not None:
            import random
            random.seed(self.random_state)
            records = random.sample(records, min(self.n_samples, len(records)))

        return records

    def evaluate_item(self, item: Dict[str, Any], response: str):
        pred_letter = _extract_letter(response)
        gold_letter = str(item["answer"]).strip().lower()
        correct = (pred_letter == gold_letter) and (pred_letter != "")
        self.f1_total += float(correct)
        self.f1_count += 1

    def get_result(self) -> float:
        return self.f1_total / self.f1_count if self.f1_count > 0 else 0.0
