"""
medqa.py — MedQA multiple-choice evaluator for KVComm + LatentMAS

Data source: KVComm/dataloader/data/medqa.json
             (copied from LatentMAS/data/medqa.json)
             Original: https://github.com/lupantech/AgentFlow

Task format:
  prompt_A = question body (A performs latent thinking about the medical question)
  prompt_B = question + formatted A/B/C/D choices (B selects the answer)
  answer   = single lowercase letter: "a", "b", "c", or "d"

Evaluation: exact match on single letter (case-insensitive after normalize).

max_tokens = 512  (MCQ reasoning; enough for \\boxed{X} + brief reasoning)

LatentMAS recommended override (--max_tokens_B):
  N=10 : 2048
  N=20 : 1536
  N=40 : 1024
"""

import os
import re
from typing import Dict, Any

from .base_evaluator import BaseEvaluator
from datasets import load_dataset


# Choice index → uppercase letter
_IDX_TO_LETTER = {0: "A", 1: "B", 2: "C", 3: "D"}


def _format_choices(options: list) -> str:
    """Format list of option strings as 'A: <text>\nB: <text>...'"""
    lines = []
    for i, opt in enumerate(options):
        label = _IDX_TO_LETTER.get(i, chr(ord("A") + i))
        lines.append(f"{label}: {opt.strip()}")
    return "\n".join(lines)


def _find_gold_letter(options: list, raw_answer: str) -> str:
    """
    Find which option (A/B/C/D) contains raw_answer substring.
    Returns lowercase letter, e.g. 'a', 'b', 'c', 'd'.
    Falls back to raw_answer.lower() if not found.
    """
    for i, opt in enumerate(options):
        if raw_answer in opt:
            return _IDX_TO_LETTER.get(i, chr(ord("A") + i)).lower()
    return raw_answer.strip().lower()


def _extract_letter(response: str) -> str:
    """
    Extract single letter answer from model response.
    Priority:
      1. Last \\boxed{X} pattern
      2. First standalone A/B/C/D in the response
    Returns lowercase letter or empty string if not found.
    """
    # Try \boxed{X}
    boxes = re.findall(r"\\boxed\{([A-Da-d])\}", response)
    if boxes:
        return boxes[-1].strip().lower()

    # Try first standalone letter
    matches = re.findall(r"\b([A-Da-d])\b", response)
    if matches:
        return matches[-1].strip().lower()

    return ""


class MedQAEvaluator(BaseEvaluator):
    """
    MedQA multiple-choice medical question evaluator.

    Attributes used by prompts_latent.py for task detection:
        self.medqa = True

    Item format returned by __iter__:
        {
            "prompt_A": str,   # full question body (for sender A latent thinking)
            "prompt_B": str,   # question + choices (for receiver B answering)
            "answer":   str,   # single letter: "a" / "b" / "c" / "d"
        }
    """

    def __init__(self, n_samples: int = None):
        super().__init__()
        self.max_tokens = 512        # MCQ: \boxed{X} + brief reasoning chain.
                                     # 512 is the KVComm-only default; for LatentMAS with
                                     # allow_b_think=True, override via --max_tokens_B:
                                     #   N=10 → 2048, N=20 → 1536, N=40 → 1024
        self.truncate_input = True
        self.multiple_answers = False
        self.n_samples = n_samples   # None = use all samples
        self.medqa = True            # task-detection flag for prompts_latent.py
        self.name = "medqa"
        self.data = self.load_data()

    def load_data(self):
        """
        Load medqa.json. The file schema per item:
            {
              "query":   <question text>,
              "options": [<option0>, <option1>, <option2>, <option3>],
              "answer":  <full answer text that matches one of the options>
            }
        We map to KVComm's prompt_A / prompt_B / answer format.
        """
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(script_dir, "data", "medqa.json")

        ds = load_dataset("json", data_files=data_path, split="train")

        # Build records with prompt_A, prompt_B, answer
        records = []
        for item in ds:
            question = item["query"].strip()
            options = item["options"]
            raw_answer = str(item["answer"]).strip()

            # Map to letter: find which option contains the raw answer text
            gold_letter = _find_gold_letter(options, raw_answer)

            formatted_choices = _format_choices(options)
            prompt_b = f"{question}\n{formatted_choices}"

            records.append({
                "prompt_A": question,           # A thinks about the medical question
                "prompt_B": prompt_b,           # B answers with choices
                "answer": gold_letter,          # "a" / "b" / "c" / "d"
            })

        if self.n_samples is not None:
            import random
            random.seed(self.random_state)
            records = random.sample(records, min(self.n_samples, len(records)))

        return records

    def evaluate_item(self, item: Dict[str, Any], response: str):
        """
        Exact match on single letter (case-insensitive).
        Increments f1_total by 1.0 if correct, 0.0 otherwise.
        (Using f1_total/f1_count to reuse BaseEvaluator.get_result().)
        """
        pred_letter = _extract_letter(response)
        gold_letter = str(item["answer"]).strip().lower()
        correct = (pred_letter == gold_letter) and (pred_letter != "")
        self.f1_total += float(correct)
        self.f1_count += 1

    def get_result(self) -> float:
        """Returns accuracy (0.0 – 1.0)."""
        return self.f1_total / self.f1_count if self.f1_count > 0 else 0.0
