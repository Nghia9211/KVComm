"""
arc_easy.py — ARC-Easy science MCQ evaluator for KVComm + LatentMAS

Data source: HuggingFace "allenai/ai2_arc" / "ARC-Easy" (split="test")

Task format:
  prompt_A = question + formatted A/B/C/D choices
  prompt_B = same question + choices
  answer   = single lowercase letter: "a", "b", "c", or "d"

Evaluation: exact match on single letter (case-insensitive).
max_tokens = 512.

Attributes used by prompts_latent.py:
  self.arc_easy = True
"""

import re
from typing import Dict, Any

from .base_evaluator import BaseEvaluator
from datasets import load_dataset


_IDX_TO_LETTER = {0: "A", 1: "B", 2: "C", 3: "D"}

# ARC datasets use labels 1,2,3,4 or A,B,C,D interchangeably
_LABEL_MAP = {"1": "A", "2": "B", "3": "C", "4": "D",
              "A": "A", "B": "B", "C": "C", "D": "D"}


def _map_label(label: str) -> str:
    return _LABEL_MAP.get(str(label).strip().upper(), str(label).strip().upper())


def _format_choices(labels, texts) -> str:
    lines = []
    for label, text in zip(labels, texts):
        mapped = _map_label(label)
        lines.append(f"{mapped}: {text.strip()}")
    return "\n".join(lines)


def _extract_letter(response: str) -> str:
    boxes = re.findall(r"\\boxed\{([A-Da-d])\}", response)
    if boxes:
        return boxes[-1].strip().lower()
    matches = re.findall(r"\b([A-Da-d])\b", response)
    if matches:
        return matches[-1].strip().lower()
    return ""


class ARCEasyEvaluator(BaseEvaluator):
    """
    ARC-Easy multiple-choice science evaluator.

    Attributes used by prompts_latent.py for task detection:
        self.arc_easy = True

    Item format returned by __iter__:
        {
            "prompt_A": str,   # question + choices (for A latent thinking)
            "prompt_B": str,   # question + choices (for B answering)
            "answer":   str,   # single letter: "a" / "b" / "c" / "d"
        }
    """

    def __init__(self, n_samples: int = None):
        super().__init__()
        self.max_tokens = 512
        self.truncate_input = True
        self.multiple_answers = False
        self.n_samples = n_samples
        self.arc_easy = True         # task-detection flag for prompts_latent.py
        self.name = "arc_easy"
        self.data = self.load_data()

    def load_data(self):
        ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split="test")

        records = []
        for item in ds:
            stem = item["question"].strip()
            choices = item["choices"]
            labels = choices["label"]
            texts  = choices["text"]

            formatted = _format_choices(labels, texts)
            question = stem + "\n" + formatted

            raw_answer = str(item.get("answerKey", "")).strip()
            gold_letter = _map_label(raw_answer).lower()

            records.append({
                "prompt_A": question,
                "prompt_B": question,
                "answer":   gold_letter,
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
