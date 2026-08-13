"""
arc_challenge.py — ARC-Challenge science MCQ evaluator for KVComm + LatentMAS

Identical to arc_easy.py but uses the ARC-Challenge subset (harder questions).
Data source: HuggingFace "allenai/ai2_arc" / "ARC-Challenge" (split="test")

Attributes used by prompts_latent.py:
  self.arc_challenge = True
"""

import re
from typing import Dict, Any

from .base_evaluator import BaseEvaluator
from datasets import load_dataset


_LABEL_MAP = {"1": "A", "2": "B", "3": "C", "4": "D",
              "A": "A", "B": "B", "C": "C", "D": "D"}


def _map_label(label: str) -> str:
    return _LABEL_MAP.get(str(label).strip().upper(), str(label).strip().upper())


def _format_choices(labels, texts) -> str:
    return "\n".join(f"{_map_label(l)}: {t.strip()}" for l, t in zip(labels, texts))


def _extract_letter(response: str) -> str:
    boxes = re.findall(r"\\boxed\{([A-Da-d])\}", response)
    if boxes:
        return boxes[-1].strip().lower()
    matches = re.findall(r"\b([A-Da-d])\b", response)
    if matches:
        return matches[-1].strip().lower()
    return ""


class ARCChallengeEvaluator(BaseEvaluator):
    """
    ARC-Challenge multiple-choice science evaluator (harder subset).

    Attributes used by prompts_latent.py for task detection:
        self.arc_challenge = True
    """

    def __init__(self, n_samples: int = None):
        super().__init__()
        self.max_tokens = 512
        self.truncate_input = True
        self.multiple_answers = False
        self.n_samples = n_samples
        self.arc_challenge = True    # task-detection flag for prompts_latent.py
        self.name = "arc_challenge"
        self.data = self.load_data()

    def load_data(self):
        ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")

        records = []
        for item in ds:
            stem = item["question"].strip()
            choices = item["choices"]
            formatted = _format_choices(choices["label"], choices["text"])
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
