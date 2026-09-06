"""Dependency-free English QA metrics compatible with LongBench."""

import re
import string
from collections import Counter
from typing import Iterable


def normalize_answer(text: str) -> str:
    text = str(text).lower()
    text = "".join(char for char in text if char not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def qa_f1_single(prediction: str, ground_truth: str) -> float:
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(prediction_tokens)
    recall = overlap / len(ground_truth_tokens)
    return 2 * precision * recall / (precision + recall)


def qa_f1_score(prediction: str, ground_truths: Iterable[str]) -> float:
    references = list(ground_truths)
    return max((qa_f1_single(prediction, answer) for answer in references), default=0.0)
