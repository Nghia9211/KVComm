"""
humanevalplus.py — HumanEval+ Python code generation evaluator for KVComm + LatentMAS

Data source: HuggingFace "evalplus/humanevalplus" (split="test")

Task format:
  prompt_A = function signature + docstring (A performs latent thinking)
  prompt_B = same question (B generates the Python implementation)
  answer   = test harness code (used to execute-evaluate generated code)

Evaluation: execution-based (same pattern as mbppplus.py).
  Generated code is extracted from ```python...``` block and executed
  with the test harness.

Notes:
  - max_tokens = 2048 (code generation needs space).
  - The test harness calls check(entry_point) to validate the solution.

Attributes used by prompts_latent.py:
  self.humanevalplus = True
"""

import re
from typing import Dict, Any, Optional

from .base_evaluator import BaseEvaluator
from datasets import load_dataset
from utils import run_with_timeout


def _format_question(prompt: str) -> str:
    return (
        "Please provide a self-contained Python script that solves the "
        "following problem in a markdown code block:\n"
        "```python\nYOUR_PYTHON_CODE\n```:\n"
        f"{prompt.strip()}\n"
    )


def _extract_python_block(text: str) -> Optional[str]:
    pattern = r"```python(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
    return matches[-1].strip() if matches else None


class HumanEvalPlusEvaluator(BaseEvaluator):
    """
    HumanEval+ code generation evaluator with execution-based scoring.

    Attributes used by prompts_latent.py for task detection:
        self.humanevalplus = True

    Item format returned by __iter__:
        {
            "prompt_A": str,   # question prompt (for A latent thinking)
            "prompt_B": str,   # same question (for B to generate code)
            "answer":   str,   # full test harness code for execution evaluation
        }
    """

    def __init__(self, n_samples: int = None):
        super().__init__()
        self.max_tokens = 2048
        self.truncate_input = True
        self.multiple_answers = False
        self.n_samples = n_samples
        self.humanevalplus = True    # task-detection flag for prompts_latent.py
        self.name = "humanevalplus"
        self.data = self.load_data()
        self._exec_timeout = 10

    def load_data(self):
        ds = load_dataset("evalplus/humanevalplus", split="test")

        records = []
        for item in ds:
            prompt = item["prompt"].strip()
            entry_point = item["entry_point"]

            question = _format_question(prompt)

            # Build test harness: replace 'candidate' with actual function name
            raw_test = str(item.get("test", ""))
            test_code = raw_test.replace("candidate", entry_point)
            test_code += f"\n\ncheck({entry_point})"

            records.append({
                "prompt_A": question,
                "prompt_B": question,
                "answer":   test_code,
            })

        if self.n_samples is not None:
            import random
            random.seed(self.random_state)
            records = random.sample(records, min(self.n_samples, len(records)))

        return records

    def evaluate_item(self, item: Dict[str, Any], response: str):
        pred_code = _extract_python_block(response)
        test_code = item.get("answer", "")

        if pred_code is None:
            self.f1_total += 0.0
        else:
            full_code = pred_code + "\n" + test_code
            ok, _ = run_with_timeout(full_code, timeout=self._exec_timeout)
            self.f1_total += float(ok)

        self.f1_count += 1

    def get_result(self) -> float:
        """Returns pass@1 accuracy (0.0 – 1.0)."""
        return self.f1_total / self.f1_count if self.f1_count > 0 else 0.0
