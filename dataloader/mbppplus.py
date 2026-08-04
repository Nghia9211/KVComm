"""
mbppplus.py — MBPP+ Python code generation evaluator for KVComm + LatentMAS

Data source: HuggingFace "evalplus/mbppplus" (split="test")

Task format:
  prompt_A = function description + test cases (A latent-thinks about the problem)
  prompt_B = same function description + test cases (B generates the actual code)
  answer   = test_list joined (used to execute-evaluate generated code)

Evaluation: execution-based.
  Generated python code is extracted from markdown block and executed together
  with the test cases. Uses run_with_timeout() from KVComm utils.

Notes:
  - max_tokens = 2048 (code generation needs enough space).
  - evaluate_item() overrides BaseEvaluator to run code execution instead of F1.
  - The evaluator stores OK/total; get_result() returns pass@1 accuracy.

Attributes used by prompts_latent.py:
  self.mbppplus = True
"""

import re
from typing import Dict, Any, Optional

from .base_evaluator import BaseEvaluator
from datasets import load_dataset
from utils import run_with_timeout


def _format_question(prompt: str, test_list: list) -> str:
    """
    Build the question string matching LatentMAS format.
    Shows up to 3 test cases so the model understands the expected interface.
    """
    test_examples = "\n".join(test_list[:3]) if test_list else ""
    return (
        "Please provide a self-contained Python script that solves the following "
        "problem in a markdown code block:\n"
        "```python\nYOUR_PYTHON_CODE\n```:\n"
        f"{prompt.strip()}\n"
        "Your answer will be tested on test cases like:\n"
        f"{test_examples}\n"
    )


def _extract_python_block(text: str) -> Optional[str]:
    """
    Extract the last ```python ... ``` block from model response.
    Returns the code string or None if not found.
    """
    pattern = r"```python(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
    return matches[-1].strip() if matches else None


class MBPPPlusEvaluator(BaseEvaluator):
    """
    MBPP+ code generation evaluator with execution-based scoring.

    Attributes used by prompts_latent.py for task detection:
        self.mbppplus = True

    Item format returned by __iter__:
        {
            "prompt_A": str,   # function description + test cases (for A)
            "prompt_B": str,   # same question (for B to generate code)
            "answer":   str,   # full test string used for execution
        }
    """

    def __init__(self, n_samples: int = None):
        super().__init__()
        self.max_tokens = 2048       # Code generation needs space
        self.truncate_input = True
        self.multiple_answers = False
        self.n_samples = n_samples   # None = use all test samples
        self.mbppplus = True         # task-detection flag for prompts_latent.py
        self.name = "mbppplus"
        self.data = self.load_data()
        self._exec_timeout = 10      # seconds per execution

    def load_data(self):
        """
        Load MBPP+ from HuggingFace evalplus/mbppplus.
        Schema: { "prompt": str, "test_list": list[str], "test": str, ... }
        """
        ds = load_dataset("evalplus/mbppplus", split="test")

        records = []
        for item in ds:
            prompt = item["prompt"].strip()
            test_list = item.get("test_list", [])
            # "test" is the full test harness code string used for execution
            test_code = str(item.get("test", ""))

            question = _format_question(prompt, test_list)

            records.append({
                "prompt_A": question,    # A thinks about the programming problem
                "prompt_B": question,    # B generates the actual code
                "answer": test_code,     # full test code for execution evaluation
            })

        if self.n_samples is not None:
            import random
            random.seed(self.random_state)
            records = random.sample(records, min(self.n_samples, len(records)))

        return records

    def evaluate_item(self, item: Dict[str, Any], response: str):
        """
        Execution-based evaluation.
        1. Extract ```python...``` block from model response.
        2. Concatenate with gold test code.
        3. Execute with timeout; correct = no exception raised.
        """
        pred_code = _extract_python_block(response)
        test_code = item.get("answer", "")

        if pred_code is None:
            # No code block found → incorrect
            self.f1_total += 0.0
        else:
            full_code = pred_code + "\n" + test_code
            ok, _ = run_with_timeout(full_code, timeout=self._exec_timeout)
            self.f1_total += float(ok)

        self.f1_count += 1

    def get_result(self) -> float:
        """Returns pass@1 accuracy (0.0 – 1.0)."""
        return self.f1_total / self.f1_count if self.f1_count > 0 else 0.0
