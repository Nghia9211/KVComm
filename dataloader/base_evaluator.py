from utils import f1_match
from utils.longbench_metrics import qa_f1_score
from typing import Dict, Any
import random

class BaseEvaluator:
    _VALID_PROFILE_VALUES = {
        "prompt_family": {"kvcomm", "latentmas"},
        "task_type": {"qa", "math", "code", "summarization"},
        "sender_input_mode": {"query_aware_context", "shared_problem", "native_split"},
        "answer_format": {"short_text", "boxed_integer", "boxed_choice", "python", "summary"},
        "primary_metric": {"legacy_match", "longbench_qa_f1"},
    }

    def __init__(self, random_state=42):
        self.index = 0
        self.f1_total = 0.0
        self.f1_count = 0
        self.max_tokens = None
        self.truncate_input = None
        self.multiple_answers = None
        self.n_samples = None
        self.random_state = random_state
        # Conservative defaults for the original KVComm QA datasets.  Every
        # concrete evaluator overrides these through configure_task_profile().
        self.prompt_family = "kvcomm"
        self.task_type = "qa"
        self.sender_input_mode = "query_aware_context"
        self.answer_format = "short_text"
        self.prompt_version = "kvcomm_qa_query_aware_v2"
        self.primary_metric = "legacy_match"
        self.sender_max_tokens = 256
        self.metric_totals = {"legacy_match": 0.0, "longbench_f1": 0.0}

    def configure_task_profile(self, **overrides):
        for key, value in overrides.items():
            if key not in self._VALID_PROFILE_VALUES and key not in {"prompt_version", "sender_max_tokens"}:
                raise ValueError(f"Unsupported task-profile field: {key}")
            setattr(self, key, value)
        self.validate_task_profile()

    def validate_task_profile(self):
        for field, allowed in self._VALID_PROFILE_VALUES.items():
            value = getattr(self, field, None)
            if value not in allowed:
                raise ValueError(f"Unsupported {field}={value!r}; expected one of {sorted(allowed)}")
        if not isinstance(self.prompt_version, str) or not self.prompt_version:
            raise ValueError("prompt_version must be a non-empty string")
        if not isinstance(self.sender_max_tokens, int) or self.sender_max_tokens <= 0:
            raise ValueError("sender_max_tokens must be a positive integer")

    def load_data(self):
        pass

    def __len__(self):
        return len(self.data)
    
    def __iter__(self):
        self.index = 0
        self.f1_total = 0.0
        self.f1_count = 0
        self.metric_totals = {"legacy_match": 0.0, "longbench_f1": 0.0}
        return self
    
    def __next__(self):
        if self.index < len(self):
            value = self.data[self.index]
            self.index += 1
            return value
        else:
            raise StopIteration

    def evaluate_item(self, item: Dict[str, Any], response: str):
        if self.multiple_answers:
            answers = item['answers']
        else:
            answers = [item['answer']]
        legacy_match = max((float(f1_match(answer, response)) for answer in answers), default=0.0)
        longbench_f1 = qa_f1_score(response, answers)
        item_metrics = {
            "legacy_match": legacy_match,
            "longbench_f1": longbench_f1,
        }
        self.metric_totals["legacy_match"] += legacy_match
        self.metric_totals["longbench_f1"] += longbench_f1
        primary_value = longbench_f1 if self.primary_metric == "longbench_qa_f1" else legacy_match
        self.f1_total += primary_value
        self.f1_count += 1
        return item_metrics
    
    def get_result(self) -> float:
        return self.f1_total / self.f1_count if self.f1_count > 0 else 0.0

    def get_results(self) -> Dict[str, float]:
        if self.f1_count == 0:
            return {"primary_metric": self.primary_metric, "primary": 0.0}
        results = {
            "primary_metric": self.primary_metric,
            "primary": self.get_result(),
        }
        if self.task_type == "qa":
            results.update({
                "legacy_accuracy": self.metric_totals["legacy_match"] / self.f1_count,
                "longbench_f1": self.metric_totals["longbench_f1"] / self.f1_count,
            })
        return results
    
    def random_sample(self, dataset):
        if self.n_samples is None:
            return dataset
        else:
            return dataset.shuffle(seed=self.random_state).select(range(self.n_samples))
