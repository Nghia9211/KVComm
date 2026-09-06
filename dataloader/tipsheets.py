from .base_evaluator import BaseEvaluator
from datasets import load_dataset
import os


class TipsheetsEvaluator(BaseEvaluator):
    def __init__(self):
        super().__init__()
        self.max_tokens = 10
        self.truncate_input = False
        self.multiple_answers = False
        self.data = self.load_data()
        self.name = "tipsheets"
        self.configure_task_profile(prompt_family="kvcomm", task_type="qa", sender_input_mode="query_aware_context", answer_format="short_text", prompt_version="kvcomm_qa_query_aware_v2", primary_metric="legacy_match", sender_max_tokens=128)
        
    def load_data(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        dataset_path = os.path.join(script_dir, "data", "tipsheets.jsonl")
        dataset = load_dataset("json", data_files=dataset_path)["train"]
        dataset = dataset.rename_column("label", "answer")
        return dataset
