from .base_evaluator import BaseEvaluator
from datasets import load_dataset


class MultiFieldQAEnEvaluator(BaseEvaluator):
    def __init__(self):
        super().__init__()
        self.max_tokens = 64
        self.truncate_input = True
        self.multiple_answers = True
        self.data = self.load_data()
        self.name = "multifieldqa_en"
        self.configure_task_profile(prompt_family="kvcomm", task_type="qa", sender_input_mode="query_aware_context", answer_format="short_text", prompt_version="kvcomm_qa_query_aware_v2", primary_metric="longbench_qa_f1", sender_max_tokens=256)
        
    def load_data(self):
        dataset = load_dataset('Xnhyacinth/LongBench', split='test', name='multifieldqa_en')
        dataset = dataset.map(lambda x: {
            "prompt_A": x["context"], 
            "prompt_B": x["question"], 
        })
        return dataset
