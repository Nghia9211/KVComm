import os
import glob
import re
from typing import Dict, Any
from datasets import load_dataset
from rouge import Rouge
from .base_evaluator import BaseEvaluator


class TMathEvaluator(BaseEvaluator):
    def __init__(self):
        super().__init__()
        self.max_tokens = 256
        self.truncate_input = True
        self.multiple_answers = False
        self.n_samples = 300
        self.data = self.load_data()
        self.rouge = Rouge()
        self.tmath = True
        self.name = "tmath"
        
    def load_data(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        dataset_path = os.path.join(script_dir, "data", "TMATH")
        all_files = []
        for split in ["hint_algebra", "hint_geometry", "hint_number_theory", 
                    "hint_intermediate_algebra", "hint_prealgebra", 
                    "hint_precalculus", "hint_counting_and_probability"]:
            files = glob.glob(f"{dataset_path}/{split}/*.json")
            all_files.extend(files)
        dataset = load_dataset("json", data_files=all_files, split="train")
        dataset = self.random_sample(dataset)
        dataset = dataset.rename_column("socratic_questions", "prompt_A")
        dataset = dataset.rename_column("problem", "prompt_B")
        dataset = dataset.rename_column("solution", "answer")
        return dataset

    @staticmethod
    def _extract_boxed(text: str) -> str:
        """Trích xuất nội dung \boxed{...} cuối cùng trong văn bản, hỗ trợ cả ngoặc nhọn lồng nhau."""
        results = []
        for m in re.finditer(r'\\boxed\{', text):
            start = m.end()
            depth = 1
            i = start
            while i < len(text) and depth > 0:
                if text[i] == '{':
                    depth += 1
                elif text[i] == '}':
                    depth -= 1
                i += 1
            if depth == 0:
                results.append(text[start:i - 1])
        return results[-1].strip() if results else ""

    @staticmethod
    def _normalize(s: str) -> str:
        """Chuẩn hóa chuỗi toán học để so sánh."""
        s = s.replace("\\,", "").replace("\\!", "").replace("\\ ", "")
        s = re.sub(r'\s+', '', s)
        s = s.replace("dfrac", "frac").replace("tfrac", "frac")
        s = s.lower()
        return s

    def evaluate_item(self, item: Dict[str, Any], response: str):
        if self.multiple_answers:
            answers = item['answers']
        else:
            answers = [item['answer']]

        score = 0.0
        for answer in answers:
            # 1. Trích xuất đáp án chuẩn trong \boxed{} từ bài giải mẫu
            gold_boxed = self._extract_boxed(answer)

            if gold_boxed:
                gold_norm = self._normalize(gold_boxed)
                resp_boxed = self._extract_boxed(response)
                
                # 2. Nếu model có trả về \boxed{...}, so sánh trực tiếp
                if resp_boxed:
                    if self._normalize(resp_boxed) == gold_norm:
                        score = 1.0
                        break
                
                # 3. Fallback: Kiểm tra xem đáp án chuẩn có xuất hiện trong response hay không
                if gold_norm in self._normalize(response):
                    score = 1.0
                    break
            else:
                # Nếu bài giải mẫu không chứa \boxed{}, fallback về ROUGE-L như cũ
                try:
                    scores = self.rouge.get_scores(response, answer)[0]
                    score = max(score, scores["rouge-l"]["f"])
                except Exception:
                    continue

        self.f1_total += score
        self.f1_count += 1





    
