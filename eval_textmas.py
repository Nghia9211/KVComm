"""Independent sequential text-channel baseline; re-exported by eval_latent."""

import json
import time
import logging
import torch
from tqdm import tqdm
from eval import apply_chat_template, is_think_model
from prompts_latent import build_latent_sender_msg, build_text_receiver_msg
from utils.response_logging import build_response_record
from utils.evaluation_config import resolve_textmas_budgets


class TextMASEvaluator:
    """
    TextMAS: Sequential 2-agent text-based baseline, faithful to the LatentMAS paper
    (arXiv 2511.20639, Section 4 "Models and Baselines").

    Paper definition — "Sequential TextMAS":
      Following the chain-of-agents design, each agent performs full text-based
      CoT reasoning and communication; the output of each agent is directly
      appended to the input of the next agent.

    2-agent pipeline (our instantiation of the paper's 4-agent chain):

      Step 1 — Agent A (Thinker):
        - Prompt:   build_latent_sender_msg()   [same as LatentMAS condition]
        - Thinking: FULLY ALLOWED (no </think> suppression)
        - Budget:   evaluator.max_tokens (uncapped, like paper)
        - Output:   full_response_A  (complete CoT + answer text)

      Step 2 — Agent B (Solver):
        - Prompt:   build_latent_receiver_msg() + "\\n\\nAgent A's reasoning:\\n{full_response_A}"
        - Thinking: FULLY ALLOWED
        - Budget:   evaluator.max_tokens
        - Output:   final answer

    Key differences from NLD (the old incorrect implementation):
      ✓ No </think> suppression for A or B
      ✓ A generates full CoT (no 128-token cap)
      ✓ No REFINE_TMPL / B-Phase-1 / debate loop
      ✓ A's full output is appended directly to B's input context
      ✓ B generates once only (not refine-style)

    This enables a fair apples-to-apples comparison:
      TextMAS (text channel, full CoT)  vs  LatentMAS (KV-cache channel, latent steps)

    Args:
        evaluator:         Task evaluator (e.g. MedQAEvaluator).
        tokenizer:         Shared tokenizer.
        use_wandb:         Whether to log metrics to W&B.
        max_input_length:  Maximum token length for B's input before truncation.
        response_log_path: Path to JSONL file for logging per-item responses.
    """

    # Template used to prepend A's reasoning to B's input.
    # Kept minimal so B sees A's thoughts as pure context, not a debate prompt.
    _A_CONTEXT_PREFIX = "Agent A's reasoning:\n{response_A}\n\n"

    def __init__(
        self,
        evaluator,
        tokenizer,
        use_wandb: bool,
        max_input_length: int,
        allow_b_think: bool = False,
        max_tokens_A: int = 0,
        max_tokens_B: int = 0,
        response_log_path: str = None,
    ):
        from eval import apply_chat_template, is_think_model
        self._apply_chat_template = apply_chat_template
        self._is_think_model = is_think_model

        self.evaluator = evaluator
        self.tokenizer = tokenizer
        self.use_wandb = use_wandb
        self.max_input_length = max_input_length
        self.allow_b_think = allow_b_think
        self.response_log_path = response_log_path
        self.name = "textmas"

        self.effective_max_tokens_A, self.effective_max_tokens_B = resolve_textmas_budgets(
            evaluator, max_tokens_A, max_tokens_B
        )

        # Sampling params aligned with LatentMAS paper (Section 4):
        # temperature=0.6, top_p=0.95 — same as LatentCommunicationEvaluator.
        common_generate_args = {
            "temperature":    0.6,
            "top_p":          0.95,
            "top_k":          None,
            "num_beams":      1,
            "do_sample":      True,
        }
        self.generate_args_A = {**common_generate_args, "max_new_tokens": self.effective_max_tokens_A}
        self.generate_args_B = {**common_generate_args, "max_new_tokens": self.effective_max_tokens_B}

        logging.info(
            f"TextMASEvaluator ready: "
            f"prompt_family={evaluator.prompt_family}, prompt_version={evaluator.prompt_version}, "
            f"max_tokens_A={self.effective_max_tokens_A}, max_tokens_B={self.effective_max_tokens_B}, "
            f"allow_b_think={allow_b_think}, temperature=0.6, top_p=0.95"
        )

    # ------------------------------------------------------------------
    # Input preparation
    # ------------------------------------------------------------------

    def _prepare_input_ids_A(self, item, model_A):
        """
        Build tokenised input for Agent A.

        Uses build_latent_sender_msg() — the same framing as LatentMAS.
        Agent A ALWAYS has allow_b_think=True so that it generates a full reasoning CoT,
        matching the latent thinking capability of Sender A in LatentMAS.
        """
        msg_A = build_latent_sender_msg(
            self.evaluator, item, is_think=self._is_think_model(model_A)
        )
        input_ids_A = self._apply_chat_template(
            self.evaluator, self.tokenizer, msg_A, model_A,
            context=False, allow_b_think=True,
        )
        return input_ids_A

    def _prepare_input_ids_B(self, item, response_A, model_B):
        """
        Build tokenised input for Agent B.

        B receives:
          build_text_receiver_msg() with Agent A's reasoning naturally formatted
          between role introduction and target question.
          ← sequential natural language communication baseline.

        Respects self.allow_b_think for B as well.

        Truncation: if the combined prompt exceeds max_input_length, we
        truncate the middle (same strategy as CommunicationEvaluator).
        """
        msg_B = build_text_receiver_msg(
            self.evaluator, item, response_A=response_A, allow_b_think=self.allow_b_think
        )

        input_ids_B = self._apply_chat_template(
            self.evaluator, self.tokenizer, msg_B, model_B,
            context=False, allow_b_think=self.allow_b_think,
        )

        # Truncate in the middle if over budget
        if input_ids_B.shape[-1] > self.max_input_length and self.evaluator.truncate_input:
            half = self.max_input_length // 2
            input_ids_B = torch.cat(
                [input_ids_B[:, :half], input_ids_B[:, -half:]], dim=-1
            )
        return input_ids_B

    # ------------------------------------------------------------------
    # Response decoding
    # ------------------------------------------------------------------

    def get_response(self, output, context_length):
        """
        Decode output tokens after context_length.

        For think-models: strip <think>...</think> before returning to
        the evaluator so scoring runs on the clean final answer only.
        (Same fix as LatentCommunicationEvaluator.get_response.)
        """
        response = self.tokenizer.decode(
            output[context_length:], skip_special_tokens=True
        ).strip()
        # Strip thinking trace — present when allow_b_think=True
        if "</think>" in response:
            after_think = response.split("</think>", 1)[1].strip()
            return after_think if after_think else response
        return response

    # ------------------------------------------------------------------
    # Single-sample inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def inference(self, model_A, model_B, item):
        """
        Run one TextMAS inference step.

        Step 1: Agent A processes its prompt and generates a full CoT response.
        Step 2: Agent B receives A's full response as context and generates
                the final answer.
        """
        input_ids_A = self._prepare_input_ids_A(item, model_A)

        # ── Step 1: A generates full CoT (thinking fully enabled) ─────────
        output_A = model_A.generate(
            input_ids_A,
            attention_mask=torch.ones_like(input_ids_A),
            **self.generate_args_A,
        )[0]
        response_A = self.tokenizer.decode(
            output_A[input_ids_A.shape[-1]:], skip_special_tokens=True
        ).strip()

        # ── Step 2: B generates final answer with A's full output ──────────
        input_ids_B = self._prepare_input_ids_B(item, response_A, model_B)
        output_B = model_B.generate(
            input_ids_B,
            attention_mask=torch.ones_like(input_ids_B),
            **self.generate_args_B,
        )[0]
        response_B = self.get_response(output_B, input_ids_B.shape[-1])
        return {
            "response_A": response_A,
            "response_B": response_B,
            "generated_tokens_A": int(output_A.shape[-1] - input_ids_A.shape[-1]),
            "generated_tokens_B": int(output_B.shape[-1] - input_ids_B.shape[-1]),
            "model_A_prompt": self.tokenizer.decode(input_ids_A[0], skip_special_tokens=False),
            "model_B_prompt": self.tokenizer.decode(input_ids_B[0], skip_special_tokens=False),
        }

    # ------------------------------------------------------------------
    # Evaluation loop
    # ------------------------------------------------------------------

    def _test(self, model_A, model_B, limit=None):
        items_all = list(self.evaluator)
        if limit is not None:
            items_all = items_all[:limit]

        progress_bar = tqdm(items_all, desc=f"{self.name} result: 0.0000")

        response_log_file = None
        if self.response_log_path:
            response_log_file = open(self.response_log_path, "a", encoding="utf-8")

        try:
            for i, item in enumerate(progress_bar):
                try:
                    inference_result = self.inference(model_A, model_B, item)
                except Exception as e:
                    logging.error(f"TextMAS inference error at item {i}: {e}")
                    continue

                response = inference_result["response_B"]
                item_metrics = self.evaluator.evaluate_item(item, response) or {}

                result = self.evaluator.get_result()
                progress_bar.set_description(f"{self.name} {self.evaluator.primary_metric}: {result:.4f}")

                if response_log_file is not None:
                    record = build_response_record(
                        idx=i, evaluator=self.evaluator, item=item, method="textmas_two_agent",
                        response=response, response_a=inference_result["response_A"],
                        model_a_prompt=inference_result["model_A_prompt"],
                        model_b_prompt=inference_result["model_B_prompt"],
                        item_metrics=item_metrics, aggregate_metrics=self.evaluator.get_results(),
                        max_tokens_a=self.effective_max_tokens_A, max_tokens_b=self.effective_max_tokens_B,
                        generated_tokens_a=inference_result["generated_tokens_A"],
                        generated_tokens_b=inference_result["generated_tokens_B"],
                        communication_type="text",
                    )
                    response_log_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                    response_log_file.flush()

        finally:
            if response_log_file is not None:
                response_log_file.close()

        return self.evaluator.get_result()

    @torch.no_grad()
    def test(self, model_A, model_B, limit=None):
        tic = time.time()
        result = self._test(model_A, model_B, limit)
        toc = time.time()
        time_used = toc - tic
        if self.use_wandb:
            import wandb
            wandb.log({f"{self.name}_{key}": value for key, value in self.evaluator.get_results().items() if isinstance(value, (int, float))})
            wandb.log({f"{self.name}_time": time_used})
        logging.info(f"{self.name} result: {result:.4f}, {self.name} time: {time_used:.2f}s")
        return result
