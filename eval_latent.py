"""
eval_latent.py — Latent evaluator for KVComm + LatentMAS integration

Extends CommunicationEvaluator with the following changes vs KVComm original:

  1. prepare_input_ids() [OVERRIDE]:
       - Sender A: uses latent-thinker prompts from prompts_latent.py
                   tokenised with add_generation_prompt=True (preserves <think>
                   for think models)
       - Receiver B: prepends LATENT_RECEIVER_PREFIX so B knows it has
                     latent context

  2. inference() [OVERRIDE]:
       - Replaces model(input_ids_A) with latent_mas.run(input_ids_A)
       - Passes latent_only flag to control whether only latent tokens or
         full T+N tokens are forwarded to model_B
       - max_new_tokens is sourced from evaluator.max_tokens, which is set
         per-evaluator class:
           MedQAEvaluator    →  512
           MBPPPlusEvaluator → 2048
           AIME2024Evaluator → 4096
         So no CLI flag is needed to set max_tokens per task.

  3. Supported LatentMAS tasks (via dataloader flags):
       # MCQ tasks
       evaluator.medqa        → MedQAEvaluator      (MCQ medical, \\boxed{A/B/C/D})
       evaluator.arc_easy     → ARCEasyEvaluator     (MCQ science easy, \\boxed{A/B/C/D})
       evaluator.arc_challenge→ ARCChallengeEvaluator(MCQ science hard, \\boxed{A/B/C/D})
       evaluator.gpqa         → GPQAEvaluator        (MCQ graduate sci, \\boxed{A/B/C/D})
       # Math tasks
       evaluator.aime         → AIME2024/2025Evaluator (competition math, \\boxed{N})
       evaluator.gsm8k        → GSM8KEvaluator        (math word problem, \\boxed{N})
       # Code tasks
       evaluator.mbppplus     → MBPPPlusEvaluator    (code gen, ```python...```, execute)
       evaluator.humanevalplus→ HumanEvalPlusEvaluator(code gen, ```python...```, execute)

Layer importance tracking, _test, test are inherited from parent unchanged.
"""

import torch
import logging
from eval import CommunicationEvaluator, apply_chat_template, is_think_model
from models_latent import LatentMAS
from models import CVCommunicator
from prompts_latent import build_latent_sender_msg, build_latent_receiver_msg


class LatentCommunicationEvaluator(CommunicationEvaluator):
    """
    CommunicationEvaluator extended with LatentMAS thinking for sender A.

    Overrides prepare_input_ids() and inference().
    All KVComm evaluation infrastructure (_test, test, layer importance,
    truncate_input, get_response) is inherited without modification.

    Operating modes (controlled by CVCommunicator layers_list):
      Mode 1: cv.layers_list = all layers  -> no layer selection
      Mode 2: cv.layers_list = subset      -> KVComm layer selection applied

    latent_only flag (controls token selection inside LatentMAS.run):
      False: B receives A's full T_input + N_latent KV tokens
      True:  B receives only the last N_latent KV tokens (compressed thoughts)
    """

    def __init__(
        self,
        evaluator,
        tokenizer,
        use_wandb: bool,
        max_input_length: int,
        latent_mas: LatentMAS,
        cv: CVCommunicator,
        latent_only: bool = False,
        response_log_path: str = None,
    ) -> None:
        """
        Args:
            evaluator:        Task evaluator (e.g. TMathEvaluator).
            tokenizer:        Shared tokenizer for model_A and model_B.
            use_wandb:        Whether to log metrics to W&B.
            max_input_length: Maximum combined token length before truncation.
            latent_mas:       LatentMAS instance wrapping model_A.
            cv:               CVCommunicator(model_A, model_B, ...).
            latent_only:      If True, only the latent_steps KV tokens are
                              forwarded to B (discard T_input prefix).
                              Default False preserves backward-compatible behaviour.
        """
        super().__init__(evaluator, tokenizer, use_wandb, max_input_length,
                         response_log_path=response_log_path)

        # Safety: ensure the LatentMAS model IS the same object as cv.A
        assert latent_mas.model is cv.A, (
            "latent_mas.model must be the exact same Python object as cv.A.\n"
            "Initialise with:\n"
            "    latent_mas = LatentMAS(model=model_A, ...)\n"
            "    cv = CVCommunicator(model_A=model_A, model_B=model_B, ...)\n"
            "using the same model_A instance."
        )

        self.latent_mas = latent_mas
        self.latent_only = latent_only
        self.name = "latent_communication"

        # Override max_new_tokens from the evaluator's own setting.
        # Each evaluator class (MedQAEvaluator, MBPPPlusEvaluator, AIME2024Evaluator, ...)
        # already sets self.max_tokens to the right value for its task.
        # This means you do NOT need --max_new_tokens on the CLI per task.
        self.generate_args["max_new_tokens"] = evaluator.max_tokens

        logging.info(
            f"LatentCommunicationEvaluator ready: "
            f"latent_steps={latent_mas.latent_steps}, "
            f"latent_space_realign={latent_mas.latent_space_realign}, "
            f"layers_list={cv.layers_list}, "
            f"latent_only={latent_only}, "
            f"max_new_tokens={evaluator.max_tokens} (from evaluator)"
        )

    # ------------------------------------------------------------------
    # Override 1: prepare_input_ids
    # Fixes vấn đề 1 (receiver not latent-aware)
    #        vấn đề 3 (inherited prepare_input_ids not overridden)
    #        vấn đề 5 (apply_chat_template strips <think> for A)
    # ------------------------------------------------------------------

    def prepare_input_ids(self, item, model_A, model_B):
        """
        Build tokenised inputs for sender A (Latent Thinker) and receiver B
        with latent-aware prompts.

        Key differences vs CommunicationEvaluator.prepare_input_ids():

          Sender A:
            - Uses LATENT_SENDER_*_INSTRUCTION (prompts_latent.py) so A knows
              its internal reasoning will be transferred to another agent.
            - Tokenised with tokenizer.apply_chat_template(add_generation_prompt=True)
              directly, WITHOUT the context=True stripping from eval.apply_chat_template.
              This preserves the <think> token for think-models so the latent loop
              starts from a proper thinking state (fix vấn đề 5).

          Receiver B:
            - Uses LATENT_RECEIVER_PREFIX + original KVComm B template so B
              knows latent context exists and to ignore irrelevant parts
              (fix vấn đề 1).
            - Tokenised via eval.apply_chat_template (unchanged behaviour for B).

        Args:
            item:    Dataset item dict with "prompt_A", "prompt_B".
            model_A: Sender model (cv.A).
            model_B: Receiver model (cv.B).

        Returns:
            (input_ids_A, input_ids_B): after truncate_input().
        """
        # ── Sender A: Latent Thinker prompt ──────────────────────────────
        msg_A = build_latent_sender_msg(self.evaluator, item, is_think=is_think_model(model_A))
        # Tokenise A with add_generation_prompt=True so the assistant prefix is preserved.
        # For think-models (e.g. Qwen3): apply_chat_template adds <think>, which lets A
        #   enter thinking mode during the latent forward pass.
        # For non-think models (e.g. Llama-3): apply_chat_template adds normal prefix.
        # We do NOT call eval.apply_chat_template(..., context=True) here because
        # context=True strips <think>, preventing thinking mode in A.
        input_ids_A = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": msg_A}],
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(model_A.device)

        # For think-models: append <think> token(s) so latent loop starts from thinking state.
        # For non-think models: no prefix added (Llama, etc. don't have <think>).
        # Note: convert_tokens_to_ids returns None when <think> is not a single special
        # token (e.g. Llama-based distills tokenize it as multiple pieces). In that case
        # fall back to encode(), which handles both single- and multi-token representations.
        if is_think_model(model_A):
            think_token_id = self.tokenizer.convert_tokens_to_ids("<think>")
            if think_token_id is None:
                # <think> is not a single special token — encode as a token sequence
                think_ids = self.tokenizer.encode("<think>", add_special_tokens=False)
                think_tensor = torch.tensor([think_ids], device=model_A.device)
            else:
                think_tensor = torch.tensor([[think_token_id]], device=model_A.device)
            input_ids_A = torch.cat([input_ids_A, think_tensor], dim=-1)

        # ── Receiver B: latent-aware prompt ──────────────────────────────
        msg_B = build_latent_receiver_msg(self.evaluator, item)
        # Use eval.apply_chat_template (unchanged) for B — handles think-model
        # prefix stripping / "The answer is:" injection as original KVComm does.
        input_ids_B = apply_chat_template(
            self.evaluator, self.tokenizer, msg_B, model_B
        )

        # ── Truncation (inherited logic) ──────────────────────────────────
        input_ids_A, input_ids_B = self.truncate_input(input_ids_A, input_ids_B)
        return input_ids_A, input_ids_B

    # ------------------------------------------------------------------
    # Override 2: inference
    # Uses updated prepare_input_ids + latent_only flag (fix vấn đề 7)
    # ------------------------------------------------------------------

    def inference(self, model, cv, item):
        """
        Run one inference sample using LatentMAS for the sender.

        Changes vs CommunicationEvaluator.inference():
          1. prepare_input_ids() now uses latent-aware prompts (overridden above).
          2. Sender uses latent_mas.run() instead of model(input_ids_A).
          3. latent_only flag controls whether B receives only latent KV tokens.

        Args:
            model: model_A (passed by _test(), kept for API compatibility).
                   Not used directly; LatentMAS wraps model_A internally.
            cv:    CVCommunicator(model_A, model_B, ...).
            item:  Dataset item dict with "prompt_A", "prompt_B", "answer".

        Returns:
            str: decoded response from model_B.
        """
        # ── Input preparation (now uses overridden prepare_input_ids) ─────
        input_ids_A, input_ids_B = self.prepare_input_ids(item, cv.A, cv.B)

        # ── Latent thinking ───────────────────────────────────────────────
        # Replace: out_A = model(input_ids_A, use_cache=True)
        # With:    latent loop on model_A → DynamicCache
        #
        # latent_only=True  → DynamicCache[i]: [1, kv_heads, N,   head_dim]
        # latent_only=False → DynamicCache[i]: [1, kv_heads, T+N, head_dim]
        latent_past_kv = self.latent_mas.run(
            input_ids_A,
            attention_mask=torch.ones_like(input_ids_A),
            latent_only=self.latent_only,
        )

        # ── FIX: prepend past_mask cho attention_mask của B ───────────────
        # LatentMAS gốc (models.py L244-252) luôn prepend một mask có shape
        # [B, past_len] trước attention_mask của judger/receiver. Nếu bỏ bước
        # này, HuggingFace sẽ build causal_mask sai: B chỉ "nhìn thấy" T_B
        # tokens của chính nó, không attend đúng vào A's KV cache.
        # Điều này đặc biệt nghiêm trọng ở latent_steps cao (N=20) vì
        # past_len = T_A + N lớn hơn nhiều, làm attention collapse hoàn toàn.
        past_len = latent_past_kv.get_seq_length()
        if past_len > 0:
            past_mask = torch.ones(
                (input_ids_B.shape[0], past_len),
                dtype=torch.long,
                device=input_ids_B.device,
            )
            attention_mask_B = torch.cat(
                [past_mask, torch.ones_like(input_ids_B)], dim=-1
            )
        else:
            attention_mask_B = torch.ones_like(input_ids_B)

        # ── Generation ────────────────────────────────────────────────────
        # cv.generate() → prepare_key_cache(latent_past_kv)
        #               → layer selection (Mode 2) or identity (Mode 1)
        #               → model_B.generate với attention_mask_B đúng
        output = cv.generate(
            input_ids_B,
            attention_mask=attention_mask_B,  # ← Bao gồm cả T_A + N_latent tokens
            out_A_past_key_values=latent_past_kv,
            **self.generate_args,
        )[0]

        context_length = input_ids_B.shape[-1]
        return self.get_response(output, context_length)
