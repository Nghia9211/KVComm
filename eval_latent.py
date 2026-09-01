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

  4. TextMASEvaluator (TextMAS baseline):
       Natural-language baseline for LatentMAS — sender A generates a short
       text summary, receiver B reads it and produces the final answer.
       Uses the same LatentMAS-task prompts as LatentCommunicationEvaluator
       for a fair comparison. Inherits from NLDEvaluator (eval.py).
"""

import torch
import logging
from eval import CommunicationEvaluator, apply_chat_template, is_think_model
from layer_importance import calc_layer_importance
from models_latent import LatentMAS
from models import CVCommunicator
from prompts_latent import build_latent_sender_msg, build_latent_receiver_msg


class LatentCommunicationEvaluator(CommunicationEvaluator):
    """
    CommunicationEvaluator extended with LatentMAS thinking for sender A.

    Overrides prepare_input_ids(), inference(), and get_response().
    All other KVComm evaluation infrastructure (_test, test, layer importance,
    truncate_input) is inherited without modification.

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
        allow_b_think: bool = False,
        max_tokens_B: int = 0,
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
            allow_b_think:    If True, allows receiver B to think instead of suppressing
                              thinking with </think>\n\nThe answer is: (Fix Bug B1).
                              Only applies in LatentMAS mode (this evaluator).
            max_tokens_B:     Maximum new tokens B is allowed to generate.
                              0 (default) = use evaluator.max_tokens (task default).
                              Set > 0 to override, e.g. 4096 when allow_b_think=True
                              so B has headroom for chain-of-thought before answering.
                              Only applies in LatentMAS mode (this evaluator).
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
        self.allow_b_think = allow_b_think
        self.name = "latent_communication"

        # ── Max output tokens for B (LatentMAS-only override) ──────────────
        # When allow_b_think=True, B needs extra headroom for <think>…</think>
        # before the actual answer. Use max_tokens_B > 0 to grant that headroom.
        # When max_tokens_B == 0, fall back to the task evaluator's default.
        effective_max_tokens = max_tokens_B if max_tokens_B > 0 else evaluator.max_tokens
        self.generate_args["max_new_tokens"] = effective_max_tokens

        # ── Align sampling params with LatentMAS paper (Section 4) ─────────
        # Paper: temperature=0.6, top_p=0.95 (stochastic decoding).
        # KVComm base CommunicationEvaluator uses greedy (do_sample=False);
        # override here only for LatentMAS-specific evaluator.
        self.generate_args["temperature"] = 0.6
        self.generate_args["top_p"]        = 0.95
        self.generate_args["top_k"]        = None
        self.generate_args["do_sample"]    = True

        logging.info(
            f"LatentCommunicationEvaluator ready: "
            f"latent_steps={latent_mas.latent_steps}, "
            f"latent_space_realign={latent_mas.latent_space_realign}, "
            f"layers_list={cv.layers_list}, "
            f"latent_only={latent_only}, "
            f"allow_b_think={allow_b_think}, "
            f"max_new_tokens={effective_max_tokens} "
            f"({'override' if max_tokens_B > 0 else 'from evaluator'}), "
            f"temperature=0.6, top_p=0.95 (paper-aligned)"
        )

    # ------------------------------------------------------------------
    # Override 0: get_response
    # Fixes F1-dilution bug when allow_b_think=True.
    # ------------------------------------------------------------------

    def get_response(self, output, context_length, truncate_response=True):
        """
        Decode B's output and strip the <think>...</think> trace when
        allow_b_think=True, so that F1 scoring runs only on the final answer.

        Problem (without this override):
          When allow_b_think=True, B generates:
            '<think>\nOkay, let me see ... Oldham County ...\n</think>\n\nOldham County.'
          The full string is passed to f1_match(answer, full_response).
          Word overlap: ref={'oldham','county'}, cand has 100+ words.
          Precision = 2/100 = 0.02 → F1 = 0.07 < threshold(0.5) → score = 0.

        Fix:
          Strip everything up to and including '</think>' (and following whitespace)
          before passing to the evaluator. The JSONL log still stores the full
          response (for analysis), but evaluation uses only the clean answer.
        """
        full_response = super().get_response(output, context_length, truncate_response)

        if self.allow_b_think and "</think>" in full_response:
            # Extract only the text after the closing </think> tag
            after_think = full_response.split("</think>", 1)[1].strip()
            # If something remains after stripping, use it; otherwise fall back
            return after_think if after_think else full_response

        return full_response

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
            # Fix B2: Guard against double <think>
            if input_ids_A.shape[-1] < think_tensor.shape[-1] or not torch.equal(input_ids_A[:, -think_tensor.shape[-1]:], think_tensor):
                input_ids_A = torch.cat([input_ids_A, think_tensor], dim=-1)

        # ── Receiver B: latent-aware prompt ──────────────────────────────
        msg_B = build_latent_receiver_msg(self.evaluator, item, allow_b_think=self.allow_b_think)
        # Use eval.apply_chat_template for B — with allow_b_think option (Fix B1)
        input_ids_B = apply_chat_template(
            self.evaluator, self.tokenizer, msg_B, model_B, allow_b_think=self.allow_b_think
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

    # ------------------------------------------------------------------
    # Batched Evaluation (batch_size > 1)
    # ------------------------------------------------------------------

    def inference_batch(self, cv, items):
        """
        Run batched inference on multiple dataset items simultaneously (batch_size > 1).
        Uses right-padded input_ids_A for LatentMAS prefill, and left-padded input_ids_B
        for Receiver B CausalLM generation.
        """
        if len(items) == 1:
            return [self.inference(cv.A, cv, items[0])]

        batch_size = len(items)
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id

        # 1. Prepare individual input_ids
        ids_A_list, ids_B_list = [], []
        real_len_B = []
        for item in items:
            ids_A, ids_B = self.prepare_input_ids(item, cv.A, cv.B)
            ids_A_list.append(ids_A[0])
            ids_B_list.append(ids_B[0])
            real_len_B.append(ids_B[0].shape[0])

        # 2. Right-padding input_ids_A for Sender A prefill + latent loop
        max_len_A = max(ids.shape[0] for ids in ids_A_list)
        input_ids_A = torch.full((batch_size, max_len_A), pad_id, dtype=torch.long, device=cv.A.device)
        attention_mask_A = torch.zeros((batch_size, max_len_A), dtype=torch.long, device=cv.A.device)

        for i, ids in enumerate(ids_A_list):
            l = ids.shape[0]
            input_ids_A[i, :l] = ids
            attention_mask_A[i, :l] = 1

        # 3. Run LatentMAS on batched Model A
        latent_past_kv = self.latent_mas.run(
            input_ids_A,
            attention_mask=attention_mask_A,
            latent_only=self.latent_only,
        )

        # 4. Left-padding input_ids_B for Receiver B CausalLM generation
        max_len_B = max(ids.shape[0] for ids in ids_B_list)
        input_ids_B = torch.full((batch_size, max_len_B), pad_id, dtype=torch.long, device=cv.B.device)
        attention_mask_B_tokens = torch.zeros((batch_size, max_len_B), dtype=torch.long, device=cv.B.device)

        for i, ids in enumerate(ids_B_list):
            l = ids.shape[0]
            input_ids_B[i, -l:] = ids
            attention_mask_B_tokens[i, -l:] = 1

        # 5. Build past_mask for A's KV cache
        # past_mask logic:
        # If latent_only=True, cache is fixed length N_latent (all active).
        # If latent_only=False, cache includes A's original KV (padded to max_len_A) + N_latent.
        # We must mask out padding tokens from A (attention_mask_A) so B does not attend to them.
        N_latent = self.latent_mas.latent_steps
        if self.latent_only:
            past_mask = torch.ones((batch_size, N_latent), dtype=torch.long, device=cv.B.device)
        else:
            latent_ones = torch.ones((batch_size, N_latent), dtype=torch.long, device=cv.B.device)
            past_mask = torch.cat([attention_mask_A.to(cv.B.device), latent_ones], dim=1)

        attention_mask_B = torch.cat([past_mask, attention_mask_B_tokens], dim=1)

        # 6. Batched Generation on B
        outputs = cv.generate(
            input_ids_B,
            attention_mask=attention_mask_B,
            out_A_past_key_values=latent_past_kv,
            **self.generate_args,
        )

        # 7. Decode responses per item
        # With left-padding B inputs, input_ids_B has shape [batch_size, max_len_B].
        # Generation appends new tokens starting strictly at position max_len_B.
        # Therefore, context_length is always max_len_B.
        responses = []
        for i in range(batch_size):
            response_i = self.get_response(outputs[i], max_len_B)
            responses.append(response_i)

        return responses

    def _test(self, model_A, cv=None, limit=None, do_calc_layer_importance=False, batch_size=1):
        if cv is None:
            return super()._test(model_A, limit=limit, do_calc_layer_importance=do_calc_layer_importance)

        import json
        items_all = list(self.evaluator)
        if limit is not None:
            items_all = items_all[:limit]

        from tqdm import tqdm
        progress_bar = tqdm(range(0, len(items_all), batch_size), desc=f"{self.name} result: 0.0000", disable=do_calc_layer_importance)

        # ── Open response log file (mirrors CommunicationEvaluator pattern) ──
        response_log_file = None
        if self.response_log_path and not do_calc_layer_importance:
            response_log_file = open(self.response_log_path, "a", encoding="utf-8")

        # Meta fields written to every log record for easy filtering in debug
        _mode_tag = (
            f"latent_only={self.latent_only}"
            f"|allow_b_think={self.allow_b_think}"
            f"|N={self.latent_mas.latent_steps}"
        )

        try:
            for start_idx in progress_bar:
                batch_items = items_all[start_idx : start_idx + batch_size]
                # When computing layer importance, always use single-item inference to
                # get per-item attention weights via cv.calc_attn_weights_from_qk()
                if len(batch_items) > 1 and not do_calc_layer_importance:
                    responses = self.inference_batch(cv, batch_items)
                else:
                    responses = [self.inference(model_A, cv, item) for item in batch_items]

                if do_calc_layer_importance:
                    cv.calc_attn_weights_from_qk()
                    self.layer_importance_total = calc_layer_importance(
                        cv.B_attn_weights, model_A.name, self.layer_importance_total
                    )

                for i, (item, resp) in enumerate(zip(batch_items, responses)):
                    # resp = clean answer (thinking trace stripped by get_response override)
                    # Score before writing so we can include `correct` in the log
                    prev_count = self.evaluator.f1_count
                    prev_total = self.evaluator.f1_total
                    self.evaluator.evaluate_item(item, resp)
                    # Detect correctness: f1_total increased by 1.0 = correct
                    item_score = self.evaluator.f1_total - prev_total

                    result = self.evaluator.get_result()
                    progress_bar.set_description(f"{self.name} result: {result:.4f}")

                    # ── Write to responses.jsonl ────────────────────────────
                    if response_log_file is not None:
                        record = {
                            "idx":            start_idx + i,
                            "mode":           _mode_tag,
                            "prompt_a":       item.get("prompt_A", ""),
                            "prompt_b":       item.get("prompt_B", ""),
                            "response":       resp,          # clean answer (post-</think>)
                            "answer":         item.get("answer", ""),
                            "item_score":     round(item_score, 4),
                            "result_so_far":  round(result, 4),
                        }
                        response_log_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                        response_log_file.flush()

        finally:
            if response_log_file is not None:
                response_log_file.close()

        return self.evaluator.get_result()


    @torch.no_grad()
    def test(self, model_A, cv, limit=None, no_wandb=False, do_calc_layer_importance=False, batch_size=1):
        import time, logging
        tic = time.time()
        result = self._test(model_A, cv, limit=limit, do_calc_layer_importance=do_calc_layer_importance, batch_size=batch_size)
        toc = time.time()
        time_used = toc - tic

        if self.use_wandb and not no_wandb and not do_calc_layer_importance:
            import wandb
            wandb.log({f"{self.name}_result": result, f"{self.name}_time": time_used})
        logging.info(f"{self.name} result: {result:.4f}, {self.name} time: {time_used:.2f}s")
        return result


# ──────────────────────────────────────────────────────────────────────────────
# TextMASEvaluator — Natural-language baseline for LatentMAS (paper-faithful)
# ──────────────────────────────────────────────────────────────────────────────

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

        effective_max_tokens = max_tokens_B if max_tokens_B > 0 else evaluator.max_tokens

        # Both A and B use the effective token budget.
        # Sampling params aligned with LatentMAS paper (Section 4):
        # temperature=0.6, top_p=0.95 — same as LatentCommunicationEvaluator.
        self.generate_args = {
            "max_new_tokens": effective_max_tokens,
            "temperature":    0.6,
            "top_p":          0.95,
            "top_k":          None,
            "num_beams":      1,
            "do_sample":      True,
        }

        logging.info(
            f"TextMASEvaluator ready: "
            f"max_new_tokens={effective_max_tokens}, "
            f"allow_b_think={allow_b_think}, temperature=0.6, top_p=0.95"
        )

    # ------------------------------------------------------------------
    # Input preparation
    # ------------------------------------------------------------------

    def _prepare_input_ids_A(self, item, model_A):
        """
        Build tokenised input for Agent A.

        Uses build_latent_sender_msg() — the same framing as LatentMAS.
        Respects self.allow_b_think: if True, <think> is enabled; if False,
        thinking is suppressed so A generates concise text.
        """
        msg_A = build_latent_sender_msg(
            self.evaluator, item, is_think=self._is_think_model(model_A)
        )
        input_ids_A = self._apply_chat_template(
            self.evaluator, self.tokenizer, msg_A, model_A,
            context=False, allow_b_think=self.allow_b_think,
        )
        return input_ids_A

    def _prepare_input_ids_B(self, item, response_A, model_B):
        """
        Build tokenised input for Agent B.

        B receives:
          build_latent_receiver_msg()        ← same framing as LatentMAS
          + Agent A's full reasoning text    ← sequential text communication

        Respects self.allow_b_think for B as well.

        Truncation: if the combined prompt exceeds max_input_length, we
        truncate the middle (same strategy as CommunicationEvaluator).
        """
        msg_B_base = build_latent_receiver_msg(self.evaluator, item, allow_b_think=self.allow_b_think)
        # Append A's full output as plain context (no debate/refine framing)
        msg_B = msg_B_base + "\n\n" + self._A_CONTEXT_PREFIX.format(response_A=response_A)

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
            **self.generate_args,
        )[0]
        response_A = self.get_response(output_A, input_ids_A.shape[-1])

        # ── Step 2: B generates final answer with A's full output ──────────
        input_ids_B = self._prepare_input_ids_B(item, response_A, model_B)
        output_B = model_B.generate(
            input_ids_B,
            attention_mask=torch.ones_like(input_ids_B),
            **self.generate_args,
        )[0]
        response_B = self.get_response(output_B, input_ids_B.shape[-1])
        return response_B

    # ------------------------------------------------------------------
    # Evaluation loop
    # ------------------------------------------------------------------

    def _test(self, model_A, model_B, limit=None):
        import json
        from tqdm import tqdm

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
                    response = self.inference(model_A, model_B, item)
                except Exception as e:
                    logging.error(f"TextMAS inference error at item {i}: {e}")
                    continue

                prev_total = self.evaluator.f1_total
                self.evaluator.evaluate_item(item, response)
                item_score = self.evaluator.f1_total - prev_total

                result = self.evaluator.get_result()
                progress_bar.set_description(f"{self.name} result: {result:.4f}")

                if response_log_file is not None:
                    record = {
                        "idx":           i,
                        "mode":          "textmas|sequential|thinking=True",
                        "prompt_a":      item.get("prompt_A", ""),
                        "prompt_b":      item.get("prompt_B", ""),
                        "response":      response,
                        "answer":        item.get("answer", ""),
                        "item_score":    round(item_score, 4),
                        "result_so_far": round(result, 4),
                    }
                    response_log_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                    response_log_file.flush()

        finally:
            if response_log_file is not None:
                response_log_file.close()

        return self.evaluator.get_result()

    @torch.no_grad()
    def test(self, model_A, model_B, limit=None):
        import time
        tic = time.time()
        result = self._test(model_A, model_B, limit)
        toc = time.time()
        time_used = toc - tic
        if self.use_wandb:
            import wandb
            wandb.log({f"{self.name}_result": result, f"{self.name}_time": time_used})
        logging.info(f"{self.name} result: {result:.4f}, {self.name} time: {time_used:.2f}s")
        return result

