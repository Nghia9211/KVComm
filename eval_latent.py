"""Two-agent latent and text evaluation.

LatentCommunicationEvaluator implements prompt preparation, fixed/adaptive
sender inference, selected/full KV handoff, metrics and per-sample response logs.
TextMASEvaluator is an independent A-summary -> B-answer evaluator; it does not
inherit NLDEvaluator. Both use the shared task-profile prompt/metric utilities.
"""

import json
import time
import logging
import torch
from tqdm import tqdm
from eval import CommunicationEvaluator, apply_chat_template, is_think_model
from layer_importance import calc_layer_importance
from models_latent import LatentMAS
from models import CVCommunicator
from prompts_latent import build_latent_sender_msg, build_latent_receiver_msg, build_text_receiver_msg
from utils.response_logging import build_response_record
from utils.method_names import latent_method
from utils.evaluation_config import resolve_textmas_budgets
from adaptive_latent import sample_id, sample_rng, decoding_seed, synchronize_models, cache_payload, content_hash


class LatentCommunicationEvaluator(CommunicationEvaluator):
    """
    CommunicationEvaluator extended with LatentMAS thinking for sender A.

    Overrides prompt preparation, inference, response decoding and evaluation.
    Uses the shared truncation and layer-importance infrastructure.

    Operating modes (controlled by CVCommunicator layers_list):
      Mode 1: cv.layers_list = all layers  -> no layer selection
      Mode 2: cv.layers_list = subset      -> KVComm layer selection applied

    LatentMAS returns the full input+actual-latent cache. Mode 2 selects whole
    layers afterwards and preserves the original attention sink.
    """

    def __init__(
        self,
        evaluator,
        tokenizer,
        use_wandb: bool,
        max_input_length: int,
        latent_mas: LatentMAS,
        cv: CVCommunicator,
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
        self.allow_b_think = allow_b_think
        self.last_context_length = None
        self.last_latent_length = None
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
        if latent_mas.greedy:
            self.generate_args.update(do_sample=False, temperature=None, top_p=None)
        self.last_inference_stats = {}

        logging.info(
            f"LatentCommunicationEvaluator ready: "
            f"latent_steps={latent_mas.latent_steps}, "
            f"latent_space_realign={latent_mas.latent_space_realign}, "
            f"layers_list={cv.layers_list}, "
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
    # Uses updated prepare_input_ids and full input+latent cache.
    # ------------------------------------------------------------------

    def inference(self, model, cv, item):
        """
        Run one inference sample using LatentMAS for the sender.

        Changes vs CommunicationEvaluator.inference():
          1. prepare_input_ids() now uses latent-aware prompts (overridden above).
          2. Sender uses latent_mas.run() instead of model(input_ids_A).
          3. CVCommunicator optionally selects whole KV layers.

        Args:
            model: model_A (passed by _test(), kept for API compatibility).
                   Not used directly; LatentMAS wraps model_A internally.
            cv:    CVCommunicator(model_A, model_B, ...).
            item:  Dataset item dict with "prompt_A", "prompt_B", "answer".

        Returns:
            str: decoded response from model_B.
        """
        # ── Input preparation (now uses overridden prepare_input_ids) ─────
        if self.latent_mas.profile_timing:
            synchronize_models(cv.A, cv.B)
            for device in {p.device for m in (cv.A, cv.B) for p in m.parameters() if p.is_cuda}:
                torch.cuda.reset_peak_memory_stats(device)
        inference_start = time.perf_counter()
        identity = sample_id(item)
        input_ids_A, input_ids_B = self.prepare_input_ids(item, cv.A, cv.B)

        # ── Latent thinking ───────────────────────────────────────────────
        # Replace: out_A = model(input_ids_A, use_cache=True)
        # With:    latent loop on model_A → DynamicCache
        latent_past_kv = self.latent_mas.run(
            input_ids_A,
            attention_mask=torch.ones_like(input_ids_A),
        )
        self.last_context_length = int(input_ids_A.shape[-1])
        self.last_latent_length = int(latent_past_kv._kvcomm_latent_length)

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
        payload = cache_payload(
            latent_past_kv, cv.layers_list if cv.layers_list is not None else range(len(latent_past_kv.key_cache)))
        if self.latent_mas.profile_timing:
            synchronize_models(cv.A, cv.B)
        receiver_start = time.perf_counter()
        with sample_rng(self.latent_mas.sample_seed, identity):
            output = cv.generate(
                input_ids_B,
                attention_mask=attention_mask_B,
                out_A_past_key_values=latent_past_kv,
                **self.generate_args,
            )[0]
        if self.latent_mas.profile_timing:
            synchronize_models(cv.A, cv.B)
        receiver_end = time.perf_counter()

        context_length = input_ids_B.shape[-1]
        response = self.get_response(output, context_length)
        self.last_inference_stats = dict(self.latent_mas.last_run_stats,
            sample_id=identity, status="ok", **payload,
            prompt_A_token_hash=content_hash(input_ids_A[0].tolist()),
            prompt_B_token_hash=content_hash(input_ids_B[0].tolist()),
            decoding_seed=(decoding_seed(self.latent_mas.sample_seed, identity)
                           if self.latent_mas.sample_seed is not None else None),
            generated_B_token_count_raw=int(output.shape[-1] - context_length),
            B_generation_including_handoff_ms=(receiver_end - receiver_start) * 1000,
            kv_handoff_ms=None,  # routing is lazy inside cv.forward; do not double count
            end_to_end_ms=(time.perf_counter() - inference_start) * 1000)
        if self.latent_mas.profile_timing:
            self.last_inference_stats["per_device_peak_allocated_bytes"] = {
                str(device): torch.cuda.max_memory_allocated(device)
                for device in {p.device for m in (cv.A, cv.B) for p in m.parameters() if p.is_cuda}}
        self.last_prompt_ids = [(input_ids_A[0], input_ids_B[0])]
        return response

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
        for item in items:
            ids_A, ids_B = self.prepare_input_ids(item, cv.A, cv.B)
            ids_A_list.append(ids_A[0])
            ids_B_list.append(ids_B[0])

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
        # Cache includes A's padded input KV plus N latent tokens.  Mask A padding.
        N_latent = self.latent_mas.latent_steps
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

        self.last_prompt_ids = list(zip(ids_A_list, ids_B_list))
        return responses

    def _test(
        self, model_A, cv=None, limit=None, do_calc_layer_importance=False,
        batch_size=1,
    ):
        if batch_size != 1 and (self.latent_mas.latent_step_policy != "fixed" or
                               self.latent_mas.sample_seed is not None or self.latent_mas.profile_timing):
            raise ValueError("Adaptive/per-sample RNG/timing require batch_size=1")
        if cv is None:
            return super()._test(model_A, limit=limit, do_calc_layer_importance=do_calc_layer_importance)

        items_all = list(self.evaluator)
        if limit is not None:
            items_all = items_all[:limit]

        collecting_importance = do_calc_layer_importance
        progress_bar = tqdm(
            range(0, len(items_all), batch_size),
            desc=f"{self.name} result: 0.0000",
            disable=collecting_importance,
        )

        # ── Open response log file (mirrors CommunicationEvaluator pattern) ──
        response_log_file = None
        if self.response_log_path and not collecting_importance:
            response_log_file = open(self.response_log_path, "a", encoding="utf-8")

        # Meta fields written to every log record for easy filtering in debug
        try:
            for start_idx in progress_bar:
                batch_items = items_all[start_idx : start_idx + batch_size]
                # When computing layer importance, always use single-item inference to
                # get per-item attention weights via cv.calc_attn_weights_from_qk()
                try:
                    if len(batch_items) > 1 and not collecting_importance:
                        responses = self.inference_batch(cv, batch_items)
                    else:
                        responses = [self.inference(model_A, cv, item) for item in batch_items]
                except Exception as exc:
                    if response_log_file is not None:
                        response_log_file.write(json.dumps({
                            "adaptive_schema_version": 1, "status": "error",
                            "sample_ids": [sample_id(item) for item in batch_items],
                            "policy": self.latent_mas.latent_step_policy,
                            "configured_max_steps": self.latent_mas.latent_steps,
                            "error_type": type(exc).__name__, "error": str(exc),
                        }, ensure_ascii=False) + "\n")
                        response_log_file.flush()
                    raise

                if do_calc_layer_importance:
                    cv.calc_attn_weights_from_qk()
                    self.layer_importance_total = calc_layer_importance(
                        cv.B_attn_weights, model_A.name, self.layer_importance_total
                    )
                for i, (item, resp) in enumerate(zip(batch_items, responses)):
                    # resp = clean answer (thinking trace stripped by get_response override)
                    item_metrics = self.evaluator.evaluate_item(item, resp) or {}

                    result = self.evaluator.get_result()
                    progress_bar.set_description(f"{self.name} {self.evaluator.primary_metric}: {result:.4f}")

                    # ── Write to responses.jsonl ────────────────────────────
                    if response_log_file is not None:
                        logged_ids_a, logged_ids_b = self.last_prompt_ids[i]
                        selected_layers = (
                            list(cv.layers_list) if cv.layers_list is not None
                            else None
                        )
                        total_layers = getattr(cv.A.config, "num_hidden_layers", None)
                        is_selective = (
                            bool(selected_layers) and total_layers is not None
                            and len(selected_layers) < total_layers
                        )
                        method = (
                            latent_method(is_selective, response=True)
                        )
                        record = build_response_record(
                            idx=start_idx + i, evaluator=self.evaluator, item=item,
                            method=method,
                            response=resp,
                            model_a_prompt=self.tokenizer.decode(logged_ids_a, skip_special_tokens=False),
                            model_b_prompt=self.tokenizer.decode(logged_ids_b, skip_special_tokens=False),
                            item_metrics=item_metrics, aggregate_metrics=self.evaluator.get_results(),
                            max_tokens_b=self.generate_args["max_new_tokens"],
                            generated_tokens_a=self.last_latent_length if len(batch_items) == 1 else self.latent_mas.latent_steps,
                            generated_tokens_b=(self.last_inference_stats["generated_B_token_count_raw"]
                                                if len(batch_items) == 1 else len(self.tokenizer.encode(resp, add_special_tokens=False))),
                            communication_type="latent_kv", latent_steps=(self.last_latent_length if len(batch_items) == 1 else self.latent_mas.latent_steps),
                            layer_selection_mode=(
                                "selected" if is_selective
                                else "full"
                            ),
                            selected_layers=selected_layers,
                            adaptive_stats=self.last_inference_stats if len(batch_items) == 1 else None,
                        )
                        response_log_file.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                        response_log_file.flush()

        finally:
            if response_log_file is not None:
                response_log_file.close()

        return self.evaluator.get_result()


    @torch.no_grad()
    def test(
        self, model_A, cv, limit=None, no_wandb=False,
        do_calc_layer_importance=False,
        batch_size=1,
    ):
        if not do_calc_layer_importance:
            for _ in range(self.latent_mas.warmup):
                self.inference(model_A, cv, self.evaluator.data[0])
        tic = time.time()
        result = self._test(
            model_A, cv, limit=limit,
            do_calc_layer_importance=do_calc_layer_importance,
            batch_size=batch_size,
        )
        toc = time.time()
        time_used = toc - tic
        self.last_time_used = time_used

        if self.use_wandb and not no_wandb and not do_calc_layer_importance:
            import wandb
            wandb.log({f"{self.name}_{key}": value for key, value in self.evaluator.get_results().items() if isinstance(value, (int, float))})
            wandb.log({f"{self.name}_time": time_used})
        logging.info(f"{self.name} result: {result:.4f}, {self.name} time: {time_used:.2f}s")
        return result

# TextMASEvaluator — Natural-language baseline for LatentMAS (paper-faithful)
# ──────────────────────────────────────────────────────────────────────────────

# Backward-compatible import for existing callers.
from eval_textmas import TextMASEvaluator

