"""
eval_latent.py — Latent evaluator for KVComm + LatentMAS integration

Extends CommunicationEvaluator with a single change:
  - KVComm original: out_A = model_A(input_ids_A) → past_key_values
  - LatentMAS:       out_A = latent_mas.run(input_ids_A) → DynamicCache

Everything else (prepare_input_ids, get_response, _test, test, layer
importance tracking) is inherited from CommunicationEvaluator unchanged.
"""

import torch
import logging
from eval import CommunicationEvaluator
from models_latent import LatentMAS
from models import CVCommunicator


class LatentCommunicationEvaluator(CommunicationEvaluator):
    """
    CommunicationEvaluator extended with LatentMAS thinking for sender A.

    Design principle: minimal diff from the parent class.
    Only inference() is overridden — all KVComm evaluation infrastructure
    (prepare_input_ids, truncate_input, get_response, _test, test,
    layer importance) is reused without modification.

    Both operating modes are controlled entirely by how CVCommunicator
    is constructed (layers_list); this class is mode-agnostic:

      Mode 1 (LatentMAS standalone):
        cv has layers_list = list(range(A_num_layers))
        → prepare_key_cache() keeps full cache for all layers (no selection)

      Mode 2 (KVComm + LatentMAS):
        cv has a specific layers_list
        → prepare_key_cache() applies layer selection on the latent cache
    """

    def __init__(
        self,
        evaluator,
        tokenizer,
        use_wandb: bool,
        max_input_length: int,
        latent_mas: LatentMAS,
        cv: CVCommunicator,
    ) -> None:
        """
        Args:
            evaluator:        Task evaluator (e.g. TMathEvaluator).
            tokenizer:        Tokenizer for model_B.
            use_wandb:        Whether to log metrics to W&B.
            max_input_length: Maximum combined token length before truncation.
            latent_mas:       LatentMAS instance wrapping model_A.
            cv:               CVCommunicator(model_A, model_B, ...) — used here
                              to verify that latent_mas.model is cv.A.
        """
        super().__init__(evaluator, tokenizer, use_wandb, max_input_length)

        # Safety: ensure the LatentMAS model IS the same object as cv.A
        assert latent_mas.model is cv.A, (
            "latent_mas.model must be the exact same Python object as cv.A.\n"
            "Initialise with:\n"
            "    latent_mas = LatentMAS(model=model_A, ...)\n"
            "    cv = CVCommunicator(model_A=model_A, model_B=model_B, ...)\n"
            "using the same model_A instance."
        )

        self.latent_mas = latent_mas
        self.name = "latent_communication"
        logging.info(
            f"LatentCommunicationEvaluator ready: "
            f"latent_steps={latent_mas.latent_steps}, "
            f"latent_space_realign={latent_mas.latent_space_realign}, "
            f"layers_list={cv.layers_list}"
        )

    # ------------------------------------------------------------------
    # Override: only this method changes vs CommunicationEvaluator
    # ------------------------------------------------------------------

    def inference(self, model, cv, item):
        """
        Run one inference sample using LatentMAS for the sender.

        Difference from CommunicationEvaluator.inference():
            BEFORE: out_A = model(input_ids_A, use_cache=True)
                            out_A_past_key_values = out_A.past_key_values
            AFTER:  latent_past_kv = latent_mas.run(input_ids_A)
                    (DynamicCache with T + latent_steps tokens per layer)

        The rest of the flow (prepare_key_cache inside cv.generate,
        decode, get_response) is identical to the parent class.

        Args:
            model: model_A (passed by _test(), kept for API compatibility).
                   Not used directly; LatentMAS wraps model_A internally.
            cv:    CVCommunicator(model_A, model_B, ...).
            item:  Dataset item dict with "prompt_A", "prompt_B", "answer".

        Returns:
            str: decoded response from model_B.
        """
        # ── Input preparation (identical to parent) ───────────────────
        input_ids_A, input_ids_B = self.prepare_input_ids(item, cv.A, cv.B)

        # ── Latent thinking (KEY CHANGE) ──────────────────────────────
        # Replace: out_A = model(input_ids_A, use_cache=True)
        # With:    latent loop on model_A → accumulated DynamicCache
        latent_past_kv = self.latent_mas.run(
            input_ids_A,
            attention_mask=torch.ones_like(input_ids_A),
        )
        # latent_past_kv.key_cache[i].shape = [1, kv_heads, T+N, head_dim]
        # where N = latent_steps

        # ── Generation (identical to parent) ──────────────────────────
        # cv.generate calls forward() → prepare_key_cache(latent_past_kv)
        # → layer selection (Mode 2) or identity (Mode 1) → model_B.generate
        output = cv.generate(
            input_ids_B,
            attention_mask=torch.ones_like(input_ids_B),
            out_A_past_key_values=latent_past_kv,
            **self.generate_args,
        )[0]

        context_length = input_ids_B.shape[-1]
        return self.get_response(output, context_length)
