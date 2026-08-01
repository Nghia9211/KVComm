"""
models_latent.py — LatentMAS wrapper for KVComm integration

Faithfully ports the latent thinking mechanism from LatentMAS/models.py
(ModelWrapper.generate_latent_batch, HuggingFace path, no vLLM).

Designed to be model-agnostic: wraps any HuggingFace PreTrainedModel
and returns a DynamicCache that CVCommunicator can consume via
out_A_past_key_values → prepare_key_cache().
"""

import torch
import logging
from typing import Optional
from transformers.cache_utils import DynamicCache
from transformers.modeling_utils import PreTrainedModel


class LatentMAS:
    """
    Model-agnostic wrapper that runs latent thinking on any HuggingFace PreTrainedModel.

    Faithfully reproduces the latent mechanism from LatentMAS
    (generate_latent_batch / _apply_latent_realignment),
    independent of the multi-agent framework used (KVComm handles agents).

    Flow
    ----
    1. Standard forward pass on input_ids
       → past_kv (DynamicCache), last_hidden [B, D]

    2. Latent loop x latent_steps:
       a. Realign last_hidden → latent_embed [B, 1, D]
       b. Forward with inputs_embeds=latent_embed → update past_kv, last_hidden

    3. Return past_kv
       Shape per layer: [B, kv_heads, T_input + latent_steps, head_dim]

    The returned DynamicCache is passed to CVCommunicator:
      cv.generate(input_ids_B, out_A_past_key_values=latent_past_kv, ...)
    which calls prepare_key_cache() for optional layer selection (Mode 2)
    or passes all layers through (Mode 1, latent_kv_select=False).
    """

    def __init__(
        self,
        model: PreTrainedModel,
        latent_steps: int = 5,
        latent_space_realign: bool = True,
    ) -> None:
        """
        Args:
            model: Any HuggingFace CausalLM. Must be in eval() mode.
            latent_steps: Number of latent thinking steps (N in the paper).
            latent_space_realign: If True, project hidden state back to
                embedding space via learned linear map W = (E_out^T E_out)^{-1} E_out^T E_in.
                If False, only L2-normalize (identity realignment).
        """
        self.model = model
        self.latent_steps = latent_steps
        self.latent_space_realign = latent_space_realign

        self._realign_matrix: Optional[torch.Tensor] = None
        self._target_norm: Optional[torch.Tensor] = None

        self._build_realign_matrix()
        logging.info(
            f"LatentMAS initialized: latent_steps={latent_steps}, "
            f"latent_space_realign={latent_space_realign}"
        )

    # ------------------------------------------------------------------
    # Realignment matrix
    # Ported from: LatentMAS/models.py ModelWrapper._build_latent_realign_matrix()
    # ------------------------------------------------------------------

    def _build_realign_matrix(self) -> None:
        """
        Build latent space realignment matrix W and target norm.

        When latent_space_realign=True:
            Solves (E_out^T E_out) W = E_out^T E_in
            Maps hidden states back to input embedding space.

        When False: W = Identity (only L2-norm rescaling applied).

        Ported from LatentMAS/models.py L171-198.
        """
        model = self.model
        device = next(model.parameters()).device

        input_embeds = (
            model.get_input_embeddings()
            if hasattr(model, "get_input_embeddings")
            else None
        )
        output_embeds = (
            model.get_output_embeddings()
            if hasattr(model, "get_output_embeddings")
            else None
        )
        if output_embeds is None:
            output_embeds = getattr(model, "lm_head", None)

        if (
            input_embeds is None
            or output_embeds is None
            or not hasattr(input_embeds, "weight")
            or not hasattr(output_embeds, "weight")
        ):
            raise RuntimeError(
                "Cannot build latent realignment matrix: "
                "embedding weights not accessible on this model."
            )

        # Work in float32 for numerical stability
        input_weight  = input_embeds.weight.detach().to(device=device, dtype=torch.float32)
        output_weight = output_embeds.weight.detach().to(device=device, dtype=torch.float32)

        # Gram matrix with Tikhonov regularisation
        gram = torch.matmul(output_weight.T, output_weight)
        reg  = 1e-5 * torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
        gram = gram + reg

        # Solve: gram @ W = output_weight^T @ input_weight
        rhs            = torch.matmul(output_weight.T, input_weight)
        realign_matrix = torch.linalg.solve(gram, rhs)           # [D, D]
        target_norm    = input_weight.norm(dim=1).mean().detach()  # scalar

        if not self.latent_space_realign:
            # Realignment disabled: identity matrix, only norm-rescaling
            realign_matrix = torch.eye(
                realign_matrix.shape[0],
                device=realign_matrix.device,
                dtype=realign_matrix.dtype,
            )

        self._realign_matrix = realign_matrix
        self._target_norm    = target_norm.to(device=device, dtype=realign_matrix.dtype)
        logging.info(
            f"LatentMAS realignment matrix: shape={tuple(realign_matrix.shape)}, "
            f"target_norm={target_norm.item():.4f}"
        )

    def _apply_realignment(self, hidden: torch.Tensor) -> torch.Tensor:
        """
        Apply latent realignment to hidden state, then rescale to target norm.

        Ported from LatentMAS/models.py L217-226
        (ModelWrapper._apply_latent_realignment).

        Args:
            hidden: [B, D] last hidden state (any dtype).
        Returns:
            [B, D] realigned tensor (same dtype as input).
        """
        assert self._realign_matrix is not None and self._target_norm is not None

        matrix      = self._realign_matrix.to(device=hidden.device, dtype=torch.float32)
        target_norm = self._target_norm.to(device=hidden.device,    dtype=torch.float32)

        hidden_fp32  = hidden.to(torch.float32)
        aligned      = torch.matmul(hidden_fp32, matrix)                      # [B, D]
        aligned_norm = aligned.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        aligned      = aligned * (target_norm / aligned_norm)                 # L2-rescale
        return aligned.to(hidden.dtype)

    # ------------------------------------------------------------------
    # Main latent thinking loop
    # Ported from: LatentMAS/models.py ModelWrapper.generate_latent_batch()
    # ------------------------------------------------------------------

    @torch.no_grad()
    def run(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> DynamicCache:
        """
        Run latent thinking on the given input and return a DynamicCache.

        Faithfully reproduces ModelWrapper.generate_latent_batch() from
        LatentMAS/models.py (HuggingFace path, no vLLM).

        The returned DynamicCache is passed to CVCommunicator:
            cv.generate(input_ids_B, out_A_past_key_values=latent_past_kv, ...)

        Mode 1 (latent_kv_select=False): cv has all layers -> full cache passes through.
        Mode 2 (latent_kv_select=True):  cv.prepare_key_cache() selects layers.

        Args:
            input_ids: [B, T] token ids.
            attention_mask: [B, T] mask (default: all ones). Handles right-padded batches.

        Returns:
            DynamicCache with per-layer shape [B, kv_heads, T + latent_steps, head_dim].
            The last latent_steps positions are latent thinking tokens.
        """
        if input_ids.dim() != 2:
            raise ValueError(
                f"input_ids must be 2D [batch, seq_len], got shape {tuple(input_ids.shape)}"
            )

        device    = next(self.model.parameters()).device
        input_ids = input_ids.to(device)

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=device)
        else:
            attention_mask = attention_mask.to(device)

        # ── Step 1: Standard forward pass ─────────────────────────────
        # Ported from generate_latent_batch() L308-327
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
        past = outputs.past_key_values   # DynamicCache

        # Right-padded batch: select last REAL (non-pad) token per sample.
        # Ported from generate_latent_batch() L318-323.
        last_token_idx = attention_mask.sum(1).long() - 1         # [B]
        batch_idx      = torch.arange(input_ids.shape[0], device=device)
        last_hidden    = outputs.hidden_states[-1][batch_idx, last_token_idx, :]  # [B, D]

        # ── Step 2: Latent loop ────────────────────────────────────────
        # Ported from generate_latent_batch() L334-362
        for step in range(self.latent_steps):
            # a. Project hidden state back to embedding space
            latent_vec   = self._apply_realignment(last_hidden)    # [B, D]
            latent_embed = latent_vec.unsqueeze(1)                  # [B, 1, D]

            # b. Attention mask covering past tokens + this new latent token
            past_len    = past.get_seq_length()
            latent_mask = torch.ones(
                (latent_embed.shape[0], past_len + 1),
                dtype=torch.long,
                device=latent_embed.device,
            )

            # c. Forward with latent embedding (inputs_embeds, no input_ids)
            outputs = self.model(
                inputs_embeds=latent_embed,
                attention_mask=latent_mask,
                past_key_values=past,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )
            past        = outputs.past_key_values                   # DynamicCache updated
            last_hidden = outputs.hidden_states[-1][:, -1, :]       # [B, D]

            logging.debug(
                f"Latent step {step + 1}/{self.latent_steps}: "
                f"past_len={past.get_seq_length()}"
            )

        # past.key_cache[i].shape = [B, kv_heads, T_input + latent_steps, head_dim]
        return past
