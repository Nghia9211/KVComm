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
import time
from dataclasses import asdict
from adaptive_latent import (PolicyConfig, StopController, feature_tensor,
                             decode_features, synchronize_models)
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
        latent_step_policy: str = "fixed",
        min_latent_steps: int = 10,
        latent_check_interval: int = 5,
        latent_patience: int = 2,
        policy_config: Optional[PolicyConfig] = None,
        policy_config_hash: Optional[str] = None,
        latent_trace: bool = False,
        profile_timing: bool = False,
        greedy: bool = False,
        sample_seed: Optional[int] = None,
        warmup: int = 0,
    ) -> None:
        """
        Args:
            model: Any HuggingFace CausalLM. Must be in eval() mode.
            latent_steps: Number of latent thinking steps (N in the paper).
            latent_space_realign: If True, project hidden state back to
                embedding space via learned linear map W = (E_out^T E_out)^{-1} E_out^T E_in.
                If False, only L2-normalize (identity realignment).
            latent_trace: Record per-step features for offline analysis without changing stopping.
        """
        self.model = model
        self.latent_steps = latent_steps
        self.latent_step_policy = latent_step_policy
        self.policy_config = policy_config
        self.policy_config_hash = policy_config_hash
        self.controller_args = dict(policy=latent_step_policy, min_steps=min_latent_steps,
                                    interval=latent_check_interval, patience=latent_patience,
                                    config=policy_config)
        StopController(max_steps=latent_steps, **self.controller_args)
        self.latent_trace = latent_trace
        self.profile_timing = profile_timing
        self.greedy = greedy
        self.sample_seed = sample_seed
        if warmup < 0:
            raise ValueError("warmup must be nonnegative")
        self.warmup = warmup
        self.last_run_stats = {}
        self.latent_space_realign = latent_space_realign
        self._realign_matrix: Optional[torch.Tensor] = None
        self._target_norm: Optional[torch.Tensor] = None

        self._build_realign_matrix()
        # Cache the layer→device mapping for device_map="auto" support.
        # Built lazily on first run() call (model must be fully loaded first).
        self._layer_devices: Optional[list] = None
        logging.info(
            f"LatentMAS initialized: latent_steps={latent_steps}, "
            f"latent_space_realign={latent_space_realign}, "
            f"latent_trace={latent_trace}"
        )

    # ------------------------------------------------------------------
    # Device helpers (for device_map="auto" / Accelerate offloading)
    # ------------------------------------------------------------------

    def _build_layer_devices(self) -> list:
        """
        Return a list of torch.device, one per transformer layer.

        When device_map="auto" is used with Accelerate, different layers
        may reside on different devices (cuda:0, cuda:1, cpu, or even
        meta for fully-offloaded layers).  We must move KV tensors to the
        device where that layer's parameters actually live *before* calling
        DynamicCache.update(), otherwise torch.cat raises:

            RuntimeError: Tensor on device cuda:0 is not on the expected
                          device meta!

        For models where all layers are on one device this is a no-op.
        """
        devices = []
        model = self.model

        # Resolve the actual layer list (handles multimodal wrappers)
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            layers = model.model.layers
        elif hasattr(model, "model") and hasattr(model.model, "language_model"):
            layers = model.model.language_model.layers
        else:
            layers = []

        for layer in layers:
            try:
                # k_proj weight lives on the actual compute device of this layer
                dev = layer.self_attn.k_proj.weight.device
            except AttributeError:
                try:
                    dev = next(layer.parameters()).device
                except StopIteration:
                    dev = next(model.parameters()).device
            devices.append(dev)

        if not devices:
            # Fallback: all layers on the same device as the first parameter
            devices = [next(model.parameters()).device]

        logging.debug(
            f"LatentMAS layer devices: "
            + ", ".join(f"L{i}={d}" for i, d in enumerate(devices[:4]))
            + (" ..." if len(devices) > 4 else "")
        )
        return devices

    def _normalize_cache_devices(self, past: DynamicCache) -> DynamicCache:
        """
        Move each layer's KV tensors to the correct device for that layer.

        This prevents the "cuda:0 is not on the expected device meta" error
        that occurs when device_map="auto" spreads layers across devices.

        Creates a *new* DynamicCache (does not mutate the input) to avoid
        in-place side-effects on the accumulated past.
        """
        if self._layer_devices is None:
            self._layer_devices = self._build_layer_devices()

        layer_devices = self._layer_devices
        n_layers = len(past.key_cache)

        # Fast path: single device — no movement needed
        if len(set(str(d) for d in layer_devices)) == 1:
            return past

        new_cache = DynamicCache()
        for i in range(n_layers):
            # Pick device for this layer (clamp index if fewer devices registered)
            dev = layer_devices[min(i, len(layer_devices) - 1)]
            k = past.key_cache[i].to(dev)
            v = past.value_cache[i].to(dev)
            new_cache.update(k, v, i)

        new_cache._seen_tokens = past._seen_tokens
        return new_cache

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
        Mode 2 (latent_kv_select=True): cv.prepare_key_cache() selects layers.

        Args:
            input_ids:    [B, T] token ids.
            attention_mask: [B, T] mask (default: all ones). Handles right-padded batches.
        Returns:
            Full DynamicCache with shape per layer
            ``[B, kv_heads, T_input + actual_steps, head_dim]``.
            Whole-layer selection happens later in CVCommunicator.
        """
        if input_ids.dim() != 2:
            raise ValueError(
                f"input_ids must be 2D [batch, seq_len], got shape {tuple(input_ids.shape)}"
            )

        if self.latent_step_policy != "fixed" and input_ids.shape[0] != 1:
            raise ValueError("Adaptive latent steps require batch_size=1")
        controller = StopController(max_steps=self.latent_steps, **self.controller_args)
        self.last_run_stats = {}
        buffered_features = []
        actual_steps = 0
        reason = "fixed_budget" if self.latent_step_policy == "fixed" else "max_steps"
        controller_ms = 0.0
        feature_config = self.policy_config or PolicyConfig(cosine_distance=0.0)
        if any(i >= self.model.config.num_hidden_layers for i in feature_config.value_layers):
            raise ValueError("value_layers contains an out-of-range model layer")
        if self.profile_timing:
            synchronize_models(self.model)
        prefill_start = time.perf_counter()

        # Resolve the input device: use the embedding layer's device so that
        # input_ids land on the first real compute device, even with
        # device_map="auto" where next(parameters()) might return "meta".
        try:
            device = self.model.get_input_embeddings().weight.device
        except Exception:
            device = next(
                (p.device for p in self.model.parameters() if p.device.type != "meta"),
                torch.device("cpu"),
            )

        # Build layer→device map lazily (needed for _normalize_cache_devices)
        if self._layer_devices is None:
            self._layer_devices = self._build_layer_devices()

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
        if self.profile_timing:
            synchronize_models(self.model)
        latent_start = time.perf_counter()

        # ── Step 2: Latent loop ────────────────────────────────────────
        # Ported from generate_latent_batch() L334-362
        for step in range(self.latent_steps):
            # a. Project hidden state back to embedding space
            latent_vec   = self._apply_realignment(last_hidden)    # [B, D]
            latent_embed = latent_vec.unsqueeze(1)                  # [B, 1, D]

            # b. Fix B3: Attention mask preserving pad-token masks + new latent tokens
            latent_ones = torch.ones(
                (input_ids.shape[0], step + 1),
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )
            latent_mask = torch.cat([attention_mask, latent_ones], dim=1)

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

            # ── Device normalization (critical for device_map="auto") ──────
            # When Accelerate spreads layers across devices, the KV tensors
            # returned from the forward pass may be on mixed devices.  The
            # NEXT forward call will then try to torch.cat a cuda:0 tensor
            # with a meta tensor, causing:
            #   RuntimeError: Tensor on device cuda:0 is not on the expected
            #                 device meta!
            # Move every layer’s K/V to the device where that layer lives.
            past        = self._normalize_cache_devices(past)
            prev_hidden = last_hidden  # store before updating
            last_hidden = outputs.hidden_states[-1][:, -1, :]       # [B, D]

            actual_steps = step + 1
            check = controller.should_check(actual_steps)
            features = None
            at_adaptive_cap = self.latent_step_policy != "fixed" and actual_steps == self.latent_steps
            if check or at_adaptive_cap or self.latent_trace:
                check_start = time.perf_counter()
                packed = feature_tensor(prev_hidden, last_hidden, past,
                                        input_ids.shape[-1], feature_config,
                                        include_values=bool(feature_config.value_layers) and
                                        (self.latent_step_policy == "hidden_value" or self.latent_trace))
                if self.latent_trace:
                    buffered_features.append((actual_steps, packed))
                if check or at_adaptive_cap:
                    features = decode_features(packed.cpu().tolist())
                controller_ms += (time.perf_counter() - check_start) * 1000
            decision = controller.observe(actual_steps, features)
            if decision:
                reason = decision
                break

            logging.debug(
                f"Latent step {step + 1}/{self.latent_steps}: "
                f"past_len={past.get_seq_length()}"
            )

        # Preserve actual context/latent lengths for logging and cache invariants. These are
        # ordinary Python attributes on DynamicCache and do not alter tensors.
        past._kvcomm_context_length = int(input_ids.shape[-1])
        past._kvcomm_latent_length = actual_steps
        past._kvcomm_logical_length = int(input_ids.shape[-1] + actual_steps)
        if past.get_seq_length() != past._kvcomm_logical_length:
            raise RuntimeError("Sender cache length differs from completed latent forwards")
        if self.profile_timing:
            synchronize_models(self.model)
        latent_end = time.perf_counter()
        trace = []
        if buffered_features:
            rows = torch.stack([entry[1] for entry in buffered_features]).cpu().tolist()
            trace = [dict(step=entry[0], **decode_features(row))
                     for entry, row in zip(buffered_features, rows)]
        self.last_run_stats = {
            "adaptive_schema_version": 1, "policy": self.latent_step_policy,
            "policy_config_hash": self.policy_config_hash,
            "feature_config": asdict(feature_config),
            "configured_max_steps": self.latent_steps, "actual_steps": actual_steps,
            "stop_reason": reason, "decision_checkpoints": controller.checkpoints,
            "context_length": int(input_ids.shape[-1]),
            "actual_cache_length_before_B": past.get_seq_length(),
            "feature_trace": trace if self.latent_trace else [],
            "timing_synchronized": self.profile_timing,
            "prefill_A_ms": (latent_start - prefill_start) * 1000 if self.profile_timing else None,
            "latent_A_ms_including_controller": (latent_end - latent_start) * 1000 if self.profile_timing else None,
            "controller_host_ms_diagnostic": controller_ms,
        }
        past._kvcomm_adaptive_stats = self.last_run_stats

        return past
