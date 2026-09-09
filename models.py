from typing import Literal, Optional
import copy
import inspect
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_utils import PreTrainedModel
from transformers.generation.utils import GenerationMixin
from transformers.cache_utils import DynamicCache, Cache
from transformers.modeling_outputs import CausalLMOutputWithPast
import logging
from model_attn import LlamaAttentionTracer, Qwen2AttentionTracer, Qwen3AttentionTracer, Gemma3AttentionTracer
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaModel, repeat_kv
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention, Qwen2Model
from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention, Qwen3Model
from transformers.models.gemma3.modeling_gemma3 import Gemma3Attention
from segmented_kv import logical_retention, slice_segmented_kv

def get_layer_map(L_A, L_B):
    layer_map = {}
    for l_a in range(L_A):
        layer_map[l_a] = round( (l_a + 0.5) * L_B / L_A - 0.5 )
    return layer_map

class CVCommunicator(PreTrainedModel, GenerationMixin):
    def __init__(
        self,
        model_A: PreTrainedModel,
        model_B: PreTrainedModel,
        layer_from: int,
        layer_to: int,
        top_layers: float = 0.0,
        layers_list: list[int] = [],
        apply_attn_tracer: bool = False,
        shift_back: bool = False,
        segmented_context_layers: Optional[list[int]] = None,
        segmented_latent_layers: Optional[list[int]] = None,
        capture_segmented_importance: bool = False,
    ) -> None:
        super().__init__(model_B.config)
        self.A = model_A
        self.B = model_B
        self.layer_from = layer_from
        self.layer_to = layer_to
        self.apply_attn_tracer = apply_attn_tracer
        self.shift_back = shift_back
        self.capture_segmented_importance = capture_segmented_importance
        self.segmented_kv = (
            segmented_context_layers is not None or segmented_latent_layers is not None
        )
        if self.segmented_kv and (
            segmented_context_layers is None or segmented_latent_layers is None
        ):
            raise ValueError(
                "segmented_context_layers and segmented_latent_layers must be provided together"
            )
        self.segmented_context_layers = (
            sorted(set(segmented_context_layers or [])) if self.segmented_kv else None
        )
        self.segmented_latent_layers = (
            sorted(set(segmented_latent_layers or [])) if self.segmented_kv else None
        )
        for p in self.A.parameters(): p.requires_grad = False
        for p in self.B.parameters(): p.requires_grad = False

        if hasattr(self.A.config, "num_hidden_layers"):
            self.A_num_layers = self.A.config.num_hidden_layers
        elif hasattr(self.A.config, "text_config") and hasattr(self.A.config.text_config, "num_hidden_layers"):
            self.A_num_layers = self.A.config.text_config.num_hidden_layers
        else:
            raise ValueError(f"num_hidden_layers not found in {self.A.config}")
        if hasattr(self.B.config, "num_hidden_layers"):
            self.B_num_layers = self.B.config.num_hidden_layers
        elif hasattr(self.B.config, "text_config") and hasattr(self.B.config.text_config, "num_hidden_layers"):
            self.B_num_layers = self.B.config.text_config.num_hidden_layers
        else:
            raise ValueError(f"num_hidden_layers not found in {self.B.config}")


        if layers_list[0] != -1:
            self.layers_list = layers_list
        elif top_layers > 0:
            self.layers_list = list(range(0, self.A_num_layers)) # set all layers at first
        else:
            # layer_to=-1 means "all layers up to the last one"
            effective_to = self.layer_to if self.layer_to >= 0 else (self.A_num_layers - 1)
            self.layers_list = list(range(self.layer_from, effective_to + 1))



        self.layer_map = get_layer_map(self.A_num_layers, self.B_num_layers)

        if self.segmented_kv:
            if self.A_num_layers != self.B_num_layers:
                raise ValueError(
                    "Segmented Dual-KV v1 requires A and B to have the same number of layers"
                )
            if type(self.A.model) is not Qwen3Model or type(self.B.model) is not Qwen3Model:
                raise NotImplementedError(
                    "Segmented Dual-KV v1 currently requires Qwen3 for both A and B"
                )
            if self.A.model.has_sliding_layers or self.B.model.has_sliding_layers:
                raise NotImplementedError(
                    "Segmented Dual-KV v1 does not support sliding-attention layers"
                )
            architecture_fields = (
                "hidden_size", "num_attention_heads", "num_key_value_heads", "head_dim",
                "sliding_window", "layer_types",
            )
            mismatched = [
                field for field in architecture_fields
                if getattr(self.A.config, field, None) != getattr(self.B.config, field, None)
            ]
            if mismatched:
                raise ValueError(
                    "Segmented Dual-KV v1 requires matching A/B Qwen3 architecture; "
                    f"mismatched fields: {mismatched}"
                )
            if not self.shift_back:
                raise ValueError(
                    "Segmented Dual-KV requires shift_back=True for position-aware routing"
                )
            valid = set(range(self.A_num_layers))
            invalid = (
                set(self.segmented_context_layers) | set(self.segmented_latent_layers)
            ) - valid
            if invalid:
                raise ValueError(f"invalid segmented layer ids: {sorted(invalid)}")

        if capture_segmented_importance and not apply_attn_tracer:
            raise ValueError("segmented importance capture requires apply_attn_tracer=True")
        if apply_attn_tracer:
            self.B_attn_weights = {}
            self.apply_B_attn_tracer()

        logging.info(
            "CVCommunicator initialized: segmented=%s, context_layers=%s, latent_layers=%s",
            self.segmented_kv,
            self.segmented_context_layers,
            self.segmented_latent_layers,
        )

    @staticmethod
    def get_dual_layers_list(
        A_num_layers: int,
        split_ratio: float = 0.4,
        context_top_ratio: float = 1.0,
        latent_top_ratio: float = 1.0,
    ) -> tuple[list[int], list[int]]:
        """
        Partition A's layers into context-optimized (shallow) and latent-optimized (deep)
        groups for §3.1 Dual-Selective KV Routing.

        Instead of ranking all layers by a single importance score (Mode 2), we split
        the depth axis into two halves and select independently from each:

          - **Context half** [0, split_point):  shallow–mid layers that preserve
            factual/verbatim representations from T_A input tokens.
            Fixing MultiFieldQA-EN regression caused by latent thinking distorting
            extraction context (Finding F2 / Problem P1 in problem_v1.md).

          - **Latent half** [split_point, L):   mid–deep layers that encode reasoning
            abstractions built during the N latent thinking steps.
            Preserving the reasoning capability that makes LatentMAS win on HotpotQA/TMATH.

        The caller should union both returned lists and pass to CVCommunicator as
        `layers_list`:
            context_l, latent_l = CVCommunicator.get_dual_layers_list(L, ...)
            layers_list = sorted(set(context_l) | set(latent_l))
            cv = CVCommunicator(..., layers_list=layers_list, ...)

        Args:
            A_num_layers:       Total number of transformer layers in model A.
            split_ratio:        Fraction of layers (from shallow end) reserved for the
                                context group.  E.g. 0.4 with 36 layers → context covers
                                layers 0–13, latent covers 14–35.  Default 0.4.
            context_top_ratio:  Fraction of context-half layers to include (1.0 = all).
                                Layers are taken from the *shallowest* end of the
                                context half (first n_ctx layers of [0, split_point)).
            latent_top_ratio:   Fraction of latent-half layers to include (1.0 = all).
                                Layers are taken from the *deepest* end of the latent
                                half (last n_lat layers of [split_point, L)).

        Returns:
            (context_layers, latent_layers): Two lists of integer layer indices.
            Indices may overlap (a layer in both lists is always transferred in full).
        """
        if A_num_layers < 2:
            raise ValueError(f"A_num_layers must be >= 2, got {A_num_layers}")

        split_point = round(split_ratio * A_num_layers)
        split_point = max(1, min(split_point, A_num_layers - 1))  # clamp to valid range

        # ── Context half: shallow layers [0, split_point) ─────────────────
        context_half = list(range(0, split_point))
        n_ctx = max(1, round(context_top_ratio * len(context_half)))
        # Keep the first n_ctx layers (shallowest end of context half)
        context_layers = context_half[:n_ctx]

        # ── Latent half: deep layers [split_point, A_num_layers) ──────────
        latent_half = list(range(split_point, A_num_layers))
        n_lat = max(1, round(latent_top_ratio * len(latent_half)))
        # Keep the last n_lat layers (deepest end of latent half)
        latent_layers = latent_half[-n_lat:]

        logging.debug(
            f"get_dual_layers_list: A_num_layers={A_num_layers}, "
            f"split_ratio={split_ratio} → split_point={split_point}, "
            f"context_layers={context_layers} (n={len(context_layers)}), "
            f"latent_layers={latent_layers} (n={len(latent_layers)})"
        )
        return context_layers, latent_layers



    def apply_B_attn_tracer(self):
        if hasattr(self.B.model, "language_model"):
            layers = self.B.model.language_model.layers
        else:
            layers = self.B.model.layers
        for i, block in enumerate(layers):
            old = block.self_attn
            device = next(old.parameters()).device
            dtype  = next(old.parameters()).dtype
            if type(old) is Qwen2Attention:
                new = Qwen2AttentionTracer(old.config, old.layer_idx).to(device, dtype)
                new.load_state_dict(old.state_dict(), strict=True)
                block.self_attn = new
            elif type(old) is Qwen3Attention:
                new = Qwen3AttentionTracer(old.config, old.layer_idx).to(device, dtype)
                new.load_state_dict(old.state_dict(), strict=True)
                block.self_attn = new
            elif type(old) is LlamaAttention:
                new = LlamaAttentionTracer(old.config, old.layer_idx).to(device, dtype)
                new.load_state_dict(old.state_dict(), strict=True)
                block.self_attn = new
            elif type(old) is Gemma3Attention:
                new = Gemma3AttentionTracer(old.config, old.layer_idx).to(device, dtype)
                new.load_state_dict(old.state_dict(), strict=True)
                block.self_attn = new
            else:
                raise ValueError(f"Unsupported attention module: {type(old)}")

    def remove_B_attn_tracer(self):
        """Restore standard attention modules after calibration.

        Tracers retain the latest Q/K tensors for offline attention scoring.
        Replacing them before the full evaluation releases those references and
        avoids paying the calibration memory overhead on every sample.
        """
        if hasattr(self.B.model, "language_model"):
            layers = self.B.model.language_model.layers
        else:
            layers = self.B.model.layers
        replacements = {
            LlamaAttentionTracer: LlamaAttention,
            Qwen2AttentionTracer: Qwen2Attention,
            Qwen3AttentionTracer: Qwen3Attention,
            Gemma3AttentionTracer: Gemma3Attention,
        }
        for block in layers:
            old = block.self_attn
            standard_class = replacements.get(type(old))
            if standard_class is None:
                continue
            device = next(old.parameters()).device
            dtype = next(old.parameters()).dtype
            new = standard_class(old.config, old.layer_idx).to(device, dtype)
            new.load_state_dict(old.state_dict(), strict=True)
            block.self_attn = new
        self.apply_attn_tracer = False
        self.B_attn_weights = {}

    def configure_segmented_attention_capture(self, context_length: int, latent_length: int):
        """Reset scalar attention-mass capture for the next B prefill."""
        if not self.capture_segmented_importance:
            return
        if hasattr(self.B.model, "language_model"):
            layers = self.B.model.language_model.layers
        else:
            layers = self.B.model.layers
        for block in layers:
            attention = block.self_attn
            if type(attention) is not Qwen3AttentionTracer:
                raise TypeError("Mode 5 calibration requires Qwen3AttentionTracer")
            attention.segment_boundaries = (int(context_length), int(latent_length))
            attention.segment_masses = None
            attention.attn_inputs = None

    def get_segmented_attention_masses(self):
        """Collect the two captured scalar masses from every B layer."""
        if hasattr(self.B.model, "language_model"):
            layers = self.B.model.language_model.layers
        else:
            layers = self.B.model.layers
        masses = {}
        for layer, block in enumerate(layers):
            layer_masses = getattr(block.self_attn, "segment_masses", None)
            if layer_masses is None:
                raise RuntimeError(f"no segmented attention mass captured for layer {layer}")
            masses[layer] = dict(layer_masses)
        return masses

    def get_layer_device(self, layer_idx: int):
        try:
            if hasattr(self.B, "model") and hasattr(self.B.model, "layers"):
                return self.B.model.layers[layer_idx].self_attn.k_proj.weight.device
            elif hasattr(self.B, "model") and hasattr(self.B.model, "language_model"):
                return self.B.model.language_model.layers[layer_idx].self_attn.k_proj.weight.device
        except Exception:
            pass
        return next(self.B.parameters()).device

    def prepare_key_cache(self, past_key_values):
        if self.segmented_kv:
            return self.prepare_segmented_key_cache(past_key_values)
        key_cache = past_key_values.key_cache
        value_cache = past_key_values.value_cache
        assert len(key_cache) == len(self.layer_map), "key_cache and layer_map must have the same length"
        past_key_values_new = DynamicCache()
        for i in range(len(key_cache)): # i is the layer index of model A
            target_layer_b = self.layer_map[i]
            target_device = self.get_layer_device(target_layer_b)
            k_i = key_cache[i].to(target_device)
            v_i = value_cache[i].to(target_device)
            if i in self.layers_list or i == 0:
                past_key_values_new.update(k_i, v_i, target_layer_b)
            else:
                # keep the first token due to attention sink
                key_cache_i = k_i[:, :, :1, :]
                value_cache_i = v_i[:, :, :1, :]
                past_key_values_new.update(key_cache_i, value_cache_i, target_layer_b)
        return past_key_values_new

    def prepare_segmented_key_cache(self, past_key_values):
        """Physically route input and latent KV segments independently per layer."""
        context_length = getattr(past_key_values, "_kvcomm_context_length", None)
        latent_length = getattr(past_key_values, "_kvcomm_latent_length", None)
        logical_length = getattr(past_key_values, "_kvcomm_logical_length", None)
        if context_length is None or latent_length is None or logical_length is None:
            raise ValueError(
                "Segmented Dual-KV requires cache boundaries produced by LatentMAS.run()"
            )
        if past_key_values.key_cache[0].shape[0] != 1:
            raise ValueError("Segmented Dual-KV v1 requires batch_size=1")

        context_set = set(self.segmented_context_layers)
        latent_set = set(self.segmented_latent_layers)
        routed = DynamicCache()
        states = {}
        prefix_lengths = {}
        actual_bytes = 0
        full_bytes = 0
        for source_layer, (key, value) in enumerate(zip(
            past_key_values.key_cache, past_key_values.value_cache
        )):
            full_bytes += key.numel() * key.element_size()
            full_bytes += value.numel() * value.element_size()
            target_layer = self.layer_map[source_layer]
            target_device = self.get_layer_device(target_layer)
            routed_key, routed_value, state = slice_segmented_kv(
                key,
                value,
                context_length=context_length,
                latent_length=latent_length,
                keep_context=source_layer in context_set,
                keep_latent=source_layer in latent_set,
            )
            # Move only the retained contiguous tensors, unlike the legacy path
            # which moves the full cache before slicing.
            routed_key = routed_key.to(target_device)
            routed_value = routed_value.to(target_device)
            routed.update(routed_key, routed_value, target_layer)
            states[target_layer] = state
            prefix_lengths[target_layer] = int(routed_key.shape[-2])
            actual_bytes += routed_key.numel() * routed_key.element_size()
            actual_bytes += routed_value.numel() * routed_value.element_size()

        accounting = logical_retention(
            self.A_num_layers,
            self.segmented_context_layers,
            self.segmented_latent_layers,
            context_length,
            latent_length,
        )
        routed._segmented_kv = True
        routed._segmented_context_length = int(context_length)
        routed._segmented_latent_length = int(latent_length)
        routed._segmented_logical_prefix_length = int(logical_length)
        routed._segmented_next_position = int(logical_length)
        routed._segmented_prefix_lengths = prefix_lengths
        routed._segmented_states = states
        routed._segmented_actual_bytes = int(actual_bytes)
        routed._segmented_accounting = accounting
        self.last_segmented_stats = {
            **accounting,
            "actual_tensor_bytes": int(actual_bytes),
            "full_tensor_bytes": int(full_bytes),
            "byte_retention_ratio": actual_bytes / full_bytes if full_bytes else 0.0,
            "context_length": int(context_length),
            "latent_length": int(latent_length),
            "states": dict(states),
        }
        return routed

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        out_A_past_key_values: Optional[Cache] = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs
    ):

        if out_A_past_key_values is None:
            raise NotImplementedError("out_A_past_key_values is required")
        else:
            # Do not infer the prefill/decode phase from input length: a valid
            # receiver prompt can contain exactly one token. Generation starts
            # with no B cache (or an empty DynamicCache) and later supplies the
            # cache returned by this communicator.
            has_receiver_cache = (
                past_key_values is not None
                and past_key_values.get_seq_length() > 0
            )
            if not has_receiver_cache:
                out_A_past_key_values = self.prepare_key_cache(out_A_past_key_values)
            else:
                out_A_past_key_values = past_key_values
        
        if self.segmented_kv:
            out_B = forward_segmented_qwen3(
                model=self.B,
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=out_A_past_key_values,
                logits_to_keep=logits_to_keep,
                **kwargs,
            )
        elif self.shift_back:
            if type(self.B.model) == LlamaModel:
                out_B = forward_shift_back_llama(
                    model=self.B,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    past_key_values=out_A_past_key_values,
                    logits_to_keep=logits_to_keep,
                    **kwargs
                )
            elif type(self.B.model) in (Qwen2Model, Qwen3Model):
                out_B = forward_shift_back_qwen2(
                    model=self.B,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    past_key_values=out_A_past_key_values,
                    logits_to_keep=logits_to_keep,
                    **kwargs
                )
            else:
                raise NotImplementedError(f"shift_back is not implemented for model type {type(self.B)}")
        else:
            if "logits_to_keep" in inspect.signature(self.B.forward).parameters:
                kwargs["logits_to_keep"] = logits_to_keep
            out_B = self.B(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=out_A_past_key_values,
                **kwargs
            )

        return out_B

    @torch.no_grad()
    def calc_attn_weights_from_qk(self):
        assert self.apply_attn_tracer, "apply_attn_tracer must be True"
        if hasattr(self.B.model, "language_model"):
            layers = self.B.model.language_model.layers
        else:
            layers = self.B.model.layers
        for i, block in enumerate(layers):
            attn_inputs = block.self_attn.attn_inputs
            attn_weights = eager_attention_forward_without_value(block.self_attn, **attn_inputs)
            # attn_weights_sdpa = sdpa_attention_forward_without_value(block.self_attn, **attn_inputs)
            self.B_attn_weights[i] = attn_weights

def eager_attention_forward_without_value(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)

    return attn_weights


def sdpa_attention_forward_without_value(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    is_causal: Optional[bool] = None,
    **kwargs,
) -> torch.Tensor:

    if hasattr(module, "num_key_value_groups"):
        key = repeat_kv(key, module.num_key_value_groups)

    if attention_mask is not None and attention_mask.ndim == 4:
        attention_mask = attention_mask[:, :, :, : key.shape[-2]]

    # SDPA with memory-efficient backend is bugged with non-contiguous inputs and custom attn_mask for some torch versions
    # Reference: https://github.com/pytorch/pytorch/issues/112577.
    query = query.contiguous()
    key = key.contiguous()
    eye = torch.eye(key.shape[-2], dtype=key.dtype, device=key.device)
    value_eye = eye.unsqueeze(0).unsqueeze(0).expand(key.shape[0], key.shape[1], -1, -1)

    # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
    # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
    # Note that it is important to check first for the shape, otherwise compile will fail with `argument 'is_causal' must be bool, not SymBool`
    if is_causal is None:
        # The last condition is for encoder (decoder) models which specify this by passing their own `is_causal` flag
        # This is mainly due to those models having mixed implementations for encoder, decoder, and encoder-decoder attns
        is_causal = query.shape[2] > 1 and attention_mask is None and getattr(module, "is_causal", True)

    # Shapes (e.g. query.shape[2]) are tensors during jit tracing, resulting in `is_causal` being a tensor.
    # We convert it to a bool for the SDPA kernel that only accepts bools.
    if torch.jit.is_tracing() and isinstance(is_causal, torch.Tensor):
        is_causal = is_causal.item()

    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query,
        key,
        value_eye,
        attn_mask=attention_mask,
        dropout_p=dropout,
        scale=scaling,
        is_causal=is_causal,
    )
    attn_weights = attn_output

    return attn_weights

def get_short_past_key_values(past_key_values: DynamicCache):
    lengths = set()
    for idx in range(len(past_key_values.key_cache)):
        if past_key_values.key_cache[idx].numel():
            lengths.add(past_key_values.key_cache[idx].shape[-2])
    assert len(lengths) <= 2
    short_past_key_values = copy.deepcopy(past_key_values)
    short_past_key_values.crop(min(lengths))
    short_length = min(lengths)
    return short_past_key_values, short_length


from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask


@torch._dynamo.disable
def forward_segmented_qwen3(
    model: PreTrainedModel,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    past_key_values=None,
    logits_to_keep: int | torch.Tensor = 0,
    **kwargs,
):
    """Qwen3 forward for per-layer segmented cache geometries.

    Each layer can have a different physical prefix length, while B's RoPE
    positions remain in the original full-cache coordinate frame.  Version 1
    intentionally supports batch size one only; padding cannot be represented
    safely by one shared logical position sequence here.
    """
    if past_key_values is None or not getattr(past_key_values, "_segmented_kv", False):
        raise ValueError("forward_segmented_qwen3 requires a segmented DynamicCache")
    if input_ids is None or input_ids.shape[0] != 1:
        raise ValueError("Segmented Dual-KV v1 requires batch_size=1")

    inputs_embeds = model.get_input_embeddings()(input_ids)
    seq_len = inputs_embeds.shape[1]
    logical_start = int(past_key_values._segmented_next_position)
    logical_position_ids = torch.arange(
        logical_start,
        logical_start + seq_len,
        device=inputs_embeds.device,
    ).unsqueeze(0)
    position_embeddings = model.model.rotary_emb(inputs_embeds, logical_position_ids)

    output_hidden_states = kwargs.get(
        "output_hidden_states", model.model.config.output_hidden_states
    )
    hidden_states = inputs_embeds
    all_hidden_states = () if output_hidden_states else None
    for layer_idx, decoder_layer in enumerate(
        model.model.layers[: model.config.num_hidden_layers]
    ):
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        physical_past = int(past_key_values.key_cache[layer_idx].shape[-2])
        physical_cache_position = torch.arange(
            physical_past,
            physical_past + seq_len,
            device=inputs_embeds.device,
        )
        physical_position_ids = physical_cache_position.unsqueeze(0)
        physical_attention_mask = torch.ones(
            (1, physical_past + seq_len),
            dtype=torch.long,
            device=inputs_embeds.device,
        )
        mask_kwargs = {
            "config": model.model.config,
            "input_embeds": hidden_states,
            "attention_mask": physical_attention_mask,
            "cache_position": physical_cache_position,
            "past_key_values": past_key_values,
            # Mask construction uses physical sequence geometry.  RoPE below
            # deliberately uses logical_position_ids instead.
            "position_ids": physical_position_ids,
        }
        masks = {"full_attention": create_causal_mask(**mask_kwargs)}
        if getattr(model.model, "has_sliding_layers", False):
            masks["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)
        attention_type = getattr(decoder_layer, "attention_type", "full_attention")

        layer_outputs = decoder_layer(
            hidden_states,
            attention_mask=masks[attention_type],
            position_ids=logical_position_ids,
            past_key_value=past_key_values,
            output_attentions=model.model.config.output_attentions,
            use_cache=model.model.config.use_cache,
            cache_position=physical_cache_position,
            position_embeddings=position_embeddings,
        )
        hidden_states = layer_outputs[0]

    past_key_values._segmented_next_position = logical_start + seq_len
    hidden_states = model.model.norm(hidden_states)
    if output_hidden_states:
        all_hidden_states += (hidden_states,)
    slice_indices = (
        slice(-logits_to_keep, None)
        if isinstance(logits_to_keep, int)
        else logits_to_keep
    )
    logits = model.lm_head(hidden_states[:, slice_indices, :])
    return CausalLMOutputWithPast(
        logits=logits,
        past_key_values=past_key_values,
        hidden_states=all_hidden_states,
        attentions=None,
    )

@torch._dynamo.disable
def forward_shift_back_llama(
    model: PreTrainedModel,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    past_key_values=None,
    logits_to_keep: int | torch.Tensor = 0,
    **kwargs,
):
    inputs_embeds = model.get_input_embeddings()(input_ids)
    if past_key_values is None:
        past_key_values = DynamicCache()

    seq_len = inputs_embeds.shape[1]  # T_B (number of B's input tokens)

    ##########  Full-cache path (selected layers: T+N tokens)  ##########
    past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
    cache_position = torch.arange(
        past_seen_tokens, past_seen_tokens + seq_len, device=inputs_embeds.device
    )
    position_ids = cache_position.unsqueeze(0)

    # attention_mask here is [1]*(past_seen_tokens + seq_len) — correct for full cache.
    causal_mask = create_causal_mask(
        config=model.model.config,
        input_embeds=inputs_embeds,
        attention_mask=attention_mask,
        cache_position=cache_position,
        past_key_values=past_key_values,
        position_ids=position_ids,
    )

    ##########  Short-cache path (non-selected layers: short_length=1 token)  ##########
    short_past_key_values, short_length = get_short_past_key_values(past_key_values)
    short_past_seen = short_past_key_values.get_seq_length() if short_past_key_values is not None else 0
    short_cache_position = torch.arange(
        short_past_seen, short_past_seen + seq_len, device=inputs_embeds.device
    )
    short_position_ids = short_cache_position.unsqueeze(0)

    # Build a short_attention_mask matching the short cache length.
    # The full attention_mask is [1]*(T+N+T_B) but short cache only has short_length tokens.
    # Passing the full mask to create_causal_mask with short_cache_position causes
    # it to build a wrong causal mask for these layers.
    # Fix: construct [1]*(short_length + T_B) so the mask matches the actual short cache.
    if attention_mask is not None:
        short_attention_mask = torch.ones(
            (attention_mask.shape[0], short_past_seen + seq_len),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
    else:
        short_attention_mask = None

    short_causal_mask = create_causal_mask(
        config=model.model.config,
        input_embeds=inputs_embeds,
        attention_mask=short_attention_mask,  # ← matched to short cache length
        cache_position=short_cache_position,
        past_key_values=short_past_key_values,
        position_ids=short_position_ids,
    )
    hidden_states = inputs_embeds

    # create position embeddings to be shared across the decoder layers
    position_embeddings = model.model.rotary_emb(hidden_states, position_ids)
    short_position_embeddings = model.model.rotary_emb(hidden_states, short_position_ids)
    
    output_hidden_states = kwargs.get(
        "output_hidden_states", model.model.config.output_hidden_states
    )
    all_hidden_states = () if output_hidden_states else None

    for i, decoder_layer in enumerate(model.model.layers[: model.config.num_hidden_layers]):

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if past_key_values.key_cache[i].shape[-2] == short_length:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=short_causal_mask,
                position_ids=short_position_ids,
                past_key_value=past_key_values,
                output_attentions=model.model.config.output_attentions,
                use_cache=model.model.config.use_cache,
                cache_position=short_cache_position,
                position_embeddings=short_position_embeddings,
            )
        else:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=model.model.config.output_attentions,
                use_cache=model.model.config.use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )

        hidden_states = layer_outputs[0]

    hidden_states = model.model.norm(hidden_states)

    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    slice_indices = (
        slice(-logits_to_keep, None)
        if isinstance(logits_to_keep, int)
        else logits_to_keep
    )
    logits = model.lm_head(hidden_states[:, slice_indices, :])

    return CausalLMOutputWithPast(
        logits=logits,
        past_key_values=past_key_values,
        hidden_states=all_hidden_states,
        attentions=None,
    )

@torch._dynamo.disable
def forward_shift_back_qwen2(
    model: PreTrainedModel,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    past_key_values=None,
    logits_to_keep: int | torch.Tensor = 0,
    **kwargs,
):
    inputs_embeds = model.get_input_embeddings()(input_ids)
    if past_key_values is None:
        past_key_values = DynamicCache()

    seq_len = inputs_embeds.shape[1]  # T_B

    ##########  Full-cache path (selected layers)  ##########
    past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
    cache_position = torch.arange(
        past_seen_tokens, past_seen_tokens + seq_len, device=inputs_embeds.device
    )
    position_ids = cache_position.unsqueeze(0)

    # It may already have been prepared by e.g. `generate`
    if not isinstance(causal_mask_mapping := attention_mask, dict):
        mask_kwargs = {
            "config": model.model.config,
            "input_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "cache_position": cache_position,
            "past_key_values": past_key_values,
            "position_ids": position_ids,
        }
        causal_mask_mapping = {
            "full_attention": create_causal_mask(**mask_kwargs),
        }
        if model.model.has_sliding_layers:
            causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

    ##########  Short-cache path (non-selected layers: short_length tokens)  ##########
    short_past_key_values, short_length = get_short_past_key_values(past_key_values)
    short_past_seen = short_past_key_values.get_seq_length() if short_past_key_values is not None else 0
    short_cache_position = torch.arange(
        short_past_seen, short_past_seen + seq_len, device=inputs_embeds.device
    )
    short_position_ids = short_cache_position.unsqueeze(0)

    # Build short_attention_mask matched to short cache length.
    # The full attention_mask is [1]*(T+N+T_B); passing it with short_cache_position
    # causes create_causal_mask to produce a wrong mask. Fix: [1]*(short_length + T_B).
    if attention_mask is not None and not isinstance(attention_mask, dict):
        short_attention_mask = torch.ones(
            (attention_mask.shape[0], short_past_seen + seq_len),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
    else:
        short_attention_mask = attention_mask

    # It may already have been prepared by e.g. `generate`
    if not isinstance(short_causal_mask_mapping := short_attention_mask, dict):
        mask_kwargs = {
            "config": model.model.config,
            "input_embeds": inputs_embeds,
            "attention_mask": short_attention_mask,  # ← matched to short cache
            "cache_position": short_cache_position,
            "past_key_values": short_past_key_values,
            "position_ids": short_position_ids,
        }
        short_causal_mask_mapping = {
            "full_attention": create_causal_mask(**mask_kwargs),
        }
        if model.model.has_sliding_layers:
            short_causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)
    hidden_states = inputs_embeds

    # create position embeddings to be shared across the decoder layers
    position_embeddings = model.model.rotary_emb(hidden_states, position_ids)
    short_position_embeddings = model.model.rotary_emb(hidden_states, short_position_ids)
    
    output_hidden_states = kwargs.get(
        "output_hidden_states", model.model.config.output_hidden_states
    )
    all_hidden_states = () if output_hidden_states else None

    for i, decoder_layer in enumerate(model.model.layers[: model.config.num_hidden_layers]):

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if past_key_values.key_cache[i].shape[-2] == short_length:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=short_causal_mask_mapping[decoder_layer.attention_type],
                position_ids=short_position_ids,
                past_key_value=past_key_values,
                output_attentions=model.model.config.output_attentions,
                use_cache=model.model.config.use_cache,
                cache_position=short_cache_position,
                position_embeddings=short_position_embeddings,
            )
        else:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=model.model.config.output_attentions,
                use_cache=model.model.config.use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )

        hidden_states = layer_outputs[0]

    hidden_states = model.model.norm(hidden_states)

    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    slice_indices = (
        slice(-logits_to_keep, None)
        if isinstance(logits_to_keep, int)
        else logits_to_keep
    )
    logits = model.lm_head(hidden_states[:, slice_indices, :])


    return CausalLMOutputWithPast(
        logits=logits,
        past_key_values=past_key_values,
        hidden_states=all_hidden_states,
        attentions=None,
    )

