"""Pure utilities for Segmented Dual-KV routing.

The functions in this module intentionally do not depend on Transformers so the
selection and tensor geometry can be unit-tested without loading an LLM.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Mapping, MutableMapping, Sequence

import numpy as np
import torch


def top_fraction_layers(scores: Mapping[int, float], ratio: float) -> list[int]:
    """Return floor(ratio * L) layer ids, ranked by score with stable ties.

    A positive ratio always keeps at least one layer.  Layer id is the
    deterministic tie-breaker so calibration is reproducible.
    """
    if not 0.0 < ratio <= 1.0:
        raise ValueError(f"ratio must be in (0, 1], got {ratio}")
    if not scores:
        raise ValueError("scores must not be empty")
    count = max(1, math.floor(ratio * len(scores)))
    ranked = sorted(scores, key=lambda layer: (-float(scores[layer]), int(layer)))
    return ranked[:count]


def mean_layer_scores(samples: Mapping[int, Sequence[float]]) -> dict[int, float]:
    """Average per-sample normalized scores for each layer."""
    if not samples:
        raise ValueError("no layer-importance samples were collected")
    return {
        int(layer): float(np.mean(values))
        for layer, values in samples.items()
        if values
    }


def select_segmented_layers(
    totals: Mapping[str, Mapping[int, Sequence[float]]],
    context_ratio: float,
    latent_ratio: float,
) -> tuple[list[int], list[int], dict[int, float], dict[int, float]]:
    """Select context and latent layer sets independently."""
    context_scores = mean_layer_scores(totals.get("context", {}))
    latent_scores = mean_layer_scores(totals.get("latent", {}))
    return (
        sorted(top_fraction_layers(context_scores, context_ratio)),
        sorted(top_fraction_layers(latent_scores, latent_ratio)),
        context_scores,
        latent_scores,
    )


def _minmax_normalize(scores: Mapping[int, float]) -> dict[int, float]:
    values = list(scores.values())
    if not values:
        return {}
    low, high = min(values), max(values)
    if high - low < 1e-12:
        return {int(layer): 0.0 for layer in scores}
    return {
        int(layer): (float(value) - low) / (high - low)
        for layer, value in scores.items()
    }


def segment_attention_mass_from_qk(
    query: torch.Tensor,
    key: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    context_length: int,
    latent_length: int,
    num_key_value_groups: int,
    query_chunk_size: int = 4,
) -> dict[str, float]:
    """Compute exact context/latent softmax mass without retaining full attention.

    Qwen3 uses grouped-query attention. The einsum keeps KV heads grouped and
    chunks the query axis, avoiding a persistent ``[layers, heads, Q, K]``
    tensor during long-context calibration.
    """
    if query.ndim != 4 or key.ndim != 4:
        raise ValueError("query and key must have shape [batch, heads, seq, dim]")
    if context_length < 1 or latent_length < 1:
        raise ValueError("context_length and latent_length must both be positive")
    latent_end = context_length + latent_length
    if key.shape[-2] < latent_end:
        raise ValueError("key sequence is shorter than the requested segments")
    if query.shape[1] != key.shape[1] * num_key_value_groups:
        raise ValueError("query/KV head counts do not match num_key_value_groups")
    if query_chunk_size < 1:
        raise ValueError("query_chunk_size must be positive")

    batch, _, query_length, head_dim = query.shape
    kv_heads = key.shape[1]
    query_grouped = query.reshape(
        batch, kv_heads, num_key_value_groups, query_length, head_dim
    )
    context_sum = torch.zeros((), dtype=torch.float32, device=query.device)
    latent_sum = torch.zeros((), dtype=torch.float32, device=query.device)
    count = 0
    for start in range(0, query_length, query_chunk_size):
        end = min(start + query_chunk_size, query_length)
        query_chunk = query_grouped[..., start:end, :]
        logits = torch.einsum("bkgqd,bkmd->bkgqm", query_chunk, key) * scaling
        if attention_mask is not None:
            mask = attention_mask[..., start:end, : key.shape[-2]]
            if mask.shape[1] == 1:
                mask = mask.unsqueeze(2)
            elif mask.shape[1] == query.shape[1]:
                mask = mask.reshape(
                    batch, kv_heads, num_key_value_groups, end - start, key.shape[-2]
                )
            else:
                raise ValueError("attention mask head dimension is not broadcastable")
            logits = logits + mask
        else:
            # Preserve causal semantics if SDPA omitted an explicit 4-D mask.
            past_length = key.shape[-2] - query_length
            key_positions = torch.arange(key.shape[-2], device=query.device)
            query_positions = past_length + torch.arange(start, end, device=query.device)
            allowed = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
            logits = logits.masked_fill(
                ~allowed.view(1, 1, 1, end - start, key.shape[-2]),
                torch.finfo(logits.dtype).min,
            )
        probabilities = torch.softmax(logits, dim=-1, dtype=torch.float32)
        context_sum += probabilities[..., 1:context_length].sum()
        latent_sum += probabilities[..., context_length:latent_end].sum()
        count += probabilities[..., 0].numel()
        del logits, probabilities

    return {
        "context": float(context_sum.item()) / count if count else 0.0,
        "latent": float(latent_sum.item()) / count if count else 0.0,
    }


def accumulate_segment_masses(
    masses_by_layer: Mapping[int, Mapping[str, float]],
    totals: MutableMapping[str, MutableMapping[int, list[float]]] | None = None,
) -> MutableMapping[str, MutableMapping[int, list[float]]]:
    """Normalize scalar segment masses per sample and append to totals."""
    if totals is None:
        totals = {"context": defaultdict(list), "latent": defaultdict(list)}
    if not masses_by_layer:
        raise ValueError("no segmented attention masses were captured")
    for segment in ("context", "latent"):
        raw = {
            int(layer): float(masses[segment])
            for layer, masses in masses_by_layer.items()
        }
        for layer, value in _minmax_normalize(raw).items():
            totals[segment][layer].append(value)
    return totals


def accumulate_segment_attention(
    attention_by_layer: Mapping[int, torch.Tensor],
    context_length: int,
    latent_length: int,
    totals: MutableMapping[str, MutableMapping[int, list[float]]] | None = None,
) -> MutableMapping[str, MutableMapping[int, list[float]]]:
    """Accumulate independently normalized attention mass for both segments.

    Attention tensors have shape ``[batch, heads, B_queries, all_keys]``.  The
    original A sink is position zero and is excluded from ContextScore.  The A
    latent segment occupies ``[context_length, context_length + latent_length)``.
    B's own keys follow those positions and are deliberately excluded.
    """
    if context_length < 1:
        raise ValueError("context_length must include the original sink token")
    if latent_length < 1:
        raise ValueError("Segmented Dual-KV requires latent_length >= 1")
    if totals is None:
        totals = {
            "context": defaultdict(list),
            "latent": defaultdict(list),
        }

    context_raw: dict[int, float] = {}
    latent_raw: dict[int, float] = {}
    latent_end = context_length + latent_length
    for layer, attention in attention_by_layer.items():
        if attention is None or attention.ndim != 4:
            raise ValueError(f"layer {layer} has invalid attention tensor")
        if attention.shape[-1] < latent_end:
            raise ValueError(
                f"layer {layer} key length {attention.shape[-1]} is shorter than "
                f"context+latent length {latent_end}"
            )
        context_mass = attention[..., 1:context_length].sum(dim=-1).mean()
        latent_mass = attention[..., context_length:latent_end].sum(dim=-1).mean()
        context_raw[int(layer)] = 0.0 if torch.isnan(context_mass) else float(context_mass.item())
        latent_raw[int(layer)] = 0.0 if torch.isnan(latent_mass) else float(latent_mass.item())

    # Normalize per sample before aggregation so long calibration examples do
    # not dominate merely because they contain more context tokens.
    for segment, normalized in (
        ("context", _minmax_normalize(context_raw)),
        ("latent", _minmax_normalize(latent_raw)),
    ):
        for layer, value in normalized.items():
            totals[segment][layer].append(value)
    return totals


def slice_segmented_kv(
    key: torch.Tensor,
    value: torch.Tensor,
    context_length: int,
    latent_length: int,
    keep_context: bool,
    keep_latent: bool,
) -> tuple[torch.Tensor, torch.Tensor, str]:
    """Build one layer's physical cache while preserving the original sink.

    Cached keys already contain RoPE in A's original coordinate system; this
    function never rotates or renumbers them.
    """
    expected = context_length + latent_length
    if key.shape[-2] < expected or value.shape[-2] < expected:
        raise ValueError(
            f"cache is shorter than declared segments: key={key.shape[-2]}, "
            f"value={value.shape[-2]}, expected={expected}"
        )
    if context_length < 1:
        raise ValueError("context_length must include the original sink token")

    key_parts = [key[..., :context_length, :]] if keep_context else [key[..., :1, :]]
    value_parts = [value[..., :context_length, :]] if keep_context else [value[..., :1, :]]
    if keep_latent and latent_length:
        key_parts.append(key[..., context_length:expected, :])
        value_parts.append(value[..., context_length:expected, :])

    state = (
        "context+latent" if keep_context and keep_latent
        else "context-only" if keep_context
        else "sink+latent" if keep_latent
        else "sink-only"
    )
    return (
        torch.cat(key_parts, dim=-2).contiguous(),
        torch.cat(value_parts, dim=-2).contiguous(),
        state,
    )


def logical_retention(
    num_layers: int,
    context_layers: Sequence[int],
    latent_layers: Sequence[int],
    context_length: int,
    latent_length: int,
) -> dict[str, int | float]:
    """Return reproducible logical element/token-position accounting."""
    context_set, latent_set = set(context_layers), set(latent_layers)
    full_positions = num_layers * (context_length + latent_length)
    retained_positions = sum(
        (context_length if layer in context_set else 1)
        + (latent_length if layer in latent_set else 0)
        for layer in range(num_layers)
    )
    return {
        "full_kv_token_positions": full_positions,
        "retained_kv_token_positions": retained_positions,
        "logical_retention_ratio": retained_positions / full_positions if full_positions else 0.0,
    }
