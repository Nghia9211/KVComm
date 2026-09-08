"""Schema-v2 response records shared by text and latent communication modes."""

from typing import Any, Mapping, Optional


def normalize_answers(item: Mapping[str, Any]) -> list[str]:
    if item.get("answers") is not None:
        value = item["answers"]
        return [str(answer) for answer in (value if isinstance(value, (list, tuple)) else [value])]
    if item.get("answer") is not None:
        return [str(item["answer"])]
    return []


def build_response_record(
    *, idx: int, evaluator: Any, item: Mapping[str, Any], method: str,
    response: str, model_a_prompt: str, model_b_prompt: str,
    response_a: Optional[str] = None, item_metrics: Optional[Mapping[str, float]] = None,
    aggregate_metrics: Optional[Mapping[str, Any]] = None,
    max_tokens_a: Optional[int] = None, max_tokens_b: Optional[int] = None,
    generated_tokens_a: Optional[int] = None, generated_tokens_b: Optional[int] = None,
    communication_type: str = "latent_kv", latent_steps: Optional[int] = None,
    layer_selection_mode: Optional[str] = None, selected_layers: Optional[list[int]] = None,
) -> dict[str, Any]:
    return {
        "schema_version": "v2",
        "metric_version": "v2",
        "idx": idx,
        "task": getattr(evaluator, "name", "unknown"),
        "method": method,
        "prompt_family": evaluator.prompt_family,
        "prompt_version": evaluator.prompt_version,
        "sender_input_mode": evaluator.sender_input_mode,
        "communication_type": communication_type,
        "prompt_A": item.get("prompt_A", ""),
        "prompt_B": item.get("prompt_B", ""),
        "model_A_prompt": model_a_prompt,
        "model_B_prompt": model_b_prompt,
        "response_A": response_a,
        "response": response,
        "answers": normalize_answers(item),
        "item_metrics": dict(item_metrics or {}),
        "aggregate_metrics": dict(aggregate_metrics or {}),
        "token_counts": {
            "max_tokens_A": max_tokens_a,
            "max_tokens_B": max_tokens_b,
            "generated_tokens_A": generated_tokens_a,
            "generated_tokens_B": generated_tokens_b,
        },
        "latent": {
            "steps": latent_steps,
            "layer_selection_mode": layer_selection_mode,
            "selected_layers": [int(x) for x in selected_layers] if selected_layers is not None else None,
        },
    }
